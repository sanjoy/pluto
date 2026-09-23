// Conditional final-MLP single-deletion sensitivity, not exclusive ownership
// of facts. All variants receive the ORIGINAL model's cached upstream values.
// Only private, checkpoint-loaded in-memory clones receive affine edits.
#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "src/cuda/executor.h"
#include "src/cuda/page_locked_host_array.h"
#include "src/dataset/compact_vocabulary.h"
#include "src/dataset/dataset.h"
#include "src/dataset/gpt2_tokenizer.h"
#include "src/llm/checkpoint.h"
#include "src/llm/experiments/one_shot_memorizer/feature_subset.h"
#include "src/llm/experiments/one_shot_memorizer/mlp_subset_tail.h"
#include "src/llm/experiments/one_shot_memorizer/token_trace.h"
#include "src/llm/gpt2.h"
#include "src/llm/layer.h"
#include "src/util/status_macros.h"

ABSL_FLAG(std::string, checkpoint, "", "Exact memorized checkpoint directory");
ABSL_FLAG(std::string, tokenizer, "", "Base GPT-2 tokenizer directory");
ABSL_FLAG(std::string, corpus, "testdata/general_facts_dataset.txt",
          "Full corpus");
ABSL_FLAG(std::string, output_dir, "",
          "Fresh directory for feature audit reports");
ABSL_FLAG(std::vector<std::string>, lines,
          (std::vector<std::string>{"80", "406", "411", "1", "258", "631"}),
          "Distinct one-based corpus lines, or all");
ABSL_FLAG(int, expected_sentences, 1024, "Must be 1024 for this experiment");
ABSL_FLAG(int, layers, 8, "Must be 8 for this checkpoint architecture");
ABSL_FLAG(int, model_width, 16, "Must be 16 for this checkpoint architecture");
ABSL_FLAG(int, attention_heads, 1,
          "Must be 1 for this checkpoint architecture");
ABSL_FLAG(int, feed_forward_width, 64,
          "Must be 64 for this checkpoint architecture");

namespace pluto::llm::one_shot_memorizer {
namespace {
namespace fs = std::filesystem;
constexpr size_t kPromptTokens = 5;
constexpr int kVocabulary = 4475;
constexpr int kFeatures = 64;
constexpr int kWidth = 16;
constexpr FeatureSubset kFullSubset = ~FeatureSubset{0};
// The fixed CreateGpt2 recipe has E/P, twelve tensors per block, final gamma/
// beta, and a tied E alias. Its final contraction bias and final beta occupy
// these slots. The tail library validates the complete expected inventory.
constexpr size_t kContractionBiasIndex = 97;
constexpr size_t kFinalBetaIndex = 99;
constexpr size_t kExposedWeightCount = 101;

absl::StatusOr<std::vector<std::string>> ReadCorpusLines(
    absl::string_view text) {
  std::vector<std::string> result;
  while (!text.empty()) {
    const size_t end = text.find('\n');
    auto line = text.substr(0, end);
    if (!line.empty() && line.back() == '\r')
      line.remove_suffix(1);
    if (line.empty() || line.find_first_not_of(" \t\r") == line.npos)
      return absl::InvalidArgumentError("empty or blank corpus line");
    result.emplace_back(line);
    if (end == text.npos)
      break;
    text.remove_prefix(end + 1);
  }
  return result;
}

absl::StatusOr<std::vector<size_t>> SelectLines() {
  const auto requested = absl::GetFlag(FLAGS_lines);
  std::vector<size_t> result;
  if (requested == std::vector<std::string>{"all"}) {
    result.resize(1024);
    std::iota(result.begin(), result.end(), 0);
    return result;
  }
  absl::flat_hash_set<size_t> seen;
  for (const auto& value : requested) {
    size_t line;
    if (value.empty() ||
        !std::all_of(value.begin(), value.end(),
                     [](char ch) { return ch >= '0' && ch <= '9'; }) ||
        !absl::SimpleAtoi(value, &line) || line < 1 || line > 1024 ||
        !seen.insert(line).second)
      return absl::InvalidArgumentError(
          "lines must be distinct unsigned IDs 1..1024, or all");
    result.push_back(line - 1);
  }
  if (result.empty())
    return absl::InvalidArgumentError("empty selected lines");
  return result;
}

struct Prediction {
  int winner = 0;  // Lowest token ID wins ties, including rival ties.
  int rival = 0;   // Largest non-target logit, not necessarily the winner.
  size_t target_rank = 1;
  double probability = 0;
  double margin = 0;
};

absl::StatusOr<Prediction> Score(absl::Span<const float> logits, int target) {
  if (logits.size() != kVocabulary || target < 0 || target >= kVocabulary)
    return absl::InvalidArgumentError(
        "wrong feature-audit logit shape or target");
  Prediction result{.rival = target == 0 ? 1 : 0};
  for (int token = 0; token < kVocabulary; ++token) {
    if (!std::isfinite(logits[token]))
      return absl::DataLossError("nonfinite audit logit");
    if (logits[token] > logits[result.winner])
      result.winner = token;
    if (token != target) {
      if (logits[token] > logits[result.rival])
        result.rival = token;
      result.target_rank += logits[token] > logits[target] ||
                            (logits[token] == logits[target] && token < target);
    }
  }
  const double maximum = logits[result.winner];
  double denominator = 0;
  for (float value : logits)
    denominator += std::exp(static_cast<double>(value) - maximum);
  result.probability =
      std::exp(static_cast<double>(logits[target]) - maximum) / denominator;
  result.margin = static_cast<double>(logits[target]) - logits[result.rival];
  return result;
}

absl::Status ExtractBf16Query(const TokenTraceResult& trace,
                              const TokenTraceSite& site,
                              absl::Span<uint16_t> output) {
  const TokenTraceActivation* found = nullptr;
  for (const auto& activation : trace.activations)
    if (activation.site.scope == site.scope &&
        activation.site.layer_name == site.layer_name &&
        activation.site.occurrence == site.occurrence &&
        activation.site.output_index == site.output_index) {
      if (found != nullptr)
        return absl::FailedPreconditionError("ambiguous activation site");
      found = &activation;
    }
  if (found == nullptr || found->data_type != DataType::BF16 ||
      found->channels != output.size() || found->first_row > trace.query_row ||
      trace.query_row - found->first_row >= found->row_count ||
      found->bytes.size() !=
          found->row_count * output.size() * sizeof(uint16_t))
    return absl::FailedPreconditionError(
        "unexpected cached activation site/type/shape");
  const size_t offset =
      (trace.query_row - found->first_row) * output.size() * sizeof(uint16_t);
  std::memcpy(output.data(), found->bytes.data() + offset,
              output.size() * sizeof(uint16_t));
  return absl::OkStatus();
}

bool SameBytes(absl::Span<const uint8_t> first,
               absl::Span<const uint8_t> second) {
  return first.size() == second.size() &&
         std::memcmp(first.data(), second.data(), first.size()) == 0;
}

bool SameLogits(absl::Span<const float> first, absl::Span<const float> second) {
  return first.size() == second.size() &&
         std::memcmp(first.data(), second.data(),
                     first.size() * sizeof(float)) == 0;
}

using HostBytes = cuda::PageLockedHostArray<uint8_t>;

absl::StatusOr<std::vector<HostBytes>> CopyMasterBytes(cuda::Executor& executor,
                                                       const Layer& model) {
  std::vector<HostBytes> result;
  for (const auto& weight : model.weights()) {
    ASSIGN_OR_RETURN(auto host,
                     HostBytes::Allocate(executor, weight.size_bytes()));
    RETURN_IF_ERROR(cuda::CudaStatus(
        cudaMemcpyAsync(host.data(), weight.data(), host.size_bytes(),
                        cudaMemcpyDeviceToHost, executor.stream()),
        "snapshot audit master bytes"));
    result.push_back(std::move(host));
  }
  RETURN_IF_ERROR(executor.Synchronize());
  return result;
}

// Verify a private clone differs from the checkpoint ONLY at its intended
// all-zero affine tensor. This includes all untouched upstream tensors.
absl::Status ValidateCloneBytes(absl::Span<const HostBytes> original,
                                absl::Span<const HostBytes> clone,
                                size_t edited_index) {
  if (original.size() != kExposedWeightCount || clone.size() != original.size())
    return absl::FailedPreconditionError("clone master inventory mismatch");
  for (size_t i = 0; i < original.size(); ++i) {
    if (original[i].size() != clone[i].size())
      return absl::FailedPreconditionError(
          "clone master tensor length mismatch");
    if (i == edited_index) {
      if (clone[i].size() != kWidth * sizeof(float) ||
          !std::all_of(clone[i].begin(), clone[i].end(),
                       [](uint8_t byte) { return byte == 0; }))
        return absl::FailedPreconditionError(
            "clone edited affine tensor is not zero");
    } else if (!SameBytes({original[i].data(), original[i].size()},
                          {clone[i].data(), clone[i].size()}))
      return absl::FailedPreconditionError("unexpected additional clone edit");
  }
  return absl::OkStatus();
}

absl::Status Run() {
  const fs::path checkpoint = absl::GetFlag(FLAGS_checkpoint);
  const fs::path directory = absl::GetFlag(FLAGS_output_dir);
  if (checkpoint.empty() || directory.empty() ||
      absl::GetFlag(FLAGS_tokenizer).empty() ||
      absl::GetFlag(FLAGS_expected_sentences) != 1024 ||
      absl::GetFlag(FLAGS_layers) != 8 ||
      absl::GetFlag(FLAGS_model_width) != kWidth ||
      absl::GetFlag(FLAGS_attention_heads) != 1 ||
      absl::GetFlag(FLAGS_feed_forward_width) != kFeatures)
    return absl::InvalidArgumentError(
        "required paths or fixed 8/16/1/64 architecture are invalid");
  ASSIGN_OR_RETURN(auto selected, SelectLines());
  ASSIGN_OR_RETURN(auto corpus, LoadTextCorpus(absl::GetFlag(FLAGS_corpus)));
  ASSIGN_OR_RETURN(auto sentences, ReadCorpusLines(corpus.text()));
  if (sentences.size() != 1024)
    return absl::InvalidArgumentError("requires 1024 corpus lines");
  ASSIGN_OR_RETURN(
      auto base, tokenizer::Gpt2Tokenizer::Load(absl::GetFlag(FLAGS_tokenizer)));
  ASSIGN_OR_RETURN(auto vocabulary,
                   tokenizer::CompactVocabularyTokenizer::LoadFromFile(
                       *base, checkpoint / "compact_vocabulary.tsv"));
  if (vocabulary->vocab_size() != kVocabulary ||
      base->eos_token_id() != vocabulary->original_eos_token_id())
    return absl::InvalidArgumentError(
        "requires the original compact 4475-token vocabulary");
  std::error_code error;
  if (!fs::create_directory(directory, error))
    return absl::InvalidArgumentError(
        absl::StrCat("output_dir must be fresh: ",
                     error ? error.message() : "directory already exists"));
  std::ofstream rows(directory / "conditions.tsv"),
      controls(directory / "controls.tsv"),
      manifest(directory / "manifest.tsv");
  for (auto* stream : {&rows, &controls, &manifest}) {
    if (!*stream)
      return absl::DataLossError("cannot create feature-audit reports");
    *stream << std::setprecision(17);
  }
  rows << "line\tvariant\tkeep_kind\tdeleted_channel\twinner\ttarget\tcorrect\t"
          "target_probability\ttarget_margin\ttarget_rank\tfixed_baseline_"
          "rival\t"
          "target_logit\tfixed_rival_logit\tfixed_rival_margin\tdynamic_rival\t"
          "dynamic_rival_logit\tdelta_probability\tdelta_fixed_margin\tdelta_"
          "dynamic_margin\n";
  controls << "line\tvariant\tcondition\tbitwise_all_logits\tmax_logit_delta\n";
  manifest << "checkpoint\t" << checkpoint.string() << "\ncorpus\t"
           << absl::GetFlag(FLAGS_corpus) << "\ntokenizer\t"
           << absl::GetFlag(FLAGS_tokenizer)
           << "\ninput\tfirst_five_tokens_future_EOS\nlabel_use\tscoring_only_"
              "not_model_input"
           << "\nmask_site\tfinal_GELU_query_row4"
           << "\nvariants\tseparate_checkpoint_clones_zero_only_weight97_or99"
           << "\ncached_upstream\toriginal_model_GELU_and_preMLP_residual_for_"
              "every_variant"
           << "\nfixed_rival\tstrongest_non_target_in_original_baseline"
           << "\nselected_controls\tlines1_80_258_406_411_631_full_empty_"
              "variants_and_original_deletions12_34"
           << "\ninterpretation\tconditional_single_deletion_sensitivity_not_"
              "necessary_or_exclusive_ownership"
           << "\nsource_checkpoint_files\tread_only\ncomplete\tfalse\n";
  manifest.flush();
  ASSIGN_OR_RETURN(auto executor, cuda::Executor::Create());
  const Gpt2Config config{.transformer_block_count = 8,
                          .model_width = kWidth,
                          .attention_heads = 1,
                          .feed_forward_width = kFeatures,
                          .vocabulary_size = kVocabulary,
                          .pad_vocabulary = false};
  std::array<std::unique_ptr<Layer>, 3> models;
  std::array<std::unique_ptr<FinalMlpSubsetTail>, 3> tails;
  const std::array<const char*, 3> variants{"original", "contraction_bias_zero",
                                            "final_norm_beta_zero"};
  std::vector<HostBytes> original_weights;
  absl::flat_hash_set<const void*> original_allocations;
  for (size_t variant = 0; variant < models.size(); ++variant) {
    ASSIGN_OR_RETURN(models[variant],
                     CreateGpt2(*executor, DataType::BF16, 0, config));
    RETURN_IF_ERROR(ReadFromDirectory(*executor, *models[variant],
                                      checkpoint.string(), false));
    const auto weights = models[variant]->weights();
    if (weights.size() != kExposedWeightCount)
      return absl::FailedPreconditionError(
          "unexpected CreateGpt2 master inventory");
    if (variant == 0) {
      ASSIGN_OR_RETURN(original_weights, CopyMasterBytes(*executor, *models[0]));
      for (const auto& weight : weights)
        original_allocations.insert(weight.data());
    } else {
      // No original model allocation may be edited through a cloned handle.
      for (const auto& weight : weights)
        if (original_allocations.contains(weight.data()))
          return absl::FailedPreconditionError(
              "private clone aliases original model storage");
      const size_t edited_index =
          variant == 1 ? kContractionBiasIndex : kFinalBetaIndex;
      if (weights[edited_index].size_bytes() != kWidth * sizeof(float))
        return absl::FailedPreconditionError(
            "cloned affine tensor has unexpected size");
      RETURN_IF_ERROR(
          cuda::CudaStatus(cudaMemsetAsync(weights[edited_index].data(), 0,
                                           weights[edited_index].size_bytes(),
                                           executor->stream()),
                           "zero one private cloned affine tensor"));
      ASSIGN_OR_RETURN(auto edited_bytes,
                       CopyMasterBytes(*executor, *models[variant]));
      RETURN_IF_ERROR(
          ValidateCloneBytes(original_weights, edited_bytes, edited_index));
    }
    ASSIGN_OR_RETURN(tails[variant], FinalMlpSubsetTail::Create(
                                         *executor, *models[variant], config));
  }
  const TokenTraceOptions plain{.vocabulary_size = kVocabulary,
                                .padding_token = vocabulary->eos_token_id()};
  TokenTraceOptions capture = plain;
  capture.capture_activations = true;
  const TokenTraceSite gelu_site{
      .scope = {"gpt2", "transformer_block_7", "ResidualLayer", "mlp"},
      .layer_name = "GeluLayer",
      .occurrence = 0};
  const TokenTraceSite residual_site{.scope = {"gpt2", "transformer_block_7"},
                                     .layer_name = "ResidualLayer",
                                     .occurrence = 0};
  size_t evaluated = 0, full_controls = 0, spot_controls = 0,
         source_post_controls = 0;
  const auto started = std::chrono::steady_clock::now();
  for (size_t index : selected) {
    const size_t line = index + 1;
    ASSIGN_OR_RETURN(auto tokens,
                     vocabulary->Encode(*executor, sentences[index]));
    if (tokens.size() < kPromptTokens || tokens.size() > kGpt2ContextLength)
      return absl::InvalidArgumentError(
          "invalid selected sentence token count");
    const absl::Span<const int> prefix(tokens.data(), kPromptTokens);
    const int target = tokens.size() == kPromptTokens
                           ? vocabulary->eos_token_id()
                           : tokens[kPromptTokens];
    // Every variant uses these ORIGINAL upstream rows. The only edited clone
    // tensors are downstream of both capture sites, which direct checks test.
    ASSIGN_OR_RETURN(auto trace,
                     TraceNextToken(*executor, *models[0], prefix, capture));
    std::array<uint16_t, kFeatures> gelu;
    std::array<uint16_t, kWidth> residual;
    RETURN_IF_ERROR(ExtractBf16Query(trace, gelu_site, absl::MakeSpan(gelu)));
    RETURN_IF_ERROR(
        ExtractBf16Query(trace, residual_site, absl::MakeSpan(residual)));
    ASSIGN_OR_RETURN(auto baseline, Score(trace.logits, target));
    if (baseline.winner != target)
      return absl::FailedPreconditionError(
          absl::StrCat("original baseline next token is wrong at line ", line));
    const int fixed_rival = baseline.rival;
    const double fixed_margin =
        static_cast<double>(trace.logits[target]) - trace.logits[fixed_rival];
    const bool spot = line == 80 || line == 406 || line == 411 || line == 1 ||
                      line == 258 || line == 631;
    for (size_t variant = 0; variant < tails.size(); ++variant) {
      std::vector<FinalMlpSubsetInput> inputs{{gelu, residual, kFullSubset},
                                              {gelu, residual, 0}};
      if (variant == 0)
        for (int channel = 0; channel < kFeatures; ++channel)
          inputs.push_back(
              {gelu, residual, kFullSubset ^ (FeatureSubset{1} << channel)});
      ASSIGN_OR_RETURN(auto logits, tails[variant]->Evaluate(*executor, inputs));
      const int stride = tails[variant]->logit_stride();
      for (size_t condition = 0; condition < inputs.size(); ++condition) {
        const absl::Span<const float> values(logits.data() + condition * stride,
                                             kVocabulary);
        const char* name = condition == 0   ? "full"
                           : condition == 1 ? "empty"
                                            : "single_deletion";
        ASSIGN_OR_RETURN(auto score, Score(values, target));
        const double pair_margin =
            static_cast<double>(values[target]) - values[fixed_rival];
        rows << line << '\t' << variants[variant] << '\t' << name << '\t'
             << (condition < 2 ? -1 : static_cast<int>(condition - 2)) << '\t'
             << score.winner << '\t' << target << '\t'
             << (score.winner == target) << '\t' << score.probability << '\t'
             << score.margin << '\t' << score.target_rank << '\t' << fixed_rival
             << '\t' << values[target] << '\t' << values[fixed_rival] << '\t'
             << pair_margin << '\t' << score.rival << '\t'
             << values[score.rival] << '\t'
             << score.probability - baseline.probability << '\t'
             << pair_margin - fixed_margin << '\t'
             << score.margin - baseline.margin << '\n';
        ++evaluated;
        const bool identity = variant == 0 && condition == 0;
        // Six facts fixed before the corpus-wide audit, never selected from
        // this run's effects. Conditions14/36 are deletions of channels12/34.
        const bool direct_check =
            spot && ((variant != 0 && condition < 2) ||
                     (variant == 0 &&
                      (condition == 1 || condition == 14 || condition == 36)));
        if (!identity && !direct_check)
          continue;
        std::vector<float> expected = trace.logits;
        if (direct_check) {
          TokenTraceOptions intervention = plain;
          const TokenTracePatch patch{
              .site = gelu_site,
              .rows = TokenTraceRows::kQuery,
              .first_channel = condition < 2 ? 0 : condition - 2,
              .channel_count = condition < 2 ? size_t{kFeatures} : size_t{1}};
          if (condition != 0)
            intervention.patches = {&patch, 1};
          ASSIGN_OR_RETURN(
              auto reference,
              TraceNextToken(*executor, *models[variant], prefix, intervention));
          expected = std::move(reference.logits);
          ++spot_controls;
        }
        double maximum_delta = 0;
        for (int token = 0; token < kVocabulary; ++token)
          maximum_delta = std::max(
              maximum_delta,
              std::abs(static_cast<double>(values[token]) - expected[token]));
        const bool equal = SameLogits(values, expected);
        controls << line << '\t' << variants[variant] << '\t' << name;
        if (condition >= 2)
          controls << '_' << condition - 2;
        controls << '\t' << equal << '\t' << maximum_delta << '\n';
        if (!equal) {
          controls.flush();
          return absl::FailedPreconditionError(
              "cached feature audit differs from direct full-model logits");
        }
        full_controls += identity;
      }
    }
    ASSIGN_OR_RETURN(auto post,
                     TraceNextToken(*executor, *models[0], prefix, plain));
    const bool unchanged = SameLogits(trace.logits, post.logits);
    controls << line << "\toriginal\tordinary_post\t" << unchanged << '\t';
    double maximum_delta = 0;
    for (int token = 0; token < kVocabulary; ++token)
      maximum_delta = std::max(
          maximum_delta, std::abs(static_cast<double>(trace.logits[token]) -
                                  post.logits[token]));
    controls << maximum_delta << '\n';
    if (!unchanged) {
      controls.flush();
      return absl::FailedPreconditionError(
          "original ordinary forward changed after feature audit");
    }
    ++source_post_controls;
    rows.flush();
    controls.flush();
    if (!rows || !controls)
      return absl::DataLossError("failed writing feature audit");
    if (selected.size() <= 6 || source_post_controls % 64 == 0)
      std::cout << "facts=" << source_post_controls
                << " conditions=" << evaluated << " seconds="
                << std::chrono::duration<double>(
                       std::chrono::steady_clock::now() - started)
                       .count()
                << std::endl;
  }
  ASSIGN_OR_RETURN(auto final_weights, CopyMasterBytes(*executor, *models[0]));
  if (final_weights.size() != original_weights.size())
    return absl::FailedPreconditionError(
        "original model tensor inventory changed");
  for (size_t i = 0; i < original_weights.size(); ++i)
    if (!SameBytes({original_weights[i].data(), original_weights[i].size()},
                   {final_weights[i].data(), final_weights[i].size()}))
      return absl::FailedPreconditionError(
          "original model master bytes changed");
  manifest << "case_count\t" << selected.size() << "\ncondition_count\t"
           << evaluated << "\noriginal_bitwise_controls\t" << full_controls
           << "\nfull_model_spot_controls\t" << spot_controls
           << "\noriginal_ordinary_post_controls\t" << source_post_controls
           << "\nclone_only_intended_affine_bytes_changed\ttrue"
           << "\noriginal_master_bytes_unchanged\ttrue\nseconds\t"
           << std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                            started)
                  .count()
           << "\ncomplete\ttrue\n";
  for (auto* stream : {&rows, &controls, &manifest}) {
    stream->close();
    if (!*stream)
      return absl::DataLossError("feature audit report write failed");
  }
  return absl::OkStatus();
}

}  // namespace
}  // namespace pluto::llm::one_shot_memorizer

int main(int argc, char** argv) {
  if (absl::ParseCommandLine(argc, argv).size() != 1)
    return 2;
  const auto status = pluto::llm::one_shot_memorizer::Run();
  if (!status.ok()) {
    std::cerr << status << '\n';
    return 1;
  }
}
