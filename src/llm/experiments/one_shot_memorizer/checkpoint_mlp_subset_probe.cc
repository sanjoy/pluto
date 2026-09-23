// Target-guided final-MLP interventions, not a decoder of exclusive fact
// owners. Captures five-token queries once, then exhaustively masks cached GELU
// rows through the production projection/add/normalization/head GPU kernels.
#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <bit>
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
#include "src/dataset/gpt2_detokenizer.h"
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
          "One fact per line");
ABSL_FLAG(std::string, output_dir, "", "Fresh directory for subset reports");
ABSL_FLAG(std::vector<std::string>, lines,
          (std::vector<std::string>{"80", "406", "411", "1", "258", "631"}),
          "Distinct one-based corpus lines, or all");
ABSL_FLAG(int, expected_sentences, 1024, "Required corpus line count");
ABSL_FLAG(int, layers, 8, "Transformer block count");
ABSL_FLAG(int, model_width, 16, "Must be 16 for this experiment");
ABSL_FLAG(int, attention_heads, 1, "Attention head count");
ABSL_FLAG(int, feed_forward_width, 64, "Must be 64 for this experiment");
ABSL_FLAG(int, max_keep, 2,
          "Exhaustive maximum retained-feature cardinality, 0..4");
ABSL_FLAG(bool, stop_after_first_success, true,
          "Finish the first successful cardinality, then omit larger levels");
ABSL_FLAG(int, batch_rows, 2048,
          "Rows per tail batch, including two identity controls, 3..2048");
ABSL_FLAG(bool, write_all_scores, false,
          "Also write scores for failed masks; potentially very large");
ABSL_FLAG(bool, spotcheck_full_model, true,
          "Compare first successful and failed mask per level with an actual "
          "full-model patched forward, bitwise over every vocabulary logit");

namespace pluto::llm::one_shot_memorizer {
namespace {
namespace fs = std::filesystem;
constexpr size_t kPromptTokens = 5;
constexpr int kWidth = 16;
constexpr int kFeatures = 64;
constexpr FeatureSubset kFullSubset = ~FeatureSubset{0};

// Keep text fields unambiguous even for partial token bytes or special paths.
std::string EscapeTsv(absl::string_view text) {
  constexpr char kHex[] = "0123456789abcdef";
  std::string result;
  for (unsigned char ch : text)
    if (ch == '\\')
      result += "\\\\";
    else if (ch == '\n')
      result += "\\n";
    else if (ch == '\r')
      result += "\\r";
    else if (ch == '\t')
      result += "\\t";
    else if (ch < 32 || ch >= 127) {
      result += "\\x";
      result += kHex[ch >> 4];
      result += kHex[ch & 15];
    } else
      result += static_cast<char>(ch);
  return result;
}

absl::StatusOr<std::vector<std::string>> ReadCorpusLines(
    absl::string_view text) {
  std::vector<std::string> lines;
  while (!text.empty()) {
    const size_t end = text.find('\n');
    auto line = text.substr(0, end);
    if (!line.empty() && line.back() == '\r')
      line.remove_suffix(1);
    if (line.empty() || line.find_first_not_of(" \t\r") == line.npos)
      return absl::InvalidArgumentError("empty or blank corpus line");
    lines.emplace_back(line);
    if (end == text.npos)
      break;
    text.remove_prefix(end + 1);
  }
  return lines;
}

absl::StatusOr<std::vector<size_t>> SelectLines(size_t count) {
  std::vector<size_t> result;
  const auto requested = absl::GetFlag(FLAGS_lines);
  if (requested == std::vector<std::string>{"all"}) {
    result.resize(count);
    std::iota(result.begin(), result.end(), 0);
    return result;
  }
  absl::flat_hash_set<size_t> seen;
  for (const auto& text : requested) {
    size_t line;
    if (text.empty() ||
        !std::all_of(text.begin(), text.end(),
                     [](char ch) { return ch >= '0' && ch <= '9'; }) ||
        !absl::SimpleAtoi(text, &line) || line < 1 || line > count ||
        !seen.insert(line).second)
      return absl::InvalidArgumentError(
          "lines must be distinct unsigned one-based IDs, or all");
    result.push_back(line - 1);
  }
  if (result.empty())
    return absl::InvalidArgumentError("no selected corpus lines");
  return result;
}

struct Prediction {
  int winner = 0;  // Stable argmax: the lowest ID wins equal logits.
  int rival = 0;   // Strongest non-target class, with the same tie rule.
  size_t target_rank = 1;
  double target_margin = 0;
};

// Cheap first pass for every mask. Softmax is needed only for successful masks
// or when all-score output was explicitly requested; do not exponentiate the
// entire vocabulary for the overwhelmingly many rejected candidate masks.
absl::StatusOr<Prediction> InspectLogits(absl::Span<const float> logits,
                                         int target) {
  if (logits.size() < 2 || target < 0 ||
      static_cast<size_t>(target) >= logits.size())
    return absl::InvalidArgumentError("invalid subset-scoring shape or target");
  Prediction result{.rival = target == 0 ? 1 : 0};
  for (size_t token = 0; token < logits.size(); ++token) {
    if (!std::isfinite(logits[token]))
      return absl::DataLossError("nonfinite subset logit");
    if (logits[token] > logits[result.winner])
      result.winner = static_cast<int>(token);
    if (static_cast<int>(token) != target) {
      if (logits[token] > logits[result.rival])
        result.rival = static_cast<int>(token);
      result.target_rank +=
          logits[token] > logits[target] ||
          (logits[token] == logits[target] && static_cast<int>(token) < target);
    }
  }
  result.target_margin =
      static_cast<double>(logits[target]) - logits[result.rival];
  return result;
}

double TargetProbability(absl::Span<const float> logits, int target,
                         int winner) {
  const double maximum = logits[winner];
  double denominator = 0;
  for (float value : logits)
    denominator += std::exp(static_cast<double>(value) - maximum);
  return std::exp(static_cast<double>(logits[target]) - maximum) / denominator;
}

// This is byte-preserving extraction, not a float decode/re-encode. Match the
// full hook identity so identically named layers elsewhere cannot be selected.
absl::Status ExtractBf16Query(const TokenTraceResult& trace,
                              const TokenTraceSite& site,
                              absl::Span<uint16_t> destination) {
  const TokenTraceActivation* found = nullptr;
  for (const auto& activation : trace.activations)
    if (activation.site.scope == site.scope &&
        activation.site.layer_name == site.layer_name &&
        activation.site.occurrence == site.occurrence &&
        activation.site.output_index == site.output_index) {
      if (found != nullptr)
        return absl::FailedPreconditionError(
            "ambiguous cached activation site");
      found = &activation;
    }
  if (found == nullptr || found->data_type != DataType::BF16 ||
      found->channels != destination.size() ||
      found->first_row > trace.query_row ||
      trace.query_row - found->first_row >= found->row_count ||
      found->bytes.size() !=
          found->row_count * destination.size() * sizeof(uint16_t))
    return absl::FailedPreconditionError(
        "cached activation has wrong site/type/shape");
  const size_t offset = (trace.query_row - found->first_row) *
                        destination.size() * sizeof(uint16_t);
  std::memcpy(destination.data(), found->bytes.data() + offset,
              destination.size() * sizeof(uint16_t));
  return absl::OkStatus();
}

bool SameLogits(absl::Span<const float> first, absl::Span<const float> second) {
  return first.size() == second.size() &&
         std::memcmp(first.data(), second.data(),
                     first.size() * sizeof(float)) == 0;
}

absl::StatusOr<TokenTraceResult> TraceMaskedGelu(
    cuda::Executor& executor, const Layer& model, absl::Span<const int> prefix,
    const TokenTraceOptions& plain, const TokenTraceSite& gelu_site,
    absl::Span<const uint16_t> original_gelu, FeatureSubset subset) {
  ASSIGN_OR_RETURN(auto masked, ApplyBf16FeatureSubset(original_gelu, subset));
  TokenTraceActivation donor{
      .site = gelu_site,
      .data_type = DataType::BF16,
      .first_row = kPromptTokens - 1,
      .row_count = 1,
      .channels = kFeatures,
      .bytes = std::vector<uint8_t>(masked.size() * sizeof(uint16_t))};
  std::memcpy(donor.bytes.data(), masked.data(), donor.bytes.size());
  const TokenTracePatch patch{.site = gelu_site,
                              .rows = TokenTraceRows::kQuery,
                              .channel_count = kFeatures,
                              .replacement = TokenTraceReplacement::kDonor,
                              .donor = &donor};
  TokenTraceOptions intervention = plain;
  intervention.patches = absl::Span<const TokenTracePatch>(&patch, 1);
  return TraceNextToken(executor, model, prefix, intervention);
}

using HostBytes = cuda::PageLockedHostArray<uint8_t>;

// Snapshot all exposed tensors, including the tied alias. Kept host storage is
// pinned; every CUDA transfer completes before the snapshots are inspected.
absl::StatusOr<std::vector<HostBytes>> CopyMasterBytes(cuda::Executor& executor,
                                                       const Layer& model) {
  std::vector<HostBytes> result;
  for (const auto& weight : model.weights()) {
    ASSIGN_OR_RETURN(auto host,
                     HostBytes::Allocate(executor, weight.size_bytes()));
    RETURN_IF_ERROR(cuda::CudaStatus(
        cudaMemcpyAsync(host.data(), weight.data(), host.size_bytes(),
                        cudaMemcpyDeviceToHost, executor.stream()),
        "snapshot model master bytes"));
    result.push_back(std::move(host));
  }
  RETURN_IF_ERROR(executor.Synchronize());
  return result;
}

std::string ChannelList(FeatureSubset subset) {
  std::string text;
  for (int feature = 0; feature < kFeatures; ++feature)
    if (subset & (FeatureSubset{1} << feature))
      absl::StrAppend(&text, text.empty() ? "" : ",", feature);
  return text.empty() ? "-" : text;
}

void WriteMaskScore(std::ostream& output, size_t line, int cardinality,
                    FeatureSubset subset, int target,
                    const Prediction& prediction, double probability) {
  output << line << '\t' << cardinality << "\t0x" << std::hex << std::setw(16)
         << std::setfill('0') << subset << std::dec << std::setfill(' ') << '\t'
         << ChannelList(subset) << '\t' << target << '\t' << prediction.winner
         << '\t' << (prediction.winner == target) << '\t' << probability << '\t'
         << prediction.target_margin << '\t' << prediction.rival << '\t'
         << prediction.target_rank << '\n';
}

absl::Status Run() {
  const fs::path checkpoint = absl::GetFlag(FLAGS_checkpoint);
  const fs::path directory = absl::GetFlag(FLAGS_output_dir);
  const int max_keep = absl::GetFlag(FLAGS_max_keep);
  const int batch_rows = absl::GetFlag(FLAGS_batch_rows);
  if (checkpoint.empty() || directory.empty() ||
      absl::GetFlag(FLAGS_tokenizer).empty() || max_keep < 0 || max_keep > 4 ||
      batch_rows < 3 || batch_rows > 2048 ||
      absl::GetFlag(FLAGS_expected_sentences) < 1 ||
      absl::GetFlag(FLAGS_layers) < 1 ||
      absl::GetFlag(FLAGS_model_width) != kWidth ||
      absl::GetFlag(FLAGS_feed_forward_width) != kFeatures ||
      absl::GetFlag(FLAGS_attention_heads) < 1 ||
      kWidth % absl::GetFlag(FLAGS_attention_heads) != 0)
    return absl::InvalidArgumentError(
        "invalid required paths, model dimensions, or subset limits");
  ASSIGN_OR_RETURN(auto corpus, LoadTextCorpus(absl::GetFlag(FLAGS_corpus)));
  ASSIGN_OR_RETURN(auto corpus_lines, ReadCorpusLines(corpus.text()));
  if (corpus_lines.size() !=
      static_cast<size_t>(absl::GetFlag(FLAGS_expected_sentences)))
    return absl::InvalidArgumentError(
        "corpus line count differs from expected_sentences");
  ASSIGN_OR_RETURN(auto selected, SelectLines(corpus_lines.size()));
  ASSIGN_OR_RETURN(
      auto base, tokenizer::Gpt2Tokenizer::Load(absl::GetFlag(FLAGS_tokenizer)));
  ASSIGN_OR_RETURN(auto detokenizer, tokenizer::Gpt2Detokenizer::Load(
                                         absl::GetFlag(FLAGS_tokenizer)));
  ASSIGN_OR_RETURN(auto vocabulary,
                   tokenizer::CompactVocabularyTokenizer::LoadFromFile(
                       *base, checkpoint / "compact_vocabulary.tsv"));
  if (base->eos_token_id() != vocabulary->original_eos_token_id() ||
      vocabulary->vocab_size() < 2)
    return absl::InvalidArgumentError(
        "tokenizer/vocabulary EOS or size mismatch");
  const Gpt2Config config{
      .transformer_block_count = absl::GetFlag(FLAGS_layers),
      .model_width = kWidth,
      .attention_heads = absl::GetFlag(FLAGS_attention_heads),
      .feed_forward_width = kFeatures,
      .vocabulary_size = vocabulary->vocab_size(),
      .pad_vocabulary = false};
  RETURN_IF_ERROR(config.Validate());
  std::error_code error;
  if (!fs::create_directory(directory, error))
    return absl::InvalidArgumentError(
        absl::StrCat("output_dir must be fresh: ",
                     error ? error.message() : "directory already exists"));
  std::ofstream levels(directory / "levels.tsv"),
      successful(directory / "successful_masks.tsv"),
      controls(directory / "controls.tsv"), cases(directory / "cases.tsv"),
      states(directory / "cached_activations.tsv"),
      spotchecks(directory / "full_model_spotchecks.tsv"),
      manifest(directory / "manifest.tsv");
  std::ofstream all_scores;
  if (absl::GetFlag(FLAGS_write_all_scores))
    all_scores.open(directory / "all_masks.tsv");
  for (auto* stream : {&levels, &successful, &controls, &cases, &states,
                       &spotchecks, &manifest}) {
    if (!*stream)
      return absl::DataLossError("failed opening subset report");
    *stream << std::setprecision(17);
  }
  if (absl::GetFlag(FLAGS_write_all_scores) && !all_scores)
    return absl::DataLossError("failed opening all-mask report");
  all_scores << std::setprecision(17);
  constexpr const char* kScoreHeader =
      "line_1based\tkeep_count\tkeep_mask_hex\tkeep_channels\ttarget_compact\t"
      "predicted_compact\tcorrect\ttarget_probability\ttarget_margin\trival_"
      "compact\ttarget_rank\n";
  successful << kScoreHeader;
  if (all_scores.is_open())
    all_scores << kScoreHeader;
  levels << "line_1based\tkeep_count\ttested\tcorrect\tminimum_"
            "proven\tcomplete_level\tseconds\n";
  controls << "line_1based\tkeep_count\tfirst_mask_index\tmask_count\tbatch_"
              "rows_padded\t"
              "control_row\tbitwise_equal\n";
  cases << "line_1based\tprefix_text\ttarget_text\ttarget_compact\tbaseline_"
           "probability\t"
           "baseline_margin\tbaseline_rank\n";
  states << "line_1based\tactivation\tchannel\tbf16_bits\n";
  spotchecks
      << "line_1based\tkeep_count\tkeep_mask_hex\tselection\tbitwise_equal\t"
         "cached_winner\tfull_model_winner\tcached_margin\tfull_model_margin\n";
  manifest
      << "checkpoint\t" << EscapeTsv(checkpoint.string()) << "\ncorpus\t"
      << EscapeTsv(absl::GetFlag(FLAGS_corpus)) << "\ntokenizer\t"
      << EscapeTsv(absl::GetFlag(FLAGS_tokenizer)) << "\nmax_keep\t" << max_keep
      << "\nbatch_rows_limit\t" << batch_rows << "\nstop_after_first_success\t"
      << absl::GetFlag(FLAGS_stop_after_first_success) << "\nwrite_all_scores\t"
      << absl::GetFlag(FLAGS_write_all_scores) << "\nlayers\t"
      << config.transformer_block_count
      << "\nmodel_width\t16\nfeed_forward_width\t64\nattention_heads\t"
      << config.attention_heads << "\nvocabulary_size\t"
      << config.vocabulary_size
      << "\ninput\tfirst_five_tokens_future_EOS\nlabel_use\tsupervised_search_"
         "scoring_not_model_input"
      << "\nintervention\tfinal_GELU_query_only_keep_original_raw_BF16_zero_"
         "others"
      << "\nprojection_bias\tpreserved\nresidual_skip\tpreserved"
      << "\ncomputation\tproduction_BF16_contraction_add_LayerNorm_tied_head"
      << "\nminimum_scope\tfixed_query_and_unchanged_upstream_computation_not_"
         "fact_ownership"
      << "\ncontrols\tfull_mask_bookends_each_actual_batch_geometry_all_vocab_"
         "logits_bitwise"
      << "\nspotcheck_full_model\t" << absl::GetFlag(FLAGS_spotcheck_full_model)
      << "\nspotcheck_selection\tfirst_success_and_first_failure_per_level"
      << "\ncomplete\tfalse\n";
  manifest.flush();
  ASSIGN_OR_RETURN(auto executor, cuda::Executor::Create());
  ASSIGN_OR_RETURN(auto model, CreateGpt2(*executor, DataType::BF16, 0, config));
  RETURN_IF_ERROR(
      ReadFromDirectory(*executor, *model, checkpoint.string(), false));
  ASSIGN_OR_RETURN(auto original_weights, CopyMasterBytes(*executor, *model));
  ASSIGN_OR_RETURN(auto tail,
                   FinalMlpSubsetTail::Create(*executor, *model, config));
  const int stride = tail->logit_stride();
  const int eos = vocabulary->eos_token_id();
  const TokenTraceOptions plain{.vocabulary_size = config.vocabulary_size,
                                .padding_token = eos};
  TokenTraceOptions capture = plain;
  capture.capture_activations = true;
  const std::string block =
      absl::StrCat("transformer_block_", config.transformer_block_count - 1);
  const TokenTraceSite gelu_site{
      .scope = {"gpt2", block, "ResidualLayer", "mlp"},
      .layer_name = "GeluLayer",
      .occurrence = 0};
  const TokenTraceSite residual_site{
      .scope = {"gpt2", block}, .layer_name = "ResidualLayer", .occurrence = 0};
  size_t mask_count = 0, control_count = 0, post_controls = 0,
         spotcheck_count = 0;
  const auto started = std::chrono::steady_clock::now();
  for (size_t index : selected) {
    ASSIGN_OR_RETURN(auto tokens,
                     vocabulary->Encode(*executor, corpus_lines[index]));
    if (tokens.size() < kPromptTokens || tokens.size() > kGpt2ContextLength)
      return absl::InvalidArgumentError(
          "selected fact has unsupported token count");
    const absl::Span<const int> prefix(tokens.data(), kPromptTokens);
    const int target =
        tokens.size() == kPromptTokens ? eos : tokens[kPromptTokens];
    // The suffix supplies evaluation metadata only. No suffix token can enter
    // TraceNextToken, whose only input here is the first five-token span.
    ASSIGN_OR_RETURN(auto baseline,
                     TraceNextToken(*executor, *model, prefix, plain));
    ASSIGN_OR_RETURN(auto trace,
                     TraceNextToken(*executor, *model, prefix, capture));
    if (!SameLogits(baseline.logits, trace.logits))
      return absl::FailedPreconditionError(
          "capture and ordinary baseline logits differ");
    ASSIGN_OR_RETURN(auto baseline_score,
                     InspectLogits(baseline.logits, target));
    if (baseline_score.winner != target)
      return absl::FailedPreconditionError(
          absl::StrCat("baseline next token is wrong at line ", index + 1));
    std::array<uint16_t, kFeatures> gelu;
    std::array<uint16_t, kWidth> residual;
    RETURN_IF_ERROR(ExtractBf16Query(trace, gelu_site, absl::MakeSpan(gelu)));
    RETURN_IF_ERROR(
        ExtractBf16Query(trace, residual_site, absl::MakeSpan(residual)));
    for (size_t channel = 0; channel < gelu.size(); ++channel)
      states << index + 1 << "\tgelu\t" << channel << '\t' << gelu[channel]
             << '\n';
    for (size_t channel = 0; channel < residual.size(); ++channel)
      states << index + 1 << "\tpre_mlp_residual\t" << channel << '\t'
             << residual[channel] << '\n';
    std::vector<int> original_prefix;
    for (int token : prefix) {
      ASSIGN_OR_RETURN(int original, vocabulary->OriginalId(token));
      original_prefix.push_back(original);
    }
    ASSIGN_OR_RETURN(int original_target, vocabulary->OriginalId(target));
    ASSIGN_OR_RETURN(auto prefix_text, detokenizer->Decode(original_prefix));
    ASSIGN_OR_RETURN(auto target_text,
                     detokenizer->Decode({&original_target, 1}));
    cases << index + 1 << '\t' << EscapeTsv(prefix_text) << '\t'
          << EscapeTsv(target_text) << '\t' << target << '\t'
          << TargetProbability(baseline.logits, target, baseline_score.winner)
          << '\t' << baseline_score.target_margin << '\t'
          << baseline_score.target_rank << '\n';
    int first_success = -1;
    for (int cardinality = 0; cardinality <= max_keep; ++cardinality) {
      ASSIGN_OR_RETURN(auto subsets,
                       EnumerateFeatureSubsets(kFeatures, cardinality));
      size_t correct_count = 0, tested = 0;
      bool checked_success = false, checked_failure = false;
      const auto level_started = std::chrono::steady_clock::now();
      for (size_t begin = 0; begin < subsets.size();) {
        const size_t count = std::min(subsets.size() - begin,
                                      static_cast<size_t>(batch_rows - 2));
        std::vector<FinalMlpSubsetInput> batch;
        batch.reserve(count + 2);
        // Same-input all-kept rows bracket EACH actual batch, so different GEMM
        // row geometries must reproduce every original vocabulary logit.
        batch.push_back({gelu, residual, kFullSubset});
        for (size_t item = 0; item < count; ++item)
          batch.push_back({gelu, residual, subsets[begin + item]});
        batch.push_back({gelu, residual, kFullSubset});
        ASSIGN_OR_RETURN(auto logits, tail->Evaluate(*executor, batch));
        for (size_t control_row : {size_t{0}, batch.size() - 1}) {
          const absl::Span<const float> values(
              logits.data() + control_row * stride, config.vocabulary_size);
          const bool equal = SameLogits(values, baseline.logits);
          controls << index + 1 << '\t' << cardinality << '\t' << begin << '\t'
                   << count << '\t' << (batch.size() + 15) / 16 * 16 << '\t'
                   << control_row << '\t' << equal << '\n';
          if (!equal) {
            controls.flush();
            return absl::FailedPreconditionError(
                "cached full-mask logits differ at actual batch geometry");
          }
          ++control_count;
        }
        for (size_t item = 0; item < count; ++item) {
          const absl::Span<const float> values(
              logits.data() + (item + 1) * stride, config.vocabulary_size);
          ASSIGN_OR_RETURN(auto prediction, InspectLogits(values, target));
          const bool correct = prediction.winner == target;
          correct_count += correct;
          if (absl::GetFlag(FLAGS_spotcheck_full_model) &&
              !(correct ? checked_success : checked_failure)) {
            ASSIGN_OR_RETURN(
                auto patched,
                TraceMaskedGelu(*executor, *model, prefix, plain, gelu_site,
                                gelu, subsets[begin + item]));
            ASSIGN_OR_RETURN(auto full_prediction,
                             InspectLogits(patched.logits, target));
            const bool equal = SameLogits(values, patched.logits);
            spotchecks << index + 1 << '\t' << cardinality << "\t0x" << std::hex
                       << std::setw(16) << std::setfill('0')
                       << subsets[begin + item] << std::dec << std::setfill(' ')
                       << '\t' << (correct ? "first_success" : "first_failure")
                       << '\t' << equal << '\t' << prediction.winner << '\t'
                       << full_prediction.winner << '\t'
                       << prediction.target_margin << '\t'
                       << full_prediction.target_margin << '\n';
            if (!equal) {
              spotchecks.flush();
              return absl::FailedPreconditionError(
                  "cached counterfactual differs from full-model patched "
                  "logits");
            }
            (correct ? checked_success : checked_failure) = true;
            ++spotcheck_count;
          }
          if (correct || all_scores.is_open()) {
            const double probability =
                TargetProbability(values, target, prediction.winner);
            if (correct)
              WriteMaskScore(successful, index + 1, cardinality,
                             subsets[begin + item], target, prediction,
                             probability);
            if (all_scores.is_open())
              WriteMaskScore(all_scores, index + 1, cardinality,
                             subsets[begin + item], target, prediction,
                             probability);
          }
        }
        begin += count;
        tested += count;
        mask_count += count;
      }
      const bool proves_minimum = correct_count != 0 && first_success < 0;
      if (proves_minimum)
        first_success = cardinality;
      levels << index + 1 << '\t' << cardinality << '\t' << tested << '\t'
             << correct_count << '\t' << proves_minimum << "\t1\t"
             << std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                              level_started)
                    .count()
             << '\n';
      levels.flush();
      successful.flush();
      controls.flush();
      spotchecks.flush();
      if (all_scores.is_open())
        all_scores.flush();
      if (!levels || !successful || !controls || !spotchecks ||
          (all_scores.is_open() && !all_scores))
        return absl::DataLossError("failed writing subset level reports");
      std::cout << "line=" << index + 1 << " keep=" << cardinality
                << " tested=" << tested << " correct=" << correct_count
                << std::endl;
      if (first_success >= 0 && absl::GetFlag(FLAGS_stop_after_first_success))
        break;
    }
    ASSIGN_OR_RETURN(auto post,
                     TraceNextToken(*executor, *model, prefix, plain));
    if (!SameLogits(post.logits, baseline.logits))
      return absl::FailedPreconditionError(
          "source model ordinary logits changed after subset search");
    ++post_controls;
    manifest << "line_" << index + 1 << "_minimum_keep_cardinality\t";
    if (first_success >= 0)
      manifest << first_success;
    else
      manifest << '>' << max_keep;
    manifest << '\n';
    manifest.flush();
  }
  ASSIGN_OR_RETURN(auto final_weights, CopyMasterBytes(*executor, *model));
  if (original_weights.size() != final_weights.size())
    return absl::FailedPreconditionError(
        "source model tensor inventory changed");
  for (size_t i = 0; i < original_weights.size(); ++i)
    if (original_weights[i].size() != final_weights[i].size() ||
        std::memcmp(original_weights[i].data(), final_weights[i].data(),
                    original_weights[i].size()) != 0)
      return absl::FailedPreconditionError("source model master bytes changed");
  manifest << "case_count\t" << selected.size() << "\nmasked_conditions\t"
           << mask_count << "\nbitwise_tail_controls\t" << control_count
           << "\nbitwise_source_post_controls\t" << post_controls
           << "\nbitwise_full_model_mask_spotchecks\t" << spotcheck_count
           << "\nsource_master_bytes_unchanged\ttrue\nelapsed_seconds\t"
           << std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                            started)
                  .count()
           << "\ncomplete\ttrue\n";
  for (auto* stream : {&levels, &successful, &controls, &cases, &states,
                       &spotchecks, &manifest}) {
    stream->close();
    if (!*stream)
      return absl::DataLossError("failed closing subset report");
  }
  if (all_scores.is_open()) {
    all_scores.close();
    if (!all_scores)
      return absl::DataLossError("failed closing all-mask report");
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
