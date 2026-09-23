// Traces the first failed greedy token of a saved quadratic construction.
// The supplied prefix ends BEFORE the wrong token. Labels only score results;
// all interventions execute the live production model on that same prefix.
#include <cuda_runtime_api.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "src/cuda/page_locked_host_array.h"
#include "src/dataset/compact_vocabulary.h"
#include "src/dataset/dataset.h"
#include "src/dataset/gpt2_detokenizer.h"
#include "src/dataset/gpt2_tokenizer.h"
#include "src/dataset/padded_line_dataset.h"
#include "src/llm/checkpoint.h"
#include "src/llm/experiments/one_shot_memorizer/mlp_probe.h"
#include "src/llm/experiments/one_shot_memorizer/quadratic_features.h"
#include "src/llm/experiments/one_shot_memorizer/quadratic_probe_artifacts.h"
#include "src/llm/experiments/one_shot_memorizer/token_trace.h"
#include "src/llm/gpt2.h"
#include "src/llm/layers/combinators.h"
#include "src/llm/layers/fully_connected.h"
#include "src/util/status_macros.h"

ABSL_FLAG(std::string, checkpoint, "", "Original trained checkpoint");
ABSL_FLAG(std::string, tokenizer, "", "Base GPT-2 tokenizer directory");
ABSL_FLAG(std::string, corpus, "testdata/general_facts_dataset.txt",
          "Original fixed corpus");
ABSL_FLAG(std::string, construction_dir, "",
          "Completed quadratic probe output with coefficients/first_failures");
ABSL_FLAG(std::string, output_dir, "", "Fresh output directory");

namespace pluto::llm::one_shot_memorizer {
namespace {
namespace fs = std::filesystem;
constexpr int kVocabulary = 4475;
constexpr int kWidth = 16;
constexpr int kBlocks = 8;

std::string Escape(absl::string_view value) {
  const char* hex = "0123456789abcdef";
  std::string result;
  for (unsigned char ch : value)
    if (ch == '\\')
      result += "\\\\";
    else if (ch < 32 || ch >= 127) {
      result += "\\x";
      result += hex[ch >> 4];
      result += hex[ch & 15];
    } else
      result += static_cast<char>(ch);
  return result;
}

absl::StatusOr<std::string> Decode(
    absl::Span<const int> ids,
    const tokenizer::CompactVocabularyTokenizer& compact,
    const tokenizer::Gpt2Detokenizer& decoder) {
  std::vector<int> original;
  for (int id : ids) {
    ASSIGN_OR_RETURN(int base, compact.OriginalId(id));
    original.push_back(base);
  }
  ASSIGN_OR_RETURN(auto text, decoder.Decode(original));
  return Escape(text);
}

absl::StatusOr<std::vector<std::vector<uint8_t>>> Snapshot(
    cuda::Executor& executor, const Layer& layer) {
  std::vector<std::vector<uint8_t>> result;
  for (const auto& weight : layer.weights()) {
    ASSIGN_OR_RETURN(auto host, cuda::PageLockedHostArray<uint8_t>::Allocate(
                                    executor, weight.size_bytes()));
    RETURN_IF_ERROR(cuda::CudaStatus(
        cudaMemcpyAsync(host.data(), weight.data(), weight.size_bytes(),
                        cudaMemcpyDeviceToHost, executor.stream()),
        "snapshot trace source weights"));
    RETURN_IF_ERROR(executor.Synchronize());
    result.emplace_back(host.begin(), host.end());
  }
  return result;
}

absl::StatusOr<std::unique_ptr<ComposedLayer>> MakeReplacement(
    cuda::Executor& executor, const QuadraticCoefficients& coefficients) {
  ASSIGN_OR_RETURN(auto projection,
                   FullyConnectedLayer::Create(
                       executor, 152, 16, DataType::BF16, kGpt2ContextLength));
  const absl::Span<const float> tensors[] = {coefficients.weights,
                                             coefficients.bias};
  for (int part = 0; part < 2; ++part) {
    ASSIGN_OR_RETURN(auto host, cuda::PageLockedHostArray<float>::CopyFrom(
                                    executor, tensors[part]));
    if (host.size_bytes() != projection->weights()[part].size_bytes())
      return absl::InvalidArgumentError("malformed saved quadratic tensor");
    RETURN_IF_ERROR(cuda::CudaStatus(
        cudaMemcpyAsync(projection->weights()[part].data(), host.data(),
                        host.size_bytes(), cudaMemcpyHostToDevice,
                        executor.stream()),
        "upload saved quadratic coefficients"));
  }
  ComposedLayerBuilder builder;
  RETURN_IF_ERROR(builder.add(
      QuadraticFeaturesLayer::Create(executor, kGpt2ContextLength)));
  RETURN_IF_ERROR(builder.add(std::move(projection)));
  return builder.create("saved_quadratic_mlp");
}

TokenTraceSite MlpSite(int block, bool inner_projection) {
  TokenTraceSite site{
      .scope = {"gpt2", absl::StrCat("transformer_block_", block),
                "ResidualLayer"},
      .layer_name = "mlp",
      .occurrence = 0};
  if (inner_projection) {
    site.scope.push_back("mlp");
    site.layer_name = "FullyConnectedLayer";
    site.occurrence = 1;
  }
  return site;
}
TokenTraceSite BlockSite(int block) {
  return {.scope = {"gpt2"},
          .layer_name = absl::StrCat("transformer_block_", block),
          .occurrence = 0};
}
absl::StatusOr<const TokenTraceActivation*> Find(const TokenTraceResult& trace,
                                                 const TokenTraceSite& site) {
  const TokenTraceActivation* found = nullptr;
  for (const auto& activation : trace.activations)
    if (activation.site.scope == site.scope &&
        activation.site.layer_name == site.layer_name &&
        activation.site.occurrence == site.occurrence &&
        activation.site.output_index == site.output_index) {
      if (found != nullptr)
        return absl::DataLossError("duplicate quadratic trace site");
      found = &activation;
    }
  if (found == nullptr || found->data_type != DataType::BF16 ||
      found->channels != 16 || found->first_row != 0 ||
      found->row_count != trace.query_row + 1 ||
      found->values.size() != found->row_count * 16)
    return absl::DataLossError("missing/malformed quadratic trace site");
  return found;
}

bool EqualLogits(const TokenTraceResult& a, const TokenTraceResult& b) {
  return a.predicted_token == b.predicted_token &&
         a.logits.size() == b.logits.size() &&
         std::memcmp(a.logits.data(), b.logits.data(),
                     a.logits.size() * sizeof(float)) == 0;
}

// This is an exact arithmetic decomposition of three OBSERVED branch vectors,
// not an additive attribution of the final prediction. The inherited term
// passes the altered input through the original nonlinear MLP; the local term
// compares the two MLP functions on that SAME altered input.
absl::Status WriteDecomposition(size_t line, const TokenTraceResult& original,
                                const TokenTraceResult& joint,
                                std::ostream& rows, std::ostream& coordinates) {
  for (int block = 0; block < kBlocks; ++block) {
    ASSIGN_OR_RETURN(auto original_update,
                     Find(original, MlpSite(block, false)));
    ASSIGN_OR_RETURN(auto learned_on_joint, Find(joint, MlpSite(block, true)));
    ASSIGN_OR_RETURN(auto quadratic_on_joint,
                     Find(joint, MlpSite(block, false)));
    for (size_t row = 0; row <= original.query_row; ++row) {
      double local_squared = 0, inherited_squared = 0, total_squared = 0;
      double inner_product = 0, reference_squared = 0;
      for (int channel = 0; channel < kWidth; ++channel) {
        const size_t index = row * kWidth + channel;
        const double a = original_update->values[index];
        const double b = learned_on_joint->values[index];
        const double c = quadratic_on_joint->values[index];
        if (!std::isfinite(a) || !std::isfinite(b) || !std::isfinite(c))
          return absl::DataLossError("nonfinite quadratic branch update");
        const double local = c - b, inherited = b - a, total = c - a;
        if (local + inherited != total)
          return absl::InternalError("BF16 difference decomposition failed");
        local_squared += local * local;
        inherited_squared += inherited * inherited;
        total_squared += total * total;
        inner_product += local * inherited;
        reference_squared += a * a;
        coordinates << line << '\t' << block << '\t' << row << '\t' << channel
                    << '\t' << a << '\t' << b << '\t' << c << '\t' << local
                    << '\t' << inherited << '\t' << total << '\n';
      }
      rows << line << '\t' << block << '\t' << row << '\t'
           << (row == original.query_row) << '\t' << std::sqrt(local_squared)
           << '\t' << std::sqrt(inherited_squared) << '\t'
           << std::sqrt(total_squared) << '\t' << inner_product << '\t'
           << std::sqrt(reference_squared) << '\n';
    }
  }
  return absl::OkStatus();
}

absl::Status Run() {
  const fs::path checkpoint = absl::GetFlag(FLAGS_checkpoint);
  const fs::path construction = absl::GetFlag(FLAGS_construction_dir);
  const fs::path output = absl::GetFlag(FLAGS_output_dir);
  if (checkpoint.empty() || construction.empty() || output.empty() ||
      absl::GetFlag(FLAGS_tokenizer).empty())
    return absl::InvalidArgumentError("all input/output paths are required");
  std::ifstream coefficient_input(construction / "coefficients.tsv");
  std::ifstream failure_input(construction / "first_failures.tsv");
  if (!coefficient_input || !failure_input)
    return absl::NotFoundError("missing construction coefficients/failures");
  ASSIGN_OR_RETURN(auto coefficients,
                   ReadQuadraticCoefficients(coefficient_input));
  ASSIGN_OR_RETURN(auto failures, ReadQuadraticFailures(failure_input));
  ASSIGN_OR_RETURN(
      auto base, tokenizer::Gpt2Tokenizer::Load(absl::GetFlag(FLAGS_tokenizer)));
  ASSIGN_OR_RETURN(auto compact,
                   tokenizer::CompactVocabularyTokenizer::LoadFromFile(
                       *base, checkpoint / "compact_vocabulary.tsv"));
  if (compact->vocab_size() != kVocabulary)
    return absl::InvalidArgumentError("expected 4475 compact tokens");
  ASSIGN_OR_RETURN(auto decoder, tokenizer::Gpt2Detokenizer::Load(
                                     absl::GetFlag(FLAGS_tokenizer)));
  ASSIGN_OR_RETURN(auto corpus, LoadTextCorpus(absl::GetFlag(FLAGS_corpus)));
  ASSIGN_OR_RETURN(auto executor, cuda::Executor::Create());
  ASSIGN_OR_RETURN(auto dataset, PaddedLineDataSetIterator::Create(
                                     *executor, corpus.text(), *compact,
                                     {.batch_size = 1,
                                      .context_length = kGpt2ContextLength,
                                      .prompt_tokens = 5,
                                      .eos_token = compact->eos_token_id()}));
  if (dataset->sample_count() != 1024 ||
      dataset->supervised_row_count() != 10002)
    return absl::InvalidArgumentError("unexpected corpus inventory");
  ASSIGN_OR_RETURN(auto model, CreateGpt2(*executor, DataType::BF16, 0,
                                          {.transformer_block_count = 8,
                                           .model_width = 16,
                                           .attention_heads = 1,
                                           .feed_forward_width = 64,
                                           .vocabulary_size = kVocabulary,
                                           .pad_vocabulary = false}));
  RETURN_IF_ERROR(
      ReadFromDirectory(*executor, *model, checkpoint.string(), false));
  ASSIGN_OR_RETURN(auto before, Snapshot(*executor, *model));
  std::vector<std::unique_ptr<ComposedLayer>> owners;
  std::vector<MlpReplacement> replacements;
  std::vector<std::vector<std::vector<uint8_t>>> replacement_before;
  for (int block = 0; block < kBlocks; ++block) {
    ASSIGN_OR_RETURN(auto layer,
                     MakeReplacement(*executor, coefficients[block]));
    ASSIGN_OR_RETURN(auto snapshot, Snapshot(*executor, *layer));
    replacement_before.push_back(std::move(snapshot));
    replacements.push_back({block, MlpSource::kLayerNorm, layer.get()});
    owners.push_back(std::move(layer));
  }
  std::error_code error;
  if (fs::exists(output, error))
    return absl::AlreadyExistsError("output_dir must be fresh");
  if (error || !fs::create_directory(output, error) || error)
    return absl::UnknownError("cannot create output directory");
  std::ofstream manifest(output / "manifest.tsv"), cases(output / "cases.tsv");
  std::ofstream scores(output / "interventions.tsv"),
      rows(output / "decomposition.tsv");
  std::ofstream coordinates(output / "coordinates.tsv");
  std::ofstream logits(output / "logits.f32", std::ios::binary);
  for (auto* stream :
       {&manifest, &cases, &scores, &rows, &coordinates, &logits}) {
    if (!*stream)
      return absl::UnknownError("cannot open quadratic trace reports");
    *stream << std::setprecision(17);
  }
  manifest
      << "key\tvalue\ncheckpoint\t" << checkpoint.string()
      << "\nconstruction_dir\t" << construction.string()
      << "\nconditions\toriginal,joint,single,prefix,restore_function,reset_"
         "state,identity_reset"
         "\nreset_scope\tall causal prefix rows after the selected complete "
         "transformer"
         "\nlocal_delta\tquadratic(x_joint)-learned(x_joint)"
         "\ninherited_delta\tlearned(x_joint)-learned(x_original)"
         "\ntotal_delta\tquadratic(x_joint)-learned(x_original)"
         "\nlogits\tnative little-endian FP32, 4475 per interventions.tsv row"
         "\nlabels\tscoring only; never passed into forward"
         "\nfailure_selection\tfirst greedy mismatch in saved joint quadratic "
         "run\n";
  cases << "line\tgroup\ttarget_position\tprefix\texpected\tpredicted\n";
  scores << "line\tcondition\tblock\tpredicted_id\tpredicted_"
            "text\tcorrect\ttarget_probability\ttarget_margin\tlogits_row\n";
  rows << "line\tblock\trow\tquery\tlocal_l2\tinherited_l2\ttotal_l2\tlocal_"
          "dot_inherited\toriginal_update_l2\n";
  coordinates << "line\tblock\trow\tchannel\toriginal_update\tlearned_on_"
                 "joint\tquadratic_on_joint\tlocal\tinherited\ttotal\n";
  size_t logit_row = 0;
  const TokenTraceOptions plain{.vocabulary_size = kVocabulary,
                                .padding_token = compact->eos_token_id()};
  auto capture = plain;
  capture.capture_activations = true;
  for (const auto& failure : failures) {
    const auto tokens = dataset->sample_tokens(failure.line - 1);
    const size_t position = failure.target_position;
    if (position > tokens.size() ||
        !std::equal(failure.prefix.begin(), failure.prefix.end(),
                    tokens.begin()) ||
        failure.expected_token != (position == tokens.size()
                                       ? compact->eos_token_id()
                                       : tokens[position]))
      return absl::DataLossError(
          "saved failure prefix/target disagrees with corpus");
    auto trace = [&](absl::Span<const MlpReplacement> selected,
                     const TokenTraceOptions& options) {
      return TraceMlpReplacements(*executor, *model, failure.prefix, selected,
                                  options);
    };
    auto record = [&](absl::string_view condition, int block,
                      const TokenTraceResult& result) -> absl::Status {
      if (result.logits.size() != kVocabulary)
        return absl::DataLossError("wrong real-vocabulary logit count");
      double rival = -std::numeric_limits<double>::infinity();
      double maximum = rival, sum = 0;
      for (float value : result.logits) {
        if (!std::isfinite(value))
          return absl::DataLossError("nonfinite intervention logits");
        maximum = std::max(maximum, static_cast<double>(value));
      }
      for (int token = 0; token < kVocabulary; ++token) {
        sum += std::exp(result.logits[token] - maximum);
        if (token != failure.expected_token)
          rival = std::max(rival, static_cast<double>(result.logits[token]));
      }
      ASSIGN_OR_RETURN(auto text,
                       Decode({&result.predicted_token, 1}, *compact, *decoder));
      scores << failure.line << '\t' << condition << '\t' << block << '\t'
             << result.predicted_token << '\t' << text << '\t'
             << (result.predicted_token == failure.expected_token) << '\t'
             << std::exp(result.logits[failure.expected_token] - maximum) / sum
             << '\t' << result.logits[failure.expected_token] - rival << '\t'
             << logit_row++ << '\n';
      logits.write(reinterpret_cast<const char*>(result.logits.data()),
                   result.logits.size() * sizeof(float));
      return absl::OkStatus();
    };
    ASSIGN_OR_RETURN(auto original,
                     TraceNextToken(*executor, *model, failure.prefix, capture));
    ASSIGN_OR_RETURN(auto joint, trace(replacements, capture));
    ASSIGN_OR_RETURN(auto joint_plain, trace(replacements, plain));
    ASSIGN_OR_RETURN(auto empty, trace({}, plain));
    if (original.predicted_token != failure.expected_token ||
        joint.predicted_token != failure.predicted_token ||
        !EqualLogits(joint, joint_plain) || !EqualLogits(original, empty))
      return absl::FailedPreconditionError(
          "saved failure/capture/empty replacement control failed");
    RETURN_IF_ERROR(record("original", -1, original));
    RETURN_IF_ERROR(record("joint", -1, joint));
    ASSIGN_OR_RETURN(auto prefix_text,
                     Decode(failure.prefix, *compact, *decoder));
    ASSIGN_OR_RETURN(auto expected_text,
                     Decode({&failure.expected_token, 1}, *compact, *decoder));
    ASSIGN_OR_RETURN(auto predicted_text,
                     Decode({&failure.predicted_token, 1}, *compact, *decoder));
    cases << failure.line << '\t'
          << ((failure.line - 1) % 5 == 0 ? "held" : "fit") << '\t' << position
          << '\t' << prefix_text << '\t' << expected_text << '\t'
          << predicted_text << '\n';
    RETURN_IF_ERROR(
        WriteDecomposition(failure.line, original, joint, rows, coordinates));
    for (int block = 0; block < kBlocks; ++block) {
      ASSIGN_OR_RETURN(auto single, trace({&replacements[block], 1}, plain));
      RETURN_IF_ERROR(record("single", block, single));
      ASSIGN_OR_RETURN(
          auto prefix,
          trace({replacements.data(), static_cast<size_t>(block + 1)}, plain));
      RETURN_IF_ERROR(record("prefix", block, prefix));
      std::vector<MlpReplacement> except;
      for (const auto& replacement : replacements)
        if (replacement.block != block)
          except.push_back(replacement);
      ASSIGN_OR_RETURN(auto restored, trace(except, plain));
      RETURN_IF_ERROR(record("restore_function", block, restored));
      ASSIGN_OR_RETURN(auto donor, Find(original, BlockSite(block)));
      const TokenTracePatch patch{.site = BlockSite(block),
                                  .rows = TokenTraceRows::kAllPrefix,
                                  .replacement = TokenTraceReplacement::kDonor,
                                  .donor = donor};
      auto patched = plain;
      patched.patches = {&patch, 1};
      ASSIGN_OR_RETURN(auto reset, trace(replacements, patched));
      RETURN_IF_ERROR(record("reset_state", block, reset));
      ASSIGN_OR_RETURN(auto identity, trace({}, patched));
      RETURN_IF_ERROR(record("identity_reset", block, identity));
      if (!EqualLogits(identity, original) ||
          (block == 7 && !EqualLogits(reset, original)) ||
          (block == 7 && !EqualLogits(prefix, joint)))
        return absl::FailedPreconditionError(
            "identity/final reset/prefix control failed");
    }
    ASSIGN_OR_RETURN(auto post,
                     TraceNextToken(*executor, *model, failure.prefix, plain));
    if (!EqualLogits(original, post))
      return absl::FailedPreconditionError("ordinary post-pass logits changed");
    std::cout << "Traced line " << failure.line << ": " << expected_text
              << " versus " << predicted_text << std::endl;
    scores.flush();
    cases.flush();
  }
  ASSIGN_OR_RETURN(auto after, Snapshot(*executor, *model));
  if (before != after)
    return absl::FailedPreconditionError(
        "original model master weights changed");
  for (int block = 0; block < kBlocks; ++block) {
    ASSIGN_OR_RETURN(auto after_replacement,
                     Snapshot(*executor, *owners[block]));
    if (after_replacement != replacement_before[block])
      return absl::FailedPreconditionError("quadratic coefficients changed");
  }
  manifest << "cases\t" << failures.size() << "\nlogit_rows\t" << logit_row
           << "\ncontrols\tcaptured/plain,empty replacement,original "
              "post,identity reset,final reset,full prefix,all weights "
              "unchanged: PASS\ncompleted\ttrue\n";
  for (auto* stream :
       {&manifest, &cases, &scores, &rows, &coordinates, &logits}) {
    stream->flush();
    if (!*stream)
      return absl::UnknownError("cannot finish quadratic trace reports");
  }
  return absl::OkStatus();
}
}  // namespace
}  // namespace pluto::llm::one_shot_memorizer

int main(int argc, char** argv) {
  static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559);
  static_assert(std::endian::native == std::endian::little);
  if (absl::ParseCommandLine(argc, argv).size() != 1) {
    std::cerr << "Unexpected positional arguments\n";
    return 1;
  }
  const auto status = pluto::llm::one_shot_memorizer::Run();
  if (!status.ok()) {
    std::cerr << status << '\n';
    return 1;
  }
}
