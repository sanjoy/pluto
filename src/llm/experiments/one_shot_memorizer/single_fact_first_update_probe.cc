// Explain an observed first Adam update with scalar CPU reference layers.
// Default mode diagnoses a supplied known fact. --permute_prompt instead
// searches all orders of an explicitly supplied five-token candidate set,
// matching predicted updates to observed weights. Neither mode trains a
// replacement model or uses CUDA; search computes gradients at fixed weights.
#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
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
#include "absl/types/span.h"
#include "src/host/buffer.h"
#include "src/llm/experiments/one_shot_memorizer/first_update_match.h"
#include "src/llm/layer.h"
#include "src/llm/layers/attention.h"
#include "src/llm/layers/combinators.h"
#include "src/llm/layers/cross_entropy_loss.h"
#include "src/llm/layers/embedding.h"
#include "src/llm/layers/fully_connected.h"
#include "src/llm/layers/gelu.h"
#include "src/llm/layers/norm.h"
#include "src/llm/layers/reference_internal.h"
#include "src/util/status_macros.h"

ABSL_FLAG(std::string, initial_checkpoint, "",
          "Checkpoint before the first update");
ABSL_FLAG(std::string, first_update_checkpoint, "",
          "Checkpoint after exactly one single-fact Adam update");
ABSL_FLAG(std::string, output_dir, "",
          "Fresh directory for summary and coordinate TSVs");
ABSL_FLAG(std::vector<std::string>, token_ids,
          (std::vector<std::string>{"4059", "329", "2114", "113", "789", "124",
                                    "328", "792", "2147", "86", "3735", "4018",
                                    "2483", "2"}),
          "Complete known fact as compact IDs, without appended EOS; default "
          "is corpus line 631 (Durian)");
ABSL_FLAG(int, prompt_token_count, 5,
          "First tokens supplied without suffix-prediction loss");
ABSL_FLAG(int, eos_token, 4474, "Compact end-of-sequence token");
ABSL_FLAG(
    int, cpu_context_length, 16,
    "CPU storage rows; multiple of 16, up to 1024, at least the fact length");
ABSL_FLAG(double, learning_rate, 0.000006,
          "Actual first-step rate (default peak 0.0006 / warmup 100)");
ABSL_FLAG(double, epsilon, 1e-8, "Adam denominator epsilon");
ABSL_FLAG(
    bool, permute_prompt, false,
    "Search all 120 orders of the first five distinct token_ids, treating them "
    "as an unordered set; remaining IDs are a fixed recovered suffix. Requires "
    "explicit token_ids and an observed first update with fresh Adam moments "
    "and zero weight decay. No optimizer update is applied during search");
ABSL_FLAG(int, max_search_seconds, 120,
          "CPU time cap for prompt search; report partial results on timeout");

namespace pluto::llm::one_shot_memorizer {
namespace {
namespace fs = std::filesystem;
namespace ri = reference_internal;

// The experiment intentionally fixes the studied checkpoint architecture.
// Every initial tensor and the observed updated embedding are size-checked;
// other updated tensors are not needed for this comparison. BF16 is the
// compute policy; all trainable checkpoint tensors are FP32 master weights.
constexpr int kWidth = 16;
constexpr int kVocabulary = 4475;
constexpr int kLayers = 8;
constexpr int kHeads = 1;
constexpr int kExpansion = 64;
constexpr int kCheckpointContext = 1024;
constexpr DataType kCompute = DataType::BF16;
constexpr size_t kEmbeddingElements = kVocabulary * kWidth;

struct Options {
  fs::path initial;
  fs::path updated;
  fs::path output;
  // Known fact in diagnostic mode; unordered first five IDs plus a fixed
  // recovered suffix in search mode. The CLI does not itself recover this set.
  std::vector<int> tokens;
  int prompt_count;
  int eos;
  int context;
  float learning_rate;
  float epsilon;
  bool permute_prompt;
  int max_search_seconds;
};

absl::StatusOr<Options> ReadOptions() {
  Options out{
      .initial = absl::GetFlag(FLAGS_initial_checkpoint),
      .updated = absl::GetFlag(FLAGS_first_update_checkpoint),
      .output = absl::GetFlag(FLAGS_output_dir),
      .prompt_count = absl::GetFlag(FLAGS_prompt_token_count),
      .eos = absl::GetFlag(FLAGS_eos_token),
      .context = absl::GetFlag(FLAGS_cpu_context_length),
      .learning_rate = static_cast<float>(absl::GetFlag(FLAGS_learning_rate)),
      .epsilon = static_cast<float>(absl::GetFlag(FLAGS_epsilon)),
      .permute_prompt = absl::GetFlag(FLAGS_permute_prompt),
      .max_search_seconds = absl::GetFlag(FLAGS_max_search_seconds)};
  // Never silently reuse the diagnostic's labeled example in decoding mode.
  if (out.permute_prompt && !FLAGS_token_ids.IsSpecifiedOnCommandLine())
    return absl::InvalidArgumentError(
        "permute_prompt requires explicitly supplied token_ids: unordered "
        "five-token candidate set followed by its recovered suffix");
  if (out.initial.empty() || out.updated.empty() || out.output.empty())
    return absl::InvalidArgumentError(
        "initial_checkpoint, first_update_checkpoint, and output_dir are "
        "required");
  if (out.context <= 0 || out.context > kCheckpointContext ||
      out.context % 16 || out.prompt_count <= 0 || out.eos < 0 ||
      out.eos >= kVocabulary || !std::isfinite(out.learning_rate) ||
      out.learning_rate <= 0 || !std::isfinite(out.epsilon) || out.epsilon <= 0)
    return absl::InvalidArgumentError(
        "invalid context, prompt count, EOS, learning rate, or epsilon");
  for (const auto& text : absl::GetFlag(FLAGS_token_ids)) {
    int token;
    if (!absl::SimpleAtoi(text, &token) || token < 0 || token >= kVocabulary ||
        token == out.eos)
      return absl::InvalidArgumentError(
          absl::StrCat("invalid fact token ID: ", text));
    out.tokens.push_back(token);
  }
  if (out.tokens.size() < static_cast<size_t>(out.prompt_count) ||
      out.tokens.size() > static_cast<size_t>(out.context))
    return absl::InvalidArgumentError(
        "fact must contain at least prompt_token_count tokens and fit CPU "
        "context");
  if (out.permute_prompt) {
    if (out.prompt_count != 5 || out.max_search_seconds <= 0)
      return absl::InvalidArgumentError(
          "prompt search requires prompt_token_count=5 and positive time cap");
    std::sort(out.tokens.begin(), out.tokens.begin() + out.prompt_count);
    if (std::adjacent_find(out.tokens.begin(),
                           out.tokens.begin() + out.prompt_count) !=
        out.tokens.begin() + out.prompt_count)
      return absl::InvalidArgumentError(
          "prompt search requires five distinct candidate IDs");
  }
  return out;
}

// Assemble the exact pre-LayerNorm architecture from existing reference
// classes. Attention and MLP each retain their residual connection. The final
// head borrows the very same embedding object, preserving tied gradients.
absl::StatusOr<std::vector<std::unique_ptr<LayerReference>>> CreateModel(
    int context) {
  std::vector<std::unique_ptr<LayerReference>> model;
  ASSIGN_OR_RETURN(auto embedding,
                   EmbeddingLookupLayerReference::Create(
                       kVocabulary, kWidth, kCompute, context, false));
  auto* tied = embedding.get();
  model.push_back(std::move(embedding));
  ASSIGN_OR_RETURN(auto position, PositionEmbeddingLayerReference::Create(
                                      context, kWidth, kCompute));
  model.push_back(std::move(position));
  for (int block = 0; block < kLayers; ++block) {
    ComposedLayerReferenceBuilder attention;
    RETURN_IF_ERROR(attention.add(
        LayerNormLayerReference::Create(kWidth, 1e-5f, kCompute, context)));
    RETURN_IF_ERROR(attention.add(FullyConnectedLayerReference::Create(
        kWidth, 3 * kWidth, kCompute, context)));
    RETURN_IF_ERROR(attention.add(
        AttentionLayerReference::Create(context, kHeads, kWidth, kCompute)));
    RETURN_IF_ERROR(attention.add(FullyConnectedLayerReference::Create(
        kWidth, kWidth, kCompute, context)));
    ASSIGN_OR_RETURN(auto branch, attention.create("attention"));
    ASSIGN_OR_RETURN(auto residual,
                     ResidualLayerReference::Create(std::move(branch)));
    model.push_back(std::move(residual));
    ComposedLayerReferenceBuilder mlp;
    RETURN_IF_ERROR(mlp.add(
        LayerNormLayerReference::Create(kWidth, 1e-5f, kCompute, context)));
    RETURN_IF_ERROR(mlp.add(FullyConnectedLayerReference::Create(
        kWidth, kExpansion, kCompute, context)));
    RETURN_IF_ERROR(
        mlp.add(GeluLayerReference::Create(kExpansion, kCompute, context)));
    RETURN_IF_ERROR(mlp.add(FullyConnectedLayerReference::Create(
        kExpansion, kWidth, kCompute, context)));
    ASSIGN_OR_RETURN(auto mlp_branch, mlp.create("mlp"));
    ASSIGN_OR_RETURN(auto mlp_residual,
                     ResidualLayerReference::Create(std::move(mlp_branch)));
    model.push_back(std::move(mlp_residual));
  }
  ASSIGN_OR_RETURN(auto final_norm, LayerNormLayerReference::Create(
                                        kWidth, 1e-5f, kCompute, context));
  model.push_back(std::move(final_norm));
  ASSIGN_OR_RETURN(auto head, LanguageModelingHeadLayerReference::Create(tied));
  model.push_back(std::move(head));
  return model;
}

absl::Status ReadFloats(const fs::path& path, size_t expected_file_elements,
                        absl::Span<float> destination) {
  if (destination.size() > expected_file_elements)
    return absl::InvalidArgumentError(
        "checkpoint destination exceeds file extent");
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file)
    return absl::NotFoundError(path.string());
  if (file.tellg() !=
      static_cast<std::streamoff>(expected_file_elements * sizeof(float)))
    return absl::DataLossError(
        absl::StrCat("wrong tensor file size: ", path.string()));
  file.seekg(0);
  file.read(reinterpret_cast<char*>(destination.data()),
            destination.size() * sizeof(float));
  if (!file)
    return absl::DataLossError(
        absl::StrCat("short tensor read: ", path.string()));
  for (float value : destination)
    if (!std::isfinite(value))
      return absl::InvalidArgumentError(
          absl::StrCat("nonfinite tensor: ", path.string()));
  return absl::OkStatus();
}

absl::Status LoadInitialWeights(
    const fs::path& initial,
    std::vector<std::unique_ptr<LayerReference>>& model) {
  absl::flat_hash_set<const void*> seen;
  size_t index = 0;
  for (auto& layer : model)
    for (auto& weight : layer->weights()) {
      if (!seen.insert(weight.data()).second)
        continue;
      const size_t elements = weight.size_bytes() / sizeof(float);
      RETURN_IF_ERROR(
          ReadFloats(initial / absl::StrCat("weight_", index, ".bin"),
                     index == 1 ? kCheckpointContext * kWidth : elements,
                     {static_cast<float*>(weight.data()), elements}));
      ++index;
    }
  if (index != 4 + 12 * kLayers)
    return absl::InternalError("unexpected reference parameter inventory");
  return absl::OkStatus();
}

int Sign(double value) { return (value > 0) - (value < 0); }

// Saved quantities needed only by the explanatory diagnostic. The model's
// gradient buffers contain the total tied-embedding gradient after this call.
struct FactBackward {
  HostBuffer normalized;
  std::vector<float> head_gradient;
};

// Evaluate the same masked mean-loss objective at fixed weights. Future
// padding cannot affect causal supervised rows; scalar reduction order can
// still differ from CUDA. Every call resets all gradient accumulators.
absl::StatusOr<FactBackward> ComputeFactGradient(
    const Options& options,
    std::vector<std::unique_ptr<LayerReference>>& model) {
  ASSIGN_OR_RETURN(auto input,
                   HostBuffer::Allocate(options.context * sizeof(int32_t)));
  ASSIGN_OR_RETURN(auto target_buffer,
                   HostBuffer::Allocate(options.context * sizeof(int32_t)));
  static_assert(sizeof(int) == sizeof(int32_t),
                "reference token schema uses int32 storage");
  auto* ids = static_cast<int*>(input.data());
  auto* labels = static_cast<int*>(target_buffer.data());
  std::fill_n(ids, options.context, options.eos);
  std::fill_n(labels, options.context, CrossEntropyLossLayer::kIgnoredTarget);
  std::copy(options.tokens.begin(), options.tokens.end(), ids);
  const int first_supervised_row = options.prompt_count - 1;
  const int fact_length = static_cast<int>(options.tokens.size());
  for (int row = first_supervised_row; row < fact_length; ++row)
    labels[row] =
        row + 1 == fact_length ? options.eos : options.tokens[row + 1];

  // Future rows have ignored targets and cannot affect earlier causal rows.
  // CE divides by supervised_rows, not context, so shortening padding from
  // 1024 to16 preserves the objective. Scalar reduction order can still differ
  // from GPU arithmetic; this diagnostic never promises bitwise equivalence.
  HostBufferVec values{input};
  std::vector<ReferenceBackwardState> states;
  HostBuffer normalized;
  for (size_t index = 0; index < model.size(); ++index) {
    ASSIGN_OR_RETURN(auto result, model[index]->fwd(values));
    values = std::move(result.outputs);
    states.push_back(std::move(result.state));
    if (index + 2 == model.size())
      normalized = values[0];
  }
  ASSIGN_OR_RETURN(auto loss, CrossEntropyLossLayerReference::Create(
                                  kVocabulary, kCompute, options.context));
  ASSIGN_OR_RETURN(auto loss_fwd, loss->fwd({values[0], target_buffer}));
  ASSIGN_OR_RETURN(auto gradients, loss->bwd({}, std::move(loss_fwd.state)));
  for (auto& layer : model)
    for (auto& gradient : layer->gradients())
      std::memset(gradient.data(), 0, gradient.size_bytes());
  std::vector<float> head_gradient;
  for (size_t index = model.size(); index-- > 0;) {
    ASSIGN_OR_RETURN(gradients,
                     model[index]->bwd(gradients, std::move(states[index])));
    if (index + 1 == model.size()) {
      const auto* head =
          static_cast<const float*>(model.front()->gradients()[0].data());
      head_gradient.assign(head, head + kEmbeddingElements);
    }
  }
  return FactBackward{std::move(normalized), std::move(head_gradient)};
}

std::string TokenIds(absl::Span<const int> tokens) {
  std::string result;
  for (int token : tokens) {
    if (!result.empty())
      result += ',';
    result += std::to_string(token);
  }
  return result;
}

// The prompt-only score is secondary: the primary all-embedding score alone
// ranks candidates. This avoids choosing a scoring rule after seeing labels.
struct PromptCandidate {
  std::vector<int> prompt;
  FirstAdamUpdateScore all_embeddings;
  FirstAdamUpdateScore prompt_rows;
  double seconds = 0;
};

absl::Status RunPermutationSearch(const Options& options) {
  ASSIGN_OR_RETURN(auto model, CreateModel(options.context));
  RETURN_IF_ERROR(LoadInitialWeights(options.initial, model));
  std::vector<float> observed(kEmbeddingElements);
  RETURN_IF_ERROR(ReadFloats(options.updated / "weight_0.bin",
                             kEmbeddingElements, absl::MakeSpan(observed)));
  const auto* initial_data =
      static_cast<const float*>(model.front()->weights()[0].data());
  const std::vector<float> initial(initial_data,
                                   initial_data + kEmbeddingElements);
  std::vector<int> permutation(options.tokens.begin(),
                               options.tokens.begin() + options.prompt_count);
  std::sort(permutation.begin(), permutation.end());
  const std::vector<int> prompt_set = permutation;
  const absl::Span<const int> suffix(
      options.tokens.data() + options.prompt_count,
      options.tokens.size() - options.prompt_count);

  // Forward/backward may change only activation/gradient buffers. Verify every
  // candidate starts and ends with the same complete set of master weights,
  // including tied weights (which are snapshotted only once).
  std::vector<HostBuffer> weights;
  std::vector<std::vector<uint8_t>> weight_bytes;
  absl::flat_hash_set<const void*> seen;
  for (const auto& layer : model)
    for (const auto& weight : layer->weights())
      if (seen.insert(weight.data()).second) {
        weights.push_back(weight);
        const auto* bytes = static_cast<const uint8_t*>(weight.data());
        weight_bytes.emplace_back(bytes, bytes + weight.size_bytes());
      }
  std::error_code error;
  if (!fs::create_directory(options.output, error))
    return absl::AlreadyExistsError("output_dir must be a fresh directory");
  std::ofstream candidates(options.output / "candidates.tsv");
  std::ofstream ranked(options.output / "ranked.tsv");
  std::ofstream manifest(options.output / "manifest.tsv");
  for (auto* file : {&candidates, &ranked, &manifest}) {
    if (!*file)
      return absl::UnknownError("cannot create prompt search reports");
    *file << std::setprecision(std::numeric_limits<double>::max_digits10);
  }
  manifest
      << "initial_checkpoint\t" << options.initial.string()
      << "\nfirst_update_checkpoint\t" << options.updated.string()
      << "\nmode\tpermute_prompt\nbackend\tCPU_reference_no_Executor"
      << "\ncandidate_source\texplicit_unordered_prompt_set_and_ordered_suffix"
      << "\ncorpus_read\tfalse\ntrue_prompt_order_used_for_search\tfalse"
      << "\nprimary_score\tall_E_FP32_predicted_weight_minus_observed_weight_L2"
      << "\nsecondary_scores\tprompt_rows_L2_and_update_sign_mismatches_not_"
         "used_for_ranking"
      << "\noptimizer_assumption\tfresh_Adam_moments_zero_weight_decay_first_"
         "step"
      << "\nlearning_rate_fp32\t" << options.learning_rate << "\nepsilon_fp32\t"
      << options.epsilon
      << "\nnumerical_caveat\tCPU_scalar_BF16_and_simplified_Adam_not_bitexact_"
         "GPU"
      << "\nknown_prompt_count\t" << options.prompt_count
      << "\nprompt_candidate_set_sorted\t" << TokenIds(prompt_set)
      << "\nsupplied_suffix_no_EOS\t" << TokenIds(suffix) << "\neos_token\t"
      << options.eos << "\ncpu_context\t" << options.context
      << "\ncheckpoint_context\t" << kCheckpointContext << "\nvocabulary\t"
      << kVocabulary << "\nmodel_width\t" << kWidth << "\nlayers\t" << kLayers
      << "\nheads\t" << kHeads << "\nfeed_forward_width\t" << kExpansion
      << "\nphase_side_information\tfirst_update_fresh_moments_masked_suffix_"
         "loss"
      << "\nlimitation\tsearch_cannot_correct_wrong_candidate_set_or_suffix"
      << "\nmax_search_seconds\t" << options.max_search_seconds << '\n';
  const char* columns =
      "ordinal\tprompt_ids\tl2\tsquared_l2\tprompt_rows_l2\t"
      "sign_mismatches\tprompt_sign_mismatches\tbit_equal_"
      "coordinates\tseconds\n";
  candidates << columns;
  ranked << "rank\t" << columns;
  manifest.flush();
  const auto write = [](std::ostream& output, size_t ordinal,
                        const PromptCandidate& candidate) {
    output << ordinal << '\t' << TokenIds(candidate.prompt) << '\t'
           << std::sqrt(candidate.all_embeddings.squared_error) << '\t'
           << candidate.all_embeddings.squared_error << '\t'
           << std::sqrt(candidate.prompt_rows.squared_error) << '\t'
           << candidate.all_embeddings.sign_mismatches << '\t'
           << candidate.prompt_rows.sign_mismatches << '\t'
           << candidate.all_embeddings.bit_equal_coordinates << '\t'
           << candidate.seconds << '\n';
  };

  std::vector<PromptCandidate> results;
  const auto started = std::chrono::steady_clock::now();
  do {
    const auto candidate_started = std::chrono::steady_clock::now();
    Options candidate = options;
    std::copy(permutation.begin(), permutation.end(), candidate.tokens.begin());
    ASSIGN_OR_RETURN(auto backward, ComputeFactGradient(candidate, model));
    const auto* gradient_data =
        static_cast<const float*>(model.front()->gradients()[0].data());
    const absl::Span<const float> gradient(gradient_data, kEmbeddingElements);
    ASSIGN_OR_RETURN(auto score, ScoreFirstAdamUpdate(
                                     initial, observed, gradient,
                                     options.learning_rate, options.epsilon));
    std::vector<float> prompt_initial, prompt_observed, prompt_gradient;
    for (int token : prompt_set)
      for (int column = 0; column < kWidth; ++column) {
        const size_t index = static_cast<size_t>(token) * kWidth + column;
        prompt_initial.push_back(initial[index]);
        prompt_observed.push_back(observed[index]);
        prompt_gradient.push_back(gradient[index]);
      }
    ASSIGN_OR_RETURN(
        auto prompt_score,
        ScoreFirstAdamUpdate(prompt_initial, prompt_observed, prompt_gradient,
                             options.learning_rate, options.epsilon));
    for (size_t i = 0; i < weights.size(); ++i)
      if (std::memcmp(weights[i].data(), weight_bytes[i].data(),
                      weight_bytes[i].size()) != 0)
        return absl::InternalError("candidate modified initial master weights");
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      candidate_started)
            .count();
    results.push_back({permutation, score, prompt_score, seconds});
    write(candidates, results.size(), results.back());
    candidates.flush();
    if (!candidates)
      return absl::DataLossError("failed writing candidate score");
    if (results.size() % 10 == 0)
      std::cout << "evaluated=" << results.size() << " elapsed_seconds="
                << std::chrono::duration<double>(
                       std::chrono::steady_clock::now() - started)
                       .count()
                << std::endl;
    if (std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      started)
            .count() > options.max_search_seconds)
      break;
  } while (std::next_permutation(permutation.begin(), permutation.end()));

  std::vector<size_t> order(results.size());
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
    const double left = results[a].all_embeddings.squared_error;
    const double right = results[b].all_embeddings.squared_error;
    return left != right ? left < right : results[a].prompt < results[b].prompt;
  });
  for (size_t rank = 0; rank < order.size(); ++rank) {
    ranked << rank + 1 << '\t';
    write(ranked, order[rank] + 1, results[order[rank]]);
  }
  const auto& best = results[order[0]];
  const size_t ties =
      std::count_if(results.begin(), results.end(), [&](const auto& row) {
        return row.all_embeddings.squared_error ==
               best.all_embeddings.squared_error;
      });
  const bool complete = results.size() == 120;
  manifest << "evaluated\t" << results.size()
           << "\nexpected_permutations\t120\ncomplete\t" << complete
           << "\nall_candidate_weights_unchanged\ttrue\nbest_prompt_ids\t"
           << TokenIds(best.prompt) << "\nbest_l2\t"
           << std::sqrt(best.all_embeddings.squared_error)
           << "\nexact_best_score_ties\t" << ties;
  if (order.size() >= 2)
    manifest << "\nrunner_up_l2\t"
             << std::sqrt(results[order[1]].all_embeddings.squared_error)
             << "\nrunner_up_squared_error_gap\t"
             << results[order[1]].all_embeddings.squared_error -
                    best.all_embeddings.squared_error;
  manifest << "\nelapsed_seconds\t"
           << std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                            started)
                  .count()
           << '\n';
  std::cout << "complete=" << complete
            << " best_prompt=" << TokenIds(best.prompt)
            << " best_l2=" << std::sqrt(best.all_embeddings.squared_error)
            << " ties=" << ties << std::endl;
  for (auto* file : {&candidates, &ranked, &manifest}) {
    file->close();
    if (!*file)
      return absl::DataLossError("failed writing prompt search report");
  }
  if (!complete)
    return absl::DeadlineExceededError(
        "time cap reached; saved partial rankings");
  return absl::OkStatus();
}

absl::Status Run(const Options& options) {
  ASSIGN_OR_RETURN(auto model, CreateModel(options.context));
  RETURN_IF_ERROR(LoadInitialWeights(options.initial, model));
  std::vector<float> observed(kEmbeddingElements);
  RETURN_IF_ERROR(ReadFloats(options.updated / "weight_0.bin",
                             kEmbeddingElements, absl::MakeSpan(observed)));
  const absl::flat_hash_set<int> inputs(options.tokens.begin(),
                                        options.tokens.end());
  absl::flat_hash_set<int> targets(
      options.tokens.begin() + options.prompt_count, options.tokens.end());
  targets.insert(options.eos);
  ASSIGN_OR_RETURN(auto backward, ComputeFactGradient(options, model));
  const auto& normalized = backward.normalized;
  const auto& head_gradient = backward.head_gradient;
  const int first_supervised_row = options.prompt_count - 1;
  const int fact_length = static_cast<int>(options.tokens.size());
  const int supervised_rows = fact_length - first_supervised_row;
  const auto* gradient =
      static_cast<const float*>(model.front()->gradients()[0].data());
  const auto* initial =
      static_cast<const float*>(model.front()->weights()[0].data());
  // Uniform softmax predicts a common absent-token head gradient. This ideal
  // formula uses real-valued probabilities; actual reference gradients retain
  // nonuniform softmax and the head's BF16 operand conversions.
  std::vector<double> uniform(kWidth);
  for (int column = 0; column < kWidth; ++column)
    for (int row = first_supervised_row; row < fact_length; ++row)
      uniform[column] +=
          ri::LoadActivation(normalized, row * kWidth + column, kCompute) /
          (static_cast<double>(supervised_rows) * kVocabulary);
  std::vector<double> actual(kEmbeddingElements), predicted(kEmbeddingElements),
      mean(kWidth);
  double error2 = 0, actual2 = 0, absent_error2 = 0, absent_actual2 = 0;
  double absent_gradient2 = 0, uniform_error2 = 0, input_gradient2 = 0;
  size_t absent_rows = 0, sign_equal = 0, bits_equal = 0,
         uniform_sign_equal = 0;
  for (int token = 0; token < kVocabulary; ++token) {
    const bool absent = !inputs.contains(token) && !targets.contains(token);
    absent_rows += absent;
    for (int column = 0; column < kWidth; ++column) {
      const size_t i = static_cast<size_t>(token) * kWidth + column;
      if (!std::isfinite(gradient[i]) || !std::isfinite(head_gradient[i]))
        return absl::InvalidArgumentError("nonfinite CPU gradient");
      // Fresh moments, bias correction, and zero weight decay reduce the first
      // Adam step to this expression in real arithmetic, independent of betas.
      // The GPU actually materializes FP32 moments/corrections, so differences
      // also include optimizer rounding, not only CPU-vs-GPU gradient error.
      const float update =
          gradient[i] / (std::abs(gradient[i]) + options.epsilon);
      const float weight = initial[i] - options.learning_rate * update;
      actual[i] = static_cast<double>(observed[i]) - initial[i];
      predicted[i] = static_cast<double>(weight) - initial[i];
      const double error = predicted[i] - actual[i];
      error2 += error * error;
      actual2 += actual[i] * actual[i];
      sign_equal += Sign(predicted[i]) == Sign(actual[i]);
      bits_equal += std::bit_cast<uint32_t>(weight) ==
                    std::bit_cast<uint32_t>(observed[i]);
      const double lookup = static_cast<double>(gradient[i]) - head_gradient[i];
      input_gradient2 += lookup * lookup;
      if (absent) {
        if (gradient[i] != head_gradient[i])
          return absl::InternalError(
              "absent token has an input-lookup gradient");
        absent_actual2 += actual[i] * actual[i];
        absent_error2 += error * error;
        absent_gradient2 += static_cast<double>(gradient[i]) * gradient[i];
        const double uniform_error = gradient[i] - uniform[column];
        uniform_error2 += uniform_error * uniform_error;
        uniform_sign_equal += Sign(gradient[i]) == Sign(uniform[column]);
        mean[column] += actual[i];
      }
    }
  }
  if (absent_rows == 0 || actual2 == 0 || absent_actual2 == 0 ||
      absent_gradient2 == 0)
    return absl::FailedPreconditionError(
        "comparison requires nonzero observed updates and absent-token "
        "gradients");
  double common_energy = 0;
  for (double& value : mean) {
    value /= absent_rows;
    common_energy += absent_rows * value * value;
  }
  std::error_code error;
  if (!fs::create_directory(options.output, error))
    return absl::AlreadyExistsError(absl::StrCat(
        "output_dir must be a new directory: ", options.output.string(), "; ",
        error.message()));
  std::ofstream coordinates(options.output / "coordinates.tsv");
  std::ofstream summary(options.output / "summary.tsv");
  if (!coordinates || !summary)
    return absl::UnknownError("cannot create probe reports");
  coordinates
      << std::setprecision(std::numeric_limits<double>::max_digits10)
      << "token\tdimension\tinput\ttarget\tgradient\thead_gradient\tuniform_"
         "gradient\tactual_update\tpredicted_update\n";
  for (int token = 0; token < kVocabulary; ++token)
    for (int column = 0; column < kWidth; ++column) {
      const size_t i = static_cast<size_t>(token) * kWidth + column;
      coordinates << token << '\t' << column << '\t' << inputs.contains(token)
                  << '\t' << targets.contains(token) << '\t' << gradient[i]
                  << '\t' << head_gradient[i] << '\t' << uniform[column] << '\t'
                  << actual[i] << '\t' << predicted[i] << '\n';
    }
  auto report = [&](std::ostream& out) {
    out << std::setprecision(std::numeric_limits<double>::max_digits10)
        << "initial_checkpoint\t" << options.initial.string()
        << "\nfirst_update_checkpoint\t" << options.updated.string()
        << "\nbackend\tCPU_reference_no_Executor\ninterpretation\tknown_fact_"
           "backward_diagnostic_not_weight_decoding"
        << "\noptimizer_assumption\tfresh_moments_zero_weight_decay_first_step"
        << "\nnumerics\tsimplified_Adam_formula_and_scalar_BF16_reference_not_"
           "bitexact_GPU"
        << "\nvocabulary\t" << kVocabulary << "\nmodel_width\t" << kWidth
        << "\nlayers\t" << kLayers << "\nheads\t" << kHeads
        << "\nfeed_forward_width\t" << kExpansion << "\ncheckpoint_context\t"
        << kCheckpointContext << "\ncpu_context\t" << options.context
        << "\nprompt_token_count\t" << options.prompt_count << "\neos_token\t"
        << options.eos << "\nsupervised_rows\t" << supervised_rows
        << "\nlearning_rate_fp32\t" << options.learning_rate
        << "\nepsilon_fp32\t" << options.epsilon << "\ntoken_ids\t";
    for (size_t i = 0; i < options.tokens.size(); ++i)
      out << (i ? "," : "") << options.tokens[i];
    out << "\nabsent_rows\t" << absent_rows
        << "\nall_relative_prediction_error\t" << std::sqrt(error2 / actual2)
        << "\nabsent_relative_prediction_error\t"
        << std::sqrt(absent_error2 / absent_actual2)
        << "\nall_prediction_sign_agree\t" << sign_equal << '\n'
        << "embedding_coordinates\t" << kEmbeddingElements
        << "\nbit_identical_predicted_weights\t" << bits_equal
        << "\nabsent_gradient_uniform_approx_relative_error\t"
        << std::sqrt(uniform_error2 / absent_gradient2)
        << "\nabsent_gradient_uniform_sign_agree\t" << uniform_sign_equal
        << "\nabsent_coordinates\t" << absent_rows * kWidth
        << "\nabsent_actual_common_mean_energy_fraction\t"
        << common_energy / absent_actual2 << "\ninput_lookup_gradient_l2\t"
        << std::sqrt(input_gradient2) << '\n';
  };
  report(summary);
  report(std::cout);
  coordinates.close();
  summary.close();
  if (!coordinates || !summary)
    return absl::UnknownError("failed writing probe reports");
  return absl::OkStatus();
}
}  // namespace
}  // namespace pluto::llm::one_shot_memorizer

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  auto options = pluto::llm::one_shot_memorizer::ReadOptions();
  if (!options.ok()) {
    std::cerr << options.status() << '\n';
    return 1;
  }
  const auto status =
      options->permute_prompt
          ? pluto::llm::one_shot_memorizer::RunPermutationSearch(*options)
          : pluto::llm::one_shot_memorizer::Run(*options);
  if (!status.ok()) {
    std::cerr << status << '\n';
    return 1;
  }
  return 0;
}
