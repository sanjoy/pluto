// Fits only one residual MLP to frozen native post-attention activations.
// Neither the original checkpoint nor its prefix/head parameters are updated.
#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_split.h"
#include "src/cuda/page_locked_host_array.h"
#include "src/dataset/compact_vocabulary.h"
#include "src/dataset/dataset.h"
#include "src/dataset/gpt2_tokenizer.h"
#include "src/dataset/padded_line_dataset.h"
#include "src/llm/adamw_optimizer.h"
#include "src/llm/checkpoint.h"
#include "src/llm/experiments/memorize_general_facts/fit_attention_readout/activation_graft.h"
#include "src/llm/experiments/memorize_general_facts/fit_attention_readout/fixed_preprocessing.h"
#include "src/llm/experiments/memorize_general_facts/fit_attention_readout/margin_loss.h"
#include "src/llm/experiments/memorize_general_facts/fit_attention_readout/readout.h"
#include "src/llm/experiments/memorize_general_facts/fit_attention_readout/subspace_graft.h"
#include "src/llm/extract_top1_ids.h"
#include "src/llm/layer_hooks.h"
#include "src/llm/layers/cross_entropy_loss.h"
#include "src/util/status_macros.h"

ABSL_FLAG(std::string, checkpoint, "", "Frozen source checkpoint.");
ABSL_FLAG(std::string, tokenizer, "", "Original GPT-2 tokenizer directory.");
ABSL_FLAG(std::string, corpus, "", "Corpus, one fact per line.");
ABSL_FLAG(std::string, output, "",
          "New directory for the best MLP checkpoint.");
ABSL_FLAG(std::string, readout_checkpoint, "",
          "Optional trainable readout checkpoint (six tensors, or eight with "
          "train_final_norm) to initialize from.");
ABSL_FLAG(
    std::string, branch_checkpoint, "",
    "Optional six-tensor MLP checkpoint; final LN starts from the source. "
    "Cannot be combined with readout_checkpoint.");
ABSL_FLAG(bool, train_final_norm, false,
          "Also train final LayerNorm scale/bias; keep the vocabulary head and "
          "source model frozen.");
ABSL_FLAG(int, block, 1, "Zero-based frozen attention boundary.");
ABSL_FLAG(
    int, graft_block, -1,
    "Optional zero-based donor attention boundary; -1 disables grafting.");
ABSL_FLAG(int, graft_dimensions, 0,
          "Number of leading raw coordinates copied from graft_block into "
          "each captured row, before LayerNorm and the residual MLP.");
ABSL_FLAG(std::string, graft_columns, "",
          "Comma-separated one-based raw dimensions, e.g. 1,7. Exclusive with "
          "graft_dimensions and graft_plane.");
ABSL_FLAG(std::string, graft_plane, "",
          "File containing model_width rows of two orthonormal FP32 basis "
          "coefficients. Graft donor projection into this shared plane, "
          "retaining the base orthogonal component.");
ABSL_FLAG(int, layers, 4, "Source checkpoint block count.");
ABSL_FLAG(int, model_width, 10, "Source hidden width.");
ABSL_FLAG(int, feed_forward_width, 20, "Source checkpoint MLP width.");
ABSL_FLAG(
    int, readout_width, 0,
    "Replacement MLP hidden width; 0 uses feed_forward_width. A different "
    "width requires fresh_branch or readout_checkpoint.");
ABSL_FLAG(int, context_length, 27, "Source context length.");
ABSL_FLAG(int, batch_size, 32, "Facts per update; must divide corpus size.");
ABSL_FLAG(
    int, steps, 20000,
    "Maximum optimizer updates; zero only evaluates the initial weights.");
ABSL_FLAG(bool, stop_on_zero_errors, true,
          "Stop when all scored tokens are correct. Disable for fixed-budget "
          "comparisons that must execute every requested update.");
ABSL_FLAG(int, eval_every, 200, "Updates between complete corpus evaluations.");
ABSL_FLAG(double, seconds, 600,
          "Wall-clock cap on fitting, excluding capture.");
ABSL_FLAG(float, learning_rate, 0.001f, "Initial Adam learning rate.");
ABSL_FLAG(float, final_rate_ratio, 0.1f, "Cosine decay end/start ratio.");
ABSL_FLAG(float, margin, 0.1f, "Desired target-minus-competitor logit margin.");
ABSL_FLAG(std::string, objective, "squared_margin",
          "Training loss: squared_margin or cross_entropy. Evaluations always "
          "report margin loss and top-1.");
ABSL_FLAG(int, seed, 0, "Minibatch shuffle seed.");
ABSL_FLAG(bool, fresh_branch, false,
          "With random_init >= 0, also reset pre-LN and biases; copy no "
          "trained branch parameters.");
ABSL_FLAG(
    int, random_init, -1,
    "-1: checkpoint MLP; otherwise matrix init seed (biases/LN retained).");
ABSL_FLAG(float, input_init_std, 0.2f,
          "Initial MLP input matrix standard deviation.");
ABSL_FLAG(float, output_init_std, 0.1f,
          "Initial MLP output matrix standard deviation; zero is allowed.");
ABSL_FLAG(
    bool, scale_output_init, false,
    "Scale output initialization by sqrt(source MLP width/readout width).");
ABSL_FLAG(bool, fresh_final_norm, false,
          "Initialize trainable final LN to identity rather than the source "
          "weights.");
ABSL_FLAG(std::string, preprocessing, "identity",
          "Fixed per-token width-preserving input transform: identity, dct, "
          "dft, sin, cos, signed_sqrt, or random_fourier. Applied before the "
          "entire residual readout, including its skip connection.");
ABSL_FLAG(float, preprocessing_scale, 1.0f,
          "Positive phase scale for sin, cos, or random_fourier.");
ABSL_FLAG(
    uint32_t, preprocessing_seed, 0,
    "Seed for the fixed random Fourier projection; no learned coefficients.");

namespace pluto::llm::fit_attention_readout {
namespace {

constexpr int kPromptTokens = 5;

absl::StatusOr<FixedPreprocessingOptions> ParsePreprocessingOptions() {
  FixedPreprocessingOptions result;
  const auto kind = absl::GetFlag(FLAGS_preprocessing);
  if (kind == "identity")
    result.kind = FixedPreprocessingKind::kIdentity;
  else if (kind == "dct")
    result.kind = FixedPreprocessingKind::kDct;
  else if (kind == "dft")
    result.kind = FixedPreprocessingKind::kRealDft;
  else if (kind == "sin")
    result.kind = FixedPreprocessingKind::kSin;
  else if (kind == "cos")
    result.kind = FixedPreprocessingKind::kCos;
  else if (kind == "signed_sqrt")
    result.kind = FixedPreprocessingKind::kSignedSqrt;
  else if (kind == "random_fourier")
    result.kind = FixedPreprocessingKind::kRandomFourier;
  else
    return absl::InvalidArgumentError("unknown fixed preprocessing");
  result.scale = absl::GetFlag(FLAGS_preprocessing_scale);
  result.seed = absl::GetFlag(FLAGS_preprocessing_seed);
  if (!std::isfinite(result.scale) || result.scale <= 0)
    return absl::InvalidArgumentError(
        "preprocessing scale must be finite and positive");
  return result;
}

// The donor is a frozen source boundary; exactly one graft representation is
// active. Columns are zero-based internally, whereas the CLI is one-based.
struct GraftOptions {
  int block = -1;
  std::vector<int> columns;
  std::vector<float> plane;
};

absl::StatusOr<GraftOptions> ParseGraftOptions() {
  GraftOptions result;
  result.block = absl::GetFlag(FLAGS_graft_block);
  const int width = absl::GetFlag(FLAGS_model_width);
  const int leading = absl::GetFlag(FLAGS_graft_dimensions);
  const std::string columns = absl::GetFlag(FLAGS_graft_columns);
  const std::string plane = absl::GetFlag(FLAGS_graft_plane);
  const int modes = (leading != 0) + !columns.empty() + !plane.empty();
  if (width <= 0 || result.block < -1 ||
      result.block >= absl::GetFlag(FLAGS_layers) ||
      (result.block == -1 && modes != 0) || (result.block >= 0 && modes != 1) ||
      leading < 0 || leading > width)
    return absl::InvalidArgumentError(
        "grafting requires a valid donor block and exactly one of "
        "graft_dimensions, graft_columns, or graft_plane");
  if (leading > 0) {
    result.columns.resize(leading);
    std::iota(result.columns.begin(), result.columns.end(), 0);
  }
  if (!columns.empty()) {
    for (absl::string_view token : absl::StrSplit(columns, ',')) {
      int column;
      if (!absl::SimpleAtoi(token, &column) || column < 1 || column > width)
        return absl::InvalidArgumentError(
            "graft_columns must contain one-based dimensions in "
            "1..model_width");
      result.columns.push_back(column - 1);
    }
    std::sort(result.columns.begin(), result.columns.end());
    if (std::adjacent_find(result.columns.begin(), result.columns.end()) !=
        result.columns.end())
      return absl::InvalidArgumentError("graft_columns contains duplicates");
  }
  if (!plane.empty()) {
    std::ifstream input(plane);
    if (!input)
      return absl::NotFoundError(
          absl::StrCat("cannot open graft plane: ", plane));
    float value;
    while (input >> value)
      result.plane.push_back(value);
    if (!input.eof() || result.plane.size() != 2 * static_cast<size_t>(width))
      return absl::InvalidArgumentError(
          "graft_plane must contain exactly two numeric columns per dimension");
    double norm0 = 0, norm1 = 0, dot = 0;
    for (int i = 0; i < width; ++i) {
      const float u = result.plane[2 * i], v = result.plane[2 * i + 1];
      if (!std::isfinite(u) || !std::isfinite(v))
        return absl::InvalidArgumentError("graft_plane must be finite");
      norm0 += static_cast<double>(u) * u;
      norm1 += static_cast<double>(v) * v;
      dot += static_cast<double>(u) * v;
    }
    if (std::abs(norm0 - 1) > 1e-5 || std::abs(norm1 - 1) > 1e-5 ||
        std::abs(dot) > 1e-5)
      return absl::InvalidArgumentError(
          "graft_plane columns must be orthonormal");
  }
  return result;
}

absl::Status Copy(cuda::Executor& executor, void* to, const void* from,
                  size_t bytes, cudaMemcpyKind kind) {
  return cuda::CudaStatus(
      cudaMemcpyAsync(to, from, bytes, kind, executor.stream()),
      "copy experiment buffer");
}

// Native BF16 snapshots, retaining padding so fitting uses the same sequence
// shape and layer kernels as inference. Only suffix-plus-EOS rows have targets.
struct Cache {
  Buffer activations;
  Buffer targets;
  cuda::PageLockedHostArray<int> host_targets;
  int sample_count;
  int scored_count;
};

// Evaluate the frozen source once, retaining the requested pre-MLP output and
// optional donor output from the SAME tokens and positions. Grafting happens
// only after source.fwd completes, into a separate buffer: modifying the base
// in its hook would also change the downstream donor and invalidate the test.
// The identical capture path serves training and greedy verification, so the
// donor is recomputed from each generated causal prefix, never future targets.
absl::StatusOr<Buffer> Capture(
    cuda::Executor& executor, const Layer& source, int block,
    const GraftOptions& graft, int model_width, const Buffer& tokens,
    std::optional<FwdResult>& full,
    const FixedPreprocessingOptions& preprocessing = {}) {
  std::vector<std::string> scopes;
  std::optional<Buffer> result;
  std::optional<Buffer> donor;
  const std::string block_name = absl::StrCat("transformer_block_", block);
  const std::string donor_name =
      absl::StrCat("transformer_block_", graft.block);
  LayerHooks hooks;
  hooks.enter_combinator = [&](auto&, auto name) {
    scopes.emplace_back(name);
    return absl::OkStatus();
  };
  hooks.exit_combinator = [&](auto&) {
    if (scopes.empty())
      return absl::DataLossError("unbalanced capture scope");
    scopes.pop_back();
    return absl::OkStatus();
  };
  hooks.activation_hook = [&](auto&, auto name, auto, auto buffers) {
    if (name == "ResidualLayer" && scopes.size() == 2 && scopes[0] == "gpt2") {
      if (buffers.size() != 1)
        return absl::DataLossError("unexpected attention output count");
      // The first residual output in a transformer block is attention; the
      // later MLP residual must not overwrite either retained snapshot.
      if (!result && scopes[1] == block_name)
        result = buffers[0];
      if (graft.block >= 0 && !donor && scopes[1] == donor_name)
        donor = buffers[0];
    }
    return absl::OkStatus();
  };
  ASSIGN_OR_RETURN(auto forward, source.fwd(executor, {&tokens, 1}, &hooks));
  forward.state = BackwardState{};
  full = std::move(forward);
  if (!result || !scopes.empty())
    return absl::DataLossError("missing post-attention capture");
  if (graft.block >= 0) {
    if (!donor)
      return absl::DataLossError("missing donor attention capture");
    if (!graft.plane.empty()) {
      ASSIGN_OR_RETURN(auto grafted, GraftBf16Plane(executor, *result, *donor,
                                                    model_width, graft.plane));
      result = std::move(grafted);
    } else {
      ASSIGN_OR_RETURN(auto grafted,
                       GraftBf16Dimensions(executor, *result, *donor,
                                           model_width, graft.columns));
      result = std::move(grafted);
    }
  }
  // This transform is outside the frozen source. Training capture and greedy
  // verification share this path, so neither can consult future target tokens.
  return PreprocessHiddenStates(executor, *result, model_width, preprocessing);
}

absl::StatusOr<Cache> CaptureCorpus(
    cuda::Executor& executor, const Layer& source, const Gpt2Config& config,
    int block, const GraftOptions& graft, const PaddedLineDataSetIterator& data,
    int eos, const FixedPreprocessingOptions& preprocessing = {}) {
  const int samples = data.sample_count();
  const int sequence = config.context_length;
  const size_t row_bytes = config.model_width * sizeof(uint16_t);
  ASSIGN_OR_RETURN(auto inputs,
                   cuda::PageLockedHostArray<int>::Allocate(executor, sequence));
  ASSIGN_OR_RETURN(auto predictions,
                   cuda::PageLockedHostArray<int>::Allocate(executor, sequence));
  ASSIGN_OR_RETURN(auto targets, cuda::PageLockedHostArray<int>::Allocate(
                                     executor, samples * sequence));
  std::fill(targets.begin(), targets.end(), -1);
  ASSIGN_OR_RETURN(auto device_input,
                   Buffer::Allocate(executor, inputs.size_bytes()));
  ASSIGN_OR_RETURN(auto device_mask,
                   Buffer::Allocate(executor, inputs.size_bytes()));
  RETURN_IF_ERROR(cuda::CudaStatus(
      cudaMemsetAsync(device_mask.data(), 0, device_mask.size_bytes(),
                      executor.stream()),
      "clear mask"));
  ASSIGN_OR_RETURN(auto activations,
                   Buffer::Allocate(executor, samples * sequence * row_bytes));
  ASSIGN_OR_RETURN(auto device_targets,
                   Buffer::Allocate(executor, targets.size_bytes()));
  int scored = 0;
  for (int sample = 0; sample < samples; ++sample) {
    const auto tokens = data.sample_tokens(sample);
    std::fill(inputs.begin(), inputs.end(), eos);
    std::copy(tokens.begin(), tokens.end(), inputs.begin());
    RETURN_IF_ERROR(Copy(executor, device_input.data(), inputs.data(),
                         inputs.size_bytes(), cudaMemcpyHostToDevice));
    std::optional<FwdResult> full;
    ASSIGN_OR_RETURN(auto captured,
                     Capture(executor, source, block, graft, config.model_width,
                             device_input, full, preprocessing));
    if (captured.size_bytes() != sequence * row_bytes)
      return absl::DataLossError("unexpected activation shape");
    RETURN_IF_ERROR(Copy(
        executor,
        static_cast<char*>(activations.data()) + sample * sequence * row_bytes,
        captured.data(), captured.size_bytes(), cudaMemcpyDeviceToDevice));
    ASSIGN_OR_RETURN(auto ids,
                     ExtractTop1Ids(executor, full->outputs[0], device_mask,
                                    config.vocabulary_size));
    RETURN_IF_ERROR(Copy(executor, predictions.data(), ids.data(),
                         predictions.size_bytes(), cudaMemcpyDeviceToHost));
    RETURN_IF_ERROR(
        executor.Synchronize());  // Host input is reused next sample.
    for (size_t row = kPromptTokens - 1; row < tokens.size(); ++row) {
      const int target = row + 1 < tokens.size() ? tokens[row + 1] : eos;
      if (predictions[row] != target)
        return absl::FailedPreconditionError(
            "source checkpoint is not fully memorized");
      targets[sample * sequence + row] = target;
      ++scored;
    }
  }
  RETURN_IF_ERROR(Copy(executor, device_targets.data(), targets.data(),
                       targets.size_bytes(), cudaMemcpyHostToDevice));
  // A nonlinear fixed transform can erase distinctions. Audit exact BF16
  // vectors on scored rows; labels are used only for this diagnostic, never
  // to construct preprocessing features. Canonicalize signed zero as equal.
  ASSIGN_OR_RETURN(auto host_activations,
                   cuda::PageLockedHostArray<uint16_t>::Allocate(
                       executor, samples * sequence * config.model_width));
  RETURN_IF_ERROR(Copy(executor, host_activations.data(), activations.data(),
                       activations.size_bytes(), cudaMemcpyDeviceToHost));
  RETURN_IF_ERROR(executor.Synchronize());
  absl::flat_hash_map<std::string, int> vector_targets;
  int conflicting_vectors = 0;
  for (int row = 0; row < samples * sequence; ++row) {
    if (targets[row] < 0)
      continue;
    auto* values = host_activations.data() + row * config.model_width;
    for (int column = 0; column < config.model_width; ++column) {
      if ((values[column] & 0x7f80) == 0x7f80)
        return absl::DataLossError("nonfinite preprocessed input");
      if (values[column] == 0x8000)
        values[column] = 0;
    }
    std::string key(reinterpret_cast<const char*>(values), row_bytes);
    auto [it, inserted] = vector_targets.emplace(std::move(key), targets[row]);
    if (!inserted && it->second != targets[row] && it->second >= 0) {
      it->second = -1;
      ++conflicting_vectors;
    }
  }
  std::cout << "preprocessed_unique_vectors=" << vector_targets.size()
            << " scored_vectors=" << scored
            << " conflicting_vectors=" << conflicting_vectors << std::endl;
  return Cache{std::move(activations), std::move(device_targets),
               std::move(targets), samples, scored};
}

// A fixed-size reusable GPU minibatch. Loading is D2D; host labels are
// consulted only to count scored rows, so padding does not change the loss
// normalization.
struct Batch {
  Buffer x;
  Buffer targets;
  int normalizer = 0;
};

absl::Status LoadBatch(cuda::Executor& executor, const Cache& cache,
                       const Gpt2Config& config, absl::Span<const int> indices,
                       Batch& batch) {
  const int sequence = config.context_length;
  const size_t x_bytes = sequence * config.model_width * sizeof(uint16_t);
  const size_t target_bytes = sequence * sizeof(int);
  batch.normalizer = 0;
  for (size_t i = 0; i < indices.size(); ++i) {
    const int index = indices[i];
    RETURN_IF_ERROR(
        Copy(executor, static_cast<char*>(batch.x.data()) + i * x_bytes,
             static_cast<char*>(cache.activations.data()) + index * x_bytes,
             x_bytes, cudaMemcpyDeviceToDevice));
    RETURN_IF_ERROR(Copy(
        executor, static_cast<char*>(batch.targets.data()) + i * target_bytes,
        static_cast<char*>(cache.targets.data()) + index * target_bytes,
        target_bytes, cudaMemcpyDeviceToDevice));
    for (int row = 0; row < sequence; ++row)
      batch.normalizer += cache.host_targets[index * sequence + row] >= 0;
  }
  if (batch.normalizer == 0)
    return absl::DataLossError("batch has no scored rows");
  return absl::OkStatus();
}

struct Metrics {
  int wrong = 0;    // Scored positions whose top-1 ID is not the target.
  int failed = 0;   // Facts with at least one wrong teacher-forced position.
  double loss = 0;  // Mean squared margin violation over scored rows.
  double cross_entropy =
      0;  // Actual mean CE, independent of the fit objective.
  float min_margin = std::numeric_limits<float>::infinity();
};

absl::StatusOr<Metrics> Evaluate(cuda::Executor& executor, const Layer& tail,
                                 const Cache& cache, const Gpt2Config& config,
                                 int batch_size, float margin, Batch& batch) {
  const int rows = batch_size * config.context_length;
  ASSIGN_OR_RETURN(auto losses,
                   cuda::PageLockedHostArray<float>::Allocate(executor, rows));
  ASSIGN_OR_RETURN(auto margins,
                   cuda::PageLockedHostArray<float>::Allocate(executor, rows));
  ASSIGN_OR_RETURN(auto ids,
                   cuda::PageLockedHostArray<int>::Allocate(executor, rows));
  ASSIGN_OR_RETURN(auto ce_values,
                   cuda::PageLockedHostArray<float>::Allocate(executor, rows));
  ASSIGN_OR_RETURN(
      auto cross_entropy,
      CrossEntropyLossLayer::Create(executor, config.vocabulary_size,
                                    DataType::BF16, config.context_length));
  std::vector<int> indices(batch_size);
  Metrics result;
  for (int start = 0; start < cache.sample_count; start += batch_size) {
    std::iota(indices.begin(), indices.end(), start);
    RETURN_IF_ERROR(LoadBatch(executor, cache, config, indices, batch));
    ASSIGN_OR_RETURN(auto forward, tail.fwd(executor, {&batch.x, 1}));
    const BufferVec ce_inputs{forward.outputs[0], batch.targets};
    ASSIGN_OR_RETURN(auto ce, cross_entropy->fwd(executor, ce_inputs));
    RETURN_IF_ERROR(Copy(executor, ce_values.data(), ce.outputs[0].data(),
                         ce_values.size_bytes(), cudaMemcpyDeviceToHost));
    ASSIGN_OR_RETURN(
        auto loss,
        SquaredMarginLoss(executor, forward.outputs[0], batch.targets,
                          config.vocabulary_size, margin, batch.normalizer));
    RETURN_IF_ERROR(Copy(executor, losses.data(), loss.losses.data(),
                         losses.size_bytes(), cudaMemcpyDeviceToHost));
    RETURN_IF_ERROR(Copy(executor, margins.data(), loss.margins.data(),
                         margins.size_bytes(), cudaMemcpyDeviceToHost));
    RETURN_IF_ERROR(Copy(executor, ids.data(), loss.predicted_ids.data(),
                         ids.size_bytes(), cudaMemcpyDeviceToHost));
    RETURN_IF_ERROR(executor.Synchronize());
    for (int sample = 0; sample < batch_size; ++sample) {
      bool failed = false;
      for (int row = 0; row < config.context_length; ++row) {
        const int local = sample * config.context_length + row;
        const int target =
            cache.host_targets[(start + sample) * config.context_length + row];
        if (target < 0)
          continue;  // Prompt and right-padding rows are not completion
                     // targets.
        if (!std::isfinite(losses[local]) || !std::isfinite(ce_values[local]) ||
            !std::isfinite(margins[local]) || ids[local] < 0)
          return absl::DataLossError("nonfinite fit output");
        result.loss += losses[local];
        result.cross_entropy += ce_values[local];
        result.min_margin = std::min(result.min_margin, margins[local]);
        result.wrong += ids[local] != target;
        failed |= ids[local] != target;
      }
      result.failed += failed;
    }
  }
  result.loss /= cache.scored_count;
  result.cross_entropy /= cache.scored_count;
  return result;
}

// Evaluate actual generated prefixes, not the fixed cached training vectors.
// Stop each fact at its first mistake: its exact completion is then disproved.
absl::StatusOr<int> VerifyCompletions(
    cuda::Executor& executor, const Layer& source, const Layer& tail,
    const Gpt2Config& config, int block, const GraftOptions& graft,
    const PaddedLineDataSetIterator& data, int eos,
    const FixedPreprocessingOptions& preprocessing = {}) {
  ASSIGN_OR_RETURN(auto input, cuda::PageLockedHostArray<int>::Allocate(
                                   executor, config.context_length));
  ASSIGN_OR_RETURN(auto ids, cuda::PageLockedHostArray<int>::Allocate(
                                 executor, config.context_length));
  ASSIGN_OR_RETURN(auto device_input,
                   Buffer::Allocate(executor, input.size_bytes()));
  ASSIGN_OR_RETURN(auto mask, Buffer::Allocate(executor, input.size_bytes()));
  RETURN_IF_ERROR(cuda::CudaStatus(
      cudaMemsetAsync(mask.data(), 0, mask.size_bytes(), executor.stream()),
      "clear verification mask"));
  int correct = 0;
  for (size_t sample = 0; sample < data.sample_count(); ++sample) {
    auto tokens = data.sample_tokens(sample);
    std::fill(input.begin(), input.end(), eos);
    std::copy_n(tokens.begin(), kPromptTokens, input.begin());
    bool failed = false;
    for (size_t row = kPromptTokens - 1; row < tokens.size(); ++row) {
      RETURN_IF_ERROR(Copy(executor, device_input.data(), input.data(),
                           input.size_bytes(), cudaMemcpyHostToDevice));
      std::optional<FwdResult> unused;
      ASSIGN_OR_RETURN(
          auto x, Capture(executor, source, block, graft, config.model_width,
                          device_input, unused, preprocessing));
      ASSIGN_OR_RETURN(auto forward, tail.fwd(executor, {&x, 1}));
      ASSIGN_OR_RETURN(auto predicted,
                       ExtractTop1Ids(executor, forward.outputs[0], mask,
                                      config.vocabulary_size));
      RETURN_IF_ERROR(Copy(executor, ids.data(), predicted.data(),
                           ids.size_bytes(), cudaMemcpyDeviceToHost));
      RETURN_IF_ERROR(executor.Synchronize());
      const int expected = row + 1 < tokens.size() ? tokens[row + 1] : eos;
      if (ids[row] != expected) {
        failed = true;
        break;
      }
      if (row + 1 < tokens.size())
        input[row + 1] = ids[row];
    }
    correct += !failed;
  }
  return correct;
}

// Snapshot only the frozen copy's weights to catch accidental optimizer scope
// expansion. The trainable prefix has six tensors, or eight with final LN.
absl::StatusOr<std::vector<uint8_t>> FrozenWeights(cuda::Executor& executor,
                                                   const Readout& readout) {
  std::vector<uint8_t> result;
  for (const auto& weight :
       readout.model->weights().subspan(readout.trainable->weights().size())) {
    ASSIGN_OR_RETURN(auto host, cuda::PageLockedHostArray<uint8_t>::Allocate(
                                    executor, weight.size_bytes()));
    RETURN_IF_ERROR(Copy(executor, host.data(), weight.data(),
                         host.size_bytes(), cudaMemcpyDeviceToHost));
    RETURN_IF_ERROR(executor.Synchronize());
    result.insert(result.end(), host.begin(), host.end());
  }
  return result;
}

absl::Status Run() {
  const auto output = absl::GetFlag(FLAGS_output);
  const auto checkpoint = absl::GetFlag(FLAGS_checkpoint);
  const int batch_size = absl::GetFlag(FLAGS_batch_size);
  const int steps = absl::GetFlag(FLAGS_steps);
  const int eval_every = absl::GetFlag(FLAGS_eval_every);
  const float margin = absl::GetFlag(FLAGS_margin);
  const float lr = absl::GetFlag(FLAGS_learning_rate);
  const float ratio = absl::GetFlag(FLAGS_final_rate_ratio);
  const double seconds = absl::GetFlag(FLAGS_seconds);
  const std::string objective = absl::GetFlag(FLAGS_objective);
  const int readout_width = absl::GetFlag(FLAGS_readout_width);
  ASSIGN_OR_RETURN(const auto graft, ParseGraftOptions());
  ASSIGN_OR_RETURN(const auto preprocessing, ParsePreprocessingOptions());
  const ReadoutInitializationOptions initialization{
      .input_standard_deviation = absl::GetFlag(FLAGS_input_init_std),
      .output_standard_deviation = absl::GetFlag(FLAGS_output_init_std),
      .scale_output_by_width = absl::GetFlag(FLAGS_scale_output_init),
      .fresh_final_norm = absl::GetFlag(FLAGS_fresh_final_norm)};
  const bool restoring_readout =
      !absl::GetFlag(FLAGS_readout_checkpoint).empty();
  const bool restoring_branch = !absl::GetFlag(FLAGS_branch_checkpoint).empty();
  const bool restoring = restoring_readout || restoring_branch;
  const bool different_width =
      readout_width > 0 &&
      readout_width != absl::GetFlag(FLAGS_feed_forward_width);
  if (objective != "squared_margin" && objective != "cross_entropy")
    return absl::InvalidArgumentError("unknown training objective");
  if (restoring_readout && restoring_branch)
    return absl::InvalidArgumentError(
        "choose only one of readout_checkpoint and branch_checkpoint");
  if (restoring && (absl::GetFlag(FLAGS_random_init) >= 0 ||
                    absl::GetFlag(FLAGS_fresh_branch)))
    return absl::InvalidArgumentError(
        "checkpoint initialization cannot be combined with random "
        "initialization");
  if (readout_width < 0 ||
      (different_width && !restoring && !absl::GetFlag(FLAGS_fresh_branch)))
    return absl::InvalidArgumentError(
        "readout_width must be nonnegative; changing width requires "
        "fresh_branch or a readout/branch checkpoint");
  if (output.empty() || checkpoint.empty() || batch_size <= 0 || steps < 0 ||
      eval_every <= 0 || !(seconds > 0) || !std::isfinite(seconds) ||
      !(margin > 0) || !std::isfinite(margin) || !(ratio > 0 && ratio <= 1) ||
      !(lr > 0) || !std::isfinite(lr))
    return absl::InvalidArgumentError("invalid paths or fitting options");
  if (std::filesystem::exists(output))
    return absl::AlreadyExistsError(
        "output directory must be new; source is never overwritten");
  ASSIGN_OR_RETURN(auto original, tokenizer::Gpt2Tokenizer::Load(
                                      absl::GetFlag(FLAGS_tokenizer)));
  ASSIGN_OR_RETURN(auto vocabulary,
                   tokenizer::CompactVocabularyTokenizer::LoadFromFile(
                       *original, std::filesystem::path(checkpoint) /
                                      "compact_vocabulary.tsv"));
  const Gpt2Config config{
      .transformer_block_count = absl::GetFlag(FLAGS_layers),
      .model_width = absl::GetFlag(FLAGS_model_width),
      .attention_heads = 1,
      .feed_forward_width = absl::GetFlag(FLAGS_feed_forward_width),
      .vocabulary_size = vocabulary->vocab_size(),
      .pad_vocabulary = false,
      .context_length = absl::GetFlag(FLAGS_context_length)};
  RETURN_IF_ERROR(config.Validate());
  ASSIGN_OR_RETURN(auto executor, cuda::Executor::Create());
  ASSIGN_OR_RETURN(auto corpus, LoadTextCorpus(absl::GetFlag(FLAGS_corpus)));
  ASSIGN_OR_RETURN(auto data, PaddedLineDataSetIterator::Create(
                                  *executor, corpus.text(), *vocabulary,
                                  {.batch_size = 1,
                                   .context_length = config.context_length,
                                   .prompt_tokens = kPromptTokens,
                                   .eos_token = vocabulary->eos_token_id()}));
  if (data->sample_count() % batch_size != 0)
    return absl::InvalidArgumentError("batch size must divide fact count");
  ASSIGN_OR_RETURN(auto source,
                   CreateGpt2(*executor, DataType::BF16, 0, config));
  RETURN_IF_ERROR(ReadFromDirectory(*executor, *source, checkpoint, false));
  // A differently sized checkpoint cannot be initialized by copying the source
  // MLP. Construct a fresh branch, then overwrite every one of its six tensors.
  // This temporary seed has no effect on the restored weights or Adam state.
  const bool fresh =
      absl::GetFlag(FLAGS_fresh_branch) || (restoring && different_width);
  const int initialization_seed =
      restoring && different_width ? 0 : absl::GetFlag(FLAGS_random_init);
  ASSIGN_OR_RETURN(
      auto readout,
      CreateReadout(*executor, *source, config, absl::GetFlag(FLAGS_block),
                    initialization_seed, fresh, readout_width,
                    absl::GetFlag(FLAGS_train_final_norm), initialization));
  if (restoring_readout)
    RETURN_IF_ERROR(ReadFromDirectory(*executor, *readout.trainable,
                                      absl::GetFlag(FLAGS_readout_checkpoint),
                                      false));
  if (restoring_branch)
    RETURN_IF_ERROR(ReadFromDirectory(*executor, *readout.branch,
                                      absl::GetFlag(FLAGS_branch_checkpoint),
                                      false));
  ASSIGN_OR_RETURN(const auto frozen, FrozenWeights(*executor, readout));
  size_t parameters = 0;
  for (const auto& weight : readout.trainable->weights())
    parameters += weight.size_bytes() / sizeof(float);
  std::cout << "Capturing block " << absl::GetFlag(FLAGS_block)
            << "; source_mlp_width=" << config.feed_forward_width
            << "; readout_mlp_width="
            << (readout_width == 0 ? config.feed_forward_width : readout_width)
            << "; train_final_norm=" << absl::GetFlag(FLAGS_train_final_norm)
            << "; graft_block=" << graft.block << "; graft_leading_dimensions="
            << absl::GetFlag(FLAGS_graft_dimensions)
            << "; graft_columns=" << absl::GetFlag(FLAGS_graft_columns)
            << "; graft_plane=" << absl::GetFlag(FLAGS_graft_plane)
            << "; preprocessing=" << absl::GetFlag(FLAGS_preprocessing)
            << "; preprocessing_scale=" << preprocessing.scale
            << "; preprocessing_seed=" << preprocessing.seed
            << "; trainable parameters=" << parameters << std::endl;
  ASSIGN_OR_RETURN(
      auto cache,
      CaptureCorpus(*executor, *source, config, absl::GetFlag(FLAGS_block),
                    graft, *data, vocabulary->eos_token_id(), preprocessing));
  std::cout << "Source verified: " << cache.sample_count << " facts; "
            << cache.scored_count << " scored targets.\n";
  const int rows = batch_size * config.context_length;
  ASSIGN_OR_RETURN(
      auto x,
      Buffer::Allocate(*executor, rows * config.model_width * sizeof(uint16_t)));
  ASSIGN_OR_RETURN(auto targets,
                   Buffer::Allocate(*executor, rows * sizeof(int)));
  Batch batch{std::move(x), std::move(targets)};
  ASSIGN_OR_RETURN(auto optimizer,
                   AdamWOptimizer::Create(*executor, *readout.trainable,
                                          {.learning_rate = lr,
                                           .beta1 = 0.9f,
                                           .beta2 = 0.999f,
                                           .epsilon = 1e-8f,
                                           .weight_decay = 0}));
  RETURN_IF_ERROR(optimizer->ZeroGrad());
  ASSIGN_OR_RETURN(
      auto cross_entropy,
      CrossEntropyLossLayer::Create(*executor, config.vocabulary_size,
                                    DataType::BF16, config.context_length));
  std::cout << "objective=" << objective << " learning_rate=" << lr
            << " batch_size=" << batch_size << " margin=" << margin
            << std::endl;
  const auto begin = std::chrono::steady_clock::now();
  auto elapsed = [&] {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                         begin)
        .count();
  };
  Metrics best;
  best.wrong = std::numeric_limits<int>::max();
  int best_step = 0;
  int last_eval = -1;
  auto evaluate = [&](int step) -> absl::StatusOr<bool> {
    ASSIGN_OR_RETURN(auto metrics, Evaluate(*executor, *readout.model, cache,
                                            config, batch_size, margin, batch));
    last_eval = step;
    std::cout << absl::StrFormat(
                     "step=%d seconds=%.2f wrong=%d/%d failed_facts=%d/%d "
                     "margin_loss=%.8f min_margin=%.6f cross_entropy=%.8f\n",
                     step, elapsed(), metrics.wrong, cache.scored_count,
                     metrics.failed, cache.sample_count, metrics.loss,
                     metrics.min_margin, metrics.cross_entropy)
              << std::flush;
    if (metrics.wrong < best.wrong ||
        (metrics.wrong == best.wrong && metrics.loss < best.loss)) {
      best = metrics;
      best_step = step;
      RETURN_IF_ERROR(
          WriteToDirectory(*executor, *readout.trainable,
                           std::filesystem::path(output) / "best_mlp"));
    }
    return metrics.wrong == 0 && absl::GetFlag(FLAGS_stop_on_zero_errors);
  };
  ASSIGN_OR_RETURN(bool done, evaluate(0));
  std::vector<int> order(cache.sample_count);
  std::iota(order.begin(), order.end(), 0);
  std::mt19937 random(absl::GetFlag(FLAGS_seed));
  int cursor = cache.sample_count;
  int completed = 0;
  for (int step = 1; step <= steps && !done && elapsed() < seconds; ++step) {
    if (cursor == cache.sample_count) {
      std::shuffle(order.begin(), order.end(), random);
      cursor = 0;
    }
    RETURN_IF_ERROR(LoadBatch(
        *executor, cache, config,
        absl::MakeConstSpan(order).subspan(cursor, batch_size), batch));
    cursor += batch_size;
    const float progress = static_cast<float>(step - 1) / steps;
    const float current_lr =
        lr * (ratio + (1 - ratio) * 0.5f *
                          (1 + std::cos(3.14159265358979323846f * progress)));
    RETURN_IF_ERROR(optimizer->SetLearningRate(current_lr));
    ASSIGN_OR_RETURN(auto forward, readout.model->fwd(*executor, {&batch.x, 1}));
    BufferVec logit_gradients;
    if (objective == "squared_margin") {
      ASSIGN_OR_RETURN(
          auto loss,
          SquaredMarginLoss(*executor, forward.outputs[0], batch.targets,
                            config.vocabulary_size, margin, batch.normalizer));
      logit_gradients.push_back(std::move(loss.gradients));
    } else {
      const BufferVec loss_inputs{forward.outputs[0], batch.targets};
      ASSIGN_OR_RETURN(auto loss, cross_entropy->fwd(*executor, loss_inputs));
      ASSIGN_OR_RETURN(logit_gradients,
                       cross_entropy->bwd(*executor, {}, std::move(loss.state)));
    }
    // The head (and final LN unless explicitly trained) receives gradients to
    // propagate into the readout but never enters the optimizer. Clear only
    // these frozen tensors; trainable final-LN gradients must reach Adam.
    for (const auto& gradient : readout.model->gradients().subspan(
             readout.trainable->weights().size()))
      RETURN_IF_ERROR(cuda::CudaStatus(
          cudaMemsetAsync(gradient.data(), 0, gradient.size_bytes(),
                          executor->stream()),
          "clear frozen gradients"));
    RETURN_IF_ERROR(
        readout.model->bwd(*executor, logit_gradients, std::move(forward.state))
            .status());
    RETURN_IF_ERROR(optimizer->ApplyStep());
    completed = step;
    if (step % eval_every == 0) {
      ASSIGN_OR_RETURN(done, evaluate(step));
    }
    // Bound enqueued work and enforce the wall-clock cap without allowing an
    // unbounded async backlog. This also exposes CUDA failures promptly.
    if (step % 32 == 0)
      RETURN_IF_ERROR(executor->Synchronize());
  }
  if (last_eval != completed) {
    ASSIGN_OR_RETURN(done, evaluate(completed));
  }
  RETURN_IF_ERROR(ReadFromDirectory(*executor, *readout.trainable,
                                    std::filesystem::path(output) / "best_mlp",
                                    false));
  ASSIGN_OR_RETURN(auto after, FrozenWeights(*executor, readout));
  if (frozen != after)
    return absl::InternalError("frozen final LN or embedding changed");
  ASSIGN_OR_RETURN(auto final_metrics,
                   Evaluate(*executor, *readout.model, cache, config,
                            batch_size, margin, batch));
  ASSIGN_OR_RETURN(const int correct,
                   VerifyCompletions(*executor, *source, *readout.model, config,
                                     absl::GetFlag(FLAGS_block), graft, *data,
                                     vocabulary->eos_token_id(), preprocessing));
  std::cout
      << absl::StrFormat(
             "FINAL best_step=%d updates=%d seconds=%.2f wrong=%d/%d "
             "autoregressive_complete=%d/%d frozen_weights_unchanged=true "
             "cross_entropy=%.8f\n",
             best_step, completed, elapsed(), final_metrics.wrong,
             cache.scored_count, correct, cache.sample_count,
             final_metrics.cross_entropy)
      << std::flush;
  return absl::OkStatus();
}

}  // namespace
}  // namespace pluto::llm::fit_attention_readout

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  const auto status = pluto::llm::fit_attention_readout::Run();
  if (!status.ok()) {
    std::cerr << status << '\n';
    return 1;
  }
  return 0;
}
