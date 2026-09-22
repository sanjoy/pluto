#include "src/llm/experiments/memorize_general_facts/discretized_model/discrete_model_generator/snapshot_execution_trace.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "src/cuda/buffer.h"
#include "src/cuda/page_locked_host_array.h"
#include "src/llm/extract_top1_ids.h"
#include "src/llm/layer_hooks.h"
#include "src/util/status_macros.h"

namespace pluto::llm::memorize_general_facts::discretized_model {
namespace {

absl::Status ValidateOptions(const CaptureOptions& options) {
  if (options.layers < 0 ||
      options.layers > (std::numeric_limits<int>::max() - 1) / 2 ||
      options.vocab_size <= 0 || options.eos_token < 0 ||
      options.eos_token >= options.vocab_size || options.prompt_tokens <= 0 ||
      options.prompt_tokens > kCaptureContext)
    return absl::InvalidArgumentError(
        "invalid capture dimensions or token IDs");
  return absl::OkStatus();
}

// Copies are queued at the observation point, before later layers can release
// their buffers. Host staging remains alive until the final synchronization.
class BoundaryRecorder {
 public:
  BoundaryRecorder(cuda::Executor& executor, size_t rows, int layers)
      : executor_(executor), rows_(rows), stage_count_(2 * layers + 1) {}

  LayerHooks Hooks() {
    return {
        .activation_hook =
            [this](auto& executor, auto name, auto types, auto buffers) {
              return Observe(executor, name, types, buffers);
            },
        .enter_combinator =
            [this](auto& executor, auto name) {
              if (&executor != &executor_)
                return absl::InvalidArgumentError(
                    "capture scope executor mismatch");
              scopes_.emplace_back(name);
              return absl::OkStatus();
            },
        .exit_combinator =
            [this](auto& executor) {
              if (&executor != &executor_ || scopes_.empty())
                return absl::InvalidArgumentError("unbalanced capture scope");
              scopes_.pop_back();
              return absl::OkStatus();
            }};
  }

  absl::Status ValidateComplete() const {
    if (!scopes_.empty() || pending_residual_ ||
        staged_.size() != static_cast<size_t>(stage_count_))
      return absl::DataLossError(
          "GPT-2 capture did not observe every residual boundary exactly once");
    return absl::OkStatus();
  }

  std::vector<std::vector<CapturedRow>> Materialize() const {
    std::vector<std::vector<CapturedRow>> result;
    result.reserve(staged_.size());
    for (const auto& stage : staged_)
      result.emplace_back(stage.begin(), stage.end());
    return result;
  }

 private:
  absl::Status Observe(cuda::Executor& executor, absl::string_view name,
                       absl::Span<const ActivationType> types,
                       absl::Span<Buffer> buffers) {
    if (name == "attention" || name == "mlp") {
      const size_t stage = staged_.size();
      if (stage == 0 || stage >= static_cast<size_t>(stage_count_) ||
          pending_residual_ || scopes_.size() != 3 || scopes_[0] != "gpt2" ||
          scopes_[1] != absl::StrCat("transformer_block_", (stage - 1) / 2) ||
          scopes_[2] != "ResidualLayer" ||
          name != (stage % 2 == 1 ? "attention" : "mlp"))
        return absl::DataLossError(
            "attention/MLP residual branches are out of order");
      pending_residual_ = true;
      return absl::OkStatus();
    }
    if (name != "PositionEmbeddingLayer" && name != "ResidualLayer")
      return absl::OkStatus();
    const size_t stage = staged_.size();
    if (stage >= static_cast<size_t>(stage_count_) || scopes_.empty() ||
        scopes_[0] != "gpt2")
      return absl::DataLossError("unexpected GPT-2 capture boundary or root");
    if (name == "PositionEmbeddingLayer") {
      if (stage != 0 || scopes_.size() != 1)
        return absl::DataLossError("token-position boundary is out of order");
    } else {
      if (stage == 0 || !pending_residual_ || scopes_.size() != 2 ||
          scopes_[1] != absl::StrCat("transformer_block_", (stage - 1) / 2))
        return absl::DataLossError(
            "residual boundary has wrong block or scope");
      pending_residual_ = false;
    }
    const ActivationType expected(
        DataType::BF16,
        {ActivationType::kBatchDimension, kCaptureContext, kCaptureWidth});
    if (&executor != &executor_ || types.size() != 1 || types[0] != expected ||
        buffers.size() != 1 || &buffers[0].executor() != &executor_ ||
        buffers[0].size_bytes() != kCaptureContext * sizeof(CapturedRow))
      return absl::InvalidArgumentError(
          "capture boundary must be batch-one native BF16 [-2,1024,16]");
    ASSIGN_OR_RETURN(
        auto staging,
        cuda::PageLockedHostArray<CapturedRow>::Allocate(executor_, rows_));
    RETURN_IF_ERROR(cuda::CudaStatus(
        cudaMemcpyAsync(staging.data(), buffers[0].data(), staging.size_bytes(),
                        cudaMemcpyDeviceToHost, executor_.stream()),
        "download native BF16 residual boundary"));
    staged_.push_back(std::move(staging));
    return absl::OkStatus();
  }

  cuda::Executor& executor_;
  size_t rows_;
  int stage_count_;
  bool pending_residual_ = false;
  std::vector<std::string> scopes_;
  std::vector<cuda::PageLockedHostArray<CapturedRow>> staged_;
};

}  // namespace

absl::StatusOr<CapturedSample> CaptureSample(cuda::Executor& executor,
                                             const Layer& model,
                                             absl::Span<const int> tokens,
                                             const CaptureOptions& options) {
  RETURN_IF_ERROR(ValidateOptions(options));
  if (tokens.empty() || tokens.size() > kCaptureContext)
    return absl::InvalidArgumentError(
        "capture needs 1 to 1024 real token rows");
  for (int token : tokens)
    if (token < 0 || token >= options.vocab_size || token == options.eos_token)
      return absl::InvalidArgumentError(
          "capture text contains invalid or EOS ID");
  const ActivationType input_type(
      DataType::INT32, {ActivationType::kBatchDimension, kCaptureContext});
  const auto output_types = model.output_types();
  const int64_t padded_vocabulary =
      (static_cast<int64_t>(options.vocab_size) + 15) / 16 * 16;
  const ActivationType output_type(
      DataType::FP32,
      {ActivationType::kBatchDimension, kCaptureContext, padded_vocabulary});
  if (model.name() != "gpt2" || model.output_type() != DataType::BF16 ||
      model.input_types().size() != 1 || model.input_types()[0] != input_type ||
      output_types.size() != 1 || output_types[0] != output_type)
    return absl::InvalidArgumentError(
        "capture requires the native BF16 fixed-context GPT-2 model");

  ASSIGN_OR_RETURN(auto context, cuda::PageLockedHostArray<int>::Allocate(
                                     executor, kCaptureContext));
  std::fill(context.begin(), context.end(), options.eos_token);
  std::copy(tokens.begin(), tokens.end(), context.begin());
  ASSIGN_OR_RETURN(auto mask, cuda::PageLockedHostArray<int>::Allocate(
                                  executor, kCaptureContext));
  std::fill(mask.begin(), mask.end(), -1);
  std::fill_n(mask.begin(), tokens.size(), 0);
  ASSIGN_OR_RETURN(auto device_context,
                   Buffer::Allocate(executor, context.size_bytes()));
  ASSIGN_OR_RETURN(auto device_mask,
                   Buffer::Allocate(executor, mask.size_bytes()));
  RETURN_IF_ERROR(cuda::CudaStatus(
      cudaMemcpyAsync(device_context.data(), context.data(),
                      context.size_bytes(), cudaMemcpyHostToDevice,
                      executor.stream()),
      "upload capture context"));
  RETURN_IF_ERROR(cuda::CudaStatus(
      cudaMemcpyAsync(device_mask.data(), mask.data(), mask.size_bytes(),
                      cudaMemcpyHostToDevice, executor.stream()),
      "upload capture real-row mask"));
  BoundaryRecorder recorder(executor, tokens.size(), options.layers);
  auto hooks = recorder.Hooks();
  ASSIGN_OR_RETURN(auto forward,
                   model.fwd(executor, {&device_context, 1}, &hooks));
  RETURN_IF_ERROR(recorder.ValidateComplete());
  forward.state = BackwardState{};
  if (forward.outputs.size() != 1 ||
      &forward.outputs[0].executor() != &executor ||
      forward.outputs[0].size_bytes() !=
          static_cast<size_t>(kCaptureContext * padded_vocabulary) *
              sizeof(float))
    return absl::InvalidArgumentError("capture logits have wrong storage");
  ASSIGN_OR_RETURN(auto predicted,
                   ExtractTop1Ids(executor, forward.outputs[0], device_mask,
                                  options.vocab_size));
  ASSIGN_OR_RETURN(
      auto host_predictions,
      cuda::PageLockedHostArray<int>::Allocate(executor, tokens.size()));
  RETURN_IF_ERROR(cuda::CudaStatus(
      cudaMemcpyAsync(host_predictions.data(), predicted.data(),
                      host_predictions.size_bytes(), cudaMemcpyDeviceToHost,
                      executor.stream()),
      "download real-row predictions"));
  RETURN_IF_ERROR(executor.Synchronize());
  for (int prediction : host_predictions)
    if (prediction < 0 || prediction >= options.vocab_size)
      return absl::DataLossError(
          "capture produced nonfinite or invalid logits");
  return CapturedSample{
      .tokens = {tokens.begin(), tokens.end()},
      .predictions = {host_predictions.begin(), host_predictions.end()},
      .boundaries = recorder.Materialize()};
}

absl::Status ValidateCapturedPredictions(const CapturedSample& sample,
                                         const CaptureOptions& options) {
  RETURN_IF_ERROR(ValidateOptions(options));
  if (sample.tokens.size() < static_cast<size_t>(options.prompt_tokens) ||
      sample.tokens.size() > kCaptureContext ||
      sample.predictions.size() != sample.tokens.size() ||
      sample.boundaries.size() != static_cast<size_t>(2 * options.layers + 1))
    return absl::InvalidArgumentError(
        "captured sample has inconsistent lengths");
  for (const auto& stage : sample.boundaries)
    if (stage.size() != sample.tokens.size())
      return absl::InvalidArgumentError(
          "captured boundary has padding or missing rows");
  for (size_t row = 0; row < sample.tokens.size(); ++row) {
    if (sample.tokens[row] < 0 || sample.tokens[row] >= options.vocab_size ||
        sample.tokens[row] == options.eos_token ||
        sample.predictions[row] < 0 ||
        sample.predictions[row] >= options.vocab_size)
      return absl::InvalidArgumentError(
          "captured sample has invalid token IDs");
    if (row + 1 < static_cast<size_t>(options.prompt_tokens))
      continue;
    const int target = row + 1 == sample.tokens.size() ? options.eos_token
                                                       : sample.tokens[row + 1];
    if (sample.predictions[row] != target)
      return absl::DataLossError(absl::StrCat(
          "checkpoint is not exact at real row ", row, ": predicted ",
          sample.predictions[row], ", expected ", target));
  }
  return absl::OkStatus();
}

absl::Status VerifyGreedyCapture(cuda::Executor& executor, const Layer& model,
                                 const CapturedSample& expected,
                                 const CaptureOptions& options) {
  RETURN_IF_ERROR(ValidateCapturedPredictions(expected, options));
  std::vector<int> prefix(expected.tokens.begin(),
                          expected.tokens.begin() + options.prompt_tokens);
  while (prefix.size() <= expected.tokens.size()) {
    ASSIGN_OR_RETURN(auto actual,
                     CaptureSample(executor, model, prefix, options));
    for (size_t stage = 0; stage < actual.boundaries.size(); ++stage)
      if (!std::equal(actual.boundaries[stage].begin(),
                      actual.boundaries[stage].end(),
                      expected.boundaries[stage].begin()))
        return absl::DataLossError(
            absl::StrCat("prefix BF16 states differ at stage ", stage,
                         " with prefix length ", prefix.size()));
    const int next = actual.predictions.back();
    const int target = prefix.size() == expected.tokens.size()
                           ? options.eos_token
                           : expected.tokens[prefix.size()];
    if (next != target)
      return absl::DataLossError(absl::StrCat(
          "autonomous completion mismatch at length ", prefix.size(),
          ": predicted ", next, ", expected ", target));
    if (next == options.eos_token)
      return absl::OkStatus();
    prefix.push_back(next);
  }
  return absl::DataLossError("autonomous completion did not predict EOS");
}

}  // namespace pluto::llm::memorize_general_facts::discretized_model
