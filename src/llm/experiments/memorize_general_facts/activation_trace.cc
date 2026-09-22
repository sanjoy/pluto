#include "src/llm/experiments/memorize_general_facts/activation_trace.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "src/cuda/page_locked_host_array.h"
#include "src/llm/layer_hooks.h"
#include "src/util/status_macros.h"

namespace pluto::llm::memorize_general_facts {
namespace {

struct DeviceBoundary {
  std::string name;
  DataType type;
  Buffer activation;
};

// BF16 has the same exponent as FP32 and supplies its upper 16 bits. Expanding
// the representation, rather than a numeric integer conversion, preserves the
// raw physical activation exactly without depending on CUDA host BF16 support.
float DecodeBfloat16(const uint8_t* bytes) {
  uint16_t bf16;
  std::memcpy(&bf16, bytes, sizeof(bf16));
  const uint32_t bits = static_cast<uint32_t>(bf16) << 16;
  float result;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

}  // namespace

absl::StatusOr<std::vector<ActivationTraceBoundary>>
CaptureActivationBoundaries(cuda::Executor& executor, const Layer& model,
                            absl::Span<const int> complete_tokens,
                            int vocabulary_size, int eos_token,
                            int transformer_blocks, int model_width) {
  if (model_width != 16)
    return absl::InvalidArgumentError(
        "activation tracing requires model_width=16");
  if (transformer_blocks < 0)
    return absl::InvalidArgumentError(
        "activation tracing requires a nonnegative transformer block count");
  if (complete_tokens.empty() || vocabulary_size <= 0 || eos_token < 0 ||
      eos_token >= vocabulary_size ||
      std::any_of(complete_tokens.begin(), complete_tokens.end(),
                  [vocabulary_size](int token) {
                    return token < 0 || token >= vocabulary_size;
                  }))
    return absl::InvalidArgumentError(
        "activation tracing requires a nonempty token sequence and IDs/EOS "
        "within the model vocabulary");
  const auto input_types = model.input_types();
  if (input_types.size() != 1)
    return absl::InvalidArgumentError(
        "activation tracing requires one token input");
  RETURN_IF_ERROR(input_types[0].Validate());
  const auto dims = input_types[0].dimensions();
  if (input_types[0].data_type() != DataType::INT32 || dims.size() != 2 ||
      dims[0] != ActivationType::kBatchDimension)
    return absl::InvalidArgumentError(
        "activation tracing requires an INT32 [-2, context] input");
  const uint64_t context = dims[1];
  if (context >
      std::numeric_limits<size_t>::max() / model_width / sizeof(float))
    return absl::InvalidArgumentError(
        "activation tracing context size overflows");
  if (complete_tokens.size() > context)
    return absl::InvalidArgumentError(
        "activation tracing tokens exceed the model context");

  const auto output_types = model.output_types();
  if (output_types.size() != 1 ||
      output_types[0].data_type() != DataType::FP32 ||
      output_types[0].dimensions().size() != 3 ||
      output_types[0].dimensions()[0] != ActivationType::kBatchDimension ||
      output_types[0].dimensions()[1] != static_cast<int64_t>(context) ||
      output_types[0].dimensions()[2] < vocabulary_size)
    return absl::InvalidArgumentError(
        "activation tracing requires compatible FP32 model logits");

  ASSIGN_OR_RETURN(
      auto host_tokens,
      cuda::PageLockedHostArray<int32_t>::Allocate(executor, context));
  std::fill(host_tokens.begin(), host_tokens.end(), eos_token);
  std::copy(complete_tokens.begin(), complete_tokens.end(),
            host_tokens.begin());
  ASSIGN_OR_RETURN(auto input,
                   Buffer::Allocate(executor, host_tokens.size_bytes()));
  RETURN_IF_ERROR(cuda::CudaStatus(
      cudaMemcpyAsync(input.data(), host_tokens.data(),
                      host_tokens.size_bytes(), cudaMemcpyHostToDevice,
                      executor.stream()),
      "cudaMemcpyAsync(activation trace input)"));

  std::vector<std::string> scopes;
  std::vector<DeviceBoundary> captured;
  LayerHooks hooks;
  const auto validate_executor = [&](cuda::Executor& hook_executor) {
    return &hook_executor == &executor
               ? absl::OkStatus()
               : absl::InvalidArgumentError(
                     "activation tracing received another executor");
  };
  hooks.enter_combinator = [&](cuda::Executor& hook_executor,
                               absl::string_view name) -> absl::Status {
    RETURN_IF_ERROR(validate_executor(hook_executor));
    scopes.emplace_back(name);
    return absl::OkStatus();
  };
  hooks.exit_combinator = [&](cuda::Executor& hook_executor) -> absl::Status {
    RETURN_IF_ERROR(validate_executor(hook_executor));
    if (scopes.empty())
      return absl::FailedPreconditionError(
          "activation tracing has an unbalanced combinator scope");
    scopes.pop_back();
    return absl::OkStatus();
  };
  hooks.activation_hook = [&](cuda::Executor& hook_executor,
                              absl::string_view name,
                              absl::Span<const ActivationType> types,
                              absl::Span<Buffer> outputs) -> absl::Status {
    RETURN_IF_ERROR(validate_executor(hook_executor));
    // A composed layer's activation callback follows its exit callback.
    // Thus these boundaries are direct children of the outer model scope.
    // Ignoring deeper scopes avoids tapping residual branches or MLP outputs
    // that happen to have the same width (or even a reused diagnostic name).
    if (scopes.size() != 1 || scopes[0] != model.name())
      return absl::OkStatus();
    if (name != "PositionEmbeddingLayer" &&
        !absl::StartsWith(name, "transformer_block_"))
      return absl::OkStatus();
    const std::string expected =
        captured.empty()
            ? "PositionEmbeddingLayer"
            : absl::StrCat("transformer_block_", captured.size() - 1);
    if (name != expected ||
        captured.size() > static_cast<size_t>(transformer_blocks))
      return absl::InvalidArgumentError(absl::StrCat(
          "activation tracing expected boundary ", expected, ", got ", name));
    if (types.size() != 1 || outputs.size() != 1)
      return absl::InvalidArgumentError(
          "activation tracing requires one residual-stream output");
    RETURN_IF_ERROR(types[0].Validate());
    const auto shape = types[0].dimensions();
    if ((types[0].data_type() != DataType::FP32 &&
         types[0].data_type() != DataType::BF16) ||
        shape.size() != 3 || shape[0] != ActivationType::kBatchDimension ||
        shape[1] != static_cast<int64_t>(context) || shape[2] != model_width)
      return absl::InvalidArgumentError(
          "activation tracing expects FP32/BF16 [-2, context, 16] boundaries");
    const size_t element_bytes = types[0].data_type() == DataType::BF16
                                     ? sizeof(uint16_t)
                                     : sizeof(float);
    if (&outputs[0].executor() != &executor ||
        outputs[0].size_bytes() != context * model_width * element_bytes)
      return absl::InvalidArgumentError(
          "activation tracing boundary has an invalid executor or byte size");
    // Keep the device allocation alive until the copies below are queued.
    // No callback modifies the model output or its backward state.
    captured.push_back({std::string(name), types[0].data_type(), outputs[0]});
    return absl::OkStatus();
  };
  {
    ASSIGN_OR_RETURN(auto result, model.fwd(executor, {&input, 1}, &hooks));
    (void)result;
    // Only captured residual buffers need to survive the replay. Release the
    // logits and the otherwise-unused backward state before downloading them.
  }
  if (!scopes.empty() ||
      captured.size() != static_cast<size_t>(transformer_blocks) + 1)
    return absl::InvalidArgumentError(
        "activation tracing did not find the expected GPT-2 boundaries");

  const size_t values_per_boundary = complete_tokens.size() * model_width;
  std::vector<cuda::PageLockedHostArray<uint8_t>> host_boundaries;
  for (const auto& boundary : captured) {
    const size_t element_bytes =
        boundary.type == DataType::BF16 ? sizeof(uint16_t) : sizeof(float);
    ASSIGN_OR_RETURN(auto host,
                     cuda::PageLockedHostArray<uint8_t>::Allocate(
                         executor, values_per_boundary * element_bytes));
    // Only the leading real rows are transferred, never EOS-filled padding.
    RETURN_IF_ERROR(cuda::CudaStatus(
        cudaMemcpyAsync(host.data(), boundary.activation.data(),
                        host.size_bytes(), cudaMemcpyDeviceToHost,
                        executor.stream()),
        "cudaMemcpyAsync(activation trace boundary)"));
    host_boundaries.push_back(std::move(host));
  }
  RETURN_IF_ERROR(executor.Synchronize());

  std::vector<ActivationTraceBoundary> boundaries;
  for (size_t i = 0; i < captured.size(); ++i) {
    ActivationTraceBoundary boundary{captured[i].name,
                                     std::vector<float>(values_per_boundary)};
    for (size_t j = 0; j < values_per_boundary; ++j) {
      float value;
      if (captured[i].type == DataType::BF16)
        value =
            DecodeBfloat16(host_boundaries[i].data() + j * sizeof(uint16_t));
      else
        std::memcpy(&value, host_boundaries[i].data() + j * sizeof(float),
                    sizeof(value));
      if (!std::isfinite(value))
        return absl::DataLossError(absl::StrCat(
            "activation tracing found a non-finite value at ", boundary.name,
            ", row ", j / model_width, ", dimension ", j % model_width));
      boundary.values[j] = value;
    }
    boundaries.push_back(std::move(boundary));
  }
  return boundaries;
}

}  // namespace pluto::llm::memorize_general_facts
