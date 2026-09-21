#include "src/llm/generate_greedy_continuation.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <limits>

#include "absl/status/status.h"
#include "src/cuda/buffer.h"
#include "src/llm/extract_top1_ids.h"
#include "src/util/status_macros.h"

namespace pluto::llm {

absl::StatusOr<cuda::PageLockedHostArray<int>> GenerateGreedyContinuation(
    cuda::Executor& executor, const Layer& model,
    absl::Span<const int> prompt_tokens, int vocabulary_size, int eos_token,
    int max_new_tokens, const GreedyGenerationOptions& options) {
  if (vocabulary_size <= 0 || eos_token < 0 || eos_token >= vocabulary_size)
    return absl::InvalidArgumentError("invalid generation vocabulary or EOS");
  if (max_new_tokens < 0)
    return absl::InvalidArgumentError("max_new_tokens must be nonnegative");
  if (prompt_tokens.empty())
    return absl::InvalidArgumentError("generation prompt must not be empty");
  for (int token : prompt_tokens)
    if (token < 0 || token >= vocabulary_size)
      return absl::InvalidArgumentError("prompt token is outside vocabulary");

  const auto inputs = model.input_types();
  const auto outputs = model.output_types();
  if (inputs.size() != 1 || outputs.size() != 1)
    return absl::InvalidArgumentError(
        "generation requires one model input and one output");
  RETURN_IF_ERROR(inputs[0].Validate());
  RETURN_IF_ERROR(outputs[0].Validate());
  const auto input_shape = inputs[0].dimensions();
  const auto output_shape = outputs[0].dimensions();
  if (inputs[0].data_type() != DataType::INT32 || input_shape.size() != 2 ||
      input_shape[0] != ActivationType::kBatchDimension ||
      input_shape[1] > std::numeric_limits<int>::max())
    return absl::InvalidArgumentError(
        "generation input must be INT32 [batch, context]");
  if (outputs[0].data_type() != DataType::FP32 || output_shape.size() != 3 ||
      output_shape[0] != ActivationType::kBatchDimension ||
      output_shape[1] != input_shape[1] || output_shape[2] < vocabulary_size ||
      output_shape[2] > std::numeric_limits<int>::max())
    return absl::InvalidArgumentError(
        "generation output must be FP32 [batch, context, padded_vocabulary]");
  const size_t context_length = static_cast<size_t>(input_shape[1]);
  const size_t logit_stride = static_cast<size_t>(output_shape[2]);
  if (context_length >
      std::numeric_limits<size_t>::max() / sizeof(float) / logit_stride)
    return absl::InvalidArgumentError("generation logits byte size overflows");
  const size_t logits_bytes = context_length * logit_stride * sizeof(float);
  if (prompt_tokens.size() > context_length)
    return absl::InvalidArgumentError("generation prompt exceeds context");
  for (const auto& weight : model.weights())
    if (&weight.executor() != &executor)
      return absl::InvalidArgumentError(
          "generation model weights belong to another executor");

  const size_t generation_limit =
      std::min(static_cast<size_t>(max_new_tokens),
               context_length - prompt_tokens.size());
  if (generation_limit == 0)
    return cuda::PageLockedHostArray<int>::Allocate(executor, 0);

  ASSIGN_OR_RETURN(auto context, cuda::PageLockedHostArray<int>::Allocate(
                                     executor, context_length));
  ASSIGN_OR_RETURN(auto targets, cuda::PageLockedHostArray<int>::Allocate(
                                     executor, context_length));
  ASSIGN_OR_RETURN(auto selected,
                   cuda::PageLockedHostArray<int>::Allocate(executor, 1));
  std::fill(context.begin(), context.end(), eos_token);
  std::copy(prompt_tokens.begin(), prompt_tokens.end(), context.begin());
  std::fill(targets.begin(), targets.end(), -1);
  ASSIGN_OR_RETURN(auto device_context,
                   cuda::Buffer::Allocate(executor, context.size_bytes()));
  ASSIGN_OR_RETURN(auto device_targets,
                   cuda::Buffer::Allocate(executor, targets.size_bytes()));

  size_t used = prompt_tokens.size();
  for (size_t generated = 0; generated < generation_limit; ++generated) {
    const size_t final_row = used - 1;
    if (generated != 0)
      targets[final_row - 1] = -1;
    targets[final_row] = 0;
    // Both staging arrays are reused only after the preceding iteration's
    // synchronization. The fixed input shape retains absolute positions.
    RETURN_IF_ERROR(cuda::CudaStatus(
        cudaMemcpyAsync(device_context.data(), context.data(),
                        context.size_bytes(), cudaMemcpyHostToDevice,
                        executor.stream()),
        "upload generation context"));
    RETURN_IF_ERROR(cuda::CudaStatus(
        cudaMemcpyAsync(device_targets.data(), targets.data(),
                        targets.size_bytes(), cudaMemcpyHostToDevice,
                        executor.stream()),
        "upload generation row mask"));
    ASSIGN_OR_RETURN(auto forward, model.fwd(executor, {&device_context, 1},
                                             options.layer_hooks));
    // Inference never calls backward. Release all saved intermediates on the
    // executor stream before the next full-context forward pass.
    forward.state = BackwardState{};
    if (forward.outputs.size() != 1 ||
        &forward.outputs[0].executor() != &executor ||
        forward.outputs[0].size_bytes() != logits_bytes)
      return absl::InvalidArgumentError(
          "generation logits do not match the declared shape and executor");
    ASSIGN_OR_RETURN(auto ids, ExtractTop1Ids(executor, forward.outputs[0],
                                              device_targets, vocabulary_size));
    RETURN_IF_ERROR(cuda::CudaStatus(
        cudaMemcpyAsync(selected.data(),
                        static_cast<const int*>(ids.data()) + final_row,
                        sizeof(int), cudaMemcpyDeviceToHost, executor.stream()),
        "download generated token"));
    RETURN_IF_ERROR(executor.Synchronize());
    const int token = selected[0];
    if (token == -2)
      return absl::DataLossError(
          "generation encountered nonfinite logical vocabulary logits");
    if (token < 0 || token >= vocabulary_size)
      return absl::DataLossError("generation produced an invalid token ID");
    if (token == eos_token)
      break;
    if (options.on_token)
      RETURN_IF_ERROR(
          options.on_token(executor, context.span().first(used), token));
    context[used++] = token;
  }
  return cuda::PageLockedHostArray<int>::CopyFrom(
      executor, context.span().subspan(prompt_tokens.size(),
                                       used - prompt_tokens.size()));
}

}  // namespace pluto::llm
