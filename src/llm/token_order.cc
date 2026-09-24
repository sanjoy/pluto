#include "src/llm/token_order.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <utility>
#include <vector>

#include "src/cuda/page_locked_host_array.h"
#include "src/util/status_macros.h"

namespace pluto::llm {

absl::Status ValidateTokenOrder(int vocabulary_size,
                                absl::Span<const int32_t> token_order) {
  if (vocabulary_size <= 0)
    return absl::InvalidArgumentError(
        "token order needs a positive vocabulary");
  if (token_order.empty())
    return absl::OkStatus();
  if (token_order.size() != static_cast<size_t>(vocabulary_size))
    return absl::InvalidArgumentError(
        "token order must contain exactly vocabulary_size entries");
  std::vector<bool> seen(vocabulary_size);
  for (int32_t token : token_order) {
    if (token < 0 || token >= vocabulary_size || seen[token])
      return absl::InvalidArgumentError(
          "token order must be a permutation of [0, vocabulary_size)");
    seen[token] = true;
  }
  return absl::OkStatus();
}

absl::StatusOr<std::optional<cuda::Buffer>> CopyTokenOrderToDevice(
    cuda::Executor& executor, int vocabulary_size,
    absl::Span<const int32_t> token_order) {
  RETURN_IF_ERROR(ValidateTokenOrder(vocabulary_size, token_order));
  bool identity = true;
  for (size_t index = 0; index < token_order.size(); ++index)
    identity &= token_order[index] == static_cast<int32_t>(index);
  if (identity)
    return std::optional<cuda::Buffer>{};
  ASSIGN_OR_RETURN(auto staging, cuda::PageLockedHostArray<int32_t>::CopyFrom(
                                     executor, token_order));
  ASSIGN_OR_RETURN(auto device,
                   cuda::Buffer::Allocate(executor, staging.size_bytes()));
  RETURN_IF_ERROR(cuda::CudaStatus(
      cudaMemcpyAsync(device.data(), staging.data(), staging.size_bytes(),
                      cudaMemcpyHostToDevice, executor.stream()),
      "upload vocabulary token order"));
  return std::optional<cuda::Buffer>(std::move(device));
}

}  // namespace pluto::llm
