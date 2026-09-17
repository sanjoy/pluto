#include "src/llm/experiments/path_kernel/update_kernel.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>

#include "absl/status/status.h"
#include "src/cuda/page_locked_host_array.h"
#include "src/util/status_macros.h"

namespace pluto::llm::path_kernel::internal {
namespace {

__global__ void Update(float* weights, size_t count, const float* jacobian,
                       size_t parameters, size_t offset, size_t first_row,
                       size_t rows, double scale, int* invalid) {
  for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < count; i += static_cast<size_t>(gridDim.x) * blockDim.x) {
    double gradient = 0;
    for (size_t row = 0; row < rows; ++row)
      gradient += jacobian[(first_row + row) * parameters + offset + i];
    const float value =
        static_cast<float>(static_cast<double>(weights[i]) - scale * gradient);
    if (!isfinite(value))
      atomicExch(invalid, 1);  // Error flag only, never a numeric reduction.
    weights[i] = value;
  }
}

}  // namespace

absl::Status ApplyGradientDescent(cuda::Executor& executor,
                                  const cuda::Buffer& weights,
                                  const cuda::Buffer& jacobian,
                                  size_t parameter_count, size_t block_offset,
                                  size_t first_loss_row, size_t loss_rows,
                                  double learning_rate) {
  if (&weights.executor() != &executor || &jacobian.executor() != &executor ||
      weights.size_bytes() == 0 || weights.size_bytes() % sizeof(float) != 0 ||
      jacobian.size_bytes() % sizeof(float) != 0 || parameter_count == 0 ||
      loss_rows == 0 || !std::isfinite(learning_rate) || learning_rate <= 0)
    return absl::InvalidArgumentError(
        "invalid path-kernel GD buffers/settings");
  const size_t count = weights.size_bytes() / sizeof(float);
  const size_t elements = jacobian.size_bytes() / sizeof(float);
  if (block_offset > parameter_count ||
      count > parameter_count - block_offset ||
      elements % parameter_count != 0 ||
      first_loss_row > elements / parameter_count ||
      loss_rows != elements / parameter_count - first_loss_row)
    return absl::InvalidArgumentError("path-kernel GD Jacobian shape mismatch");
  ASSIGN_OR_RETURN(auto invalid, cuda::Buffer::Allocate(executor, sizeof(int)));
  RETURN_IF_ERROR(cuda::CudaStatus(
      cudaMemsetAsync(invalid.data(), 0, sizeof(int), executor.stream()),
      "clear path-kernel GD error flag"));
  const size_t blocks = (count - 1) / 256 + 1;
  Update<<<static_cast<unsigned>(blocks > 65535 ? 65535 : blocks), 256, 0,
           executor.stream()>>>(
      static_cast<float*>(weights.data()), count,
      static_cast<const float*>(jacobian.data()), parameter_count, block_offset,
      first_loss_row, loss_rows, learning_rate / static_cast<double>(loss_rows),
      static_cast<int*>(invalid.data()));
  RETURN_IF_ERROR(
      cuda::CudaStatus(cudaGetLastError(), "path-kernel GD launch"));
  ASSIGN_OR_RETURN(auto flag,
                   cuda::PageLockedHostArray<int>::Allocate(executor, 1));
  RETURN_IF_ERROR(cuda::CudaStatus(
      cudaMemcpyAsync(flag.data(), invalid.data(), sizeof(int),
                      cudaMemcpyDeviceToHost, executor.stream()),
      "read path-kernel GD error flag"));
  RETURN_IF_ERROR(executor.Synchronize());
  if (flag[0] != 0)
    return absl::OutOfRangeError("path-kernel GD produced a nonfinite weight");
  return absl::OkStatus();
}

}  // namespace pluto::llm::path_kernel::internal
