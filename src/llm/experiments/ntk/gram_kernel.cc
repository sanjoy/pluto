#include "src/llm/experiments/ntk/gram_kernel.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <limits>
#include <vector>

#include "absl/status/status.h"
#include "src/cuda/page_locked_host_array.h"
#include "src/util/status_macros.h"

namespace pluto::llm::ntk::internal {
namespace {

constexpr size_t kParametersPerChunk = 4096;
constexpr int kThreads = 256;

// One block reduces one chunk of a lower-triangular Gram entry. Cast BEFORE
// multiplying: FP32 products can overflow even when the final FP64 dot
// product is perfectly representable. Chunks provide parallelism for models
// with millions of parameters even when we select only a few output rows.
__global__ void DotChunks(const float* jacobian, size_t parameters, size_t rows,
                          size_t chunks, double* partials) {
  const size_t i = blockIdx.y;
  const size_t j = blockIdx.z;
  if (j > i)
    return;
  const size_t begin = static_cast<size_t>(blockIdx.x) * kParametersPerChunk;
  double sum = 0;
  for (size_t p = begin + threadIdx.x;
       p < parameters && p < begin + kParametersPerChunk; p += kThreads)
    sum += static_cast<double>(jacobian[i * parameters + p]) *
           static_cast<double>(jacobian[j * parameters + p]);
  __shared__ double shared[kThreads];
  shared[threadIdx.x] = sum;
  __syncthreads();
  for (int stride = kThreads / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride)
      shared[threadIdx.x] += shared[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0)
    partials[(i * rows + j) * chunks + blockIdx.x] = shared[0];
}

// Fixed second-stage reduction. No block reads unused upper-triangular
// scratch, and only one block writes either member of a symmetric pair.
__global__ void FinishGram(const double* partials, size_t rows, size_t chunks,
                           double* gram) {
  const size_t i = blockIdx.x;
  const size_t j = blockIdx.y;
  if (j > i)
    return;
  double sum = 0;
  for (size_t chunk = threadIdx.x; chunk < chunks; chunk += kThreads)
    sum += partials[(i * rows + j) * chunks + chunk];
  __shared__ double shared[kThreads];
  shared[threadIdx.x] = sum;
  __syncthreads();
  for (int stride = kThreads / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride)
      shared[threadIdx.x] += shared[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    gram[i * rows + j] = shared[0];
    gram[j * rows + i] = shared[0];
  }
}

absl::StatusOr<size_t> Product(size_t left, size_t right) {
  if (right != 0 && left > std::numeric_limits<size_t>::max() / right)
    return absl::InvalidArgumentError("NTK Gram tensor size overflows size_t");
  return left * right;
}

}  // namespace

absl::StatusOr<Matrix> ComputeGram(cuda::Executor& executor,
                                   const cuda::Buffer& jacobian, size_t rows,
                                   size_t parameters) {
  if (rows == 0 || parameters == 0)
    return absl::InvalidArgumentError(
        "NTK Gram requires positive row and parameter counts");
  if (rows > 65535)
    return absl::ResourceExhaustedError("NTK Gram rows exceed CUDA grid limit");
  ASSIGN_OR_RETURN(const size_t elements, Product(rows, parameters));
  ASSIGN_OR_RETURN(const size_t jacobian_bytes,
                   Product(elements, sizeof(float)));
  const size_t chunks = (parameters - 1) / kParametersPerChunk + 1;
  if (chunks > static_cast<size_t>(std::numeric_limits<int>::max()))
    return absl::ResourceExhaustedError(
        "NTK Gram reduction exceeds CUDA grid limit");
  if (&jacobian.executor() != &executor ||
      jacobian.size_bytes() != jacobian_bytes)
    return absl::InvalidArgumentError(
        "NTK Jacobian must be a matching FP32 buffer on the supplied executor");

  ASSIGN_OR_RETURN(const size_t entries, Product(rows, rows));
  ASSIGN_OR_RETURN(const size_t partial_count, Product(entries, chunks));
  ASSIGN_OR_RETURN(const size_t partial_bytes,
                   Product(partial_count, sizeof(double)));
  ASSIGN_OR_RETURN(const size_t gram_bytes, Product(entries, sizeof(double)));
  ASSIGN_OR_RETURN(auto partials,
                   cuda::Buffer::Allocate(executor, partial_bytes));
  ASSIGN_OR_RETURN(auto gram, cuda::Buffer::Allocate(executor, gram_bytes));
  DotChunks<<<dim3(static_cast<unsigned int>(chunks),
                   static_cast<unsigned int>(rows),
                   static_cast<unsigned int>(rows)),
              kThreads, 0, executor.stream()>>>(
      static_cast<const float*>(jacobian.data()), parameters, rows, chunks,
      static_cast<double*>(partials.data()));
  RETURN_IF_ERROR(cuda::CudaStatus(cudaGetLastError(), "NTK DotChunks launch"));
  FinishGram<<<dim3(static_cast<unsigned int>(rows),
                    static_cast<unsigned int>(rows)),
               kThreads, 0, executor.stream()>>>(
      static_cast<const double*>(partials.data()), rows, chunks,
      static_cast<double*>(gram.data()));
  RETURN_IF_ERROR(
      cuda::CudaStatus(cudaGetLastError(), "NTK FinishGram launch"));
  ASSIGN_OR_RETURN(auto host_gram, cuda::PageLockedHostArray<double>::Allocate(
                                       executor, entries));
  RETURN_IF_ERROR(cuda::CudaStatus(
      cudaMemcpyAsync(host_gram.data(), gram.data(), gram_bytes,
                      cudaMemcpyDeviceToHost, executor.stream()),
      "read NTK Gram matrix"));
  RETURN_IF_ERROR(executor.Synchronize());
  Matrix result{rows, rows,
                std::vector<double>(host_gram.begin(), host_gram.end())};
  RETURN_IF_ERROR(ValidateMatrix(result));
  return result;
}

}  // namespace pluto::llm::ntk::internal
