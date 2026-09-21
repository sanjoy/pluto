#include "src/llm/gradient_clipper.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/memory/memory.h"
#include "src/cuda/page_locked_host_array.h"
#include "src/util/status_macros.h"

namespace pluto::llm {
namespace {

constexpr int kThreads = 256;
constexpr int kChunkElements = 1024;
constexpr double kNormEpsilon = 1e-6;

// All chunks own disjoint gradient elements. Making this list once lets each
// step use three launches total, rather than two launches per parameter tensor.
struct GradientChunk {
  float* values;
  int elements;
};

__global__ void PartialSquaredNormKernel(const GradientChunk* chunks,
                                         double* squared_norms) {
  __shared__ double partial[kThreads];
  const GradientChunk chunk = chunks[blockIdx.x];
  double sum = 0;
  for (int index = threadIdx.x; index < chunk.elements; index += kThreads) {
    // A double accumulator also avoids overflow when squaring finite FP32
    // gradients. This norm is sensitive metadata, not model computation.
    const double value = chunk.values[index];
    sum += value * value;
  }
  partial[threadIdx.x] = sum;
  __syncthreads();
  for (int stride = kThreads / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride)
      partial[threadIdx.x] += partial[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0)
    squared_norms[blockIdx.x] = partial[0];
}

__global__ void ClippingScaleKernel(const double* squared_norms, int chunks,
                                    float max_norm, float* scale) {
  __shared__ double partial[kThreads];
  double sum = 0;
  for (size_t index = threadIdx.x; index < static_cast<size_t>(chunks);
       index += kThreads)
    sum += squared_norms[index];
  partial[threadIdx.x] = sum;
  __syncthreads();
  for (int stride = kThreads / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride)
      partial[threadIdx.x] += partial[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0)
    scale[0] =
        static_cast<float>(fmin(1.0, static_cast<double>(max_norm) /
                                         (sqrt(partial[0]) + kNormEpsilon)));
}

__global__ void ScaleGradientsKernel(const GradientChunk* chunks,
                                     const float* scale) {
  const float factor = scale[0];
  if (factor == 1.0f)
    return;
  const GradientChunk chunk = chunks[blockIdx.x];
  for (int index = threadIdx.x; index < chunk.elements; index += kThreads)
    chunk.values[index] *= factor;
}

}  // namespace

absl::StatusOr<std::unique_ptr<GradientClipper>> GradientClipper::Create(
    cuda::Executor& executor, Layer& model, float max_norm) {
  if (!(max_norm > 0) || !std::isfinite(max_norm))
    return absl::InvalidArgumentError(
        "gradient clipping norm must be finite and strictly positive");
  const auto weights = model.weights();
  const auto model_gradients = model.gradients();
  if (weights.empty() || weights.size() != model_gradients.size())
    return absl::InvalidArgumentError(
        "gradient clipping requires matching nonempty weights and gradients");

  absl::flat_hash_set<void*> seen;
  std::vector<cuda::Buffer> gradients;
  size_t chunk_count = 0;
  for (size_t index = 0; index < model_gradients.size(); ++index) {
    const auto& weight = weights[index];
    const auto& gradient = model_gradients[index];
    if (&weight.executor() != &executor || &gradient.executor() != &executor ||
        weight.size_bytes() != gradient.size_bytes() ||
        gradient.size_bytes() == 0 ||
        gradient.size_bytes() % sizeof(float) != 0)
      return absl::InvalidArgumentError(
          "gradient clipping needs matching FP32 buffers on its executor");
    if (!seen.insert(gradient.data()).second)
      continue;
    const size_t elements = gradient.size_bytes() / sizeof(float);
    const size_t count = 1 + (elements - 1) / kChunkElements;
    if (count >
        static_cast<size_t>(std::numeric_limits<int>::max()) - chunk_count)
      return absl::InvalidArgumentError("too many gradient clipping chunks");
    chunk_count += count;
    gradients.push_back(gradient);
  }

  ASSIGN_OR_RETURN(
      auto host_chunks,
      cuda::PageLockedHostArray<GradientChunk>::Allocate(executor, chunk_count));
  size_t chunk = 0;
  for (const auto& gradient : gradients) {
    const size_t elements = gradient.size_bytes() / sizeof(float);
    for (size_t offset = 0; offset < elements; offset += kChunkElements)
      host_chunks[chunk++] = {static_cast<float*>(gradient.data()) + offset,
                              static_cast<int>(std::min<size_t>(
                                  kChunkElements, elements - offset))};
  }
  ASSIGN_OR_RETURN(auto chunks,
                   cuda::Buffer::Allocate(executor, host_chunks.size_bytes()));
  ASSIGN_OR_RETURN(
      auto partial_squared_norms,
      cuda::Buffer::Allocate(executor, chunk_count * sizeof(double)));
  ASSIGN_OR_RETURN(auto scale, cuda::Buffer::Allocate(executor, sizeof(float)));
  RETURN_IF_ERROR(cuda::CudaStatus(
      cudaMemcpyAsync(chunks.data(), host_chunks.data(),
                      host_chunks.size_bytes(), cudaMemcpyHostToDevice,
                      executor.stream()),
      "upload gradient clipping chunks"));
  // Page-locked storage is released after this upload on the same stream;
  // neither setup nor Clip() requires a host synchronization.
  return absl::WrapUnique(
      new GradientClipper(executor, max_norm, std::move(gradients),
                          std::move(chunks), std::move(partial_squared_norms),
                          std::move(scale), static_cast<int>(chunk_count)));
}

absl::Status GradientClipper::Clip() {
  const auto* chunks = static_cast<const GradientChunk*>(chunks_.data());
  auto* partial = static_cast<double*>(partial_squared_norms_.data());
  auto* scale = static_cast<float*>(scale_.data());
  PartialSquaredNormKernel<<<chunk_count_, kThreads, 0, executor_.stream()>>>(
      chunks, partial);
  RETURN_IF_ERROR(
      cuda::CudaStatus(cudaGetLastError(), "PartialSquaredNormKernel launch"));
  ClippingScaleKernel<<<1, kThreads, 0, executor_.stream()>>>(
      partial, chunk_count_, max_norm_, scale);
  RETURN_IF_ERROR(
      cuda::CudaStatus(cudaGetLastError(), "ClippingScaleKernel launch"));
  ScaleGradientsKernel<<<chunk_count_, kThreads, 0, executor_.stream()>>>(
      chunks, scale);
  return cuda::CudaStatus(cudaGetLastError(), "ScaleGradientsKernel launch");
}

}  // namespace pluto::llm
