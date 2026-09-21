#include "src/llm/gradient_clipper.h"

#include <cuda_runtime.h>
#include <cuda_tile.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
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

constexpr int kChunkElements = 1024;
constexpr int kReductionElements = 256;
constexpr double kNormEpsilon = 1e-6;

// Each metadata row stores [gradient_address_bits, valid_elements]. Keeping
// these as scalar integer columns lets cuTile load them without interpreting a
// C++ struct. The address is a lossless bitcast, not a numeric conversion.
// All chunks own disjoint elements, including after deduplicating tied tensors.
static_assert(sizeof(float*) == sizeof(uint64_t));

__tile__ auto ChunkView(const uint64_t* chunks, int count) {
  namespace ct = ::cuda::tiles;
  using namespace ct::literals;
  return ct::partition_view{ct::tensor_span{chunks, ct::extents{count, 2}},
                            ct::shape{1_ic, 1_ic}};
}

__tile_global__ void PartialSquaredNormKernel(const uint64_t* chunks, int count,
                                              double* squared_norms) {
  namespace ct = ::cuda::tiles;
  using namespace ct::literals;
  const int chunk = ct::bid().x;
  auto metadata = ChunkView(chunks, count);
  const auto* values =
      static_cast<float*>(ct::element_bitcast<float*>(metadata.load(chunk, 0)));
  const int elements = static_cast<int>(metadata.load(chunk, 1));
  auto gradients =
      ct::partition_view{ct::tensor_span{values, ct::extents{elements}},
                         ct::shape<kChunkElements>{}};
  // Widen before squaring: every finite FP32 gradient has a finite FP64
  // square, including values whose squares (or whole norm) overflow FP32.
  // A fixed tile reduction gives one writer per partial, without atomics.
  auto values64 = ct::element_cast<double>(gradients.load_masked(0));
  auto output = ct::partition_view{
      ct::tensor_span{squared_norms, ct::extents{count}}, ct::shape{1_ic}};
  output.store(ct::sum(values64 * values64, 0_ic), chunk);
}

__tile_global__ void ClippingScaleKernel(const double* squared_norms,
                                         int chunks, float max_norm,
                                         float* scale) {
  namespace ct = ::cuda::tiles;
  using namespace ct::literals;
  auto partials =
      ct::partition_view{ct::tensor_span{squared_norms, ct::extents{chunks}},
                         ct::shape<kReductionElements>{}};
  auto sum = ct::zeros<ct::tile<double, ct::shape<kReductionElements>>>();
  // Traversing the partials in a fixed order keeps the whole-model norm
  // deterministic even when the chunk-producing programs finish out of order.
  const int tiles = 1 + (chunks - 1) / kReductionElements;
  for (int tile = 0; tile < tiles; ++tile)
    sum = sum + partials.load_masked(tile);
  auto factor = ct::min(1.0, static_cast<double>(max_norm) /
                                 (ct::sqrt(ct::sum(sum, 0_ic)) + kNormEpsilon));
  auto output = ct::partition_view{ct::tensor_span{scale, ct::extents{1}},
                                   ct::shape{1_ic}};
  output.store(ct::element_cast<float>(factor), 0);
}

__tile_global__ void ScaleGradientsKernel(const uint64_t* chunks, int count,
                                          const float* scale) {
  namespace ct = ::cuda::tiles;
  using namespace ct::literals;
  auto scale_view = ct::partition_view{ct::tensor_span{scale, ct::extents{1}},
                                       ct::shape{1_ic}};
  auto factor = scale_view.load(0);
  if (static_cast<float>(factor) == 1.0f)
    return;
  const int chunk = ct::bid().x;
  auto metadata = ChunkView(chunks, count);
  auto* values =
      static_cast<float*>(ct::element_bitcast<float*>(metadata.load(chunk, 0)));
  const int elements = static_cast<int>(metadata.load(chunk, 1));
  auto gradients =
      ct::partition_view{ct::tensor_span{values, ct::extents{elements}},
                         ct::shape<kChunkElements>{}};
  gradients.store_masked(gradients.load_masked(0) * factor, 0);
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
      cuda::PageLockedHostArray<uint64_t>::Allocate(executor, 2 * chunk_count));
  size_t chunk = 0;
  for (const auto& gradient : gradients) {
    const size_t elements = gradient.size_bytes() / sizeof(float);
    for (size_t offset = 0; offset < elements; offset += kChunkElements) {
      host_chunks[2 * chunk] = reinterpret_cast<uintptr_t>(
          static_cast<float*>(gradient.data()) + offset);
      host_chunks[2 * chunk + 1] =
          std::min<size_t>(kChunkElements, elements - offset);
      ++chunk;
    }
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
  const auto* chunks = static_cast<const uint64_t*>(chunks_.data());
  auto* partial = static_cast<double*>(partial_squared_norms_.data());
  auto* scale = static_cast<float*>(scale_.data());
  PartialSquaredNormKernel<<<chunk_count_, 1, 0, executor_.stream()>>>(
      chunks, chunk_count_, partial);
  RETURN_IF_ERROR(
      cuda::CudaStatus(cudaGetLastError(), "PartialSquaredNormKernel launch"));
  ClippingScaleKernel<<<1, 1, 0, executor_.stream()>>>(partial, chunk_count_,
                                                       max_norm_, scale);
  RETURN_IF_ERROR(
      cuda::CudaStatus(cudaGetLastError(), "ClippingScaleKernel launch"));
  ScaleGradientsKernel<<<chunk_count_, 1, 0, executor_.stream()>>>(
      chunks, chunk_count_, scale);
  return cuda::CudaStatus(cudaGetLastError(), "ScaleGradientsKernel launch");
}

}  // namespace pluto::llm
