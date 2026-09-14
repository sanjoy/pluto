#include "src/llm/layers/gelu.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda_tile.h>

#include <cstddef>
#include <memory>
#include <utility>

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "src/llm/layers/util.h"
#include "src/util/status_macros.h"

namespace pluto::llm {
namespace {

template <class Activation>
__tile_global__ void GeluForwardKernel(const Activation* __restrict__ input,
                                       int elements,
                                       Activation* __restrict__ output) {
  namespace ct = ::cuda::tiles;
  using namespace ct::literals;
  auto input_view = ct::partition_view{
      ct::tensor_span{input, ct::extents{elements}}, ct::shape{16_ic}};
  auto output_view = ct::partition_view{
      ct::tensor_span{output, ct::extents{elements}}, ct::shape{16_ic}};
  const int block = ct::bid().x;
  auto x = ct::element_cast<float>(input_view.load(block));
  constexpr float kSqrtTwoOverPi = 0.7978845608f;
  auto inner = kSqrtTwoOverPi * (x + 0.044715f * x * x * x);
  auto result = 0.5f * x * (1.0f + ct::tanh(inner));
  output_view.store(ct::element_cast<Activation>(result), block);
}

template <class Activation>
__tile_global__ void GeluBackwardKernel(
    const Activation* __restrict__ input,
    const float* __restrict__ output_gradient, int elements,
    float* __restrict__ input_gradient) {
  namespace ct = ::cuda::tiles;
  using namespace ct::literals;
  auto input_view = ct::partition_view{
      ct::tensor_span{input, ct::extents{elements}}, ct::shape{16_ic}};
  auto gradient_view = ct::partition_view{
      ct::tensor_span{output_gradient, ct::extents{elements}},
      ct::shape{16_ic}};
  auto input_gradient_view = ct::partition_view{
      ct::tensor_span{input_gradient, ct::extents{elements}}, ct::shape{16_ic}};
  const int block = ct::bid().x;
  auto x = ct::element_cast<float>(input_view.load(block));
  constexpr float kSqrtTwoOverPi = 0.7978845608f;
  auto inner = kSqrtTwoOverPi * (x + 0.044715f * x * x * x);
  auto tanh_inner = ct::tanh(inner);
  auto derivative = 0.5f * (1.0f + tanh_inner) +
                    0.5f * x * (1.0f - tanh_inner * tanh_inner) *
                        kSqrtTwoOverPi * (1.0f + 3.0f * 0.044715f * x * x);
  input_gradient_view.store(gradient_view.load(block) * derivative, block);
}

}  // namespace

absl::StatusOr<std::unique_ptr<GeluLayer>> GeluLayer::Create(
    cuda::Executor& executor, int embedding_dim, DataType data_type,
    int sequence_length) {
  RETURN_IF_ERROR(internal::ValidateComputeType(data_type));
  if (sequence_length <= 0)
    return absl::InvalidArgumentError("sequence_length must be positive");
  // Elementwise tiles cross row boundaries. Only the total element count,
  // checked by fwd(), needs to be a multiple of the tile width.
  if (embedding_dim <= 0)
    return absl::InvalidArgumentError("embedding_dim must be positive");
  return absl::WrapUnique(
      new GeluLayer(executor, embedding_dim, data_type, sequence_length));
}

absl::StatusOr<FwdResult> GeluLayer::fwd_impl(
    cuda::Executor& executor, absl::Span<const Buffer> inputs) const {
  BackwardState state;
  RETURN_IF_ERROR(internal::ValidateExecutor(executor_, executor, "GeluLayer"));
  if (inputs.size() != 1)
    return absl::InvalidArgumentError("GeluLayer fwd expects one input");
  ASSIGN_OR_RETURN(
      int elements,
      internal::ElementCount(executor, inputs[0],
                             internal::ActivationElementBytes(output_type_),
                             "GELU input"));
  RETURN_IF_ERROR(
      internal::ValidateTiledExtent(elements, "GELU element count"));
  ASSIGN_OR_RETURN(auto output,
                   Buffer::Allocate(executor, inputs[0].size_bytes()));
  state.intermediates = {inputs[0]};
  state.children.clear();
  if (output_type_ == DataType::BF16) {
    GeluForwardKernel<__nv_bfloat16>
        <<<internal::TileCount(elements), 1, 0, executor.stream()>>>(
            static_cast<const __nv_bfloat16*>(inputs[0].data()), elements,
            static_cast<__nv_bfloat16*>(output.data()));
  } else {
    GeluForwardKernel<float>
        <<<internal::TileCount(elements), 1, 0, executor.stream()>>>(
            static_cast<const float*>(inputs[0].data()), elements,
            static_cast<float*>(output.data()));
  }
  RETURN_IF_ERROR(
      cuda::CudaStatus(cudaGetLastError(), "GeluForwardKernel launch"));
  return FwdResult{{std::move(output)}, std::move(state)};
}

absl::StatusOr<BufferVec> GeluLayer::bwd_impl(
    cuda::Executor& executor, absl::Span<const Buffer> output_gradients,
    BackwardState state) {
  RETURN_IF_ERROR(internal::ValidateExecutor(executor_, executor, "GeluLayer"));
  if (output_gradients.size() != 1 || state.intermediates.size() != 1) {
    return absl::InvalidArgumentError(
        "GeluLayer bwd received an incompatible gradient or state");
  }
  ASSIGN_OR_RETURN(int elements, internal::ElementCount(
                                     executor, output_gradients[0],
                                     sizeof(float), "GELU output gradient"));
  RETURN_IF_ERROR(internal::ValidateBuffer(
      executor, state.intermediates[0],
      static_cast<size_t>(elements) *
          internal::ActivationElementBytes(output_type_),
      "GELU saved input"));
  ASSIGN_OR_RETURN(
      auto input_gradient,
      Buffer::Allocate(executor, static_cast<size_t>(elements) * sizeof(float)));
  if (output_type_ == DataType::BF16) {
    GeluBackwardKernel<__nv_bfloat16>
        <<<internal::TileCount(elements), 1, 0, executor.stream()>>>(
            static_cast<const __nv_bfloat16*>(state.intermediates[0].data()),
            static_cast<const float*>(output_gradients[0].data()), elements,
            static_cast<float*>(input_gradient.data()));
  } else {
    GeluBackwardKernel<float>
        <<<internal::TileCount(elements), 1, 0, executor.stream()>>>(
            static_cast<const float*>(state.intermediates[0].data()),
            static_cast<const float*>(output_gradients[0].data()), elements,
            static_cast<float*>(input_gradient.data()));
  }
  RETURN_IF_ERROR(
      cuda::CudaStatus(cudaGetLastError(), "GeluBackwardKernel launch"));
  return BufferVec{std::move(input_gradient)};
}

}  // namespace pluto::llm
