#include "src/llm/layers/inference.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda_tile.h>

#include <cmath>
#include <cstdint>
#include <utility>

#include "absl/memory/memory.h"
#include "src/util/status_macros.h"

namespace pluto::llm {
namespace {

using inference_internal::kMaximumDimension;

absl::Status CheckBuffer(cuda::Executor& executor, const Buffer& buffer,
                         size_t bytes) {
  if (&buffer.executor() != &executor)
    return absl::InvalidArgumentError(
        "inference buffer belongs to another executor");
  if (buffer.size_bytes() != bytes)
    return absl::InvalidArgumentError(
        "inference buffer has incorrect byte size");
  return absl::OkStatus();
}

absl::Status CheckExecutor(cuda::Executor& expected, cuda::Executor& actual) {
  if (&expected != &actual)
    return absl::InvalidArgumentError(
        "inference layer belongs to another executor");
  return absl::OkStatus();
}

absl::Status CheckDimension(int n) {
  if (n <= 0 || n > kMaximumDimension)
    return absl::InvalidArgumentError(
        "inference dimension must be in [1, 1048576]");
  return absl::OkStatus();
}

absl::StatusOr<size_t> ElementBytes(inference_ops::MatrixStorage storage) {
  switch (storage) {
    case inference_ops::MatrixStorage::kFloat32:
      return sizeof(float);
    case inference_ops::MatrixStorage::kBFloat16:
      return sizeof(__nv_bfloat16);
    case inference_ops::MatrixStorage::kFp8E4M3:
      return 1;
  }
  return absl::InvalidArgumentError("unknown inference matrix storage");
}

// The imported operators accumulate in FP32. Converting at each public layer
// boundary lets ordinary ResidualLayer supply exactly BF16 residual rounding.
template <class In, class Out>
__tile_global__ void CastKernel(const In* input, int n, Out* output) {
  namespace ct = ::cuda::tiles;
  using namespace ct::literals;
  auto x = ct::partition_view{ct::tensor_span{input, ct::extents{n}},
                              ct::shape{256_ic}};
  auto y = ct::partition_view{ct::tensor_span{output, ct::extents{n}},
                              ct::shape{256_ic}};
  y.store_masked(ct::element_cast<Out>(x.load_masked(ct::bid().x)),
                 ct::bid().x);
}

template <class In, class Out>
absl::StatusOr<Buffer> Cast(cuda::Executor& executor, const Buffer& input,
                            int n) {
  RETURN_IF_ERROR(CheckDimension(n));
  RETURN_IF_ERROR(CheckBuffer(executor, input, size_t(n) * sizeof(In)));
  ASSIGN_OR_RETURN(Buffer output,
                   Buffer::Allocate(executor, size_t(n) * sizeof(Out)));
  CastKernel<<<1 + (n - 1) / 256, 1, 0, executor.stream()>>>(
      static_cast<const In*>(input.data()), n,
      static_cast<Out*>(output.data()));
  RETURN_IF_ERROR(cuda::CudaStatus(cudaGetLastError(), "inference CastKernel"));
  return output;
}

absl::Status UnsupportedBackward() {
  return absl::UnimplementedError(
      "imported inference layers do not support backward");
}

float* Floats(const Buffer& buffer) {
  return static_cast<float*>(buffer.data());
}

}  // namespace

namespace inference_internal {

absl::StatusOr<Buffer> AllocateFloatVector(cuda::Executor& executor,
                                           int elements) {
  RETURN_IF_ERROR(CheckDimension(elements));
  return Buffer::Allocate(executor, size_t(elements) * sizeof(float));
}

absl::StatusOr<Buffer> ToFloat(cuda::Executor& executor, const Buffer& input,
                               int elements) {
  return Cast<__nv_bfloat16, float>(executor, input, elements);
}

absl::StatusOr<Buffer> ToBFloat16(cuda::Executor& executor, const Buffer& input,
                                  int elements) {
  return Cast<float, __nv_bfloat16>(executor, input, elements);
}

absl::Status ValidateInputs(cuda::Executor& executor,
                            absl::Span<const Buffer> inputs,
                            absl::Span<const int> element_counts) {
  if (inputs.size() != element_counts.size())
    return absl::InvalidArgumentError(
        "inference layer received incorrect input count");
  for (size_t i = 0; i < inputs.size(); ++i) {
    RETURN_IF_ERROR(CheckDimension(element_counts[i]));
    RETURN_IF_ERROR(
        CheckBuffer(executor, inputs[i],
                    size_t(element_counts[i]) * sizeof(__nv_bfloat16)));
  }
  return absl::OkStatus();
}

}  // namespace inference_internal

absl::StatusOr<std::unique_ptr<InferenceLinearLayer>>
InferenceLinearLayer::Create(cuda::Executor& executor, Buffer weights,
                             MatrixStorage storage, int input_dim,
                             int output_dim, std::optional<Buffer> scales) {
  RETURN_IF_ERROR(CheckDimension(input_dim));
  RETURN_IF_ERROR(CheckDimension(output_dim));
  ASSIGN_OR_RETURN(size_t bytes, ElementBytes(storage));
  RETURN_IF_ERROR(
      CheckBuffer(executor, weights, size_t(input_dim) * output_dim * bytes));
  if ((storage == MatrixStorage::kFp8E4M3) != scales.has_value())
    return absl::InvalidArgumentError(
        "only FP8 linear weights require block scales");
  BufferVec buffers{std::move(weights)};
  if (scales) {
    RETURN_IF_ERROR(CheckBuffer(executor, *scales,
                                size_t((input_dim + 127) / 128) *
                                    ((output_dim + 127) / 128) *
                                    sizeof(float)));
    buffers.push_back(std::move(*scales));
  }
  return absl::WrapUnique(new InferenceLinearLayer(
      executor, std::move(buffers), storage, input_dim, output_dim));
}

absl::StatusOr<FwdResult> InferenceLinearLayer::fwd_impl(
    cuda::Executor& executor, absl::Span<const Buffer> inputs,
    LayerHooks*) const {
  RETURN_IF_ERROR(CheckExecutor(executor_, executor));
  RETURN_IF_ERROR(
      inference_internal::ValidateInputs(executor, inputs, {input_dim_}));
  ASSIGN_OR_RETURN(Buffer input,
                   inference_internal::ToFloat(executor, inputs[0], input_dim_));
  if (storage_ == MatrixStorage::kFp8E4M3) {
    ASSIGN_OR_RETURN(Buffer quantized, inference_internal::AllocateFloatVector(
                                           executor, input_dim_));
    RETURN_IF_ERROR(inference_ops::QuantizeFp8Input(
        executor, Floats(input), input_dim_, Floats(quantized)));
    input = std::move(quantized);
  }
  ASSIGN_OR_RETURN(Buffer result, inference_internal::AllocateFloatVector(
                                      executor, output_dim_));
  const float* scales = weights_.size() == 2 ? Floats(weights_[1]) : nullptr;
  RETURN_IF_ERROR(inference_ops::MatVec(executor, weights_[0].data(), storage_,
                                        scales, Floats(input), input_dim_,
                                        output_dim_, Floats(result)));
  ASSIGN_OR_RETURN(Buffer output, inference_internal::ToBFloat16(
                                      executor, result, output_dim_));
  return FwdResult{{std::move(output)}, {}};
}

absl::StatusOr<BufferVec> InferenceLinearLayer::bwd_impl(
    cuda::Executor&, absl::Span<const Buffer>, BackwardState, LayerHooks*) {
  return UnsupportedBackward();
}

absl::StatusOr<std::unique_ptr<InferenceEmbeddingLayer>>
InferenceEmbeddingLayer::Create(cuda::Executor& executor, Buffer weights,
                                MatrixStorage storage, int vocab_size,
                                int embedding_dim) {
  RETURN_IF_ERROR(CheckDimension(vocab_size));
  RETURN_IF_ERROR(CheckDimension(embedding_dim));
  ASSIGN_OR_RETURN(size_t bytes, ElementBytes(storage));
  if (storage == MatrixStorage::kFp8E4M3)
    return absl::InvalidArgumentError(
        "inference embedding requires BF16 or FP32 weights");
  RETURN_IF_ERROR(CheckBuffer(executor, weights,
                              size_t(vocab_size) * embedding_dim * bytes));
  return absl::WrapUnique(new InferenceEmbeddingLayer(
      executor, std::move(weights), storage, vocab_size, embedding_dim));
}

absl::StatusOr<FwdResult> InferenceEmbeddingLayer::fwd_impl(
    cuda::Executor& executor, absl::Span<const Buffer> inputs,
    LayerHooks*) const {
  RETURN_IF_ERROR(CheckExecutor(executor_, executor));
  if (inputs.size() != 1)
    return absl::InvalidArgumentError("inference embedding requires one input");
  RETURN_IF_ERROR(CheckBuffer(executor, inputs[0], sizeof(int32_t)));
  ASSIGN_OR_RETURN(Buffer result, inference_internal::AllocateFloatVector(
                                      executor, embedding_dim_));
  RETURN_IF_ERROR(inference_ops::EmbeddingLookupDevice(
      executor, weight_.data(), storage_,
      static_cast<const int32_t*>(inputs[0].data()), vocab_size_,
      embedding_dim_, Floats(result)));
  ASSIGN_OR_RETURN(Buffer output, inference_internal::ToBFloat16(
                                      executor, result, embedding_dim_));
  return FwdResult{{std::move(output)}, {}};
}

absl::StatusOr<BufferVec> InferenceEmbeddingLayer::bwd_impl(
    cuda::Executor&, absl::Span<const Buffer>, BackwardState, LayerHooks*) {
  return UnsupportedBackward();
}

absl::StatusOr<std::unique_ptr<RmsNormLayer>> RmsNormLayer::Create(
    cuda::Executor& executor, int width, Buffer weight, float epsilon) {
  if (width <= 0 || width > 16384 || !std::isfinite(epsilon) || epsilon <= 0)
    return absl::InvalidArgumentError(
        "invalid inference RMSNorm dimensions or epsilon");
  RETURN_IF_ERROR(CheckBuffer(executor, weight, size_t(width) * sizeof(float)));
  return absl::WrapUnique(
      new RmsNormLayer(executor, width, std::move(weight), epsilon));
}

absl::StatusOr<FwdResult> RmsNormLayer::fwd_impl(
    cuda::Executor& executor, absl::Span<const Buffer> inputs,
    LayerHooks*) const {
  RETURN_IF_ERROR(CheckExecutor(executor_, executor));
  RETURN_IF_ERROR(
      inference_internal::ValidateInputs(executor, inputs, {width_}));
  ASSIGN_OR_RETURN(Buffer input,
                   inference_internal::ToFloat(executor, inputs[0], width_));
  ASSIGN_OR_RETURN(Buffer result,
                   inference_internal::AllocateFloatVector(executor, width_));
  RETURN_IF_ERROR(inference_ops::RmsNorm(executor, Floats(input),
                                         Floats(weight_), width_, epsilon_,
                                         Floats(result)));
  ASSIGN_OR_RETURN(Buffer output,
                   inference_internal::ToBFloat16(executor, result, width_));
  return FwdResult{{std::move(output)}, {}};
}

absl::StatusOr<BufferVec> RmsNormLayer::bwd_impl(cuda::Executor&,
                                                 absl::Span<const Buffer>,
                                                 BackwardState, LayerHooks*) {
  return UnsupportedBackward();
}

absl::StatusOr<std::unique_ptr<SwiGluLayer>> SwiGluLayer::Create(int width) {
  RETURN_IF_ERROR(CheckDimension(width));
  return absl::WrapUnique(new SwiGluLayer(width));
}

absl::StatusOr<FwdResult> SwiGluLayer::fwd_impl(cuda::Executor& executor,
                                                absl::Span<const Buffer> inputs,
                                                LayerHooks*) const {
  RETURN_IF_ERROR(
      inference_internal::ValidateInputs(executor, inputs, {width_, width_}));
  ASSIGN_OR_RETURN(Buffer gate,
                   inference_internal::ToFloat(executor, inputs[0], width_));
  ASSIGN_OR_RETURN(Buffer up,
                   inference_internal::ToFloat(executor, inputs[1], width_));
  ASSIGN_OR_RETURN(Buffer result,
                   inference_internal::AllocateFloatVector(executor, width_));
  RETURN_IF_ERROR(inference_ops::SwiGlu(executor, Floats(gate), Floats(up),
                                        width_, Floats(result)));
  ASSIGN_OR_RETURN(Buffer output,
                   inference_internal::ToBFloat16(executor, result, width_));
  return FwdResult{{std::move(output)}, {}};
}

absl::StatusOr<BufferVec> SwiGluLayer::bwd_impl(cuda::Executor&,
                                                absl::Span<const Buffer>,
                                                BackwardState, LayerHooks*) {
  return UnsupportedBackward();
}

}  // namespace pluto::llm
