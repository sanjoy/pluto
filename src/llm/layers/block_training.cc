#include "src/llm/layers/block_training.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cuda_tile.h>

#include <cmath>
#include <limits>
#include <type_traits>

#include "absl/memory/memory.h"
#include "src/util/status_macros.h"

namespace pluto::llm {
namespace {
namespace ct = ::cuda::tiles;
using namespace ct::literals;

template <class T>
__tile_global__ void DequantizeKernel(const T* input, const float* scales,
                                      int rows, int cols,
                                      __nv_bfloat16* output) {
  auto x = ct::partition_view{ct::tensor_span{input, ct::extents{rows, cols}},
                              ct::shape{1_ic, 128_ic}};
  auto y = ct::partition_view{ct::tensor_span{output, ct::extents{rows, cols}},
                              ct::shape{1_ic, 128_ic}};
  auto v = ct::element_cast<float>(x.load_masked(ct::bid().x, ct::bid().y));
  if constexpr (std::is_same_v<T, __nv_fp8_e4m3>) {
    auto s = ct::partition_view{
        ct::tensor_span{scales,
                        ct::extents{(rows + 127) / 128, (cols + 127) / 128}},
        ct::shape{1_ic, 1_ic}};
    v = v * s.load(ct::bid().x / 128, ct::bid().y);
  }
  y.store_masked(ct::element_cast<__nv_bfloat16>(v), ct::bid().x, ct::bid().y);
}

template <class Output>
__tile_global__ void LinearForward(const __nv_bfloat16* weight,
                                   const __nv_bfloat16* input, int cols,
                                   int rows, int tokens, Output* output) {
  auto w = ct::partition_view{ct::tensor_span{weight, ct::extents{rows, cols}},
                              ct::shape{8_ic, 128_ic}};
  auto x = ct::partition_view{ct::tensor_span{input, ct::extents{tokens, cols}},
                              ct::shape{1_ic, 128_ic}};
  auto y =
      ct::partition_view{ct::tensor_span{output, ct::extents{tokens, rows}},
                         ct::shape{1_ic, 8_ic}};
  auto sum = ct::zeros<ct::tile<float, ct::shape<8, 128>>>();
  for (int c = 0; c < (cols + 127) / 128; ++c)
    sum = sum + ct::element_cast<float>(w.load_masked(ct::bid().x, c)) *
                    ct::element_cast<float>(x.load_masked(ct::bid().y, c));
  auto result = ct::reshape(ct::sum(sum, 1_ic), ct::shape{1_ic, 8_ic});
  y.store_masked(ct::element_cast<Output>(result), ct::bid().y, ct::bid().x);
}

// A tile owns an input-gradient column range for one token. Reductions visit
// output rows in a fixed order, so frozen and active weights have identical dx.
__tile_global__ void LinearInputGradient(const __nv_bfloat16* weight,
                                         const float* dy, int cols, int rows,
                                         int tokens, float* dx) {
  auto w = ct::partition_view{ct::tensor_span{weight, ct::extents{rows, cols}},
                              ct::shape{32_ic, 128_ic}};
  auto g = ct::partition_view{ct::tensor_span{dy, ct::extents{tokens, rows}},
                              ct::shape{1_ic, 32_ic}};
  auto out = ct::partition_view{ct::tensor_span{dx, ct::extents{tokens, cols}},
                                ct::shape{1_ic, 128_ic}};
  auto sum = ct::zeros<ct::tile<float, ct::shape<32, 128>>>();
  for (int r = 0; r < (rows + 31) / 32; ++r) {
    auto grad =
        ct::reshape(g.load_masked(ct::bid().y, r), ct::shape{32_ic, 1_ic});
    sum = sum + ct::element_cast<float>(w.load_masked(r, ct::bid().x)) * grad;
  }
  out.store_masked(ct::reshape(ct::sum(sum, 0_ic), ct::shape{1_ic, 128_ic}),
                   ct::bid().y, ct::bid().x);
}

// Each weight has one writer. Sum tokens serially, then add to the accumulator
// to support tied weights and gradient accumulation without floating atomics.
__tile_global__ void LinearWeightGradient(const __nv_bfloat16* input,
                                          const float* dy, int cols, int rows,
                                          int tokens, float* dw) {
  auto x = ct::partition_view{ct::tensor_span{input, ct::extents{tokens, cols}},
                              ct::shape{1_ic, 128_ic}};
  auto g = ct::partition_view{ct::tensor_span{dy, ct::extents{tokens, rows}},
                              ct::shape{1_ic, 8_ic}};
  auto out = ct::partition_view{ct::tensor_span{dw, ct::extents{rows, cols}},
                                ct::shape{8_ic, 128_ic}};
  auto sum = ct::zeros<ct::tile<float, ct::shape<8, 128>>>();
  for (int t = 0; t < tokens; ++t)
    sum = sum +
          ct::reshape(g.load_masked(t, ct::bid().y), ct::shape{8_ic, 1_ic}) *
              ct::element_cast<float>(x.load_masked(t, ct::bid().x));
  out.store_masked(out.load_masked(ct::bid().y, ct::bid().x) + sum, ct::bid().y,
                   ct::bid().x);
}

template <bool Backward>
__tile_global__ void EmbeddingKernel(const __nv_bfloat16* weight,
                                     const int32_t* ids, const float* dy,
                                     int vocab, int width, int tokens,
                                     __nv_bfloat16* output, float* dw) {
  auto index = ct::partition_view{ct::tensor_span{ids, ct::extents{tokens}},
                                  ct::shape{1_ic}};
  auto y =
      ct::partition_view{ct::tensor_span{output, ct::extents{tokens, width}},
                         ct::shape{1_ic, 256_ic}};
  auto w =
      ct::partition_view{ct::tensor_span{weight, ct::extents{vocab, width}},
                         ct::shape{1_ic, 256_ic}};
  if constexpr (!Backward) {
    const int token = static_cast<int>(index.load(ct::bid().y));
    if (token >= 0 && token < vocab)
      y.store_masked(w.load_masked(token, ct::bid().x), ct::bid().y,
                     ct::bid().x);
    else
      y.store_masked(ct::zeros<ct::tile<__nv_bfloat16, ct::shape<1, 256>>>(),
                     ct::bid().y, ct::bid().x);
  } else {
    auto grad =
        ct::partition_view{ct::tensor_span{dy, ct::extents{tokens, width}},
                           ct::shape{1_ic, 256_ic}};
    auto out =
        ct::partition_view{ct::tensor_span{dw, ct::extents{vocab, width}},
                           ct::shape{1_ic, 256_ic}};
    // Each block owns a disjoint range of feature columns. Repeated IDs are
    // processed serially within it, not concurrently by separate token blocks.
    for (int t = 0; t < tokens; ++t) {
      const int token = static_cast<int>(index.load(t));
      if (token >= 0 && token < vocab)
        out.store_masked(out.load_masked(token, ct::bid().x) +
                             grad.load_masked(t, ct::bid().x),
                         token, ct::bid().x);
    }
  }
}

template <int Size, bool Backward>
__tile_global__ void RmsKernel(const __nv_bfloat16* input, const float* weight,
                               const float* dy, int width, int tokens,
                               float eps, __nv_bfloat16* output, float* dx,
                               float* inverse_rms) {
  auto x = ct::partition_view{
      ct::tensor_span{input, ct::extents{tokens, width}}, ct::shape<1, Size>{}};
  auto w = ct::partition_view{ct::tensor_span{weight, ct::extents{width}},
                              ct::shape<Size>{}};
  const int t = ct::bid().x;
  auto value = ct::element_cast<float>(x.load_masked(t, 0));
  auto inv = ct::rsqrt(ct::sum(value * value, 1_ic) / float(width) + eps);
  if constexpr (Backward) {
    auto g = ct::partition_view{ct::tensor_span{dy, ct::extents{tokens, width}},
                                ct::shape<1, Size>{}};
    auto result = ct::partition_view{
        ct::tensor_span{dx, ct::extents{tokens, width}}, ct::shape<1, Size>{}};
    auto scaled_gradient =
        g.load_masked(t, 0) *
        (1.0f + ct::reshape(w.load_masked(0), ct::shape<1, Size>{}));
    auto dot = ct::sum(scaled_gradient * value, 1_ic);
    result.store_masked(
        inv * scaled_gradient - value * (inv * inv * inv * dot / float(width)),
        t, 0);
  } else {
    auto y =
        ct::partition_view{ct::tensor_span{output, ct::extents{tokens, width}},
                           ct::shape<1, Size>{}};
    auto inverse = ct::partition_view{
        ct::tensor_span{inverse_rms, ct::extents{tokens}}, ct::shape{1_ic}};
    y.store_masked(
        ct::element_cast<__nv_bfloat16>(
            value * inv *
            (1.0f + ct::reshape(w.load_masked(0), ct::shape<1, Size>{}))),
        t, 0);
    inverse.store(ct::reshape(inv, ct::shape{1_ic}), t);
  }
}

__tile_global__ void RmsWeightGradient(const __nv_bfloat16* input,
                                       const float* dy, const float* inverse,
                                       int width, int tokens, float* dw) {
  auto x =
      ct::partition_view{ct::tensor_span{input, ct::extents{tokens, width}},
                         ct::shape{1_ic, 256_ic}};
  auto g = ct::partition_view{ct::tensor_span{dy, ct::extents{tokens, width}},
                              ct::shape{1_ic, 256_ic}};
  auto inv = ct::partition_view{ct::tensor_span{inverse, ct::extents{tokens}},
                                ct::shape{1_ic}};
  auto out = ct::partition_view{ct::tensor_span{dw, ct::extents{width}},
                                ct::shape{256_ic}};
  auto sum = ct::zeros<ct::tile<float, ct::shape<1, 256>>>();
  for (int t = 0; t < tokens; ++t)
    sum = sum + ct::element_cast<float>(x.load_masked(t, ct::bid().x)) *
                    g.load_masked(t, ct::bid().x) * inv.load(t);
  out.store_masked(
      out.load_masked(ct::bid().x) + ct::reshape(sum, ct::shape{256_ic}),
      ct::bid().x);
}

template <bool Backward>
__tile_global__ void SwiGluKernel(const __nv_bfloat16* gate,
                                  const __nv_bfloat16* up, const float* dy,
                                  int elements, __nv_bfloat16* output,
                                  float* dgate, float* dup) {
  auto a = ct::partition_view{ct::tensor_span{gate, ct::extents{elements}},
                              ct::shape{256_ic}};
  auto b = ct::partition_view{ct::tensor_span{up, ct::extents{elements}},
                              ct::shape{256_ic}};
  const int tile = ct::bid().x;
  auto g = ct::element_cast<float>(a.load_masked(tile));
  auto u = ct::element_cast<float>(b.load_masked(tile));
  auto sigmoid = 1.0f / (1.0f + ct::exp(-g));
  auto silu =
      ct::element_cast<float>(ct::element_cast<__nv_bfloat16>(g * sigmoid));
  if constexpr (Backward) {
    auto grad = ct::partition_view{ct::tensor_span{dy, ct::extents{elements}},
                                   ct::shape{256_ic}};
    auto ga = ct::partition_view{ct::tensor_span{dgate, ct::extents{elements}},
                                 ct::shape{256_ic}};
    auto gb = ct::partition_view{ct::tensor_span{dup, ct::extents{elements}},
                                 ct::shape{256_ic}};
    ga.store_masked(
        grad.load_masked(tile) * u * sigmoid * (1.0f + g * (1.0f - sigmoid)),
        tile);
    gb.store_masked(grad.load_masked(tile) * silu, tile);
  } else {
    auto y = ct::partition_view{ct::tensor_span{output, ct::extents{elements}},
                                ct::shape{256_ic}};
    y.store_masked(ct::element_cast<__nv_bfloat16>(silu * u), tile);
  }
}

absl::Status CheckBuffer(cuda::Executor& executor, const Buffer& buffer,
                         size_t bytes) {
  if (&buffer.executor() != &executor || buffer.size_bytes() != bytes)
    return absl::InvalidArgumentError(
        "block training requires the owning executor, exact dimensions and "
        "batch size one");
  return absl::OkStatus();
}

absl::Status CheckParameter(cuda::Executor& executor,
                            const std::shared_ptr<BlockParameter>& parameter,
                            int first, int second, int sequence,
                            DataType storage) {
  if (!parameter || first <= 0 || second <= 0 || sequence <= 0 ||
      first > 1048576 || second > 1048576 ||
      static_cast<int64_t>(sequence) * first >
          std::numeric_limits<int>::max() ||
      static_cast<int64_t>(sequence) * second >
          std::numeric_limits<int>::max() ||
      parameter->storage() != storage ||
      parameter->elements() != static_cast<size_t>(first) * second ||
      &parameter->value().executor() != &executor)
    return absl::InvalidArgumentError(
        "invalid block parameter dimensions, storage or executor");
  return absl::OkStatus();
}

template <bool Backward>
void LaunchRms(cuda::Executor& executor, const Buffer& input,
               const Buffer& weight, const float* dy, int width, int sequence,
               float eps, __nv_bfloat16* output, float* dx, float* inverse) {
  if (width <= 256)
    RmsKernel<256, Backward><<<sequence, 1, 0, executor.stream()>>>(
        static_cast<const __nv_bfloat16*>(input.data()),
        static_cast<const float*>(weight.data()), dy, width, sequence, eps,
        output, dx, inverse);
  else if (width <= 8192)
    RmsKernel<8192, Backward><<<sequence, 1, 0, executor.stream()>>>(
        static_cast<const __nv_bfloat16*>(input.data()),
        static_cast<const float*>(weight.data()), dy, width, sequence, eps,
        output, dx, inverse);
  else
    RmsKernel<16384, Backward><<<sequence, 1, 0, executor.stream()>>>(
        static_cast<const __nv_bfloat16*>(input.data()),
        static_cast<const float*>(weight.data()), dy, width, sequence, eps,
        output, dx, inverse);
}
}  // namespace

absl::StatusOr<Buffer> DequantizeMatrix(cuda::Executor& executor,
                                        const Buffer& input,
                                        inference_ops::MatrixStorage storage,
                                        const std::optional<Buffer>& scales,
                                        int rows, int cols) {
  using Storage = inference_ops::MatrixStorage;
  const size_t size = storage == Storage::kBFloat16  ? 2
                      : storage == Storage::kFloat32 ? 4
                                                     : 1;
  if (rows <= 0 || cols <= 0 || rows > 1048576 || cols > 1048576 ||
      (storage != Storage::kBFloat16 && storage != Storage::kFloat32 &&
       storage != Storage::kFp8E4M3))
    return absl::InvalidArgumentError("invalid imported matrix shape/storage");
  RETURN_IF_ERROR(
      CheckBuffer(executor, input, static_cast<size_t>(rows) * cols * size));
  if (storage == Storage::kFp8E4M3) {
    if (!scales)
      return absl::InvalidArgumentError("FP8 matrix requires scales");
    RETURN_IF_ERROR(CheckBuffer(
        executor, *scales,
        static_cast<size_t>((rows + 127) / 128) * ((cols + 127) / 128) * 4));
  } else if (scales) {
    return absl::InvalidArgumentError("non-FP8 matrix must not have scales");
  }
  if (storage == Storage::kBFloat16)
    return input;
  ASSIGN_OR_RETURN(
      auto output,
      Buffer::Allocate(executor, static_cast<size_t>(rows) * cols * 2));
  const dim3 grid(rows, (cols + 127) / 128);
  if (storage == Storage::kFloat32)
    DequantizeKernel<<<grid, 1, 0, executor.stream()>>>(
        static_cast<const float*>(input.data()), nullptr, rows, cols,
        static_cast<__nv_bfloat16*>(output.data()));
  else
    DequantizeKernel<<<grid, 1, 0, executor.stream()>>>(
        static_cast<const __nv_fp8_e4m3*>(input.data()),
        static_cast<const float*>(scales->data()), rows, cols,
        static_cast<__nv_bfloat16*>(output.data()));
  RETURN_IF_ERROR(
      cuda::CudaStatus(cudaGetLastError(), "dequantize block matrix"));
  return output;
}

absl::StatusOr<std::unique_ptr<BlockLinearLayer>> BlockLinearLayer::Create(
    cuda::Executor& executor, std::shared_ptr<BlockParameter> weight,
    int input_dim, int output_dim, int sequence_length, bool output_float32) {
  RETURN_IF_ERROR(CheckParameter(executor, weight, input_dim, output_dim,
                                 sequence_length, DataType::BF16));
  return absl::WrapUnique(new BlockLinearLayer(std::move(weight), input_dim,
                                               output_dim, sequence_length,
                                               output_float32));
}

absl::StatusOr<FwdResult> BlockLinearLayer::fwd_impl(
    cuda::Executor& executor, absl::Span<const Buffer> inputs,
    LayerHooks*) const {
  if (inputs.size() != 1)
    return absl::InvalidArgumentError("linear requires one input");
  RETURN_IF_ERROR(
      CheckBuffer(executor, inputs[0],
                  static_cast<size_t>(sequence_length_) * input_dim_ * 2));
  RETURN_IF_ERROR(
      CheckBuffer(executor, parameter_->value(),
                  static_cast<size_t>(input_dim_) * output_dim_ * 2));
  ASSIGN_OR_RETURN(
      auto output,
      Buffer::Allocate(executor, static_cast<size_t>(sequence_length_) *
                                     output_dim_ * (output_float32_ ? 4 : 2)));
  const dim3 grid((output_dim_ + 7) / 8, sequence_length_);
  if (output_float32_)
    LinearForward<<<grid, 1, 0, executor.stream()>>>(
        static_cast<const __nv_bfloat16*>(parameter_->value().data()),
        static_cast<const __nv_bfloat16*>(inputs[0].data()), input_dim_,
        output_dim_, sequence_length_, static_cast<float*>(output.data()));
  else
    LinearForward<<<grid, 1, 0, executor.stream()>>>(
        static_cast<const __nv_bfloat16*>(parameter_->value().data()),
        static_cast<const __nv_bfloat16*>(inputs[0].data()), input_dim_,
        output_dim_, sequence_length_,
        static_cast<__nv_bfloat16*>(output.data()));
  RETURN_IF_ERROR(cuda::CudaStatus(cudaGetLastError(), "block linear forward"));
  return FwdResult{{std::move(output)},
                   BackwardState{.intermediates = {inputs[0]}}};
}

absl::StatusOr<BufferVec> BlockLinearLayer::bwd_impl(
    cuda::Executor& executor, absl::Span<const Buffer> gradients,
    BackwardState state, LayerHooks*) {
  if (gradients.size() != 1 || state.intermediates.size() != 1)
    return absl::InvalidArgumentError("invalid linear backward state/gradient");
  RETURN_IF_ERROR(
      CheckBuffer(executor, gradients[0],
                  static_cast<size_t>(sequence_length_) * output_dim_ * 4));
  RETURN_IF_ERROR(
      CheckBuffer(executor, state.intermediates[0],
                  static_cast<size_t>(sequence_length_) * input_dim_ * 2));
  ASSIGN_OR_RETURN(
      auto dx,
      Buffer::Allocate(executor,
                       static_cast<size_t>(sequence_length_) * input_dim_ * 4));
  LinearInputGradient<<<dim3((input_dim_ + 127) / 128, sequence_length_), 1, 0,
                        executor.stream()>>>(
      static_cast<const __nv_bfloat16*>(parameter_->value().data()),
      static_cast<const float*>(gradients[0].data()), input_dim_, output_dim_,
      sequence_length_, static_cast<float*>(dx.data()));
  RETURN_IF_ERROR(
      cuda::CudaStatus(cudaGetLastError(), "block linear input gradient"));
  if (parameter_->active()) {
    LinearWeightGradient<<<dim3((input_dim_ + 127) / 128,
                                (output_dim_ + 7) / 8),
                           1, 0, executor.stream()>>>(
        static_cast<const __nv_bfloat16*>(state.intermediates[0].data()),
        static_cast<const float*>(gradients[0].data()), input_dim_, output_dim_,
        sequence_length_, static_cast<float*>(parameter_->gradient().data()));
    RETURN_IF_ERROR(
        cuda::CudaStatus(cudaGetLastError(), "block linear weight gradient"));
  }
  return BufferVec{std::move(dx)};
}

absl::StatusOr<std::unique_ptr<BlockEmbeddingLayer>>
BlockEmbeddingLayer::Create(cuda::Executor& executor,
                            std::shared_ptr<BlockParameter> weight,
                            int vocab_size, int width, int sequence_length) {
  RETURN_IF_ERROR(CheckParameter(executor, weight, vocab_size, width,
                                 sequence_length, DataType::BF16));
  return absl::WrapUnique(new BlockEmbeddingLayer(std::move(weight), vocab_size,
                                                  width, sequence_length));
}

absl::StatusOr<FwdResult> BlockEmbeddingLayer::fwd_impl(
    cuda::Executor& executor, absl::Span<const Buffer> inputs,
    LayerHooks*) const {
  if (inputs.size() != 1)
    return absl::InvalidArgumentError("embedding requires one input");
  RETURN_IF_ERROR(CheckBuffer(executor, inputs[0],
                              static_cast<size_t>(sequence_length_) * 4));
  RETURN_IF_ERROR(CheckBuffer(executor, parameter_->value(),
                              static_cast<size_t>(vocab_size_) * width_ * 2));
  ASSIGN_OR_RETURN(
      auto output,
      Buffer::Allocate(executor,
                       static_cast<size_t>(sequence_length_) * width_ * 2));
  EmbeddingKernel<false><<<dim3((width_ + 255) / 256, sequence_length_), 1, 0,
                           executor.stream()>>>(
      static_cast<const __nv_bfloat16*>(parameter_->value().data()),
      static_cast<const int32_t*>(inputs[0].data()), nullptr, vocab_size_,
      width_, sequence_length_, static_cast<__nv_bfloat16*>(output.data()),
      nullptr);
  RETURN_IF_ERROR(
      cuda::CudaStatus(cudaGetLastError(), "block embedding forward"));
  return FwdResult{{std::move(output)},
                   BackwardState{.intermediates = {inputs[0]}}};
}

absl::StatusOr<BufferVec> BlockEmbeddingLayer::bwd_impl(
    cuda::Executor& executor, absl::Span<const Buffer> gradients,
    BackwardState state, LayerHooks*) {
  if (gradients.size() != 1 || state.intermediates.size() != 1)
    return absl::InvalidArgumentError(
        "invalid embedding backward state/gradient");
  RETURN_IF_ERROR(
      CheckBuffer(executor, gradients[0],
                  static_cast<size_t>(sequence_length_) * width_ * 4));
  RETURN_IF_ERROR(CheckBuffer(executor, state.intermediates[0],
                              static_cast<size_t>(sequence_length_) * 4));
  if (parameter_->active()) {
    EmbeddingKernel<true><<<(width_ + 255) / 256, 1, 0, executor.stream()>>>(
        static_cast<const __nv_bfloat16*>(parameter_->value().data()),
        static_cast<const int32_t*>(state.intermediates[0].data()),
        static_cast<const float*>(gradients[0].data()), vocab_size_, width_,
        sequence_length_, nullptr,
        static_cast<float*>(parameter_->gradient().data()));
    RETURN_IF_ERROR(
        cuda::CudaStatus(cudaGetLastError(), "block embedding gradient"));
  }
  return BufferVec{};
}

absl::StatusOr<std::unique_ptr<BlockRmsNormLayer>> BlockRmsNormLayer::Create(
    cuda::Executor& executor, std::shared_ptr<BlockParameter> weight, int width,
    int sequence_length, float epsilon) {
  RETURN_IF_ERROR(CheckParameter(executor, weight, width, 1, sequence_length,
                                 DataType::FP32));
  if (width > 16384 || !std::isfinite(epsilon) || epsilon <= 0)
    return absl::InvalidArgumentError("invalid RMSNorm width or epsilon");
  return absl::WrapUnique(new BlockRmsNormLayer(std::move(weight), width,
                                                sequence_length, epsilon));
}

absl::StatusOr<FwdResult> BlockRmsNormLayer::fwd_impl(
    cuda::Executor& executor, absl::Span<const Buffer> inputs,
    LayerHooks*) const {
  if (inputs.size() != 1)
    return absl::InvalidArgumentError("RMSNorm requires one input");
  RETURN_IF_ERROR(CheckBuffer(
      executor, inputs[0], static_cast<size_t>(sequence_length_) * width_ * 2));
  RETURN_IF_ERROR(CheckBuffer(executor, parameter_->value(),
                              static_cast<size_t>(width_) * 4));
  ASSIGN_OR_RETURN(
      auto output,
      Buffer::Allocate(executor,
                       static_cast<size_t>(sequence_length_) * width_ * 2));
  ASSIGN_OR_RETURN(
      auto inverse,
      Buffer::Allocate(executor, static_cast<size_t>(sequence_length_) * 4));
  LaunchRms<false>(executor, inputs[0], parameter_->value(), nullptr, width_,
                   sequence_length_, epsilon_,
                   static_cast<__nv_bfloat16*>(output.data()), nullptr,
                   static_cast<float*>(inverse.data()));
  RETURN_IF_ERROR(
      cuda::CudaStatus(cudaGetLastError(), "block RMSNorm forward"));
  return FwdResult{
      {std::move(output)},
      BackwardState{.intermediates = {inputs[0], std::move(inverse)}}};
}

absl::StatusOr<BufferVec> BlockRmsNormLayer::bwd_impl(
    cuda::Executor& executor, absl::Span<const Buffer> gradients,
    BackwardState state, LayerHooks*) {
  if (gradients.size() != 1 || state.intermediates.size() != 2)
    return absl::InvalidArgumentError(
        "invalid RMSNorm backward state/gradient");
  RETURN_IF_ERROR(
      CheckBuffer(executor, gradients[0],
                  static_cast<size_t>(sequence_length_) * width_ * 4));
  RETURN_IF_ERROR(
      CheckBuffer(executor, state.intermediates[0],
                  static_cast<size_t>(sequence_length_) * width_ * 2));
  RETURN_IF_ERROR(CheckBuffer(executor, state.intermediates[1],
                              static_cast<size_t>(sequence_length_) * 4));
  ASSIGN_OR_RETURN(
      auto dx,
      Buffer::Allocate(executor,
                       static_cast<size_t>(sequence_length_) * width_ * 4));
  LaunchRms<true>(executor, state.intermediates[0], parameter_->value(),
                  static_cast<const float*>(gradients[0].data()), width_,
                  sequence_length_, epsilon_, nullptr,
                  static_cast<float*>(dx.data()), nullptr);
  RETURN_IF_ERROR(
      cuda::CudaStatus(cudaGetLastError(), "block RMSNorm input gradient"));
  if (parameter_->active()) {
    RmsWeightGradient<<<(width_ + 255) / 256, 1, 0, executor.stream()>>>(
        static_cast<const __nv_bfloat16*>(state.intermediates[0].data()),
        static_cast<const float*>(gradients[0].data()),
        static_cast<const float*>(state.intermediates[1].data()), width_,
        sequence_length_, static_cast<float*>(parameter_->gradient().data()));
    RETURN_IF_ERROR(
        cuda::CudaStatus(cudaGetLastError(), "block RMSNorm weight gradient"));
  }
  return BufferVec{std::move(dx)};
}

absl::StatusOr<std::unique_ptr<BlockSwiGluLayer>> BlockSwiGluLayer::Create(
    int width, int sequence_length) {
  if (width <= 0 || sequence_length <= 0 ||
      static_cast<int64_t>(width) * sequence_length >
          std::numeric_limits<int>::max())
    return absl::InvalidArgumentError("invalid SwiGLU dimensions");
  return absl::WrapUnique(new BlockSwiGluLayer(width, sequence_length));
}

absl::StatusOr<FwdResult> BlockSwiGluLayer::fwd_impl(
    cuda::Executor& executor, absl::Span<const Buffer> inputs,
    LayerHooks*) const {
  if (inputs.size() != 2)
    return absl::InvalidArgumentError("SwiGLU requires gate and up inputs");
  const int n = width_ * sequence_length_;
  for (const auto& input : inputs)
    RETURN_IF_ERROR(CheckBuffer(executor, input, static_cast<size_t>(n) * 2));
  ASSIGN_OR_RETURN(auto output,
                   Buffer::Allocate(executor, static_cast<size_t>(n) * 2));
  SwiGluKernel<false><<<(n + 255) / 256, 1, 0, executor.stream()>>>(
      static_cast<const __nv_bfloat16*>(inputs[0].data()),
      static_cast<const __nv_bfloat16*>(inputs[1].data()), nullptr, n,
      static_cast<__nv_bfloat16*>(output.data()), nullptr, nullptr);
  RETURN_IF_ERROR(cuda::CudaStatus(cudaGetLastError(), "block SwiGLU forward"));
  return FwdResult{{std::move(output)},
                   BackwardState{.intermediates = {inputs[0], inputs[1]}}};
}

absl::StatusOr<BufferVec> BlockSwiGluLayer::bwd_impl(
    cuda::Executor& executor, absl::Span<const Buffer> gradients,
    BackwardState state, LayerHooks*) {
  if (gradients.size() != 1 || state.intermediates.size() != 2)
    return absl::InvalidArgumentError("invalid SwiGLU backward state/gradient");
  const int n = width_ * sequence_length_;
  RETURN_IF_ERROR(
      CheckBuffer(executor, gradients[0], static_cast<size_t>(n) * 4));
  for (const auto& input : state.intermediates)
    RETURN_IF_ERROR(CheckBuffer(executor, input, static_cast<size_t>(n) * 2));
  ASSIGN_OR_RETURN(auto da,
                   Buffer::Allocate(executor, static_cast<size_t>(n) * 4));
  ASSIGN_OR_RETURN(auto db,
                   Buffer::Allocate(executor, static_cast<size_t>(n) * 4));
  SwiGluKernel<true><<<(n + 255) / 256, 1, 0, executor.stream()>>>(
      static_cast<const __nv_bfloat16*>(state.intermediates[0].data()),
      static_cast<const __nv_bfloat16*>(state.intermediates[1].data()),
      static_cast<const float*>(gradients[0].data()), n, nullptr,
      static_cast<float*>(da.data()), static_cast<float*>(db.data()));
  RETURN_IF_ERROR(
      cuda::CudaStatus(cudaGetLastError(), "block SwiGLU backward"));
  return BufferVec{std::move(da), std::move(db)};
}
}  // namespace pluto::llm
