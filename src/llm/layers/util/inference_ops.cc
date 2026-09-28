#include "src/llm/layers/util/inference_ops.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cuda_tile.h>

#include <cmath>
#include <type_traits>

#include "src/util/status_macros.h"

namespace pluto::llm::inference_ops {
namespace {

namespace ct = ::cuda::tiles;
using namespace ct::literals;

template <class T>
__tile_global__ void MatVecKernel(const T* weights, const float* scales,
                                  const float* input, int cols, int rows,
                                  float* output, bool round_bf16) {
  // Eight consecutive output rows share each input group. The 128-column
  // boundary is also the checkpoint's quantization-block boundary.
  auto w = ct::partition_view{ct::tensor_span{weights, ct::extents{rows, cols}},
                              ct::shape{8_ic, 128_ic}};
  auto x = ct::partition_view{ct::tensor_span{input, ct::extents{cols}},
                              ct::shape{128_ic}};
  auto s = ct::partition_view{
      ct::tensor_span{scales,
                      ct::extents{(rows + 127) / 128, (cols + 127) / 128}},
      ct::shape{1_ic, 1_ic}};
  auto y = ct::partition_view{ct::tensor_span{output, ct::extents{rows}},
                              ct::shape{8_ic}};
  const int row = ct::bid().x;
  auto sum = ct::zeros<ct::tile<float, ct::shape<8, 128>>>();
  for (int col = 0; col < (cols + 127) / 128; ++col) {
    auto value = ct::element_cast<float>(w.load_masked(row, col));
    if constexpr (std::is_same_v<T, __nv_fp8_e4m3>)
      value = value * s.load(row / 16, col);
    sum = sum +
          value * ct::broadcast(x.load_masked(col), ct::shape{8_ic, 128_ic});
  }
  auto result = ct::reshape(ct::sum(sum, 1_ic), ct::shape{8_ic});
  if (round_bf16)
    result = ct::element_cast<float>(ct::element_cast<__nv_bfloat16>(result));
  y.store_masked(result, row);
}

__tile_global__ void QuantizeKernel(const float* input, int n, float* output) {
  auto x = ct::partition_view{ct::tensor_span{input, ct::extents{n}},
                              ct::shape{128_ic}};
  auto y = ct::partition_view{ct::tensor_span{output, ct::extents{n}},
                              ct::shape{128_ic}};
  const int block = ct::bid().x;
  auto v = x.load_masked(block);
  auto magnitude = ct::max(v, -v);
  auto scale = ct::reduce_max(magnitude, 0_ic) / 448.0f;
  // Match HF finegrained-fp8: guard the quantization divisor, but retain the
  // original scale for dequantization (including all-zero/tiny groups).
  auto quantized = ct::element_cast<__nv_fp8_e4m3>(v / ct::max(scale, 1e-12f));
  y.store_masked(ct::element_cast<float>(quantized) * scale, block);
}

template <int Size>
__tile_global__ void NormKernel(const float* input, const float* weight, int n,
                                float eps, float* output, bool round_bf16) {
  auto x = ct::partition_view{ct::tensor_span{input, ct::extents{n}},
                              ct::shape<Size>{}};
  auto w = ct::partition_view{ct::tensor_span{weight, ct::extents{n}},
                              ct::shape<Size>{}};
  auto y = ct::partition_view{ct::tensor_span{output, ct::extents{n}},
                              ct::shape<Size>{}};
  auto v = x.load_masked(0);
  auto result = v * ct::rsqrt(ct::sum(v * v, 0_ic) / float(n) + eps) *
                (1.0f + w.load_masked(0));
  if (round_bf16)
    result = ct::element_cast<float>(ct::element_cast<__nv_bfloat16>(result));
  y.store_masked(result, 0);
}

template <bool Swi>
__tile_global__ void BinaryKernel(const float* first, const float* second,
                                  int n, float* output, bool round_bf16) {
  auto x = ct::partition_view{ct::tensor_span{first, ct::extents{n}},
                              ct::shape{256_ic}};
  auto z = ct::partition_view{ct::tensor_span{second, ct::extents{n}},
                              ct::shape{256_ic}};
  auto y = ct::partition_view{ct::tensor_span{output, ct::extents{n}},
                              ct::shape{256_ic}};
  const int block = ct::bid().x;
  auto a = x.load_masked(block);
  auto b = z.load_masked(block);
  auto result = a + b;
  if constexpr (Swi) {
    auto silu = a / (1.0f + ct::exp(-a));
    if (round_bf16)
      silu = ct::element_cast<float>(ct::element_cast<__nv_bfloat16>(silu));
    result = silu * b;
  }
  if (round_bf16)
    result = ct::element_cast<float>(ct::element_cast<__nv_bfloat16>(result));
  y.store_masked(result, block);
}

template <class T>
__tile_global__ void EmbeddingKernel(const T* weights, int token, int width,
                                     float* output) {
  auto x = ct::partition_view{
      ct::tensor_span{weights + static_cast<int64_t>(token) * width,
                      ct::extents{width}},
      ct::shape{256_ic}};
  auto y = ct::partition_view{ct::tensor_span{output, ct::extents{width}},
                              ct::shape{256_ic}};
  y.store_masked(ct::element_cast<float>(x.load_masked(ct::bid().x)),
                 ct::bid().x);
}

template <class T>
__tile_global__ void DeviceEmbeddingKernel(const T* weights,
                                           const int32_t* token, int vocab,
                                           int width, float* output) {
  auto index = ct::partition_view{ct::tensor_span{token, ct::extents{1}},
                                  ct::shape{1_ic}};
  auto y = ct::partition_view{ct::tensor_span{output, ct::extents{width}},
                              ct::shape{256_ic}};
  const int id = static_cast<int>(index.load(0));
  if (id < 0 || id >= vocab) {
    y.store_masked(ct::zeros<ct::tile<float, ct::shape<256>>>(), ct::bid().x);
    return;
  }
  auto x = ct::partition_view{
      ct::tensor_span{weights + static_cast<int64_t>(id) * width,
                      ct::extents{width}},
      ct::shape{256_ic}};
  y.store_masked(ct::element_cast<float>(x.load_masked(ct::bid().x)),
                 ct::bid().x);
}

absl::Status CheckElements(int n, const void* x, const void* y) {
  if (n <= 0 || x == nullptr || y == nullptr)
    return absl::InvalidArgumentError(
        "operator requires non-null buffers and positive size");
  return absl::OkStatus();
}

}  // namespace

absl::Status MatVec(cuda::Executor& executor, const void* weights,
                    MatrixStorage storage, const float* scales,
                    const float* input, int input_dim, int output_dim,
                    float* output, bool round_bf16) {
  if (input_dim <= 0 || output_dim <= 0 || input_dim > 1048576 ||
      output_dim > 1048576 || !weights || !input || !output)
    return absl::InvalidArgumentError(
        "invalid matrix-vector dimensions or pointers");
  if (storage == MatrixStorage::kFp8E4M3 && !scales)
    return absl::InvalidArgumentError("FP8 weights require block scales");
  const int grid = (output_dim + 7) / 8;
  switch (storage) {
    case MatrixStorage::kBFloat16:
      MatVecKernel<<<grid, 1, 0, executor.stream()>>>(
          static_cast<const __nv_bfloat16*>(weights), scales, input, input_dim,
          output_dim, output, round_bf16);
      break;
    case MatrixStorage::kFloat32:
      MatVecKernel<<<grid, 1, 0, executor.stream()>>>(
          static_cast<const float*>(weights), scales, input, input_dim,
          output_dim, output, round_bf16);
      break;
    case MatrixStorage::kFp8E4M3:
      MatVecKernel<<<grid, 1, 0, executor.stream()>>>(
          static_cast<const __nv_fp8_e4m3*>(weights), scales, input, input_dim,
          output_dim, output, round_bf16);
      break;
    default:
      return absl::InvalidArgumentError("unknown matrix storage type");
  }
  return cuda::CudaStatus(cudaGetLastError(), "inference MatVecKernel");
}

absl::Status QuantizeFp8Input(cuda::Executor& executor, const float* input,
                              int elements, float* output) {
  RETURN_IF_ERROR(CheckElements(elements, input, output));
  QuantizeKernel<<<1 + (elements - 1) / 128, 1, 0, executor.stream()>>>(
      input, elements, output);
  return cuda::CudaStatus(cudaGetLastError(), "inference QuantizeKernel");
}

absl::Status RmsNorm(cuda::Executor& executor, const float* input,
                     const float* weight, int width, float epsilon,
                     float* output, bool round_bf16) {
  if (width <= 0 || width > 16384 || !input || !weight || !output ||
      !std::isfinite(epsilon) || epsilon <= 0)
    return absl::InvalidArgumentError("invalid RMSNorm input");
  if (width <= 256)
    NormKernel<256><<<1, 1, 0, executor.stream()>>>(
        input, weight, width, epsilon, output, round_bf16);
  else if (width <= 8192)
    NormKernel<8192><<<1, 1, 0, executor.stream()>>>(
        input, weight, width, epsilon, output, round_bf16);
  else
    NormKernel<16384><<<1, 1, 0, executor.stream()>>>(
        input, weight, width, epsilon, output, round_bf16);
  return cuda::CudaStatus(cudaGetLastError(), "inference NormKernel");
}

absl::Status SwiGlu(cuda::Executor& executor, const float* gate,
                    const float* up, int elements, float* output,
                    bool round_bf16) {
  if (!up)
    return absl::InvalidArgumentError("SwiGLU requires up input");
  RETURN_IF_ERROR(CheckElements(elements, gate, output));
  BinaryKernel<true><<<1 + (elements - 1) / 256, 1, 0, executor.stream()>>>(
      gate, up, elements, output, round_bf16);
  return cuda::CudaStatus(cudaGetLastError(), "inference SwiGlu");
}

absl::Status ResidualAdd(cuda::Executor& executor, const float* input,
                         const float* update, int elements, float* output,
                         bool round_bf16) {
  if (!update)
    return absl::InvalidArgumentError("residual requires update input");
  RETURN_IF_ERROR(CheckElements(elements, input, output));
  BinaryKernel<false><<<1 + (elements - 1) / 256, 1, 0, executor.stream()>>>(
      input, update, elements, output, round_bf16);
  return cuda::CudaStatus(cudaGetLastError(), "inference ResidualAdd");
}

absl::Status EmbeddingLookup(cuda::Executor& executor, const void* weights,
                             MatrixStorage storage, int token, int vocab_size,
                             int width, float* output) {
  if (token < 0 || token >= vocab_size || width <= 0 || !weights || !output)
    return absl::InvalidArgumentError("invalid embedding lookup");
  if (storage == MatrixStorage::kBFloat16)
    EmbeddingKernel<<<1 + (width - 1) / 256, 1, 0, executor.stream()>>>(
        static_cast<const __nv_bfloat16*>(weights), token, width, output);
  else if (storage == MatrixStorage::kFloat32)
    EmbeddingKernel<<<1 + (width - 1) / 256, 1, 0, executor.stream()>>>(
        static_cast<const float*>(weights), token, width, output);
  else
    return absl::UnimplementedError("FP8 token embeddings are not supported");
  return cuda::CudaStatus(cudaGetLastError(), "inference EmbeddingKernel");
}

absl::Status EmbeddingLookupDevice(cuda::Executor& executor,
                                   const void* weights, MatrixStorage storage,
                                   const int32_t* token, int vocab_size,
                                   int width, float* output) {
  if (!token || vocab_size <= 0 || width <= 0 || !weights || !output)
    return absl::InvalidArgumentError("invalid device embedding lookup");
  if (storage == MatrixStorage::kBFloat16)
    DeviceEmbeddingKernel<<<1 + (width - 1) / 256, 1, 0, executor.stream()>>>(
        static_cast<const __nv_bfloat16*>(weights), token, vocab_size, width,
        output);
  else if (storage == MatrixStorage::kFloat32)
    DeviceEmbeddingKernel<<<1 + (width - 1) / 256, 1, 0, executor.stream()>>>(
        static_cast<const float*>(weights), token, vocab_size, width, output);
  else
    return absl::UnimplementedError("FP8 token embeddings are not supported");
  return cuda::CudaStatus(cudaGetLastError(), "DeviceEmbeddingKernel");
}

}  // namespace pluto::llm::inference_ops
