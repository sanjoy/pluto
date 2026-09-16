#include "src/llm/vocabulary_readout.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <algorithm>
#include <cstddef>
#include <limits>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "src/util/status_macros.h"

namespace pluto::llm {
namespace {

constexpr int kThreads = 256;
constexpr int kRowsPerChunk = 16;
constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = kThreads / kWarpSize;

__device__ float ActivationValue(float value) { return value; }
__device__ float ActivationValue(__nv_bfloat16 value) {
  return __bfloat162float(value);
}

// Each warp owns one dot product. Its lanes read adjacent embedding columns;
// fixed per-lane loops and a fixed reduction tree avoid inter-block atomics.
// Visit only logical vocabulary rows, so padded embeddings cannot participate.
template <class Element>
__global__ void EmbeddingLogitsKernel(const Element* activations,
                                      const float* embedding, int width,
                                      int vocab_size, float* logits) {
  const size_t token = static_cast<size_t>(blockIdx.x) * kWarpsPerBlock +
                       threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  if (token >= static_cast<size_t>(vocab_size))
    return;
  const size_t row = blockIdx.y;
  float sum = 0;
  for (size_t column = lane; column < static_cast<size_t>(width);
       column += kWarpSize)
    sum = fmaf(ActivationValue(activations[row * width + column]),
               embedding[token * width + column], sum);
  for (int offset = kWarpSize / 2; offset > 0; offset /= 2)
    sum += __shfl_down_sync(0xffffffff, sum, offset);
  if (lane == 0)
    logits[row * vocab_size + token] = sum;
}

// Three fixed-order argmax reductions select distinct IDs, then a fourth pass
// computes the full softmax denominator. This deliberately favors obvious
// tie/normalization semantics over a more intricate top-k merge algorithm.
__global__ void TopThreeKernel(const float* logits, int vocab_size,
                               int row_stride, TopThreeTokens* output) {
  __shared__ float values[kThreads];
  __shared__ int ids[kThreads];
  __shared__ float winners[3];
  __shared__ int chosen[3];
  const int lane = threadIdx.x;
  const float* row = logits + static_cast<size_t>(blockIdx.x) * row_stride;
  for (int rank = 0; rank < 3; ++rank) {
    float best = -CUDART_INF_F;
    int id = 0x7fffffff;
    int invalid = 0;
    for (size_t token = lane; token < static_cast<size_t>(vocab_size);
         token += kThreads) {
      const float value = row[token];
      invalid |= !isfinite(value);
      bool used = false;
      for (int previous = 0; previous < rank; ++previous)
        used |= static_cast<int>(token) == chosen[previous];
      if (!used &&
          (value > best || (value == best && static_cast<int>(token) < id))) {
        best = value;
        id = token;
      }
    }
    if (__syncthreads_or(invalid)) {
      if (lane == 0)
        for (int i = 0; i < 3; ++i) {
          output[blockIdx.x].tokens[i] = -1;
          output[blockIdx.x].probabilities[i] = CUDART_NAN_F;
        }
      return;
    }
    values[lane] = best;
    ids[lane] = id;
    __syncthreads();
    for (int offset = kThreads / 2; offset > 0; offset /= 2) {
      if (lane < offset && (values[lane + offset] > values[lane] ||
                            (values[lane + offset] == values[lane] &&
                             ids[lane + offset] < ids[lane]))) {
        values[lane] = values[lane + offset];
        ids[lane] = ids[lane + offset];
      }
      __syncthreads();
    }
    if (lane == 0) {
      winners[rank] = values[0];
      chosen[rank] = ids[0];
    }
    // All threads finish reading reduction scratch before the next rank.
    __syncthreads();
  }
  float sum = 0;
  for (size_t token = lane; token < static_cast<size_t>(vocab_size);
       token += kThreads)
    sum += expf(row[token] - winners[0]);
  values[lane] = sum;
  __syncthreads();
  for (int offset = kThreads / 2; offset > 0; offset /= 2) {
    if (lane < offset)
      values[lane] += values[lane + offset];
    __syncthreads();
  }
  if (lane == 0)
    for (int rank = 0; rank < 3; ++rank) {
      output[blockIdx.x].tokens[rank] = chosen[rank];
      output[blockIdx.x].probabilities[rank] =
          expf(winners[rank] - winners[0]) / values[0];
    }
}

absl::StatusOr<size_t> CheckedBytes(size_t rows, size_t columns,
                                    size_t element_bytes) {
  if (columns == 0 || element_bytes == 0 ||
      columns > std::numeric_limits<size_t>::max() / element_bytes ||
      rows > std::numeric_limits<size_t>::max() / element_bytes / columns)
    return absl::InvalidArgumentError("readout tensor byte size overflows");
  return rows * columns * element_bytes;
}

absl::Status ValidateMatrixPrefix(cuda::Executor& executor,
                                  const cuda::Buffer& buffer, int rows,
                                  int columns, size_t element_bytes,
                                  const char* description) {
  ASSIGN_OR_RETURN(const size_t row_bytes,
                   CheckedBytes(1, columns, element_bytes));
  ASSIGN_OR_RETURN(const size_t required,
                   CheckedBytes(rows, columns, element_bytes));
  if (&buffer.executor() != &executor)
    return absl::InvalidArgumentError(
        absl::StrCat(description, " belongs to another executor"));
  if (buffer.size_bytes() < required || buffer.size_bytes() % row_bytes != 0)
    return absl::InvalidArgumentError(
        absl::StrCat(description, " must contain the requested whole rows"));
  return absl::OkStatus();
}

template <class Element>
absl::Status ProjectChunk(cuda::Executor& executor,
                          const cuda::Buffer& activations,
                          const cuda::Buffer& embedding, int start, int rows,
                          int vocab_size, int width,
                          const cuda::Buffer& logits) {
  // Subtract before rounding up so even INT_MAX vocabulary counts cannot
  // overflow signed dimension arithmetic.
  const int blocks = (vocab_size - 1) / kWarpsPerBlock + 1;
  EmbeddingLogitsKernel<<<dim3(blocks, rows), kThreads, 0, executor.stream()>>>(
      static_cast<const Element*>(activations.data()) +
          static_cast<size_t>(start) * width,
      static_cast<const float*>(embedding.data()), width, vocab_size,
      static_cast<float*>(logits.data()));
  return cuda::CudaStatus(cudaGetLastError(), "EmbeddingLogitsKernel launch");
}

}  // namespace

absl::StatusOr<cuda::Buffer> ReadTopThreeTokens(cuda::Executor& executor,
                                                const cuda::Buffer& fp32_logits,
                                                int rows, int logical_vocab,
                                                int row_stride) {
  if (rows <= 0 || logical_vocab < 3 || row_stride < logical_vocab)
    return absl::InvalidArgumentError(
        "top-three requires rows > 0, vocab >= 3, and stride >= vocab");
  RETURN_IF_ERROR(ValidateMatrixPrefix(executor, fp32_logits, rows, row_stride,
                                       sizeof(float), "readout logits"));
  ASSIGN_OR_RETURN(const size_t output_bytes,
                   CheckedBytes(rows, 1, sizeof(TopThreeTokens)));
  ASSIGN_OR_RETURN(auto output, cuda::Buffer::Allocate(executor, output_bytes));
  TopThreeKernel<<<rows, kThreads, 0, executor.stream()>>>(
      static_cast<const float*>(fp32_logits.data()), logical_vocab, row_stride,
      static_cast<TopThreeTokens*>(output.data()));
  RETURN_IF_ERROR(
      cuda::CudaStatus(cudaGetLastError(), "TopThreeKernel launch"));
  return output;
}

absl::StatusOr<cuda::Buffer> ReadEmbeddingNeighbors(
    cuda::Executor& executor, const cuda::Buffer& activations, DataType storage,
    const cuda::Buffer& fp32_embedding, int rows, int logical_vocab,
    int width) {
  if (rows <= 0 || logical_vocab < 3 || width <= 0)
    return absl::InvalidArgumentError(
        "embedding readout requires rows > 0, vocab >= 3, and width > 0");
  if (storage != DataType::FP32 && storage != DataType::BF16)
    return absl::UnimplementedError(
        "embedding readout supports physical FP32 or BF16 activations");
  const size_t element_bytes =
      storage == DataType::BF16 ? sizeof(__nv_bfloat16) : sizeof(float);
  RETURN_IF_ERROR(ValidateMatrixPrefix(executor, activations, rows, width,
                                       element_bytes, "readout activations"));
  RETURN_IF_ERROR(ValidateMatrixPrefix(executor, fp32_embedding, logical_vocab,
                                       width, sizeof(float),
                                       "readout embedding"));
  ASSIGN_OR_RETURN(const size_t output_bytes,
                   CheckedBytes(rows, 1, sizeof(TopThreeTokens)));
  ASSIGN_OR_RETURN(auto output, cuda::Buffer::Allocate(executor, output_bytes));
  ASSIGN_OR_RETURN(
      const size_t logits_bytes,
      CheckedBytes(std::min(rows, kRowsPerChunk), logical_vocab, sizeof(float)));
  ASSIGN_OR_RETURN(auto logits, cuda::Buffer::Allocate(executor, logits_bytes));
  for (int start = 0; start < rows;) {
    const int chunk_rows = std::min(kRowsPerChunk, rows - start);
    if (storage == DataType::BF16)
      RETURN_IF_ERROR(ProjectChunk<__nv_bfloat16>(
          executor, activations, fp32_embedding, start, chunk_rows,
          logical_vocab, width, logits));
    else
      RETURN_IF_ERROR(ProjectChunk<float>(executor, activations, fp32_embedding,
                                          start, chunk_rows, logical_vocab,
                                          width, logits));
    TopThreeKernel<<<chunk_rows, kThreads, 0, executor.stream()>>>(
        static_cast<const float*>(logits.data()), logical_vocab, logical_vocab,
        static_cast<TopThreeTokens*>(output.data()) + start);
    RETURN_IF_ERROR(
        cuda::CudaStatus(cudaGetLastError(), "TopThreeKernel launch"));
    start += chunk_rows;
  }
  // The temporary's stream-ordered free follows every projection/reduction.
  return output;
}

}  // namespace pluto::llm
