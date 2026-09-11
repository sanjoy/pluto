#include "src/llm/recipes/mlp_automaton/softmax.h"

#include <cuda_runtime.h>
#include <math_constants.h>

#include <cstddef>
#include <limits>
#include <utility>

#include "absl/status/status.h"

namespace pluto::llm::mlp_automaton {
namespace {

constexpr int kThreads = 256;

// One block owns a row. The two coalesced passes retain only reduction scratch
// and the winner, rather than materializing a second [rows,vocabulary] array.
__global__ void TopTransitionsKernel(const float* __restrict__ logits,
                                     int logical_vocab, int padded_vocab,
                                     TopTransition* __restrict__ output) {
  __shared__ float values[kThreads];
  __shared__ int tokens[kThreads];
  __shared__ int invalid[kThreads];
  const int lane = threadIdx.x;
  const float* row = logits + static_cast<size_t>(blockIdx.x) * padded_vocab;
  float maximum = -CUDART_INF_F;
  int token = 0x7fffffff;
  int nonfinite = 0;
  for (size_t column = lane; column < static_cast<size_t>(logical_vocab);
       column += kThreads) {
    const float value = row[column];
    if (!isfinite(value)) {
      nonfinite = 1;
    } else if (value > maximum ||
               (value == maximum && static_cast<int>(column) < token)) {
      maximum = value;
      token = static_cast<int>(column);
    }
  }
  values[lane] = maximum;
  tokens[lane] = token;
  invalid[lane] = nonfinite;
  __syncthreads();
  for (int offset = kThreads / 2; offset > 0; offset /= 2) {
    if (lane < offset) {
      const float other = values[lane + offset];
      const int other_token = tokens[lane + offset];
      if (other > values[lane] ||
          (other == values[lane] && other_token < tokens[lane])) {
        values[lane] = other;
        tokens[lane] = other_token;
      }
      invalid[lane] |= invalid[lane + offset];
    }
    __syncthreads();
  }
  if (invalid[0]) {
    if (lane == 0)
      output[blockIdx.x] = {-1, CUDART_NAN_F};
    return;
  }

  maximum = values[0];
  token = tokens[0];
  // Every thread must consume the maximum before shared scratch is reused.
  __syncthreads();
  float denominator = 0.0f;
  for (size_t column = lane; column < static_cast<size_t>(logical_vocab);
       column += kThreads) {
    denominator += expf(row[column] - maximum);
  }
  values[lane] = denominator;
  __syncthreads();
  for (int offset = kThreads / 2; offset > 0; offset /= 2) {
    if (lane < offset)
      values[lane] += values[lane + offset];
    __syncthreads();
  }
  // At the maximum the unnormalized probability is exactly exp(0)=1.
  if (lane == 0)
    output[blockIdx.x] = {token, 1.0f / values[0]};
}

}  // namespace

absl::StatusOr<cuda::Buffer> ReadTopTransitions(cuda::Executor& executor,
                                                const cuda::Buffer& fp32_logits,
                                                int rows, int logical_vocab,
                                                int padded_vocab) {
  if (rows <= 0 || logical_vocab <= 0 || padded_vocab < logical_vocab) {
    return absl::InvalidArgumentError(
        "top transitions require positive rows and 0 < logical_vocab <= "
        "padded_vocab");
  }
  if (&fp32_logits.executor() != &executor) {
    return absl::InvalidArgumentError(
        "top transitions input belongs to a different executor");
  }
  const size_t row_bytes = static_cast<size_t>(padded_vocab) * sizeof(float);
  if (static_cast<size_t>(padded_vocab) >
          std::numeric_limits<size_t>::max() / sizeof(float) ||
      static_cast<size_t>(rows) >
          std::numeric_limits<size_t>::max() / row_bytes ||
      static_cast<size_t>(rows) >
          std::numeric_limits<size_t>::max() / sizeof(TopTransition)) {
    return absl::InvalidArgumentError("top transitions byte size overflows");
  }
  if (fp32_logits.size_bytes() != static_cast<size_t>(rows) * row_bytes) {
    return absl::InvalidArgumentError(
        "top transitions require an exact rows*padded_vocab FP32 buffer");
  }
  auto output = cuda::Buffer::Allocate(
      executor, static_cast<size_t>(rows) * sizeof(TopTransition));
  if (!output.ok())
    return output.status();
  TopTransitionsKernel<<<rows, kThreads, 0, executor.stream()>>>(
      static_cast<const float*>(fp32_logits.data()), logical_vocab,
      padded_vocab, static_cast<TopTransition*>(output->data()));
  auto status =
      cuda::CudaStatus(cudaGetLastError(), "TopTransitionsKernel launch");
  if (!status.ok())
    return status;
  return std::move(*output);
}

}  // namespace pluto::llm::mlp_automaton
