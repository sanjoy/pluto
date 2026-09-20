#include "src/llm/experiments/memorize_general_facts/predictions.h"

#include <cuda_runtime.h>
#include <math_constants.h>

#include <cstddef>
#include <limits>

#include "absl/status/status.h"
#include "src/util/status_macros.h"

namespace pluto::llm::memorize_general_facts {
namespace {

constexpr int kThreads = 256;

// One block owns a token row. The fixed reduction tree and explicit tie rule
// make the result independent of thread/block scheduling. Unsupervised rows
// exit immediately, which matters for this mostly-padded corpus.
__global__ void PredictKernel(const float* logits, const int* targets,
                              int vocabulary_size, int stride, int* output) {
  const int lane = threadIdx.x;
  const int row = blockIdx.x;
  if (targets[row] == -1) {
    if (lane == 0) output[row] = -1;
    return;
  }
  __shared__ float values[kThreads];
  __shared__ int ids[kThreads];
  float best = -CUDART_INF_F;
  int best_id = 0x7fffffff;
  bool invalid = false;
  for (int token = lane; token < vocabulary_size; token += kThreads) {
    const float value = logits[static_cast<size_t>(row) * stride + token];
    invalid |= !isfinite(value);
    if (value > best || (value == best && token < best_id)) {
      best = value;
      best_id = token;
    }
  }
  if (__syncthreads_or(invalid)) {
    if (lane == 0) output[row] = -2;
    return;
  }
  values[lane] = best;
  ids[lane] = best_id;
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
  if (lane == 0) output[row] = ids[0];
}

}  // namespace

absl::StatusOr<cuda::Buffer> PredictMaskedTokens(cuda::Executor& executor,
                                                 const cuda::Buffer& logits,
                                                 const cuda::Buffer& targets,
                                                 int vocabulary_size) {
  if (&logits.executor() != &executor || &targets.executor() != &executor)
    return absl::InvalidArgumentError("predictions require one executor");
  if (vocabulary_size <= 0 || targets.size_bytes() == 0 ||
      targets.size_bytes() % sizeof(int) != 0)
    return absl::InvalidArgumentError("invalid vocabulary or target shape");
  const size_t rows = targets.size_bytes() / sizeof(int);
  if (logits.size_bytes() % rows != 0 ||
      logits.size_bytes() / rows % sizeof(float) != 0)
    return absl::InvalidArgumentError("logits must contain whole FP32 rows");
  const size_t stride = logits.size_bytes() / rows / sizeof(float);
  if (stride < static_cast<size_t>(vocabulary_size) ||
      rows > std::numeric_limits<int>::max() ||
      stride > std::numeric_limits<int>::max())
    return absl::InvalidArgumentError("invalid prediction matrix dimensions");
  ASSIGN_OR_RETURN(auto result,
                   cuda::Buffer::Allocate(executor, targets.size_bytes()));
  PredictKernel<<<rows, kThreads, 0, executor.stream()>>>(
      static_cast<const float*>(logits.data()),
      static_cast<const int*>(targets.data()), vocabulary_size, stride,
      static_cast<int*>(result.data()));
  RETURN_IF_ERROR(cuda::CudaStatus(cudaGetLastError(), "PredictKernel"));
  return result;
}

}  // namespace pluto::llm::memorize_general_facts
