#pragma once

#include <cstdint>

#include "absl/functional/function_ref.h"
#include "absl/status/statusor.h"
#include "src/cuda/buffer.h"
#include "src/cuda/executor.h"
#include "src/cuda/page_locked_host_array.h"
#include "src/llm/experiments/weight_sensitivity/evaluation.h"
#include "src/llm/experiments/weight_sensitivity/weights.h"

namespace pluto::llm::weight_sensitivity {

struct CorruptionResult {
  CompletionScores scores;
  double noise_stddev = 0;
};

// Copies only through pinned host storage. Original is the complete physical
// tensor, even when target selects a strided Q/K/V slice. Restores that tensor
// and waits before returning, including when uploading or evaluation fails.
// Original must already be ready for CPU reads (synchronize its download
// first). The callback must use this executor and must not mutate other
// parameters.
absl::StatusOr<CorruptionResult> EvaluateCorruption(
    cuda::Executor& executor, const cuda::Buffer& weight,
    const WeightTarget& target,
    const cuda::PageLockedHostArray<float>& original, uint64_t seed,
    double noise_scale, double zero_rms_stddev,
    absl::FunctionRef<absl::StatusOr<CompletionScores>()> evaluate);

}  // namespace pluto::llm::weight_sensitivity
