#include "src/llm/experiments/weight_sensitivity/runner.h"

#include <cuda_runtime.h>

#include <utility>

#include "src/util/combine_statuses.h"
#include "src/util/status_macros.h"

namespace pluto::llm::weight_sensitivity {

absl::StatusOr<CorruptionResult> EvaluateCorruption(
    cuda::Executor& executor, const cuda::Buffer& weight,
    const WeightTarget& target,
    const cuda::PageLockedHostArray<float>& original, uint64_t seed,
    double noise_scale, double zero_rms_stddev,
    absl::FunctionRef<absl::StatusOr<CompletionScores>()> evaluate) {
  if (&weight.executor() != &executor || &original.executor() != &executor ||
      weight.size_bytes() != original.size_bytes())
    return absl::InvalidArgumentError("original tensor/executor mismatch");
  ASSIGN_OR_RETURN(auto corrupted, cuda::PageLockedHostArray<float>::Allocate(
                                       executor, original.size()));
  ASSIGN_OR_RETURN(double standard_deviation,
                   ReplaceWithNoise(target, original.span(), corrupted.span(),
                                    seed, noise_scale, zero_rms_stddev));
  auto outcome = [&]() -> absl::StatusOr<CompletionScores> {
    RETURN_IF_ERROR(cuda::CudaStatus(
        cudaMemcpyAsync(weight.data(), corrupted.data(), weight.size_bytes(),
                        cudaMemcpyHostToDevice, executor.stream()),
        "upload corrupted weight"));
    return evaluate();
  }();
  // Do not RETURN_IF_ERROR before restoration: an unsuccessful evaluation
  // must not leave noise in the model and contaminate later interventions.
  auto restore = cuda::CudaStatus(
      cudaMemcpyAsync(weight.data(), original.data(), weight.size_bytes(),
                      cudaMemcpyHostToDevice, executor.stream()),
      "restore original weight");
  auto ready = executor.Synchronize();
  RETURN_IF_ERROR(util::CombineStatuses({outcome.status(), restore, ready}));
  return CorruptionResult{std::move(*outcome), standard_deviation};
}

}  // namespace pluto::llm::weight_sensitivity
