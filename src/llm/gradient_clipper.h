#pragma once

#include <memory>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "src/cuda/buffer.h"
#include "src/cuda/executor.h"
#include "src/llm/layer.h"

namespace pluto::llm {

// Clips the joint L2 norm of a model's unique FP32 gradient allocations. Call
// Clip() after backward and before the optimizer's ApplyStep(). Tied gradients
// contribute once to the norm and are scaled once; weights are never modified.
//
// Create() allocates all scratch storage and builds a device list of gradient
// chunks. Clip() then queues three fixed-order CUDA kernels without allocating,
// synchronizing, or copying a result to the host. Reduction ordering is fixed,
// not determined by atomic scheduling. The clipping scale is
// min(1, max_norm / (global_norm + 1e-6)).
//
// The model must not replace its gradient allocations while this object is in
// use. Handles keep those allocations alive, but the executor must outlive the
// clipper. Like the optimizer, use only on the owning training thread.
class GradientClipper final {
 public:
  static absl::StatusOr<std::unique_ptr<GradientClipper>> Create(
      cuda::Executor& executor, Layer& model, float max_norm = 1.0f);

  absl::Status Clip();

 private:
  GradientClipper(cuda::Executor& executor, float max_norm,
                  std::vector<cuda::Buffer> gradients, cuda::Buffer chunks,
                  cuda::Buffer partial_squared_norms, cuda::Buffer scale,
                  int chunk_count)
      : executor_(executor),
        max_norm_(max_norm),
        gradients_(std::move(gradients)),
        chunks_(std::move(chunks)),
        partial_squared_norms_(std::move(partial_squared_norms)),
        scale_(std::move(scale)),
        chunk_count_(chunk_count) {}

  cuda::Executor& executor_;
  float max_norm_;
  std::vector<cuda::Buffer> gradients_;
  cuda::Buffer chunks_;
  cuda::Buffer partial_squared_norms_;
  cuda::Buffer scale_;
  int chunk_count_;
};

}  // namespace pluto::llm
