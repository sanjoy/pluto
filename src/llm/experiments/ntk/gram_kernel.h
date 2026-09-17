#pragma once

#include <cstddef>

#include "absl/status/statusor.h"
#include "src/cuda/buffer.h"
#include "src/cuda/executor.h"
#include "src/llm/experiments/ntk/kernel_regression.h"

namespace pluto::llm::ntk::internal {

// Computes J J^T from an FP32 device Jacobian stored row-major with exactly
// rows * parameters entries on executor. Requires 1 <= rows <= 65535 and a
// positive parameter count. Products and reductions use FP64, a fixed tree,
// and no floating-point atomics; symmetry is exact because each pair is
// computed once and copied to both entries. Repeatability is for the same
// hardware/build, not an across-architecture floating-point guarantee.
//
// Synchronizes executor after a page-locked D2H copy and returns a finite CPU
// matrix. Scratch uses O(rows^2 * ceil(parameters / 4096)) doubles. Rejects
// overflowing dimensions, mismatched storage/executor, and CUDA grid overflow.
absl::StatusOr<Matrix> ComputeGram(cuda::Executor& executor,
                                   const cuda::Buffer& jacobian, size_t rows,
                                   size_t parameters);

}  // namespace pluto::llm::ntk::internal
