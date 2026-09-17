#pragma once

#include <cstddef>

#include "absl/status/status.h"
#include "src/cuda/buffer.h"

namespace pluto::llm::path_kernel::internal {

// Update one unique parameter block using the final loss_rows Jacobian rows
// starting at first_loss_row. Sum examples in fixed order using FP64, apply
// theta -= learning_rate/ loss_rows * sum(gradient), and round once to FP32.
// No atomics combine examples. A nonfinite input/update is an error; caller
// owns rollback because some parameters may have been updated on failure.
absl::Status ApplyGradientDescent(cuda::Executor& executor,
                                  const cuda::Buffer& weights,
                                  const cuda::Buffer& jacobian,
                                  size_t parameter_count, size_t block_offset,
                                  size_t first_loss_row, size_t loss_rows,
                                  double learning_rate);

}  // namespace pluto::llm::path_kernel::internal
