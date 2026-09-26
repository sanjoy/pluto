#pragma once

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "src/cuda/buffer.h"
#include "src/cuda/executor.h"

namespace pluto::llm::fit_attention_readout {

// Copies base into independent storage, replacing the selected columns in every
// row with donor's exact BF16 bits. Dimensions must be nonempty, unique, and in
// [0, model_width); their order does not matter. Neither input is mutated; base
// and donor may share storage. Both inputs must be nonempty BF16 matrices with
// the same size and model_width columns, allocated on executor. The caller must
// align their sample/position rows. Copies are queued on executor's stream.
absl::StatusOr<cuda::Buffer> GraftBf16Dimensions(
    cuda::Executor& executor, const cuda::Buffer& base,
    const cuda::Buffer& donor, int model_width,
    absl::Span<const int> zero_based_dimensions);

// Equivalent to GraftBf16Dimensions with columns [0, dimensions).
absl::StatusOr<cuda::Buffer> GraftBf16LeadingDimensions(
    cuda::Executor& executor, const cuda::Buffer& base,
    const cuda::Buffer& donor, int model_width, int dimensions);

}  // namespace pluto::llm::fit_attention_readout
