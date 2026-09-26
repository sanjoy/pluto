#pragma once

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "src/cuda/buffer.h"
#include "src/cuda/executor.h"

namespace pluto::llm::fit_attention_readout {

// Replaces each base row's component in an orthonormal plane with the aligned
// donor row's component: output = base + U U^T (donor - base). A shared PCA
// center cancels in this difference, so no center is required here.
//
// plane contains model_width * 2 finite FP32 values in row-major order
// [u_0,0, u_0,1, u_1,0, u_1,1, ...]. Its columns must be orthonormal to 1e-5.
// Widths 2 through 1024 are supported. Inputs must be equal-sized, nonempty,
// contiguous BF16 matrices allocated on executor; the caller aligns rows.
// The returned storage is independent and neither input is modified. All
// arithmetic uses FP32 before rounding the output to BF16. Work and temporary
// storage lifetimes are ordered on executor's stream without a compute wait.
absl::StatusOr<cuda::Buffer> GraftBf16Plane(cuda::Executor& executor,
                                            const cuda::Buffer& base,
                                            const cuda::Buffer& donor,
                                            int model_width,
                                            absl::Span<const float> plane);

}  // namespace pluto::llm::fit_attention_readout
