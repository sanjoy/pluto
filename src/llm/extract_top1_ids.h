#pragma once

#include "absl/status/statusor.h"
#include "src/cuda/buffer.h"
#include "src/cuda/executor.h"

namespace pluto::llm {

// Extracts one exact top-1 token ID from each FP32 [rows, padded_vocabulary]
// logits row. The lowest ID wins ties, and vocabulary padding cannot win.
// Targets are int32 [rows]: -1 skips a row without reading its logits and
// produces -1; every other target value enables the row. Any nonfinite logit
// inside the logical vocabulary produces -2 instead of a predicted token ID.
//
// Both inputs must belong to executor. The returned int32 [rows] device buffer
// is ordered on that same executor; no device-to-host copy or wait is needed.
absl::StatusOr<cuda::Buffer> ExtractTop1Ids(cuda::Executor& executor,
                                            const cuda::Buffer& logits,
                                            const cuda::Buffer& targets,
                                            int vocabulary_size);

}  // namespace pluto::llm
