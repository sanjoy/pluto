#pragma once

#include "absl/status/statusor.h"
#include "src/cuda/buffer.h"
#include "src/cuda/executor.h"

namespace pluto::llm::memorize_general_facts {

// Exact top-1 IDs for FP32 [rows, padded_vocabulary] logits, with lowest-ID
// tie breaking. Targets are int32 [rows]; -1 skips a row without reading its
// logits (prompt/padding). Padded vocabulary columns cannot win. Nonfinite
// logical logits produce -2, which the evaluator treats as a numerical error.
// Returns a device int32 array ordered on executor, without synchronization.
absl::StatusOr<cuda::Buffer> PredictMaskedTokens(cuda::Executor& executor,
                                                 const cuda::Buffer& logits,
                                                 const cuda::Buffer& targets,
                                                 int vocabulary_size);

}  // namespace pluto::llm::memorize_general_facts
