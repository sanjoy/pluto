#pragma once

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "src/cuda/executor.h"
#include "src/cuda/page_locked_host_array.h"
#include "src/llm/layer.h"

namespace pluto::llm {

// Generates a batch-one greedy continuation, resolving ties by lowest token
// ID. The model must accept INT32 [batch, context] and return FP32
// [batch, context, padded_vocabulary]. Only logical vocabulary IDs participate
// in selection. Nonfinite logical logits on the selected row return DataLoss.
//
// Returns newly generated tokens only, excluding EOS. Stops at EOS, the
// requested count, or a full context; positions never shift and there is no KV
// cache. Prompts must be nonempty and fit the context. A zero count or an
// already-full context returns an empty continuation after input validation.
// The returned pinned storage is CPU-ready and must not outlive executor.
absl::StatusOr<cuda::PageLockedHostArray<int>> GenerateGreedyContinuation(
    cuda::Executor& executor, const Layer& model,
    absl::Span<const int> prompt_tokens, int vocabulary_size, int eos_token,
    int max_new_tokens);

}  // namespace pluto::llm
