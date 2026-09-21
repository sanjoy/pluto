#pragma once

#include <functional>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "src/cuda/executor.h"
#include "src/cuda/page_locked_host_array.h"
#include "src/llm/layer.h"

namespace pluto::llm {

// Optional instrumentation, borrowed for one generation call. Hooks observe
// every forward pass (including the pass that selects EOS). on_token runs only
// after selecting a valid, non-EOS token, with the unpadded input prefix that
// produced it; the new token is not yet included in that prefix. Thus its
// zero-based output position is input_prefix.size(). The prefix is borrowed
// only for the callback. Callback errors stop generation and propagate.
struct GreedyGenerationOptions {
  LayerHooks* layer_hooks = nullptr;
  std::function<absl::Status(cuda::Executor& executor,
                             absl::Span<const int> input_prefix, int token)>
      on_token;
};

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
    int max_new_tokens, const GreedyGenerationOptions& options = {});

}  // namespace pluto::llm
