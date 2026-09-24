#pragma once

#include <cstdint>
#include <optional>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "src/cuda/buffer.h"
#include "src/cuda/executor.h"

namespace pluto::llm {

// A token order maps canonical vocabulary positions to the token IDs currently
// stored in logits/embedding rows: token_order[canonical_id] = physical_id.
// It changes reduction traversal, not the meaning/layout of layer inputs or
// outputs. Empty means ordinary ID order. Otherwise it must be a permutation
// of [0, vocabulary_size), without entries for padding lanes. Sequence
// positions and hidden dimensions are not vocabulary IDs and must not be
// permuted.
absl::Status ValidateTokenOrder(int vocabulary_size,
                                absl::Span<const int32_t> token_order);

// Validates and owns a device copy of a host order. The caller can immediately
// release or mutate its array. Uploads use pinned staging memory with an
// executor-ordered lifetime; no stream synchronization is needed. Empty and
// explicit identity orders return nullopt, preserving the existing fast path.
absl::StatusOr<std::optional<cuda::Buffer>> CopyTokenOrderToDevice(
    cuda::Executor& executor, int vocabulary_size,
    absl::Span<const int32_t> token_order);

}  // namespace pluto::llm
