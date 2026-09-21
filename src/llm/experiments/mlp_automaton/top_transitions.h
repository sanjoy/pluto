#pragma once

#include <cstddef>
#include <type_traits>

#include "absl/status/statusor.h"
#include "src/cuda/buffer.h"
#include "src/cuda/executor.h"

namespace pluto::llm::mlp_automaton {

// One temperature-one softmax winner, with ties resolved by lower token ID.
// Any nonfinite logical logit makes the row invalid: token=-1, probability=NaN.
struct TopTransition {
  int token;
  float probability;
};

static_assert(std::is_standard_layout_v<TopTransition>);
static_assert(std::is_trivially_copyable_v<TopTransition>);
// The cuTile writer uses separate strided views into these packed fields.
static_assert(sizeof(int) == sizeof(float));
static_assert(offsetof(TopTransition, token) == 0);
static_assert(offsetof(TopTransition, probability) == sizeof(int));
static_assert(sizeof(TopTransition) == 8);

// Enqueues a read-only reduction of an exact [rows,padded_vocab] FP32 buffer.
// Only columns [0,logical_vocab) participate; padding may contain any values.
// All dimensions must be positive, and logical_vocab must not exceed padding.
// The returned device buffer contains exactly rows TopTransition records, and
// belongs to executor. No vocabulary-sized probability buffer is allocated.
// Synchronize the executor after downloading before reading the host records.
absl::StatusOr<cuda::Buffer> ReadTopTransitions(cuda::Executor& executor,
                                                const cuda::Buffer& fp32_logits,
                                                int rows, int logical_vocab,
                                                int padded_vocab);

}  // namespace pluto::llm::mlp_automaton
