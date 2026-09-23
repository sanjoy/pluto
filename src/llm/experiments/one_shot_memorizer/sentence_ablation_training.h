#pragma once

#include <cstddef>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "src/cuda/buffer.h"
#include "src/cuda/executor.h"

namespace pluto::llm::one_shot_memorizer {

// Erases the selected sentence's contribution AFTER cross-entropy backward.
// logits_gradient is its owned FP32 [batch, sequence_length, vocabulary_width]
// output; batch_sample_indices identifies the corpus sentence in each slot.
// vocabulary_width is the physical PADDED width reported by the loss layer.
// CE has already normalized by the ORIGINAL batch's supervised-target count.
// Zeroing here therefore preserves every other sample's scaling and the batch
// schedule, unlike marking omitted targets ignored before computing the loss.
// Repeated occurrences are all zeroed; an absent sentence leaves every byte
// unchanged. Returns the number of zeroed batch slots.
//
// Queues only cudaMemsetAsync on the buffer's supplied executor: no allocation,
// synchronization, host copy, or custom kernel. All validation precedes writes.
absl::StatusOr<int> ZeroSentenceContribution(
    cuda::Executor& executor, cuda::Buffer& logits_gradient,
    absl::Span<const size_t> batch_sample_indices, size_t omitted_sentence,
    int sequence_length, int vocabulary_width);

// One-based linear warmup followed by cosine decay to 10% of peak, matching the
// memorization experiment. schedule_horizon is the ORIGINAL training budget
// (for example 40,000), not the shorter ablation pilot's number of updates.
// Requires 0 <= warmup_steps < schedule_horizon, positive step and finite peak;
// steps beyond the horizon retain the 10% floor. The result must fit a positive
// finite float so it can be passed directly to AdamW's learning-rate setter.
absl::StatusOr<float> AblationLearningRate(int step, double peak_learning_rate,
                                           int warmup_steps,
                                           int schedule_horizon);

}  // namespace pluto::llm::one_shot_memorizer
