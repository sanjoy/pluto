#include "src/llm/experiments/one_shot_memorizer/sentence_ablation_training.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <limits>

#include "absl/status/status.h"
#include "src/util/status_macros.h"

namespace pluto::llm::one_shot_memorizer {

absl::StatusOr<int> ZeroSentenceContribution(
    cuda::Executor& executor, cuda::Buffer& logits_gradient,
    absl::Span<const size_t> batch_sample_indices, size_t omitted_sentence,
    int sequence_length, int vocabulary_width) {
  if (&logits_gradient.executor() != &executor)
    return absl::InvalidArgumentError(
        "sentence ablation gradient belongs to another executor");
  if (sequence_length <= 0 || vocabulary_width <= 0)
    return absl::InvalidArgumentError(
        "sentence ablation sequence and vocabulary dimensions must be "
        "positive");
  const size_t sequence = sequence_length, vocabulary = vocabulary_width;
  if (sequence > std::numeric_limits<size_t>::max() / vocabulary ||
      sequence * vocabulary >
          std::numeric_limits<size_t>::max() / sizeof(float))
    return absl::OutOfRangeError(
        "sentence ablation sample byte size overflows");
  const size_t sample_bytes = sequence * vocabulary * sizeof(float);
  if (batch_sample_indices.size() >
          static_cast<size_t>(std::numeric_limits<int>::max()) ||
      batch_sample_indices.size() >
          std::numeric_limits<size_t>::max() / sample_bytes)
    return absl::OutOfRangeError("sentence ablation batch byte size overflows");
  if (logits_gradient.size_bytes() !=
      batch_sample_indices.size() * sample_bytes)
    return absl::InvalidArgumentError(
        "sentence ablation gradient must be FP32 [batch, sequence, "
        "vocabulary]");

  int masked_slots = 0;
  auto* bytes = static_cast<unsigned char*>(logits_gradient.data());
  for (size_t sample = 0; sample < batch_sample_indices.size(); ++sample) {
    if (batch_sample_indices[sample] != omitted_sentence)
      continue;
    RETURN_IF_ERROR(
        cuda::CudaStatus(cudaMemsetAsync(bytes + sample * sample_bytes, 0,
                                         sample_bytes, executor.stream()),
                         "cudaMemsetAsync(omitted sentence gradient)"));
    ++masked_slots;
  }
  return masked_slots;
}

absl::StatusOr<float> AblationLearningRate(int step, double peak_learning_rate,
                                           int warmup_steps,
                                           int schedule_horizon) {
  if (step <= 0 || !std::isfinite(peak_learning_rate) ||
      peak_learning_rate <= 0 || warmup_steps < 0 || schedule_horizon <= 0 ||
      warmup_steps >= schedule_horizon)
    return absl::InvalidArgumentError(
        "invalid ablation learning-rate schedule");
  double rate;
  if (step <= warmup_steps) {
    // The positive step and this comparison imply warmup_steps > 0.
    rate = peak_learning_rate * step / warmup_steps;
  } else {
    const double progress =
        std::clamp(static_cast<double>(step - warmup_steps) /
                       (schedule_horizon - warmup_steps),
                   0.0, 1.0);
    rate = peak_learning_rate *
           (0.1 + 0.9 * (1 + std::cos(3.14159265358979323846 * progress)) / 2);
  }
  const float stored = static_cast<float>(rate);
  if (!std::isfinite(stored) || stored <= 0)
    return absl::OutOfRangeError(
        "ablation learning rate must fit a positive finite float");
  return stored;
}

}  // namespace pluto::llm::one_shot_memorizer
