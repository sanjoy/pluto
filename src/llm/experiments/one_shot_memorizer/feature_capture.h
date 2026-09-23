#pragma once

#include <vector>

#include "absl/status/statusor.h"
#include "src/cuda/executor.h"
#include "src/dataset/dataset.h"
#include "src/llm/layer.h"

namespace pluto::llm::one_shot_memorizer {

// Only supervised rows, in sample/position order. These features already
// depend on the trained backbone; reconstructing their readout is conditional
// decoding, NOT an end-to-end construction of the trained model from data.
struct ReadoutFeatures {
  int width = 0;
  std::vector<float> values;  // [labels.size(), width], expanded from BF16.
  std::vector<int> labels;
  std::vector<int> original_predictions;
  std::vector<int> sample_indices;  // Indices within the supplied batch.
  std::vector<int> positions;       // Query positions, before the target.
};

// Captures the direct final LayerNorm child of "gpt2", not any block's norm.
// Requires BF16 features and FP32 logits. Performs one unmodified forward and
// downloads only features, targets, and top-1 IDs, using pinned staging memory.
// Targets -1 exclude prompt/padding rows. No parameter or activation is
// changed.
absl::StatusOr<ReadoutFeatures> CaptureGpt2ReadoutBatch(
    cuda::Executor& executor, const Layer& model, const DataBatch& batch,
    int width, int vocabulary_size);

// Downloads the FP32 tied embedding and rounds it to the BF16 values actually
// multiplied by the BF16 language-modeling head. Returns only logical rows.
absl::StatusOr<std::vector<float>> CopyEffectiveBf16Readout(
    cuda::Executor& executor, const Buffer& embedding, int width,
    int vocabulary_size);

}  // namespace pluto::llm::one_shot_memorizer
