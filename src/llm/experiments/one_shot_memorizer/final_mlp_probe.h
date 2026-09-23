#pragma once

#include <memory>
#include <vector>

#include "absl/status/statusor.h"
#include "src/cuda/executor.h"
#include "src/dataset/dataset.h"
#include "src/llm/layer.h"

namespace pluto::llm::one_shot_memorizer {

struct FinalMlpBatch {
  int width = 0;
  int feature_width = 0;
  // Only rows whose target is not -1, in original sample/position order.
  // BF16 activations are expanded to float without additional rounding.
  std::vector<float> normalized_inputs;  // [supervised_rows, width]
  std::vector<float> features;           // [supervised_rows, feature_width]
  std::vector<float> residuals;          // [supervised_rows, width], before MLP
  std::vector<int> labels;
  std::vector<int> sample_indices;  // Within this batch, not corpus-global.
  std::vector<int> positions;
  std::vector<int> original_predictions;
  // The original final, direct GPT-2 LayerNorm's FP32 parameters. Located by
  // its forward-state identity/nesting, never by flattened weight offsets.
  std::vector<float> finalnorm_gamma;
  std::vector<float> finalnorm_beta;
};

// Captures the last MLP's own LayerNorm output, fresh GELU features and input
// residual, plus labels and original predictions. Does NOT expose teacher
// branch outputs. Requires the declared BF16 GPT-2 dimensions/block count.
// All downloads use pinned memory and finish before returning.
absl::StatusOr<FinalMlpBatch> CaptureFinalMlpBatch(
    cuda::Executor& executor, const Layer& model, const DataBatch& batch,
    int width, int feature_width, int block_count, int vocabulary_size);

// Forward-only borrowed view: all four referenced layers must outlive it.
// A nonnull projection replaces the last MLP's branch output using this SAME
// forward's GELU output. A null projection retains the original MLP branch.
// The direct final norm and head are replaced using the fresh last-block
// residual and fresh replacement-normalization output, respectively.
// Input embeddings and all original parameter/activation bytes are untouched.
// Replacement layers run without hooks; caller hooks still see original GPT-2
// nesting and the substituted outputs. No backward pass is supported.
// Head signature must match the original logits, including vocabulary padding.
absl::StatusOr<std::unique_ptr<Layer>> CreateFinalMlpReplacement(
    const Layer& model, const Layer* projection, const Layer& norm,
    const Layer& head, int last_block);

}  // namespace pluto::llm::one_shot_memorizer
