#pragma once

#include <memory>

#include "absl/status/statusor.h"
#include "src/llm/gpt2.h"
#include "src/llm/layer.h"
#include "src/llm/layers/combinators.h"
#include "src/llm/layers/embedding.h"

namespace pluto::llm::memorize_general_facts {

// The head is tied to an independent, frozen copy of the source embedding.
// Its owner precedes model so the embedding outlives the head. Source need not
// outlive this object; none of its buffers is changed or shared with training.
struct PuzzleReadout {
  std::unique_ptr<EmbeddingLookupLayer> embedding;
  std::unique_ptr<ComposedLayer> model;
  // Owned by model. Only this layer may be passed to the optimizer/checkpoint:
  // input LN, FC1, FC2, final LN (eight FP32 tensors, in that order).
  Layer* trainable;
};

// Builds head(final_LN(x + FC2(GELU(FC1(input_LN(x)))))). Input LN starts
// at identity, biases at zero, FC1/FC2 at normal stddev .2/.1 using
// seed/seed+1. Final LN is an independent, trainable copy of the source final
// LN. The head follows the eight trainable tensors in model.weights(). For
// model width 10, there are exactly 21 * mlp_width + 50 trainable parameters.
absl::StatusOr<PuzzleReadout> CreatePuzzleReadout(cuda::Executor& executor,
                                                  const Layer& source,
                                                  const Gpt2Config& config,
                                                  int mlp_width, int seed);

struct PuzzleCapture {
  Buffer hidden;  // BF16 post-attention residual in transformer_block_2.
  Buffer logits;  // FP32 full-source logits, for baseline verification only.
};

// Runs the source with integer token sequences, retaining the FIRST residual
// output of block 2, before its MLP. Later blocks run only to provide logits;
// their activations never enter hidden. Saved backward state is discarded.
absl::StatusOr<PuzzleCapture> CaptureThirdAttention(cuda::Executor& executor,
                                                    const Layer& source,
                                                    const Buffer& tokens);

}  // namespace pluto::llm::memorize_general_facts
