#pragma once

#include <memory>

#include "absl/status/statusor.h"
#include "src/cuda/executor.h"
#include "src/llm/layer.h"
#include "src/llm/layers/combinators.h"

namespace pluto::llm {

// Public dimensions of the GPT-2 recipe. Keeping these beside CreateGpt2()
// gives dataset, loss, and inference code one source of truth for tensor
// shapes without coupling those callers to the recipe implementation.
inline constexpr int kGpt2VocabularySize = 50'257;
inline constexpr int kGpt2PaddedVocabularySize = 50'272;
inline constexpr int kGpt2ContextLength = 1'024;
inline constexpr int kGpt2TransformerBlockCount = 8;
inline constexpr int kGpt2ModelWidth = 512;
inline constexpr int kGpt2AttentionHeads = 8;
inline constexpr int kGpt2AttentionHeadDimension = 64;
inline constexpr int kGpt2FeedForwardWidth = 2'048;

static_assert(kGpt2ModelWidth ==
              kGpt2AttentionHeads * kGpt2AttentionHeadDimension);
static_assert(kGpt2FeedForwardWidth == 4 * kGpt2ModelWidth);

// Builds the token-to-activation prefix of the GPT-2 recipe. The returned
// layer applies the token and learned position embeddings followed by exactly
// transformer_block_count pre-LayerNorm transformer blocks. It deliberately
// omits the final LayerNorm and language-modeling head so its output is the
// [token_count, kGpt2ModelWidth] residual-stream activation that follows the
// requested block. Passing zero taps the summed token and position embeddings.
//
// transformer_block_count must be in [0, kGpt2TransformerBlockCount]. The seed
// and initialization scheme match CreateGpt2(), so the returned layer has the
// same parameter prefix as a complete model created with the same arguments.
absl::StatusOr<std::unique_ptr<Layer>> CreateActivationGenerator(
    cuda::Executor& executor, int transformer_block_count, DataType output_type,
    int seed);

// Builds the fixed GPT-2-style architecture used by the training binaries:
// learned token and position embeddings, eight pre-LayerNorm transformer
// blocks, a final LayerNorm, and a tied language-modeling head. Activations use
// output_type while parameters and gradients remain FP32.
absl::StatusOr<std::unique_ptr<ComposedLayer>> CreateGpt2(
    cuda::Executor& executor, DataType output_type, int seed);

}  // namespace pluto::llm
