#pragma once

#include <memory>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "src/cuda/executor.h"
#include "src/llm/layer.h"
#include "src/llm/layers/combinators.h"

namespace pluto::llm {

// Default dimensions of the GPT-2 recipe. Context remains fixed while
// Gpt2Config can select vocabulary, width, and depth. Existing dataset,
// loss, and inference callers can continue to use these default dimensions.
inline constexpr int kGpt2VocabularySize = 50'257;
inline constexpr int kGpt2PaddedVocabularySize = 50'272;
inline constexpr int kGpt2ContextLength = 1'024;
// The historical default also anchors residual initialization. It is not a
// ceiling on explicitly configured depth; keep it fixed in depth comparisons.
inline constexpr int kGpt2TransformerBlockCount = 8;
inline constexpr int kGpt2ModelWidth = 512;
inline constexpr int kGpt2AttentionHeads = 8;
inline constexpr int kGpt2AttentionHeadDimension = 64;
inline constexpr int kGpt2FeedForwardWidth = 2'048;

static_assert(kGpt2ModelWidth ==
              kGpt2AttentionHeads * kGpt2AttentionHeadDimension);
static_assert(kGpt2FeedForwardWidth == 4 * kGpt2ModelWidth);

// Shape choices for controlled depth/width experiments. The MLP width is
// explicit rather than implicitly four times model_width, so changing one
// dimension never silently changes another. Every attention head has width
// model_width / attention_heads. All other architectural and initialization
// choices, including the tied head and eight-block residual initialization
// scaling, remain the same as the default recipe.
struct Gpt2Config {
  int transformer_block_count = kGpt2TransformerBlockCount;
  int model_width = kGpt2ModelWidth;
  int attention_heads = kGpt2AttentionHeads;
  int feed_forward_width = kGpt2FeedForwardWidth;
  int vocabulary_size = kGpt2VocabularySize;
  // Keep historical padded embedding checkpoints by default. False stores
  // exactly vocabulary_size trainable rows; logits remain padded to 16 lanes.
  bool pad_vocabulary = true;

  // Checks shape and current CUDA-kernel limits without allocating memory.
  // Depth may be any nonnegative int; construction time and memory grow with
  // depth. Model/MLP/head widths must be positive. Compute-tile padding never
  // adds stored parameters or contributes to model statistics.
  // Parameter tensors and single-sample intermediate tensors must fit the
  // backend's 32-bit element counts. Larger batches still need to respect
  // the per-layer runtime buffer/grid limits.
  absl::Status Validate() const;
};

// Builds the token-to-activation prefix of the GPT-2 recipe. The returned
// layer applies the token and learned position embeddings followed by exactly
// transformer_block_count pre-LayerNorm transformer blocks. It deliberately
// omits the final LayerNorm and language-modeling head so its output is the
// [-2, kGpt2ContextLength, kGpt2ModelWidth] residual-stream activation that
// follows the requested block. The batch and context axes are contiguous and
// flattened into token rows by kernels. Passing zero taps the summed token and
// position embeddings.
//
// transformer_block_count must be nonnegative. The seed and initialization
// scheme match CreateGpt2(), so the returned layer shares the parameter prefix
// of a complete model created with the same arguments.
absl::StatusOr<std::unique_ptr<Layer>> CreateActivationGenerator(
    cuda::Executor& executor, int transformer_block_count, DataType output_type,
    int seed);

// Configurable counterpart. The output width is config.model_width and the
// prefix includes config.transformer_block_count blocks. Its weights match
// the corresponding prefix of CreateGpt2() with the same widths and seed,
// even if that full model has more blocks.
absl::StatusOr<std::unique_ptr<Layer>> CreateActivationGenerator(
    cuda::Executor& executor, const Gpt2Config& config, DataType output_type,
    int seed);

// Builds the GPT-2-style architecture used by the training binaries:
// learned token and position embeddings, by default eight pre-LayerNorm
// transformer blocks, a final LayerNorm, and a tied language-modeling head.
// Every layer declares the fixed context length and symbolic batch dimension
// (-2). The head returns FP32 [-2, kGpt2ContextLength,
// kGpt2PaddedVocabularySize] logits. output_type selects the compute policy;
// parameters and gradients remain FP32. transformer_block_count must be
// nonnegative; the default of eight is not a maximum. Changing it affects depth
// only: width, vocabulary, context, initialization, final norm, and tied head
// stay fixed for controlled memorization experiments.
absl::StatusOr<std::unique_ptr<ComposedLayer>> CreateGpt2(
    cuda::Executor& executor, DataType output_type, int seed,
    int transformer_block_count = kGpt2TransformerBlockCount);

// Configurable counterpart; the legacy overload above delegates here with
// the default widths and vocabulary. The logits' vocabulary dimension is
// config.vocabulary_size rounded up to 16, independently of table padding.
absl::StatusOr<std::unique_ptr<ComposedLayer>> CreateGpt2(
    cuda::Executor& executor, DataType output_type, int seed,
    const Gpt2Config& config);

}  // namespace pluto::llm
