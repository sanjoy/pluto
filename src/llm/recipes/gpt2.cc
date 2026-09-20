#include "src/llm/recipes/gpt2.h"

#include <cmath>
#include <cstdint>
#include <memory>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "src/cuda/executor.h"
#include "src/llm/layer.h"
#include "src/llm/layers/attention.h"
#include "src/llm/layers/combinators.h"
#include "src/llm/layers/embedding.h"
#include "src/llm/layers/fully_connected.h"
#include "src/llm/layers/gelu.h"
#include "src/llm/layers/norm.h"
#include "src/util/status_macros.h"

namespace pluto::llm {
namespace {

constexpr float kLayerNormEpsilon = 1e-5f;
constexpr float kInitializationStandardDeviation = 0.02f;

// Builds one pre-LayerNorm GPT-2 transformer block:
//
//   x = x + W_o CausalMHA(W_qkv LayerNorm(x))
//   x = x + W_2 GELU(W_1 LayerNorm(x))
//
// W_qkv maps the model width to three independent Q/K/V tensors. CausalMHA
// uses online FP32 softmax statistics. The MLP expands the representation by
// four. Dropout and attention dropout are exactly zero, so no dropout layers
// appear. Every block receives independently initialized parameters.
absl::StatusOr<std::unique_ptr<ComposedLayer>> CreateTransformerBlock(
    cuda::Executor& executor, DataType output_type, int initialization_seed,
    int block_index) {
  const float residual_standard_deviation =
      kInitializationStandardDeviation /
      std::sqrt(2.0f * kGpt2TransformerBlockCount);
  const uint64_t seed_base =
      static_cast<uint64_t>(static_cast<uint32_t>(initialization_seed)) +
      1'000 + static_cast<uint64_t>(block_index) * 100;

  ComposedLayerBuilder attention_builder;
  RETURN_IF_ERROR(attention_builder.add(
      LayerNormLayer::Create(executor, kGpt2ModelWidth, kLayerNormEpsilon,
                             output_type, kGpt2ContextLength)));
  RETURN_IF_ERROR(attention_builder.add(FullyConnectedLayer::Create(
      executor, kGpt2ModelWidth, 3 * kGpt2ModelWidth, output_type,
      kGpt2ContextLength)));
  auto* qkv_projection =
      static_cast<FullyConnectedLayer*>(attention_builder.back());
  RETURN_IF_ERROR(qkv_projection->InitializeNormal(
      kInitializationStandardDeviation, seed_base + 1));
  RETURN_IF_ERROR(attention_builder.add(
      AttentionLayer::Create(executor, kGpt2ContextLength, kGpt2AttentionHeads,
                             kGpt2ModelWidth, output_type)));
  RETURN_IF_ERROR(attention_builder.add(
      FullyConnectedLayer::Create(executor, kGpt2ModelWidth, kGpt2ModelWidth,
                                  output_type, kGpt2ContextLength)));
  auto* attention_projection =
      static_cast<FullyConnectedLayer*>(attention_builder.back());
  RETURN_IF_ERROR(attention_projection->InitializeNormal(
      residual_standard_deviation, seed_base + 2));

  ComposedLayerBuilder mlp_builder;
  RETURN_IF_ERROR(mlp_builder.add(
      LayerNormLayer::Create(executor, kGpt2ModelWidth, kLayerNormEpsilon,
                             output_type, kGpt2ContextLength)));
  RETURN_IF_ERROR(mlp_builder.add(FullyConnectedLayer::Create(
      executor, kGpt2ModelWidth, kGpt2FeedForwardWidth, output_type,
      kGpt2ContextLength)));
  auto* mlp_input = static_cast<FullyConnectedLayer*>(mlp_builder.back());
  RETURN_IF_ERROR(mlp_input->InitializeNormal(kInitializationStandardDeviation,
                                              seed_base + 3));
  RETURN_IF_ERROR(mlp_builder.add(GeluLayer::Create(
      executor, kGpt2FeedForwardWidth, output_type, kGpt2ContextLength)));
  RETURN_IF_ERROR(mlp_builder.add(FullyConnectedLayer::Create(
      executor, kGpt2FeedForwardWidth, kGpt2ModelWidth, output_type,
      kGpt2ContextLength)));
  auto* mlp_output = static_cast<FullyConnectedLayer*>(mlp_builder.back());
  RETURN_IF_ERROR(
      mlp_output->InitializeNormal(residual_standard_deviation, seed_base + 4));

  ASSIGN_OR_RETURN(auto attention, attention_builder.create("attention"));
  ASSIGN_OR_RETURN(auto mlp, mlp_builder.create("mlp"));
  ComposedLayerBuilder block_builder;
  RETURN_IF_ERROR(
      block_builder.add(ResidualLayer::Create(std::move(attention))));
  RETURN_IF_ERROR(block_builder.add(ResidualLayer::Create(std::move(mlp))));
  return block_builder.create(absl::StrCat("transformer_block_", block_index));
}

// Adds the shared token-to-residual-stream prefix to builder and returns the
// embedding layer so CreateGpt2() can tie the language-modeling head to it.
// Keeping this construction in one place prevents activation taps from
// silently drifting away from the model recipe they are meant to inspect.
absl::StatusOr<EmbeddingLookupLayer*> AddActivationGeneratorLayers(
    cuda::Executor& executor, ComposedLayerBuilder& builder,
    int transformer_block_count, DataType output_type, int seed) {
  if (transformer_block_count < 0 ||
      transformer_block_count > kGpt2TransformerBlockCount) {
    return absl::InvalidArgumentError(
        "transformer_block_count must be between zero and "
        "kGpt2TransformerBlockCount");
  }

  RETURN_IF_ERROR(builder.add(EmbeddingLookupLayer::Create(
      executor, kGpt2VocabularySize, kGpt2ModelWidth, output_type,
      kGpt2ContextLength)));
  auto* embedding = static_cast<EmbeddingLookupLayer*>(builder.back());
  RETURN_IF_ERROR(embedding->InitializeNormal(kInitializationStandardDeviation,
                                              static_cast<uint64_t>(seed)));

  RETURN_IF_ERROR(builder.add(PositionEmbeddingLayer::Create(
      executor, kGpt2ContextLength, kGpt2ModelWidth, output_type)));
  auto* positions = static_cast<PositionEmbeddingLayer*>(builder.back());
  RETURN_IF_ERROR(positions->InitializeNormal(kInitializationStandardDeviation,
                                              static_cast<uint64_t>(seed) + 1));

  for (int index = 0; index < transformer_block_count; ++index) {
    RETURN_IF_ERROR(builder.add(
        CreateTransformerBlock(executor, output_type, seed, index)));
  }
  return embedding;
}

}  // namespace

absl::StatusOr<std::unique_ptr<Layer>> CreateActivationGenerator(
    cuda::Executor& executor, int transformer_block_count, DataType output_type,
    int seed) {
  ComposedLayerBuilder builder;
  ASSIGN_OR_RETURN(
      auto* embedding,
      AddActivationGeneratorLayers(executor, builder, transformer_block_count,
                                   output_type, seed));
  (void)embedding;
  ASSIGN_OR_RETURN(auto generator, builder.create(absl::StrCat(
                                       "gpt2_activation_generator_",
                                       transformer_block_count, "_blocks")));
  return std::unique_ptr<Layer>(std::move(generator));
}

absl::StatusOr<std::unique_ptr<ComposedLayer>> CreateGpt2(
    cuda::Executor& executor, DataType output_type, int seed) {
  ComposedLayerBuilder builder;
  ASSIGN_OR_RETURN(
      auto* embedding,
      AddActivationGeneratorLayers(
          executor, builder, kGpt2TransformerBlockCount, output_type, seed));

  RETURN_IF_ERROR(builder.add(
      LayerNormLayer::Create(executor, kGpt2ModelWidth, kLayerNormEpsilon,
                             output_type, kGpt2ContextLength)));
  RETURN_IF_ERROR(builder.add(LanguageModelingHeadLayer::Create(embedding)));
  return builder.create("gpt2");
}

}  // namespace pluto::llm
