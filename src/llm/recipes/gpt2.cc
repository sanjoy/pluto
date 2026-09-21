#include "src/llm/recipes/gpt2.h"

#include <cmath>
#include <cstdint>
#include <limits>
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
// uses online FP32 softmax statistics. The MLP expands the representation to
// config.feed_forward_width. Dropout and attention dropout are exactly zero.
// Every block receives independently initialized parameters.
absl::StatusOr<std::unique_ptr<ComposedLayer>> CreateTransformerBlock(
    cuda::Executor& executor, DataType output_type, int initialization_seed,
    int block_index, const Gpt2Config& config) {
  // Intentionally use the original eight-block scaling at every depth. Shared
  // blocks therefore initialize identically in depth comparisons, and adding
  // this configuration API does not change existing checkpoint trajectories.
  const float residual_standard_deviation =
      kInitializationStandardDeviation /
      std::sqrt(2.0f * kGpt2TransformerBlockCount);
  const uint64_t seed_base =
      static_cast<uint64_t>(static_cast<uint32_t>(initialization_seed)) +
      1'000 + static_cast<uint64_t>(block_index) * 100;

  ComposedLayerBuilder attention_builder;
  RETURN_IF_ERROR(attention_builder.add(
      LayerNormLayer::Create(executor, config.model_width, kLayerNormEpsilon,
                             output_type, kGpt2ContextLength)));
  RETURN_IF_ERROR(attention_builder.add(FullyConnectedLayer::Create(
      executor, config.model_width, 3 * config.model_width, output_type,
      kGpt2ContextLength)));
  auto* qkv_projection =
      static_cast<FullyConnectedLayer*>(attention_builder.back());
  RETURN_IF_ERROR(qkv_projection->InitializeNormal(
      kInitializationStandardDeviation, seed_base + 1));
  RETURN_IF_ERROR(attention_builder.add(AttentionLayer::Create(
      executor, kGpt2ContextLength, config.attention_heads, config.model_width,
      output_type)));
  RETURN_IF_ERROR(attention_builder.add(FullyConnectedLayer::Create(
      executor, config.model_width, config.model_width, output_type,
      kGpt2ContextLength)));
  auto* attention_projection =
      static_cast<FullyConnectedLayer*>(attention_builder.back());
  RETURN_IF_ERROR(attention_projection->InitializeNormal(
      residual_standard_deviation, seed_base + 2));

  ComposedLayerBuilder mlp_builder;
  RETURN_IF_ERROR(mlp_builder.add(
      LayerNormLayer::Create(executor, config.model_width, kLayerNormEpsilon,
                             output_type, kGpt2ContextLength)));
  RETURN_IF_ERROR(mlp_builder.add(FullyConnectedLayer::Create(
      executor, config.model_width, config.feed_forward_width, output_type,
      kGpt2ContextLength)));
  auto* mlp_input = static_cast<FullyConnectedLayer*>(mlp_builder.back());
  RETURN_IF_ERROR(mlp_input->InitializeNormal(kInitializationStandardDeviation,
                                              seed_base + 3));
  RETURN_IF_ERROR(mlp_builder.add(GeluLayer::Create(
      executor, config.feed_forward_width, output_type, kGpt2ContextLength)));
  RETURN_IF_ERROR(mlp_builder.add(FullyConnectedLayer::Create(
      executor, config.feed_forward_width, config.model_width, output_type,
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
    const Gpt2Config& config, DataType output_type, int seed) {
  RETURN_IF_ERROR(config.Validate());

  RETURN_IF_ERROR(builder.add(EmbeddingLookupLayer::Create(
      executor, config.vocabulary_size, config.model_width, output_type,
      kGpt2ContextLength, config.pad_vocabulary)));
  auto* embedding = static_cast<EmbeddingLookupLayer*>(builder.back());
  RETURN_IF_ERROR(embedding->InitializeNormal(kInitializationStandardDeviation,
                                              static_cast<uint64_t>(seed)));

  RETURN_IF_ERROR(builder.add(PositionEmbeddingLayer::Create(
      executor, kGpt2ContextLength, config.model_width, output_type)));
  auto* positions = static_cast<PositionEmbeddingLayer*>(builder.back());
  RETURN_IF_ERROR(positions->InitializeNormal(kInitializationStandardDeviation,
                                              static_cast<uint64_t>(seed) + 1));

  for (int index = 0; index < config.transformer_block_count; ++index) {
    RETURN_IF_ERROR(builder.add(
        CreateTransformerBlock(executor, output_type, seed, index, config)));
  }
  return embedding;
}

}  // namespace

absl::Status Gpt2Config::Validate() const {
  if (transformer_block_count < 0)
    return absl::InvalidArgumentError(
        "transformer_block_count must be nonnegative");
  // Compute tiles are masked independently of logical channel widths. In
  // particular, a narrow model has no extra trainable padding channels and
  // LayerNorm/attention statistics use only its actual dimensions.
  if (model_width <= 0 || feed_forward_width <= 0)
    return absl::InvalidArgumentError(
        "model_width and feed_forward_width must be positive");
  if (attention_heads <= 0 || model_width % attention_heads != 0)
    return absl::InvalidArgumentError(
        "attention_heads must be positive and divide model_width");

  if (vocabulary_size <= 0)
    return absl::InvalidArgumentError("vocabulary_size must be positive");

  // Widen before padding or multiplying. A tiny configured vocabulary no
  // longer bounds the width, so test the QKV product by division instead of
  // forming 3 * width * width (which can overflow even int64_t).
  constexpr int64_t kMaxElements = std::numeric_limits<int>::max();
  const int64_t width = model_width;
  const int64_t ff_width = feed_forward_width;
  const int64_t logit_stride = (int64_t{vocabulary_size} + 15) / 16 * 16;
  const int64_t stored_vocabulary =
      pad_vocabulary ? logit_stride : vocabulary_size;
  if (stored_vocabulary * width > kMaxElements)
    return absl::InvalidArgumentError(
        "token embedding exceeds the backend's 32-bit element-count limit");
  if (width > kMaxElements / 3 / width || width * ff_width > kMaxElements ||
      kGpt2ContextLength * logit_stride > kMaxElements ||
      kGpt2ContextLength * 3 * width > kMaxElements ||
      kGpt2ContextLength * ff_width > kMaxElements)
    return absl::InvalidArgumentError(
        "GPT-2 parameter or activation tensor exceeds the backend's 32-bit "
        "element-count limit");
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<Layer>> CreateActivationGenerator(
    cuda::Executor& executor, int transformer_block_count, DataType output_type,
    int seed) {
  Gpt2Config config;
  config.transformer_block_count = transformer_block_count;
  return CreateActivationGenerator(executor, config, output_type, seed);
}

absl::StatusOr<std::unique_ptr<Layer>> CreateActivationGenerator(
    cuda::Executor& executor, const Gpt2Config& config, DataType output_type,
    int seed) {
  ComposedLayerBuilder builder;
  ASSIGN_OR_RETURN(auto* embedding,
                   AddActivationGeneratorLayers(executor, builder, config,
                                                output_type, seed));
  (void)embedding;
  ASSIGN_OR_RETURN(
      auto generator,
      builder.create(absl::StrCat("gpt2_activation_generator_",
                                  config.transformer_block_count, "_blocks")));
  return std::unique_ptr<Layer>(std::move(generator));
}

absl::StatusOr<std::unique_ptr<ComposedLayer>> CreateGpt2(
    cuda::Executor& executor, DataType output_type, int seed,
    int transformer_block_count) {
  Gpt2Config config;
  config.transformer_block_count = transformer_block_count;
  return CreateGpt2(executor, output_type, seed, config);
}

absl::StatusOr<std::unique_ptr<ComposedLayer>> CreateGpt2(
    cuda::Executor& executor, DataType output_type, int seed,
    const Gpt2Config& config) {
  ComposedLayerBuilder builder;
  ASSIGN_OR_RETURN(auto* embedding,
                   AddActivationGeneratorLayers(executor, builder, config,
                                                output_type, seed));

  RETURN_IF_ERROR(builder.add(
      LayerNormLayer::Create(executor, config.model_width, kLayerNormEpsilon,
                             output_type, kGpt2ContextLength)));
  RETURN_IF_ERROR(builder.add(LanguageModelingHeadLayer::Create(embedding)));
  return builder.create("gpt2");
}

}  // namespace pluto::llm
