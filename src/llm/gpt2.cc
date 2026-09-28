#include "src/llm/gpt2.h"

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
#include "src/llm/token_order.h"
#include "src/util/status_macros.h"

namespace pluto::llm {
namespace {

constexpr float kLayerNormEpsilon = 1e-5f;
constexpr float kInitializationStandardDeviation = 0.02f;

float ResidualInitializationStandardDeviation() {
  // Intentionally use the original eight-block scaling at every depth. Shared
  // blocks therefore initialize identically in depth comparisons, and adding
  // this configuration API does not change existing checkpoint trajectories.
  return kInitializationStandardDeviation /
         std::sqrt(2.0f * kGpt2TransformerBlockCount);
}

uint64_t BlockSeedBase(int initialization_seed, int block_index) {
  return static_cast<uint64_t>(static_cast<uint32_t>(initialization_seed)) +
         1'000 + static_cast<uint64_t>(block_index) * 100;
}

// Pre-LayerNorm attention branch, without its outer residual addition.
absl::StatusOr<std::unique_ptr<ComposedLayer>> CreateAttentionBranch(
    cuda::Executor& executor, DataType output_type, int initialization_seed,
    int block_index, const Gpt2Config& config) {
  const uint64_t seed_base = BlockSeedBase(initialization_seed, block_index);

  ComposedLayerBuilder attention_builder;
  RETURN_IF_ERROR(attention_builder.add(
      LayerNormLayer::Create(executor, config.model_width, kLayerNormEpsilon,
                             output_type, config.context_length)));
  RETURN_IF_ERROR(attention_builder.add(FullyConnectedLayer::Create(
      executor, config.model_width, 3 * config.model_width, output_type,
      config.context_length)));
  auto* qkv_projection =
      static_cast<FullyConnectedLayer*>(attention_builder.back());
  RETURN_IF_ERROR(qkv_projection->InitializeNormal(
      kInitializationStandardDeviation, seed_base + 1));
  RETURN_IF_ERROR(attention_builder.add(AttentionLayer::Create(
      executor, config.context_length, config.attention_heads,
      config.model_width, output_type)));
  RETURN_IF_ERROR(attention_builder.add(FullyConnectedLayer::Create(
      executor, config.model_width, config.model_width, output_type,
      config.context_length)));
  auto* attention_projection =
      static_cast<FullyConnectedLayer*>(attention_builder.back());
  RETURN_IF_ERROR(attention_projection->InitializeNormal(
      ResidualInitializationStandardDeviation(), seed_base + 2));
  return attention_builder.create("attention");
}

// Pre-LayerNorm MLP branch, without its outer residual addition.
absl::StatusOr<std::unique_ptr<ComposedLayer>> CreateMlpBranch(
    cuda::Executor& executor, DataType output_type, int initialization_seed,
    int block_index, const Gpt2Config& config) {
  const uint64_t seed_base = BlockSeedBase(initialization_seed, block_index);
  ComposedLayerBuilder mlp_builder;
  RETURN_IF_ERROR(mlp_builder.add(
      LayerNormLayer::Create(executor, config.model_width, kLayerNormEpsilon,
                             output_type, config.context_length)));
  RETURN_IF_ERROR(mlp_builder.add(FullyConnectedLayer::Create(
      executor, config.model_width, config.feed_forward_width, output_type,
      config.context_length)));
  auto* mlp_input = static_cast<FullyConnectedLayer*>(mlp_builder.back());
  RETURN_IF_ERROR(mlp_input->InitializeNormal(kInitializationStandardDeviation,
                                              seed_base + 3));
  RETURN_IF_ERROR(
      mlp_builder.add(GeluLayer::Create(executor, config.feed_forward_width,
                                        output_type, config.context_length)));
  RETURN_IF_ERROR(mlp_builder.add(FullyConnectedLayer::Create(
      executor, config.feed_forward_width, config.model_width, output_type,
      config.context_length)));
  auto* mlp_output = static_cast<FullyConnectedLayer*>(mlp_builder.back());
  RETURN_IF_ERROR(mlp_output->InitializeNormal(
      ResidualInitializationStandardDeviation(), seed_base + 4));
  return mlp_builder.create("mlp");
}

// One ordinary GPT-2 block retains the historical topology, diagnostic names,
// parameter order, and initialization seeds when sharing these branch helpers.
absl::StatusOr<std::unique_ptr<ComposedLayer>> CreateTransformerBlock(
    cuda::Executor& executor, DataType output_type, int initialization_seed,
    int block_index, const Gpt2Config& config) {
  ASSIGN_OR_RETURN(auto attention, CreateAttentionBranch(executor, output_type,
                                                         initialization_seed,
                                                         block_index, config));
  ASSIGN_OR_RETURN(auto mlp,
                   CreateMlpBranch(executor, output_type, initialization_seed,
                                   block_index, config));
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
      config.context_length, config.pad_vocabulary)));
  auto* embedding = static_cast<EmbeddingLookupLayer*>(builder.back());
  RETURN_IF_ERROR(embedding->InitializeNormal(kInitializationStandardDeviation,
                                              static_cast<uint64_t>(seed)));

  RETURN_IF_ERROR(builder.add(PositionEmbeddingLayer::Create(
      executor, config.context_length, config.model_width, output_type)));
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
  if (context_length <= 0)
    return absl::InvalidArgumentError("context_length must be positive");

  // Widen before padding or multiplying. A tiny configured vocabulary no
  // longer bounds the width, so test the QKV product by division instead of
  // forming 3 * width * width (which can overflow even int64_t).
  constexpr int64_t kMaxElements = std::numeric_limits<int>::max();
  const int64_t width = model_width;
  const int64_t ff_width = feed_forward_width;
  const int64_t context = context_length;
  const int64_t logit_stride = (int64_t{vocabulary_size} + 15) / 16 * 16;
  const int64_t stored_vocabulary =
      pad_vocabulary ? logit_stride : vocabulary_size;
  if (stored_vocabulary * width > kMaxElements)
    return absl::InvalidArgumentError(
        "token embedding exceeds the backend's 32-bit element-count limit");
  if (width > kMaxElements / 3 / width || width * ff_width > kMaxElements ||
      context > kMaxElements / logit_stride ||
      context > kMaxElements / 3 / width || context > kMaxElements / ff_width)
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
    int transformer_block_count, absl::Span<const int32_t> token_order) {
  Gpt2Config config;
  config.transformer_block_count = transformer_block_count;
  return CreateGpt2(executor, output_type, seed, config, token_order);
}

absl::StatusOr<std::unique_ptr<ComposedLayer>> CreateGpt2(
    cuda::Executor& executor, DataType output_type, int seed,
    const Gpt2Config& config, absl::Span<const int32_t> token_order) {
  RETURN_IF_ERROR(ValidateTokenOrder(config.vocabulary_size, token_order));
  ComposedLayerBuilder builder;
  ASSIGN_OR_RETURN(auto* embedding,
                   AddActivationGeneratorLayers(executor, builder, config,
                                                output_type, seed));

  RETURN_IF_ERROR(builder.add(
      LayerNormLayer::Create(executor, config.model_width, kLayerNormEpsilon,
                             output_type, config.context_length)));
  RETURN_IF_ERROR(
      builder.add(LanguageModelingHeadLayer::Create(embedding, token_order)));
  return builder.create("gpt2");
}

absl::StatusOr<std::unique_ptr<ComposedLayer>> CreateGpt2WithA3MlpStack(
    cuda::Executor& executor, DataType output_type, int seed,
    const Gpt2Config& config, absl::Span<const int32_t> token_order) {
  if (config.transformer_block_count != 4)
    return absl::InvalidArgumentError(
        "gpt2_a3_mlp_stack requires transformer_block_count == 4");
  RETURN_IF_ERROR(config.Validate());
  RETURN_IF_ERROR(ValidateTokenOrder(config.vocabulary_size, token_order));

  ComposedLayerBuilder builder;
  Gpt2Config prefix_config = config;
  prefix_config.transformer_block_count = 2;
  ASSIGN_OR_RETURN(auto* embedding,
                   AddActivationGeneratorLayers(
                       executor, builder, prefix_config, output_type, seed));
  ASSIGN_OR_RETURN(auto attention, CreateAttentionBranch(executor, output_type,
                                                         seed, 2, config));
  RETURN_IF_ERROR(builder.add(ResidualLayer::Create(std::move(attention))));
  for (int mlp_index = 2; mlp_index < 5; ++mlp_index) {
    ASSIGN_OR_RETURN(auto mlp, CreateMlpBranch(executor, output_type, seed,
                                               mlp_index, config));
    RETURN_IF_ERROR(builder.add(ResidualLayer::Create(std::move(mlp))));
  }
  RETURN_IF_ERROR(builder.add(
      LayerNormLayer::Create(executor, config.model_width, kLayerNormEpsilon,
                             output_type, config.context_length)));
  RETURN_IF_ERROR(
      builder.add(LanguageModelingHeadLayer::Create(embedding, token_order)));
  return builder.create("gpt2_a3_mlp_stack");
}

}  // namespace pluto::llm
