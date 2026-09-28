#include "src/llm/experiments/memorize_general_facts/transformer_readout.h"

#include <cuda_runtime_api.h>

#include <cstdint>
#include <iterator>
#include <memory>
#include <utility>
#include <vector>

#include "src/llm/layers/attention.h"
#include "src/llm/layers/fully_connected.h"
#include "src/llm/layers/gelu.h"
#include "src/llm/layers/norm.h"
#include "src/util/status_macros.h"

namespace pluto::llm::memorize_general_facts {
namespace {

constexpr size_t kSourceTailStart = 32;
constexpr size_t kSourceFinalNorm = 50;
constexpr size_t kTrainableTensorCount = 20;

// Tensor order is part of this experiment's contract: reject a different
// source layout before interpreting indices as MLP3, block4, and final LN.
absl::Status ValidateSource(cuda::Executor& executor, const Layer& source,
                            const Gpt2Config& config) {
  const auto weights = source.weights();
  if (source.name() != "gpt2" || source.output_type() != DataType::BF16 ||
      weights.size() != 53 || weights.front().data() != weights.back().data())
    return absl::InvalidArgumentError(
        "transformer readout source must be a four-block BF16 GPT-2 with a "
        "tied head");
  const int64_t logits = (int64_t{config.vocabulary_size} + 15) / 16 * 16;
  const auto inputs = source.input_types();
  const auto outputs = source.output_types();
  if (inputs.size() != 1 || outputs.size() != 1 ||
      inputs[0] !=
          ActivationType(DataType::INT32, {-2, config.context_length}) ||
      outputs[0] !=
          ActivationType(DataType::FP32, {-2, config.context_length, logits}))
    return absl::InvalidArgumentError("source/config GPT-2 layout mismatch");
  const size_t width = config.model_width;
  const size_t hidden = config.feed_forward_width;
  const size_t vocabulary =
      config.pad_vocabulary ? logits : config.vocabulary_size;
  std::vector<size_t> elements{vocabulary * width,
                               config.context_length * width};
  for (int block = 0; block < 4; ++block) {
    const size_t block_elements[] = {
        width, width, 3 * width * width, 3 * width, width * width,  width,
        width, width, width * hidden,    hidden,    hidden * width, width};
    elements.insert(elements.end(), std::begin(block_elements),
                    std::end(block_elements));
  }
  elements.insert(elements.end(), {width, width, vocabulary * width});
  for (size_t index = 0; index < weights.size(); ++index) {
    if (&weights[index].executor() != &executor)
      return absl::InvalidArgumentError("source executor mismatch");
    if (weights[index].size_bytes() != elements[index] * sizeof(float))
      return absl::InvalidArgumentError(
          "source/config GPT-2 weight shape mismatch");
  }
  return absl::OkStatus();
}

absl::Status CopyWeight(cuda::Executor& executor, const Buffer& from,
                        const Buffer& to) {
  if (from.size_bytes() != to.size_bytes() || &from.executor() != &executor ||
      &to.executor() != &executor)
    return absl::InvalidArgumentError(
        "transformer readout weight shape/executor mismatch");
  return cuda::CudaStatus(
      cudaMemcpyAsync(to.data(), from.data(), to.size_bytes(),
                      cudaMemcpyDeviceToDevice, executor.stream()),
      "copy transformer readout weight");
}

absl::StatusOr<std::unique_ptr<ResidualLayer>> CreateMlp(
    cuda::Executor& executor, const Gpt2Config& config, uint64_t seed) {
  ComposedLayerBuilder branch;
  RETURN_IF_ERROR(branch.add(
      LayerNormLayer::Create(executor, config.model_width, 1e-5f,
                             DataType::BF16, config.context_length)));
  RETURN_IF_ERROR(branch.add(FullyConnectedLayer::Create(
      executor, config.model_width, config.feed_forward_width, DataType::BF16,
      config.context_length)));
  RETURN_IF_ERROR(static_cast<FullyConnectedLayer*>(branch.back())
                      ->InitializeNormal(0.2f, seed));
  RETURN_IF_ERROR(
      branch.add(GeluLayer::Create(executor, config.feed_forward_width,
                                   DataType::BF16, config.context_length)));
  RETURN_IF_ERROR(branch.add(FullyConnectedLayer::Create(
      executor, config.feed_forward_width, config.model_width, DataType::BF16,
      config.context_length)));
  RETURN_IF_ERROR(static_cast<FullyConnectedLayer*>(branch.back())
                      ->InitializeNormal(0.1f, seed + 1));
  ASSIGN_OR_RETURN(auto mlp, branch.create("mlp"));
  return ResidualLayer::Create(std::move(mlp));
}

absl::StatusOr<std::unique_ptr<ResidualLayer>> CreateAttention(
    cuda::Executor& executor, const Gpt2Config& config, uint64_t seed) {
  ComposedLayerBuilder branch;
  RETURN_IF_ERROR(branch.add(
      LayerNormLayer::Create(executor, config.model_width, 1e-5f,
                             DataType::BF16, config.context_length)));
  RETURN_IF_ERROR(branch.add(FullyConnectedLayer::Create(
      executor, config.model_width, 3 * config.model_width, DataType::BF16,
      config.context_length)));
  RETURN_IF_ERROR(static_cast<FullyConnectedLayer*>(branch.back())
                      ->InitializeNormal(0.02f, seed));
  RETURN_IF_ERROR(branch.add(AttentionLayer::Create(
      executor, config.context_length, config.attention_heads,
      config.model_width, DataType::BF16)));
  RETURN_IF_ERROR(branch.add(FullyConnectedLayer::Create(
      executor, config.model_width, config.model_width, DataType::BF16,
      config.context_length)));
  // Same .02 / sqrt(2 * 8) residual scale as the GPT-2 recipe; unlike the
  // MLP initialization, this projection has no counterpart in the stack.
  RETURN_IF_ERROR(static_cast<FullyConnectedLayer*>(branch.back())
                      ->InitializeNormal(0.005f, seed + 1));
  ASSIGN_OR_RETURN(auto attention, branch.create("attention"));
  return ResidualLayer::Create(std::move(attention));
}

}  // namespace

absl::StatusOr<MlpReadoutParameterBudget>
ResolveMlpTransformerReadoutParameterBudget(const Gpt2Config& config) {
  RETURN_IF_ERROR(config.Validate());
  if (config.transformer_block_count != 4)
    return absl::InvalidArgumentError(
        "transformer readout requires exactly four source blocks");
  // Validation bounds each matrix's element count, and this fixed-depth sum
  // of 20 tensors fits int64_t even at the backend's largest legal widths.
  const int64_t d = config.model_width;
  const int64_t f = config.feed_forward_width;
  const int64_t mlp = 2 * ((2 * d + 1) * f + d);
  const int64_t attention = 4 * d * d + 4 * d;
  const int64_t total = mlp + attention + 8 * d;
  return MlpReadoutParameterBudget{
      config.feed_forward_width, config.feed_forward_width, total, mlp, total};
}

absl::StatusOr<MlpReadout> CreateMlpTransformerReadout(cuda::Executor& executor,
                                                       const Layer& source,
                                                       const Gpt2Config& config,
                                                       int seed,
                                                       bool copy_source_tail) {
  ASSIGN_OR_RETURN(auto budget,
                   ResolveMlpTransformerReadoutParameterBudget(config));
  if (seed < 0)
    return absl::InvalidArgumentError("readout seed must be nonnegative");
  RETURN_IF_ERROR(ValidateSource(executor, source, config));
  const auto source_weights = source.weights();
  ASSIGN_OR_RETURN(
      auto embedding,
      EmbeddingLookupLayer::Create(
          executor, config.vocabulary_size, config.model_width, DataType::BF16,
          config.context_length, config.pad_vocabulary));
  RETURN_IF_ERROR(CopyWeight(executor, source_weights[0], embedding->weight()));

  const uint64_t initial_seed = static_cast<uint64_t>(seed);
  ComposedLayerBuilder suffix;
  RETURN_IF_ERROR(suffix.add(CreateMlp(executor, config, initial_seed)));
  RETURN_IF_ERROR(
      suffix.add(CreateAttention(executor, config, initial_seed + 4)));
  RETURN_IF_ERROR(suffix.add(CreateMlp(executor, config, initial_seed + 2)));
  RETURN_IF_ERROR(suffix.add(
      LayerNormLayer::Create(executor, config.model_width, 1e-5f,
                             DataType::BF16, config.context_length)));
  for (size_t index = 0; index < 2; ++index)
    RETURN_IF_ERROR(CopyWeight(executor,
                               source_weights[kSourceFinalNorm + index],
                               suffix.back()->weights()[index]));
  ASSIGN_OR_RETURN(auto trainable,
                   suffix.create("mlp_transformer_readout_trainable"));
  const auto trainable_weights = trainable->weights();
  if (trainable_weights.size() != kTrainableTensorCount)
    return absl::InternalError("transformer readout tensor count mismatch");
  int64_t source_parameters = 0;
  int64_t trainable_parameters = 0;
  for (size_t index = 0; index < kTrainableTensorCount; ++index) {
    const auto& original = source_weights[kSourceTailStart + index];
    const auto& replacement = trainable_weights[index];
    if (original.size_bytes() != replacement.size_bytes())
      return absl::InternalError("transformer readout suffix shape mismatch");
    source_parameters += original.size_bytes() / sizeof(float);
    trainable_parameters += replacement.size_bytes() / sizeof(float);
    if (copy_source_tail)
      RETURN_IF_ERROR(CopyWeight(executor, original, replacement));
  }
  if (source_parameters != budget.source_tail_parameters ||
      trainable_parameters != budget.trainable_parameters)
    return absl::InternalError(
        "transformer readout parameter budget disagrees with weights");
  Layer* trainable_pointer = trainable.get();
  ComposedLayerBuilder tail;
  RETURN_IF_ERROR(tail.add(std::move(trainable)));
  RETURN_IF_ERROR(tail.add(LanguageModelingHeadLayer::Create(embedding.get())));
  ASSIGN_OR_RETURN(auto model, tail.create("mlp_transformer_readout"));
  return MlpReadout{std::move(embedding), std::move(model), trainable_pointer,
                    budget};
}

}  // namespace pluto::llm::memorize_general_facts
