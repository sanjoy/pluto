#include "src/llm/experiments/memorize_general_facts/puzzle_readout.h"

#include <cuda_runtime_api.h>

#include <cstdint>
#include <iterator>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "src/llm/layer_hooks.h"
#include "src/llm/layers/fully_connected.h"
#include "src/llm/layers/gelu.h"
#include "src/llm/layers/norm.h"
#include "src/util/status_macros.h"

namespace pluto::llm::memorize_general_facts {
namespace {

absl::Status ValidateSource(cuda::Executor& executor, const Layer& source) {
  const auto weights = source.weights();
  if (source.name() != "gpt2" || source.output_type() != DataType::BF16 ||
      weights.size() < 41 || (weights.size() - 5) % 12 != 0 ||
      weights.front().data() != weights.back().data())
    return absl::InvalidArgumentError(
        "source must be a BF16 GPT-2 with at least three blocks and a tied "
        "head");
  for (const auto& weight : weights)
    if (&weight.executor() != &executor)
      return absl::InvalidArgumentError("source executor mismatch");
  const auto inputs = source.input_types();
  const auto outputs = source.output_types();
  if (inputs.size() != 1 || outputs.size() != 1 ||
      inputs[0].data_type() != DataType::INT32 ||
      inputs[0].dimensions().size() != 2 ||
      inputs[0].dimensions()[0] != ActivationType::kBatchDimension ||
      inputs[0].dimensions()[1] <= 0 ||
      outputs[0].data_type() != DataType::FP32 ||
      outputs[0].dimensions().size() != 3 ||
      outputs[0].dimensions()[0] != ActivationType::kBatchDimension ||
      outputs[0].dimensions()[1] != inputs[0].dimensions()[1] ||
      outputs[0].dimensions()[2] <= 0)
    return absl::InvalidArgumentError("source has invalid GPT-2 signatures");
  return absl::OkStatus();
}

absl::Status CopyWeight(cuda::Executor& executor, const Buffer& from,
                        const Buffer& to) {
  if (from.size_bytes() != to.size_bytes() || &from.executor() != &executor ||
      &to.executor() != &executor)
    return absl::InvalidArgumentError("readout weight shape/executor mismatch");
  return cuda::CudaStatus(
      cudaMemcpyAsync(to.data(), from.data(), to.size_bytes(),
                      cudaMemcpyDeviceToDevice, executor.stream()),
      "copy puzzle readout weight");
}

}  // namespace

absl::StatusOr<PuzzleReadout> CreatePuzzleReadout(cuda::Executor& executor,
                                                  const Layer& source,
                                                  const Gpt2Config& config,
                                                  int mlp_width, int seed) {
  RETURN_IF_ERROR(config.Validate());
  if (mlp_width <= 0 || seed < 0)
    return absl::InvalidArgumentError(
        "readout MLP width must be positive and seed nonnegative");
  auto readout_config = config;
  readout_config.feed_forward_width = mlp_width;
  RETURN_IF_ERROR(readout_config.Validate());
  RETURN_IF_ERROR(ValidateSource(executor, source));
  const auto weights = source.weights();
  const size_t final_ln =
      2 + 12 * static_cast<size_t>(config.transformer_block_count);
  const size_t width = config.model_width;
  const size_t hidden = config.feed_forward_width;
  const size_t sequence = config.context_length;
  const size_t logits =
      (static_cast<size_t>(config.vocabulary_size) + 15) / 16 * 16;
  const size_t vocabulary =
      config.pad_vocabulary ? logits : config.vocabulary_size;
  if (weights.size() != final_ln + 3 ||
      source.input_types()[0] !=
          ActivationType(DataType::INT32, {-2, config.context_length}) ||
      source.output_types()[0] !=
          ActivationType(DataType::FP32, {-2, config.context_length,
                                          static_cast<int64_t>(logits)}))
    return absl::InvalidArgumentError("source/config GPT-2 layout mismatch");
  // Validate every tensor, including source MLP widths that are not copied.
  std::vector<size_t> elements{vocabulary * width, sequence * width};
  for (int block = 0; block < config.transformer_block_count; ++block) {
    const size_t block_elements[] = {
        width, width, 3 * width * width, 3 * width, width * width,  width,
        width, width, width * hidden,    hidden,    hidden * width, width};
    elements.insert(elements.end(), std::begin(block_elements),
                    std::end(block_elements));
  }
  elements.insert(elements.end(), {width, width, vocabulary * width});
  for (size_t index = 0; index < weights.size(); ++index)
    if (weights[index].size_bytes() != elements[index] * sizeof(float))
      return absl::InvalidArgumentError(
          "source/config GPT-2 weight shape mismatch");

  ASSIGN_OR_RETURN(
      auto embedding,
      EmbeddingLookupLayer::Create(
          executor, config.vocabulary_size, config.model_width, DataType::BF16,
          config.context_length, config.pad_vocabulary));
  RETURN_IF_ERROR(CopyWeight(executor, weights[0], embedding->weight()));
  ComposedLayerBuilder branch;
  RETURN_IF_ERROR(branch.add(
      LayerNormLayer::Create(executor, config.model_width, 1e-5f,
                             DataType::BF16, config.context_length)));
  RETURN_IF_ERROR(branch.add(
      FullyConnectedLayer::Create(executor, config.model_width, mlp_width,
                                  DataType::BF16, config.context_length)));
  RETURN_IF_ERROR(static_cast<FullyConnectedLayer*>(branch.back())
                      ->InitializeNormal(0.2f, static_cast<uint64_t>(seed)));
  RETURN_IF_ERROR(branch.add(GeluLayer::Create(
      executor, mlp_width, DataType::BF16, config.context_length)));
  RETURN_IF_ERROR(branch.add(
      FullyConnectedLayer::Create(executor, mlp_width, config.model_width,
                                  DataType::BF16, config.context_length)));
  RETURN_IF_ERROR(
      static_cast<FullyConnectedLayer*>(branch.back())
          ->InitializeNormal(0.1f, static_cast<uint64_t>(seed) + 1));
  ASSIGN_OR_RETURN(auto mlp, branch.create("puzzle_mlp"));
  ASSIGN_OR_RETURN(auto residual, ResidualLayer::Create(std::move(mlp)));
  ASSIGN_OR_RETURN(
      auto norm, LayerNormLayer::Create(executor, config.model_width, 1e-5f,
                                        DataType::BF16, config.context_length));
  for (size_t index = 0; index < 2; ++index)
    RETURN_IF_ERROR(CopyWeight(executor, weights[final_ln + index],
                               norm->weights()[index]));
  ComposedLayerBuilder suffix;
  RETURN_IF_ERROR(suffix.add(std::move(residual)));
  RETURN_IF_ERROR(suffix.add(std::move(norm)));
  ASSIGN_OR_RETURN(auto trainable, suffix.create("puzzle_trainable"));
  Layer* trainable_pointer = trainable.get();
  ComposedLayerBuilder tail;
  RETURN_IF_ERROR(tail.add(std::move(trainable)));
  RETURN_IF_ERROR(tail.add(LanguageModelingHeadLayer::Create(embedding.get())));
  ASSIGN_OR_RETURN(auto model, tail.create("puzzle_readout"));
  return PuzzleReadout{std::move(embedding), std::move(model),
                       trainable_pointer};
}

absl::StatusOr<PuzzleCapture> CaptureThirdAttention(cuda::Executor& executor,
                                                    const Layer& source,
                                                    const Buffer& tokens) {
  RETURN_IF_ERROR(ValidateSource(executor, source));
  const size_t sequence = source.input_types()[0].dimensions()[1];
  if (&tokens.executor() != &executor || tokens.size_bytes() == 0 ||
      tokens.size_bytes() % (sequence * sizeof(int32_t)) != 0)
    return absl::InvalidArgumentError("capture token shape/executor mismatch");
  const auto weights = source.weights();
  const size_t width = weights[weights.size() - 3].size_bytes() / sizeof(float);
  const size_t token_count = tokens.size_bytes() / sizeof(int32_t);
  if (width == 0 ||
      token_count > std::numeric_limits<size_t>::max() / width / 2)
    return absl::InvalidArgumentError("capture hidden shape overflow");
  const ActivationType hidden_type(
      DataType::BF16,
      {-2, static_cast<int64_t>(sequence), static_cast<int64_t>(width)});
  std::vector<std::string> scopes;
  std::optional<Buffer> hidden;
  LayerHooks hooks;
  hooks.enter_combinator = [&](auto&, auto name) {
    scopes.emplace_back(name);
    return absl::OkStatus();
  };
  hooks.exit_combinator = [&](auto&) {
    if (scopes.empty())
      return absl::DataLossError("unbalanced puzzle capture scope");
    scopes.pop_back();
    return absl::OkStatus();
  };
  hooks.activation_hook = [&](auto&, auto name, auto types, auto buffers) {
    if (!hidden && name == "ResidualLayer" && scopes.size() == 2 &&
        scopes[0] == "gpt2" && scopes[1] == "transformer_block_2") {
      if (types.size() != 1 || types[0] != hidden_type || buffers.size() != 1 ||
          buffers[0].size_bytes() != token_count * width * sizeof(uint16_t) ||
          &buffers[0].executor() != &executor)
        return absl::DataLossError(
            "unexpected third-attention capture type/shape");
      hidden = buffers[0];
    }
    return absl::OkStatus();
  };
  ASSIGN_OR_RETURN(auto forward, source.fwd(executor, {&tokens, 1}, &hooks));
  forward.state = BackwardState{};
  if (!hidden || !scopes.empty())
    return absl::DataLossError("missing third-attention residual capture");
  const size_t columns = source.output_types()[0].dimensions()[2];
  if (forward.outputs.size() != 1 ||
      forward.outputs[0].size_bytes() != token_count * columns * sizeof(float))
    return absl::DataLossError("unexpected source logit shape");
  return PuzzleCapture{std::move(*hidden), std::move(forward.outputs[0])};
}

}  // namespace pluto::llm::memorize_general_facts
