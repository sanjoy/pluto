#include "src/llm/experiments/memorize_general_facts/mlp_readout.h"

#include <cuda_runtime_api.h>

#include <algorithm>
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
      "copy MLP readout weight");
}

}  // namespace

absl::StatusOr<MlpReadoutParameterBudget> ResolveMlpReadoutParameterBudget(
    const Gpt2Config& config, int requested_min_mlp_width, int mlp_depth,
    bool match_parameter_budget, bool iso_parameter_budget) {
  RETURN_IF_ERROR(config.Validate());
  if (config.transformer_block_count < 3 || mlp_depth <= 0 ||
      (!iso_parameter_budget && requested_min_mlp_width <= 0))
    return absl::InvalidArgumentError(
        "MLP readout budget requires at least three source blocks and positive "
        "MLP width and depth");
  const int64_t d = config.model_width;
  const int64_t f = config.feed_forward_width;
  // FC1/FC2 use 2*d*f matrix entries plus f+d biases. Each pre-LN
  // contributes 2*d parameters. Attention has QKV and output projections.
  // config.Validate() bounds each tensor, so these per-block sums fit int64.
  const int64_t mlp_with_norm = (2 * d + 1) * f + 3 * d;
  const int64_t attention_with_norm = 4 * d * d + 6 * d;
  const int64_t block = mlp_with_norm + attention_with_norm;
  const int64_t later_blocks = config.transformer_block_count - 3;
  const int64_t initial_tail = mlp_with_norm + 2 * d;  // MLP3 and final LN.
  if (later_blocks >
      (std::numeric_limits<int64_t>::max() - initial_tail) / block)
    return absl::InvalidArgumentError(
        "MLP readout source-tail parameter count overflows");
  const int64_t source_tail = initial_tail + later_blocks * block;
  const int64_t per_hidden_unit = 2 * d + 1;
  if (iso_parameter_budget) {
    // Each block contributes d FC2 biases and 2*d input-LN parameters,
    // independently of width; the shared final LN contributes another 2*d.
    // Every additional hidden unit then costs exactly 2*d+1 parameters, so
    // rounding the total hidden width gives the closest total-size match.
    const int64_t per_block_overhead = 3 * d;
    if (mlp_depth >
        (std::numeric_limits<int64_t>::max() - 2 * d) / per_block_overhead)
      return absl::InvalidArgumentError(
          "MLP readout parameter count overflows");
    const int64_t overhead = mlp_depth * per_block_overhead + 2 * d;
    const int64_t remaining = std::max(int64_t{0}, source_tail - overhead);
    const int64_t quotient = remaining / per_hidden_unit;
    const int64_t remainder = remaining % per_hidden_unit;
    // The per-unit price is odd, so there is no exactly tied rounding case.
    const int64_t total_width = std::max(
        int64_t{mlp_depth}, quotient + (remainder > per_hidden_unit / 2));
    const int64_t rounded_up = quotient + (remainder != 0);
    const int64_t minimum = std::max(
        int64_t{1}, rounded_up / mlp_depth + (rounded_up % mlp_depth != 0));
    const int64_t lower_width = total_width / mlp_depth;
    const int64_t wider_blocks = total_width % mlp_depth;
    const int64_t width = lower_width + (wider_blocks != 0);
    if (width > std::numeric_limits<int>::max() ||
        minimum > std::numeric_limits<int>::max())
      return absl::InvalidArgumentError(
          "required MLP readout width exceeds int32");
    auto readout_config = config;
    readout_config.feed_forward_width = static_cast<int>(width);
    RETURN_IF_ERROR(readout_config.Validate());
    if (total_width >
        (std::numeric_limits<int64_t>::max() - overhead) / per_hidden_unit)
      return absl::InvalidArgumentError(
          "MLP readout parameter count overflows");
    const int64_t variable_parameters = total_width * per_hidden_unit;
    std::vector<int> widths(mlp_depth, static_cast<int>(lower_width));
    for (int64_t index = 0; index < wider_blocks; ++index)
      ++widths[index];
    return MlpReadoutParameterBudget{static_cast<int>(minimum),
                                     static_cast<int>(width),
                                     source_tail,
                                     variable_parameters + mlp_depth * d,
                                     variable_parameters + overhead,
                                     std::move(widths)};
  }
  // Each bare MLP has (2*d+1)*h+d parameters. Divide the source budget
  // across all blocks before rounding up the width, avoiding depth products
  // until the final count is checked. Input/final LNs are extra capacity.
  const int64_t per_mlp =
      source_tail / mlp_depth + (source_tail % mlp_depth != 0);
  const int64_t required = std::max(int64_t{0}, per_mlp - d);
  const int64_t minimum =
      std::max(int64_t{1},
               required / per_hidden_unit + (required % per_hidden_unit != 0));
  if (minimum > std::numeric_limits<int>::max())
    return absl::InvalidArgumentError(
        "required MLP readout width exceeds int32");
  const int width = match_parameter_budget ? std::max(requested_min_mlp_width,
                                                      static_cast<int>(minimum))
                                           : requested_min_mlp_width;
  auto readout_config = config;
  readout_config.feed_forward_width = width;
  RETURN_IF_ERROR(readout_config.Validate());
  const int64_t affine_per_block = per_hidden_unit * width + d;
  const int64_t trainable_per_block = affine_per_block + 2 * d;
  if (mlp_depth >
      (std::numeric_limits<int64_t>::max() - 2 * d) / trainable_per_block)
    return absl::InvalidArgumentError("MLP readout parameter count overflows");
  const int64_t mlp = mlp_depth * affine_per_block;
  return MlpReadoutParameterBudget{static_cast<int>(minimum),
                                   width,
                                   source_tail,
                                   mlp,
                                   mlp_depth * trainable_per_block + 2 * d,
                                   std::vector<int>(mlp_depth, width)};
}

absl::StatusOr<MlpReadout> CreateMlpReadout(
    cuda::Executor& executor, const Layer& source, const Gpt2Config& config,
    int mlp_width, int seed, int mlp_depth, bool match_parameter_budget,
    bool residual_connections, bool iso_parameter_budget) {
  ASSIGN_OR_RETURN(auto budget,
                   ResolveMlpReadoutParameterBudget(
                       config, mlp_width, mlp_depth, match_parameter_budget,
                       iso_parameter_budget));
  if (seed < 0)
    return absl::InvalidArgumentError("readout seed must be nonnegative");
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

  // Embeddings occupy two tensors; each block has six attention tensors and
  // six MLP tensors. Start at MLP3 and stop before the final tied head alias.
  int64_t actual_source_tail = 0;
  for (size_t index = 2 + 2 * 12 + 6; index + 1 < weights.size(); ++index)
    actual_source_tail += weights[index].size_bytes() / sizeof(float);
  if (actual_source_tail != budget.source_tail_parameters)
    return absl::InternalError(
        "MLP readout source parameter budget disagrees with weights");

  ASSIGN_OR_RETURN(
      auto embedding,
      EmbeddingLookupLayer::Create(
          executor, config.vocabulary_size, config.model_width, DataType::BF16,
          config.context_length, config.pad_vocabulary));
  RETURN_IF_ERROR(CopyWeight(executor, weights[0], embedding->weight()));
  ComposedLayerBuilder suffix;
  for (int block = 0; block < mlp_depth; ++block) {
    const int block_width = budget.mlp_widths[block];
    const uint64_t block_seed =
        static_cast<uint64_t>(seed) + 2 * static_cast<uint64_t>(block);
    ComposedLayerBuilder branch;
    RETURN_IF_ERROR(branch.add(
        LayerNormLayer::Create(executor, config.model_width, 1e-5f,
                               DataType::BF16, config.context_length)));
    RETURN_IF_ERROR(branch.add(
        FullyConnectedLayer::Create(executor, config.model_width, block_width,
                                    DataType::BF16, config.context_length)));
    RETURN_IF_ERROR(static_cast<FullyConnectedLayer*>(branch.back())
                        ->InitializeNormal(0.2f, block_seed));
    RETURN_IF_ERROR(branch.add(GeluLayer::Create(
        executor, block_width, DataType::BF16, config.context_length)));
    RETURN_IF_ERROR(branch.add(
        FullyConnectedLayer::Create(executor, block_width, config.model_width,
                                    DataType::BF16, config.context_length)));
    // Keep the output projection scale fixed so depth is the only change.
    RETURN_IF_ERROR(static_cast<FullyConnectedLayer*>(branch.back())
                        ->InitializeNormal(0.1f, block_seed + 1));
    ASSIGN_OR_RETURN(auto mlp, branch.create("mlp_readout_block"));
    if (residual_connections)
      RETURN_IF_ERROR(suffix.add(ResidualLayer::Create(std::move(mlp))));
    else
      RETURN_IF_ERROR(suffix.add(std::move(mlp)));
  }
  ASSIGN_OR_RETURN(
      auto norm, LayerNormLayer::Create(executor, config.model_width, 1e-5f,
                                        DataType::BF16, config.context_length));
  for (size_t index = 0; index < 2; ++index)
    RETURN_IF_ERROR(CopyWeight(executor, weights[final_ln + index],
                               norm->weights()[index]));
  RETURN_IF_ERROR(suffix.add(std::move(norm)));
  ASSIGN_OR_RETURN(auto trainable, suffix.create("mlp_readout_trainable"));
  Layer* trainable_pointer = trainable.get();
  ComposedLayerBuilder tail;
  RETURN_IF_ERROR(tail.add(std::move(trainable)));
  RETURN_IF_ERROR(tail.add(LanguageModelingHeadLayer::Create(embedding.get())));
  ASSIGN_OR_RETURN(auto model, tail.create("mlp_readout"));
  int64_t actual_trainable = 0;
  for (const auto& weight : trainable_pointer->weights())
    actual_trainable += weight.size_bytes() / sizeof(float);
  if (actual_trainable != budget.trainable_parameters)
    return absl::InternalError(
        "MLP readout parameter budget disagrees with weights");
  return MlpReadout{std::move(embedding), std::move(model), trainable_pointer,
                    budget};
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
