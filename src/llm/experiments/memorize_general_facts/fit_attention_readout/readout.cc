#include "src/llm/experiments/memorize_general_facts/fit_attention_readout/readout.h"

#include <cuda_runtime_api.h>

#include <cmath>
#include <cstdint>
#include <utility>

#include "src/llm/layers/fully_connected.h"
#include "src/llm/layers/gelu.h"
#include "src/llm/layers/norm.h"
#include "src/util/status_macros.h"

namespace pluto::llm::fit_attention_readout {
namespace {

absl::Status CopyWeight(cuda::Executor& executor, const Buffer& from,
                        const Buffer& to) {
  if (from.size_bytes() != to.size_bytes() || &from.executor() != &executor ||
      &to.executor() != &executor)
    return absl::InvalidArgumentError("readout weight shape/executor mismatch");
  return cuda::CudaStatus(
      cudaMemcpyAsync(to.data(), from.data(), to.size_bytes(),
                      cudaMemcpyDeviceToDevice, executor.stream()),
      "copy frozen checkpoint weight");
}

}  // namespace

absl::StatusOr<Readout> CreateReadout(
    cuda::Executor& executor, const Layer& source, const Gpt2Config& config,
    int block, int random_seed, bool fresh_branch, int replacement_width,
    bool train_final_norm, const ReadoutInitializationOptions& initialization) {
  RETURN_IF_ERROR(config.Validate());
  if (block < 0 || block >= config.transformer_block_count ||
      random_seed < -1 || (fresh_branch && random_seed < 0))
    return absl::InvalidArgumentError("invalid readout block or seed");
  if (replacement_width < 0)
    return absl::InvalidArgumentError(
        "replacement MLP width must be nonnegative");
  if (!std::isfinite(initialization.input_standard_deviation) ||
      initialization.input_standard_deviation <= 0 ||
      !std::isfinite(initialization.output_standard_deviation) ||
      initialization.output_standard_deviation < 0)
    return absl::InvalidArgumentError(
        "readout initialization deviations must be finite; input must be "
        "positive and output nonnegative");
  if (random_seed < 0 && (initialization.input_standard_deviation != 0.2f ||
                          initialization.output_standard_deviation != 0.1f ||
                          initialization.scale_output_by_width))
    return absl::InvalidArgumentError(
        "custom matrix initialization requires a nonnegative random_seed");
  if (initialization.fresh_final_norm && (!fresh_branch || !train_final_norm))
    return absl::InvalidArgumentError(
        "fresh final normalization requires fresh_branch and train_final_norm");
  const int hidden_width =
      replacement_width == 0 ? config.feed_forward_width : replacement_width;
  if (hidden_width != config.feed_forward_width && !fresh_branch)
    return absl::InvalidArgumentError(
        "a different replacement MLP width requires fresh_branch and a "
        "nonnegative random_seed; checkpoint branch shapes are incompatible");
  // GPT-2 exposes token and position embeddings, twelve tensors per block,
  // final LN's two tensors, and the tied embedding again at the head. Check
  // both count and aliasing so an unrelated graph cannot silently be loaded.
  const auto source_weights = source.weights();
  const size_t final_ln = 2 + 12 * config.transformer_block_count;
  if (source_weights.size() != final_ln + 3 ||
      source_weights.front().data() != source_weights.back().data())
    return absl::InvalidArgumentError(
        "source does not have GPT-2 weight layout");
  const auto type = source.output_type();
  const int width = config.model_width;
  const int sequence = config.context_length;
  ASSIGN_OR_RETURN(auto embedding, EmbeddingLookupLayer::Create(
                                       executor, config.vocabulary_size, width,
                                       type, sequence, config.pad_vocabulary));
  RETURN_IF_ERROR(CopyWeight(executor, source_weights[0], embedding->weight()));

  ComposedLayerBuilder branch;
  RETURN_IF_ERROR(branch.add(
      LayerNormLayer::Create(executor, width, 1e-5f, type, sequence)));
  RETURN_IF_ERROR(branch.add(FullyConnectedLayer::Create(
      executor, width, hidden_width, type, sequence)));
  auto* input = static_cast<FullyConnectedLayer*>(branch.back());
  RETURN_IF_ERROR(
      branch.add(GeluLayer::Create(executor, hidden_width, type, sequence)));
  RETURN_IF_ERROR(branch.add(FullyConnectedLayer::Create(
      executor, hidden_width, width, type, sequence)));
  auto* output = static_cast<FullyConnectedLayer*>(branch.back());
  ASSIGN_OR_RETURN(auto mlp, branch.create("replacement_mlp"));
  // Within a block: LN1(2), QKV(2), attention projection(2), then the six
  // trainable LN2/FC1/FC2 tensors. Attention itself has no parameters.
  const size_t first_mlp = 2 + 12 * block + 6;
  if (!fresh_branch)
    for (size_t i = 0; i < mlp->weights().size(); ++i)
      RETURN_IF_ERROR(CopyWeight(executor, source_weights[first_mlp + i],
                                 mlp->weights()[i]));
  if (random_seed >= 0) {
    RETURN_IF_ERROR(input->InitializeNormal(
        initialization.input_standard_deviation, random_seed));
    float output_deviation = initialization.output_standard_deviation;
    if (initialization.scale_output_by_width)
      output_deviation *= std::sqrt(
          static_cast<float>(config.feed_forward_width) / hidden_width);
    if (!std::isfinite(output_deviation))
      return absl::InvalidArgumentError(
          "width-scaled readout initialization deviation overflowed");
    if (output_deviation == 0) {
      // FullyConnectedLayer's normal initializer deliberately rejects zero.
      // Zero only the matrix here: a non-fresh branch keeps its copied bias.
      RETURN_IF_ERROR(cuda::CudaStatus(
          cudaMemsetAsync(output->weights()[0].data(), 0,
                          output->weights()[0].size_bytes(), executor.stream()),
          "zero readout output matrix"));
    } else {
      RETURN_IF_ERROR(output->InitializeNormal(
          output_deviation, uint64_t{static_cast<uint32_t>(random_seed)} + 1));
    }
  }
  ASSIGN_OR_RETURN(auto residual, ResidualLayer::Create(std::move(mlp)));
  Layer* residual_branch = residual.get();
  Layer* trainable = residual_branch;
  ASSIGN_OR_RETURN(
      auto norm, LayerNormLayer::Create(executor, width, 1e-5f, type, sequence));
  if (!initialization.fresh_final_norm)
    for (size_t i = 0; i < 2; ++i)
      RETURN_IF_ERROR(CopyWeight(executor, source_weights[final_ln + i],
                                 norm->weights()[i]));
  ComposedLayerBuilder tail;
  if (train_final_norm) {
    ComposedLayerBuilder suffix;
    RETURN_IF_ERROR(suffix.add(std::move(residual)));
    RETURN_IF_ERROR(suffix.add(std::move(norm)));
    ASSIGN_OR_RETURN(auto trainable_suffix,
                     suffix.create("replacement_mlp_and_final_norm"));
    trainable = trainable_suffix.get();
    RETURN_IF_ERROR(tail.add(std::move(trainable_suffix)));
  } else {
    RETURN_IF_ERROR(tail.add(std::move(residual)));
    RETURN_IF_ERROR(tail.add(std::move(norm)));
  }
  RETURN_IF_ERROR(tail.add(LanguageModelingHeadLayer::Create(embedding.get())));
  ASSIGN_OR_RETURN(auto model, tail.create("replacement_readout"));
  return Readout{std::move(embedding), std::move(model), residual_branch,
                 trainable};
}

}  // namespace pluto::llm::fit_attention_readout
