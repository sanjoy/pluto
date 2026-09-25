#include "src/llm/experiments/memorize_general_facts/fit_attention_readout/readout.h"

#include <cuda_runtime_api.h>

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

absl::StatusOr<Readout> CreateReadout(cuda::Executor& executor,
                                      const Layer& source,
                                      const Gpt2Config& config, int block,
                                      int random_seed, bool fresh_branch) {
  RETURN_IF_ERROR(config.Validate());
  if (block < 0 || block >= config.transformer_block_count ||
      random_seed < -1 || (fresh_branch && random_seed < 0))
    return absl::InvalidArgumentError("invalid readout block or seed");
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
      executor, width, config.feed_forward_width, type, sequence)));
  auto* input = static_cast<FullyConnectedLayer*>(branch.back());
  RETURN_IF_ERROR(branch.add(
      GeluLayer::Create(executor, config.feed_forward_width, type, sequence)));
  RETURN_IF_ERROR(branch.add(FullyConnectedLayer::Create(
      executor, config.feed_forward_width, width, type, sequence)));
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
    RETURN_IF_ERROR(input->InitializeNormal(0.2f, random_seed));
    RETURN_IF_ERROR(output->InitializeNormal(
        0.1f, uint64_t{static_cast<uint32_t>(random_seed)} + 1));
  }
  ASSIGN_OR_RETURN(auto residual, ResidualLayer::Create(std::move(mlp)));
  Layer* trainable = residual.get();
  ComposedLayerBuilder tail;
  RETURN_IF_ERROR(tail.add(std::move(residual)));
  ASSIGN_OR_RETURN(
      auto norm, LayerNormLayer::Create(executor, width, 1e-5f, type, sequence));
  for (size_t i = 0; i < 2; ++i)
    RETURN_IF_ERROR(
        CopyWeight(executor, source_weights[final_ln + i], norm->weights()[i]));
  RETURN_IF_ERROR(tail.add(std::move(norm)));
  RETURN_IF_ERROR(tail.add(LanguageModelingHeadLayer::Create(embedding.get())));
  ASSIGN_OR_RETURN(auto model, tail.create("replacement_readout"));
  return Readout{std::move(embedding), std::move(model), trainable};
}

}  // namespace pluto::llm::fit_attention_readout
