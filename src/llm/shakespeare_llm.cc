#include "src/llm/shakespeare_llm.h"

#include <cuda_runtime_api.h>

#include <memory>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "src/llm/layer.h"
#include "src/llm/layers.h"

namespace pluto::llm {

absl::StatusOr<std::unique_ptr<Layer>> CreateShakespeareLlm(
    DataType data_type, float learning_rate, cudaStream_t stream) {
  ShakespeareLlmConfig config{
      .data_type = data_type,
      .learning_rate = learning_rate,
  };
  return CreateShakespeareLlm(config, stream);
}

absl::StatusOr<std::unique_ptr<Layer>> CreateShakespeareLlm(
    const ShakespeareLlmConfig& config, cudaStream_t stream) {
  if (config.dense_repetitions < 0) {
    return absl::InvalidArgumentError(
        "dense_repetitions must be non-negative");
  }
  auto embedding =
      EmbeddingLookupLayer::Create(config.data_type, config.learning_rate,
                                   stream);
  if (!embedding.ok()) return embedding.status();

  std::vector<std::unique_ptr<Layer>> layers;
  layers.push_back(std::move(*embedding));
  if (config.dense_repetitions > 0) {
    std::vector<std::unique_ptr<ComposedLayer>> blocks;
    blocks.reserve(config.dense_repetitions);
    for (int repeat = 0; repeat < config.dense_repetitions; ++repeat) {
      // Dense blocks are frozen. Their forward and backward cuTile MMAs remain
      // part of the model while the trainable bigram table stays a convex,
      // quick optimization target for the integration test.
      auto projection =
          FullyConnectedLayer::Create(config.data_type, 0.0f, stream);
      if (!projection.ok()) return projection.status();
      if (auto status = (*projection)->InitializeIdentity(); !status.ok()) {
        return status;
      }
      std::vector<std::unique_ptr<Layer>> block_layers;
      block_layers.push_back(std::move(*projection));
      blocks.push_back(std::make_unique<ComposedLayer>(
          config.data_type, std::move(block_layers)));
    }
    layers.push_back(
        std::make_unique<RepeatedLayer<ComposedLayer>>(config.data_type,
                                                       std::move(blocks)));
  }
  return std::unique_ptr<Layer>(
      new ComposedLayer(config.data_type, std::move(layers)));
}

}  // namespace pluto::llm
