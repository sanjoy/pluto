#ifndef PLUTO_SRC_LLM_SIMPLE_LLM_H_
#define PLUTO_SRC_LLM_SIMPLE_LLM_H_

#include <cuda_runtime_api.h>

#include <memory>

#include "absl/status/statusor.h"
#include "src/llm/layer.h"

namespace pluto::llm {

struct SimpleLlmConfig {
  DataType data_type = DataType::FP16;
  float learning_rate = 6.0f;
  // Each repetition is a ComposedLayer containing an identity-initialized
  // dense layer. The default keeps the integration model fast; increasing the
  // count exercises a deeper model without changing the execution API.
  int dense_repetitions = 1;
};

// Builds the predictor for a deliberately small byte-level bigram LLM:
//
//   token -> trainable embedding/logits
//         -> RepeatedLayer<ComposedLayer<identity dense>>
//
// The identity dense layer makes the example exercise a composed cuTile model
// while leaving the optimization problem convex and fast enough for a Bazel
// integration test. CrossEntropyLossLayer is created separately because labels
// are training-only inputs and prompting needs the predictor by itself.
absl::StatusOr<std::unique_ptr<Layer>> CreateSimpleLlm(
    DataType data_type, float learning_rate, cudaStream_t stream);
absl::StatusOr<std::unique_ptr<Layer>> CreateSimpleLlm(
    const SimpleLlmConfig& config, cudaStream_t stream);

}  // namespace pluto::llm

#endif  // PLUTO_SRC_LLM_SIMPLE_LLM_H_
