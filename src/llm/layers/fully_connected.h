#ifndef PLUTO_SRC_LLM_LAYERS_FULLY_CONNECTED_H_
#define PLUTO_SRC_LLM_LAYERS_FULLY_CONNECTED_H_

#include <cuda_runtime_api.h>

#include <cstdint>
#include <memory>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "src/llm/layer.h"

namespace pluto::llm {

// A bias-bearing rectangular projection. Activations use the selected compute
// type, MMA reductions accumulate in FP32, and parameters/gradients remain
// FP32 master buffers for the external optimizer.
class FullyConnectedLayer final : public Layer {
 public:
  static absl::StatusOr<std::unique_ptr<FullyConnectedLayer>> Create(
      int input_dim, int output_dim, DataType data_type,
      cudaStream_t stream);
  static absl::StatusOr<std::unique_ptr<FullyConnectedLayer>> Create(
      int model_width, DataType data_type, cudaStream_t stream) {
    return Create(model_width, model_width, data_type, stream);
  }

  // Initializes the rectangular matrix to a scaled identity on its available
  // diagonal and clears the bias.
  absl::Status InitializeIdentity(float scale = 1.0f);
  absl::Status InitializeNormal(float standard_deviation, uint64_t seed);

  absl::StatusOr<Buffer> fwd(absl::Span<const Buffer> inputs,
                              Tape* tape) override;
  absl::StatusOr<BufferVec> bwd(
      absl::Span<const Buffer> output_gradients, Tape tape) override;
  absl::Span<Buffer> weights() override { return absl::MakeSpan(weights_); }
  absl::Span<Buffer> gradients() override {
    return absl::MakeSpan(gradients_);
  }
  DataType output_type() const override { return output_type_; }

  int input_dim() const { return input_dim_; }
  int output_dim() const { return output_dim_; }

 private:
  FullyConnectedLayer(int input_dim, int output_dim, DataType data_type,
                      cudaStream_t stream, Buffer matrix, Buffer bias,
                      Buffer matrix_gradient, Buffer bias_gradient);

  int input_dim_;
  int output_dim_;
  DataType output_type_;
  cudaStream_t stream_;
  BufferVec weights_;
  BufferVec gradients_;
};

}  // namespace pluto::llm

#endif  // PLUTO_SRC_LLM_LAYERS_FULLY_CONNECTED_H_
