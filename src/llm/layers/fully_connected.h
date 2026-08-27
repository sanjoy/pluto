#ifndef PLUTO_SRC_LLM_LAYERS_FULLY_CONNECTED_H_
#define PLUTO_SRC_LLM_LAYERS_FULLY_CONNECTED_H_

#include <cuda_runtime_api.h>

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "src/llm/layer.h"

namespace pluto::llm {

// A bias-bearing model_width x model_width dense layer. Matrix products are
// implemented as 16x16 cuTile MMAs. FP32 master parameters are rounded to FP16
// at the MMA boundary, which keeps updates stable while exercising FP16 math.
class FullyConnectedLayer final : public Layer {
 public:
  static absl::StatusOr<std::unique_ptr<FullyConnectedLayer>> Create(
      int model_width, DataType data_type, float learning_rate,
      cudaStream_t stream);

  // Makes this layer an exact identity at its FP16 compute boundary. Useful
  // when inserting it into a small model without changing its initial logits.
  absl::Status InitializeIdentity(float scale = 1.0f);

  absl::StatusOr<Buffer> fwd(absl::Span<const Buffer> inputs,
                              Tape* tape) override;
  absl::StatusOr<BufferVec> bwd(
      absl::Span<const Buffer> output_gradients, Tape tape) override;
  absl::Span<Buffer> weights() override {
    return absl::MakeSpan(weights_);
  }
  DataType output_type() const override { return output_type_; }

 private:
  FullyConnectedLayer(int model_width, DataType data_type,
                      float learning_rate, cudaStream_t stream, Buffer matrix,
                      Buffer bias);

  int model_width_;
  DataType output_type_;
  float learning_rate_;
  cudaStream_t stream_;
  BufferVec weights_;
};

}  // namespace pluto::llm

#endif  // PLUTO_SRC_LLM_LAYERS_FULLY_CONNECTED_H_
