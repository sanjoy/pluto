#ifndef PLUTO_SRC_LLM_LAYERS_GELU_H_
#define PLUTO_SRC_LLM_LAYERS_GELU_H_

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

// Elementwise Gaussian Error Linear Unit used by GPT-2's feed-forward block.
class GeluLayer final : public Layer {
 public:
  static absl::StatusOr<std::unique_ptr<GeluLayer>> Create(
      DataType data_type, cudaStream_t stream);

  absl::StatusOr<Buffer> fwd(absl::Span<const Buffer> inputs,
                              Tape* tape) override;
  absl::StatusOr<BufferVec> bwd(
      absl::Span<const Buffer> output_gradients, Tape tape) override;
  absl::Span<Buffer> weights() override { return {}; }
  DataType output_type() const override { return output_type_; }

 private:
  GeluLayer(DataType data_type, cudaStream_t stream)
      : output_type_(data_type), stream_(stream) {}

  DataType output_type_;
  cudaStream_t stream_;
};

}  // namespace pluto::llm

#endif  // PLUTO_SRC_LLM_LAYERS_GELU_H_
