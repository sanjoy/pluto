#ifndef PLUTO_SRC_LLM_LAYERS_NORM_H_
#define PLUTO_SRC_LLM_LAYERS_NORM_H_

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

// Per-token layer normalization without affine parameters. Omitting gamma and
// beta keeps this compact fixture focused on attention while preserving GPT-2's
// pre-normalization topology.
class LayerNormLayer final : public Layer {
 public:
  static absl::StatusOr<std::unique_ptr<LayerNormLayer>> Create(
      int embedding_dim, float epsilon, DataType data_type,
      cudaStream_t stream);

  absl::StatusOr<Buffer> fwd(absl::Span<const Buffer> inputs,
                              Tape* tape) override;
  absl::StatusOr<BufferVec> bwd(
      absl::Span<const Buffer> output_gradients, Tape tape) override;
  absl::Span<Buffer> weights() override { return {}; }
  DataType output_type() const override { return output_type_; }

 private:
  LayerNormLayer(int embedding_dim, float epsilon, DataType data_type,
                 cudaStream_t stream)
      : embedding_dim_(embedding_dim),
        epsilon_(epsilon),
        output_type_(data_type),
        stream_(stream) {}

  int embedding_dim_;
  float epsilon_;
  DataType output_type_;
  cudaStream_t stream_;
};

}  // namespace pluto::llm

#endif  // PLUTO_SRC_LLM_LAYERS_NORM_H_
