#ifndef PLUTO_SRC_LLM_LAYERS_CROSS_ENTROPY_LOSS_H_
#define PLUTO_SRC_LLM_LAYERS_CROSS_ENTROPY_LOSS_H_

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

// Computes one cross-entropy value per batch element. fwd() takes two inputs:
// [logits(float), target_token(int32)]. bwd() takes no upstream gradient
// because this is a terminal loss and returns the mean-loss gradient for the
// logits. Keeping per-example losses makes diagnostics and tests more useful.
class CrossEntropyLossLayer final : public Layer {
 public:
  static absl::StatusOr<std::unique_ptr<CrossEntropyLossLayer>> Create(
      int vocabulary_size, DataType data_type, cudaStream_t stream);

  absl::StatusOr<Buffer> fwd(absl::Span<const Buffer> inputs,
                              Tape* tape) override;
  absl::StatusOr<BufferVec> bwd(
      absl::Span<const Buffer> output_gradients, Tape tape) override;
  absl::Span<Buffer> weights() override { return {}; }
  DataType output_type() const override { return output_type_; }

 private:
  CrossEntropyLossLayer(int vocabulary_size, DataType data_type,
                        cudaStream_t stream)
      : vocabulary_size_(vocabulary_size),
        output_type_(data_type),
        stream_(stream) {}

  int vocabulary_size_;
  DataType output_type_;
  cudaStream_t stream_;
};

}  // namespace pluto::llm

#endif  // PLUTO_SRC_LLM_LAYERS_CROSS_ENTROPY_LOSS_H_
