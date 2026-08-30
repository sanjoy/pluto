#pragma once

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
                              Tape* tape) const override;
  absl::StatusOr<BufferVec> bwd(
      absl::Span<const Buffer> output_gradients, Tape tape) override;
  absl::Span<Buffer> weights() override { return {}; }
  DataType output_type() const override { return output_type_; }

  int vocab_size() const { return vocab_size_; }
  int padded_vocab_size() const { return padded_vocab_size_; }

 private:
  CrossEntropyLossLayer(int vocab_size, int padded_vocab_size,
                        DataType data_type,
                        cudaStream_t stream)
      : vocab_size_(vocab_size),
        padded_vocab_size_(padded_vocab_size),
        output_type_(data_type),
        stream_(stream) {}

  int vocab_size_;
  int padded_vocab_size_;
  DataType output_type_;
  cudaStream_t stream_;
};

// Scalar log-sum-exp reference for the terminal cross-entropy operation.
class CrossEntropyLossLayerReference final : public LayerReference {
 public:
  static absl::StatusOr<std::unique_ptr<CrossEntropyLossLayerReference>>
  Create(int vocabulary_size, DataType data_type);

  absl::StatusOr<HostBuffer> fwd(absl::Span<const HostBuffer> inputs,
                                  ReferenceTape* tape) override;
  absl::StatusOr<HostBufferVec> bwd(
      absl::Span<const HostBuffer> output_gradients,
      ReferenceTape tape) override;
  absl::Span<HostBuffer> weights() override { return {}; }
  DataType output_type() const override { return output_type_; }

  int vocab_size() const { return vocab_size_; }
  int padded_vocab_size() const { return padded_vocab_size_; }

 private:
  CrossEntropyLossLayerReference(int vocab_size, int padded_vocab_size,
                                 DataType data_type)
      : vocab_size_(vocab_size),
        padded_vocab_size_(padded_vocab_size),
        output_type_(data_type) {}

  int vocab_size_;
  int padded_vocab_size_;
  DataType output_type_;
};

}  // namespace pluto::llm
