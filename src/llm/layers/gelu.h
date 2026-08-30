#pragma once

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
      DataType data_type, cuda::Executor& executor);

  absl::StatusOr<Buffer> fwd(absl::Span<const Buffer> inputs, Tape* tape,
                             cuda::Executor& executor) const override;
  absl::StatusOr<BufferVec> bwd(absl::Span<const Buffer> output_gradients,
                                Tape tape, cuda::Executor& executor) override;
  absl::Span<Buffer> weights() override { return {}; }
  DataType output_type() const override { return output_type_; }

 private:
  GeluLayer(DataType data_type, cuda::Executor& executor)
      : output_type_(data_type), executor_(executor) {}

  DataType output_type_;
  cuda::Executor& executor_;
};

class GeluLayerReference final : public LayerReference {
 public:
  static absl::StatusOr<std::unique_ptr<GeluLayerReference>> Create(
      DataType data_type);

  absl::StatusOr<HostBuffer> fwd(absl::Span<const HostBuffer> inputs,
                                 ReferenceTape* tape) override;
  absl::StatusOr<HostBufferVec> bwd(
      absl::Span<const HostBuffer> output_gradients,
      ReferenceTape tape) override;
  absl::Span<HostBuffer> weights() override { return {}; }
  DataType output_type() const override { return output_type_; }

 private:
  explicit GeluLayerReference(DataType data_type) : output_type_(data_type) {}

  DataType output_type_;
};

}  // namespace pluto::llm
