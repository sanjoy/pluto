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

// Elementwise Gaussian Error Linear Unit.
class GeluLayer final : public Layer {
 public:
  static absl::StatusOr<std::unique_ptr<GeluLayer>> Create(
      cuda::Executor& executor, DataType data_type);

  absl::Span<Buffer> weights() override { return {}; }
  DataType output_type() const override { return output_type_; }

 private:
  absl::StatusOr<FwdResult> fwd_impl(
      cuda::Executor& executor, absl::Span<const Buffer> inputs) const override;
  absl::StatusOr<BufferVec> bwd_impl(cuda::Executor& executor,
                                     absl::Span<const Buffer> output_gradients,
                                     BackwardState state) override;

  GeluLayer(cuda::Executor& executor, DataType data_type)
      : output_type_(data_type), executor_(executor) {}

  DataType output_type_;
  cuda::Executor& executor_;
};

class GeluLayerReference final : public LayerReference {
 public:
  static absl::StatusOr<std::unique_ptr<GeluLayerReference>> Create(
      DataType data_type);

  absl::Span<HostBuffer> weights() override { return {}; }
  DataType output_type() const override { return output_type_; }

 private:
  absl::StatusOr<ReferenceFwdResult> fwd_impl(
      absl::Span<const HostBuffer> inputs) const override;
  absl::StatusOr<HostBufferVec> bwd_impl(
      absl::Span<const HostBuffer> output_gradients,
      ReferenceBackwardState state) override;

  explicit GeluLayerReference(DataType data_type) : output_type_(data_type) {}

  DataType output_type_;
};

}  // namespace pluto::llm
