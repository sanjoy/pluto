#pragma once

#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "src/llm/layer.h"

namespace pluto::llm {

// Wraps a unary layer as x + layer(x), retaining the child's state and weights.
class ResidualLayer final : public Layer {
 public:
  explicit ResidualLayer(std::unique_ptr<Layer> layer);

  absl::Status ValidateSequenceLength(int sequence_length) const override;

  absl::Span<Buffer> weights() override { return absl::MakeSpan(weights_); }
  absl::Span<Buffer> gradients() override { return absl::MakeSpan(gradients_); }
  DataType output_type() const override { return layer_->output_type(); }

 private:
  absl::StatusOr<Buffer> fwd_impl(cuda::Executor& executor,
                                  absl::Span<const Buffer> inputs,
                                  BackwardState& state) const override;
  absl::StatusOr<BufferVec> bwd_impl(cuda::Executor& executor,
                                     absl::Span<const Buffer> output_gradients,
                                     BackwardState state) override;

  std::unique_ptr<Layer> layer_;
  std::vector<Buffer> weights_;
  std::vector<Buffer> gradients_;
};

// Sequentially composes unary layers. Multi-input terminal operations, such as
// cross entropy with labels, intentionally remain outside the predictor.
class ComposedLayer final : public Layer {
 public:
  ComposedLayer(DataType data_type, std::vector<std::unique_ptr<Layer>> layers);

  absl::Status ValidateSequenceLength(int sequence_length) const override;

  absl::Span<Buffer> weights() override { return absl::MakeSpan(weights_); }
  absl::Span<Buffer> gradients() override { return absl::MakeSpan(gradients_); }
  DataType output_type() const override { return output_type_; }

 private:
  absl::StatusOr<Buffer> fwd_impl(cuda::Executor& executor,
                                  absl::Span<const Buffer> inputs,
                                  BackwardState& state) const override;
  absl::StatusOr<BufferVec> bwd_impl(cuda::Executor& executor,
                                     absl::Span<const Buffer> output_gradients,
                                     BackwardState state) override;

  DataType output_type_;
  std::vector<std::unique_ptr<Layer>> layers_;
  std::vector<Buffer> weights_;
  std::vector<Buffer> gradients_;
};

// Incrementally assembles a ComposedLayer while retaining ownership of every
// child. Pointers returned by back() remain valid when more children are added
// and after create() transfers the children into the resulting layer.
class ComposedLayerBuilder final {
 public:
  // Adds an infallibly-created child to the end of the composition. A null
  // child is rejected so back() and create() never expose an invalid layer.
  absl::Status add(std::unique_ptr<Layer> layer);

  // Propagates a failed layer factory, or transfers its successful result into
  // the composition. Accepting the concrete LayerType preserves convenient
  // calls such as `RETURN_IF_ERROR(builder.add(MyLayer::Create(...)))`.
  template <class LayerType>
  absl::Status add(absl::StatusOr<std::unique_ptr<LayerType>> layer_or_error) {
    static_assert(std::is_base_of_v<Layer, LayerType>,
                  "ComposedLayerBuilder children must derive from Layer");
    if (!layer_or_error.ok())
      return layer_or_error.status();
    return add(std::move(layer_or_error).value());
  }

  // Returns the most recently added child, or nullptr when the builder is
  // empty. The builder or created ComposedLayer retains ownership.
  Layer* back();
  const Layer* back() const;

  // Consumes the accumulated children. The composed output type is inferred
  // from the final child. Building an empty composition is an error.
  absl::StatusOr<std::unique_ptr<ComposedLayer>> create();

 private:
  std::vector<std::unique_ptr<Layer>> layers_;
};

// CPU counterpart of ResidualLayer. The branch is itself a reference layer,
// so the complete residual forward and backward graphs stay on the host.
class ResidualLayerReference final : public LayerReference {
 public:
  explicit ResidualLayerReference(std::unique_ptr<LayerReference> layer);

  absl::Span<HostBuffer> weights() override { return absl::MakeSpan(weights_); }
  absl::Span<HostBuffer> gradients() override {
    return absl::MakeSpan(gradients_);
  }
  DataType output_type() const override { return layer_->output_type(); }

 private:
  absl::StatusOr<HostBuffer> fwd_impl(
      absl::Span<const HostBuffer> inputs,
      ReferenceBackwardState& state) const override;
  absl::StatusOr<HostBufferVec> bwd_impl(
      absl::Span<const HostBuffer> output_gradients,
      ReferenceBackwardState state) override;

  std::unique_ptr<LayerReference> layer_;
  std::vector<HostBuffer> weights_;
  std::vector<HostBuffer> gradients_;
};

// CPU counterpart of ComposedLayer. Each child retains its own
// ReferenceBackwardState, making reverse traversal match the production graph
// exactly.
class ComposedLayerReference final : public LayerReference {
 public:
  ComposedLayerReference(DataType data_type,
                         std::vector<std::unique_ptr<LayerReference>> layers);

  absl::Span<HostBuffer> weights() override { return absl::MakeSpan(weights_); }
  absl::Span<HostBuffer> gradients() override {
    return absl::MakeSpan(gradients_);
  }
  DataType output_type() const override { return output_type_; }

 private:
  absl::StatusOr<HostBuffer> fwd_impl(
      absl::Span<const HostBuffer> inputs,
      ReferenceBackwardState& state) const override;
  absl::StatusOr<HostBufferVec> bwd_impl(
      absl::Span<const HostBuffer> output_gradients,
      ReferenceBackwardState state) override;

  DataType output_type_;
  std::vector<std::unique_ptr<LayerReference>> layers_;
  std::vector<HostBuffer> weights_;
  std::vector<HostBuffer> gradients_;
};

class ComposedLayerReferenceBuilder final {
 public:
  absl::Status add(std::unique_ptr<LayerReference> layer);
  template <class LayerType>
  absl::Status add(absl::StatusOr<std::unique_ptr<LayerType>> layer_or_error) {
    static_assert(std::is_base_of_v<LayerReference, LayerType>,
                  "reference children must derive from LayerReference");
    if (!layer_or_error.ok())
      return layer_or_error.status();
    return add(std::move(layer_or_error).value());
  }
  LayerReference* back();
  const LayerReference* back() const;
  absl::StatusOr<std::unique_ptr<ComposedLayerReference>> create();

 private:
  std::vector<std::unique_ptr<LayerReference>> layers_;
};

}  // namespace pluto::llm
