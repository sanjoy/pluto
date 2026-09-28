#pragma once

#include <memory>
#include <string>
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
  absl::string_view name() const override { return "ResidualLayer"; }

  // Rejects null, non-unary, or shape/dtype-changing branches before use.
  static absl::StatusOr<std::unique_ptr<ResidualLayer>> Create(
      std::unique_ptr<Layer> layer);

  absl::Span<Buffer> weights() override { return absl::MakeSpan(weights_); }
  absl::Span<Buffer> gradients() override { return absl::MakeSpan(gradients_); }
  DataType output_type() const override { return layer_->output_type(); }

  absl::Span<const ActivationType> input_types() const override {
    return layer_->input_types();
  }
  absl::Span<const ActivationType> output_types() const override {
    return layer_->output_types();
  }

 private:
  explicit ResidualLayer(std::unique_ptr<Layer> layer);

  absl::StatusOr<FwdResult> fwd_impl(cuda::Executor& executor,
                                     absl::Span<const Buffer> inputs,
                                     LayerHooks* hooks) const override;
  absl::StatusOr<BufferVec> bwd_impl(cuda::Executor& executor,
                                     absl::Span<const Buffer> output_gradients,
                                     BackwardState state,
                                     LayerHooks* hooks) override;

  // Work inside this layer's hook scope; the impl wrappers bracket these calls.
  absl::StatusOr<FwdResult> fwd_body(cuda::Executor& executor,
                                     absl::Span<const Buffer> inputs,
                                     LayerHooks* hooks) const;
  absl::StatusOr<BufferVec> bwd_body(cuda::Executor& executor,
                                     absl::Span<const Buffer> output_gradients,
                                     BackwardState state, LayerHooks* hooks);

  std::unique_ptr<Layer> layer_;
  std::vector<Buffer> weights_;
  std::vector<Buffer> gradients_;
};

// Sequentially composes layers, passing complete output and gradient vectors.
// Adjacent forward signatures must match exactly, including arity, dtype,
// rank, and dimensions. No broadcasting or flattening is implicit.
class ComposedLayer final : public Layer {
 public:
  absl::string_view name() const override { return name_; }

  // Owns the supplied nonempty diagnostic name. Requires at least one non-null
  // child and exact adjacent signatures. Names need not be globally unique.
  static absl::StatusOr<std::unique_ptr<ComposedLayer>> Create(
      std::string name, std::vector<std::unique_ptr<Layer>> layers);

  absl::Span<Buffer> weights() override { return absl::MakeSpan(weights_); }
  absl::Span<Buffer> gradients() override { return absl::MakeSpan(gradients_); }
  DataType output_type() const override { return output_type_; }

  absl::Span<const ActivationType> input_types() const override {
    return layers_.front()->input_types();
  }
  absl::Span<const ActivationType> output_types() const override {
    return layers_.back()->output_types();
  }

 private:
  ComposedLayer(std::string name, std::vector<std::unique_ptr<Layer>> layers);

  absl::StatusOr<FwdResult> fwd_impl(cuda::Executor& executor,
                                     absl::Span<const Buffer> inputs,
                                     LayerHooks* hooks) const override;
  absl::StatusOr<BufferVec> bwd_impl(cuda::Executor& executor,
                                     absl::Span<const Buffer> output_gradients,
                                     BackwardState state,
                                     LayerHooks* hooks) override;

  // Work inside this layer's hook scope; the impl wrappers bracket these calls.
  absl::StatusOr<FwdResult> fwd_body(cuda::Executor& executor,
                                     absl::Span<const Buffer> inputs,
                                     LayerHooks* hooks) const;
  absl::StatusOr<BufferVec> bwd_body(cuda::Executor& executor,
                                     absl::Span<const Buffer> output_gradients,
                                     BackwardState state, LayerHooks* hooks);

  std::string name_;
  DataType output_type_;
  std::vector<std::unique_ptr<Layer>> layers_;
  std::vector<Buffer> weights_;
  std::vector<Buffer> gradients_;
};

// Sends the complete input vector to every child and concatenates their
// outputs in child order. Backward splits that vector and sums corresponding
// FP32 input gradients without modifying the buffers returned by a child.
// Children must have identical input signatures; output signatures may differ.
// Children consuming nondifferentiable inputs may all return no gradients.
class ParallelLayer final : public Layer {
 public:
  absl::string_view name() const override { return name_; }

  // Owns the nonempty name and at least one non-null child. output_type() is
  // the final child compute policy; output_types() describes every buffer.
  static absl::StatusOr<std::unique_ptr<ParallelLayer>> Create(
      std::string name, std::vector<std::unique_ptr<Layer>> layers);

  absl::Span<Buffer> weights() override { return absl::MakeSpan(weights_); }
  absl::Span<Buffer> gradients() override { return absl::MakeSpan(gradients_); }
  DataType output_type() const override {
    return layers_.back()->output_type();
  }
  absl::Span<const ActivationType> input_types() const override {
    return layers_.front()->input_types();
  }
  absl::Span<const ActivationType> output_types() const override {
    return output_types_;
  }

 private:
  ParallelLayer(std::string name, std::vector<std::unique_ptr<Layer>> layers);

  absl::StatusOr<FwdResult> fwd_impl(cuda::Executor& executor,
                                     absl::Span<const Buffer> inputs,
                                     LayerHooks* hooks) const override;
  absl::StatusOr<BufferVec> bwd_impl(cuda::Executor& executor,
                                     absl::Span<const Buffer> output_gradients,
                                     BackwardState state,
                                     LayerHooks* hooks) override;
  absl::StatusOr<FwdResult> fwd_body(cuda::Executor& executor,
                                     absl::Span<const Buffer> inputs,
                                     LayerHooks* hooks) const;
  absl::StatusOr<BufferVec> bwd_body(cuda::Executor& executor,
                                     absl::Span<const Buffer> output_gradients,
                                     BackwardState state, LayerHooks* hooks);

  std::string name_;
  std::vector<std::unique_ptr<Layer>> layers_;
  std::vector<ActivationType> output_types_;
  std::vector<Buffer> weights_;
  std::vector<Buffer> gradients_;
};

// Incrementally assembles a ComposedLayer while retaining ownership of every
// child. Pointers returned by back() remain valid when more children are added
// and after create() transfers the children into the resulting layer.
class ComposedLayerBuilder final {
 public:
  // Adds an infallibly-created child to the end of the composition. A null
  // child or an incompatible signature is rejected without modifying the
  // builder.
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

  // Consumes the accumulated children and owns the nonempty name. The composed
  // output type is inferred from the final child. An empty name is rejected
  // without consuming any children; building an empty composition is an error.
  absl::StatusOr<std::unique_ptr<ComposedLayer>> create(std::string name);

 private:
  std::vector<std::unique_ptr<Layer>> layers_;
};

// CPU counterpart of ResidualLayer. The branch is itself a reference layer,
// so the complete residual forward and backward graphs stay on the host.
class ResidualLayerReference final : public LayerReference {
 public:
  absl::string_view name() const override { return "ResidualLayerReference"; }

  // Rejects null, non-unary, or shape/dtype-changing branches before use.
  static absl::StatusOr<std::unique_ptr<ResidualLayerReference>> Create(
      std::unique_ptr<LayerReference> layer);

  absl::Span<HostBuffer> weights() override { return absl::MakeSpan(weights_); }
  absl::Span<HostBuffer> gradients() override {
    return absl::MakeSpan(gradients_);
  }
  DataType output_type() const override { return layer_->output_type(); }

  absl::Span<const ActivationType> input_types() const override {
    return layer_->input_types();
  }
  absl::Span<const ActivationType> output_types() const override {
    return layer_->output_types();
  }

 private:
  explicit ResidualLayerReference(std::unique_ptr<LayerReference> layer);

  absl::StatusOr<ReferenceFwdResult> fwd_impl(
      absl::Span<const HostBuffer> inputs) const override;
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
  absl::string_view name() const override { return name_; }

  // Owns the supplied nonempty diagnostic name. Requires at least one non-null
  // child and exact adjacent signatures, just like the GPU composition.
  static absl::StatusOr<std::unique_ptr<ComposedLayerReference>> Create(
      std::string name, std::vector<std::unique_ptr<LayerReference>> layers);

  absl::Span<HostBuffer> weights() override { return absl::MakeSpan(weights_); }
  absl::Span<HostBuffer> gradients() override {
    return absl::MakeSpan(gradients_);
  }
  DataType output_type() const override { return output_type_; }

  absl::Span<const ActivationType> input_types() const override {
    return layers_.front()->input_types();
  }
  absl::Span<const ActivationType> output_types() const override {
    return layers_.back()->output_types();
  }

 private:
  ComposedLayerReference(std::string name,
                         std::vector<std::unique_ptr<LayerReference>> layers);

  absl::StatusOr<ReferenceFwdResult> fwd_impl(
      absl::Span<const HostBuffer> inputs) const override;
  absl::StatusOr<HostBufferVec> bwd_impl(
      absl::Span<const HostBuffer> output_gradients,
      ReferenceBackwardState state) override;

  std::string name_;
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
  // Same ownership/error rules as ComposedLayerBuilder::create: an empty name
  // leaves all children in this builder so the caller can retry.
  absl::StatusOr<std::unique_ptr<ComposedLayerReference>> create(
      std::string name);

 private:
  std::vector<std::unique_ptr<LayerReference>> layers_;
};

}  // namespace pluto::llm
