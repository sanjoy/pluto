#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "gtest/gtest.h"
#include "src/llm/layer.h"
#include "src/llm/layers/combinators.h"

namespace pluto::llm {
namespace {

// Exercise the same ownership contract entirely on the CPU. Keeping the
// implementations private also checks that callers use the inherited API.
class IdentityLayerReference final : public LayerReference {
 public:
  absl::Span<HostBuffer> weights() override { return {}; }
  DataType output_type() const override { return DataType::FP16; }

  mutable int forward_calls = 0;
  int backward_calls = 0;
  bool fail_forward = false;
  bool fail_backward = false;

 private:
  absl::StatusOr<ReferenceFwdResult> fwd_impl(
      absl::Span<const HostBuffer> inputs) const override {
    ReferenceBackwardState state;
    ++forward_calls;
    EXPECT_EQ(state.layer, nullptr);
    EXPECT_TRUE(state.intermediates.empty());
    EXPECT_TRUE(state.children.empty());
    // Populate state even on failure to exercise partially written states.
    state.intermediates = {inputs[0]};
    if (fail_forward) return absl::ResourceExhaustedError("forward failed");
    return ReferenceFwdResult{{std::move(inputs[0])}, std::move(state)};
  }

  absl::StatusOr<HostBufferVec> bwd_impl(
      absl::Span<const HostBuffer>, ReferenceBackwardState state) override {
    ++backward_calls;
    EXPECT_EQ(state.layer, this);
    if (fail_backward) return absl::InternalError("backward failed");
    return std::move(state.intermediates);
  }
};

// Expose a parameter as a second output, like SAE's decoder output. Recording
// both derivatives makes the composition's reverse routing observable.
class ParameterOutputLayer final : public LayerReference {
 public:
  explicit ParameterOutputLayer(HostBuffer parameter)
      : parameters_{std::move(parameter)} {}
  absl::Span<HostBuffer> weights() override {
    return absl::MakeSpan(parameters_);
  }
  DataType output_type() const override { return DataType::FP16; }
  HostBufferVec received_gradients;
  HostBufferVec saved_inputs;

 private:
  absl::StatusOr<ReferenceFwdResult> fwd_impl(
      absl::Span<const HostBuffer> inputs) const override {
    if (inputs.size() != 1)
      return absl::InvalidArgumentError("expected one input");
    ReferenceBackwardState state;
    state.intermediates = {inputs[0]};
    return ReferenceFwdResult{{inputs[0], parameters_[0]}, std::move(state)};
  }
  absl::StatusOr<HostBufferVec> bwd_impl(
      absl::Span<const HostBuffer> gradients,
      ReferenceBackwardState state) override {
    if (gradients.size() != 2)
      return absl::InvalidArgumentError("expected two gradients");
    received_gradients.assign(gradients.begin(), gradients.end());
    saved_inputs = std::move(state.intermediates);
    return HostBufferVec{gradients[0]};
  }
  HostBufferVec parameters_;
};

class SwapOutputsLayer final : public LayerReference {
 public:
  absl::Span<HostBuffer> weights() override { return {}; }
  DataType output_type() const override { return DataType::FP16; }

 private:
  absl::StatusOr<ReferenceFwdResult> fwd_impl(
      absl::Span<const HostBuffer> inputs) const override {
    if (inputs.size() != 2)
      return absl::InvalidArgumentError("expected two inputs");
    return ReferenceFwdResult{{inputs[1], inputs[0]}, {}};
  }
  absl::StatusOr<HostBufferVec> bwd_impl(absl::Span<const HostBuffer> gradients,
                                         ReferenceBackwardState) override {
    if (gradients.size() != 2)
      return absl::InvalidArgumentError("expected two gradients");
    return HostBufferVec{gradients[1], gradients[0]};
  }
};

class LayerReferenceStateTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto input = HostBuffer::Allocate(sizeof(float));
    ASSERT_TRUE(input.ok()) << input.status();
    inputs_.push_back(std::move(*input));
  }

  HostBufferVec inputs_;
};

TEST_F(LayerReferenceStateTest, ResultContainsOutputAndStateWithMatchingOwner) {
  IdentityLayerReference layer;
  const LayerReference& forward_layer = layer;
  auto result = forward_layer.fwd(inputs_);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result->outputs[0].data(), inputs_[0].data());
  EXPECT_EQ(result->state.layer, &layer);
  EXPECT_EQ(layer.forward_calls, 1);

  // Moving the complete result and copying its state preserve layer identity.
  ReferenceFwdResult moved = std::move(*result);
  ReferenceBackwardState copy = moved.state;
  LayerReference& backward_layer = layer;
  auto gradient = backward_layer.bwd(inputs_, std::move(copy));
  ASSERT_TRUE(gradient.ok()) << gradient.status();
  ASSERT_EQ(gradient->size(), 1);
  EXPECT_EQ((*gradient)[0].data(), inputs_[0].data());
  EXPECT_TRUE(layer.bwd(inputs_, std::move(moved.state)).ok());
  EXPECT_EQ(layer.backward_calls, 2);
  // Consuming backward state does not consume the returned output handle.
  EXPECT_EQ(moved.outputs[0].data(), inputs_[0].data());
}

TEST(LayerReferenceApiTest, ForwardReturnsResultWithoutStateArgument) {
  static_assert(
      std::is_same_v<std::invoke_result_t<decltype(&LayerReference::fwd),
                                          const LayerReference&,
                                          absl::Span<const HostBuffer>>,
                     absl::StatusOr<ReferenceFwdResult>>);
  static_assert(
      !std::is_invocable_v<decltype(&LayerReference::fwd),
                           const LayerReference&, absl::Span<const HostBuffer>,
                           ReferenceBackwardState&>);
}

TEST_F(LayerReferenceStateTest, UnusedStateDoesNotDispatchBackward) {
  IdentityLayerReference layer;
  auto gradient = layer.bwd(inputs_, ReferenceBackwardState{});
  EXPECT_EQ(gradient.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(layer.backward_calls, 0);
}

TEST_F(LayerReferenceStateTest, AnotherLayersResultDoesNotDispatchBackward) {
  IdentityLayerReference first;
  IdentityLayerReference second;
  auto result = first.fwd(inputs_);
  ASSERT_TRUE(result.ok()) << result.status();
  auto gradient = second.bwd(inputs_, result->state);
  EXPECT_EQ(gradient.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(second.backward_calls, 0);
  EXPECT_TRUE(first.bwd(inputs_, std::move(result->state)).ok());
}

TEST_F(LayerReferenceStateTest,
       FailedForwardPublishesNoStateAndPreservesEarlierResult) {
  IdentityLayerReference layer;
  auto earlier = layer.fwd(inputs_);
  ASSERT_TRUE(earlier.ok()) << earlier.status();
  layer.fail_forward = true;
  auto failed = layer.fwd(inputs_);
  EXPECT_EQ(failed.status(), absl::ResourceExhaustedError("forward failed"));
  EXPECT_EQ(layer.forward_calls, 2);
  EXPECT_EQ(earlier->state.layer, &layer);
  EXPECT_EQ(earlier->outputs[0].data(), inputs_[0].data());
  EXPECT_TRUE(layer.bwd(inputs_, std::move(earlier->state)).ok());
  EXPECT_EQ(layer.backward_calls, 1);
}

TEST_F(LayerReferenceStateTest, ConsecutiveResultsKeepIndependentSavedInputs) {
  IdentityLayerReference layer;
  auto first = layer.fwd(inputs_);
  ASSERT_TRUE(first.ok()) << first.status();
  auto other_input = HostBuffer::Allocate(sizeof(float));
  ASSERT_TRUE(other_input.ok()) << other_input.status();
  HostBufferVec other_inputs{std::move(*other_input)};
  auto second = layer.fwd(other_inputs);
  ASSERT_TRUE(second.ok()) << second.status();
  ASSERT_EQ(first->state.intermediates.size(), 1);
  ASSERT_EQ(second->state.intermediates.size(), 1);
  EXPECT_EQ(first->state.intermediates[0].data(), inputs_[0].data());
  EXPECT_EQ(second->state.intermediates[0].data(), other_inputs[0].data());
  EXPECT_EQ(first->state.layer, &layer);
  EXPECT_EQ(second->state.layer, &layer);
}

TEST_F(LayerReferenceStateTest, BackwardImplementationErrorIsPreserved) {
  IdentityLayerReference layer;
  auto result = layer.fwd(inputs_);
  ASSERT_TRUE(result.ok()) << result.status();
  layer.fail_backward = true;
  auto gradient = layer.bwd(inputs_, std::move(result->state));
  EXPECT_EQ(gradient.status(), absl::InternalError("backward failed"));
  EXPECT_EQ(layer.backward_calls, 1);
}

TEST_F(LayerReferenceStateTest,
       NestedCompositionChecksIndividualChildIdentity) {
  auto first = std::make_unique<IdentityLayerReference>();
  auto second = std::make_unique<IdentityLayerReference>();
  const auto* first_ptr = first.get();
  const auto* second_ptr = second.get();
  std::vector<std::unique_ptr<LayerReference>> children;
  children.push_back(std::move(first));
  children.push_back(std::move(second));
  auto inner = std::make_unique<ComposedLayerReference>(DataType::FP16,
                                                        std::move(children));
  const auto* inner_ptr = inner.get();
  std::vector<std::unique_ptr<LayerReference>> outer_children;
  outer_children.push_back(std::move(inner));
  ComposedLayerReference outer(DataType::FP16, std::move(outer_children));

  auto result = outer.fwd(inputs_);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result->state.layer, &outer);
  ASSERT_EQ(result->state.children.size(), 1);
  EXPECT_EQ(result->state.children[0].layer, inner_ptr);
  auto& child_states = result->state.children[0].children;
  ASSERT_EQ(child_states.size(), 2);
  EXPECT_EQ(child_states[0].layer, first_ptr);
  EXPECT_EQ(child_states[1].layer, second_ptr);
  ASSERT_TRUE(outer.bwd(inputs_, result->state).ok());
  EXPECT_EQ(first_ptr->backward_calls, 1);
  EXPECT_EQ(second_ptr->backward_calls, 1);

  // The parent state is valid, but the children are attached to wrong slots.
  std::swap(child_states[0], child_states[1]);
  auto gradient = outer.bwd(inputs_, std::move(result->state));
  EXPECT_EQ(gradient.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(first_ptr->backward_calls, 1);
  EXPECT_EQ(second_ptr->backward_calls, 1);
}

TEST_F(LayerReferenceStateTest,
       CompositionRoutesMultipleOutputsAndInterleavedStates) {
  auto parameter = HostBuffer::Allocate(sizeof(float));
  auto other_input = HostBuffer::Allocate(sizeof(float));
  ASSERT_TRUE(parameter.ok()) << parameter.status();
  ASSERT_TRUE(other_input.ok()) << other_input.status();
  auto source = std::make_unique<ParameterOutputLayer>(*parameter);
  auto* source_ptr = source.get();
  std::vector<std::unique_ptr<LayerReference>> children;
  children.push_back(std::move(source));
  children.push_back(std::make_unique<SwapOutputsLayer>());
  ComposedLayerReference model(DataType::FP16, std::move(children));

  auto first = model.fwd(inputs_);
  auto second = model.fwd(HostBufferVec{*other_input});
  ASSERT_TRUE(first.ok()) << first.status();
  ASSERT_TRUE(second.ok()) << second.status();
  ASSERT_EQ(first->outputs.size(), 2u);
  ASSERT_EQ(second->outputs.size(), 2u);
  EXPECT_EQ(first->outputs[0].data(), parameter->data());
  EXPECT_EQ(first->outputs[1].data(), inputs_[0].data());
  EXPECT_EQ(second->outputs[1].data(), other_input->data());

  HostBufferVec gradients{*parameter, *other_input};
  auto first_gradient = model.bwd(gradients, std::move(first->state));
  ASSERT_TRUE(first_gradient.ok()) << first_gradient.status();
  ASSERT_EQ(first_gradient->size(), 1u);
  EXPECT_EQ((*first_gradient)[0].data(), other_input->data());
  ASSERT_EQ(source_ptr->received_gradients.size(), 2u);
  EXPECT_EQ(source_ptr->received_gradients[0].data(), other_input->data());
  EXPECT_EQ(source_ptr->received_gradients[1].data(), parameter->data());
  EXPECT_EQ(source_ptr->saved_inputs[0].data(), inputs_[0].data());

  auto second_gradient = model.bwd(gradients, std::move(second->state));
  ASSERT_TRUE(second_gradient.ok()) << second_gradient.status();
  EXPECT_EQ(source_ptr->saved_inputs[0].data(), other_input->data());
  EXPECT_EQ(first->outputs[1].data(), inputs_[0].data());
}

TEST_F(LayerReferenceStateTest,
       CompositionAcceptsAndReturnsMultipleInputsAndGradients) {
  auto second = HostBuffer::Allocate(sizeof(float));
  ASSERT_TRUE(second.ok()) << second.status();
  std::vector<std::unique_ptr<LayerReference>> children;
  children.push_back(std::make_unique<SwapOutputsLayer>());
  ComposedLayerReference model(DataType::FP16, std::move(children));
  HostBufferVec inputs{inputs_[0], *second};
  auto result = model.fwd(inputs);
  ASSERT_TRUE(result.ok()) << result.status();
  ASSERT_EQ(result->outputs.size(), 2u);
  EXPECT_EQ(result->outputs[0].data(), second->data());
  EXPECT_EQ(result->outputs[1].data(), inputs_[0].data());
  auto gradients = model.bwd(inputs, std::move(result->state));
  ASSERT_TRUE(gradients.ok()) << gradients.status();
  ASSERT_EQ(gradients->size(), 2u);
  EXPECT_EQ((*gradients)[0].data(), second->data());
  EXPECT_EQ((*gradients)[1].data(), inputs_[0].data());
}

TEST_F(LayerReferenceStateTest, ResidualRejectsMultipleBranchOutputs) {
  ResidualLayerReference residual(
      std::make_unique<ParameterOutputLayer>(inputs_[0]));
  auto result = residual.fwd(inputs_);
  EXPECT_EQ(result.status().code(), absl::StatusCode::kInvalidArgument);
}

}  // namespace
}  // namespace pluto::llm
