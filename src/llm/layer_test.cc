#include "src/llm/layer.h"

#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "gtest/gtest.h"
#include "src/llm/layers/combinators.h"

namespace pluto::llm {
namespace {

// No kernels are needed to test dispatch and state ownership. Keeping the
// implementations private also checks that callers use the inherited API.
class IdentityLayer final : public Layer {
 public:
  absl::string_view name() const override { return "IdentityLayer"; }

  absl::Span<Buffer> weights() override { return {}; }
  DataType output_type() const override { return DataType::FP16; }

  mutable int forward_calls = 0;
  int backward_calls = 0;
  bool fail_forward = false;
  bool fail_backward = false;

  absl::Span<const ActivationType> input_types() const override {
    return input_types_;
  }
  absl::Span<const ActivationType> output_types() const override {
    return output_types_;
  }

 private:
  const ActivationType input_types_[1] = {{DataType::FP32, {}}};
  const ActivationType output_types_[1] = {{DataType::FP32, {}}};

  absl::StatusOr<FwdResult> fwd_impl(cuda::Executor&,
                                     absl::Span<const Buffer> inputs,
                                     LayerHooks*) const override {
    BackwardState state;
    ++forward_calls;
    EXPECT_EQ(state.layer, nullptr);
    EXPECT_TRUE(state.intermediates.empty());
    EXPECT_TRUE(state.children.empty());
    // Populate state even on failure to exercise partially written states.
    state.intermediates = {inputs[0]};
    if (fail_forward)
      return absl::ResourceExhaustedError("forward failed");
    return FwdResult{{std::move(inputs[0])}, std::move(state)};
  }

  absl::StatusOr<BufferVec> bwd_impl(cuda::Executor&, absl::Span<const Buffer>,
                                     BackwardState state,
                                     LayerHooks*) override {
    ++backward_calls;
    EXPECT_EQ(state.layer, this);
    if (fail_backward)
      return absl::InternalError("backward failed");
    return std::move(state.intermediates);
  }
};

// Expose a parameter as a second output, like SAE's decoder output. Recording
// both derivatives makes the composition's reverse routing observable.
class ParameterOutputLayer final : public Layer {
 public:
  absl::string_view name() const override { return "ParameterOutputLayer"; }

  explicit ParameterOutputLayer(Buffer parameter)
      : parameters_{std::move(parameter)} {}
  absl::Span<Buffer> weights() override { return absl::MakeSpan(parameters_); }
  DataType output_type() const override { return DataType::FP16; }
  BufferVec received_gradients;
  BufferVec saved_inputs;

  absl::Span<const ActivationType> input_types() const override {
    return input_types_;
  }
  absl::Span<const ActivationType> output_types() const override {
    return output_types_;
  }

 private:
  const ActivationType input_types_[1] = {{DataType::FP32, {}}};
  const ActivationType output_types_[2] = {{DataType::FP32, {}},
                                           {DataType::FP32, {}}};

  absl::StatusOr<FwdResult> fwd_impl(cuda::Executor&,
                                     absl::Span<const Buffer> inputs,
                                     LayerHooks*) const override {
    if (inputs.size() != 1)
      return absl::InvalidArgumentError("expected one input");
    BackwardState state;
    state.intermediates = {inputs[0]};
    return FwdResult{{inputs[0], parameters_[0]}, std::move(state)};
  }
  absl::StatusOr<BufferVec> bwd_impl(cuda::Executor&,
                                     absl::Span<const Buffer> gradients,
                                     BackwardState state,
                                     LayerHooks*) override {
    if (gradients.size() != 2)
      return absl::InvalidArgumentError("expected two gradients");
    received_gradients.assign(gradients.begin(), gradients.end());
    saved_inputs = std::move(state.intermediates);
    return BufferVec{gradients[0]};
  }
  BufferVec parameters_;
};

class SwapOutputsLayer final : public Layer {
 public:
  absl::string_view name() const override { return "SwapOutputsLayer"; }

  absl::Span<Buffer> weights() override { return {}; }
  DataType output_type() const override { return DataType::FP16; }

  absl::Span<const ActivationType> input_types() const override {
    return input_types_;
  }
  absl::Span<const ActivationType> output_types() const override {
    return output_types_;
  }

 private:
  const ActivationType input_types_[2] = {{DataType::FP32, {}},
                                          {DataType::FP32, {}}};
  const ActivationType output_types_[2] = {{DataType::FP32, {}},
                                           {DataType::FP32, {}}};

  absl::StatusOr<FwdResult> fwd_impl(cuda::Executor&,
                                     absl::Span<const Buffer> inputs,
                                     LayerHooks*) const override {
    if (inputs.size() != 2)
      return absl::InvalidArgumentError("expected two inputs");
    return FwdResult{{inputs[1], inputs[0]}, {}};
  }
  absl::StatusOr<BufferVec> bwd_impl(cuda::Executor&,
                                     absl::Span<const Buffer> gradients,
                                     BackwardState, LayerHooks*) override {
    if (gradients.size() != 2)
      return absl::InvalidArgumentError("expected two gradients");
    return BufferVec{gradients[1], gradients[0]};
  }
};

class LayerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto executor = cuda::Executor::Create();
    ASSERT_TRUE(executor.ok()) << executor.status();
    executor_ = std::move(*executor);
    auto input = Buffer::Allocate(*executor_, sizeof(float));
    ASSERT_TRUE(input.ok()) << input.status();
    inputs_.push_back(std::move(*input));
  }

  // The executor must outlive every buffer allocated through it.
  std::unique_ptr<cuda::Executor> executor_;
  BufferVec inputs_;
};

TEST_F(LayerTest, ResultContainsOutputAndStateWithMatchingOwner) {
  IdentityLayer layer;
  const Layer& forward_layer = layer;
  auto result = forward_layer.fwd(*executor_, inputs_);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result->outputs[0].data(), inputs_[0].data());
  EXPECT_EQ(result->state.layer, &layer);
  EXPECT_EQ(layer.forward_calls, 1);

  // Moving the complete result and copying its state preserve layer identity.
  FwdResult moved = std::move(*result);
  BackwardState copy = moved.state;
  Layer& backward_layer = layer;
  auto gradient = backward_layer.bwd(*executor_, inputs_, std::move(copy));
  ASSERT_TRUE(gradient.ok()) << gradient.status();
  ASSERT_EQ(gradient->size(), 1);
  EXPECT_EQ((*gradient)[0].data(), inputs_[0].data());
  EXPECT_TRUE(layer.bwd(*executor_, inputs_, std::move(moved.state)).ok());
  EXPECT_EQ(layer.backward_calls, 2);
  // Consuming backward state does not consume the returned output handle.
  EXPECT_EQ(moved.outputs[0].data(), inputs_[0].data());
}

TEST(LayerApiTest, ForwardReturnsResultWithoutStateArgument) {
  static_assert(
      std::is_same_v<std::invoke_result_t<
                         decltype(&Layer::fwd), const Layer&, cuda::Executor&,
                         absl::Span<const Buffer>, LayerHooks*>,
                     absl::StatusOr<FwdResult>>);
  static_assert(
      !std::is_invocable_v<decltype(&Layer::fwd), const Layer&, cuda::Executor&,
                           absl::Span<const Buffer>, BackwardState&>);
}

TEST_F(LayerTest, UnusedStateDoesNotDispatchBackward) {
  IdentityLayer layer;
  auto gradient = layer.bwd(*executor_, inputs_, BackwardState{});
  EXPECT_EQ(gradient.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(layer.backward_calls, 0);
}

TEST_F(LayerTest, AnotherLayersResultDoesNotDispatchBackward) {
  IdentityLayer first;
  IdentityLayer second;
  auto result = first.fwd(*executor_, inputs_);
  ASSERT_TRUE(result.ok()) << result.status();
  auto gradient = second.bwd(*executor_, inputs_, result->state);
  EXPECT_EQ(gradient.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(second.backward_calls, 0);
  EXPECT_TRUE(first.bwd(*executor_, inputs_, std::move(result->state)).ok());
}

TEST_F(LayerTest, FailedForwardPublishesNoStateAndPreservesEarlierResult) {
  IdentityLayer layer;
  auto earlier = layer.fwd(*executor_, inputs_);
  ASSERT_TRUE(earlier.ok()) << earlier.status();
  layer.fail_forward = true;
  auto failed = layer.fwd(*executor_, inputs_);
  EXPECT_EQ(failed.status(), absl::ResourceExhaustedError("forward failed"));
  EXPECT_EQ(layer.forward_calls, 2);
  EXPECT_EQ(earlier->state.layer, &layer);
  EXPECT_EQ(earlier->outputs[0].data(), inputs_[0].data());
  EXPECT_TRUE(layer.bwd(*executor_, inputs_, std::move(earlier->state)).ok());
  EXPECT_EQ(layer.backward_calls, 1);
}

TEST_F(LayerTest, ConsecutiveResultsKeepIndependentSavedInputs) {
  IdentityLayer layer;
  auto first = layer.fwd(*executor_, inputs_);
  ASSERT_TRUE(first.ok()) << first.status();
  auto other_input = Buffer::Allocate(*executor_, sizeof(float));
  ASSERT_TRUE(other_input.ok()) << other_input.status();
  BufferVec other_inputs{std::move(*other_input)};
  auto second = layer.fwd(*executor_, other_inputs);
  ASSERT_TRUE(second.ok()) << second.status();
  ASSERT_EQ(first->state.intermediates.size(), 1);
  ASSERT_EQ(second->state.intermediates.size(), 1);
  EXPECT_EQ(first->state.intermediates[0].data(), inputs_[0].data());
  EXPECT_EQ(second->state.intermediates[0].data(), other_inputs[0].data());
  EXPECT_EQ(first->state.layer, &layer);
  EXPECT_EQ(second->state.layer, &layer);
}

TEST_F(LayerTest, BackwardImplementationErrorIsPreserved) {
  IdentityLayer layer;
  auto result = layer.fwd(*executor_, inputs_);
  ASSERT_TRUE(result.ok()) << result.status();
  layer.fail_backward = true;
  auto gradient = layer.bwd(*executor_, inputs_, std::move(result->state));
  EXPECT_EQ(gradient.status(), absl::InternalError("backward failed"));
  EXPECT_EQ(layer.backward_calls, 1);
}

TEST_F(LayerTest, NestedCompositionChecksIndividualChildIdentity) {
  auto first = std::make_unique<IdentityLayer>();
  auto second = std::make_unique<IdentityLayer>();
  const auto* first_ptr = first.get();
  const auto* second_ptr = second.get();
  std::vector<std::unique_ptr<Layer>> children;
  children.push_back(std::move(first));
  children.push_back(std::move(second));
  auto inner_result = ComposedLayer::Create(std::move(children));
  ASSERT_TRUE(inner_result.ok()) << inner_result.status();
  auto inner = std::move(*inner_result);
  const auto* inner_ptr = inner.get();
  std::vector<std::unique_ptr<Layer>> outer_children;
  outer_children.push_back(std::move(inner));
  auto outer_result = ComposedLayer::Create(std::move(outer_children));
  ASSERT_TRUE(outer_result.ok()) << outer_result.status();
  auto& outer = **outer_result;

  auto result = outer.fwd(*executor_, inputs_);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result->state.layer, &outer);
  ASSERT_EQ(result->state.children.size(), 1);
  EXPECT_EQ(result->state.children[0].layer, inner_ptr);
  auto& child_states = result->state.children[0].children;
  ASSERT_EQ(child_states.size(), 2);
  EXPECT_EQ(child_states[0].layer, first_ptr);
  EXPECT_EQ(child_states[1].layer, second_ptr);
  ASSERT_TRUE(outer.bwd(*executor_, inputs_, result->state).ok());
  EXPECT_EQ(first_ptr->backward_calls, 1);
  EXPECT_EQ(second_ptr->backward_calls, 1);

  // The parent state is valid, but the children are attached to wrong slots.
  std::swap(child_states[0], child_states[1]);
  auto gradient = outer.bwd(*executor_, inputs_, std::move(result->state));
  EXPECT_EQ(gradient.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(first_ptr->backward_calls, 1);
  EXPECT_EQ(second_ptr->backward_calls, 1);
}

TEST_F(LayerTest, CompositionRoutesMultipleOutputsAndInterleavedStates) {
  auto parameter = Buffer::Allocate(*executor_, sizeof(float));
  auto other_input = Buffer::Allocate(*executor_, sizeof(float));
  ASSERT_TRUE(parameter.ok()) << parameter.status();
  ASSERT_TRUE(other_input.ok()) << other_input.status();
  auto source = std::make_unique<ParameterOutputLayer>(*parameter);
  auto* source_ptr = source.get();
  std::vector<std::unique_ptr<Layer>> children;
  children.push_back(std::move(source));
  children.push_back(std::make_unique<SwapOutputsLayer>());
  auto model_result = ComposedLayer::Create(std::move(children));
  ASSERT_TRUE(model_result.ok()) << model_result.status();
  auto& model = **model_result;

  auto first = model.fwd(*executor_, inputs_);
  auto second = model.fwd(*executor_, BufferVec{*other_input});
  ASSERT_TRUE(first.ok()) << first.status();
  ASSERT_TRUE(second.ok()) << second.status();
  ASSERT_EQ(first->outputs.size(), 2u);
  ASSERT_EQ(second->outputs.size(), 2u);
  EXPECT_EQ(first->outputs[0].data(), parameter->data());
  EXPECT_EQ(first->outputs[1].data(), inputs_[0].data());
  EXPECT_EQ(second->outputs[1].data(), other_input->data());

  BufferVec gradients{*parameter, *other_input};
  auto first_gradient =
      model.bwd(*executor_, gradients, std::move(first->state));
  ASSERT_TRUE(first_gradient.ok()) << first_gradient.status();
  ASSERT_EQ(first_gradient->size(), 1u);
  EXPECT_EQ((*first_gradient)[0].data(), other_input->data());
  ASSERT_EQ(source_ptr->received_gradients.size(), 2u);
  EXPECT_EQ(source_ptr->received_gradients[0].data(), other_input->data());
  EXPECT_EQ(source_ptr->received_gradients[1].data(), parameter->data());
  EXPECT_EQ(source_ptr->saved_inputs[0].data(), inputs_[0].data());

  auto second_gradient =
      model.bwd(*executor_, gradients, std::move(second->state));
  ASSERT_TRUE(second_gradient.ok()) << second_gradient.status();
  EXPECT_EQ(source_ptr->saved_inputs[0].data(), other_input->data());
  EXPECT_EQ(first->outputs[1].data(), inputs_[0].data());
}

TEST_F(LayerTest, CompositionAcceptsAndReturnsMultipleInputsAndGradients) {
  auto second = Buffer::Allocate(*executor_, sizeof(float));
  ASSERT_TRUE(second.ok()) << second.status();
  std::vector<std::unique_ptr<Layer>> children;
  children.push_back(std::make_unique<SwapOutputsLayer>());
  auto model_result = ComposedLayer::Create(std::move(children));
  ASSERT_TRUE(model_result.ok()) << model_result.status();
  auto& model = **model_result;
  BufferVec inputs{inputs_[0], *second};
  auto result = model.fwd(*executor_, inputs);
  ASSERT_TRUE(result.ok()) << result.status();
  ASSERT_EQ(result->outputs.size(), 2u);
  EXPECT_EQ(result->outputs[0].data(), second->data());
  EXPECT_EQ(result->outputs[1].data(), inputs_[0].data());
  auto gradients = model.bwd(*executor_, inputs, std::move(result->state));
  ASSERT_TRUE(gradients.ok()) << gradients.status();
  ASSERT_EQ(gradients->size(), 2u);
  EXPECT_EQ((*gradients)[0].data(), second->data());
  EXPECT_EQ((*gradients)[1].data(), inputs_[0].data());
}

TEST_F(LayerTest, ResidualRejectsMultipleBranchOutputs) {
  auto result =
      ResidualLayer::Create(std::make_unique<ParameterOutputLayer>(inputs_[0]));
  EXPECT_EQ(result.status().code(), absl::StatusCode::kInvalidArgument);
}

}  // namespace
}  // namespace pluto::llm
