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
  absl::StatusOr<HostBuffer> fwd_impl(
      absl::Span<const HostBuffer> inputs,
      ReferenceBackwardState& state) const override {
    ++forward_calls;
    EXPECT_EQ(state.layer, nullptr);
    EXPECT_TRUE(state.intermediates.empty());
    EXPECT_TRUE(state.children.empty());
    // Populate state even on failure to exercise partially written states.
    state.intermediates = {inputs[0]};
    if (fail_forward) return absl::ResourceExhaustedError("forward failed");
    return inputs[0];
  }

  absl::StatusOr<HostBufferVec> bwd_impl(
      absl::Span<const HostBuffer>, ReferenceBackwardState state) override {
    ++backward_calls;
    EXPECT_EQ(state.layer, this);
    if (fail_backward) return absl::InternalError("backward failed");
    return std::move(state.intermediates);
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

TEST_F(LayerReferenceStateTest,
       SuccessfulForwardAndCopiedOrMovedStateDispatchThroughBase) {
  IdentityLayerReference layer;
  const LayerReference& forward_layer = layer;
  ReferenceBackwardState state;
  auto output = forward_layer.fwd(inputs_, state);
  ASSERT_TRUE(output.ok()) << output.status();
  EXPECT_EQ(output->data(), inputs_[0].data());
  EXPECT_EQ(state.layer, &layer);
  EXPECT_EQ(layer.forward_calls, 1);

  LayerReference& backward_layer = layer;
  // Copying saved state is supported; identity follows copies and moves.
  ReferenceBackwardState copy = state;
  auto gradient = backward_layer.bwd(inputs_, std::move(copy));
  ASSERT_TRUE(gradient.ok()) << gradient.status();
  ASSERT_EQ(gradient->size(), 1);
  EXPECT_EQ((*gradient)[0].data(), inputs_[0].data());
  EXPECT_TRUE(layer.bwd(inputs_, std::move(state)).ok());
  EXPECT_EQ(layer.backward_calls, 2);
}

TEST(LayerReferenceApiTest, ForwardRequiresStateReference) {
  static_assert(
      std::is_invocable_v<decltype(&LayerReference::fwd), const LayerReference&,
                          absl::Span<const HostBuffer>,
                          ReferenceBackwardState&>);
  static_assert(
      !std::is_invocable_v<decltype(&LayerReference::fwd),
                           const LayerReference&, absl::Span<const HostBuffer>,
                           ReferenceBackwardState*>);
}

TEST_F(LayerReferenceStateTest, UnusedStateDoesNotDispatchBackward) {
  IdentityLayerReference layer;
  auto gradient = layer.bwd(inputs_, ReferenceBackwardState{});
  EXPECT_EQ(gradient.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(layer.backward_calls, 0);
}

TEST_F(LayerReferenceStateTest,
       SameTypeAndShapeDoNotMakeAnotherLayersStateValid) {
  IdentityLayerReference first;
  IdentityLayerReference second;
  ReferenceBackwardState state;
  ASSERT_TRUE(first.fwd(inputs_, state).ok());
  auto gradient = second.bwd(inputs_, state);
  EXPECT_EQ(gradient.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(second.backward_calls, 0);
  EXPECT_TRUE(first.bwd(inputs_, std::move(state)).ok());
}

TEST_F(LayerReferenceStateTest, FailedForwardInvalidatesReusedState) {
  IdentityLayerReference layer;
  ReferenceBackwardState state;
  ASSERT_TRUE(layer.fwd(inputs_, state).ok());
  state.children.emplace_back();
  layer.fail_forward = true;
  auto output = layer.fwd(inputs_, state);
  EXPECT_EQ(output.status(), absl::ResourceExhaustedError("forward failed"));
  EXPECT_EQ(state.layer, nullptr);
  EXPECT_EQ(layer.forward_calls, 2);
  auto gradient = layer.bwd(inputs_, std::move(state));
  EXPECT_EQ(gradient.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(layer.backward_calls, 0);
}

TEST_F(LayerReferenceStateTest,
       ReusingStateAssociatesItWithLatestSuccessfulLayer) {
  IdentityLayerReference first;
  IdentityLayerReference second;
  ReferenceBackwardState state;
  ASSERT_TRUE(first.fwd(inputs_, state).ok());
  ASSERT_TRUE(second.fwd(inputs_, state).ok());
  EXPECT_EQ(state.layer, &second);
  EXPECT_EQ(first.bwd(inputs_, state).status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(second.bwd(inputs_, std::move(state)).ok());
}

TEST_F(LayerReferenceStateTest, BackwardImplementationErrorIsPreserved) {
  IdentityLayerReference layer;
  ReferenceBackwardState state;
  ASSERT_TRUE(layer.fwd(inputs_, state).ok());
  layer.fail_backward = true;
  auto gradient = layer.bwd(inputs_, std::move(state));
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

  ReferenceBackwardState state;
  ASSERT_TRUE(outer.fwd(inputs_, state).ok());
  EXPECT_EQ(state.layer, &outer);
  ASSERT_EQ(state.children.size(), 1);
  EXPECT_EQ(state.children[0].layer, inner_ptr);
  auto& child_states = state.children[0].children;
  ASSERT_EQ(child_states.size(), 2);
  EXPECT_EQ(child_states[0].layer, first_ptr);
  EXPECT_EQ(child_states[1].layer, second_ptr);
  ASSERT_TRUE(outer.bwd(inputs_, state).ok());
  EXPECT_EQ(first_ptr->backward_calls, 1);
  EXPECT_EQ(second_ptr->backward_calls, 1);

  // The parent state is valid, but the children are attached to wrong slots.
  std::swap(child_states[0], child_states[1]);
  auto gradient = outer.bwd(inputs_, std::move(state));
  EXPECT_EQ(gradient.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(first_ptr->backward_calls, 1);
  EXPECT_EQ(second_ptr->backward_calls, 1);
}

}  // namespace
}  // namespace pluto::llm
