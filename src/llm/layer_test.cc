#include "src/llm/layer.h"

#include <memory>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "gtest/gtest.h"
#include "src/llm/layers/combinators.h"

namespace pluto::llm {
namespace {

// No kernels are needed to test dispatch and tape ownership. Keeping the
// implementations private also checks that callers use the inherited API.
class IdentityLayer final : public Layer {
 public:
  absl::Span<Buffer> weights() override { return {}; }
  DataType output_type() const override { return DataType::FP16; }

  mutable int forward_calls = 0;
  int backward_calls = 0;
  bool fail_forward = false;
  bool fail_backward = false;

 private:
  absl::StatusOr<Buffer> fwd_impl(cuda::Executor&,
                                  absl::Span<const Buffer> inputs,
                                  Tape* tape) const override {
    ++forward_calls;
    EXPECT_EQ(tape->layer, nullptr);
    EXPECT_TRUE(tape->intermediates.empty());
    EXPECT_TRUE(tape->children.empty());
    // Populate state even on failure to exercise partially written tapes.
    tape->intermediates = {inputs[0]};
    if (fail_forward) return absl::ResourceExhaustedError("forward failed");
    return inputs[0];
  }

  absl::StatusOr<BufferVec> bwd_impl(cuda::Executor&, absl::Span<const Buffer>,
                                     Tape tape) override {
    ++backward_calls;
    EXPECT_EQ(tape.layer, this);
    if (fail_backward) return absl::InternalError("backward failed");
    return std::move(tape.intermediates);
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

TEST_F(LayerTest, SuccessfulForwardAndCopiedOrMovedTapeDispatchThroughBase) {
  IdentityLayer layer;
  const Layer& forward_layer = layer;
  Tape tape;
  auto output = forward_layer.fwd(*executor_, inputs_, &tape);
  ASSERT_TRUE(output.ok()) << output.status();
  EXPECT_EQ(output->data(), inputs_[0].data());
  EXPECT_EQ(tape.layer, &layer);
  EXPECT_EQ(layer.forward_calls, 1);

  Layer& backward_layer = layer;
  // Copying saved state is supported; identity follows copies and moves.
  Tape copy = tape;
  auto gradient = backward_layer.bwd(*executor_, inputs_, std::move(copy));
  ASSERT_TRUE(gradient.ok()) << gradient.status();
  ASSERT_EQ(gradient->size(), 1);
  EXPECT_EQ((*gradient)[0].data(), inputs_[0].data());
  EXPECT_TRUE(layer.bwd(*executor_, inputs_, std::move(tape)).ok());
  EXPECT_EQ(layer.backward_calls, 2);
}

TEST_F(LayerTest, NullForwardTapeDoesNotDispatch) {
  IdentityLayer layer;
  auto output = layer.fwd(*executor_, inputs_, nullptr);
  EXPECT_EQ(output.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(layer.forward_calls, 0);
}

TEST_F(LayerTest, UnusedTapeDoesNotDispatchBackward) {
  IdentityLayer layer;
  auto gradient = layer.bwd(*executor_, inputs_, Tape{});
  EXPECT_EQ(gradient.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(layer.backward_calls, 0);
}

TEST_F(LayerTest, SameTypeAndShapeDoNotMakeAnotherLayersTapeValid) {
  IdentityLayer first;
  IdentityLayer second;
  Tape tape;
  ASSERT_TRUE(first.fwd(*executor_, inputs_, &tape).ok());
  auto gradient = second.bwd(*executor_, inputs_, tape);
  EXPECT_EQ(gradient.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(second.backward_calls, 0);
  EXPECT_TRUE(first.bwd(*executor_, inputs_, std::move(tape)).ok());
}

TEST_F(LayerTest, FailedForwardInvalidatesReusedTape) {
  IdentityLayer layer;
  Tape tape;
  ASSERT_TRUE(layer.fwd(*executor_, inputs_, &tape).ok());
  tape.children.emplace_back();
  layer.fail_forward = true;
  auto output = layer.fwd(*executor_, inputs_, &tape);
  EXPECT_EQ(output.status(), absl::ResourceExhaustedError("forward failed"));
  EXPECT_EQ(tape.layer, nullptr);
  EXPECT_EQ(layer.forward_calls, 2);
  auto gradient = layer.bwd(*executor_, inputs_, std::move(tape));
  EXPECT_EQ(gradient.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(layer.backward_calls, 0);
}

TEST_F(LayerTest, ReusingTapeAssociatesItWithLatestSuccessfulLayer) {
  IdentityLayer first;
  IdentityLayer second;
  Tape tape;
  ASSERT_TRUE(first.fwd(*executor_, inputs_, &tape).ok());
  ASSERT_TRUE(second.fwd(*executor_, inputs_, &tape).ok());
  EXPECT_EQ(tape.layer, &second);
  EXPECT_EQ(first.bwd(*executor_, inputs_, tape).status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(second.bwd(*executor_, inputs_, std::move(tape)).ok());
}

TEST_F(LayerTest, BackwardImplementationErrorIsPreserved) {
  IdentityLayer layer;
  Tape tape;
  ASSERT_TRUE(layer.fwd(*executor_, inputs_, &tape).ok());
  layer.fail_backward = true;
  auto gradient = layer.bwd(*executor_, inputs_, std::move(tape));
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
  auto inner =
      std::make_unique<ComposedLayer>(DataType::FP16, std::move(children));
  const auto* inner_ptr = inner.get();
  std::vector<std::unique_ptr<Layer>> outer_children;
  outer_children.push_back(std::move(inner));
  ComposedLayer outer(DataType::FP16, std::move(outer_children));

  Tape tape;
  ASSERT_TRUE(outer.fwd(*executor_, inputs_, &tape).ok());
  EXPECT_EQ(tape.layer, &outer);
  ASSERT_EQ(tape.children.size(), 1);
  EXPECT_EQ(tape.children[0].layer, inner_ptr);
  auto& child_tapes = tape.children[0].children;
  ASSERT_EQ(child_tapes.size(), 2);
  EXPECT_EQ(child_tapes[0].layer, first_ptr);
  EXPECT_EQ(child_tapes[1].layer, second_ptr);
  ASSERT_TRUE(outer.bwd(*executor_, inputs_, tape).ok());
  EXPECT_EQ(first_ptr->backward_calls, 1);
  EXPECT_EQ(second_ptr->backward_calls, 1);

  // The parent tape is valid, but the children are attached to wrong slots.
  std::swap(child_tapes[0], child_tapes[1]);
  auto gradient = outer.bwd(*executor_, inputs_, std::move(tape));
  EXPECT_EQ(gradient.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(first_ptr->backward_calls, 1);
  EXPECT_EQ(second_ptr->backward_calls, 1);
}

}  // namespace
}  // namespace pluto::llm
