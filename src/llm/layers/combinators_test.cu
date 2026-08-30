#include "src/llm/layers/combinators.h"

#include <cuda_runtime.h>

#include <memory>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "gtest/gtest.h"
#include "src/common/status_macros.h"
#include "src/gpu/buffer.h"
#include "src/llm/layer.h"
#include "src/llm/layers/test_util.h"
#include "src/llm/layers/fully_connected.h"

namespace pluto::llm {
namespace {

// A CPU-only layer is enough to exercise builder ownership and type inference;
// its data path is deliberately unavailable because these tests never run it.
class TestLayer final : public Layer {
 public:
  explicit TestLayer(DataType output_type) : output_type_(output_type) {}

  absl::StatusOr<Buffer> fwd(absl::Span<const Buffer>, Tape*) override {
    return absl::UnimplementedError("TestLayer has no data path");
  }
  absl::StatusOr<BufferVec> bwd(absl::Span<const Buffer>, Tape) override {
    return absl::UnimplementedError("TestLayer has no data path");
  }
  absl::Span<Buffer> weights() override { return {}; }
  DataType output_type() const override { return output_type_; }

 private:
  DataType output_type_;
};

absl::StatusOr<std::unique_ptr<TestLayer>> MakeTestLayer(
    DataType output_type, int* evaluations) {
  ++*evaluations;
  return std::make_unique<TestLayer>(output_type);
}

absl::StatusOr<std::unique_ptr<ComposedLayer>> BuildTestComposition(
    int* evaluations, Layer** first, Layer** last) {
  ComposedLayerBuilder builder;
  RETURN_IF_ERROR(
      builder.add(MakeTestLayer(DataType::FP16, evaluations)));
  *first = builder.back();
  RETURN_IF_ERROR(builder.add(MakeTestLayer(DataType::FP8, evaluations)));
  *last = builder.back();
  return builder.create();
}

absl::Status BuildWithFactoryError(int* evaluations, bool* reached_end) {
  ComposedLayerBuilder builder;
  RETURN_IF_ERROR(
      builder.add(MakeTestLayer(DataType::FP16, evaluations)));
  RETURN_IF_ERROR(
      builder.add(([&]() -> absl::StatusOr<std::unique_ptr<TestLayer>> {
        ++*evaluations;
        return absl::NotFoundError("missing layer");
      })()));
  *reached_end = true;
  return absl::OkStatus();
}

absl::Status BuildWithNullLayer() {
  ComposedLayerBuilder builder;
  RETURN_IF_ERROR(builder.add(
      absl::StatusOr<std::unique_ptr<TestLayer>>(
          std::unique_ptr<TestLayer>())));
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<ComposedLayer>> BuildDenseComposition(
    cudaStream_t stream) {
  ComposedLayerBuilder builder;
  RETURN_IF_ERROR(builder.add(FullyConnectedLayer::Create(
      kTestModelWidth, DataType::FP16, stream)));
  RETURN_IF_ERROR(builder.add(FullyConnectedLayer::Create(
      kTestModelWidth, DataType::FP16, stream)));
  return builder.create();
}

TEST(ComposedLayerBuilderTest, BackIsStableAndCreateInfersFinalOutputType) {
  int evaluations = 0;
  Layer* first = nullptr;
  Layer* last = nullptr;
  auto composed = BuildTestComposition(&evaluations, &first, &last);
  ASSERT_TRUE(composed.ok()) << composed.status();

  EXPECT_EQ(evaluations, 2);
  ASSERT_NE(first, nullptr);
  ASSERT_NE(last, nullptr);
  EXPECT_EQ(first->output_type(), DataType::FP16);
  EXPECT_EQ(last->output_type(), DataType::FP8);
  EXPECT_EQ((*composed)->output_type(), DataType::FP8);
}

TEST(ComposedLayerBuilderTest, AddPropagatesFactoryErrorExactlyOnce) {
  int evaluations = 0;
  bool reached_end = false;
  const absl::Status status =
      BuildWithFactoryError(&evaluations, &reached_end);

  EXPECT_EQ(status.code(), absl::StatusCode::kNotFound);
  EXPECT_EQ(status.message(), "missing layer");
  EXPECT_EQ(evaluations, 2);
  EXPECT_FALSE(reached_end);
}

TEST(ComposedLayerBuilderTest, RejectsEmptyAndNullLayers) {
  ComposedLayerBuilder builder;
  EXPECT_EQ(builder.back(), nullptr);
  const ComposedLayerBuilder& const_builder = builder;
  EXPECT_EQ(const_builder.back(), nullptr);

  const auto empty = builder.create();
  EXPECT_EQ(empty.status().code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_EQ(BuildWithNullLayer().code(), absl::StatusCode::kInvalidArgument);
}

TEST_F(LayersTest, CreatedCompositionCollectsChildWeights) {
  auto composed = BuildDenseComposition(stream_);
  ASSERT_TRUE(composed.ok()) << composed.status();

  // Each dense child contributes its own matrix and bias.
  EXPECT_EQ((*composed)->weights().size(), 4u);
  EXPECT_EQ((*composed)->gradients().size(), 4u);
}


}  // namespace
}  // namespace pluto::llm
