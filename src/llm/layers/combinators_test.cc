#include "src/llm/layers/combinators.h"

#include <cuda_runtime.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "gtest/gtest.h"
#include "src/cuda/buffer.h"
#include "src/llm/layer.h"
#include "src/llm/layers/fully_connected.h"
#include "src/llm/layers/gelu.h"
#include "src/llm/layers/test_util.h"
#include "src/util/status_macros.h"

namespace pluto::llm {
namespace {

// A CPU-only layer is enough to exercise builder ownership and type inference;
// its data path is deliberately unavailable because these tests never run it.
class TestLayer final : public Layer {
 public:
  absl::string_view name() const override { return "TestLayer"; }

  // This fake explicitly converts FP32 input to the requested output storage.
  // In particular, the FP8 builder test is a real declared conversion, not an
  // exception to exact producer/consumer matching.
  explicit TestLayer(DataType output_type)
      : output_type_(output_type),
        inputs_{ActivationType(DataType::FP32, {-2, 1, 16})},
        outputs_{ActivationType(ActivationDataType(output_type), {-2, 1, 16})} {
  }
  TestLayer(std::vector<ActivationType> inputs,
            std::vector<ActivationType> outputs,
            DataType output_type = DataType::FP16)
      : output_type_(output_type),
        inputs_(std::move(inputs)),
        outputs_(std::move(outputs)) {}

  absl::Span<const ActivationType> input_types() const override {
    return inputs_;
  }
  absl::Span<const ActivationType> output_types() const override {
    return outputs_;
  }

  absl::Span<Buffer> weights() override { return {}; }
  DataType output_type() const override { return output_type_; }

 private:
  absl::StatusOr<FwdResult> fwd_impl(cuda::Executor&, absl::Span<const Buffer>,
                                     LayerHooks*) const override {
    return absl::UnimplementedError("TestLayer has no data path");
  }
  absl::StatusOr<BufferVec> bwd_impl(cuda::Executor&, absl::Span<const Buffer>,
                                     BackwardState, LayerHooks*) override {
    return absl::UnimplementedError("TestLayer has no data path");
  }

  DataType output_type_;
  const std::vector<ActivationType> inputs_;
  const std::vector<ActivationType> outputs_;
};

absl::StatusOr<std::unique_ptr<TestLayer>> MakeTestLayer(DataType output_type,
                                                         int* evaluations) {
  ++*evaluations;
  return absl::make_unique<TestLayer>(output_type);
}

absl::StatusOr<std::unique_ptr<ComposedLayer>> BuildTestComposition(
    int* evaluations, Layer** first, Layer** last) {
  ComposedLayerBuilder builder;
  RETURN_IF_ERROR(builder.add(MakeTestLayer(DataType::FP16, evaluations)));
  *first = builder.back();
  RETURN_IF_ERROR(builder.add(MakeTestLayer(DataType::FP8, evaluations)));
  *last = builder.back();
  return builder.create("MixedPrecisionTestComposition");
}

absl::Status BuildWithFactoryError(int* evaluations, bool* reached_end) {
  ComposedLayerBuilder builder;
  RETURN_IF_ERROR(builder.add(MakeTestLayer(DataType::FP16, evaluations)));
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
  RETURN_IF_ERROR(builder.add(absl::StatusOr<std::unique_ptr<TestLayer>>(
      std::unique_ptr<TestLayer>())));
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<ComposedLayer>> BuildDenseComposition(
    cuda::Executor& executor) {
  ComposedLayerBuilder builder;
  RETURN_IF_ERROR(builder.add(
      FullyConnectedLayer::Create(executor, kTestModelWidth, DataType::FP16)));
  RETURN_IF_ERROR(builder.add(
      FullyConnectedLayer::Create(executor, kTestModelWidth, DataType::FP16)));
  return builder.create("TwoDenseLayers");
}

TEST(ComposedLayerTest, OwnsNamesBeyondCallerStringLifetime) {
  std::string source = "transformer block zero / attention and MLP";
  const std::string expected = source;
  std::vector<std::unique_ptr<Layer>> first_children;
  first_children.push_back(absl::make_unique<TestLayer>(DataType::FP16));
  auto first = ComposedLayer::Create(source, std::move(first_children));
  ASSERT_TRUE(first.ok()) << first.status();
  const absl::string_view retained_name = (*first)->name();
  source.assign(512, 'x');
  EXPECT_EQ((*first)->name(), expected);
  EXPECT_EQ(retained_name, expected);

  std::vector<std::unique_ptr<Layer>> second_children;
  second_children.push_back(absl::make_unique<TestLayer>(DataType::FP16));
  auto second = ComposedLayer::Create(
      std::string("transformer block one / ") + "attention and MLP",
      std::move(second_children));
  ASSERT_TRUE(second.ok()) << second.status();
  EXPECT_EQ((*second)->name(), "transformer block one / attention and MLP");
  EXPECT_NE((*first)->name(), (*second)->name());
  EXPECT_EQ(retained_name, expected);
}

TEST(ComposedLayerTest, DirectFactoryRejectsEmptyName) {
  std::vector<std::unique_ptr<Layer>> children;
  children.push_back(absl::make_unique<TestLayer>(DataType::FP16));
  EXPECT_EQ(ComposedLayer::Create("", std::move(children)).status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(ComposedLayer::Create("", {}).status().code(),
            absl::StatusCode::kInvalidArgument);
}

TEST(ComposedLayerBuilderTest, InvalidNamePreservesChildrenForRetryAndReuse) {
  ComposedLayerBuilder builder;
  EXPECT_EQ(builder.create("").status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(builder.back(), nullptr);
  ASSERT_TRUE(builder.add(absl::make_unique<TestLayer>(DataType::FP16)).ok());
  const Layer* first = builder.back();
  ASSERT_TRUE(builder.add(absl::make_unique<TestLayer>(DataType::FP8)).ok());
  const Layer* last = builder.back();
  EXPECT_EQ(builder.create("").status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(builder.back(), last);
  const ComposedLayerBuilder& const_builder = builder;
  EXPECT_EQ(const_builder.back(), last);

  std::string name = "RetriedNamedComposition";
  auto graph = builder.create(name);
  ASSERT_TRUE(graph.ok()) << graph.status();
  name.assign(512, 'x');
  EXPECT_EQ((*graph)->name(), "RetriedNamedComposition");
  EXPECT_EQ((*graph)->input_types().data(), first->input_types().data());
  EXPECT_EQ((*graph)->output_types().data(), last->output_types().data());
  EXPECT_EQ((*graph)->output_type(), DataType::FP8);
  EXPECT_EQ(builder.back(), nullptr);

  ASSERT_TRUE(builder.add(absl::make_unique<TestLayer>(DataType::FP16)).ok());
  auto reused = builder.create("ReusedNamedComposition");
  ASSERT_TRUE(reused.ok()) << reused.status();
  EXPECT_EQ((*reused)->name(), "ReusedNamedComposition");
  EXPECT_EQ((*graph)->name(), "RetriedNamedComposition");
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
  const absl::Status status = BuildWithFactoryError(&evaluations, &reached_end);

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

  const auto empty = builder.create("EmptyComposition");
  EXPECT_EQ(empty.status().code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_EQ(BuildWithNullLayer().code(), absl::StatusCode::kInvalidArgument);
}

TEST_F(LayersTest, CreatedCompositionCollectsChildWeights) {
  auto composed = BuildDenseComposition(*executor_);
  ASSERT_TRUE(composed.ok()) << composed.status();

  // Each dense child contributes its own matrix and bias.
  EXPECT_EQ((*composed)->weights().size(), 4u);
  EXPECT_EQ((*composed)->gradients().size(), 4u);
}

std::unique_ptr<TestLayer> SignatureLayer(std::vector<ActivationType> inputs,
                                          std::vector<ActivationType> outputs) {
  return absl::make_unique<TestLayer>(std::move(inputs), std::move(outputs));
}

TEST(ComposedLayerBuilderTest, RejectsEveryKindOfIncompatibleConnection) {
  const ActivationType actual(DataType::FP32, {-2, 7, 32});
  // Same byte counts do not make different ranks/sample boundaries equivalent.
  for (const auto& expected : {ActivationType(DataType::FP32, {2, 7, 32}),
                               ActivationType(DataType::FP32, {-2, 7, 64}),
                               ActivationType(DataType::BF16, {-2, 7, 32}),
                               ActivationType(DataType::FP32, {-2, 224}),
                               ActivationType(DataType::FP32, {-2, 14, 16})}) {
    ComposedLayerBuilder builder;
    ASSERT_TRUE(builder.add(SignatureLayer({actual}, {actual})).ok());
    const auto status = builder.add(SignatureLayer({expected}, {expected}));
    EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  }
  // Batch is symbolic equality, not a wildcard in either direction.
  const ActivationType fixed(DataType::FP32, {2, 7, 32});
  ComposedLayerBuilder reverse;
  ASSERT_TRUE(reverse.add(SignatureLayer({fixed}, {fixed})).ok());
  EXPECT_EQ(reverse.add(SignatureLayer({actual}, {actual})).code(),
            absl::StatusCode::kInvalidArgument);
}

TEST(ComposedLayerBuilderTest, RejectsArityMismatch) {
  const ActivationType activation(DataType::FP32, {-2, 7, 32});
  ComposedLayerBuilder builder;
  ASSERT_TRUE(
      builder.add(SignatureLayer({activation}, {activation, activation})).ok());
  EXPECT_EQ(builder.add(SignatureLayer({activation}, {activation})).code(),
            absl::StatusCode::kInvalidArgument);
  // A rejected addition leaves the producer available for a compatible retry.
  EXPECT_TRUE(
      builder.add(SignatureLayer({activation, activation}, {activation})).ok());
  EXPECT_TRUE(builder.create("RetriedArityConnection").ok());
}

TEST(ComposedLayerBuilderTest, RejectsInvalidDimensionsInEitherSignature) {
  const ActivationType valid(DataType::FP32, {-2, 7, 32});
  for (const auto& invalid : {ActivationType(DataType::FP32, {-1, 7, 32}),
                              ActivationType(DataType::FP32, {2, -2, 32}),
                              ActivationType(DataType::FP32, {-2, 0, 32}),
                              ActivationType(DataType::FP32, {-2, -3, 32})}) {
    for (bool invalid_input : {false, true}) {
      ComposedLayerBuilder builder;
      const auto status =
          builder.add(SignatureLayer({invalid_input ? invalid : valid},
                                     {invalid_input ? valid : invalid}));
      EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
      EXPECT_EQ(builder.back(), nullptr);
    }
  }
}

TEST(ComposedLayerBuilderTest, FailedAddPreservesChildrenAndCanRetry) {
  const ActivationType narrow(DataType::FP32, {-2, 7, 32});
  const ActivationType wide(DataType::FP32, {-2, 7, 64});
  ComposedLayerBuilder builder;
  auto first = SignatureLayer({narrow}, {narrow});
  const Layer* first_pointer = first.get();
  ASSERT_TRUE(builder.add(std::move(first)).ok());

  EXPECT_EQ(builder.add(SignatureLayer({wide}, {wide})).code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(builder.back(), first_pointer);
  EXPECT_EQ(builder.add(std::unique_ptr<Layer>()).code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(builder.back(), first_pointer);

  auto second = SignatureLayer({narrow}, {wide});
  const Layer* second_pointer = second.get();
  ASSERT_TRUE(builder.add(std::move(second)).ok());
  EXPECT_EQ(builder.back(), second_pointer);
  auto composed = builder.create("RetriedWidthConnection");
  ASSERT_TRUE(composed.ok()) << composed.status();
  EXPECT_EQ(builder.back(), nullptr);
  EXPECT_EQ((*composed)->input_types().data(),
            first_pointer->input_types().data());
  EXPECT_EQ((*composed)->output_types().data(),
            second_pointer->output_types().data());
  EXPECT_EQ((*composed)->input_types()[0], narrow);
  EXPECT_EQ((*composed)->output_types()[0], wide);

  // create() consumes the successful graph; the same builder can be reused.
  EXPECT_TRUE(builder.add(SignatureLayer({wide}, {wide})).ok());
  EXPECT_TRUE(builder.create("ReusedBuilder").ok());
}

TEST(ComposedLayerTest, DirectFactoryCannotBypassConnectionValidation) {
  const ActivationType narrow(DataType::FP32, {-2, 7, 32});
  const ActivationType wide(DataType::FP32, {-2, 7, 64});
  std::vector<std::unique_ptr<Layer>> children;
  children.push_back(SignatureLayer({narrow}, {narrow}));
  children.push_back(SignatureLayer({narrow}, {narrow}));
  children.push_back(SignatureLayer({wide}, {wide}));
  const auto composed =
      ComposedLayer::Create("InvalidConnection", std::move(children));
  EXPECT_EQ(composed.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_NE(composed.status().message().find("child 2"), std::string::npos);

  EXPECT_EQ(ComposedLayer::Create("EmptyComposition", {}).status().code(),
            absl::StatusCode::kFailedPrecondition);
  std::vector<std::unique_ptr<Layer>> null_children;
  null_children.push_back(nullptr);
  EXPECT_EQ(ComposedLayer::Create("NullChild", std::move(null_children))
                .status()
                .code(),
            absl::StatusCode::kInvalidArgument);

  const ActivationType invalid(DataType::FP32, {-1, 7, 32});
  std::vector<std::unique_ptr<Layer>> invalid_children;
  invalid_children.push_back(SignatureLayer({narrow}, {invalid}));
  EXPECT_EQ(ComposedLayer::Create("InvalidChildSignature",
                                  std::move(invalid_children))
                .status()
                .code(),
            absl::StatusCode::kInvalidArgument);
}

TEST(ComposedLayerTest, NestedGraphsExposeFirstInputAndLastOutputs) {
  const ActivationType tokens(DataType::INT32, {-2, 7});
  const ActivationType activation(DataType::BF16, {-2, 7, 32});
  const ActivationType logits(DataType::FP32, {-2, 7, 48});
  const ActivationType statistics(DataType::FP32, {32});
  ComposedLayerBuilder inner;
  ASSERT_TRUE(inner.add(SignatureLayer({tokens}, {activation})).ok());
  ASSERT_TRUE(inner.add(SignatureLayer({activation}, {activation})).ok());
  ComposedLayerBuilder outer;
  ASSERT_TRUE(outer.add(inner.create("TokenEmbeddingPipeline")).ok());
  ASSERT_TRUE(
      outer.add(SignatureLayer({activation}, {logits, statistics})).ok());
  auto model = outer.create("LogitsAndStatisticsPipeline");
  ASSERT_TRUE(model.ok()) << model.status();
  ASSERT_EQ((*model)->input_types().size(), 1);
  ASSERT_EQ((*model)->output_types().size(), 2);
  EXPECT_EQ((*model)->input_types()[0], tokens);
  EXPECT_EQ((*model)->output_types()[0], logits);
  EXPECT_EQ((*model)->output_types()[1], statistics);
}

TEST(ResidualLayerTest, PreservesExactUnarySignatures) {
  for (const DataType storage : {DataType::FP32, DataType::BF16}) {
    const ActivationType activation(storage, {-2, 7, 32});
    auto residual =
        ResidualLayer::Create(SignatureLayer({activation}, {activation}));
    ASSERT_TRUE(residual.ok()) << residual.status();
    ASSERT_EQ((*residual)->input_types().size(), 1);
    ASSERT_EQ((*residual)->output_types().size(), 1);
    EXPECT_EQ((*residual)->input_types()[0], activation);
    EXPECT_EQ((*residual)->output_types()[0], activation);
  }
}

TEST(ResidualLayerTest, RejectsNullAndNonUnaryBranches) {
  const ActivationType activation(DataType::FP32, {-2, 7, 32});
  EXPECT_EQ(ResidualLayer::Create(nullptr).status().code(),
            absl::StatusCode::kInvalidArgument);
  for (const auto& inputs :
       {std::vector<ActivationType>{},
        std::vector<ActivationType>{activation, activation}}) {
    EXPECT_EQ(ResidualLayer::Create(SignatureLayer(inputs, {activation}))
                  .status()
                  .code(),
              absl::StatusCode::kInvalidArgument);
    EXPECT_EQ(ResidualLayer::Create(SignatureLayer({activation}, inputs))
                  .status()
                  .code(),
              absl::StatusCode::kInvalidArgument);
  }
}

TEST(ResidualLayerTest, RejectsShapeDtypeAndBatchChanges) {
  const ActivationType activation(DataType::FP32, {-2, 7, 32});
  for (const auto& different : {ActivationType(DataType::FP32, {2, 7, 32}),
                                ActivationType(DataType::FP32, {-2, 7, 64}),
                                ActivationType(DataType::BF16, {-2, 7, 32}),
                                ActivationType(DataType::FP32, {-2, 224}),
                                ActivationType(DataType::FP32, {-2, 14, 16}),
                                ActivationType(DataType::FP32, {-1, 7, 32})}) {
    EXPECT_EQ(ResidualLayer::Create(SignatureLayer({activation}, {different}))
                  .status()
                  .code(),
              absl::StatusCode::kInvalidArgument);
  }
}

TEST(ResidualLayerTest,
     RejectsUnsupportedPhysicalStorageDespiteMatchingShapes) {
  for (const DataType storage :
       {DataType::FP8, DataType::FP16, DataType::INT32}) {
    const ActivationType activation(storage, {-2, 7, 32});
    const auto residual =
        ResidualLayer::Create(SignatureLayer({activation}, {activation}));
    EXPECT_FALSE(residual.ok());
  }
}

TEST_F(LayersTest, GeluCannotHideADenseWidthMismatch) {
  ComposedLayerBuilder builder;
  ASSERT_TRUE(
      builder
          .add(FullyConnectedLayer::Create(*executor_, 32, DataType::FP16, 7))
          .ok());
  ASSERT_TRUE(
      builder.add(GeluLayer::Create(*executor_, 32, DataType::FP16, 7)).ok());
  const Layer* gelu = builder.back();
  EXPECT_EQ(
      builder
          .add(FullyConnectedLayer::Create(*executor_, 64, DataType::FP16, 7))
          .code(),
      absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(builder.back(), gelu);
  EXPECT_TRUE(builder
                  .add(FullyConnectedLayer::Create(*executor_, 32, 64,
                                                   DataType::FP16, 7))
                  .ok());
  EXPECT_TRUE(builder.create("DenseGeluProjection").ok());
}

}  // namespace
}  // namespace pluto::llm
