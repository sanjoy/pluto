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
#include "src/llm/layer_hooks.h"
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

// A permutation exposes buffer ordering and gradient fan-in without hiding
// either operation behind another numerical kernel.
class RoutingLayer final : public Layer {
 public:
  RoutingLayer(std::string name, std::vector<ActivationType> inputs,
               bool reverse = false)
      : name_(std::move(name)), inputs_(std::move(inputs)), reverse_(reverse) {
    for (size_t i = 0; i < inputs_.size(); ++i)
      outputs_.push_back(inputs_[reverse_ ? inputs_.size() - 1 - i : i]);
  }
  absl::string_view name() const override { return name_; }
  absl::Span<const ActivationType> input_types() const override {
    return inputs_;
  }
  absl::Span<const ActivationType> output_types() const override {
    return outputs_;
  }
  absl::Span<Buffer> weights() override { return {}; }
  DataType output_type() const override { return DataType::FP32; }

 private:
  BufferVec Route(absl::Span<const Buffer> values) const {
    BufferVec result;
    for (size_t i = 0; i < values.size(); ++i)
      result.push_back(values[reverse_ ? values.size() - 1 - i : i]);
    return result;
  }
  absl::StatusOr<FwdResult> fwd_impl(cuda::Executor&,
                                     absl::Span<const Buffer> inputs,
                                     LayerHooks*) const override {
    return FwdResult{Route(inputs), {}};
  }
  absl::StatusOr<BufferVec> bwd_impl(cuda::Executor&,
                                     absl::Span<const Buffer> gradients,
                                     BackwardState, LayerHooks*) override {
    return Route(gradients);
  }
  std::string name_;
  std::vector<ActivationType> inputs_;
  std::vector<ActivationType> outputs_;
  bool reverse_;
};

absl::StatusOr<Buffer> UploadParallelValues(cuda::Executor& executor,
                                            const std::vector<float>& values) {
  auto staging = CopyToPageLockedHostArray(executor, values);
  ASSIGN_OR_RETURN(auto buffer,
                   Buffer::Allocate(executor, values.size() * sizeof(float)));
  RETURN_IF_ERROR(cuda::CudaStatus(
      cudaMemcpyAsync(buffer.data(), staging.data(), buffer.size_bytes(),
                      cudaMemcpyHostToDevice, executor.stream()),
      "upload parallel test values"));
  return buffer;
}

void ExpectParallelValues(cuda::Executor& executor, const Buffer& buffer,
                          const std::vector<float>& expected) {
  ASSERT_EQ(buffer.size_bytes(), expected.size() * sizeof(float));
  auto values = AllocatePageLockedHostArray<float>(executor, expected.size());
  ASSERT_EQ(cudaMemcpyAsync(values.data(), buffer.data(), buffer.size_bytes(),
                            cudaMemcpyDeviceToHost, executor.stream()),
            cudaSuccess);
  ASSERT_TRUE(executor.Synchronize().ok());
  for (size_t i = 0; i < expected.size(); ++i)
    EXPECT_FLOAT_EQ(values[i], expected[i]) << i;
}

absl::StatusOr<std::unique_ptr<ParallelLayer>> BuildRoutingParallel() {
  const std::vector<ActivationType> inputs = {{DataType::FP32, {-2, 1, 3}},
                                              {DataType::FP32, {-2, 1, 5}}};
  std::vector<std::unique_ptr<Layer>> children;
  children.push_back(absl::make_unique<RoutingLayer>("left", inputs));
  children.push_back(absl::make_unique<RoutingLayer>("right", inputs, true));
  return ParallelLayer::Create("routing", std::move(children));
}

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

TEST(ParallelLayerTest, OwnsNameAndConcatenatesDifferentOutputSignatures) {
  const ActivationType input(DataType::BF16, {-2, 1, 32});
  const ActivationType narrow(DataType::BF16, {-2, 1, 16});
  const ActivationType statistics(DataType::FP32, {32});
  std::vector<std::unique_ptr<Layer>> children;
  children.push_back(SignatureLayer({input}, {narrow}));
  children.push_back(SignatureLayer({input}, {input, statistics}));
  std::string name = "Parallel projections";
  auto parallel = ParallelLayer::Create(name, std::move(children));
  ASSERT_TRUE(parallel.ok()) << parallel.status();
  name.assign(256, 'x');
  EXPECT_EQ((*parallel)->name(), "Parallel projections");
  ASSERT_EQ((*parallel)->input_types().size(), 1);
  EXPECT_EQ((*parallel)->input_types()[0], input);
  ASSERT_EQ((*parallel)->output_types().size(), 3);
  EXPECT_EQ((*parallel)->output_types()[0], narrow);
  EXPECT_EQ((*parallel)->output_types()[1], input);
  EXPECT_EQ((*parallel)->output_types()[2], statistics);
}

TEST(ParallelLayerTest, RejectsInvalidChildrenAndInputSignatures) {
  EXPECT_EQ(ParallelLayer::Create("", {}).status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(ParallelLayer::Create("empty", {}).status().code(),
            absl::StatusCode::kFailedPrecondition);
  std::vector<std::unique_ptr<Layer>> null_children;
  null_children.push_back(nullptr);
  EXPECT_EQ(
      ParallelLayer::Create("null", std::move(null_children)).status().code(),
      absl::StatusCode::kInvalidArgument);

  const ActivationType input(DataType::FP32, {-2, 1, 16});
  for (const auto& mismatch :
       {std::vector<ActivationType>{},
        std::vector<ActivationType>{input, input},
        std::vector<ActivationType>{{DataType::BF16, {-2, 1, 16}}},
        std::vector<ActivationType>{{DataType::FP32, {1, 1, 16}}},
        std::vector<ActivationType>{{DataType::FP32, {-2, 16}}},
        std::vector<ActivationType>{{DataType::FP32, {-2, 1, 32}}},
        std::vector<ActivationType>{{DataType::FP32, {-2, 0, 16}}}}) {
    std::vector<std::unique_ptr<Layer>> children;
    children.push_back(SignatureLayer({input}, {input}));
    children.push_back(SignatureLayer(mismatch, {input}));
    EXPECT_EQ(
        ParallelLayer::Create("mismatch", std::move(children)).status().code(),
        absl::StatusCode::kInvalidArgument);
  }
  std::vector<std::unique_ptr<Layer>> invalid_output;
  invalid_output.push_back(
      SignatureLayer({input}, {ActivationType(DataType::FP32, {-2, -1, 16})}));
  EXPECT_EQ(ParallelLayer::Create("invalid output", std::move(invalid_output))
                .status()
                .code(),
            absl::StatusCode::kInvalidArgument);
}

TEST_F(LayersTest, ParallelPreservesParameterOrder) {
  std::vector<std::unique_ptr<Layer>> children;
  std::vector<const void*> weights;
  std::vector<const void*> gradients;
  for (int width : {16, 32}) {
    auto child =
        FullyConnectedLayer::Create(*executor_, 16, width, DataType::BF16);
    ASSERT_TRUE(child.ok()) << child.status();
    for (const auto& weight : (*child)->weights())
      weights.push_back(weight.data());
    for (const auto& gradient : (*child)->gradients())
      gradients.push_back(gradient.data());
    children.push_back(std::move(*child));
  }
  auto parallel = ParallelLayer::Create("projections", std::move(children));
  ASSERT_TRUE(parallel.ok()) << parallel.status();
  ASSERT_EQ((*parallel)->weights().size(), weights.size());
  ASSERT_EQ((*parallel)->gradients().size(), gradients.size());
  for (size_t i = 0; i < weights.size(); ++i) {
    EXPECT_EQ((*parallel)->weights()[i].data(), weights[i]);
    EXPECT_EQ((*parallel)->gradients()[i].data(), gradients[i]);
  }
}

TEST_F(LayersTest, ParallelFansOutEveryInputAndSumsGradientsWithoutMutation) {
  auto parallel = BuildRoutingParallel();
  ASSERT_TRUE(parallel.ok()) << parallel.status();
  auto first = UploadParallelValues(*executor_, {1, 2, 3});
  auto second = UploadParallelValues(*executor_, {4, 5, 6, 7, 8});
  ASSERT_TRUE(first.ok()) << first.status();
  ASSERT_TRUE(second.ok()) << second.status();
  auto fwd = (*parallel)->fwd(*executor_, {*first, *second});
  ASSERT_TRUE(fwd.ok()) << fwd.status();
  ASSERT_EQ(fwd->outputs.size(), 4);
  EXPECT_EQ(fwd->outputs[0].data(), first->data());
  EXPECT_EQ(fwd->outputs[1].data(), second->data());
  EXPECT_EQ(fwd->outputs[2].data(), second->data());
  EXPECT_EQ(fwd->outputs[3].data(), first->data());

  auto third = UploadParallelValues(*executor_, {8, 7, 6, 5, 4});
  auto fourth = UploadParallelValues(*executor_, {3, 2, 1});
  ASSERT_TRUE(third.ok()) << third.status();
  ASSERT_TRUE(fourth.ok()) << fourth.status();
  auto bwd = (*parallel)->bwd(*executor_, {*first, *second, *third, *fourth},
                              std::move(fwd->state));
  ASSERT_TRUE(bwd.ok()) << bwd.status();
  ASSERT_EQ(bwd->size(), 2);
  ExpectParallelValues(*executor_, (*bwd)[0], {4, 4, 4});
  ExpectParallelValues(*executor_, (*bwd)[1], {12, 12, 12, 12, 12});
  ExpectParallelValues(*executor_, *first, {1, 2, 3});
  ExpectParallelValues(*executor_, *second, {4, 5, 6, 7, 8});
  ExpectParallelValues(*executor_, *third, {8, 7, 6, 5, 4});
  ExpectParallelValues(*executor_, *fourth, {3, 2, 1});
}

TEST_F(LayersTest, ParallelRejectsIncompatibleStatesAndGradientSizes) {
  auto parallel = BuildRoutingParallel();
  ASSERT_TRUE(parallel.ok()) << parallel.status();
  auto first = UploadParallelValues(*executor_, {1, 2, 3});
  auto second = UploadParallelValues(*executor_, {4, 5, 6, 7, 8});
  ASSERT_TRUE(first.ok()) << first.status();
  ASSERT_TRUE(second.ok()) << second.status();
  EXPECT_EQ((*parallel)->fwd(*executor_, {*first}).status().code(),
            absl::StatusCode::kInvalidArgument);
  auto fwd = (*parallel)->fwd(*executor_, {*first, *second});
  ASSERT_TRUE(fwd.ok()) << fwd.status();
  const BufferVec gradients = {*first, *second, *second, *first};
  auto missing_child = fwd->state;
  missing_child.children.pop_back();
  EXPECT_EQ((*parallel)
                ->bwd(*executor_, gradients, std::move(missing_child))
                .status()
                .code(),
            absl::StatusCode::kInvalidArgument);
  auto wrong_child = fwd->state;
  std::swap(wrong_child.children[0], wrong_child.children[1]);
  EXPECT_EQ((*parallel)
                ->bwd(*executor_, gradients, std::move(wrong_child))
                .status()
                .code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ((*parallel)->bwd(*executor_, {*first}, fwd->state).status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(
      (*parallel)
          ->bwd(*executor_, {*first, *second, *first, *second}, fwd->state)
          .status()
          .code(),
      absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(
      (*parallel)->bwd(*executor_, gradients, std::move(fwd->state)).ok());
}

TEST_F(LayersTest, ParallelHooksFollowScopesAndForwardReplacements) {
  auto parallel = BuildRoutingParallel();
  ASSERT_TRUE(parallel.ok()) << parallel.status();
  auto first = UploadParallelValues(*executor_, {1, 2, 3});
  auto second = UploadParallelValues(*executor_, {4, 5, 6, 7, 8});
  auto replacement = UploadParallelValues(*executor_, {9, 9, 9});
  ASSERT_TRUE(first.ok()) << first.status();
  ASSERT_TRUE(second.ok()) << second.status();
  ASSERT_TRUE(replacement.ok()) << replacement.status();
  std::vector<std::string> events;
  LayerHooks hooks;
  hooks.enter_combinator = [&](auto&, auto name) {
    events.push_back("enter:" + std::string(name));
    return absl::OkStatus();
  };
  hooks.exit_combinator = [&](auto&) {
    events.push_back("exit");
    return absl::OkStatus();
  };
  hooks.activation_hook = [&](auto&, auto name, auto, auto buffers) {
    events.push_back("forward:" + std::string(name));
    if (name == "left")
      buffers[0] = *replacement;
    return absl::OkStatus();
  };
  hooks.gradient_hook = [&](auto&, auto name, auto, auto buffers) {
    events.push_back("backward:" + std::string(name));
    if (name == "left")
      buffers[0] = *replacement;
    return absl::OkStatus();
  };
  auto fwd = (*parallel)->fwd(*executor_, {*first, *second}, &hooks);
  ASSERT_TRUE(fwd.ok()) << fwd.status();
  EXPECT_EQ(events, (std::vector<std::string>{"enter:routing", "forward:left",
                                              "forward:right", "exit",
                                              "forward:routing"}));
  EXPECT_EQ(fwd->outputs[0].data(), replacement->data());
  EXPECT_EQ(fwd->outputs[3].data(), first->data());
  events.clear();
  auto bwd = (*parallel)->bwd(*executor_, {*first, *second, *second, *first},
                              std::move(fwd->state), &hooks);
  ASSERT_TRUE(bwd.ok()) << bwd.status();
  EXPECT_EQ(events, (std::vector<std::string>{"backward:routing",
                                              "enter:routing", "backward:left",
                                              "backward:right", "exit"}));
  ExpectParallelValues(*executor_, (*bwd)[0], {10, 11, 12});
  ExpectParallelValues(*executor_, *first, {1, 2, 3});
}

TEST_F(LayersTest, ParallelHooksUnwindFailuresAndSkipFailedEntries) {
  auto parallel = BuildRoutingParallel();
  ASSERT_TRUE(parallel.ok()) << parallel.status();
  auto first = UploadParallelValues(*executor_, {1, 2, 3});
  auto second = UploadParallelValues(*executor_, {4, 5, 6, 7, 8});
  ASSERT_TRUE(first.ok()) << first.status();
  ASSERT_TRUE(second.ok()) << second.status();
  LayerHooks hooks;
  int entered = 0;
  int exited = 0;
  int child_calls = 0;
  hooks.enter_combinator = [&](auto&, auto) {
    ++entered;
    return absl::OkStatus();
  };
  hooks.exit_combinator = [&](auto&) {
    ++exited;
    return absl::UnavailableError("exit failed");
  };
  hooks.activation_hook = [&](auto&, auto, auto, auto) {
    ++child_calls;
    return absl::AbortedError("child failed");
  };
  const auto failed = (*parallel)->fwd(*executor_, {*first, *second}, &hooks);
  EXPECT_EQ(failed.status().code(), absl::StatusCode::kAborted);
  EXPECT_NE(failed.status().message().find("child failed"), std::string::npos);
  EXPECT_NE(failed.status().message().find("exit failed"), std::string::npos);
  EXPECT_EQ(entered, 1);
  EXPECT_EQ(exited, 1);
  EXPECT_EQ(child_calls, 1);
  hooks.enter_combinator = [&](auto&, auto) {
    ++entered;
    return absl::CancelledError("enter failed");
  };
  EXPECT_EQ(
      (*parallel)->fwd(*executor_, {*first, *second}, &hooks).status().code(),
      absl::StatusCode::kCancelled);
  EXPECT_EQ(entered, 2);
  EXPECT_EQ(exited, 1);
  EXPECT_EQ(child_calls, 1);
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
