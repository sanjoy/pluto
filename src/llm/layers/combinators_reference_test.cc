#include <cmath>
#include <cstddef>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "gtest/gtest.h"
#include "src/llm/layer.h"
#include "src/llm/layers/combinators.h"
#include "src/llm/layers/fully_connected.h"
#include "src/llm/layers/gelu.h"
#include "src/llm/layers/reference_test_util.h"

namespace pluto::llm {
namespace {

// These fakes isolate signature plumbing from numerical kernels. The reference
// fake can declare arbitrary signatures for construction-only rejection tests;
// forward/backward are identity operations only for its equal-signature uses.
class SignatureReference final : public LayerReference {
 public:
  absl::string_view name() const override { return "SignatureReference"; }

  SignatureReference(std::vector<ActivationType> inputs,
                     std::vector<ActivationType> outputs,
                     DataType policy = DataType::FP16)
      : inputs_(std::move(inputs)),
        outputs_(std::move(outputs)),
        policy_(policy) {}

  absl::Span<const ActivationType> input_types() const override {
    return inputs_;
  }
  absl::Span<const ActivationType> output_types() const override {
    return outputs_;
  }
  absl::Span<HostBuffer> weights() override { return {}; }
  DataType output_type() const override { return policy_; }

 private:
  absl::StatusOr<ReferenceFwdResult> fwd_impl(
      absl::Span<const HostBuffer> inputs) const override {
    return ReferenceFwdResult{HostBufferVec(inputs.begin(), inputs.end()), {}};
  }
  absl::StatusOr<HostBufferVec> bwd_impl(
      absl::Span<const HostBuffer> gradients,
      ReferenceBackwardState state) override {
    return HostBufferVec(gradients.begin(), gradients.end());
  }

  const std::vector<ActivationType> inputs_;
  const std::vector<ActivationType> outputs_;
  const DataType policy_;
};

class SignatureIdentity final : public Layer {
 public:
  absl::string_view name() const override { return "SignatureIdentity"; }

  SignatureIdentity(DataType storage, DataType policy, int width = 16,
                    int sequence_length = 1)
      : type_{storage,
              {ActivationType::kBatchDimension, sequence_length, width}},
        policy_(policy) {}

  absl::Span<const ActivationType> input_types() const override {
    return absl::Span<const ActivationType>(&type_, 1);
  }
  absl::Span<const ActivationType> output_types() const override {
    return input_types();
  }
  absl::Span<Buffer> weights() override { return {}; }
  DataType output_type() const override { return policy_; }

 private:
  absl::StatusOr<FwdResult> fwd_impl(cuda::Executor& executor,
                                     absl::Span<const Buffer> inputs,
                                     LayerHooks*) const override {
    return FwdResult{BufferVec(inputs.begin(), inputs.end()), {}};
  }
  absl::StatusOr<BufferVec> bwd_impl(cuda::Executor& executor,
                                     absl::Span<const Buffer> gradients,
                                     BackwardState state,
                                     LayerHooks*) override {
    return BufferVec(gradients.begin(), gradients.end());
  }

  const ActivationType type_;
  const DataType policy_;
};

TEST(ReferenceCombinatorNamesTest, OwnsNamesBeyondCallerStringLifetime) {
  const ActivationType type(DataType::FP32, {-2, 1, 16});
  const auto identity = [&]() {
    return absl::make_unique<SignatureReference>(
        std::vector<ActivationType>{type}, std::vector<ActivationType>{type});
  };
  std::string source = "reference transformer block zero / attention and MLP";
  const std::string expected = source;
  std::vector<std::unique_ptr<LayerReference>> first_children;
  first_children.push_back(identity());
  auto first =
      ComposedLayerReference::Create(source, std::move(first_children));
  ASSERT_TRUE(first.ok()) << first.status();
  const absl::string_view retained_name = (*first)->name();
  source.assign(512, 'x');
  EXPECT_EQ((*first)->name(), expected);
  EXPECT_EQ(retained_name, expected);

  std::vector<std::unique_ptr<LayerReference>> second_children;
  second_children.push_back(identity());
  auto second = ComposedLayerReference::Create(
      std::string("reference transformer block one / ") + "attention and MLP",
      std::move(second_children));
  ASSERT_TRUE(second.ok()) << second.status();
  EXPECT_EQ((*second)->name(),
            "reference transformer block one / attention and MLP");
  EXPECT_NE((*first)->name(), (*second)->name());
  EXPECT_EQ(retained_name, expected);
}

TEST(ReferenceCombinatorNamesTest, DirectFactoryRejectsEmptyName) {
  const ActivationType type(DataType::FP32, {-2, 1, 16});
  std::vector<std::unique_ptr<LayerReference>> children;
  children.push_back(absl::make_unique<SignatureReference>(
      std::vector<ActivationType>{type}, std::vector<ActivationType>{type}));
  EXPECT_EQ(
      ComposedLayerReference::Create("", std::move(children)).status().code(),
      absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(ComposedLayerReference::Create("", {}).status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(
      ComposedLayerReference::Create("EmptyReference", {}).status().code(),
      absl::StatusCode::kFailedPrecondition);
}

TEST(ReferenceCombinatorNamesTest,
     InvalidNamePreservesChildrenForRetryAndReuse) {
  const ActivationType type(DataType::FP32, {-2, 1, 16});
  const auto identity = [&]() {
    return absl::make_unique<SignatureReference>(
        std::vector<ActivationType>{type}, std::vector<ActivationType>{type});
  };
  ComposedLayerReferenceBuilder builder;
  EXPECT_EQ(builder.create("").status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(builder.back(), nullptr);
  ASSERT_TRUE(builder.add(identity()).ok());
  const LayerReference* first = builder.back();
  ASSERT_TRUE(builder.add(identity()).ok());
  const LayerReference* last = builder.back();
  EXPECT_EQ(builder.create("").status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(builder.back(), last);
  const ComposedLayerReferenceBuilder& const_builder = builder;
  EXPECT_EQ(const_builder.back(), last);

  std::string name = "RetriedNamedReferenceComposition";
  auto graph = builder.create(name);
  ASSERT_TRUE(graph.ok()) << graph.status();
  name.assign(512, 'x');
  EXPECT_EQ((*graph)->name(), "RetriedNamedReferenceComposition");
  EXPECT_EQ((*graph)->input_types().data(), first->input_types().data());
  EXPECT_EQ((*graph)->output_types().data(), last->output_types().data());
  EXPECT_EQ(builder.back(), nullptr);

  ASSERT_TRUE(builder.add(identity()).ok());
  auto reused = builder.create("ReusedNamedReferenceComposition");
  ASSERT_TRUE(reused.ok()) << reused.status();
  EXPECT_EQ((*reused)->name(), "ReusedNamedReferenceComposition");
  EXPECT_EQ((*graph)->name(), "RetriedNamedReferenceComposition");
}

TEST_F(LayerReferenceTest, ResidualStorageFollowsSignatureNotComputePolicy) {
  for (DataType storage : {DataType::FP32, DataType::BF16}) {
    // Deliberately disagree in both directions. Counting FP32 bytes as BF16
    // overreads the buffer; counting BF16 bytes as FP32 leaves half unwritten.
    const DataType policy =
        storage == DataType::FP32 ? DataType::BF16 : DataType::FP16;
    const DataType storage_policy =
        storage == DataType::FP32 ? DataType::FP16 : DataType::BF16;
    const ActivationType type(storage,
                              {ActivationType::kBatchDimension, 1, 16});
    auto device = ResidualLayer::Create(
        absl::make_unique<SignatureIdentity>(storage, policy));
    auto reference =
        ResidualLayerReference::Create(absl::make_unique<SignatureReference>(
            std::vector<ActivationType>{type},
            std::vector<ActivationType>{type}, policy));
    ASSERT_TRUE(device.ok()) << device.status();
    ASSERT_TRUE(reference.ok()) << reference.status();

    std::vector<float> input(32);
    std::vector<float> expected(32);
    for (size_t index = 0; index < input.size(); ++index) {
      input[index] = (static_cast<int>(index % 11) - 5) / 8.0f;
      expected[index] = 2.0f * input[index];
    }
    auto buffers = MakeActivationBufferPair(*executor_, input, storage_policy);
    ASSERT_TRUE(buffers.ok()) << buffers.status();
    BufferVec device_inputs{buffers->device};
    HostBufferVec reference_inputs{buffers->host};
    auto actual = (*device)->fwd(*executor_, device_inputs);
    auto expected_reference = (*reference)->fwd(reference_inputs);
    ASSERT_TRUE(actual.ok()) << actual.status();
    ASSERT_TRUE(expected_reference.ok()) << expected_reference.status();
    ASSERT_EQ(actual->outputs.size(), 1);
    ASSERT_EQ(expected_reference->outputs.size(), 1);
    EXPECT_EQ(actual->outputs[0].size_bytes(), buffers->device.size_bytes());
    EXPECT_EQ(expected_reference->outputs[0].size_bytes(),
              buffers->host.size_bytes());
    auto readback =
        ReadDeviceActivations(*executor_, actual->outputs[0], storage_policy);
    ASSERT_TRUE(readback.ok()) << readback.status();
    EXPECT_TRUE(VectorsNear(readback->span(), expected, 0.0f));
    EXPECT_TRUE(VectorsNear(
        ReadHostActivations(expected_reference->outputs[0], storage_policy),
        expected, 0.0f));
  }
}

TEST_F(LayerReferenceTest, ResidualMasksPartialTilesInForwardAndBackward) {
  for (DataType policy : {DataType::FP16, DataType::BF16}) {
    const DataType storage =
        policy == DataType::BF16 ? DataType::BF16 : DataType::FP32;
    // Include fewer than one tile and the real model's one/two sequence
    // batches. Identity branches isolate residual addition from other layers.
    for (const auto [rows, width] :
         {std::tuple{1, 1}, std::tuple{1, 13}, std::tuple{27, 13},
          std::tuple{54, 13}, std::tuple{27, 26}}) {
      SCOPED_TRACE(testing::Message()
                   << "policy=" << static_cast<int>(policy) << " rows=" << rows
                   << " width=" << width);
      const ActivationType type(storage,
                                {ActivationType::kBatchDimension, rows, width});
      auto device = ResidualLayer::Create(
          absl::make_unique<SignatureIdentity>(storage, policy, width, rows));
      auto reference =
          ResidualLayerReference::Create(absl::make_unique<SignatureReference>(
              std::vector<ActivationType>{type},
              std::vector<ActivationType>{type}, policy));
      ASSERT_TRUE(device.ok()) << device.status();
      ASSERT_TRUE(reference.ok()) << reference.status();

      std::vector<float> input(rows * width);
      std::vector<float> upstream(rows * width);
      std::vector<float> expected_output(rows * width);
      std::vector<float> expected_gradient(rows * width);
      for (size_t i = 0; i < input.size(); ++i) {
        // Exact binary fractions keep the expected sums exact in both dtypes.
        // Every tail element is nonzero, so silently skipping it cannot pass.
        input[i] = static_cast<float>(1 + i % 19) / 32.0f;
        upstream[i] = -static_cast<float>(1 + i % 13) / 16.0f;
        expected_output[i] = 2.0f * input[i];
        expected_gradient[i] = 2.0f * upstream[i];
      }
      auto inputs = MakeActivationBufferPair(*executor_, input, policy);
      auto gradients = MakeRawBufferPair<float>(*executor_, upstream);
      ASSERT_TRUE(inputs.ok()) << inputs.status();
      ASSERT_TRUE(gradients.ok()) << gradients.status();
      auto actual = (*device)->fwd(*executor_, BufferVec{inputs->device});
      auto expected = (*reference)->fwd(HostBufferVec{inputs->host});
      ASSERT_TRUE(actual.ok()) << actual.status();
      ASSERT_TRUE(expected.ok()) << expected.status();
      EXPECT_TRUE(ActivationBuffersNear(actual->outputs[0],
                                        expected->outputs[0], policy, 0.0f));
      auto actual_values =
          ReadDeviceActivations(*executor_, actual->outputs[0], policy);
      ASSERT_TRUE(actual_values.ok()) << actual_values.status();
      EXPECT_TRUE(VectorsNear(actual_values->span(), expected_output, 0.0f));

      auto actual_backward = (*device)->bwd(
          *executor_, BufferVec{gradients->device}, std::move(actual->state));
      auto expected_backward =
          (*reference)
              ->bwd(HostBufferVec{gradients->host}, std::move(expected->state));
      ASSERT_TRUE(actual_backward.ok()) << actual_backward.status();
      ASSERT_TRUE(expected_backward.ok()) << expected_backward.status();
      ASSERT_EQ(actual_backward->size(), 1u);
      ASSERT_EQ(expected_backward->size(), 1u);
      EXPECT_TRUE(FloatBuffersNear(actual_backward->front(),
                                   expected_backward->front(), 0.0f));
      auto actual_gradients = ReadDeviceActivations(
          *executor_, actual_backward->front(), DataType::FP16);
      ASSERT_TRUE(actual_gradients.ok()) << actual_gradients.status();
      EXPECT_TRUE(
          VectorsNear(actual_gradients->span(), expected_gradient, 0.0f));
    }
  }
}

TEST(ReferenceCombinatorTypesTest, ConnectionsRequireExactSignatures) {
  constexpr int64_t kBatch = ActivationType::kBatchDimension;
  const ActivationType base(DataType::FP32, {kBatch, 8, 32});
  const std::vector<std::vector<ActivationType>> mismatches = {
      {{DataType::FP32, {3, 8, 32}}},       // Batch is not a fixed number.
      {{DataType::FP32, {kBatch, 8, 64}}},  // Feature width differs.
      {{DataType::FP32, {kBatch, 256}}},    // Same size, different rank.
      {{DataType::BF16, {kBatch, 8, 32}}},  // Different physical dtype.
      {base, base},                         // Different buffer arity.
  };
  for (const auto& mismatch : mismatches) {
    ComposedLayerReferenceBuilder builder;
    ASSERT_TRUE(builder
                    .add(absl::make_unique<SignatureReference>(
                        std::vector<ActivationType>{base},
                        std::vector<ActivationType>{base}))
                    .ok());
    const LayerReference* first = builder.back();
    const auto status = builder.add(absl::make_unique<SignatureReference>(
        mismatch, std::vector<ActivationType>{base}));
    EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
    EXPECT_EQ(builder.back(), first);

    // Rejection leaves the existing composition usable.
    ASSERT_TRUE(builder
                    .add(absl::make_unique<SignatureReference>(
                        std::vector<ActivationType>{base},
                        std::vector<ActivationType>{base}))
                    .ok());
    EXPECT_TRUE(builder.create("RetriedReferenceConnection").ok());

    std::vector<std::unique_ptr<LayerReference>> children;
    children.push_back(absl::make_unique<SignatureReference>(
        std::vector<ActivationType>{base}, std::vector<ActivationType>{base}));
    children.push_back(absl::make_unique<SignatureReference>(
        mismatch, std::vector<ActivationType>{base}));
    EXPECT_EQ(ComposedLayerReference::Create("InvalidReferenceConnection",
                                             std::move(children))
                  .status()
                  .code(),
              absl::StatusCode::kInvalidArgument);
    EXPECT_EQ(ResidualLayerReference::Create(
                  absl::make_unique<SignatureReference>(
                      std::vector<ActivationType>{base}, mismatch))
                  .status()
                  .code(),
              absl::StatusCode::kInvalidArgument);
  }
}

TEST(ReferenceCombinatorTypesTest, InvalidDimensionsNeverMatchEvenThemselves) {
  for (const ActivationType invalid :
       {ActivationType(DataType::FP32, {-1, 8, 32}),
        ActivationType(DataType::FP32, {8, -2, 32}),
        ActivationType(DataType::FP32, {-2, 0, 32})}) {
    ComposedLayerReferenceBuilder builder;
    EXPECT_EQ(builder
                  .add(absl::make_unique<SignatureReference>(
                      std::vector<ActivationType>{invalid},
                      std::vector<ActivationType>{invalid}))
                  .code(),
              absl::StatusCode::kInvalidArgument);
    EXPECT_EQ(builder.back(), nullptr);
    EXPECT_EQ(ResidualLayerReference::Create(
                  absl::make_unique<SignatureReference>(
                      std::vector<ActivationType>{invalid},
                      std::vector<ActivationType>{invalid}))
                  .status()
                  .code(),
              absl::StatusCode::kInvalidArgument);
  }
  const ActivationType integers(DataType::INT32, {-2, 8});
  EXPECT_EQ(
      ResidualLayerReference::Create(absl::make_unique<SignatureReference>(
                                         std::vector<ActivationType>{integers},
                                         std::vector<ActivationType>{integers}))
          .status()
          .code(),
      absl::StatusCode::kUnimplemented);
}

TEST(ReferenceCombinatorTypesTest, NestedSignaturesPropagateWithoutFlattening) {
  const ActivationType tokens(DataType::INT32, {-2, 8});
  const ActivationType hidden(DataType::BF16, {-2, 8, 32});
  const ActivationType logits(DataType::FP32, {-2, 8, 48});
  ComposedLayerReferenceBuilder inside;
  ASSERT_TRUE(inside
                  .add(absl::make_unique<SignatureReference>(
                      std::vector<ActivationType>{tokens},
                      std::vector<ActivationType>{hidden}))
                  .ok());
  ASSERT_TRUE(inside
                  .add(ResidualLayerReference::Create(
                      absl::make_unique<SignatureReference>(
                          std::vector<ActivationType>{hidden},
                          std::vector<ActivationType>{hidden}, DataType::BF16)))
                  .ok());
  auto nested = inside.create("ReferenceEmbeddingPipeline");
  ASSERT_TRUE(nested.ok()) << nested.status();
  EXPECT_EQ((*nested)->input_types()[0], tokens);
  EXPECT_EQ((*nested)->output_types()[0], hidden);

  ComposedLayerReferenceBuilder outside;
  ASSERT_TRUE(outside.add(std::move(nested)).ok());
  ASSERT_TRUE(outside
                  .add(absl::make_unique<SignatureReference>(
                      std::vector<ActivationType>{hidden},
                      std::vector<ActivationType>{logits}))
                  .ok());
  auto model = outside.create("ReferenceLogitsPipeline");
  ASSERT_TRUE(model.ok()) << model.status();
  EXPECT_EQ((*model)->input_types()[0], tokens);
  EXPECT_EQ((*model)->output_types()[0], logits);
}

TEST_F(LayerReferenceTest, ResidualCompositionAndBuildersMatchBothPasses) {
  for (DataType type : {DataType::FP16, DataType::BF16}) {
    for (const auto [rows, width] : {std::tuple{16, 16}, std::tuple{32, 32}}) {
      SCOPED_TRACE(testing::Message()
                   << "type=" << static_cast<int>(type) << " rows=" << rows
                   << " width=" << width);
      auto device_dense =
          FullyConnectedLayer::Create(*executor_, width, width, type);
      auto reference_dense =
          FullyConnectedLayerReference::Create(width, width, type);
      ASSERT_TRUE(device_dense.ok()) << device_dense.status();
      ASSERT_TRUE(reference_dense.ok()) << reference_dense.status();
      std::vector<float> matrix(static_cast<size_t>(width) * width);
      std::vector<float> bias(width);
      for (size_t index = 0; index < matrix.size(); ++index)
        matrix[index] = 0.05f * std::sin(static_cast<float>(index) * 0.23f);
      for (int index = 0; index < width; ++index)
        bias[index] = 0.02f * std::cos(index * 0.31f);
      auto device_dense_weights = (*device_dense)->weights();
      auto reference_dense_weights = (*reference_dense)->weights();
      ASSERT_TRUE(SetFloatBufferPair(*executor_, device_dense_weights[0],
                                     &reference_dense_weights[0], matrix)
                      .ok());
      ASSERT_TRUE(SetFloatBufferPair(*executor_, device_dense_weights[1],
                                     &reference_dense_weights[1], bias)
                      .ok());

      ComposedLayerBuilder device_builder;
      ComposedLayerReferenceBuilder reference_builder;
      ASSERT_TRUE(
          device_builder.add(ResidualLayer::Create(std::move(*device_dense)))
              .ok());
      ASSERT_TRUE(
          reference_builder
              .add(ResidualLayerReference::Create(std::move(*reference_dense)))
              .ok());
      ASSERT_NE(device_builder.back(), nullptr);
      ASSERT_NE(reference_builder.back(), nullptr);
      ASSERT_TRUE(
          device_builder.add(GeluLayer::Create(*executor_, width, type)).ok());
      ASSERT_TRUE(
          reference_builder.add(GeluLayerReference::Create(width, type)).ok());
      auto device_model = device_builder.create("ResidualDenseGelu");
      auto reference_model = reference_builder.create("ResidualDenseGelu");
      ASSERT_TRUE(device_model.ok()) << device_model.status();
      ASSERT_TRUE(reference_model.ok()) << reference_model.status();
      EXPECT_EQ((*device_model)->name(), "ResidualDenseGelu");
      EXPECT_EQ((*reference_model)->name(), (*device_model)->name());
      EXPECT_EQ(device_builder.back(), nullptr);
      EXPECT_EQ(reference_builder.back(), nullptr);

      std::vector<float> input(static_cast<size_t>(rows) * width);
      std::vector<float> gradient(input.size());
      for (size_t index = 0; index < input.size(); ++index) {
        input[index] = 0.5f * std::sin(static_cast<float>(index) * 0.1f);
        gradient[index] = 0.2f * std::cos(static_cast<float>(index) * 0.13f);
      }
      auto input_pair = MakeActivationBufferPair(*executor_, input, type);
      auto gradient_pair = MakeRawBufferPair<float>(*executor_, gradient);
      ASSERT_TRUE(input_pair.ok()) << input_pair.status();
      ASSERT_TRUE(gradient_pair.ok()) << gradient_pair.status();

      BufferVec device_inputs = {input_pair->device};
      HostBufferVec reference_inputs = {input_pair->host};
      auto device_output = (*device_model)->fwd(*executor_, device_inputs);

      auto reference_output = (*reference_model)->fwd(reference_inputs);

      ASSERT_TRUE(device_output.ok()) << device_output.status();
      ASSERT_TRUE(reference_output.ok()) << reference_output.status();
      EXPECT_TRUE(ActivationBuffersNear(device_output->outputs[0],
                                        reference_output->outputs[0], type,
                                        4e-3f, 3e-3f));

      BufferVec device_gradients = {gradient_pair->device};
      HostBufferVec reference_gradients = {gradient_pair->host};
      auto device_input = (*device_model)
                              ->bwd(*executor_, device_gradients,
                                    std::move(device_output->state));
      auto reference_input =
          (*reference_model)
              ->bwd(reference_gradients, std::move(reference_output->state));
      ASSERT_TRUE(device_input.ok()) << device_input.status();
      ASSERT_TRUE(reference_input.ok()) << reference_input.status();
      EXPECT_TRUE(FloatBuffersNear(device_input->front(),
                                   reference_input->front(), 5e-3f, 4e-3f));
      auto device_parameter_gradients = (*device_model)->gradients();
      auto reference_parameter_gradients = (*reference_model)->gradients();
      ASSERT_EQ(device_parameter_gradients.size(),
                reference_parameter_gradients.size());
      for (size_t index = 0; index < device_parameter_gradients.size();
           ++index) {
        EXPECT_TRUE(FloatBuffersNear(device_parameter_gradients[index],
                                     reference_parameter_gradients[index],
                                     5e-3f, 4e-3f));
      }
    }
  }
}

TEST_F(LayerReferenceTest, BuildersPropagateFP8Rejection) {
  ComposedLayerBuilder device_builder;
  ComposedLayerReferenceBuilder reference_builder;
  const absl::Status device_status =
      device_builder.add(GeluLayer::Create(*executor_, 16, DataType::FP8));
  const absl::Status reference_status =
      reference_builder.add(GeluLayerReference::Create(16, DataType::FP8));
  EXPECT_EQ(device_status.code(), absl::StatusCode::kUnimplemented);
  EXPECT_EQ(reference_status.code(), absl::StatusCode::kUnimplemented);
  EXPECT_EQ(device_builder.back(), nullptr);
  EXPECT_EQ(reference_builder.back(), nullptr);
}

}  // namespace
}  // namespace pluto::llm
