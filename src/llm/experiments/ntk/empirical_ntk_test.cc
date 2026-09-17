#include "src/llm/experiments/ntk/empirical_ntk.h"

#include <cuda_runtime_api.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "gtest/gtest.h"
#include "src/cuda/page_locked_host_array.h"
#include "src/llm/layer.h"
#include "src/llm/layers/combinators.h"
#include "src/llm/layers/embedding.h"
#include "src/llm/layers/fully_connected.h"
#include "src/llm/layers/gelu.h"
#include "src/util/status_macros.h"

namespace pluto::llm::ntk {
namespace {

constexpr int kWidth = 16;
constexpr int kRows = 16;

template <class T>
absl::Status Write(cuda::Executor& executor, const Buffer& destination,
                   const std::vector<T>& values) {
  if (destination.size_bytes() != values.size() * sizeof(T))
    return absl::InvalidArgumentError("test upload has the wrong size");
  ASSIGN_OR_RETURN(auto host, cuda::PageLockedHostArray<T>::CopyFrom(
                                  executor, absl::MakeConstSpan(values)));
  return cuda::CudaStatus(
      cudaMemcpyAsync(destination.data(), host.data(), host.size_bytes(),
                      cudaMemcpyHostToDevice, executor.stream()),
      "cudaMemcpyAsync(NTK test upload)");
}

template <class T>
absl::StatusOr<Buffer> Upload(cuda::Executor& executor,
                              const std::vector<T>& values) {
  ASSIGN_OR_RETURN(auto result,
                   Buffer::Allocate(executor, values.size() * sizeof(T)));
  RETURN_IF_ERROR(Write(executor, result, values));
  return result;
}

absl::StatusOr<std::vector<float>> Read(cuda::Executor& executor,
                                        const Buffer& source) {
  ASSIGN_OR_RETURN(auto host,
                   cuda::PageLockedHostArray<float>::Allocate(
                       executor, source.size_bytes() / sizeof(float)));
  RETURN_IF_ERROR(cuda::CudaStatus(
      cudaMemcpyAsync(host.data(), source.data(), source.size_bytes(),
                      cudaMemcpyDeviceToHost, executor.stream()),
      "cudaMemcpyAsync(NTK test download)"));
  RETURN_IF_ERROR(executor.Synchronize());
  return std::vector<float>(host.begin(), host.end());
}

absl::StatusOr<std::vector<std::vector<float>>> Snapshot(
    cuda::Executor& executor, absl::Span<const Buffer> buffers) {
  std::vector<std::vector<float>> values;
  for (const Buffer& buffer : buffers) {
    ASSIGN_OR_RETURN(auto value, Read(executor, buffer));
    values.push_back(std::move(value));
  }
  return values;
}

absl::Status SeedGradients(cuda::Executor& executor, Layer& model) {
  int tensor = 0;
  for (const Buffer& gradient : model.gradients()) {
    ++tensor;
    std::vector<float> values(gradient.size_bytes() / sizeof(float));
    for (size_t i = 0; i < values.size(); ++i)
      values[i] = static_cast<float>((i % 13) + tensor) / 8.0f;
    RETURN_IF_ERROR(Write(executor, gradient, values));
  }
  return absl::OkStatus();
}

// A construction-only probe catches invalid parameter aliasing before any
// forward work. Its failing backward deliberately changes a gradient first,
// ensuring restoration is tested after a real mutation, not a preflight error.
class ProbeLayer final : public Layer {
 public:
  ProbeLayer(BufferVec weights, BufferVec gradients, bool fail_backward = false)
      : weights_(std::move(weights)),
        gradients_(std::move(gradients)),
        fail_backward_(fail_backward) {}
  absl::string_view name() const override { return "ProbeLayer"; }
  absl::Span<const ActivationType> input_types() const override {
    return {&type_, 1};
  }
  absl::Span<const ActivationType> output_types() const override {
    return input_types();
  }
  absl::Span<Buffer> weights() override { return absl::MakeSpan(weights_); }
  absl::Span<Buffer> gradients() override { return absl::MakeSpan(gradients_); }
  DataType output_type() const override { return DataType::FP32; }
  mutable int forward_calls = 0;
  int backward_calls = 0;

 private:
  absl::StatusOr<FwdResult> fwd_impl(cuda::Executor&,
                                     absl::Span<const Buffer> inputs,
                                     LayerHooks*) const override {
    ++forward_calls;
    return FwdResult{.outputs = {inputs[0]}};
  }
  absl::StatusOr<BufferVec> bwd_impl(cuda::Executor& executor,
                                     absl::Span<const Buffer> gradients,
                                     BackwardState, LayerHooks*) override {
    ++backward_calls;
    if (fail_backward_) {
      RETURN_IF_ERROR(cuda::CudaStatus(
          cudaMemsetAsync(gradients_[0].data(), 0, gradients_[0].size_bytes(),
                          executor.stream()),
          "cudaMemsetAsync(injected NTK backward failure)"));
      return absl::DataLossError("injected backward failure");
    }
    return BufferVec(gradients.begin(), gradients.end());
  }
  BufferVec weights_;
  BufferVec gradients_;
  bool fail_backward_;
  ActivationType type_{DataType::FP32, {-2, 1, kWidth}};
};

class EmpiricalNtkTest : public testing::Test {
 protected:
  void SetUp() override {
    auto executor = cuda::Executor::Create();
    ASSERT_TRUE(executor.ok()) << executor.status();
    executor_ = std::move(*executor);
  }
  void TearDown() override {
    if (executor_ != nullptr)
      EXPECT_TRUE(executor_->Synchronize().ok());
  }
  std::unique_ptr<cuda::Executor> executor_;
};

TEST_F(EmpiricalNtkTest,
       AffineKernelMatchesAnalyticGramAcrossTokensAndOutputs) {
  auto model =
      FullyConnectedLayer::Create(*executor_, kWidth, DataType::FP16, 4);
  ASSERT_TRUE(model.ok()) << model.status();
  ASSERT_TRUE((*model)->InitializeIdentity(0.5f).ok());
  ASSERT_TRUE(SeedGradients(*executor_, **model).ok());
  auto original_weights = Snapshot(*executor_, (*model)->weights());
  auto original_gradients = Snapshot(*executor_, (*model)->gradients());
  ASSERT_TRUE(original_weights.ok()) << original_weights.status();
  ASSERT_TRUE(original_gradients.ok()) << original_gradients.status();
  std::vector<float> x(kRows * kWidth), y(x.size());
  for (size_t i = 0; i < x.size(); ++i) {
    x[i] = (static_cast<int>(i % 11) - 5) / 8.0f;
    y[i] = (static_cast<int>(i % 7) - 3) / 4.0f;
  }
  auto x_device = Upload(*executor_, x);
  auto y_device = Upload(*executor_, y);
  ASSERT_TRUE(x_device.ok()) << x_device.status();
  ASSERT_TRUE(y_device.ok()) << y_device.status();
  const std::vector<Sample> samples = {
      {{*x_device}, {{0, 0}, {0, 1}, {0, 2 * kWidth}, {0, 0}}},
      {{*y_device}, {{0, 0}, {0, 1}, {0, 3 * kWidth}, {0, 3 * kWidth + 1}}}};
  const std::vector<std::pair<const std::vector<float>*, size_t>> selected = {
      {&x, 0}, {&x, 1}, {&x, 2 * kWidth}, {&x, 0},
      {&y, 0}, {&y, 1}, {&y, 3 * kWidth}, {&y, 3 * kWidth + 1}};
  auto result = ComputeEmpiricalKernel(*executor_, **model, samples);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result->parameter_count, kWidth * kWidth + kWidth);
  ASSERT_EQ(result->parameters.size(), 2);
  EXPECT_EQ(result->parameters[0].weight_index, 0);
  EXPECT_EQ(result->parameters[0].elements, kWidth * kWidth);
  EXPECT_EQ(result->parameters[0].offset, 0);
  EXPECT_EQ(result->parameters[1].weight_index, 1);
  EXPECT_EQ(result->parameters[1].elements, kWidth);
  EXPECT_EQ(result->parameters[1].offset, kWidth * kWidth);
  ASSERT_EQ(result->gram.rows, selected.size());
  ASSERT_EQ(result->gram.columns, selected.size());
  ASSERT_EQ(result->initial_values.size(), selected.size());
  for (size_t i = 0; i < selected.size(); ++i) {
    const auto [input_i, element_i] = selected[i];
    EXPECT_DOUBLE_EQ(result->initial_values[i], 0.5 * (*input_i)[element_i]);
    for (size_t j = 0; j < selected.size(); ++j) {
      const auto [input_j, element_j] = selected[j];
      double expected = 0.0;
      if (element_i % kWidth == element_j % kWidth) {
        expected = 1.0;  // Bias contributes one, without batch normalization.
        for (int column = 0; column < kWidth; ++column)
          expected += (*input_i)[element_i / kWidth * kWidth + column] *
                      (*input_j)[element_j / kWidth * kWidth + column];
      }
      EXPECT_DOUBLE_EQ(result->gram(i, j), expected) << i << ',' << j;
      EXPECT_DOUBLE_EQ(result->gram(i, j), result->gram(j, i));
    }
  }
  // Duplicate coordinates have identical Jacobians; all tested quadratic
  // forms are nonnegative, including vectors spanning both output channels.
  for (size_t i = 0; i < selected.size(); ++i)
    EXPECT_DOUBLE_EQ(result->gram(0, i), result->gram(3, i));
  for (int probe = 0; probe < 5; ++probe) {
    double quadratic = 0.0;
    for (size_t i = 0; i < selected.size(); ++i)
      for (size_t j = 0; j < selected.size(); ++j)
        quadratic += std::sin((i + 1) * (probe + 1)) * result->gram(i, j) *
                     std::sin((j + 1) * (probe + 1));
    EXPECT_GE(quadratic, -1e-12);
  }
  auto repeated = ComputeEmpiricalKernel(*executor_, **model, samples);
  ASSERT_TRUE(repeated.ok()) << repeated.status();
  EXPECT_EQ(repeated->gram.values, result->gram.values);
  EXPECT_EQ(repeated->initial_values, result->initial_values);
  auto weights_after = Snapshot(*executor_, (*model)->weights());
  auto gradients_after = Snapshot(*executor_, (*model)->gradients());
  ASSERT_TRUE(weights_after.ok()) << weights_after.status();
  ASSERT_TRUE(gradients_after.ok()) << gradients_after.status();
  EXPECT_EQ(*weights_after, *original_weights);
  EXPECT_EQ(*gradients_after, *original_gradients);
}

TEST_F(EmpiricalNtkTest, NonlinearKernelMatchesFiniteDifferences) {
  auto first = FullyConnectedLayer::Create(*executor_, kWidth, DataType::FP16);
  auto last = FullyConnectedLayer::Create(*executor_, kWidth, DataType::FP16);
  ASSERT_TRUE(first.ok()) << first.status();
  ASSERT_TRUE(last.ok()) << last.status();
  ASSERT_TRUE((*first)->InitializeIdentity(0.5f).ok());
  ASSERT_TRUE((*last)->InitializeIdentity(0.5f).ok());
  ComposedLayerBuilder builder;
  ASSERT_TRUE(builder.add(std::move(*first)).ok());
  ASSERT_TRUE(
      builder.add(GeluLayer::Create(*executor_, kWidth, DataType::FP16)).ok());
  ASSERT_TRUE(builder.add(std::move(*last)).ok());
  auto model = builder.create("FiniteDifferenceMlp");
  ASSERT_TRUE(model.ok()) << model.status();
  std::vector<float> inputs(kRows * kWidth, 0.0f);
  inputs[0] = 0.5f;
  inputs[kWidth] = -0.25f;
  auto input = Upload(*executor_, inputs);
  ASSERT_TRUE(input.ok()) << input.status();
  const std::vector<Sample> samples = {{{*input}, {{0, 0}, {0, kWidth}}}};
  auto result = ComputeEmpiricalKernel(*executor_, **model, samples);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result->parameter_count, 2 * (kWidth * kWidth + kWidth));
  auto weights = Snapshot(*executor_, (*model)->weights());
  ASSERT_TRUE(weights.ok()) << weights.status();
  ASSERT_EQ(weights->size(), 4);

  // At diagonal weights with input confined to channel zero, these two outputs
  // depend on exactly four parameters: W1[0,0], b1[0], W2[0,0], b2[0]. Every
  // other parameter's first derivative is zero. This sparse support permits a
  // complete finite-difference Gram oracle with only eight forward passes.
  // A binary 1/32 step is deliberately larger than FP16 operand quantization;
  // an infinitesimal step would test rounding plateaus, not the backward rule.
  constexpr float kEpsilon = 1.0f / 32.0f;
  std::array<std::array<double, 4>, 2> jacobian{};
  for (size_t parameter = 0; parameter < weights->size(); ++parameter) {
    std::array<std::vector<float>, 2> outputs;
    for (size_t sign = 0; sign < 2; ++sign) {
      auto perturbed = (*weights)[parameter];
      perturbed[0] += sign == 0 ? kEpsilon : -kEpsilon;
      ASSERT_TRUE(
          Write(*executor_, (*model)->weights()[parameter], perturbed).ok());
      auto forward = (*model)->fwd(*executor_, {*input});
      ASSERT_TRUE(forward.ok()) << forward.status();
      auto values = Read(*executor_, forward->outputs[0]);
      ASSERT_TRUE(values.ok()) << values.status();
      outputs[sign] = std::move(*values);
    }
    ASSERT_TRUE(
        Write(*executor_, (*model)->weights()[parameter], (*weights)[parameter])
            .ok());
    for (size_t row = 0; row < 2; ++row)
      jacobian[row][parameter] =
          (outputs[0][row * kWidth] - outputs[1][row * kWidth]) /
          (2.0 * kEpsilon);
  }
  for (size_t i = 0; i < 2; ++i)
    for (size_t j = 0; j < 2; ++j) {
      double expected = 0.0;
      for (size_t parameter = 0; parameter < 4; ++parameter)
        expected += jacobian[i][parameter] * jacobian[j][parameter];
      EXPECT_NEAR(result->gram(i, j), expected, 0.003) << i << ',' << j;
    }
}

TEST_F(EmpiricalNtkTest,
       TiedEmbeddingHeadAccumulatesAndCountsSharedWeightsOnce) {
  for (DataType type : {DataType::FP16, DataType::BF16}) {
    SCOPED_TRACE(static_cast<int>(type));
    auto embedding =
        EmbeddingLookupLayer::Create(*executor_, kWidth, kWidth, type);
    ASSERT_TRUE(embedding.ok()) << embedding.status();
    ASSERT_TRUE((*embedding)->InitializeIdentity().ok());
    auto head = LanguageModelingHeadLayer::Create(embedding->get());
    ASSERT_TRUE(head.ok()) << head.status();
    ComposedLayerBuilder builder;
    ASSERT_TRUE(builder.add(std::move(*embedding)).ok());
    ASSERT_TRUE(builder.add(std::move(*head)).ok());
    auto model = builder.create("TiedEmbeddingReadout");
    ASSERT_TRUE(model.ok()) << model.status();
    ASSERT_EQ((*model)->weights().size(), 2);
    ASSERT_EQ((*model)->weights()[0].data(), (*model)->weights()[1].data());
    ASSERT_TRUE(SeedGradients(*executor_, **model).ok());
    auto gradients_before = Snapshot(*executor_, (*model)->gradients());
    ASSERT_TRUE(gradients_before.ok()) << gradients_before.status();
    std::vector<int> token_ids(kRows);
    for (int row = 0; row < kRows; ++row)
      token_ids[row] = row;
    auto tokens = Upload(*executor_, token_ids);
    ASSERT_TRUE(tokens.ok()) << tokens.status();
    const std::vector<size_t> elements{0, 1, kWidth, kWidth + 1, 2, 2 * kWidth};
    Sample sample{.inputs = {*tokens}};
    std::vector<std::vector<double>> jacobian;
    for (size_t element : elements) {
      sample.coordinates.push_back({0, element});
      const size_t token = element / kWidth;
      const size_t prediction = element % kWidth;
      std::vector<double> derivative(kWidth * kWidth, 0.0);
      // f(i)_j = E_i dot E_j: both occurrences differentiate the SAME E.
      derivative[token * kWidth + prediction] += 1.0;
      derivative[prediction * kWidth + token] += 1.0;
      jacobian.push_back(std::move(derivative));
    }
    auto result = ComputeEmpiricalKernel(*executor_, **model, {&sample, 1});
    ASSERT_TRUE(result.ok()) << result.status();
    EXPECT_EQ(result->parameter_count, kWidth * kWidth);
    ASSERT_EQ(result->parameters.size(), 1);
    EXPECT_EQ(result->parameters[0].weight_index, 0);
    EXPECT_EQ(result->parameters[0].elements, kWidth * kWidth);
    EXPECT_EQ(result->parameters[0].offset, 0);
    for (size_t i = 0; i < elements.size(); ++i) {
      EXPECT_DOUBLE_EQ(
          result->initial_values[i],
          elements[i] / kWidth == elements[i] % kWidth ? 1.0 : 0.0);
      for (size_t j = 0; j < elements.size(); ++j) {
        double expected = 0.0;
        for (size_t parameter = 0; parameter < jacobian[i].size(); ++parameter)
          expected += jacobian[i][parameter] * jacobian[j][parameter];
        EXPECT_DOUBLE_EQ(result->gram(i, j), expected) << i << ',' << j;
      }
    }
    EXPECT_DOUBLE_EQ(result->gram(0, 0), 4.0);
    EXPECT_DOUBLE_EQ(result->gram(1, 1), 2.0);
    EXPECT_DOUBLE_EQ(result->gram(1, 2), 2.0);
    auto gradients_after = Snapshot(*executor_, (*model)->gradients());
    ASSERT_TRUE(gradients_after.ok()) << gradients_after.status();
    EXPECT_EQ(*gradients_after, *gradients_before);
  }
}

TEST_F(EmpiricalNtkTest, ProgressFailureRestoresWeightsAndExistingGradients) {
  auto model = FullyConnectedLayer::Create(*executor_, kWidth, DataType::FP16);
  ASSERT_TRUE(model.ok()) << model.status();
  ASSERT_TRUE((*model)->InitializeIdentity().ok());
  ASSERT_TRUE(SeedGradients(*executor_, **model).ok());
  auto weights_before = Snapshot(*executor_, (*model)->weights());
  auto gradients_before = Snapshot(*executor_, (*model)->gradients());
  ASSERT_TRUE(weights_before.ok()) << weights_before.status();
  ASSERT_TRUE(gradients_before.ok()) << gradients_before.status();
  auto input = Upload(*executor_, std::vector<float>(kRows * kWidth, 0.5f));
  ASSERT_TRUE(input.ok()) << input.status();
  Sample sample{.inputs = {*input}, .coordinates = {{0, 0}, {0, 1}}};
  KernelOptions options;
  bool cancelled = false;
  options.progress = [&](size_t completed, size_t total) {
    EXPECT_EQ(total, 2);
    if (completed == 0)
      return absl::OkStatus();
    cancelled = true;
    return absl::AbortedError("cancelled by test");
  };
  auto result =
      ComputeEmpiricalKernel(*executor_, **model, {&sample, 1}, options);
  EXPECT_EQ(result.status().code(), absl::StatusCode::kAborted);
  EXPECT_EQ(result.status().message(), "cancelled by test");
  EXPECT_TRUE(cancelled);
  auto weights_after = Snapshot(*executor_, (*model)->weights());
  auto gradients_after = Snapshot(*executor_, (*model)->gradients());
  ASSERT_TRUE(weights_after.ok()) << weights_after.status();
  ASSERT_TRUE(gradients_after.ok()) << gradients_after.status();
  EXPECT_EQ(*weights_after, *weights_before);
  EXPECT_EQ(*gradients_after, *gradients_before);
}

TEST_F(EmpiricalNtkTest, BackwardFailureRestoresMutatedGradientAccumulators) {
  auto weight = Upload(*executor_, std::vector<float>(kWidth, 2.0f));
  auto gradient = Upload(*executor_, std::vector<float>(kWidth, 3.0f));
  auto input = Upload(*executor_, std::vector<float>(kRows * kWidth, 0.5f));
  ASSERT_TRUE(weight.ok()) << weight.status();
  ASSERT_TRUE(gradient.ok()) << gradient.status();
  ASSERT_TRUE(input.ok()) << input.status();
  ProbeLayer model({*weight}, {*gradient}, true);
  Sample sample{.inputs = {*input}, .coordinates = {{0, 0}}};
  auto result = ComputeEmpiricalKernel(*executor_, model, {&sample, 1});
  EXPECT_EQ(result.status().code(), absl::StatusCode::kDataLoss);
  EXPECT_EQ(result.status().message(), "injected backward failure");
  EXPECT_EQ(model.backward_calls, 1);
  auto weight_after = Read(*executor_, *weight);
  auto gradient_after = Read(*executor_, *gradient);
  ASSERT_TRUE(weight_after.ok()) << weight_after.status();
  ASSERT_TRUE(gradient_after.ok()) << gradient_after.status();
  EXPECT_EQ(*weight_after, std::vector<float>(kWidth, 2.0f));
  EXPECT_EQ(*gradient_after, std::vector<float>(kWidth, 3.0f));
}

TEST_F(EmpiricalNtkTest, RejectsInvalidCoordinatesBudgetAndForeignInputs) {
  auto model = FullyConnectedLayer::Create(*executor_, kWidth, DataType::FP16);
  ASSERT_TRUE(model.ok()) << model.status();
  ASSERT_TRUE((*model)->InitializeIdentity().ok());
  auto input = Upload(*executor_, std::vector<float>(kRows * kWidth, 0.5f));
  ASSERT_TRUE(input.ok()) << input.status();
  EXPECT_FALSE(ComputeEmpiricalKernel(*executor_, **model, {}).ok());
  Sample empty_coordinates{.inputs = {*input}};
  EXPECT_FALSE(
      ComputeEmpiricalKernel(*executor_, **model, {&empty_coordinates, 1})
          .ok());
  for (const OutputCoordinate coordinate :
       {OutputCoordinate{1, 0}, OutputCoordinate{0, kRows * kWidth}}) {
    Sample sample{.inputs = {*input}, .coordinates = {coordinate}};
    EXPECT_FALSE(
        ComputeEmpiricalKernel(*executor_, **model, {&sample, 1}).ok());
  }
  Sample sample{.inputs = {*input}, .coordinates = {{0, 0}}};
  KernelOptions options;
  options.max_jacobian_bytes = 1;
  EXPECT_FALSE(
      ComputeEmpiricalKernel(*executor_, **model, {&sample, 1}, options).ok());
  auto other = cuda::Executor::Create();
  ASSERT_TRUE(other.ok()) << other.status();
  auto foreign = Upload(**other, std::vector<float>(kRows * kWidth, 0.5f));
  ASSERT_TRUE(foreign.ok()) << foreign.status();
  Sample foreign_sample{.inputs = {*foreign}, .coordinates = {{0, 0}}};
  EXPECT_FALSE(
      ComputeEmpiricalKernel(*executor_, **model, {&foreign_sample, 1}).ok());
}

TEST_F(EmpiricalNtkTest, RejectsNonFp32PhysicalOutputs) {
  auto model = FullyConnectedLayer::Create(*executor_, kWidth, DataType::BF16);
  ASSERT_TRUE(model.ok()) << model.status();
  ASSERT_TRUE((*model)->InitializeIdentity().ok());
  // Zero bytes encode BF16 zero without involving host compiler BF16 support.
  auto input = Upload(*executor_, std::vector<uint16_t>(kRows * kWidth, 0));
  ASSERT_TRUE(input.ok()) << input.status();
  Sample sample{.inputs = {*input}, .coordinates = {{0, 0}}};
  EXPECT_FALSE(ComputeEmpiricalKernel(*executor_, **model, {&sample, 1}).ok());
}

TEST_F(EmpiricalNtkTest,
       RejectsInconsistentWeightGradientAliasesBeforeForward) {
  auto first_weight = Upload(*executor_, std::vector<float>(kWidth, 2.0f));
  auto second_weight = Upload(*executor_, std::vector<float>(kWidth, 4.0f));
  auto first_gradient = Upload(*executor_, std::vector<float>(kWidth, 3.0f));
  auto second_gradient = Upload(*executor_, std::vector<float>(kWidth, 5.0f));
  auto input = Upload(*executor_, std::vector<float>(kRows * kWidth, 0.5f));
  ASSERT_TRUE(first_weight.ok()) << first_weight.status();
  ASSERT_TRUE(second_weight.ok()) << second_weight.status();
  ASSERT_TRUE(first_gradient.ok()) << first_gradient.status();
  ASSERT_TRUE(second_gradient.ok()) << second_gradient.status();
  ASSERT_TRUE(input.ok()) << input.status();
  Sample sample{.inputs = {*input}, .coordinates = {{0, 0}}};
  ProbeLayer mismatched_tie({*first_weight, *first_weight},
                            {*first_gradient, *second_gradient});
  EXPECT_FALSE(
      ComputeEmpiricalKernel(*executor_, mismatched_tie, {&sample, 1}).ok());
  EXPECT_EQ(mismatched_tie.forward_calls, 0);
  ProbeLayer shared_gradient({*first_weight, *second_weight},
                             {*first_gradient, *first_gradient});
  EXPECT_FALSE(
      ComputeEmpiricalKernel(*executor_, shared_gradient, {&sample, 1}).ok());
  EXPECT_EQ(shared_gradient.forward_calls, 0);
  ProbeLayer parameter_gradient_alias({*first_weight}, {*first_weight});
  EXPECT_FALSE(
      ComputeEmpiricalKernel(*executor_, parameter_gradient_alias, {&sample, 1})
          .ok());
  EXPECT_EQ(parameter_gradient_alias.forward_calls, 0);
  ProbeLayer no_parameters({}, {});
  EXPECT_FALSE(
      ComputeEmpiricalKernel(*executor_, no_parameters, {&sample, 1}).ok());
  EXPECT_EQ(no_parameters.forward_calls, 0);
}

}  // namespace
}  // namespace pluto::llm::ntk
