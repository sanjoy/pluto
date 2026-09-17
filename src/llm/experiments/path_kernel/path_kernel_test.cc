#include "src/llm/experiments/path_kernel/path_kernel.h"

#include <cuda_runtime_api.h>

#include <cmath>
#include <cstddef>
#include <limits>
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
#include "src/util/status_macros.h"

namespace pluto::llm::path_kernel {
namespace {

absl::Status Write(cuda::Executor& executor, const Buffer& destination,
                   const std::vector<float>& values) {
  if (destination.size_bytes() != values.size() * sizeof(float))
    return absl::InvalidArgumentError("path-kernel test upload size mismatch");
  ASSIGN_OR_RETURN(auto host, cuda::PageLockedHostArray<float>::CopyFrom(
                                  executor, absl::MakeConstSpan(values)));
  return cuda::CudaStatus(
      cudaMemcpyAsync(destination.data(), host.data(), host.size_bytes(),
                      cudaMemcpyHostToDevice, executor.stream()),
      "upload path-kernel test values");
}

absl::StatusOr<Buffer> Upload(cuda::Executor& executor,
                              const std::vector<float>& values) {
  ASSIGN_OR_RETURN(auto result,
                   Buffer::Allocate(executor, values.size() * sizeof(float)));
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
      "download path-kernel test values"));
  RETURN_IF_ERROR(executor.Synchronize());
  return std::vector<float>(host.begin(), host.end());
}

// Deliberately tiny, smooth models isolate path attribution from GPT-2's
// reduced-precision matrix multiplies. All tensors really are device buffers;
// only this oracle layer's scalar forward/backward arithmetic runs on the CPU.
// The production collector, Gram reductions, and GD updates still run on GPU.
//
// Affine:    f(x; w) = [x*w, -x*w].
// Quadratic: f(x; w) = [x*w*w, 0].
// Tied:      f(x; w) = [x*(w+w), -x*(w+w)], exposing the same parameter and
//            gradient buffers twice, just as embedding/head weight tying does.
class ScalarClassifier final : public Layer {
 public:
  enum class Form { kAffine, kQuadratic, kTied };

  static absl::StatusOr<std::unique_ptr<ScalarClassifier>> Create(
      cuda::Executor& executor, float initial_weight,
      Form form = Form::kAffine) {
    ASSIGN_OR_RETURN(auto weight, Upload(executor, {initial_weight}));
    // A nonzero preexisting gradient exposes any accidental leakage into the
    // optimizer and makes restoration distinguishable from simply clearing it.
    ASSIGN_OR_RETURN(auto gradient, Upload(executor, {7.25f}));
    return absl::WrapUnique(
        new ScalarClassifier(std::move(weight), std::move(gradient), form));
  }

  absl::string_view name() const override { return "ScalarClassifier"; }
  absl::Span<const ActivationType> input_types() const override {
    return {&input_type_, 1};
  }
  absl::Span<const ActivationType> output_types() const override {
    return {&output_type_, 1};
  }
  absl::Span<Buffer> weights() override { return absl::MakeSpan(weights_); }
  absl::Span<Buffer> gradients() override { return absl::MakeSpan(gradients_); }
  DataType output_type() const override { return DataType::FP32; }

  bool fail_backward = false;
  int backward_calls = 0;

 private:
  ScalarClassifier(Buffer weight, Buffer gradient, Form form)
      : weights_{weight}, gradients_{gradient}, form_(form) {
    if (form == Form::kTied) {
      weights_.push_back(weight);
      gradients_.push_back(gradient);
    }
  }

  absl::StatusOr<FwdResult> fwd_impl(cuda::Executor& executor,
                                     absl::Span<const Buffer> inputs,
                                     LayerHooks*) const override {
    if (inputs.size() != 1 || inputs[0].size_bytes() != sizeof(float))
      return absl::InvalidArgumentError("test classifier expects one scalar");
    ASSIGN_OR_RETURN(auto input, Read(executor, inputs[0]));
    ASSIGN_OR_RETURN(auto weight, Read(executor, weights_[0]));
    const double x = input[0];
    const double w = weight[0];
    double first, second, dw_first, dw_second, dx_first, dx_second;
    if (form_ == Form::kQuadratic) {
      first = x * w * w;
      second = 0.0;
      dw_first = 2.0 * x * w;
      dw_second = 0.0;
      dx_first = w * w;
      dx_second = 0.0;
    } else {
      const double multiplicity = form_ == Form::kTied ? 2.0 : 1.0;
      first = multiplicity * x * w;
      second = -first;
      dw_first = multiplicity * x;
      dw_second = -dw_first;
      dx_first = multiplicity * w;
      dx_second = -dx_first;
    }
    ASSIGN_OR_RETURN(auto output,
                     Upload(executor, {static_cast<float>(first),
                                       static_cast<float>(second)}));
    ASSIGN_OR_RETURN(
        auto derivatives,
        Upload(executor,
               {static_cast<float>(dw_first), static_cast<float>(dw_second),
                static_cast<float>(dx_first), static_cast<float>(dx_second)}));
    return FwdResult{.outputs = {output},
                     .state = {.intermediates = {derivatives}}};
  }

  absl::StatusOr<BufferVec> bwd_impl(cuda::Executor& executor,
                                     absl::Span<const Buffer> output_gradients,
                                     BackwardState state,
                                     LayerHooks*) override {
    ++backward_calls;
    ASSIGN_OR_RETURN(auto upstream, Read(executor, output_gradients[0]));
    ASSIGN_OR_RETURN(auto derivatives, Read(executor, state.intermediates[0]));
    ASSIGN_OR_RETURN(auto gradient, Read(executor, gradients_[0]));
    const double increment = static_cast<double>(upstream[0]) * derivatives[0] +
                             static_cast<double>(upstream[1]) * derivatives[1];
    RETURN_IF_ERROR(Write(
        executor, gradients_[0],
        {static_cast<float>(static_cast<double>(gradient[0]) + increment)}));
    // Failure happens after the accumulator was modified. An error-path test
    // must prove restoration after mutation, not merely preflight validation.
    if (fail_backward)
      return absl::DataLossError("injected classifier backward failure");
    ASSIGN_OR_RETURN(auto input_gradient,
                     Upload(executor, {upstream[0] * derivatives[2] +
                                       upstream[1] * derivatives[3]}));
    return BufferVec{input_gradient};
  }

  BufferVec weights_;
  BufferVec gradients_;
  Form form_;
  ActivationType input_type_{DataType::FP32, {-2, 1}};
  ActivationType output_type_{DataType::FP32, {-2, 2}};
};

class PathKernelTest : public testing::Test {
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

TEST_F(PathKernelTest, AffineFullVocabularyAttributionsHaveAnalyticSigns) {
  auto model = ScalarClassifier::Create(*executor_, 0.0f);
  auto first = Upload(*executor_, {1.0f});
  auto second = Upload(*executor_, {2.0f});
  auto query = Upload(*executor_, {3.0f});
  ASSERT_TRUE(model.ok()) << model.status();
  ASSERT_TRUE(first.ok()) << first.status();
  ASSERT_TRUE(second.ok()) << second.status();
  ASSERT_TRUE(query.ok()) << query.status();
  const std::vector<TrainingExample> training{
      {{*first}, CrossEntropy({0, 0}, 2, 0)},
      {{*second}, CrossEntropy({0, 0}, 2, 1)}};
  const std::vector<ntk::Sample> queries{{{*query}, {{0, 0}, {0, 1}}}};
  Options options;
  options.steps = 1;
  options.learning_rate = 0.125;
  auto result =
      path_kernel::Run(*executor_, **model, training, queries, options);
  ASSERT_TRUE(result.ok()) << result.status();
  ASSERT_EQ(result->steps.size(), 1);
  ASSERT_EQ(result->parameter_count, 1);
  ASSERT_EQ(result->parameters.size(), 1);
  EXPECT_EQ(result->parameters[0].elements, 1);

  // At w=0 the two loss gradients are -1 and +2. Their mean is +1/2,
  // hence w_next=-1/16. Query derivatives are +3 and -3. There is no Taylor
  // remainder for an affine model, even though softmax/CE itself is nonlinear.
  EXPECT_EQ(result->initial_values, (std::vector<double>{0.0, 0.0}));
  EXPECT_EQ(result->final_values, (std::vector<double>{-0.1875, 0.1875}));
  EXPECT_EQ(result->reconstructed_values, result->final_values);
  EXPECT_EQ(result->residual, (std::vector<double>{0.0, 0.0}));
  ASSERT_EQ(result->contributions.rows, 2);
  ASSERT_EQ(result->contributions.columns, 2);
  EXPECT_DOUBLE_EQ(result->contributions(0, 0), 0.1875);
  EXPECT_DOUBLE_EQ(result->contributions(0, 1), -0.375);
  EXPECT_DOUBLE_EQ(result->contributions(1, 0), -0.1875);
  EXPECT_DOUBLE_EQ(result->contributions(1, 1), 0.375);
  EXPECT_DOUBLE_EQ(result->path_kernel(0, 0), 1.125);
  EXPECT_DOUBLE_EQ(result->path_kernel(0, 1), -1.125);
  EXPECT_DOUBLE_EQ(result->path_kernel(1, 0), -1.125);
  EXPECT_DOUBLE_EQ(result->path_kernel(1, 1), 1.125);
  const StepResult& step = result->steps[0];
  EXPECT_EQ(step.values_before, result->initial_values);
  EXPECT_EQ(step.values_after, result->final_values);
  EXPECT_EQ(step.predicted_delta, result->final_values);
  EXPECT_EQ(step.residual, result->residual);
  EXPECT_EQ(step.contributions.values, result->contributions.values);
  ASSERT_EQ(step.training_losses.size(), 2);
  EXPECT_NEAR(step.training_losses[0], std::log(2.0), 1e-12);
  EXPECT_NEAR(step.training_losses[1], std::log(2.0), 1e-12);
  ASSERT_EQ(result->final_training_losses.size(), 2);
  EXPECT_NEAR(result->final_training_losses[0], std::log1p(std::exp(0.125)),
              1e-12);
  EXPECT_NEAR(result->final_training_losses[1], std::log1p(std::exp(-0.25)),
              1e-12);
  EXPECT_DOUBLE_EQ(step.tangent_kernel(0, 0), 9.0);
  EXPECT_DOUBLE_EQ(step.tangent_kernel(0, 1), -9.0);
  auto weight = Read(*executor_, (*model)->weights()[0]);
  auto gradient = Read(*executor_, (*model)->gradients()[0]);
  ASSERT_TRUE(weight.ok()) << weight.status();
  ASSERT_TRUE(gradient.ok()) << gradient.status();
  EXPECT_EQ(*weight, (std::vector<float>{-0.0625f}));
  EXPECT_EQ(*gradient, (std::vector<float>{7.25f}));
}

TEST_F(PathKernelTest, SelectingOneQueryLogitDoesNotTruncateTrainingSoftmax) {
  auto first = Upload(*executor_, {1.0f});
  auto second = Upload(*executor_, {2.0f});
  auto query = Upload(*executor_, {3.0f});
  ASSERT_TRUE(first.ok()) << first.status();
  ASSERT_TRUE(second.ok()) << second.status();
  ASSERT_TRUE(query.ok()) << query.status();
  const std::vector<TrainingExample> training{
      {{*first}, CrossEntropy({0, 0}, 2, 0)},
      {{*second}, CrossEntropy({0, 0}, 2, 1)}};
  Options options;
  options.steps = 2;
  options.learning_rate = 0.125;
  auto selected_model = ScalarClassifier::Create(*executor_, 0.0f);
  auto complete_model = ScalarClassifier::Create(*executor_, 0.0f);
  ASSERT_TRUE(selected_model.ok()) << selected_model.status();
  ASSERT_TRUE(complete_model.ok()) << complete_model.status();
  const std::vector<ntk::Sample> selected{{{*query}, {{0, 0}}}};
  const std::vector<ntk::Sample> complete{{{*query}, {{0, 0}, {0, 1}}}};
  auto one = path_kernel::Run(*executor_, **selected_model, training, selected,
                              options);
  auto both = path_kernel::Run(*executor_, **complete_model, training, complete,
                               options);
  ASSERT_TRUE(one.ok()) << one.status();
  ASSERT_TRUE(both.ok()) << both.status();
  EXPECT_DOUBLE_EQ(one->final_values[0], both->final_values[0]);
  EXPECT_DOUBLE_EQ(one->contributions(0, 0), both->contributions(0, 0));
  EXPECT_DOUBLE_EQ(one->contributions(0, 1), both->contributions(0, 1));
  EXPECT_DOUBLE_EQ(one->path_kernel(0, 0), both->path_kernel(0, 0));
  ASSERT_EQ(one->steps.size(), both->steps.size());
  for (size_t step = 0; step < one->steps.size(); ++step)
    EXPECT_EQ(one->steps[step].training_losses,
              both->steps[step].training_losses);
  // With a mistaken single-selected-class CE, both the probability and the
  // gradient would degenerate, rather than giving this nonzero first update.
  EXPECT_DOUBLE_EQ(one->steps[0].predicted_delta[0], -0.1875);
  auto one_weight = Read(*executor_, (*selected_model)->weights()[0]);
  auto both_weight = Read(*executor_, (*complete_model)->weights()[0]);
  ASSERT_TRUE(one_weight.ok()) << one_weight.status();
  ASSERT_TRUE(both_weight.ok()) << both_weight.status();
  EXPECT_EQ(*one_weight, *both_weight);
}

TEST_F(PathKernelTest, QuadraticFiniteStepResidualIsNotSilentlyDiscarded) {
  auto input = Upload(*executor_, {1.0f});
  auto query = Upload(*executor_, {2.0f});
  ASSERT_TRUE(input.ok()) << input.status();
  ASSERT_TRUE(query.ok()) << query.status();
  const std::vector<TrainingExample> training{
      {{*input}, CrossEntropy({0, 0}, 2, 1)}};
  const std::vector<ntk::Sample> queries{{{*query}, {{0, 0}}}};
  std::vector<double> residuals;
  for (double rate : {0.125, 0.0625}) {
    auto model = ScalarClassifier::Create(*executor_, 0.5f,
                                          ScalarClassifier::Form::kQuadratic);
    ASSERT_TRUE(model.ok()) << model.status();
    Options options;
    options.steps = 1;
    options.learning_rate = rate;
    auto result =
        path_kernel::Run(*executor_, **model, training, queries, options);
    ASSERT_TRUE(result.ok()) << result.status();
    // f_train=[1/4,0], label=1, so dL/dw=sigmoid(1/4). For the query,
    // f=2w² and df/dw=4w=2. Its exact real-arithmetic Taylor remainder is
    // 2*(delta_w)². The loose final few ulps below allow FP32 storage only;
    // the residual itself is several orders of magnitude larger.
    const double loss_gradient = 1.0 / (1.0 + std::exp(-0.25));
    const double delta = -rate * loss_gradient;
    const double predicted_delta = 2.0 * delta;
    const double expected_residual = 2.0 * delta * delta;
    EXPECT_NEAR(result->contributions(0, 0), predicted_delta, 2e-8);
    EXPECT_NEAR(result->reconstructed_values[0], 0.5 + predicted_delta, 2e-8);
    EXPECT_NEAR(result->final_values[0], 2.0 * std::pow(0.5 + delta, 2), 7e-8);
    EXPECT_NEAR(result->residual[0], expected_residual, 7e-8);
    EXPECT_GT(result->residual[0], 0.0);
    EXPECT_NEAR(result->final_values[0] - result->reconstructed_values[0],
                result->residual[0], 1e-15);
    residuals.push_back(result->residual[0]);
  }
  EXPECT_NEAR(residuals[0] / residuals[1], 4.0, 0.0002);
}

TEST_F(PathKernelTest, IntegratesChangingKernelAtActualPreUpdateWeights) {
  auto model = ScalarClassifier::Create(*executor_, 0.5f,
                                        ScalarClassifier::Form::kQuadratic);
  auto input = Upload(*executor_, {1.0f});
  auto query = Upload(*executor_, {2.0f});
  ASSERT_TRUE(model.ok()) << model.status();
  ASSERT_TRUE(input.ok()) << input.status();
  ASSERT_TRUE(query.ok()) << query.status();
  const std::vector<TrainingExample> training{
      {{*input}, CrossEntropy({0, 0}, 2, 0)}};
  const std::vector<ntk::Sample> queries{{{*query}, {{0, 0}}}};
  Options options;
  options.steps = 3;
  options.learning_rate = 0.125;
  auto result =
      path_kernel::Run(*executor_, **model, training, queries, options);
  ASSERT_TRUE(result.ok()) << result.status();
  ASSERT_EQ(result->steps.size(), 3);
  float weight = 0.5f;
  double expected_kernel = 0.0;
  double expected_contribution = 0.0;
  double reported_residual = 0.0;
  for (const StepResult& step : result->steps) {
    // This independent scalar trajectory rounds only at the same declared
    // storage boundaries as the fixture: logits, backward seeds/gradients,
    // and updated FP32 master parameters. The formulas do not use Run's
    // Jacobians, its reported loss, or any values from its GPU Gram matrix.
    const float logit =
        static_cast<float>(static_cast<double>(weight) * weight);
    const double probability =
        1.0 / (1.0 + std::exp(-static_cast<double>(logit)));
    const float seed = static_cast<float>(probability - 1.0);
    const float gradient = static_cast<float>(2.0 * weight * seed);
    const double query_derivative = 4.0 * weight;
    const double tangent = query_derivative * query_derivative;
    const double contribution =
        -options.learning_rate * query_derivative * gradient;
    EXPECT_NEAR(step.values_before[0], 2.0 * weight * weight, 1e-7);
    EXPECT_NEAR(step.tangent_kernel(0, 0), tangent, 1e-10);
    EXPECT_NEAR(step.contributions(0, 0), contribution, 1e-9);
    EXPECT_NEAR(step.predicted_delta[0], contribution, 1e-9);
    EXPECT_NEAR(step.training_losses[0],
                std::log1p(std::exp(-static_cast<double>(logit))), 1e-12);
    expected_kernel += options.learning_rate * tangent;
    expected_contribution += contribution;
    reported_residual += step.residual[0];
    weight = static_cast<float>(weight - options.learning_rate * gradient);
    EXPECT_NEAR(step.values_after[0], 2.0 * weight * weight, 1e-7);
  }
  EXPECT_GT(result->steps[2].tangent_kernel(0, 0),
            result->steps[0].tangent_kernel(0, 0));
  EXPECT_NEAR(result->path_kernel(0, 0), expected_kernel, 1e-10);
  EXPECT_NEAR(result->contributions(0, 0), expected_contribution, 1e-9);
  EXPECT_NEAR(result->reconstructed_values[0], 0.5 + expected_contribution,
              1e-9);
  EXPECT_NEAR(result->residual[0], reported_residual, 1e-15);
  EXPECT_NEAR(result->final_values[0] - result->initial_values[0],
              expected_contribution + reported_residual, 1e-9);
  auto actual_weight = Read(*executor_, (*model)->weights()[0]);
  ASSERT_TRUE(actual_weight.ok()) << actual_weight.status();
  EXPECT_FLOAT_EQ((*actual_weight)[0], weight);
}

TEST_F(PathKernelTest, ZeroStepsReturnsTheInitialModelAndZeroIntegrals) {
  auto model = ScalarClassifier::Create(*executor_, 0.25f);
  auto input = Upload(*executor_, {1.0f});
  auto query = Upload(*executor_, {3.0f});
  ASSERT_TRUE(model.ok()) << model.status();
  ASSERT_TRUE(input.ok()) << input.status();
  ASSERT_TRUE(query.ok()) << query.status();
  const std::vector<TrainingExample> training{
      {{*input}, CrossEntropy({0, 0}, 2, 0)}};
  const std::vector<ntk::Sample> queries{{{*query}, {{0, 0}, {0, 1}}}};
  Options options;
  options.steps = 0;
  options.progress = [](const StepResult&) {
    ADD_FAILURE() << "zero steps must not publish completed steps";
    return absl::OkStatus();
  };
  auto result =
      path_kernel::Run(*executor_, **model, training, queries, options);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_TRUE(result->steps.empty());
  EXPECT_EQ(result->initial_values, (std::vector<double>{0.75, -0.75}));
  EXPECT_EQ(result->initial_values, result->final_values);
  EXPECT_EQ(result->initial_values, result->reconstructed_values);
  EXPECT_EQ(result->residual, (std::vector<double>{0.0, 0.0}));
  EXPECT_EQ(result->path_kernel.rows, 2);
  EXPECT_EQ(result->path_kernel.columns, 2);
  EXPECT_EQ(result->path_kernel.values, std::vector<double>(4, 0.0));
  EXPECT_EQ(result->contributions.rows, 2);
  EXPECT_EQ(result->contributions.columns, 1);
  EXPECT_EQ(result->contributions.values, std::vector<double>(2, 0.0));
  auto weight = Read(*executor_, (*model)->weights()[0]);
  auto gradient = Read(*executor_, (*model)->gradients()[0]);
  ASSERT_TRUE(weight.ok()) << weight.status();
  ASSERT_TRUE(gradient.ok()) << gradient.status();
  EXPECT_EQ(*weight, (std::vector<float>{0.25f}));
  EXPECT_EQ(*gradient, (std::vector<float>{7.25f}));
}

TEST_F(PathKernelTest, SharedParameterContributesBothUsesButUpdatesOnlyOnce) {
  auto model =
      ScalarClassifier::Create(*executor_, 0.0f, ScalarClassifier::Form::kTied);
  auto input = Upload(*executor_, {1.0f});
  auto query = Upload(*executor_, {3.0f});
  ASSERT_TRUE(model.ok()) << model.status();
  ASSERT_TRUE(input.ok()) << input.status();
  ASSERT_TRUE(query.ok()) << query.status();
  ASSERT_EQ((*model)->weights().size(), 2);
  ASSERT_EQ((*model)->weights()[0].data(), (*model)->weights()[1].data());
  const std::vector<TrainingExample> training{
      {{*input}, CrossEntropy({0, 0}, 2, 0)}};
  const std::vector<ntk::Sample> queries{{{*query}, {{0, 0}}}};
  Options options;
  options.steps = 1;
  options.learning_rate = 0.125;
  auto result =
      path_kernel::Run(*executor_, **model, training, queries, options);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result->parameter_count, 1);
  ASSERT_EQ(result->parameters.size(), 1);
  EXPECT_DOUBLE_EQ(result->path_kernel(0, 0), 4.5);
  EXPECT_DOUBLE_EQ(result->contributions(0, 0), 1.5);
  EXPECT_EQ(result->final_values, (std::vector<double>{1.5}));
  EXPECT_EQ(result->residual, (std::vector<double>{0.0}));
  auto weight = Read(*executor_, (*model)->weights()[0]);
  auto gradient = Read(*executor_, (*model)->gradients()[0]);
  ASSERT_TRUE(weight.ok()) << weight.status();
  ASSERT_TRUE(gradient.ok()) << gradient.status();
  EXPECT_EQ(*weight, (std::vector<float>{0.25f}));
  EXPECT_EQ(*gradient, (std::vector<float>{7.25f}));
}

TEST_F(PathKernelTest, CancellationRollsBackCompletedUpdatesAndGradients) {
  auto model = ScalarClassifier::Create(*executor_, 0.0f);
  auto input = Upload(*executor_, {1.0f});
  ASSERT_TRUE(model.ok()) << model.status();
  ASSERT_TRUE(input.ok()) << input.status();
  const std::vector<TrainingExample> training{
      {{*input}, CrossEntropy({0, 0}, 2, 0)}};
  const std::vector<ntk::Sample> queries{{{*input}, {{0, 0}}}};
  Options options;
  options.steps = 3;
  options.learning_rate = 0.125;
  int callbacks = 0;
  options.progress = [&](const StepResult& step) -> absl::Status {
    ++callbacks;
    EXPECT_GT(step.values_after[0], 0.0);
    ASSIGN_OR_RETURN(auto weight, Read(*executor_, (*model)->weights()[0]));
    EXPECT_GT(weight[0], 0.0f);
    if (callbacks == 2)
      return absl::AbortedError("cancel after two actual updates");
    return absl::OkStatus();
  };
  auto result =
      path_kernel::Run(*executor_, **model, training, queries, options);
  EXPECT_EQ(result.status().code(), absl::StatusCode::kAborted);
  EXPECT_EQ(result.status().message(), "cancel after two actual updates");
  EXPECT_EQ(callbacks, 2);
  auto weight = Read(*executor_, (*model)->weights()[0]);
  auto gradient = Read(*executor_, (*model)->gradients()[0]);
  ASSERT_TRUE(weight.ok()) << weight.status();
  ASSERT_TRUE(gradient.ok()) << gradient.status();
  EXPECT_EQ(*weight, (std::vector<float>{0.0f}));
  EXPECT_EQ(*gradient, (std::vector<float>{7.25f}));
}

TEST_F(PathKernelTest, BackwardFailureRestoresAnAlreadyMutatedAccumulator) {
  auto model = ScalarClassifier::Create(*executor_, 0.0f);
  auto input = Upload(*executor_, {1.0f});
  ASSERT_TRUE(model.ok()) << model.status();
  ASSERT_TRUE(input.ok()) << input.status();
  (*model)->fail_backward = true;
  const std::vector<TrainingExample> training{
      {{*input}, CrossEntropy({0, 0}, 2, 0)}};
  const std::vector<ntk::Sample> queries{{{*input}, {{0, 0}}}};
  auto result = path_kernel::Run(*executor_, **model, training, queries);
  EXPECT_EQ(result.status().code(), absl::StatusCode::kDataLoss);
  EXPECT_EQ((*model)->backward_calls, 1);
  auto weight = Read(*executor_, (*model)->weights()[0]);
  auto gradient = Read(*executor_, (*model)->gradients()[0]);
  ASSERT_TRUE(weight.ok()) << weight.status();
  ASSERT_TRUE(gradient.ok()) << gradient.status();
  EXPECT_EQ(*weight, (std::vector<float>{0.0f}));
  EXPECT_EQ(*gradient, (std::vector<float>{7.25f}));
}

TEST_F(PathKernelTest, RejectsInvalidOptionsAndEmptyInputsWithoutMutation) {
  auto model = ScalarClassifier::Create(*executor_, 0.25f);
  auto input = Upload(*executor_, {1.0f});
  ASSERT_TRUE(model.ok()) << model.status();
  ASSERT_TRUE(input.ok()) << input.status();
  const std::vector<TrainingExample> training{
      {{*input}, CrossEntropy({0, 0}, 2, 0)}};
  const std::vector<ntk::Sample> queries{{{*input}, {{0, 0}}}};
  EXPECT_FALSE(path_kernel::Run(*executor_, **model, {}, queries).ok());
  EXPECT_FALSE(path_kernel::Run(*executor_, **model, training, {}).ok());
  Options options;
  options.steps = -1;
  EXPECT_FALSE(
      path_kernel::Run(*executor_, **model, training, queries, options).ok());
  options.steps = 1;
  for (double rate : {0.0, -0.125, std::numeric_limits<double>::infinity(),
                      std::numeric_limits<double>::quiet_NaN()}) {
    options.learning_rate = rate;
    EXPECT_FALSE(
        path_kernel::Run(*executor_, **model, training, queries, options).ok());
  }
  options.learning_rate = 0.125;
  options.max_jacobian_bytes = 1;
  EXPECT_EQ(path_kernel::Run(*executor_, **model, training, queries, options)
                .status()
                .code(),
            absl::StatusCode::kResourceExhausted);
  auto weight = Read(*executor_, (*model)->weights()[0]);
  auto gradient = Read(*executor_, (*model)->gradients()[0]);
  ASSERT_TRUE(weight.ok()) << weight.status();
  ASSERT_TRUE(gradient.ok()) << gradient.status();
  EXPECT_EQ(*weight, (std::vector<float>{0.25f}));
  EXPECT_EQ(*gradient, (std::vector<float>{7.25f}));
}

TEST_F(PathKernelTest, RejectsMalformedLossCoordinatesAndTargets) {
  auto model = ScalarClassifier::Create(*executor_, 0.25f);
  auto input = Upload(*executor_, {1.0f});
  ASSERT_TRUE(model.ok()) << model.status();
  ASSERT_TRUE(input.ok()) << input.status();
  const std::vector<ntk::Sample> queries{{{*input}, {{0, 0}}}};
  for (auto loss :
       {CrossEntropy({0, 0}, 0, 0), CrossEntropy({0, 0}, 2, 2),
        CrossEntropy({1, 0}, 2, 0), CrossEntropy({0, 1}, 2, 0),
        CrossEntropy({0, std::numeric_limits<size_t>::max()}, 2, 0)}) {
    const std::vector<TrainingExample> training{{{*input}, std::move(loss)}};
    EXPECT_FALSE(path_kernel::Run(*executor_, **model, training, queries).ok());
  }
  auto weight = Read(*executor_, (*model)->weights()[0]);
  auto gradient = Read(*executor_, (*model)->gradients()[0]);
  ASSERT_TRUE(weight.ok()) << weight.status();
  ASSERT_TRUE(gradient.ok()) << gradient.status();
  EXPECT_EQ(*weight, (std::vector<float>{0.25f}));
  EXPECT_EQ(*gradient, (std::vector<float>{7.25f}));
}

TEST_F(PathKernelTest, RepeatedRunsAreBitwiseReproducible) {
  auto input = Upload(*executor_, {0.75f});
  auto query = Upload(*executor_, {-1.25f});
  ASSERT_TRUE(input.ok()) << input.status();
  ASSERT_TRUE(query.ok()) << query.status();
  const std::vector<TrainingExample> training{
      {{*input}, CrossEntropy({0, 0}, 2, 1)}};
  const std::vector<ntk::Sample> queries{{{*query}, {{0, 0}, {0, 1}}}};
  Options options;
  options.steps = 4;
  options.learning_rate = 0.125;
  auto first_model = ScalarClassifier::Create(
      *executor_, 0.25f, ScalarClassifier::Form::kQuadratic);
  auto second_model = ScalarClassifier::Create(
      *executor_, 0.25f, ScalarClassifier::Form::kQuadratic);
  ASSERT_TRUE(first_model.ok()) << first_model.status();
  ASSERT_TRUE(second_model.ok()) << second_model.status();
  auto first =
      path_kernel::Run(*executor_, **first_model, training, queries, options);
  auto second =
      path_kernel::Run(*executor_, **second_model, training, queries, options);
  ASSERT_TRUE(first.ok()) << first.status();
  ASSERT_TRUE(second.ok()) << second.status();
  EXPECT_EQ(first->initial_values, second->initial_values);
  EXPECT_EQ(first->final_values, second->final_values);
  EXPECT_EQ(first->path_kernel.values, second->path_kernel.values);
  EXPECT_EQ(first->contributions.values, second->contributions.values);
  EXPECT_EQ(first->reconstructed_values, second->reconstructed_values);
  EXPECT_EQ(first->residual, second->residual);
  ASSERT_EQ(first->steps.size(), second->steps.size());
  for (size_t step = 0; step < first->steps.size(); ++step) {
    EXPECT_EQ(first->steps[step].tangent_kernel.values,
              second->steps[step].tangent_kernel.values);
    EXPECT_EQ(first->steps[step].training_losses,
              second->steps[step].training_losses);
    EXPECT_EQ(first->steps[step].predicted_delta,
              second->steps[step].predicted_delta);
    EXPECT_EQ(first->steps[step].residual, second->steps[step].residual);
  }
}

TEST_F(PathKernelTest, CrossEntropyExcludesPaddingOtherPositionsAndOutputs) {
  // A large padding logit makes an incorrect denominator immediately visible.
  // The selected row is offsets [1,4), with target class 1 (offset 2).
  auto logits = Upload(*executor_, {-9.0f, 1.0f, 2.0f, 3.0f, 1000.0f, -7.0f});
  auto auxiliary = Upload(*executor_, {1000.0f});
  ASSERT_TRUE(logits.ok()) << logits.status();
  ASSERT_TRUE(auxiliary.ok()) << auxiliary.status();
  const BufferVec outputs{*logits, *auxiliary};
  auto value = CrossEntropy({0, 1}, 3, 1)(*executor_, outputs);
  ASSERT_TRUE(value.ok()) << value.status();
  const double denominator = 1.0 + std::exp(-1.0) + std::exp(-2.0);
  EXPECT_NEAR(value->value, 1.0 + std::log(denominator), 1e-12);
  ASSERT_EQ(value->gradients.size(), 2);
  auto gradient = Read(*executor_, value->gradients[0]);
  auto other_gradient = Read(*executor_, value->gradients[1]);
  ASSERT_TRUE(gradient.ok()) << gradient.status();
  ASSERT_TRUE(other_gradient.ok()) << other_gradient.status();
  ASSERT_EQ(gradient->size(), 6);
  EXPECT_FLOAT_EQ((*gradient)[0], 0.0f);
  EXPECT_NEAR((*gradient)[1], std::exp(-2.0) / denominator, 3e-8);
  EXPECT_NEAR((*gradient)[2], std::exp(-1.0) / denominator - 1.0, 3e-8);
  EXPECT_NEAR((*gradient)[3], 1.0 / denominator, 3e-8);
  EXPECT_FLOAT_EQ((*gradient)[4], 0.0f);
  EXPECT_FLOAT_EQ((*gradient)[5], 0.0f);
  EXPECT_EQ(*other_gradient, (std::vector<float>{0.0f}));
}

TEST_F(PathKernelTest, CrossEntropyUsesStableLogSumExpAndRejectsInvalidValues) {
  auto logits = Upload(*executor_, {1000.0f, 999.0f});
  ASSERT_TRUE(logits.ok()) << logits.status();
  const BufferVec outputs{*logits};
  auto loss = CrossEntropy({0, 0}, 2, 1);
  auto value = loss(*executor_, outputs);
  ASSERT_TRUE(value.ok()) << value.status();
  EXPECT_NEAR(value->value, 1.0 + std::log1p(std::exp(-1.0)), 1e-12);
  auto gradient = Read(*executor_, value->gradients[0]);
  ASSERT_TRUE(gradient.ok()) << gradient.status();
  EXPECT_NEAR((*gradient)[0], 1.0 / (1.0 + std::exp(-1.0)), 3e-8);
  EXPECT_NEAR((*gradient)[1], -1.0 / (1.0 + std::exp(-1.0)), 3e-8);

  for (const auto& values :
       {std::vector<float>{std::numeric_limits<float>::quiet_NaN(), 0.0f},
        std::vector<float>{std::numeric_limits<float>::infinity(), 0.0f},
        std::vector<float>{0.0f, -std::numeric_limits<float>::infinity()},
        std::vector<float>{-std::numeric_limits<float>::infinity(),
                           -std::numeric_limits<float>::infinity()}}) {
    ASSERT_TRUE(Write(*executor_, *logits, values).ok());
    EXPECT_FALSE(loss(*executor_, outputs).ok());
  }
}

}  // namespace
}  // namespace pluto::llm::path_kernel
