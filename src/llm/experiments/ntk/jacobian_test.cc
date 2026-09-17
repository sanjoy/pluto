#include <cuda_runtime_api.h>

#include <cstddef>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "gtest/gtest.h"
#include "src/cuda/page_locked_host_array.h"
#include "src/llm/experiments/ntk/empirical_ntk.h"
#include "src/llm/layers/fully_connected.h"
#include "src/util/status_macros.h"

namespace pluto::llm::ntk {
namespace {

constexpr size_t kWidth = 16;
constexpr size_t kElements = kWidth * kWidth;

absl::Status Write(cuda::Executor& executor, const Buffer& destination,
                   absl::Span<const float> values) {
  if (destination.size_bytes() != values.size() * sizeof(float))
    return absl::InvalidArgumentError("test upload size mismatch");
  ASSIGN_OR_RETURN(auto host,
                   cuda::PageLockedHostArray<float>::CopyFrom(executor, values));
  return cuda::CudaStatus(
      cudaMemcpyAsync(destination.data(), host.data(), host.size_bytes(),
                      cudaMemcpyHostToDevice, executor.stream()),
      "write Jacobian test values");
}

absl::StatusOr<Buffer> Upload(cuda::Executor& executor,
                              absl::Span<const float> values) {
  ASSIGN_OR_RETURN(auto buffer,
                   Buffer::Allocate(executor, values.size() * sizeof(float)));
  RETURN_IF_ERROR(Write(executor, buffer, values));
  return buffer;
}

absl::StatusOr<std::vector<float>> Read(cuda::Executor& executor,
                                        const Buffer& source) {
  ASSIGN_OR_RETURN(auto host,
                   cuda::PageLockedHostArray<float>::Allocate(
                       executor, source.size_bytes() / sizeof(float)));
  RETURN_IF_ERROR(cuda::CudaStatus(
      cudaMemcpyAsync(host.data(), source.data(), source.size_bytes(),
                      cudaMemcpyDeviceToHost, executor.stream()),
      "read Jacobian test values"));
  RETURN_IF_ERROR(executor.Synchronize());
  return std::vector<float>(host.begin(), host.end());
}

class JacobianTest : public testing::Test {
 protected:
  void SetUp() override {
    auto executor = cuda::Executor::Create();
    ASSERT_TRUE(executor.ok()) << executor.status();
    executor_ = std::move(*executor);
    auto model =
        FullyConnectedLayer::Create(*executor_, kWidth, DataType::FP16);
    ASSERT_TRUE(model.ok()) << model.status();
    model_ = std::move(*model);
    ASSERT_TRUE(model_->InitializeIdentity(0.5f).ok());
    for (const Buffer& gradient : model_->gradients()) {
      std::vector<float> values(gradient.size_bytes() / sizeof(float), 7);
      ASSERT_TRUE(Write(*executor_, gradient, values).ok());
    }
  }

  void ExpectGradientsRestored() {
    for (const Buffer& gradient : model_->gradients()) {
      auto values = Read(*executor_, gradient);
      ASSERT_TRUE(values.ok()) << values.status();
      EXPECT_EQ(*values, std::vector<float>(values->size(), 7));
    }
  }

  // Declared before model_ so all buffers are destroyed before their executor.
  std::unique_ptr<cuda::Executor> executor_;
  std::unique_ptr<FullyConnectedLayer> model_;
};

TEST_F(JacobianTest, ArbitraryScalarAndCoordinateRowsHaveAnalyticDerivatives) {
  std::vector<float> x(kElements, 0);
  x[0] = 2;
  x[1] = -1;
  auto input = Upload(*executor_, x);
  ASSERT_TRUE(input.ok()) << input.status();
  // L = 1/2 ||f||^2 uses the entire output, not a selected-logit Jacobian.
  // dL/df = f; this callback can share the read-only output as its seed.
  const ScalarFunction squared_norm =
      [](cuda::Executor& executor,
         absl::Span<const Buffer> outputs) -> absl::StatusOr<ScalarOutput> {
    ASSIGN_OR_RETURN(auto values, Read(executor, outputs[0]));
    double loss = 0;
    for (float value : values)
      loss += 0.5 * value * value;
    return ScalarOutput{loss, {outputs[0]}};
  };
  DifferentiationSample sample{
      .inputs = {*input},
      .outputs = {MakeOutputCoordinate({0, 0}), squared_norm}};
  size_t completed = 0;
  KernelOptions options;
  options.progress = [&](size_t row, size_t total) {
    EXPECT_EQ(row, ++completed);
    EXPECT_EQ(total, 2);
    return absl::OkStatus();
  };
  auto result = ComputeJacobian(*executor_, *model_, {&sample, 1}, options);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result->values, (std::vector<double>{1, 0.625}));
  EXPECT_EQ(result->parameter_count, kElements + kWidth);
  ASSERT_EQ(result->parameters.size(), 2);
  EXPECT_EQ(result->parameters[0].offset, 0);
  EXPECT_EQ(result->parameters[1].offset, kElements);
  EXPECT_EQ(completed, 2);
  auto derivatives = Read(*executor_, result->derivatives);
  ASSERT_TRUE(derivatives.ok()) << derivatives.status();
  std::vector<float> expected(2 * result->parameter_count, 0);
  // f_0 = x W[:,0] + b_0. The matrix layout is [input, output].
  expected[0] = 2;
  expected[kWidth] = -1;
  expected[kElements] = 1;
  const size_t second = result->parameter_count;
  expected[second] = 2;
  expected[second + 1] = -1;
  expected[second + kWidth] = -1;
  expected[second + kWidth + 1] = 0.5;
  expected[second + kElements] = 1;
  expected[second + kElements + 1] = -0.5;
  EXPECT_EQ(*derivatives, expected);
  ExpectGradientsRestored();
  // Jacobian collection must never update model parameters.
  auto forward = model_->fwd(*executor_, {*input});
  ASSERT_TRUE(forward.ok()) << forward.status();
  auto outputs = Read(*executor_, forward->outputs[0]);
  ASSERT_TRUE(outputs.ok()) << outputs.status();
  EXPECT_EQ((*outputs)[0], 1);
  EXPECT_EQ((*outputs)[1], -0.5);
}

TEST_F(JacobianTest, RejectsMalformedCallbacksAndRestoresClearedGradients) {
  auto input = Upload(*executor_, std::vector<float>(kElements, 0.5));
  auto small_seed = Upload(*executor_, std::vector<float>(1, 1));
  auto other = cuda::Executor::Create();
  ASSERT_TRUE(input.ok()) << input.status();
  ASSERT_TRUE(small_seed.ok()) << small_seed.status();
  ASSERT_TRUE(other.ok()) << other.status();
  auto foreign_seed = Upload(**other, std::vector<float>(kElements, 1));
  ASSERT_TRUE(foreign_seed.ok()) << foreign_seed.status();
  const std::vector<ScalarFunction> callbacks{
      // Empty callable must fail preflight, not throw std::bad_function_call.
      {},
      [](cuda::Executor&,
         absl::Span<const Buffer>) -> absl::StatusOr<ScalarOutput> {
        return absl::AbortedError("injected scalar callback failure");
      },
      [](cuda::Executor&,
         absl::Span<const Buffer> outputs) -> absl::StatusOr<ScalarOutput> {
        return ScalarOutput{std::numeric_limits<double>::infinity(),
                            {outputs[0]}};
      },
      [](cuda::Executor&, absl::Span<const Buffer>)
          -> absl::StatusOr<ScalarOutput> { return ScalarOutput{0, {}}; },
      [](cuda::Executor&,
         absl::Span<const Buffer> outputs) -> absl::StatusOr<ScalarOutput> {
        return ScalarOutput{0, {outputs[0], outputs[0]}};
      },
      [seed = *small_seed](cuda::Executor&, absl::Span<const Buffer>)
          -> absl::StatusOr<ScalarOutput> { return ScalarOutput{0, {seed}}; },
      [seed = *foreign_seed](cuda::Executor&, absl::Span<const Buffer>)
          -> absl::StatusOr<ScalarOutput> { return ScalarOutput{0, {seed}}; }};
  for (size_t index = 0; index < callbacks.size(); ++index) {
    SCOPED_TRACE(index);
    DifferentiationSample sample{.inputs = {*input},
                                 .outputs = {callbacks[index]}};
    auto result = ComputeJacobian(*executor_, *model_, {&sample, 1});
    EXPECT_FALSE(result.ok());
    if (index == 1)
      EXPECT_EQ(result.status().code(), absl::StatusCode::kAborted);
    if (index == 2)
      EXPECT_EQ(result.status().code(), absl::StatusCode::kFailedPrecondition);
    ExpectGradientsRestored();
  }
}

TEST_F(JacobianTest, CancellationAfterCompletedRowRestoresGradients) {
  auto input = Upload(*executor_, std::vector<float>(kElements, 0.5));
  ASSERT_TRUE(input.ok()) << input.status();
  DifferentiationSample sample{
      .inputs = {*input},
      .outputs = {MakeOutputCoordinate({0, 0}), MakeOutputCoordinate({0, 1})}};
  KernelOptions options;
  options.progress = [](size_t row, size_t total) {
    EXPECT_EQ(row, 1);
    EXPECT_EQ(total, 2);
    return absl::CancelledError("stop after first derivative");
  };
  auto result = ComputeJacobian(*executor_, *model_, {&sample, 1}, options);
  EXPECT_EQ(result.status().code(), absl::StatusCode::kCancelled);
  ExpectGradientsRestored();
}

TEST_F(JacobianTest, CoordinateCallbackSeedsExactlyOneComponentAcrossOutputs) {
  auto first = Upload(*executor_, std::vector<float>{1, 2});
  auto second = Upload(*executor_, std::vector<float>{3, 4, 5});
  ASSERT_TRUE(first.ok()) << first.status();
  ASSERT_TRUE(second.ok()) << second.status();
  auto callback = MakeOutputCoordinate({1, 1});
  auto result = callback(*executor_, {*first, *second});
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result->value, 4);
  ASSERT_EQ(result->gradients.size(), 2);
  auto first_seed = Read(*executor_, result->gradients[0]);
  auto second_seed = Read(*executor_, result->gradients[1]);
  ASSERT_TRUE(first_seed.ok()) << first_seed.status();
  ASSERT_TRUE(second_seed.ok()) << second_seed.status();
  EXPECT_EQ(*first_seed, (std::vector<float>{0, 0}));
  EXPECT_EQ(*second_seed, (std::vector<float>{0, 1, 0}));
  EXPECT_FALSE(
      MakeOutputCoordinate({2, 0})(*executor_, {*first, *second}).ok());
  EXPECT_FALSE(
      MakeOutputCoordinate({1, 3})(*executor_, {*first, *second}).ok());
}

TEST_F(JacobianTest, BudgetAndArityFailBeforeCallbacksRun) {
  auto input = Upload(*executor_, std::vector<float>(kElements, 0.5));
  ASSERT_TRUE(input.ok()) << input.status();
  bool called = false;
  ScalarFunction callback =
      [&](cuda::Executor&,
          absl::Span<const Buffer>) -> absl::StatusOr<ScalarOutput> {
    called = true;
    return absl::InternalError("should not execute");
  };
  DifferentiationSample sample{.inputs = {*input}, .outputs = {callback}};
  KernelOptions options;
  options.max_jacobian_bytes = 1;
  EXPECT_EQ(ComputeJacobian(*executor_, *model_, {&sample, 1}, options)
                .status()
                .code(),
            absl::StatusCode::kResourceExhausted);
  sample.inputs.clear();
  EXPECT_EQ(ComputeJacobian(*executor_, *model_, {&sample, 1}).status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_FALSE(called);
  ExpectGradientsRestored();
}

}  // namespace
}  // namespace pluto::llm::ntk
