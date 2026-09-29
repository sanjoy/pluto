#include "src/llm/layers/rms_norm.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "src/cuda/page_locked_host_array.h"
#include "src/llm/layer_hooks.h"
#include "src/util/status_macros.h"

namespace pluto::llm {
namespace {

float Bf16(float value) { return static_cast<float>(__nv_bfloat16(value)); }

class RmsNormTest : public testing::Test {
 protected:
  void SetUp() override {
    auto executor = cuda::Executor::Create();
    ASSERT_TRUE(executor.ok()) << executor.status();
    executor_ = std::move(*executor);
  }

  void TearDown() override {
    if (executor_)
      EXPECT_TRUE(executor_->Synchronize().ok());
  }

  template <class T>
  absl::StatusOr<Buffer> Upload(const std::vector<T>& values) {
    ASSIGN_OR_RETURN(auto host,
                     cuda::PageLockedHostArray<T>::CopyFrom(*executor_, values));
    ASSIGN_OR_RETURN(Buffer result,
                     Buffer::Allocate(*executor_, values.size() * sizeof(T)));
    RETURN_IF_ERROR(cuda::CudaStatus(
        cudaMemcpyAsync(result.data(), host.data(), result.size_bytes(),
                        cudaMemcpyHostToDevice, executor_->stream()),
        "upload RMSNorm test"));
    RETURN_IF_ERROR(executor_->Synchronize());
    return result;
  }

  absl::StatusOr<Buffer> Input(const std::vector<float>& values) {
    std::vector<__nv_bfloat16> bf16;
    for (float value : values)
      bf16.emplace_back(value);
    return Upload(bf16);
  }

  std::vector<float> Read(const Buffer& buffer) {
    auto host = cuda::PageLockedHostArray<__nv_bfloat16>::Allocate(
        *executor_, buffer.size_bytes() / sizeof(__nv_bfloat16));
    EXPECT_TRUE(host.ok());
    if (!host.ok())
      return {};
    EXPECT_EQ(cudaMemcpyAsync(host->data(), buffer.data(), buffer.size_bytes(),
                              cudaMemcpyDeviceToHost, executor_->stream()),
              cudaSuccess);
    EXPECT_TRUE(executor_->Synchronize().ok());
    std::vector<float> result;
    for (auto value : *host)
      result.push_back(static_cast<float>(value));
    return result;
  }

  std::unique_ptr<cuda::Executor> executor_;
};

TEST_F(RmsNormTest, ZeroCenteredWeightAndBfloat16OutputSupportHooks) {
  auto weight = Upload<float>({0.f, 0.5f, -0.25f});
  auto input = Input({1.f, -2.f, 0.5f});
  ASSERT_TRUE(weight.ok());
  ASSERT_TRUE(input.ok());
  auto layer = RmsNormLayer::Create(*executor_, 3, *weight, 1e-6f);
  ASSERT_TRUE(layer.ok()) << layer.status();
  EXPECT_EQ((*layer)->input_types()[0],
            ActivationType(DataType::BF16, {-2, 1, 3}));
  EXPECT_EQ((*layer)->output_types()[0], (*layer)->input_types()[0]);
  EXPECT_EQ((*layer)->output_type(), DataType::BF16);
  ASSERT_EQ((*layer)->weights().size(), 1);
  EXPECT_EQ((*layer)->weights()[0].data(), weight->data());
  EXPECT_TRUE((*layer)->gradients().empty());
  LayerHooks hooks;
  int calls = 0;
  hooks.activation_hook = [&](cuda::Executor& executor, absl::string_view name,
                              absl::Span<const ActivationType> types,
                              absl::Span<Buffer> outputs) {
    EXPECT_EQ(&executor, executor_.get());
    EXPECT_EQ(name, "RmsNormLayer");
    EXPECT_EQ(types[0], (*layer)->output_types()[0]);
    EXPECT_EQ(outputs.size(), 1);
    ++calls;
    return absl::OkStatus();
  };
  auto result = (*layer)->fwd(*executor_, {*input}, &hooks);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(calls, 1);
  const float inverse = 1.f / std::sqrt(1.75f + 1e-6f);
  EXPECT_EQ(Read(result->outputs[0]),
            (std::vector<float>{Bf16(inverse), Bf16(-3.f * inverse),
                                Bf16(0.375f * inverse)}));
  EXPECT_TRUE(absl::IsUnimplemented(
      (*layer)->bwd(*executor_, {}, std::move(result->state)).status()));
}

TEST_F(RmsNormTest, ScalarReferenceMatchesAllReductionSizesAndMaskedTails) {
  for (int width : {1, 3, 256, 257, 5120, 8192, 8193, 16384}) {
    SCOPED_TRACE(width);
    std::vector<float> input(width), weights(width);
    double squares = 0;
    for (int i = 0; i < width; ++i) {
      input[i] = Bf16(std::sin(i * 0.7f + 0.2f));
      weights[i] = i % 7 == 0 ? -1.0f : 0.3f * std::cos(i * 0.13f);
      squares += static_cast<double>(input[i]) * input[i];
    }
    auto x = Input(input);
    auto w = Upload(weights);
    ASSERT_TRUE(x.ok());
    ASSERT_TRUE(w.ok());
    auto layer = RmsNormLayer::Create(*executor_, width, *w, 1e-6f);
    ASSERT_TRUE(layer.ok()) << layer.status();
    auto result = (*layer)->fwd(*executor_, {*x});
    ASSERT_TRUE(result.ok()) << result.status();
    const auto actual = Read(result->outputs[0]);
    ASSERT_EQ(actual.size(), input.size());
    const double inverse = 1.0 / std::sqrt(squares / width + 1e-6);
    for (int i = 0; i < width; ++i) {
      const float expected = Bf16(input[i] * inverse * (1.0 + weights[i]));
      // FP32 parallel reduction and scalar FP64 can land on opposite sides
      // of a BF16 midpoint; tolerate at most one BF16 spacing.
      EXPECT_NEAR(actual[i], expected, 1e-6f + 0.008f * std::abs(expected))
          << i;
      if (weights[i] == -1.f)
        EXPECT_EQ(actual[i], 0.f) << i;
    }
  }
}

TEST_F(RmsNormTest, ZeroInputUsesEpsilonAndStaysFinite) {
  auto weight = Upload<float>({0, 1, -1});
  auto input = Input({0, 0, 0});
  ASSERT_TRUE(weight.ok());
  ASSERT_TRUE(input.ok());
  auto layer = RmsNormLayer::Create(*executor_, 3, *weight, 1e-6f);
  ASSERT_TRUE(layer.ok());
  auto result = (*layer)->fwd(*executor_, {*input});
  ASSERT_TRUE(result.ok());
  EXPECT_EQ(Read(result->outputs[0]), (std::vector<float>{0, 0, 0}));
}

TEST_F(RmsNormTest, RejectsInvalidDimensionsWeightsEpsilonAndExecutor) {
  auto weight = Upload<float>({1, 2, 3});
  auto input = Input({1, 2, 3});
  auto short_input = Input({1, 2});
  ASSERT_TRUE(weight.ok());
  ASSERT_TRUE(input.ok());
  ASSERT_TRUE(short_input.ok());
  for (int width : {-1, 0, 2, 16385, std::numeric_limits<int>::max()})
    EXPECT_FALSE(RmsNormLayer::Create(*executor_, width, *weight, 1e-6f).ok());
  for (float epsilon : {0.f, -1.f, std::numeric_limits<float>::infinity(),
                        std::numeric_limits<float>::quiet_NaN()})
    EXPECT_FALSE(RmsNormLayer::Create(*executor_, 3, *weight, epsilon).ok());
  EXPECT_FALSE(RmsNormLayer::Create(*executor_, 3, *input, 1e-6f).ok());
  auto layer = RmsNormLayer::Create(*executor_, 3, *weight, 1e-6f);
  ASSERT_TRUE(layer.ok());
  EXPECT_FALSE((*layer)->fwd(*executor_, {}).ok());
  EXPECT_FALSE((*layer)->fwd(*executor_, {*input, *input}).ok());
  EXPECT_FALSE((*layer)->fwd(*executor_, {*short_input}).ok());
  EXPECT_FALSE((*layer)->fwd(*executor_, {*weight}).ok());
  auto other = cuda::Executor::Create();
  ASSERT_TRUE(other.ok());
  EXPECT_FALSE(RmsNormLayer::Create(**other, 3, *weight, 1e-6f).ok());
  EXPECT_FALSE((*layer)->fwd(**other, {*input}).ok());
  auto other_input = Buffer::Allocate(**other, input->size_bytes());
  ASSERT_TRUE(other_input.ok());
  EXPECT_FALSE((*layer)->fwd(*executor_, {*other_input}).ok());
  EXPECT_TRUE((*other)->Synchronize().ok());
}

}  // namespace
}  // namespace pluto::llm
