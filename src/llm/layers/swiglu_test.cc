#include "src/llm/layers/swiglu.h"

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

class SwiGluTest : public testing::Test {
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

  absl::StatusOr<Buffer> Input(const std::vector<float>& values) {
    ASSIGN_OR_RETURN(auto host,
                     cuda::PageLockedHostArray<__nv_bfloat16>::Allocate(
                         *executor_, values.size()));
    for (size_t i = 0; i < values.size(); ++i)
      host[i] = __nv_bfloat16(values[i]);
    ASSIGN_OR_RETURN(Buffer result,
                     Buffer::Allocate(*executor_, host.size_bytes()));
    RETURN_IF_ERROR(cuda::CudaStatus(
        cudaMemcpyAsync(result.data(), host.data(), result.size_bytes(),
                        cudaMemcpyHostToDevice, executor_->stream()),
        "upload SwiGLU test"));
    RETURN_IF_ERROR(executor_->Synchronize());
    return result;
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

TEST_F(SwiGluTest, RoundsSiluBeforeMultiplicationAndSupportsHooks) {
  auto gate = Input({-3.f, -0.5f, 1.5f});
  auto up = Input({2.f, -4.f, 0.25f});
  ASSERT_TRUE(gate.ok());
  ASSERT_TRUE(up.ok());
  auto layer = SwiGluLayer::Create(3);
  ASSERT_TRUE(layer.ok());
  EXPECT_TRUE((*layer)->weights().empty());
  EXPECT_TRUE((*layer)->gradients().empty());
  EXPECT_EQ((*layer)->output_type(), DataType::BF16);
  ASSERT_EQ((*layer)->input_types().size(), 2);
  EXPECT_EQ((*layer)->input_types()[0],
            ActivationType(DataType::BF16, {-2, 1, 3}));
  EXPECT_EQ((*layer)->input_types()[1], (*layer)->input_types()[0]);
  LayerHooks hooks;
  int calls = 0;
  hooks.activation_hook = [&](cuda::Executor& executor, absl::string_view name,
                              absl::Span<const ActivationType> types,
                              absl::Span<Buffer> outputs) {
    EXPECT_EQ(&executor, executor_.get());
    EXPECT_EQ(name, "SwiGluLayer");
    EXPECT_EQ(types[0], (*layer)->input_types()[0]);
    EXPECT_EQ(outputs.size(), 1);
    ++calls;
    return absl::OkStatus();
  };
  auto result = (*layer)->fwd(*executor_, {*gate, *up}, &hooks);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(calls, 1);
  EXPECT_EQ(
      Read(result->outputs[0]),
      (std::vector<float>{Bf16(Bf16(-3.f / (1 + std::exp(3.f))) * 2),
                          Bf16(Bf16(-0.5f / (1 + std::exp(0.5f))) * -4),
                          Bf16(Bf16(1.5f / (1 + std::exp(-1.5f))) * 0.25f)}));
  EXPECT_TRUE(absl::IsUnimplemented(
      (*layer)->bwd(*executor_, {}, std::move(result->state)).status()));
}

TEST_F(SwiGluTest, ScalarReferenceMatchesMaskedTailsAndQwenExpansion) {
  for (int width : {1, 7, 256, 257, 17408}) {
    SCOPED_TRACE(width);
    std::vector<float> gate(width), up(width);
    for (int i = 0; i < width; ++i) {
      gate[i] = Bf16(std::sin(i * 0.31f) * 4.0f);
      up[i] = Bf16(std::cos(i * 0.29f));
    }
    auto g = Input(gate);
    auto u = Input(up);
    ASSERT_TRUE(g.ok());
    ASSERT_TRUE(u.ok());
    auto layer = SwiGluLayer::Create(width);
    ASSERT_TRUE(layer.ok());
    auto result = (*layer)->fwd(*executor_, {*g, *u});
    ASSERT_TRUE(result.ok()) << result.status();
    const auto actual = Read(result->outputs[0]);
    ASSERT_EQ(actual.size(), gate.size());
    for (int i = 0; i < width; ++i) {
      const float silu = Bf16(gate[i] / (1.0f + std::exp(-gate[i])));
      const float expected = Bf16(silu * up[i]);
      EXPECT_NEAR(actual[i], expected, 1e-6f + 0.008f * std::abs(expected))
          << i;
    }
  }
}

TEST_F(SwiGluTest, RejectsInvalidDimensionsInputCountSizeAndExecutor) {
  for (int width : {-1, 0, 1048577, std::numeric_limits<int>::max()})
    EXPECT_FALSE(SwiGluLayer::Create(width).ok());
  EXPECT_TRUE(SwiGluLayer::Create(1048576).ok());
  auto layer = SwiGluLayer::Create(3);
  auto input = Input({1, 2, 3});
  auto wrong_size = Input({1, 2});
  ASSERT_TRUE(layer.ok());
  ASSERT_TRUE(input.ok());
  ASSERT_TRUE(wrong_size.ok());
  EXPECT_FALSE((*layer)->fwd(*executor_, {}).ok());
  EXPECT_FALSE((*layer)->fwd(*executor_, {*input}).ok());
  EXPECT_FALSE((*layer)->fwd(*executor_, {*input, *input, *input}).ok());
  EXPECT_FALSE((*layer)->fwd(*executor_, {*input, *wrong_size}).ok());
  EXPECT_FALSE((*layer)->fwd(*executor_, {*wrong_size, *input}).ok());
  auto other = cuda::Executor::Create();
  ASSERT_TRUE(other.ok());
  EXPECT_FALSE((*layer)->fwd(**other, {*input, *input}).ok());
  auto other_input = Buffer::Allocate(**other, input->size_bytes());
  ASSERT_TRUE(other_input.ok());
  EXPECT_FALSE((*layer)->fwd(*executor_, {*input, *other_input}).ok());
  EXPECT_TRUE((*other)->Synchronize().ok());
}

}  // namespace
}  // namespace pluto::llm
