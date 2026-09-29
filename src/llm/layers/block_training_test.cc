#include "src/llm/layers/block_training.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cmath>
#include <memory>
#include <vector>

#include "gtest/gtest.h"
#include "src/cuda/page_locked_host_array.h"
#include "src/util/status_macros.h"

namespace pluto::llm {
namespace {
float Bf16(float value) { return static_cast<float>(__nv_bfloat16(value)); }

class BlockTrainingTest : public testing::Test {
 protected:
  void SetUp() override {
    auto executor = cuda::Executor::Create();
    ASSERT_TRUE(executor.ok()) << executor.status();
    executor_ = std::move(*executor);
  }
  void TearDown() override { EXPECT_TRUE(executor_->Synchronize().ok()); }

  template <class T>
  absl::StatusOr<Buffer> Upload(const std::vector<T>& values) {
    ASSIGN_OR_RETURN(auto host,
                     cuda::PageLockedHostArray<T>::CopyFrom(*executor_, values));
    ASSIGN_OR_RETURN(auto device,
                     Buffer::Allocate(*executor_, values.size() * sizeof(T)));
    RETURN_IF_ERROR(cuda::CudaStatus(
        cudaMemcpyAsync(device.data(), host.data(), device.size_bytes(),
                        cudaMemcpyHostToDevice, executor_->stream()),
        "upload test values"));
    return device;
  }
  absl::StatusOr<Buffer> Input(const std::vector<float>& values) {
    std::vector<__nv_bfloat16> converted;
    for (float value : values)
      converted.emplace_back(value);
    return Upload(converted);
  }
  template <class T = float>
  std::vector<float> Read(const Buffer& buffer) {
    auto host = cuda::PageLockedHostArray<T>::Allocate(
        *executor_, buffer.size_bytes() / sizeof(T));
    EXPECT_TRUE(host.ok()) << host.status();
    if (!host.ok())
      return {};
    EXPECT_EQ(cudaMemcpyAsync(host->data(), buffer.data(), buffer.size_bytes(),
                              cudaMemcpyDeviceToHost, executor_->stream()),
              cudaSuccess);
    EXPECT_TRUE(executor_->Synchronize().ok());
    std::vector<float> result;
    for (T value : *host)
      result.push_back(static_cast<float>(value));
    return result;
  }
  absl::StatusOr<std::shared_ptr<BlockParameter>> Parameter(
      std::vector<float> values, DataType type = DataType::BF16) {
    ASSIGN_OR_RETURN(auto value,
                     type == DataType::BF16 ? Input(values) : Upload(values));
    return BlockParameter::Create(*executor_, std::move(value), type);
  }
  void Near(const std::vector<float>& actual,
            const std::vector<float>& expected, float tolerance = 1e-5) {
    ASSERT_EQ(actual.size(), expected.size());
    for (size_t i = 0; i < actual.size(); ++i)
      EXPECT_NEAR(actual[i], expected[i], tolerance) << "element " << i;
  }
  std::unique_ptr<cuda::Executor> executor_;
};

TEST_F(BlockTrainingTest,
       ActivationPublishingAndDeactivationKeepResidentAddress) {
  for (DataType type : {DataType::BF16, DataType::FP32}) {
    auto parameter = Parameter({1.25, -2.5, 0.75}, type);
    ASSERT_TRUE(parameter.ok()) << parameter.status();
    auto& p = **parameter;
    void* address = p.value().data();
    EXPECT_FALSE(p.active());
    ASSERT_TRUE(p.Activate().ok());
    Near(Read(p.master()), {1.25, -2.5, 0.75});
    Near(Read(p.gradient()), {0, 0, 0});
    void* master = p.master().data();
    ASSERT_TRUE(p.Activate().ok());
    EXPECT_EQ(master, p.master().data());
    auto updated = Upload<float>({1.12345, -0.12839, 5.98765});
    ASSERT_TRUE(updated.ok());
    ASSERT_EQ(cudaMemcpyAsync(p.master().data(), updated->data(),
                              updated->size_bytes(), cudaMemcpyDeviceToDevice,
                              executor_->stream()),
              cudaSuccess);
    ASSERT_TRUE(p.Publish().ok());
    ASSERT_TRUE(p.Deactivate().ok());
    EXPECT_FALSE(p.active());
    EXPECT_EQ(address, p.value().data());
    auto values = type == DataType::BF16 ? Read<__nv_bfloat16>(p.value())
                                         : Read(p.value());
    std::vector<float> expected{1.12345, -0.12839, 5.98765};
    if (type == DataType::BF16)
      for (float& value : expected)
        value = Bf16(value);
    Near(values, expected);
    ASSERT_TRUE(p.Deactivate().ok());
    ASSERT_TRUE(p.Activate().ok());
    Near(Read(p.master()), expected);
    Near(Read(p.gradient()), {0, 0, 0});
  }
}

TEST_F(BlockTrainingTest, LinearMatchesCpuForwardAndBothGradients) {
  const std::vector<float> x{1, 2, -3, 4, -2, 1};
  const std::vector<float> w{0.25, -1, 2, -0.5, 0.75, 1};
  const std::vector<float> dy{2, -1, -0.5, 3};
  auto weight = Parameter(w);
  auto input = Input(x);
  auto gradient = Upload(dy);
  ASSERT_TRUE(weight.ok());
  ASSERT_TRUE(input.ok());
  ASSERT_TRUE(gradient.ok());
  for (bool float_output : {false, true}) {
    auto layer =
        BlockLinearLayer::Create(*executor_, *weight, 3, 2, 2, float_output);
    ASSERT_TRUE(layer.ok()) << layer.status();
    EXPECT_EQ((*layer)->output_types()[0].data_type(),
              float_output ? DataType::FP32 : DataType::BF16);
    auto result = (*layer)->fwd(*executor_, {*input});
    ASSERT_TRUE(result.ok()) << result.status();
    std::vector<float> y(4), dx(6), dw(6);
    for (int t = 0; t < 2; ++t)
      for (int o = 0; o < 2; ++o)
        for (int i = 0; i < 3; ++i) {
          y[t * 2 + o] += w[o * 3 + i] * x[t * 3 + i];
          dx[t * 3 + i] += w[o * 3 + i] * dy[t * 2 + o];
          dw[o * 3 + i] += dy[t * 2 + o] * x[t * 3 + i];
        }
    if (!float_output)
      for (float& value : y)
        value = Bf16(value);
    Near(float_output ? Read(result->outputs[0])
                      : Read<__nv_bfloat16>(result->outputs[0]),
         y);
    // A frozen block propagates dx without acquiring optimizer storage.
    auto backward = (*layer)->bwd(*executor_, {*gradient}, result->state);
    ASSERT_TRUE(backward.ok()) << backward.status();
    Near(Read((*backward)[0]), dx);
    EXPECT_FALSE((*weight)->active());
    ASSERT_TRUE((*weight)->Activate().ok());
    for (int accumulated = 1; accumulated <= 2; ++accumulated) {
      backward = (*layer)->bwd(*executor_, {*gradient}, result->state);
      ASSERT_TRUE(backward.ok()) << backward.status();
      Near(Read((*backward)[0]), dx);
      auto expected = dw;
      for (float& value : expected)
        value *= accumulated;
      Near(Read((*weight)->gradient()), expected);
    }
    ASSERT_TRUE((*weight)->Deactivate().ok());
  }
}

TEST_F(BlockTrainingTest, EmbeddingAccumulatesRepeatedTokensWithoutAtomics) {
  auto weight = Parameter({1, 2, 3, 4, 5, 6});
  auto ids = Upload<int32_t>({1, 0, 1});
  auto dy = Upload<float>({1, 2, 3, 4, 5, 6});
  ASSERT_TRUE(weight.ok());
  ASSERT_TRUE(ids.ok());
  ASSERT_TRUE(dy.ok());
  auto layer = BlockEmbeddingLayer::Create(*executor_, *weight, 3, 2, 3);
  ASSERT_TRUE(layer.ok()) << layer.status();
  auto result = (*layer)->fwd(*executor_, {*ids});
  ASSERT_TRUE(result.ok()) << result.status();
  Near(Read<__nv_bfloat16>(result->outputs[0]), {3, 4, 1, 2, 3, 4});
  auto backward = (*layer)->bwd(*executor_, {*dy}, result->state);
  ASSERT_TRUE(backward.ok());
  EXPECT_TRUE(backward->empty());
  EXPECT_FALSE((*weight)->active());
  ASSERT_TRUE((*weight)->Activate().ok());
  backward = (*layer)->bwd(*executor_, {*dy}, result->state);
  ASSERT_TRUE(backward.ok()) << backward.status();
  Near(Read((*weight)->gradient()), {3, 4, 6, 8, 0, 0});
}

TEST_F(BlockTrainingTest, LinearMasksTailTilesAndProducesRepeatableGradients) {
  constexpr int cols = 131, rows = 35, tokens = 3;
  std::vector<float> x(tokens * cols), w(rows * cols), dy(tokens * rows);
  for (size_t i = 0; i < x.size(); ++i)
    x[i] = (static_cast<int>(i % 13) - 6) * 0.125f;
  for (size_t i = 0; i < w.size(); ++i)
    w[i] = (static_cast<int>(i % 7) - 3) * 0.125f;
  for (size_t i = 0; i < dy.size(); ++i)
    dy[i] = (static_cast<int>(i % 5) - 2) * 0.0625f;
  auto input = Input(x);
  auto weight = Parameter(w);
  auto grad = Upload(dy);
  ASSERT_TRUE(input.ok());
  ASSERT_TRUE(weight.ok());
  ASSERT_TRUE(grad.ok());
  auto layer =
      BlockLinearLayer::Create(*executor_, *weight, cols, rows, tokens);
  ASSERT_TRUE(layer.ok());
  auto result = (*layer)->fwd(*executor_, {*input});
  ASSERT_TRUE(result.ok()) << result.status();
  std::vector<float> y(tokens * rows), dx(tokens * cols), dw(rows * cols);
  for (int t = 0; t < tokens; ++t)
    for (int o = 0; o < rows; ++o)
      for (int i = 0; i < cols; ++i) {
        y[t * rows + o] += w[o * cols + i] * x[t * cols + i];
        dx[t * cols + i] += w[o * cols + i] * dy[t * rows + o];
        dw[o * cols + i] += x[t * cols + i] * dy[t * rows + o];
      }
  for (float& value : y)
    value = Bf16(value);
  Near(Read<__nv_bfloat16>(result->outputs[0]), y, 0);
  std::vector<float> previous;
  for (int repeat = 0; repeat < 3; ++repeat) {
    ASSERT_TRUE((*weight)->Activate().ok());
    auto backward = (*layer)->bwd(*executor_, {*grad}, result->state);
    ASSERT_TRUE(backward.ok()) << backward.status();
    Near(Read((*backward)[0]), dx, 0);
    auto actual = Read((*weight)->gradient());
    Near(actual, dw, 0);
    if (repeat)
      EXPECT_EQ(actual, previous);
    previous = std::move(actual);
    ASSERT_TRUE((*weight)->Deactivate().ok());
  }
}

TEST_F(BlockTrainingTest, RmsNormHandlesProductionSizedWidthsAndZeroInput) {
  for (int width : {257, 5120, 8193}) {
    SCOPED_TRACE(width);
    std::vector<float> x(width, 0), weights(width, 0.25f), dy(width, 0.5f);
    auto input = Input(x);
    auto weight = Parameter(weights, DataType::FP32);
    auto grad = Upload(dy);
    ASSERT_TRUE(input.ok());
    ASSERT_TRUE(weight.ok());
    ASSERT_TRUE(grad.ok());
    auto layer =
        BlockRmsNormLayer::Create(*executor_, *weight, width, 1, 0.25f);
    ASSERT_TRUE(layer.ok());
    ASSERT_TRUE((*weight)->Activate().ok());
    auto result = (*layer)->fwd(*executor_, {*input});
    ASSERT_TRUE(result.ok()) << result.status();
    auto backward = (*layer)->bwd(*executor_, {*grad}, result->state);
    ASSERT_TRUE(backward.ok()) << backward.status();
    Near(Read<__nv_bfloat16>(result->outputs[0]), x, 0);
    Near(Read((*backward)[0]), std::vector<float>(width, 1.25f));
    Near(Read((*weight)->gradient()), x, 0);
  }
}

TEST_F(BlockTrainingTest, RmsNormMatchesExplicitDerivative) {
  constexpr int width = 7, tokens = 2;
  constexpr float eps = 1e-5;
  std::vector<float> x(width * tokens), w(width), dy(width * tokens);
  for (int i = 0; i < width * tokens; ++i) {
    x[i] = (i - 4) * 0.25f;
    dy[i] = (i + 1) * 0.125f;
  }
  for (int i = 0; i < width; ++i)
    w[i] = (i - 2) * 0.1f;
  auto input = Input(x);
  auto weight = Parameter(w, DataType::FP32);
  auto grad = Upload(dy);
  ASSERT_TRUE(input.ok());
  ASSERT_TRUE(weight.ok());
  ASSERT_TRUE(grad.ok());
  auto layer =
      BlockRmsNormLayer::Create(*executor_, *weight, width, tokens, eps);
  ASSERT_TRUE(layer.ok()) << layer.status();
  ASSERT_TRUE((*weight)->Activate().ok());
  auto forward = (*layer)->fwd(*executor_, {*input});
  ASSERT_TRUE(forward.ok()) << forward.status();
  auto backward = (*layer)->bwd(*executor_, {*grad}, forward->state);
  ASSERT_TRUE(backward.ok()) << backward.status();
  std::vector<float> y(x.size()), dx(x.size()), dw(width);
  for (int t = 0; t < tokens; ++t) {
    float sum = 0, dot = 0;
    for (int i = 0; i < width; ++i) {
      sum += x[t * width + i] * x[t * width + i];
      dot += dy[t * width + i] * (1 + w[i]) * x[t * width + i];
    }
    float inv = 1 / std::sqrt(sum / width + eps);
    for (int i = 0; i < width; ++i) {
      const int j = t * width + i;
      y[j] = Bf16(x[j] * inv * (1 + w[i]));
      dx[j] = inv * dy[j] * (1 + w[i]) - x[j] * inv * inv * inv * dot / width;
      dw[i] += dy[j] * x[j] * inv;
    }
  }
  Near(Read<__nv_bfloat16>(forward->outputs[0]), y);
  Near(Read((*backward)[0]), dx);
  Near(Read((*weight)->gradient()), dw);
}

TEST_F(BlockTrainingTest, SwiGluUsesRoundedSiluWithStraightThroughDerivative) {
  std::vector<float> a{-2, -1, 0, 1, 2, 3}, b{2, -1, 3, 0.5, -2, 1},
      g{1, 2, 3, 4, 5, 6};
  auto gate = Input(a);
  auto up = Input(b);
  auto dy = Upload(g);
  ASSERT_TRUE(gate.ok());
  ASSERT_TRUE(up.ok());
  ASSERT_TRUE(dy.ok());
  auto layer = BlockSwiGluLayer::Create(3, 2);
  ASSERT_TRUE(layer.ok());
  auto forward = (*layer)->fwd(*executor_, {*gate, *up});
  ASSERT_TRUE(forward.ok()) << forward.status();
  auto backward = (*layer)->bwd(*executor_, {*dy}, forward->state);
  ASSERT_TRUE(backward.ok()) << backward.status();
  std::vector<float> y(6), da(6), db(6);
  for (int i = 0; i < 6; ++i) {
    float sigmoid = 1 / (1 + std::exp(-a[i]));
    y[i] = Bf16(Bf16(a[i] * sigmoid) * b[i]);
    da[i] = g[i] * b[i] * sigmoid * (1 + a[i] * (1 - sigmoid));
    db[i] = g[i] * Bf16(a[i] * sigmoid);
  }
  Near(Read<__nv_bfloat16>(forward->outputs[0]), y);
  Near(Read((*backward)[0]), da);
  Near(Read((*backward)[1]), db);
}

TEST_F(BlockTrainingTest, DequantizeRespectsBothScaleBlockBoundaries) {
  using Storage = MatrixStorage;
  std::vector<__nv_fp8_e4m3> values(129 * 129, __nv_fp8_e4m3(0.5f));
  auto input = Upload(values);
  auto scales = Upload<float>({1, 2, 3, 4});
  ASSERT_TRUE(input.ok());
  ASSERT_TRUE(scales.ok());
  auto result = DequantizeMatrix(*executor_, *input, Storage::kFp8E4M3, *scales,
                                 129, 129);
  ASSERT_TRUE(result.ok()) << result.status();
  auto actual = Read<__nv_bfloat16>(*result);
  ASSERT_EQ(actual.size(), values.size());
  for (int row = 0; row < 129; ++row)
    for (int col = 0; col < 129; ++col)
      EXPECT_EQ(actual[row * 129 + col],
                (1 + 2 * (row / 128) + col / 128) * 0.5f);
  auto shared = DequantizeMatrix(*executor_, *result, Storage::kBFloat16,
                                 std::nullopt, 129, 129);
  ASSERT_TRUE(shared.ok());
  EXPECT_EQ(shared->data(), result->data());
  EXPECT_FALSE(DequantizeMatrix(*executor_, *input, Storage::kFp8E4M3,
                                std::nullopt, 129, 129)
                   .ok());
}

TEST_F(BlockTrainingTest, RejectsIncorrectStorageAndBatchSize) {
  auto weight = Parameter({1, 2, 3, 4});
  auto input = Input({1, 2, 3, 4});
  ASSERT_TRUE(weight.ok());
  ASSERT_TRUE(input.ok());
  auto layer = BlockLinearLayer::Create(*executor_, *weight, 2, 2, 1);
  ASSERT_TRUE(layer.ok());
  EXPECT_FALSE((*layer)->fwd(*executor_, {*input}).ok());
  EXPECT_FALSE(BlockRmsNormLayer::Create(*executor_, *weight, 4, 1, 1e-5).ok());
  EXPECT_FALSE(BlockLinearLayer::Create(*executor_, *weight, 3, 2, 1).ok());
  EXPECT_FALSE(BlockSwiGluLayer::Create(-1, 1).ok());
}
}  // namespace
}  // namespace pluto::llm
