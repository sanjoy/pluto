#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "src/cuda/page_locked_host_array.h"
#include "src/llm/layer_hooks.h"
#include "src/llm/layers/fully_connected.h"
#include "src/util/status_macros.h"

namespace pluto::llm {
namespace {

float Bf16(float x) { return static_cast<float>(__nv_bfloat16(x)); }
float Fp8(float x) { return static_cast<float>(__nv_fp8_e4m3(x)); }

// Scalar checkpoint-input quantization. Public inputs have already crossed a
// BF16 boundary; the dynamic scale is recomputed independently for each group.
std::vector<float> QuantizeInput(std::vector<float> input) {
  for (size_t start = 0; start < input.size(); start += 128) {
    const size_t end = std::min(start + 128, input.size());
    float maximum = 0;
    for (size_t i = start; i < end; ++i)
      maximum = std::max(maximum, std::abs(input[i]));
    const float scale = maximum / 448.0f;
    for (size_t i = start; i < end; ++i)
      input[i] = Fp8(input[i] / std::max(scale, 1e-12f)) * scale;
  }
  return input;
}

class ImportedLinearTest : public testing::Test {
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
    ASSIGN_OR_RETURN(Buffer buffer,
                     Buffer::Allocate(*executor_, values.size() * sizeof(T)));
    RETURN_IF_ERROR(cuda::CudaStatus(
        cudaMemcpyAsync(buffer.data(), host.data(), buffer.size_bytes(),
                        cudaMemcpyHostToDevice, executor_->stream()),
        "upload imported linear test"));
    return buffer;
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
    EXPECT_TRUE(host.ok()) << host.status();
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

TEST_F(ImportedLinearTest, SharesAllStorageWithoutMastersOrGradients) {
  auto input = Input({1, 0, -1});
  auto fp32 = Upload<float>({1, 2, 3, 4, 5, 6});
  auto bf16 = Input({1, 2, 3, 4, 5, 6});
  auto fp8 = Upload<__nv_fp8_e4m3>({__nv_fp8_e4m3(1.f), __nv_fp8_e4m3(2.f),
                                    __nv_fp8_e4m3(3.f), __nv_fp8_e4m3(4.f),
                                    __nv_fp8_e4m3(5.f), __nv_fp8_e4m3(6.f)});
  auto scales = Upload<float>({1});
  ASSERT_TRUE(input.ok());
  ASSERT_TRUE(fp32.ok());
  ASSERT_TRUE(bf16.ok());
  ASSERT_TRUE(fp8.ok());
  ASSERT_TRUE(scales.ok());
  for (MatrixStorage storage :
       {MatrixStorage::kFloat32, MatrixStorage::kBFloat16,
        MatrixStorage::kFp8E4M3}) {
    SCOPED_TRACE(static_cast<int>(storage));
    const Buffer& weights = storage == MatrixStorage::kFloat32    ? *fp32
                            : storage == MatrixStorage::kBFloat16 ? *bf16
                                                                  : *fp8;
    auto layer = FullyConnectedLayer::Create(
        *executor_, weights, storage, 3, 2,
        storage == MatrixStorage::kFp8E4M3 ? std::optional<Buffer>(*scales)
                                           : std::nullopt);
    ASSERT_TRUE(layer.ok()) << layer.status();
    EXPECT_EQ((*layer)->weights()[0].data(), weights.data());
    EXPECT_EQ((*layer)->weights().size(),
              storage == MatrixStorage::kFp8E4M3 ? 2 : 1);
    if (storage == MatrixStorage::kFp8E4M3)
      EXPECT_EQ((*layer)->weights()[1].data(), scales->data());
    EXPECT_TRUE((*layer)->gradients().empty());
    EXPECT_EQ((*layer)->input_types()[0],
              ActivationType(DataType::BF16, {-2, 1, 3}));
    EXPECT_EQ((*layer)->output_types()[0],
              ActivationType(DataType::BF16, {-2, 1, 2}));
    EXPECT_EQ((*layer)->output_type(), DataType::BF16);
    EXPECT_EQ((*layer)->input_dim(), 3);
    EXPECT_EQ((*layer)->output_dim(), 2);
    EXPECT_TRUE(absl::IsFailedPrecondition((*layer)->InitializeIdentity()));
    EXPECT_TRUE(absl::IsFailedPrecondition((*layer)->InitializeNormal(1, 123)));
    auto result = (*layer)->fwd(*executor_, {*input});
    ASSERT_TRUE(result.ok()) << result.status();
    // Initialization rejection must not mutate the retained checkpoint.
    EXPECT_EQ(Read(result->outputs[0]), (std::vector<float>{-2, -2}));
    EXPECT_TRUE(result->state.intermediates.empty());
    EXPECT_TRUE(absl::IsUnimplemented(
        (*layer)->bwd(*executor_, {}, std::move(result->state)).status()));
  }
}

TEST_F(ImportedLinearTest, BlockScalesTailsAndRealWidthMatchScalarProjection) {
  // Straddle 8-row compute tiles and 128x128 scale blocks, including the real
  // 5120-wide checkpoint input. Every storage uses the same BF16 activation.
  for (const auto& [cols, rows] : {std::pair{263, 259}, std::pair{5120, 131}}) {
    SCOPED_TRACE(cols);
    SCOPED_TRACE(rows);
    std::vector<float> source(size_t(cols) * rows);
    std::vector<__nv_bfloat16> bf16(source.size());
    std::vector<__nv_fp8_e4m3> fp8(source.size());
    for (size_t i = 0; i < source.size(); ++i) {
      source[i] = 1.9f * std::sin(float(i % 1009) * 0.037f);
      bf16[i] = __nv_bfloat16(source[i]);
      fp8[i] = __nv_fp8_e4m3(source[i]);
    }
    std::vector<float> input(cols);
    for (int c = 0; c < cols; ++c)
      input[c] = Bf16(std::cos(c * 0.123f) * 0.25f);
    const int scale_cols = (cols + 127) / 128;
    const int scale_rows = (rows + 127) / 128;
    std::vector<float> scales(scale_cols * scale_rows);
    for (int r = 0; r < scale_rows; ++r)
      for (int c = 0; c < scale_cols; ++c)
        scales[r * scale_cols + c] = (r + 1) * (c + 1) * 0.03125f;
    auto df32 = Upload(source);
    auto dbf16 = Upload(bf16);
    auto dfp8 = Upload(fp8);
    auto dx = Input(input);
    auto ds = Upload(scales);
    ASSERT_TRUE(df32.ok());
    ASSERT_TRUE(dbf16.ok());
    ASSERT_TRUE(dfp8.ok());
    ASSERT_TRUE(dx.ok());
    ASSERT_TRUE(ds.ok());
    for (MatrixStorage storage :
         {MatrixStorage::kFloat32, MatrixStorage::kBFloat16,
          MatrixStorage::kFp8E4M3}) {
      SCOPED_TRACE(static_cast<int>(storage));
      const Buffer& weights = storage == MatrixStorage::kFloat32    ? *df32
                              : storage == MatrixStorage::kBFloat16 ? *dbf16
                                                                    : *dfp8;
      auto layer = FullyConnectedLayer::Create(
          *executor_, weights, storage, cols, rows,
          storage == MatrixStorage::kFp8E4M3 ? std::optional<Buffer>(*ds)
                                             : std::nullopt);
      ASSERT_TRUE(layer.ok()) << layer.status();
      auto result = (*layer)->fwd(*executor_, {*dx});
      ASSERT_TRUE(result.ok()) << result.status();
      const auto actual = Read(result->outputs[0]);
      ASSERT_EQ(actual.size(), rows);
      const auto quantized =
          storage == MatrixStorage::kFp8E4M3 ? QuantizeInput(input) : input;
      for (int r = 0; r < rows; ++r) {
        double sum = 0;
        for (int c = 0; c < cols; ++c) {
          const size_t i = size_t(r) * cols + c;
          float weight = source[i];
          if (storage == MatrixStorage::kBFloat16)
            weight = float(bf16[i]);
          if (storage == MatrixStorage::kFp8E4M3)
            weight = float(fp8[i]) * scales[(r / 128) * scale_cols + c / 128];
          sum += double(weight) * quantized[c];
        }
        const float expected = Bf16(sum);
        EXPECT_NEAR(actual[r], expected, 5e-5f + 0.008f * std::abs(expected))
            << "row " << r;
      }
    }
  }
}

TEST_F(ImportedLinearTest, DynamicFp8QuantizationHandlesZeroTinyTiesAndTail) {
  constexpr int kWidth = 389;
  std::vector<float> input(kWidth);
  // Zero group, exact-unit scale with midpoint ties, tiny group testing the
  // divisor floor, and a final partial group all require distinct treatment.
  for (int i = 128; i < 256; ++i)
    input[i] = ((i % 31) - 15) * 0.0625f;
  input[128] = 448;
  input[129] = -448;
  input[130] = 1.0625f;
  input[131] = 1.1875f;
  for (int i = 256; i < 384; ++i)
    input[i] = Bf16((i % 5 - 2) * 1e-14f);
  for (int i = 384; i < kWidth; ++i)
    input[i] = Bf16((i - 386) * 25.3f);
  // An identity matrix exposes every quantized input through the layer API.
  std::vector<__nv_fp8_e4m3> identity(size_t(kWidth) * kWidth,
                                      __nv_fp8_e4m3(0.f));
  for (int i = 0; i < kWidth; ++i)
    identity[size_t(i) * kWidth + i] = __nv_fp8_e4m3(1.f);
  auto weights = Upload(identity);
  auto scales = Upload<float>(std::vector<float>(16, 1));
  auto dx = Input(input);
  ASSERT_TRUE(weights.ok());
  ASSERT_TRUE(scales.ok());
  ASSERT_TRUE(dx.ok());
  auto layer = FullyConnectedLayer::Create(
      *executor_, *weights, MatrixStorage::kFp8E4M3, kWidth, kWidth, *scales);
  ASSERT_TRUE(layer.ok()) << layer.status();
  auto output = (*layer)->fwd(*executor_, {*dx});
  ASSERT_TRUE(output.ok()) << output.status();
  const auto actual = Read(output->outputs[0]);
  const auto expected = QuantizeInput(input);
  ASSERT_EQ(actual.size(), expected.size());
  for (size_t i = 0; i < input.size(); ++i)
    EXPECT_NEAR(actual[i], Bf16(expected[i]),
                1e-20f + 0.008f * std::abs(expected[i]))
        << i;
  EXPECT_EQ(actual[130], 1.f);
  EXPECT_EQ(actual[131], 1.25f);
}

TEST_F(ImportedLinearTest, RejectsBadStorageDimensionsScalesAndExecutors) {
  auto weights = Upload<float>({1, 2, 3, 4});
  auto scales = Upload<float>({1});
  auto fp8 = Upload<uint8_t>({0, 0, 0, 0});
  auto input = Input({1, 2});
  ASSERT_TRUE(weights.ok());
  ASSERT_TRUE(scales.ok());
  ASSERT_TRUE(fp8.ok());
  ASSERT_TRUE(input.ok());
  for (int bad : {-1, 0, FullyConnectedLayer::kMaximumDimension + 1,
                  std::numeric_limits<int>::max()}) {
    EXPECT_FALSE(FullyConnectedLayer::Create(*executor_, *weights,
                                             MatrixStorage::kFloat32, bad, 2)
                     .ok());
    EXPECT_FALSE(FullyConnectedLayer::Create(*executor_, *weights,
                                             MatrixStorage::kFloat32, 2, bad)
                     .ok());
  }
  EXPECT_FALSE(FullyConnectedLayer::Create(*executor_, *weights,
                                           MatrixStorage::kFloat32, 3, 2)
                   .ok());
  EXPECT_FALSE(FullyConnectedLayer::Create(
                   *executor_, *weights, MatrixStorage::kFloat32, 2, 2, *scales)
                   .ok());
  EXPECT_FALSE(FullyConnectedLayer::Create(*executor_, *fp8,
                                           MatrixStorage::kFp8E4M3, 2, 2)
                   .ok());
  EXPECT_FALSE(FullyConnectedLayer::Create(
                   *executor_, *fp8, MatrixStorage::kFp8E4M3, 2, 2, *weights)
                   .ok());
  EXPECT_FALSE(FullyConnectedLayer::Create(*executor_, *weights,
                                           static_cast<MatrixStorage>(99), 2, 2)
                   .ok());
  auto layer = FullyConnectedLayer::Create(*executor_, *weights,
                                           MatrixStorage::kFloat32, 2, 2);
  ASSERT_TRUE(layer.ok());
  EXPECT_FALSE((*layer)->fwd(*executor_, {}).ok());
  EXPECT_FALSE((*layer)->fwd(*executor_, {*input, *input}).ok());
  EXPECT_FALSE((*layer)->fwd(*executor_, {*weights}).ok());
  auto other = cuda::Executor::Create();
  ASSERT_TRUE(other.ok());
  EXPECT_FALSE(FullyConnectedLayer::Create(**other, *weights,
                                           MatrixStorage::kFloat32, 2, 2)
                   .ok());
  auto foreign = Buffer::Allocate(**other, sizeof(float));
  ASSERT_TRUE(foreign.ok());
  EXPECT_FALSE(FullyConnectedLayer::Create(
                   *executor_, *fp8, MatrixStorage::kFp8E4M3, 2, 2, *foreign)
                   .ok());
  EXPECT_FALSE((*layer)->fwd(**other, {*input}).ok());
  EXPECT_FALSE((*layer)->fwd(*executor_, {*foreign}).ok());
  EXPECT_TRUE((*other)->Synchronize().ok());
}

TEST_F(ImportedLinearTest, ImportedPathRetainsNormalLayerHooks) {
  auto weights = Upload<float>({1, 2, 3, 4});
  auto input = Input({1, 2});
  ASSERT_TRUE(weights.ok());
  ASSERT_TRUE(input.ok());
  auto layer = FullyConnectedLayer::Create(*executor_, *weights,
                                           MatrixStorage::kFloat32, 2, 2);
  ASSERT_TRUE(layer.ok());
  int calls = 0;
  LayerHooks hooks;
  hooks.activation_hook = [&](cuda::Executor& executor, absl::string_view name,
                              absl::Span<const ActivationType> types,
                              absl::Span<Buffer> outputs) {
    EXPECT_EQ(&executor, executor_.get());
    EXPECT_EQ(name, "FullyConnectedLayer");
    EXPECT_EQ(types[0], ActivationType(DataType::BF16, {-2, 1, 2}));
    EXPECT_EQ(outputs.size(), 1);
    ++calls;
    return absl::OkStatus();
  };
  auto result = (*layer)->fwd(*executor_, {*input}, &hooks);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(Read(result->outputs[0]), (std::vector<float>{5, 11}));
  EXPECT_EQ(calls, 1);
}

}  // namespace
}  // namespace pluto::llm
