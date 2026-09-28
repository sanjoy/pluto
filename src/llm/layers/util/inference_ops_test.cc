#include "src/llm/layers/util/inference_ops.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "gtest/gtest.h"
#include "src/cuda/buffer.h"
#include "src/cuda/executor.h"
#include "src/cuda/page_locked_host_array.h"

namespace pluto::llm::inference_ops {
namespace {

float Bfloat16(float x) { return static_cast<float>(__nv_bfloat16(x)); }
float Fp8(float x) { return static_cast<float>(__nv_fp8_e4m3(x)); }
float Round(float x, bool round) { return round ? Bfloat16(x) : x; }

class OpsTest : public testing::Test {
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
  absl::StatusOr<cuda::Buffer> Upload(const std::vector<T>& values) {
    auto staging = cuda::PageLockedHostArray<T>::CopyFrom(*executor_, values);
    if (!staging.ok())
      return staging.status();
    auto buffer = cuda::Buffer::Allocate(*executor_, values.size() * sizeof(T));
    if (!buffer.ok())
      return buffer.status();
    auto status = cuda::CudaStatus(
        cudaMemcpyAsync(buffer->data(), staging->data(), buffer->size_bytes(),
                        cudaMemcpyHostToDevice, executor_->stream()),
        "upload ops test input");
    if (!status.ok())
      return status;
    const auto synchronized = executor_->Synchronize();
    if (!synchronized.ok())
      return synchronized;
    return std::move(*buffer);
  }

  std::vector<float> Read(cuda::Buffer& buffer) {
    auto staging = cuda::PageLockedHostArray<float>::Allocate(
        *executor_, buffer.size_bytes() / sizeof(float));
    EXPECT_TRUE(staging.ok()) << staging.status();
    if (!staging.ok())
      return {};
    const auto copied =
        cudaMemcpyAsync(staging->data(), buffer.data(), buffer.size_bytes(),
                        cudaMemcpyDeviceToHost, executor_->stream());
    EXPECT_EQ(copied, cudaSuccess);
    if (copied != cudaSuccess)
      return {};
    const auto synchronized = executor_->Synchronize();
    EXPECT_TRUE(synchronized.ok()) << synchronized;
    if (!synchronized.ok())
      return {};
    return std::vector<float>(staging->begin(), staging->end());
  }

  void Near(const std::vector<float>& actual,
            const std::vector<float>& expected, bool round = false,
            float absolute = 1e-5f) {
    ASSERT_EQ(actual.size(), expected.size());
    for (size_t i = 0; i < actual.size(); ++i)
      EXPECT_NEAR(actual[i], expected[i],
                  absolute + (round ? 0.008f : 2e-5f) * std::abs(expected[i]))
          << "element " << i;
  }

  static float* Data(cuda::Buffer& buffer) {
    return static_cast<float*>(buffer.data());
  }
  std::unique_ptr<cuda::Executor> executor_;
};

TEST_F(OpsTest,
       MatrixStorageBlockScalesAndTailsMatchIndependentDequantization) {
  // Three scale blocks in each dimension, plus the real hidden width spanning
  // forty FP8 groups. Row counts straddle both 8-row tiles and 128-row scales.
  for (const auto& [cols, rows] : {std::pair{263, 259}, std::pair{5120, 131}}) {
    SCOPED_TRACE(cols);
    SCOPED_TRACE(rows);
    std::vector<float> source(static_cast<size_t>(cols) * rows);
    std::vector<__nv_bfloat16> bf16(source.size());
    std::vector<__nv_fp8_e4m3> fp8(source.size());
    for (size_t i = 0; i < source.size(); ++i) {
      source[i] = 1.9f * std::sin(static_cast<float>(i % 1009) * 0.037f);
      bf16[i] = __nv_bfloat16(source[i]);
      fp8[i] = __nv_fp8_e4m3(source[i]);
    }
    std::vector<float> input(cols);
    for (int c = 0; c < cols; ++c)
      input[c] = std::cos(c * 0.123f) * 0.25f;
    const int scale_cols = (cols + 127) / 128;
    const int scale_rows = (rows + 127) / 128;
    std::vector<float> scales(scale_cols * scale_rows);
    for (int r = 0; r < scale_rows; ++r)
      for (int c = 0; c < scale_cols; ++c)
        scales[r * scale_cols + c] = (r + 1) * (c + 1) * 0.03125f;
    auto df32 = Upload(source);
    auto dbf16 = Upload(bf16);
    auto dfp8 = Upload(fp8);
    auto dx = Upload(input);
    auto ds = Upload(scales);
    auto output = Upload(std::vector<float>(rows));
    ASSERT_TRUE(df32.ok());
    ASSERT_TRUE(dbf16.ok());
    ASSERT_TRUE(dfp8.ok());
    ASSERT_TRUE(dx.ok());
    ASSERT_TRUE(ds.ok());
    ASSERT_TRUE(output.ok());
    for (MatrixStorage storage :
         {MatrixStorage::kFloat32, MatrixStorage::kBFloat16,
          MatrixStorage::kFp8E4M3}) {
      SCOPED_TRACE(static_cast<int>(storage));
      const void* weights = storage == MatrixStorage::kFloat32 ? df32->data()
                            : storage == MatrixStorage::kBFloat16
                                ? dbf16->data()
                                : dfp8->data();
      for (bool round : {false, true}) {
        SCOPED_TRACE(round);
        ASSERT_TRUE(
            MatVec(*executor_, weights, storage,
                   storage == MatrixStorage::kFp8E4M3 ? Data(*ds) : nullptr,
                   Data(*dx), cols, rows, Data(*output), round)
                .ok());
        std::vector<float> expected(rows);
        for (int r = 0; r < rows; ++r) {
          double sum = 0;
          for (int c = 0; c < cols; ++c) {
            const size_t i = static_cast<size_t>(r) * cols + c;
            float weight = source[i];
            if (storage == MatrixStorage::kBFloat16)
              weight = static_cast<float>(bf16[i]);
            if (storage == MatrixStorage::kFp8E4M3)
              weight = static_cast<float>(fp8[i]) *
                       scales[(r / 128) * scale_cols + c / 128];
            sum += static_cast<double>(weight) * input[c];
          }
          expected[r] = Round(sum, round);
        }
        Near(Read(*output), expected, round, 5e-5f);
      }
    }
  }
}

TEST_F(OpsTest, DynamicFp8QuantizationUsesIndependent128ElementGroups) {
  std::vector<float> input(389);
  // First group is all zero. Second includes E4M3 midpoint ties with an exact
  // unit scale. Third exercises the quantization-divisor floor; last is
  // partial.
  for (int i = 128; i < 256; ++i)
    input[i] = ((i % 31) - 15) * 0.0625f;
  input[128] = 448.0f;
  input[129] = -448.0f;
  input[130] = 1.0625f;
  input[131] = 1.1875f;
  for (int i = 256; i < 384; ++i)
    input[i] = (i % 5 - 2) * 1e-14f;
  for (int i = 384; i < 389; ++i)
    input[i] = (i - 386) * 25.3f;
  auto dx = Upload(input);
  auto output = Upload(std::vector<float>(input.size()));
  ASSERT_TRUE(dx.ok());
  ASSERT_TRUE(output.ok());
  ASSERT_TRUE(
      QuantizeFp8Input(*executor_, Data(*dx), input.size(), Data(*output))
          .ok());
  std::vector<float> expected(input.size());
  for (size_t start = 0; start < input.size(); start += 128) {
    const size_t end = std::min(start + 128, input.size());
    float maximum = 0;
    for (size_t i = start; i < end; ++i)
      maximum = std::max(maximum, std::abs(input[i]));
    const float scale = maximum / 448.0f;
    for (size_t i = start; i < end; ++i)
      expected[i] = Fp8(input[i] / std::max(scale, 1e-12f)) * scale;
  }
  const auto actual = Read(*output);
  for (size_t i = 0; i < input.size(); ++i)
    EXPECT_NEAR(actual[i], expected[i], 1e-20f + 1e-6f * std::abs(expected[i]))
        << i;
}

TEST_F(OpsTest, ZeroCenteredRmsNormMatchesScalarAtAllDispatchWidths) {
  for (int width : {3, 256, 257, 5120, 8193}) {
    SCOPED_TRACE(width);
    std::vector<float> x(width), weights(width);
    double squares = 0;
    for (int i = 0; i < width; ++i) {
      x[i] = std::sin(i * 0.7f + 0.2f);
      weights[i] = i % 7 == 0 ? -1.0f : 0.3f * std::cos(i * 0.13f);
      squares += static_cast<double>(x[i]) * x[i];
    }
    auto dx = Upload(x);
    auto dw = Upload(weights);
    auto output = Upload(std::vector<float>(width));
    ASSERT_TRUE(dx.ok());
    ASSERT_TRUE(dw.ok());
    ASSERT_TRUE(output.ok());
    for (bool round : {false, true}) {
      ASSERT_TRUE(RmsNorm(*executor_, Data(*dx), Data(*dw), width, 1e-6f,
                          Data(*output), round)
                      .ok());
      std::vector<float> expected(width);
      const double inverse = 1.0 / std::sqrt(squares / width + 1e-6);
      for (int i = 0; i < width; ++i)
        expected[i] = Round(x[i] * inverse * (1.0 + weights[i]), round);
      Near(Read(*output), expected, round);
    }
  }
}

TEST_F(OpsTest, SwiGluAndInPlaceResidualRespectBfloat16Boundaries) {
  for (int width : {7, 257, 17408}) {
    SCOPED_TRACE(width);
    std::vector<float> x(width), update(width);
    for (int i = 0; i < width; ++i) {
      x[i] = Bfloat16(std::sin(i * 0.31f) * 4.0f);
      update[i] = Bfloat16(std::cos(i * 0.29f));
    }
    for (bool round : {false, true}) {
      SCOPED_TRACE(round);
      auto dx = Upload(x);
      auto du = Upload(update);
      auto output = Upload(std::vector<float>(width));
      ASSERT_TRUE(dx.ok());
      ASSERT_TRUE(du.ok());
      ASSERT_TRUE(output.ok());
      ASSERT_TRUE(
          SwiGlu(*executor_, Data(*dx), Data(*du), width, Data(*output), round)
              .ok());
      std::vector<float> expected(width);
      for (int i = 0; i < width; ++i) {
        const float silu = Round(x[i] / (1.0f + std::exp(-x[i])), round);
        expected[i] = Round(silu * update[i], round);
      }
      Near(Read(*output), expected, round);
      for (int i = 0; i < width; ++i)
        expected[i] = Round(x[i] + update[i], round);
      ASSERT_TRUE(
          ResidualAdd(*executor_, Data(*dx), Data(*du), width, Data(*dx), round)
              .ok());
      EXPECT_EQ(Read(*dx), expected);
      for (int i = 0; i < width; ++i)
        expected[i] = Round(expected[i] + update[i], round);
      ASSERT_TRUE(
          ResidualAdd(*executor_, Data(*dx), Data(*du), width, Data(*du), round)
              .ok());
      EXPECT_EQ(Read(*du), expected);
    }
  }
}

TEST_F(OpsTest, EmbeddingLookupSelectsCorrectRowsIncludingPartialTiles) {
  for (int width : {7, 257, 5120}) {
    SCOPED_TRACE(width);
    std::vector<float> weights(5 * width);
    std::vector<__nv_bfloat16> bf16(weights.size());
    for (size_t i = 0; i < weights.size(); ++i) {
      weights[i] = std::sin(i * 0.035f) + i / width;
      bf16[i] = __nv_bfloat16(weights[i]);
    }
    auto df32 = Upload(weights);
    auto dbf16 = Upload(bf16);
    auto output = Upload(std::vector<float>(width));
    ASSERT_TRUE(df32.ok());
    ASSERT_TRUE(dbf16.ok());
    ASSERT_TRUE(output.ok());
    for (int token : {0, 3, 4}) {
      SCOPED_TRACE(token);
      for (MatrixStorage storage :
           {MatrixStorage::kFloat32, MatrixStorage::kBFloat16}) {
        ASSERT_TRUE(EmbeddingLookup(*executor_,
                                    storage == MatrixStorage::kFloat32
                                        ? df32->data()
                                        : dbf16->data(),
                                    storage, token, 5, width, Data(*output))
                        .ok());
        std::vector<float> expected(width);
        for (int i = 0; i < width; ++i)
          expected[i] = storage == MatrixStorage::kFloat32
                            ? weights[token * width + i]
                            : static_cast<float>(bf16[token * width + i]);
        EXPECT_EQ(Read(*output), expected);
      }
    }
  }
}

TEST_F(OpsTest, InvalidArgumentsAreRejectedBeforeLaunch) {
  auto data = Upload(std::vector<float>(16, 1.0f));
  ASSERT_TRUE(data.ok());
  EXPECT_EQ(MatVec(*executor_, data->data(), static_cast<MatrixStorage>(999),
                   nullptr, Data(*data), 4, 4, Data(*data))
                .code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(MatVec(*executor_, data->data(), MatrixStorage::kFp8E4M3, nullptr,
                   Data(*data), 4, 4, Data(*data))
                .code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(QuantizeFp8Input(*executor_, Data(*data), 0, Data(*data)).code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(
      RmsNorm(*executor_, Data(*data), Data(*data), 4, 0, Data(*data)).code(),
      absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(SwiGlu(*executor_, nullptr, Data(*data), 4, Data(*data)).code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(
      ResidualAdd(*executor_, Data(*data), nullptr, 4, Data(*data)).code(),
      absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(EmbeddingLookup(*executor_, data->data(), MatrixStorage::kFloat32,
                            4, 4, 4, Data(*data))
                .code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(EmbeddingLookup(*executor_, data->data(), MatrixStorage::kFp8E4M3,
                            0, 4, 4, Data(*data))
                .code(),
            absl::StatusCode::kUnimplemented);
}

}  // namespace
}  // namespace pluto::llm::inference_ops
