#include "src/llm/experiments/memorize_general_facts/fit_attention_readout/fixed_preprocessing.h"

#include <cuda_runtime_api.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <numbers>
#include <random>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "src/cuda/page_locked_host_array.h"
#include "src/util/status_macros.h"

namespace pluto::llm::fit_attention_readout {
namespace {

uint16_t ToBf16(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  bits += 0x7fff + ((bits >> 16) & 1);
  return bits >> 16;
}

float FromBf16(uint16_t value) {
  const uint32_t bits = static_cast<uint32_t>(value) << 16;
  float result;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

// Deliberately scalar CPU oracle: each output is computed only from one row.
// Double-precision summation differs from the GPU's FP32 reduction order, so
// comparisons allow the final BF16 rounding unit rather than requiring bits.
std::vector<float> Reference(const std::vector<uint16_t>& values, int width,
                             const FixedPreprocessingOptions& options) {
  const double pi = std::numbers::pi;
  std::vector<double> random_projection(width * width / 2);
  if (options.kind == FixedPreprocessingKind::kRandomFourier) {
    std::mt19937 random(options.seed);
    for (auto& coefficient : random_projection) {
      const double u = (static_cast<double>(random()) + 0.5) / 4294967296.0;
      const double v = (static_cast<double>(random()) + 0.5) / 4294967296.0;
      coefficient =
          std::sqrt(-2.0 * std::log(u) / width) * std::cos(2.0 * pi * v);
    }
  }
  std::vector<float> result(values.size());
  for (size_t row = 0; row < values.size() / width; ++row)
    for (int output = 0; output < width; ++output) {
      const size_t index = row * width + output;
      const double value = FromBf16(values[index]);
      if (options.kind == FixedPreprocessingKind::kIdentity)
        result[index] = value;
      else if (options.kind == FixedPreprocessingKind::kSin)
        result[index] = std::sin(options.scale * value);
      else if (options.kind == FixedPreprocessingKind::kCos)
        result[index] = std::cos(options.scale * value);
      else if (options.kind == FixedPreprocessingKind::kSignedSqrt)
        result[index] = std::copysign(std::sqrt(std::abs(value)), value);
      else {
        double sum = 0.0;
        for (int input = 0; input < width; ++input) {
          double coefficient;
          if (options.kind == FixedPreprocessingKind::kDct)
            coefficient = std::sqrt((output == 0 ? 1.0 : 2.0) / width) *
                          std::cos(pi * (input + 0.5) * output / width);
          else if (options.kind == FixedPreprocessingKind::kRandomFourier)
            coefficient =
                random_projection[(output % (width / 2)) * width + input];
          else if (output == 0)
            coefficient = 1.0 / std::sqrt(width);
          else if (output == width - 1)
            coefficient = (input % 2 == 0 ? 1.0 : -1.0) / std::sqrt(width);
          else {
            const double angle = 2.0 * pi * input * ((output + 1) / 2) / width;
            coefficient =
                std::sqrt(2.0 / width) *
                (output % 2 == 1 ? std::cos(angle) : -std::sin(angle));
          }
          sum += coefficient * FromBf16(values[row * width + input]);
        }
        if (options.kind == FixedPreprocessingKind::kRandomFourier)
          sum = output < width / 2 ? std::cos(options.scale * sum)
                                   : std::sin(options.scale * sum);
        result[index] = sum;
      }
    }
  return result;
}

class FixedPreprocessingTest : public testing::Test {
 protected:
  void SetUp() override {
    auto executor = cuda::Executor::Create();
    ASSERT_TRUE(executor.ok()) << executor.status();
    executor_ = std::move(*executor);
  }

  void TearDown() override {
    if (executor_ == nullptr)
      return;
    EXPECT_TRUE(executor_->Synchronize().ok());
  }

  absl::StatusOr<cuda::Buffer> Upload(const std::vector<uint16_t>& values) {
    ASSIGN_OR_RETURN(auto host, cuda::PageLockedHostArray<uint16_t>::CopyFrom(
                                    *executor_, values));
    ASSIGN_OR_RETURN(auto device,
                     cuda::Buffer::Allocate(*executor_, host.size_bytes()));
    RETURN_IF_ERROR(cuda::CudaStatus(
        cudaMemcpyAsync(device.data(), host.data(), host.size_bytes(),
                        cudaMemcpyHostToDevice, executor_->stream()),
        "upload fixed preprocessing test"));
    return device;
  }

  absl::StatusOr<std::vector<uint16_t>> Download(const cuda::Buffer& device) {
    ASSIGN_OR_RETURN(auto host, cuda::PageLockedHostArray<uint16_t>::Allocate(
                                    *executor_, device.size_bytes() / 2));
    RETURN_IF_ERROR(cuda::CudaStatus(
        cudaMemcpyAsync(host.data(), device.data(), host.size_bytes(),
                        cudaMemcpyDeviceToHost, executor_->stream()),
        "download fixed preprocessing test"));
    RETURN_IF_ERROR(executor_->Synchronize());
    return std::vector<uint16_t>(host.begin(), host.end());
  }

  std::unique_ptr<cuda::Executor> executor_;
};

class EveryFixedPreprocessingTest
    : public FixedPreprocessingTest,
      public testing::WithParamInterface<FixedPreprocessingKind> {};

TEST_P(EveryFixedPreprocessingTest,
       MatchesCpuAndIsDeterministicWithoutMutation) {
  for (const int width : {2, 10, 128}) {
    SCOPED_TRACE(width);
    std::vector<uint16_t> values(7 * width);
    for (size_t i = 0; i < values.size(); ++i)
      values[i] = ToBf16((static_cast<int>(i * 7 % 51) - 25) / 8.0f);
    auto input = Upload(values);
    ASSERT_TRUE(input.ok()) << input.status();
    const FixedPreprocessingOptions options{GetParam(), 0.7f, 42};
    const auto expected = Reference(values, width, options);
    auto first = PreprocessHiddenStates(*executor_, *input, width, options);
    auto second = PreprocessHiddenStates(*executor_, *input, width, options);
    ASSERT_TRUE(first.ok()) << first.status();
    ASSERT_TRUE(second.ok()) << second.status();
    EXPECT_EQ(first->size_bytes(), input->size_bytes());
    EXPECT_TRUE(GetParam() == FixedPreprocessingKind::kIdentity ||
                first->data() != input->data());
    auto first_values = Download(*first);
    auto second_values = Download(*second);
    auto input_after = Download(*input);
    ASSERT_TRUE(first_values.ok()) << first_values.status();
    ASSERT_TRUE(second_values.ok()) << second_values.status();
    ASSERT_TRUE(input_after.ok()) << input_after.status();
    EXPECT_EQ(*first_values, *second_values);
    EXPECT_EQ(*input_after, values);
    for (size_t i = 0; i < values.size(); ++i)
      EXPECT_NEAR(FromBf16((*first_values)[i]), expected[i],
                  0.006f * std::abs(expected[i]) + 0.0001f)
          << "index " << i;
  }
}

TEST_P(EveryFixedPreprocessingTest, NeverMixesRowsOrPositions) {
  constexpr int kWidth = 10;
  std::vector<uint16_t> single(kWidth);
  for (int i = 0; i < kWidth; ++i)
    single[i] = ToBf16((i - 4) / 3.0f);
  std::vector<uint16_t> batch(7 * kWidth, ToBf16(15.0f));
  for (const int row : {0, 3, 6})
    for (int column = 0; column < kWidth; ++column)
      batch[row * kWidth + column] = single[column];
  auto single_input = Upload(single);
  auto batch_input = Upload(batch);
  ASSERT_TRUE(single_input.ok());
  ASSERT_TRUE(batch_input.ok());
  const FixedPreprocessingOptions options{GetParam(), 1.3f, 7};
  auto single_output =
      PreprocessHiddenStates(*executor_, *single_input, kWidth, options);
  auto batch_output =
      PreprocessHiddenStates(*executor_, *batch_input, kWidth, options);
  ASSERT_TRUE(single_output.ok()) << single_output.status();
  ASSERT_TRUE(batch_output.ok()) << batch_output.status();
  auto single_values = Download(*single_output);
  auto batch_values = Download(*batch_output);
  ASSERT_TRUE(single_values.ok());
  ASSERT_TRUE(batch_values.ok());
  for (const int row : {0, 3, 6})
    for (int column = 0; column < kWidth; ++column)
      EXPECT_EQ((*batch_values)[row * kWidth + column],
                (*single_values)[column]);
}

INSTANTIATE_TEST_SUITE_P(
    AllKinds, EveryFixedPreprocessingTest,
    testing::Values(FixedPreprocessingKind::kIdentity,
                    FixedPreprocessingKind::kDct,
                    FixedPreprocessingKind::kRealDft,
                    FixedPreprocessingKind::kSin, FixedPreprocessingKind::kCos,
                    FixedPreprocessingKind::kSignedSqrt,
                    FixedPreprocessingKind::kRandomFourier));

TEST_F(FixedPreprocessingTest, FourierSeedAndScaleActuallyChangeFeatures) {
  std::vector<uint16_t> values(10);
  for (int i = 0; i < 10; ++i)
    values[i] = ToBf16(i / 4.0f);
  auto input = Upload(values);
  ASSERT_TRUE(input.ok());
  std::vector<std::vector<uint16_t>> results;
  for (const auto options :
       {FixedPreprocessingOptions{FixedPreprocessingKind::kRandomFourier, 1.0f,
                                  1},
        FixedPreprocessingOptions{FixedPreprocessingKind::kRandomFourier, 1.0f,
                                  2},
        FixedPreprocessingOptions{FixedPreprocessingKind::kRandomFourier, 2.0f,
                                  1}}) {
    auto output = PreprocessHiddenStates(*executor_, *input, 10, options);
    ASSERT_TRUE(output.ok()) << output.status();
    auto actual = Download(*output);
    ASSERT_TRUE(actual.ok());
    results.push_back(std::move(*actual));
  }
  EXPECT_NE(results[0], results[1]);
  EXPECT_NE(results[0], results[2]);
}

TEST_F(FixedPreprocessingTest, RejectsInvalidShapeExecutorAndOptions) {
  auto input = Upload(std::vector<uint16_t>(10, ToBf16(1.0f)));
  ASSERT_TRUE(input.ok());
  for (const int width : {0, -1, 3, 129})
    EXPECT_FALSE(PreprocessHiddenStates(*executor_, *input, width, {}).ok());
  for (const auto kind : {FixedPreprocessingKind::kRealDft,
                          FixedPreprocessingKind::kRandomFourier})
    EXPECT_FALSE(PreprocessHiddenStates(*executor_, *input, 5, {kind}).ok());
  for (const float scale :
       {0.0f, -1.0f, std::numeric_limits<float>::quiet_NaN(),
        std::numeric_limits<float>::infinity()})
    EXPECT_FALSE(
        PreprocessHiddenStates(*executor_, *input, 10,
                               {FixedPreprocessingKind::kIdentity, scale})
            .ok());
  EXPECT_FALSE(
      PreprocessHiddenStates(*executor_, *input, 10,
                             {static_cast<FixedPreprocessingKind>(999)})
          .ok());
  auto other = cuda::Executor::Create();
  ASSERT_TRUE(other.ok());
  EXPECT_FALSE(PreprocessHiddenStates(**other, *input, 10, {}).ok());
}

}  // namespace
}  // namespace pluto::llm::fit_attention_readout
