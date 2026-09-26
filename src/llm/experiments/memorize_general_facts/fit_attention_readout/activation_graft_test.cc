#include "src/llm/experiments/memorize_general_facts/fit_attention_readout/activation_graft.h"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "gtest/gtest.h"
#include "src/cuda/page_locked_host_array.h"
#include "src/util/status_macros.h"

namespace pluto::llm::fit_attention_readout {
namespace {

class ActivationGraftTest : public testing::Test {
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
        "upload BF16 graft fixture"));
    return device;
  }

  absl::StatusOr<std::vector<uint16_t>> Download(const cuda::Buffer& buffer) {
    ASSIGN_OR_RETURN(auto host,
                     cuda::PageLockedHostArray<uint16_t>::Allocate(
                         *executor_, buffer.size_bytes() / sizeof(uint16_t)));
    RETURN_IF_ERROR(cuda::CudaStatus(
        cudaMemcpyAsync(host.data(), buffer.data(), host.size_bytes(),
                        cudaMemcpyDeviceToHost, executor_->stream()),
        "download BF16 graft fixture"));
    RETURN_IF_ERROR(executor_->Synchronize());
    return std::vector<uint16_t>(host.begin(), host.end());
  }

  std::unique_ptr<cuda::Executor> executor_;
};

TEST_F(ActivationGraftTest, CopiesOnlyFirstTwoDimensionsOfEveryRowExactly) {
  // Include an odd width, and both a complete and a shorter batch's rows.
  for (const int width : {7, 10}) {
    SCOPED_TRACE(width);
    for (const int rows : {5, 3}) {
      SCOPED_TRACE(rows);
      std::vector<uint16_t> base_values(rows * width);
      std::vector<uint16_t> donor_values(rows * width);
      for (size_t i = 0; i < base_values.size(); ++i) {
        base_values[i] = 0x3f00 + i;
        donor_values[i] = 0xc100 + i;
      }
      // Raw bits preserve signed zero and NaN payloads as well as finite data.
      base_values[0] = 0x8000;
      donor_values[0] = 0x0000;
      base_values[2] = 0x7fc1;
      donor_values[width + 1] = 0xffc3;
      auto base = Upload(base_values);
      auto donor = Upload(donor_values);
      ASSERT_TRUE(base.ok()) << base.status();
      ASSERT_TRUE(donor.ok()) << donor.status();
      auto graft =
          GraftBf16LeadingDimensions(*executor_, *base, *donor, width, 2);
      ASSERT_TRUE(graft.ok()) << graft.status();
      EXPECT_NE(graft->data(), base->data());
      EXPECT_NE(graft->data(), donor->data());
      EXPECT_EQ(graft->size_bytes(), base->size_bytes());
      EXPECT_EQ(&graft->executor(), executor_.get());
      auto actual = Download(*graft);
      auto base_after = Download(*base);
      auto donor_after = Download(*donor);
      ASSERT_TRUE(actual.ok()) << actual.status();
      ASSERT_TRUE(base_after.ok()) << base_after.status();
      ASSERT_TRUE(donor_after.ok()) << donor_after.status();
      EXPECT_EQ(*base_after, base_values);
      EXPECT_EQ(*donor_after, donor_values);
      for (int row = 0; row < rows; ++row) {
        for (int column = 0; column < width; ++column) {
          const int i = row * width + column;
          EXPECT_EQ((*actual)[i], column < 2 ? donor_values[i] : base_values[i])
              << "row " << row << ", column " << column;
        }
      }
    }
  }
}

TEST_F(ActivationGraftTest, CopiesScatteredColumnsExactlyInEitherOrder) {
  constexpr int kWidth = 10;
  for (const int rows : {5, 3}) {
    SCOPED_TRACE(rows);
    std::vector<uint16_t> base_values(rows * kWidth);
    std::vector<uint16_t> donor_values(rows * kWidth);
    for (size_t i = 0; i < base_values.size(); ++i) {
      base_values[i] = 0x3f00 + i;
      donor_values[i] = 0xc100 + i;
    }
    // Preserve exact signed-zero and NaN bits in selected and untouched
    // columns.
    base_values[0] = 0x7fc1;
    base_values[1] = 0x0000;
    donor_values[1] = 0x8000;
    donor_values[kWidth + 8] = 0xffc3;
    auto base = Upload(base_values);
    auto donor = Upload(donor_values);
    ASSERT_TRUE(base.ok()) << base.status();
    ASSERT_TRUE(donor.ok()) << donor.status();
    auto forward =
        GraftBf16Dimensions(*executor_, *base, *donor, kWidth, {1, 8});
    auto reversed =
        GraftBf16Dimensions(*executor_, *base, *donor, kWidth, {8, 1});
    ASSERT_TRUE(forward.ok()) << forward.status();
    ASSERT_TRUE(reversed.ok()) << reversed.status();
    EXPECT_NE(forward->data(), base->data());
    EXPECT_NE(forward->data(), donor->data());
    EXPECT_NE(reversed->data(), forward->data());
    EXPECT_EQ(forward->size_bytes(), base->size_bytes());
    EXPECT_EQ(&forward->executor(), executor_.get());
    auto actual = Download(*forward);
    auto reversed_values = Download(*reversed);
    auto base_after = Download(*base);
    auto donor_after = Download(*donor);
    ASSERT_TRUE(actual.ok()) << actual.status();
    ASSERT_TRUE(reversed_values.ok()) << reversed_values.status();
    ASSERT_TRUE(base_after.ok()) << base_after.status();
    ASSERT_TRUE(donor_after.ok()) << donor_after.status();
    EXPECT_EQ(*actual, *reversed_values);
    EXPECT_EQ(*base_after, base_values);
    EXPECT_EQ(*donor_after, donor_values);
    for (size_t i = 0; i < base_values.size(); ++i) {
      const size_t column = i % kWidth;
      EXPECT_EQ((*actual)[i],
                column == 1 || column == 8 ? donor_values[i] : base_values[i])
          << "element " << i;
    }
  }
}

TEST_F(ActivationGraftTest,
       CopiesFirstAndLastColumnsAndSeparateContiguousGroups) {
  constexpr int kWidth = 7;
  const std::vector<uint16_t> base_values = {
      0x0000, 0x8000, 0x7fc1, 0xffc2, 0x7f80, 0xff80, 0x3f80,
      0x4000, 0xc000, 0x4040, 0xc040, 0x4080, 0xc080, 0x40a0};
  const std::vector<uint16_t> donor_values = {
      0x8000, 0x0000, 0xffc3, 0x7fc4, 0xff80, 0x7f80, 0xbf80,
      0xc000, 0x4000, 0xc040, 0x4040, 0xc080, 0x4080, 0xc0a0};
  auto base = Upload(base_values);
  auto donor = Upload(donor_values);
  ASSERT_TRUE(base.ok()) << base.status();
  ASSERT_TRUE(donor.ok()) << donor.status();
  for (const auto& dimensions :
       std::vector<std::vector<int>>{{0, 6}, {6, 0, 5, 1}}) {
    auto graft =
        GraftBf16Dimensions(*executor_, *base, *donor, kWidth, dimensions);
    ASSERT_TRUE(graft.ok()) << graft.status();
    auto actual = Download(*graft);
    ASSERT_TRUE(actual.ok()) << actual.status();
    auto expected = base_values;
    for (size_t row = 0; row < base_values.size() / kWidth; ++row)
      for (const int column : dimensions)
        expected[row * kWidth + column] = donor_values[row * kWidth + column];
    EXPECT_EQ(*actual, expected);
  }
}

TEST_F(ActivationGraftTest, LeadingDimensionsMatchesExplicitSelection) {
  constexpr int kWidth = 3;
  const std::vector<uint16_t> base_values = {0x3f00, 0x8000, 0x7fc1,
                                             0x3f80, 0x4000, 0x7f80};
  const std::vector<uint16_t> donor_values = {0xbf00, 0x0000, 0xffc2,
                                              0xbf80, 0xc000, 0xff80};
  auto base = Upload(base_values);
  auto donor = Upload(donor_values);
  ASSERT_TRUE(base.ok()) << base.status();
  ASSERT_TRUE(donor.ok()) << donor.status();
  for (int count = 1; count <= kWidth; ++count) {
    SCOPED_TRACE(count);
    std::vector<int> dimensions(count);
    std::iota(dimensions.begin(), dimensions.end(), 0);
    auto leading =
        GraftBf16LeadingDimensions(*executor_, *base, *donor, kWidth, count);
    auto explicit_selection =
        GraftBf16Dimensions(*executor_, *base, *donor, kWidth, dimensions);
    ASSERT_TRUE(leading.ok()) << leading.status();
    ASSERT_TRUE(explicit_selection.ok()) << explicit_selection.status();
    auto leading_values = Download(*leading);
    auto explicit_values = Download(*explicit_selection);
    ASSERT_TRUE(leading_values.ok()) << leading_values.status();
    ASSERT_TRUE(explicit_values.ok()) << explicit_values.status();
    EXPECT_EQ(*leading_values, *explicit_values);
  }
}

TEST_F(ActivationGraftTest, SupportsSingleDimensionAndFullWidth) {
  constexpr int kWidth = 3;
  const std::vector<uint16_t> base_values = {0x3f00, 0x8000, 0x7fc1,
                                             0x3f80, 0x4000, 0x7f80};
  const std::vector<uint16_t> donor_values = {0xbf00, 0x0000, 0xffc2,
                                              0xbf80, 0xc000, 0xff80};
  auto base = Upload(base_values);
  auto donor = Upload(donor_values);
  ASSERT_TRUE(base.ok()) << base.status();
  ASSERT_TRUE(donor.ok()) << donor.status();
  for (const int dimensions : {1, kWidth}) {
    SCOPED_TRACE(dimensions);
    auto graft = GraftBf16LeadingDimensions(*executor_, *base, *donor, kWidth,
                                            dimensions);
    ASSERT_TRUE(graft.ok()) << graft.status();
    EXPECT_NE(graft->data(), base->data());
    EXPECT_NE(graft->data(), donor->data());
    auto actual = Download(*graft);
    ASSERT_TRUE(actual.ok()) << actual.status();
    for (size_t i = 0; i < base_values.size(); ++i)
      EXPECT_EQ((*actual)[i], i % kWidth < static_cast<size_t>(dimensions)
                                  ? donor_values[i]
                                  : base_values[i]);
  }
}

TEST_F(ActivationGraftTest, AliasedInputsStillProduceIndependentOutputs) {
  const std::vector<uint16_t> values = {0x0000, 0x8000, 0x7fc1,
                                        0xffc2, 0x7f80, 0xff80};
  auto input = Upload(values);
  ASSERT_TRUE(input.ok()) << input.status();
  auto first = GraftBf16LeadingDimensions(*executor_, *input, *input, 3, 2);
  auto second = GraftBf16Dimensions(*executor_, *input, *input, 3, {0, 2});
  ASSERT_TRUE(first.ok()) << first.status();
  ASSERT_TRUE(second.ok()) << second.status();
  EXPECT_NE(first->data(), input->data());
  EXPECT_NE(second->data(), input->data());
  EXPECT_NE(first->data(), second->data());
  auto first_values = Download(*first);
  ASSERT_TRUE(first_values.ok()) << first_values.status();
  EXPECT_EQ(*first_values, values);
  ASSERT_EQ(cudaMemsetAsync(first->data(), 0, first->size_bytes(),
                            executor_->stream()),
            cudaSuccess);
  auto input_after = Download(*input);
  auto second_after = Download(*second);
  ASSERT_TRUE(input_after.ok()) << input_after.status();
  ASSERT_TRUE(second_after.ok()) << second_after.status();
  EXPECT_EQ(*input_after, values);
  EXPECT_EQ(*second_after, values);
}

TEST_F(ActivationGraftTest, RejectsInvalidDimensionsAndWidth) {
  auto input = Upload({0x3f00, 0x3f80, 0x4000});
  ASSERT_TRUE(input.ok()) << input.status();
  for (const int dimensions : {-1, 0, 4}) {
    SCOPED_TRACE(dimensions);
    EXPECT_TRUE(absl::IsInvalidArgument(
        GraftBf16LeadingDimensions(*executor_, *input, *input, 3, dimensions)
            .status()));
  }
  for (const int width : {-1, 0}) {
    SCOPED_TRACE(width);
    EXPECT_TRUE(absl::IsInvalidArgument(
        GraftBf16LeadingDimensions(*executor_, *input, *input, width, 1)
            .status()));
  }
}

TEST_F(ActivationGraftTest, RejectsEmptyDuplicateAndOutOfRangeSelections) {
  auto input = Upload({0x3f00, 0x3f80, 0x4000});
  ASSERT_TRUE(input.ok()) << input.status();
  for (const auto& dimensions : std::vector<std::vector<int>>{
           {}, {1, 1}, {2, 0, 2}, {-1}, {3}, {0, 3}, {1, -1}}) {
    SCOPED_TRACE(testing::PrintToString(dimensions));
    EXPECT_TRUE(absl::IsInvalidArgument(
        GraftBf16Dimensions(*executor_, *input, *input, 3, dimensions)
            .status()));
  }
  for (const int width : {-1, 0}) {
    SCOPED_TRACE(width);
    EXPECT_TRUE(absl::IsInvalidArgument(
        GraftBf16Dimensions(*executor_, *input, *input, width, {0}).status()));
  }
}

TEST_F(ActivationGraftTest, RejectsEmptyUnequalAndIncompleteRows) {
  auto empty = cuda::Buffer::Allocate(*executor_, 0);
  auto whole = cuda::Buffer::Allocate(*executor_, 20);
  auto shorter = cuda::Buffer::Allocate(*executor_, 18);
  auto odd_bytes = cuda::Buffer::Allocate(*executor_, 19);
  ASSERT_TRUE(empty.ok()) << empty.status();
  ASSERT_TRUE(whole.ok()) << whole.status();
  ASSERT_TRUE(shorter.ok()) << shorter.status();
  ASSERT_TRUE(odd_bytes.ok()) << odd_bytes.status();
  EXPECT_TRUE(absl::IsInvalidArgument(
      GraftBf16Dimensions(*executor_, *empty, *empty, 10, {1, 8}).status()));
  EXPECT_TRUE(absl::IsInvalidArgument(
      GraftBf16Dimensions(*executor_, *whole, *shorter, 10, {1, 8}).status()));
  EXPECT_TRUE(absl::IsInvalidArgument(
      GraftBf16Dimensions(*executor_, *shorter, *whole, 10, {1, 8}).status()));
  EXPECT_TRUE(absl::IsInvalidArgument(
      GraftBf16Dimensions(*executor_, *shorter, *shorter, 10, {1, 8})
          .status()));
  EXPECT_TRUE(absl::IsInvalidArgument(
      GraftBf16Dimensions(*executor_, *odd_bytes, *odd_bytes, 10, {1, 8})
          .status()));
}

TEST_F(ActivationGraftTest, RejectsEitherInputOnAnotherExecutor) {
  auto other_executor = cuda::Executor::Create();
  ASSERT_TRUE(other_executor.ok()) << other_executor.status();
  auto local = cuda::Buffer::Allocate(*executor_, 20);
  auto foreign = cuda::Buffer::Allocate(**other_executor, 20);
  ASSERT_TRUE(local.ok()) << local.status();
  ASSERT_TRUE(foreign.ok()) << foreign.status();
  EXPECT_TRUE(absl::IsInvalidArgument(
      GraftBf16Dimensions(*executor_, *local, *foreign, 10, {1, 8}).status()));
  EXPECT_TRUE(absl::IsInvalidArgument(
      GraftBf16Dimensions(*executor_, *foreign, *local, 10, {1, 8}).status()));
  EXPECT_TRUE(absl::IsInvalidArgument(
      GraftBf16Dimensions(*executor_, *foreign, *foreign, 10, {1, 8})
          .status()));
  EXPECT_TRUE((*other_executor)->Synchronize().ok());
}

}  // namespace
}  // namespace pluto::llm::fit_attention_readout
