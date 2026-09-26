#include "src/llm/experiments/memorize_general_facts/fit_attention_readout/subspace_graft.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "gtest/gtest.h"
#include "src/cuda/page_locked_host_array.h"
#include "src/util/status_macros.h"

namespace pluto::llm::fit_attention_readout {
namespace {

float FromBf16(uint16_t value) {
  const uint32_t bits = static_cast<uint32_t>(value) << 16;
  float result;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

uint16_t ToBf16(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  // Test inputs and reference results are finite; round to nearest, ties even.
  bits += 0x7fff + ((bits >> 16) & 1);
  return static_cast<uint16_t>(bits >> 16);
}

std::vector<float> CanonicalPlane(int width) {
  std::vector<float> plane(2 * width, 0.0f);
  plane[0] = plane[3] = 1.0f;
  return plane;
}

std::vector<float> RotatedPlane(int width) {
  // A constant vector and a centered ramp span a non-coordinate plane and
  // exercise every input channel in the reductions.
  std::vector<float> plane(2 * width);
  double ramp_norm = 0.0;
  for (int column = 0; column < width; ++column) {
    const double ramp = column - 0.5 * (width - 1);
    ramp_norm += ramp * ramp;
  }
  for (int column = 0; column < width; ++column) {
    plane[2 * column] = 1.0 / std::sqrt(static_cast<double>(width));
    plane[2 * column + 1] = (column - 0.5 * (width - 1)) / std::sqrt(ramp_norm);
  }
  return plane;
}

std::vector<uint16_t> Reference(const std::vector<uint16_t>& base,
                                const std::vector<uint16_t>& donor, int width,
                                const std::vector<float>& plane) {
  std::vector<uint16_t> output(base.size());
  for (size_t row = 0; row < base.size() / width; ++row) {
    float projections[2] = {0.0f, 0.0f};
    for (int column = 0; column < width; ++column) {
      const size_t index = row * width + column;
      const float delta = FromBf16(donor[index]) - FromBf16(base[index]);
      for (int axis = 0; axis < 2; ++axis)
        projections[axis] += plane[2 * column + axis] * delta;
    }
    for (int column = 0; column < width; ++column) {
      const size_t index = row * width + column;
      const float correction = plane[2 * column] * projections[0] +
                               plane[2 * column + 1] * projections[1];
      output[index] = correction == 0.0f
                          ? base[index]
                          : ToBf16(FromBf16(base[index]) + correction);
    }
  }
  return output;
}

class SubspaceGraftTest : public testing::Test {
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
        "upload plane graft fixture"));
    return device;
  }

  absl::StatusOr<std::vector<uint16_t>> Download(const cuda::Buffer& buffer) {
    ASSIGN_OR_RETURN(auto host,
                     cuda::PageLockedHostArray<uint16_t>::Allocate(
                         *executor_, buffer.size_bytes() / sizeof(uint16_t)));
    RETURN_IF_ERROR(cuda::CudaStatus(
        cudaMemcpyAsync(host.data(), buffer.data(), host.size_bytes(),
                        cudaMemcpyDeviceToHost, executor_->stream()),
        "download plane graft fixture"));
    RETURN_IF_ERROR(executor_->Synchronize());
    return std::vector<uint16_t>(host.begin(), host.end());
  }

  std::unique_ptr<cuda::Executor> executor_;
};

TEST_F(SubspaceGraftTest, CanonicalPlaneCopiesFirstTwoColumnsExactly) {
  for (const int width : {2, 7, 128, 129, 1024}) {
    SCOPED_TRACE(width);
    constexpr int kRows = 7;
    std::vector<uint16_t> base_values(kRows * width);
    std::vector<uint16_t> donor_values(kRows * width);
    for (size_t i = 0; i < base_values.size(); ++i) {
      base_values[i] = ToBf16((static_cast<int>(i % 37) - 18) / 8.0f);
      donor_values[i] = ToBf16((static_cast<int>(i % 29) - 14) / 16.0f);
    }
    if (width > 2)
      base_values[2] = 0x8000;  // Retained negative zero stays bit-exact.
    auto base = Upload(base_values);
    auto donor = Upload(donor_values);
    ASSERT_TRUE(base.ok()) << base.status();
    ASSERT_TRUE(donor.ok()) << donor.status();
    auto graft =
        GraftBf16Plane(*executor_, *base, *donor, width, CanonicalPlane(width));
    ASSERT_TRUE(graft.ok()) << graft.status();
    EXPECT_NE(graft->data(), base->data());
    EXPECT_NE(graft->data(), donor->data());
    EXPECT_EQ(graft->size_bytes(), base->size_bytes());
    EXPECT_EQ(&graft->executor(), executor_.get());
    auto actual = Download(*graft);
    ASSERT_TRUE(actual.ok()) << actual.status();
    for (size_t i = 0; i < base_values.size(); ++i)
      EXPECT_EQ((*actual)[i], i % width < 2 ? donor_values[i] : base_values[i])
          << "index " << i;
    auto base_after = Download(*base);
    auto donor_after = Download(*donor);
    ASSERT_TRUE(base_after.ok()) << base_after.status();
    ASSERT_TRUE(donor_after.ok()) << donor_after.status();
    EXPECT_EQ(*base_after, base_values);
    EXPECT_EQ(*donor_after, donor_values);
  }
}

TEST_F(SubspaceGraftTest, RotatedPlaneMatchesScalarBf16Reference) {
  for (const int width : {4, 7, 128, 129, 1024}) {
    SCOPED_TRACE(width);
    // Seven rows also exercise the final partially populated row tile.
    std::vector<uint16_t> base_values(7 * width);
    std::vector<uint16_t> donor_values(7 * width);
    for (size_t i = 0; i < base_values.size(); ++i) {
      base_values[i] = ToBf16((static_cast<int>(i * 7 % 37) - 18) / 8.0f);
      donor_values[i] = ToBf16((static_cast<int>(i * 11 % 29) - 14) / 16.0f);
    }
    auto plane = RotatedPlane(width);
    const auto expected = Reference(base_values, donor_values, width, plane);
    auto base = Upload(base_values);
    auto donor = Upload(donor_values);
    ASSERT_TRUE(base.ok()) << base.status();
    ASSERT_TRUE(donor.ok()) << donor.status();
    auto graft = GraftBf16Plane(*executor_, *base, *donor, width, plane);
    ASSERT_TRUE(graft.ok()) << graft.status();
    // The host basis may be reused immediately after the asynchronous call.
    std::fill(plane.begin(), plane.end(),
              std::numeric_limits<float>::quiet_NaN());
    auto actual = Download(*graft);
    ASSERT_TRUE(actual.ok()) << actual.status();
    for (size_t i = 0; i < expected.size(); ++i) {
      const float want = FromBf16(expected[i]);
      const float got = FromBf16((*actual)[i]);
      EXPECT_TRUE(std::isfinite(got)) << i;
      // Tree reductions can round differently at a BF16 midpoint; allow at
      // most one BF16 step while comparing the straightforward scalar sum.
      EXPECT_NEAR(got, want, 0.0078125f * std::max(1.0f, std::abs(want))) << i;
    }
  }
}

TEST_F(SubspaceGraftTest, IdenticalInputsPreserveValuesAndOwnTheirStorage) {
  constexpr int kWidth = 7;
  std::vector<uint16_t> values(3 * kWidth);
  for (size_t i = 0; i < values.size(); ++i)
    values[i] = ToBf16((static_cast<int>(i) - 11) / 4.0f);
  values[0] = 0x8000;
  auto input = Upload(values);
  ASSERT_TRUE(input.ok()) << input.status();
  auto graft =
      GraftBf16Plane(*executor_, *input, *input, kWidth, RotatedPlane(kWidth));
  ASSERT_TRUE(graft.ok()) << graft.status();
  EXPECT_NE(graft->data(), input->data());
  auto actual = Download(*graft);
  ASSERT_TRUE(actual.ok()) << actual.status();
  EXPECT_EQ(*actual, values);
  ASSERT_EQ(cudaMemsetAsync(graft->data(), 0, graft->size_bytes(),
                            executor_->stream()),
            cudaSuccess);
  auto input_after = Download(*input);
  ASSERT_TRUE(input_after.ok()) << input_after.status();
  EXPECT_EQ(*input_after, values);
}

TEST_F(SubspaceGraftTest, RejectsInvalidWidthAndBasis) {
  auto input = Upload({0x3f80, 0x4000, 0x4040, 0x4080});
  ASSERT_TRUE(input.ok()) << input.status();
  for (const int width : {-1, 0, 1, 1025})
    EXPECT_TRUE(absl::IsInvalidArgument(
        GraftBf16Plane(*executor_, *input, *input, width, CanonicalPlane(4))
            .status()));
  for (const size_t size : {0, 7, 9}) {
    auto plane = CanonicalPlane(4);
    plane.resize(size);
    EXPECT_TRUE(absl::IsInvalidArgument(
        GraftBf16Plane(*executor_, *input, *input, 4, plane).status()));
  }
  for (const float invalid :
       {std::numeric_limits<float>::quiet_NaN(),
        std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity(), 0.0f, 0.999f, 2.0f}) {
    auto plane = CanonicalPlane(4);
    plane[0] = invalid;
    EXPECT_TRUE(absl::IsInvalidArgument(
        GraftBf16Plane(*executor_, *input, *input, 4, plane).status()));
  }
  // Individually unit-length columns must also be orthogonal.
  auto parallel = CanonicalPlane(4);
  parallel[1] = 1.0f;
  parallel[3] = 0.0f;
  EXPECT_TRUE(absl::IsInvalidArgument(
      GraftBf16Plane(*executor_, *input, *input, 4, parallel).status()));
}

TEST_F(SubspaceGraftTest, RejectsEmptyUnequalAndIncompleteRows) {
  auto empty = cuda::Buffer::Allocate(*executor_, 0);
  auto whole = cuda::Buffer::Allocate(*executor_, 8);
  auto shorter = cuda::Buffer::Allocate(*executor_, 6);
  auto odd_bytes = cuda::Buffer::Allocate(*executor_, 7);
  ASSERT_TRUE(empty.ok()) << empty.status();
  ASSERT_TRUE(whole.ok()) << whole.status();
  ASSERT_TRUE(shorter.ok()) << shorter.status();
  ASSERT_TRUE(odd_bytes.ok()) << odd_bytes.status();
  const auto plane = CanonicalPlane(4);
  EXPECT_TRUE(absl::IsInvalidArgument(
      GraftBf16Plane(*executor_, *empty, *empty, 4, plane).status()));
  EXPECT_TRUE(absl::IsInvalidArgument(
      GraftBf16Plane(*executor_, *whole, *shorter, 4, plane).status()));
  EXPECT_TRUE(absl::IsInvalidArgument(
      GraftBf16Plane(*executor_, *shorter, *whole, 4, plane).status()));
  EXPECT_TRUE(absl::IsInvalidArgument(
      GraftBf16Plane(*executor_, *shorter, *shorter, 4, plane).status()));
  EXPECT_TRUE(absl::IsInvalidArgument(
      GraftBf16Plane(*executor_, *odd_bytes, *odd_bytes, 4, plane).status()));
}

TEST_F(SubspaceGraftTest, RejectsEitherInputFromAnotherExecutor) {
  auto other_executor = cuda::Executor::Create();
  ASSERT_TRUE(other_executor.ok()) << other_executor.status();
  auto local = cuda::Buffer::Allocate(*executor_, 8);
  auto foreign = cuda::Buffer::Allocate(**other_executor, 8);
  ASSERT_TRUE(local.ok()) << local.status();
  ASSERT_TRUE(foreign.ok()) << foreign.status();
  const auto plane = CanonicalPlane(4);
  EXPECT_TRUE(absl::IsInvalidArgument(
      GraftBf16Plane(*executor_, *local, *foreign, 4, plane).status()));
  EXPECT_TRUE(absl::IsInvalidArgument(
      GraftBf16Plane(*executor_, *foreign, *local, 4, plane).status()));
  EXPECT_TRUE(absl::IsInvalidArgument(
      GraftBf16Plane(*executor_, *foreign, *foreign, 4, plane).status()));
  EXPECT_TRUE((*other_executor)->Synchronize().ok());
}

}  // namespace
}  // namespace pluto::llm::fit_attention_readout
