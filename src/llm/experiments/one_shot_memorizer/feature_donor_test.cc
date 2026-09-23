#include "src/llm/experiments/one_shot_memorizer/feature_donor.h"

#include <algorithm>
#include <cstdint>
#include <vector>

#include "gtest/gtest.h"

namespace pluto::llm::one_shot_memorizer {
namespace {

TEST(FeatureDonorTest, SparseCopiesOnlySelectedDonorBits) {
  const std::vector<uint16_t> recipient{0x3f80, 0x8000, 0x7fc1, 0xff80};
  const std::vector<uint16_t> donor{0x8000, 0x7fc2, 0x7f80, 0x4000};
  auto result = MakeFeatureDonorRow(recipient, donor, 0b0111,
                                    FeatureDonorBackground::kSparse);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(*result, (std::vector<uint16_t>{0x8000, 0x7fc2, 0x7f80, 0}));
  EXPECT_EQ(recipient, (std::vector<uint16_t>{0x3f80, 0x8000, 0x7fc1, 0xff80}));
  EXPECT_EQ(donor, (std::vector<uint16_t>{0x8000, 0x7fc2, 0x7f80, 0x4000}));
}

TEST(FeatureDonorTest, IntactPreservesUnselectedRecipientBits) {
  const std::vector<uint16_t> recipient{0x3f80, 0x8000, 0x7fc1, 0xff80};
  const std::vector<uint16_t> donor{0x8000, 0x7fc2, 0x7f80, 0x4000};
  auto result = MakeFeatureDonorRow(recipient, donor, 0b0101,
                                    FeatureDonorBackground::kIntact);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(*result, (std::vector<uint16_t>{0x8000, 0x8000, 0x7f80, 0xff80}));
}

TEST(FeatureDonorTest, EmptyMaskSelectsBackgroundAndFullMaskSelectsDonor) {
  const std::vector<uint16_t> recipient{1, 2, 3}, donor{4, 5, 6};
  for (auto background :
       {FeatureDonorBackground::kSparse, FeatureDonorBackground::kIntact}) {
    auto empty = MakeFeatureDonorRow(recipient, donor, 0, background);
    ASSERT_TRUE(empty.ok()) << empty.status();
    EXPECT_EQ(*empty, background == FeatureDonorBackground::kSparse
                          ? std::vector<uint16_t>(3, 0)
                          : recipient);
    auto full = MakeFeatureDonorRow(recipient, donor, 0b111, background);
    ASSERT_TRUE(full.ok()) << full.status();
    EXPECT_EQ(*full, donor);
  }
}

TEST(FeatureDonorTest, SelfDonorMatchesIdentityOrOrdinarySparseMask) {
  const std::vector<uint16_t> values{0x8000, 0x7fc1, 0xff80, 0x3f80};
  for (FeatureSubset subset = 0; subset < 16; ++subset) {
    auto intact = MakeFeatureDonorRow(values, values, subset,
                                      FeatureDonorBackground::kIntact);
    ASSERT_TRUE(intact.ok()) << intact.status();
    EXPECT_EQ(*intact, values);
    auto sparse = MakeFeatureDonorRow(values, values, subset,
                                      FeatureDonorBackground::kSparse);
    auto expected = ApplyBf16FeatureSubset(values, subset);
    ASSERT_TRUE(sparse.ok()) << sparse.status();
    ASSERT_TRUE(expected.ok()) << expected.status();
    EXPECT_EQ(*sparse, *expected);
  }
}

TEST(FeatureDonorTest, HighestChannelHasNoShiftOverflow) {
  std::vector<uint16_t> recipient(64, 0x8000), donor(64, 0x7fc1);
  auto intact = MakeFeatureDonorRow(recipient, donor, uint64_t{1} << 63,
                                    FeatureDonorBackground::kIntact);
  ASSERT_TRUE(intact.ok()) << intact.status();
  EXPECT_EQ(intact->back(), 0x7fc1);
  EXPECT_TRUE(std::all_of(intact->begin(), intact->end() - 1,
                          [](uint16_t bits) { return bits == 0x8000; }));
  auto full = MakeFeatureDonorRow(recipient, donor, ~uint64_t{0},
                                  FeatureDonorBackground::kSparse);
  ASSERT_TRUE(full.ok()) << full.status();
  EXPECT_EQ(*full, donor);
}

TEST(FeatureDonorTest, RejectsBadWidthsMasksAndBackground) {
  const std::vector<uint16_t> one{0}, two{0, 0};
  EXPECT_FALSE(
      MakeFeatureDonorRow(one, two, 1, FeatureDonorBackground::kSparse).ok());
  EXPECT_FALSE(
      MakeFeatureDonorRow({}, {}, 0, FeatureDonorBackground::kSparse).ok());
  EXPECT_FALSE(MakeFeatureDonorRow(std::vector<uint16_t>(65),
                                   std::vector<uint16_t>(65), 0,
                                   FeatureDonorBackground::kSparse)
                   .ok());
  EXPECT_FALSE(
      MakeFeatureDonorRow(one, one, 2, FeatureDonorBackground::kIntact).ok());
  EXPECT_FALSE(
      MakeFeatureDonorRow(one, one, 0, static_cast<FeatureDonorBackground>(99))
          .ok());
}

}  // namespace
}  // namespace pluto::llm::one_shot_memorizer
