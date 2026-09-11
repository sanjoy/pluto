#include "ai-slop/weight_analysis/block_ablation_arms.h"

#include <algorithm>
#include <array>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "gtest/gtest.h"

namespace pluto::weight_analysis {
namespace {

// Literal checkpoint coordinates make these tests independent of the indexing
// arithmetic in the schedule builder. Each pair is output matrix plus bias.
constexpr std::array<std::array<int, 2>, 8> kAttention{{
    {6, 7},
    {18, 19},
    {30, 31},
    {42, 43},
    {54, 55},
    {66, 67},
    {78, 79},
    {90, 91},
}};
constexpr std::array<std::array<int, 2>, 8> kMlp{{
    {12, 13},
    {24, 25},
    {36, 37},
    {48, 49},
    {60, 61},
    {72, 73},
    {84, 85},
    {96, 97},
}};

std::set<int> Expected(int first, int end, bool attention, bool mlp) {
  std::set<int> result;
  for (int block = first; block < end; ++block) {
    if (attention)
      result.insert(kAttention[block].begin(), kAttention[block].end());
    if (mlp) result.insert(kMlp[block].begin(), kMlp[block].end());
  }
  return result;
}

std::map<std::string, std::set<int>> ArmIndices() {
  std::map<std::string, std::set<int>> result;
  for (const auto& arm : BlockAblationArms()) {
    result.emplace(arm.name, std::set<int>(arm.weight_indices.begin(),
                                           arm.weight_indices.end()));
  }
  return result;
}

TEST(BlockAblationArmsTest, FrozenOrderAndCleanAnchors) {
  const std::vector<std::string> expected{
      "clean_before",       "clean_repeat",    "drop_b0_attention",
      "drop_b0_mlp",        "drop_b1",         "drop_b1_attention",
      "drop_b1_mlp",        "drop_b2",         "drop_b2_attention",
      "drop_b2_mlp",        "drop_b3",         "drop_b3_attention",
      "drop_b3_mlp",        "drop_b4",         "drop_b4_attention",
      "drop_b4_mlp",        "drop_b5",         "drop_b5_attention",
      "drop_b5_mlp",        "drop_b6",         "drop_b6_attention",
      "drop_b6_mlp",        "drop_b7",         "drop_b7_attention",
      "drop_b7_mlp",        "keep_through_b0", "keep_through_b1",
      "keep_through_b2",    "keep_through_b3", "keep_through_b4",
      "keep_through_b5",    "keep_through_b6", "drop_later_attention",
      "drop_later_mlp",     "isolated_b0",     "b0_mlp_with_positions",
      "drop_all_attention", "drop_all_mlp",    "clean_after"};
  const auto arms = BlockAblationArms();
  ASSERT_EQ(arms.size(), 39u);
  ASSERT_EQ(expected.size(), arms.size());
  std::set<std::string> names;
  for (size_t index = 0; index < arms.size(); ++index) {
    const auto& arm = arms[index];
    EXPECT_EQ(arm.name, expected[index]);
    EXPECT_TRUE(names.insert(arm.name).second);
    if (index == 0 || index == 1 || index + 1 == arms.size()) {
      EXPECT_TRUE(arm.weight_indices.empty());
      EXPECT_EQ(arm.scale, 1.0f);
    } else {
      EXPECT_FALSE(arm.weight_indices.empty());
      EXPECT_EQ(arm.scale, 0.0f);
    }
  }
}

TEST(BlockAblationArmsTest, IndependentBlocksUseOutputMatricesAndBiases) {
  const auto arms = ArmIndices();
  EXPECT_EQ(arms.at("drop_b0_attention"), (std::set<int>{6, 7}));
  EXPECT_EQ(arms.at("drop_b0_mlp"), (std::set<int>{12, 13}));
  for (int block = 1; block < 8; ++block) {
    const std::string name = "drop_b" + std::to_string(block);
    EXPECT_EQ(arms.at(name), Expected(block, block + 1, true, true));
    EXPECT_EQ(arms.at(name + "_attention"),
              Expected(block, block + 1, true, false));
    EXPECT_EQ(arms.at(name + "_mlp"), Expected(block, block + 1, false, true));
  }
  EXPECT_EQ(arms.at("drop_b1"), (std::set<int>{18, 19, 24, 25}));
  EXPECT_EQ(arms.at("drop_b7"), (std::set<int>{90, 91, 96, 97}));
}

TEST(BlockAblationArmsTest, CumulativeStopsDisableExactlyTheFollowingBlocks) {
  const auto arms = ArmIndices();
  for (int last = 0; last < 7; ++last) {
    const auto& removed = arms.at("keep_through_b" + std::to_string(last));
    EXPECT_EQ(removed, Expected(last + 1, 8, true, true));
    EXPECT_EQ(removed.size(), static_cast<size_t>(7 - last) * 4);
    for (int retained : Expected(0, last + 1, true, true)) {
      EXPECT_EQ(removed.count(retained), 0u);
    }
  }
}

TEST(BlockAblationArmsTest, GroupAndIsolationArmsHaveExactSets) {
  const auto arms = ArmIndices();
  EXPECT_EQ(arms.at("drop_later_attention"), Expected(1, 8, true, false));
  EXPECT_EQ(arms.at("drop_later_mlp"), Expected(1, 8, false, true));
  EXPECT_EQ(arms.at("drop_all_attention"), Expected(0, 8, true, false));
  EXPECT_EQ(arms.at("drop_all_mlp"), Expected(0, 8, false, true));

  auto isolated = Expected(1, 8, true, true);
  isolated.insert({6, 7});
  EXPECT_EQ(arms.at("b0_mlp_with_positions"), isolated);
  EXPECT_EQ(isolated.size(), 30u);
  isolated.insert(1);
  EXPECT_EQ(arms.at("isolated_b0"), isolated);
  EXPECT_EQ(isolated.size(), 31u);
  for (const std::string name : {"isolated_b0", "b0_mlp_with_positions"}) {
    for (int b0_mlp : {8, 9, 10, 11, 12, 13}) {
      EXPECT_EQ(arms.at(name).count(b0_mlp), 0u);
    }
  }
}

TEST(BlockAblationArmsTest, EveryArmHasUniqueValidIndicesAndProtectsReadout) {
  auto allowed = Expected(0, 8, true, true);
  allowed.insert(1);
  for (const auto& arm : BlockAblationArms()) {
    SCOPED_TRACE(arm.name);
    const std::set<int> indices(arm.weight_indices.begin(),
                                arm.weight_indices.end());
    EXPECT_EQ(indices.size(), arm.weight_indices.size());
    EXPECT_TRUE(
        std::is_sorted(arm.weight_indices.begin(), arm.weight_indices.end()));
    for (int index : indices) {
      EXPECT_GE(index, 0);
      EXPECT_LT(index, 100);
      EXPECT_EQ(allowed.count(index), 1u);
      EXPECT_NE(index, 0);
      EXPECT_NE(index, 98);
      EXPECT_NE(index, 99);
      if (index == 1) {
        EXPECT_EQ(arm.name, "isolated_b0");
      }
    }
  }
}

TEST(BlockAblationArmsTest, ReturnedPlansAndCleanArmsAreIndependent) {
  auto first = BlockAblationArms();
  first[0].weight_indices.push_back(99);
  first[2].weight_indices[0] = 0;
  first.back().scale = 0.0f;
  EXPECT_TRUE(first[1].weight_indices.empty());
  EXPECT_EQ(first[1].scale, 1.0f);
  const auto second = BlockAblationArms();
  EXPECT_TRUE(second.front().weight_indices.empty());
  EXPECT_TRUE(second.back().weight_indices.empty());
  EXPECT_EQ(second.front().scale, 1.0f);
  EXPECT_EQ(second.back().scale, 1.0f);
  EXPECT_EQ(second[2].weight_indices, (std::vector<int>{6, 7}));
}

}  // namespace
}  // namespace pluto::weight_analysis
