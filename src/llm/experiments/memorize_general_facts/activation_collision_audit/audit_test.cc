#include "src/llm/experiments/memorize_general_facts/activation_collision_audit/audit.h"

#include <algorithm>
#include <cstdint>
#include <vector>

#include "gtest/gtest.h"

namespace pluto::llm::activation_collision_audit {
namespace {

constexpr uint16_t kOne = 0x3f80;
constexpr uint16_t kTwo = 0x4000;

TEST(AuditTest, EmptyAuditHasNoRowsOrErrors) {
  EXPECT_EQ(Audit(10).Summarize(), Summary{});
}

TEST(AuditTest, EqualVectorsWithEqualTargetsAreNotConflicts) {
  Audit audit(2);
  ASSERT_TRUE(audit.Add({kOne, kTwo}, {0, 4, 0}).ok());
  ASSERT_TRUE(audit.Add({kOne, kTwo}, {1, 7, 0}).ok());
  const auto summary = audit.Summarize();
  EXPECT_EQ(summary.total_rows, 2);
  EXPECT_EQ(summary.unique_vectors, 1);
  EXPECT_EQ(summary.repeated_groups, 1);
  EXPECT_EQ(summary.conflicting_groups, 0);
  EXPECT_EQ(summary.conflicting_rows, 0);
  EXPECT_EQ(summary.minimum_errors, 0);
  EXPECT_TRUE(summary.conflicts.empty());
}

TEST(AuditTest, ConflictingTargetsRetainEveryOccurrenceAndBits) {
  Audit audit(2);
  ASSERT_TRUE(audit.Add({kOne, kTwo}, {8, 4, 17}).ok());
  ASSERT_TRUE(audit.Add({kOne, kTwo}, {2, 9, 0}).ok());
  const auto summary = audit.Summarize();
  EXPECT_EQ(summary.total_rows, 2);
  EXPECT_EQ(summary.unique_vectors, 1);
  EXPECT_EQ(summary.repeated_groups, 1);
  EXPECT_EQ(summary.conflicting_groups, 1);
  EXPECT_EQ(summary.conflicting_rows, 2);
  EXPECT_EQ(summary.minimum_errors, 1);
  ASSERT_EQ(summary.conflicts.size(), 1u);
  EXPECT_EQ(summary.conflicts[0].bits, (std::vector<uint16_t>{kOne, kTwo}));
  EXPECT_EQ(summary.conflicts[0].occurrences,
            (std::vector<Occurrence>{{2, 9, 0}, {8, 4, 17}}));
  EXPECT_EQ(audit.Summarize(), summary);
}

TEST(AuditTest, MinimumErrorsUsesTargetFrequencyNotJustDistinctTargetCount) {
  Audit audit(1);
  int sample = 0;
  for (int target : {3, 3, 3, 5, 5, 8})
    ASSERT_TRUE(audit.Add({kOne}, {sample++, 4, target}).ok());
  for (int target : {7, 7, 7, 7, 9})
    ASSERT_TRUE(audit.Add({kTwo}, {sample++, 4, target}).ok());
  for (int target : {0, 0})
    ASSERT_TRUE(audit.Add({0}, {sample++, 4, target}).ok());
  const auto summary = audit.Summarize();
  EXPECT_EQ(summary.total_rows, 13);
  EXPECT_EQ(summary.unique_vectors, 3);
  EXPECT_EQ(summary.repeated_groups, 3);
  EXPECT_EQ(summary.conflicting_groups, 2);
  EXPECT_EQ(summary.conflicting_rows, 11);
  EXPECT_EQ(summary.minimum_errors, (6 - 3) + (5 - 4));
  EXPECT_EQ(summary.conflicts.size(), 2u);
}

TEST(AuditTest, SignedZerosAndLateCoordinateDifferencesStayDistinct) {
  Audit audit(10);
  std::vector<uint16_t> bits(10, kOne);
  bits.back() = 0;
  ASSERT_TRUE(audit.Add(bits, {0, 4, 1}).ok());
  bits.back() = 0x8000;  // Numerically equal to +0, but different BF16 bits.
  ASSERT_TRUE(audit.Add(bits, {1, 4, 2}).ok());
  bits.back() = 1;  // A finite subnormal; no tolerance-based grouping is used.
  ASSERT_TRUE(audit.Add(bits, {2, 4, 3}).ok());
  EXPECT_EQ(audit.Summarize().unique_vectors, 3);
  EXPECT_EQ(audit.Summarize().minimum_errors, 0);
  EXPECT_TRUE(audit.Summarize().conflicts.empty());
}

TEST(AuditTest, InputMemoryIsCopiedAndIdenticalOccurrencesCountAsRows) {
  Audit audit(1);
  std::vector<uint16_t> bits{kOne};
  ASSERT_TRUE(audit.Add(bits, {0, 4, 7}).ok());
  bits[0] = kTwo;
  ASSERT_TRUE(audit.Add({kOne}, {0, 4, 7}).ok());
  ASSERT_TRUE(audit.Add(bits, {0, 4, 8}).ok());
  const auto summary = audit.Summarize();
  EXPECT_EQ(summary.total_rows, 3);
  EXPECT_EQ(summary.unique_vectors, 2);
  EXPECT_EQ(summary.repeated_groups, 1);
  EXPECT_EQ(summary.conflicting_groups, 0);
}

TEST(AuditTest, ConflictsAreDeterministicAcrossInsertionOrder) {
  struct Row {
    std::vector<uint16_t> bits;
    Occurrence occurrence;
  };
  std::vector<Row> rows = {{{kTwo}, {3, 6, 7}},
                           {{kOne}, {2, 9, 4}},
                           {{kTwo}, {1, 2, 5}},
                           {{kOne}, {2, 8, 8}},
                           {{kOne}, {2, 8, 4}}};
  Audit forward(1), reverse(1);
  for (const auto& row : rows)
    ASSERT_TRUE(forward.Add(row.bits, row.occurrence).ok());
  std::reverse(rows.begin(), rows.end());
  for (const auto& row : rows)
    ASSERT_TRUE(reverse.Add(row.bits, row.occurrence).ok());
  const auto summary = forward.Summarize();
  EXPECT_EQ(summary, reverse.Summarize());
  ASSERT_EQ(summary.conflicts.size(), 2u);
  EXPECT_EQ(summary.conflicts[0].bits, (std::vector<uint16_t>{kOne}));
  EXPECT_EQ(summary.conflicts[0].occurrences,
            (std::vector<Occurrence>{{2, 8, 4}, {2, 8, 8}, {2, 9, 4}}));
}

TEST(AuditTest, RejectsMalformedRowsWithoutChangingState) {
  for (int width : {-1, 0}) {
    Audit invalid(width);
    EXPECT_EQ(invalid.Add({}, {0, 0, 0}).code(),
              absl::StatusCode::kInvalidArgument);
    EXPECT_EQ(invalid.Summarize(), Summary{});
  }
  Audit audit(2);
  ASSERT_TRUE(audit.Add({kOne, kTwo}, {0, 4, 8}).ok());
  const auto before = audit.Summarize();
  for (const auto& bits : std::vector<std::vector<uint16_t>>{{},
                                                             {kOne},
                                                             {kOne, kTwo, 0},
                                                             {0x7f80, 0},
                                                             {0, 0xff80},
                                                             {0x7fc0, 0},
                                                             {0, 0xffff}})
    EXPECT_EQ(audit.Add(bits, {0, 4, 8}).code(),
              absl::StatusCode::kInvalidArgument);
  for (const Occurrence occurrence :
       {Occurrence{-1, 4, 8}, Occurrence{0, -1, 8}, Occurrence{0, 4, -1}})
    EXPECT_EQ(audit.Add({kOne, kTwo}, occurrence).code(),
              absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(audit.Summarize(), before);
}

}  // namespace
}  // namespace pluto::llm::activation_collision_audit
