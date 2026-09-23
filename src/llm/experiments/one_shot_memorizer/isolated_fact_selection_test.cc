#include "src/llm/experiments/one_shot_memorizer/isolated_fact_selection.h"

#include <cstddef>
#include <limits>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "gtest/gtest.h"

namespace pluto::llm::one_shot_memorizer {
namespace {

TEST(IsolatedFactSelectionTest,
     SortsByOriginalLineAndJoinsWithoutFinalNewline) {
  auto selection =
      SelectIsolatedFacts({"first", "second", "third"}, {"3", "1"});
  ASSERT_TRUE(selection.ok()) << selection.status();
  EXPECT_EQ(selection->corpus_indices, (std::vector<size_t>{0, 2}));
  EXPECT_EQ(selection->text, "first\nthird");
  auto first = selection->CorpusIndex(0);
  auto second = selection->CorpusIndex(1);
  ASSERT_TRUE(first.ok()) << first.status();
  ASSERT_TRUE(second.ok()) << second.status();
  EXPECT_EQ(*first, 0u);
  EXPECT_EQ(*second, 2u);
}

TEST(IsolatedFactSelectionTest, PreservesSpacesAndTrailingCrBytes) {
  auto selection =
      SelectIsolatedFacts({"  first\r", "second \t\r"}, {"1", "2"});
  ASSERT_TRUE(selection.ok()) << selection.status();
  EXPECT_EQ(selection->text, "  first\r\nsecond \t\r");
}

TEST(IsolatedFactSelectionTest, RetainsIdenticalTextAtDifferentLineIds) {
  auto selection = SelectIsolatedFacts({"same", "same"}, {"2", "1"});
  ASSERT_TRUE(selection.ok()) << selection.status();
  EXPECT_EQ(selection->corpus_indices, (std::vector<size_t>{0, 1}));
  EXPECT_EQ(selection->text, "same\nsame");
}

TEST(IsolatedFactSelectionTest, RejectsEmbeddedNewlinesOnlyInSelectedEntries) {
  EXPECT_EQ(
      SelectIsolatedFacts({"first", "second\nthird"}, {"2"}).status().code(),
      absl::StatusCode::kInvalidArgument);
  auto selection = SelectIsolatedFacts({"first", "second\nthird"}, {"1"});
  ASSERT_TRUE(selection.ok()) << selection.status();
  EXPECT_EQ(selection->text, "first");
}

TEST(IsolatedFactSelectionTest, RejectsEmptySelectionAndCorpus) {
  EXPECT_EQ(SelectIsolatedFacts({"first"}, {}).status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(SelectIsolatedFacts({}, {"1"}).status().code(),
            absl::StatusCode::kInvalidArgument);
}

TEST(IsolatedFactSelectionTest, RejectsMalformedZeroAndOutOfRangeNumbers) {
  for (const char* number :
       {"", "0", "-1", "+1", "3", "1.0", "one", "1x", " 1", "1 ",
        "184467440737095516160000000000000000000000"}) {
    SCOPED_TRACE(number);
    EXPECT_EQ(
        SelectIsolatedFacts({"first", "second"}, {number}).status().code(),
        absl::StatusCode::kInvalidArgument);
  }
}

TEST(IsolatedFactSelectionTest, RejectsDuplicateLineIdentities) {
  EXPECT_EQ(
      SelectIsolatedFacts({"first", "second"}, {"1", "1"}).status().code(),
      absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(
      SelectIsolatedFacts({"first", "second"}, {"1", "01"}).status().code(),
      absl::StatusCode::kInvalidArgument);
}

TEST(IsolatedFactSelectionTest, RejectsOnlySelectedBlankLines) {
  for (const char* blank : {"", " ", "\t\r", " \n\r\t"}) {
    SCOPED_TRACE(blank);
    EXPECT_EQ(SelectIsolatedFacts({"first", blank}, {"2"}).status().code(),
              absl::StatusCode::kInvalidArgument);
    auto selection = SelectIsolatedFacts({"first", blank}, {"1"});
    ASSERT_TRUE(selection.ok()) << selection.status();
    EXPECT_EQ(selection->text, "first");
  }
}

TEST(IsolatedFactSelectionTest, BoundsChecksLocalSampleIndices) {
  auto selection = SelectIsolatedFacts({"first", "second"}, {"2"});
  ASSERT_TRUE(selection.ok()) << selection.status();
  EXPECT_EQ(selection->CorpusIndex(1).status().code(),
            absl::StatusCode::kOutOfRange);
  EXPECT_EQ(selection->CorpusIndex(std::numeric_limits<size_t>::max())
                .status()
                .code(),
            absl::StatusCode::kOutOfRange);
  EXPECT_EQ(IsolatedFactSelection{}.CorpusIndex(0).status().code(),
            absl::StatusCode::kOutOfRange);
}

}  // namespace
}  // namespace pluto::llm::one_shot_memorizer
