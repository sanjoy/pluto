#include "src/llm/experiments/memorize_general_facts/puzzle_report.h"

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "gtest/gtest.h"

namespace pluto::llm::memorize_general_facts {
namespace {

PuzzleReportData Example() {
  PuzzleReportData data;
  data.model_width = 3;
  data.context_length = 3;
  data.facts = {"first fact", "second fact"};
  data.points = {
      {0, 0, 10, -1, false, {0, 0, 0}}, {0, 1, 20, 30, false, {1, 2, 3}},
      {0, 2, 0, -1, true, {0, 0, 0}},   {1, 0, 10, -1, false, {0, 0, 0}},
      {1, 1, 21, 31, false, {4, 5, 6}}, {1, 2, 31, 32, false, {7, 8, 9}}};
  return data;
}

void ExpectInvalid(const PuzzleReportData& data) {
  const auto result = VerifyPuzzleSeparation(data);
  EXPECT_EQ(result.status().code(), absl::StatusCode::kInvalidArgument)
      << result.status();
}

TEST(PuzzleSeparationTest, CountsOnlyScoredRows) {
  const auto result = VerifyPuzzleSeparation(Example());
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result->scored_points, 3);
  EXPECT_EQ(result->unique_vectors, 3);
  EXPECT_EQ(result->distinct_targets, 3);
}

TEST(PuzzleSeparationTest, RejectsCollisionAndIdentifiesBothRows) {
  PuzzleReportData data = Example();
  data.points[4].coordinates = data.points[1].coordinates;
  const auto result = VerifyPuzzleSeparation(data);
  EXPECT_EQ(result.status().code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_NE(result.status().message().find("fact 0, position 1 (target 30)"),
            absl::string_view::npos);
  EXPECT_NE(result.status().message().find("fact 1, position 1 (target 31)"),
            absl::string_view::npos);
}

TEST(PuzzleSeparationTest, AllowsDuplicateVectorsWithSameTarget) {
  PuzzleReportData data = Example();
  data.points[4].coordinates = data.points[1].coordinates;
  data.points[4].target_token = data.points[1].target_token;
  const auto result = VerifyPuzzleSeparation(data);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result->scored_points, 3);
  EXPECT_EQ(result->unique_vectors, 2);
  EXPECT_EQ(result->distinct_targets, 2);
}

TEST(PuzzleSeparationTest, CanonicalizesSignedZero) {
  PuzzleReportData data = Example();
  data.points[1].coordinates = {0.0f, -0.0f, 0.0f};
  data.points[4].coordinates = {-0.0f, 0.0f, -0.0f};
  EXPECT_EQ(VerifyPuzzleSeparation(data).status().code(),
            absl::StatusCode::kFailedPrecondition);
  data.points[4].target_token = data.points[1].target_token;
  const auto result = VerifyPuzzleSeparation(data);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result->unique_vectors, 2);
}

TEST(PuzzleSeparationTest, ChecksAcrossPositionsGlobally) {
  PuzzleReportData data = Example();
  data.points[5].coordinates = data.points[1].coordinates;
  const auto result = VerifyPuzzleSeparation(data);
  EXPECT_EQ(result.status().code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_NE(result.status().message().find("fact 0, position 1"),
            absl::string_view::npos);
  EXPECT_NE(result.status().message().find("fact 1, position 2"),
            absl::string_view::npos);
}

TEST(PuzzleSeparationTest, ComparesAllCoordinatesWithoutTolerance) {
  PuzzleReportData data = Example();
  data.points[4].coordinates = data.points[1].coordinates;
  data.points[4].coordinates.back() = std::nextafter(3.0f, 4.0f);
  const auto result = VerifyPuzzleSeparation(data);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result->unique_vectors, 3);
}

TEST(PuzzleSeparationTest, IgnoresMaskedPromptAndPaddingCollisions) {
  PuzzleReportData data = Example();
  data.points[0].coordinates = data.points[1].coordinates;
  data.points[2].coordinates = data.points[4].coordinates;
  data.points[3].coordinates = data.points[5].coordinates;
  const auto result = VerifyPuzzleSeparation(data);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result->scored_points, 3);
  EXPECT_EQ(result->unique_vectors, 3);
}

TEST(PuzzleSeparationTest, AllowsNoScoredRowsWithoutInventingCounts) {
  PuzzleReportData data = Example();
  for (PuzzlePoint& point : data.points)
    point.target_token = -1;
  const auto result = VerifyPuzzleSeparation(data);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result->scored_points, 0);
  EXPECT_EQ(result->unique_vectors, 0);
  EXPECT_EQ(result->distinct_targets, 0);
}

TEST(PuzzleSeparationTest, RejectsNonfiniteEvenOnIgnoredRows) {
  for (float invalid : {std::numeric_limits<float>::quiet_NaN(),
                        std::numeric_limits<float>::infinity(),
                        -std::numeric_limits<float>::infinity()}) {
    for (size_t index : {0, 1, 2}) {
      PuzzleReportData data = Example();
      data.points[index].coordinates.back() = invalid;
      ExpectInvalid(data);
    }
  }
}

TEST(PuzzleSeparationTest, RequiresExactFactPositionCoverage) {
  PuzzleReportData data = Example();
  data.points.pop_back();
  ExpectInvalid(data);
  data = Example();
  data.points.push_back(data.points.back());
  ExpectInvalid(data);
  data = Example();
  data.points.back() = data.points.front();
  ExpectInvalid(data);
  data = Example();
  // Input order is irrelevant as long as coverage is complete.
  std::swap(data.points.front(), data.points.back());
  EXPECT_TRUE(VerifyPuzzleSeparation(data).ok());
}

TEST(PuzzleSeparationTest, RejectsInvalidShapeAndBounds) {
  ExpectInvalid(PuzzleReportData{});
  for (int invalid : {-1, 0}) {
    PuzzleReportData data = Example();
    data.model_width = invalid;
    ExpectInvalid(data);
    data = Example();
    data.context_length = invalid;
    ExpectInvalid(data);
  }
  PuzzleReportData data = Example();
  data.facts.clear();
  ExpectInvalid(data);
  data = Example();
  data.points.front().coordinates.pop_back();
  ExpectInvalid(data);
  data = Example();
  data.points.front().coordinates.push_back(0);
  ExpectInvalid(data);
  for (int invalid : {-1, 2}) {
    data = Example();
    data.points.front().fact_index = invalid;
    ExpectInvalid(data);
  }
  for (int invalid : {-1, 3}) {
    data = Example();
    data.points.front().position = invalid;
    ExpectInvalid(data);
  }
}

TEST(PuzzleSeparationTest, RejectsInvalidTokenAndPaddingMetadata) {
  PuzzleReportData data = Example();
  data.points.front().input_token = -1;
  ExpectInvalid(data);
  data = Example();
  data.points.front().target_token = -2;
  ExpectInvalid(data);
  data = Example();
  data.points[2].target_token = 1;
  ExpectInvalid(data);
}

class PuzzleHtmlTest : public testing::Test {
 protected:
  void SetUp() override {
    std::string pattern =
        (std::filesystem::path(testing::TempDir()) / "puzzle-report.XXXXXX")
            .string();
    std::vector<char> buffer(pattern.begin(), pattern.end());
    buffer.push_back('\0');
    char* directory = mkdtemp(buffer.data());
    ASSERT_NE(directory, nullptr);
    directory_ = directory;
    path_ = (directory_ / "puzzle.html").string();
  }

  void TearDown() override {
    if (!directory_.empty()) {
      std::error_code error;
      std::filesystem::remove_all(directory_, error);
      EXPECT_FALSE(error) << error.message();
    }
  }

  std::string Render(const PuzzleReportData& data) {
    const auto separation = VerifyPuzzleSeparation(data);
    EXPECT_TRUE(separation.ok()) << separation.status();
    if (!separation.ok())
      return {};
    const absl::Status status = WritePuzzleHtml(data, *separation, path_);
    EXPECT_TRUE(status.ok()) << status;
    std::ifstream input(path_, std::ios::binary);
    return {std::istreambuf_iterator<char>(input),
            std::istreambuf_iterator<char>()};
  }

  std::filesystem::path directory_;
  std::string path_;
};

TEST_F(PuzzleHtmlTest, EmbedsEveryPointAndExplainsAuditScope) {
  const std::string html = Render(Example());
  EXPECT_NE(html.find("3 scored points / 3 distinct full residual vectors / "
                      "3 distinct target token IDs"),
            std::string::npos);
  EXPECT_NE(html.find("third attention layer (A3), before that block's MLP"),
            std::string::npos);
  EXPECT_NE(html.find("not PCA"), std::string::npos);
  EXPECT_NE(html.find("five-token prompt mask and padding are excluded"),
            std::string::npos);
  EXPECT_NE(html.find("does not establish unambiguous next-token prediction"),
            std::string::npos);
  EXPECT_NE(html.find("\"model_width\":3,\"context_length\":3"),
            std::string::npos);
  EXPECT_NE(
      html.find("\"fact_index\":0,\"position\":2,\"input_token\":0,"
                "\"target_token\":-1,\"padding\":true,\"coordinates\":[0,0,0]"),
      std::string::npos);
  size_t count = 0;
  size_t position = 0;
  while ((position = html.find("\"fact_index\":", position)) !=
         std::string::npos) {
    ++count;
    ++position;
  }
  EXPECT_EQ(count, size_t{6});
  EXPECT_NE(html.find("type=\"checkbox\" checked"), std::string::npos);
  EXPECT_NE(html.find("valid / "), std::string::npos);
  EXPECT_NE(html.find("scored / "), std::string::npos);
  EXPECT_NE(html.find("Input token ID:"), std::string::npos);
  EXPECT_NE(html.find("Target token ID:"), std::string::npos);
}

TEST_F(PuzzleHtmlTest, EscapesUntrustedFactsAndNeedsNoExternalResources) {
  PuzzleReportData data = Example();
  data.facts[0] = "</script><script>alert(\"x\")</script>&'\\\n\x01";
  data.facts[1] = "Unicode: \u2028 and \u2029, caf\u00e9";
  const std::string html = Render(data);
  EXPECT_EQ(html.find(data.facts[0]), std::string::npos);
  EXPECT_NE(
      html.find("\\u003c/script\\u003e\\u003cscript\\u003ealert(\\\"x\\\")"),
      std::string::npos);
  EXPECT_NE(html.find("\\u0026'\\\\\\u000a\\u0001"), std::string::npos);
  EXPECT_NE(html.find("Unicode: \\u2028 and \\u2029, caf\u00e9"),
            std::string::npos);
  EXPECT_EQ(html.find("<script>alert"), std::string::npos);
  EXPECT_EQ(html.find("<script src="), std::string::npos);
  EXPECT_EQ(html.find("<link "), std::string::npos);
  EXPECT_EQ(html.find("fetch("), std::string::npos);
  EXPECT_EQ(html.find("innerHTML"), std::string::npos);
  EXPECT_NE(html.find("tooltip.textContent"), std::string::npos);
  EXPECT_NE(html.find("<noscript>"), std::string::npos);
  const size_t closing_script = html.find("</script>");
  ASSERT_NE(closing_script, std::string::npos);
  EXPECT_EQ(html.find("</script>", closing_script + 1), std::string::npos);
}

TEST_F(PuzzleHtmlTest, HandlesOddWidthConstantAndExtremeCoordinates) {
  PuzzleReportData data = Example();
  data.model_width = 1;
  for (PuzzlePoint& point : data.points) {
    point.coordinates = {std::numeric_limits<float>::max()};
    if (point.target_token >= 0)
      point.target_token = 30;
  }
  const std::string html = Render(data);
  EXPECT_NE(html.find("\"model_width\":1"), std::string::npos);
  EXPECT_NE(html.find("3.40282347e+38"), std::string::npos);
  EXPECT_NE(html.find("against zero"), std::string::npos);
  EXPECT_NE(html.find("Math.max(1, Math.abs(range[0]) * .1)"),
            std::string::npos);
  EXPECT_NE(html.find("? extents[yDimension] : [-1, 1]"), std::string::npos);
}

TEST_F(PuzzleHtmlTest, RejectsStaleAuditCountsBeforeCreatingFile) {
  const PuzzleReportData data = Example();
  EXPECT_EQ(WritePuzzleHtml(data, PuzzleSeparation{}, path_).code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_FALSE(std::filesystem::exists(path_));
}

TEST_F(PuzzleHtmlTest, RevalidatesCoverageAndCollisionsBeforeCreatingFile) {
  PuzzleReportData data = Example();
  const auto audit = VerifyPuzzleSeparation(data);
  ASSERT_TRUE(audit.ok());
  data.points.pop_back();
  EXPECT_EQ(WritePuzzleHtml(data, *audit, path_).code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_FALSE(std::filesystem::exists(path_));
  data = Example();
  data.points[4].coordinates = data.points[1].coordinates;
  EXPECT_EQ(WritePuzzleHtml(data, *audit, path_).code(),
            absl::StatusCode::kFailedPrecondition);
  EXPECT_FALSE(std::filesystem::exists(path_));
}

TEST_F(PuzzleHtmlTest, RejectsEmptyAndNulPathsAndReportsIoFailure) {
  const PuzzleReportData data = Example();
  const auto audit = VerifyPuzzleSeparation(data);
  ASSERT_TRUE(audit.ok());
  EXPECT_EQ(WritePuzzleHtml(data, *audit, "").code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(WritePuzzleHtml(data, *audit, std::string("abc\0def", 7)).code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(WritePuzzleHtml(data, *audit, directory_.string()).code(),
            absl::StatusCode::kInternal);
}

}  // namespace
}  // namespace pluto::llm::memorize_general_facts
