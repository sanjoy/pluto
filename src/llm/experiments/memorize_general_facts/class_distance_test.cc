#include "src/llm/experiments/memorize_general_facts/class_distance.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <regex>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "gtest/gtest.h"

namespace pluto::llm::memorize_general_facts {
namespace {

// Use multiple states per class so centroid and state-pair histograms differ.
// One unscored row is coincident with a scored row and must not participate.
PuzzleReportData Example() {
  return {.model_width = 2,
          .context_length = 3,
          .facts = {"first fact", "second fact"},
          .points = {{0, 0, 0, 1, false, {0, 0}},
                     {0, 1, 0, 1, false, {10, 0}},
                     {0, 2, 0, 7, false, {3, 4}},
                     {1, 0, 0, 7, false, {20, 0}},
                     {1, 1, 0, 12, false, {24, 0}},
                     {1, 2, 0, -1, true, {24, 0}}}};
}

// Keep outputs in Bazel's per-test temporary directory, not the repository.
// Distinct test names avoid accidental reuse between report-writing cases.
std::string OutputPath() {
  const char* directory = std::getenv("TEST_TMPDIR");
  return (std::filesystem::path(directory ? directory : "/tmp") /
          (std::string("class_distance_") +
           ::testing::UnitTest::GetInstance()->current_test_info()->name() +
           ".html"))
      .string();
}

std::string Read(const std::string& path) {
  std::ifstream input(path);
  return {std::istreambuf_iterator<char>(input), {}};
}

// Recover rendered bin counts so tests check histogram mass, not just markup.
// Every SVG bar has a title ending in its exact integer observation count.
size_t HistogramCount(const std::string& html) {
  const std::regex pattern(R"(: ([0-9]+) pairs</title>)");
  size_t total = 0;
  for (auto it = std::sregex_iterator(html.begin(), html.end(), pattern);
       it != std::sregex_iterator(); ++it)
    total += std::stoull((*it)[1]);
  return total;
}

TEST(ClassDistanceTest, MinimumPerUnorderedClassPairNotPerStateOrCentroid) {
  const auto result = ComputeClassDistances(Example());
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result->scored_points, 5);
  EXPECT_EQ(result->target_classes, 3);
  ASSERT_EQ(result->pairs.size(), 3);
  const auto& a = result->pairs[0];
  EXPECT_EQ(a.first_token, 1);
  EXPECT_EQ(a.second_token, 7);
  EXPECT_DOUBLE_EQ(a.distance, 5);
  EXPECT_EQ(a.first_point, 0);
  EXPECT_EQ(a.second_point, 2);
  const auto& b = result->pairs[1];
  EXPECT_EQ(b.first_token, 1);
  EXPECT_EQ(b.second_token, 12);
  EXPECT_DOUBLE_EQ(b.distance, 14);
  EXPECT_EQ(b.first_point, 1);
  EXPECT_EQ(b.second_point, 4);
  const auto& c = result->pairs[2];
  EXPECT_EQ(c.first_token, 7);
  EXPECT_EQ(c.second_token, 12);
  EXPECT_DOUBLE_EQ(c.distance, 4);
  EXPECT_EQ(c.first_point, 3);
  EXPECT_EQ(c.second_point, 4);
}

TEST(ClassDistanceTest, IgnoresPromptPaddingAndUnusedVocabularyClasses) {
  auto data = Example();
  data.points[5].padding = false;
  data.points[5].input_token = 100000;
  data.points[5].coordinates = {0, 0};
  const auto result = ComputeClassDistances(data);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result->target_classes, 3);
  EXPECT_EQ(result->pairs.size(), 3);
  EXPECT_DOUBLE_EQ(result->pairs[0].distance, 5);
}

TEST(ClassDistanceTest, PoolsAllPositionsAndUsesFirstWitnessOnTies) {
  auto data = Example();
  data.points[1].coordinates = data.points[0].coordinates;
  std::reverse(data.points.begin(), data.points.end());
  const auto result = ComputeClassDistances(data);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_DOUBLE_EQ(result->pairs[0].distance, 5);
  EXPECT_EQ(data.points[result->pairs[0].first_point].position, 1);
}

TEST(ClassDistanceTest, AllowsActualCrossClassCollisions) {
  auto data = Example();
  data.points[2].coordinates = {-0.0f, 0.0f};
  const auto result = ComputeClassDistances(data);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_DOUBLE_EQ(result->pairs[0].distance, 0);
}

TEST(ClassDistanceTest, DirectDoubleDifferencesPreserveNearbyLargeVectors) {
  auto data = Example();
  for (auto& point : data.points)
    point.target_token = -1;
  const float x = 1e10f;
  const float y = std::nextafter(x, std::numeric_limits<float>::infinity());
  data.points[0].target_token = 1;
  data.points[0].coordinates = {x, x};
  data.points[1].target_token = 7;
  data.points[1].coordinates = {y, x};
  const auto result = ComputeClassDistances(data);
  ASSERT_TRUE(result.ok()) << result.status();
  ASSERT_EQ(result->pairs.size(), 1);
  EXPECT_DOUBLE_EQ(result->pairs[0].distance, static_cast<double>(y) - x);
}

TEST(ClassDistanceTest, LargeFiniteFloatCoordinatesDoNotOverflow) {
  auto data = Example();
  const float largest = std::numeric_limits<float>::max();
  for (auto& point : data.points)
    point.target_token = -1;
  data.points[0].target_token = 1;
  data.points[0].coordinates = {largest, largest};
  data.points[1].target_token = 2;
  data.points[1].coordinates = {-largest, -largest};
  const auto result = ComputeClassDistances(data);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_DOUBLE_EQ(result->pairs[0].distance,
                   2 * static_cast<double>(largest) * std::sqrt(2.0));
}

TEST(ClassDistanceTest, NoPairsForZeroOrOneObservedClass) {
  for (int target : {-1, 7}) {
    auto data = Example();
    for (auto& point : data.points)
      if (!point.padding)
        point.target_token = target;
    const auto result = ComputeClassDistances(data);
    ASSERT_TRUE(result.ok()) << result.status();
    EXPECT_EQ(result->target_classes, target < 0 ? 0 : 1);
    EXPECT_TRUE(result->pairs.empty());
  }
}

TEST(ClassDistanceTest, RejectsMalformedCaptureEvenOnMaskedRows) {
  auto data = Example();
  data.points.back().coordinates[0] = std::numeric_limits<float>::infinity();
  EXPECT_EQ(ComputeClassDistances(data).status().code(),
            absl::StatusCode::kInvalidArgument);
  data = Example();
  data.points[0].coordinates.pop_back();
  EXPECT_EQ(ComputeClassDistances(data).status().code(),
            absl::StatusCode::kInvalidArgument);
  data = Example();
  data.points.pop_back();
  EXPECT_EQ(ComputeClassDistances(data).status().code(),
            absl::StatusCode::kInvalidArgument);
}

TEST(ClassDistanceHtmlTest, RendersBothHistogramsQuantilesLabelsAndWitnesses) {
  auto data = Example();
  data.facts[0] = "unsafe </script> & <img>";
  std::vector<std::string> labels(13);
  labels[1] = " <Paris>";
  labels[7] = "\n";
  labels[12] = "EOS";
  ASSERT_TRUE(WriteClassDistanceHtml(data, labels, OutputPath()).ok());
  const std::string html = Read(OutputPath());
  EXPECT_NE(html.find("unordered class pairs: 3"), std::string::npos);
  EXPECT_NE(html.find("<td>Median</td><td>5</td>"), std::string::npos);
  EXPECT_NE(html.find("fact 2, input position 1"), std::string::npos);
  EXPECT_NE(html.find("&lt;Paris&gt;"), std::string::npos);
  EXPECT_NE(html.find("&lt;/script&gt; &amp; &lt;img&gt;"), std::string::npos);
  EXPECT_EQ(html.find("<script>"), std::string::npos);
  EXPECT_EQ(html.find("unsafe </script>"), std::string::npos);
  EXPECT_EQ(HistogramCount(html), 6);
}

TEST(ClassDistanceHtmlTest, ZerosAppearOnlyInLinearHistogram) {
  auto data = Example();
  data.points[2].coordinates = data.points[0].coordinates;
  ASSERT_TRUE(WriteClassDistanceHtml(data, {}, OutputPath()).ok());
  const std::string html = Read(OutputPath());
  EXPECT_NE(html.find("Zero-distance class pairs: 1"), std::string::npos);
  EXPECT_EQ(HistogramCount(html), 5);
}

TEST(ClassDistanceHtmlTest, ConstantZeroDistributionAvoidsDivisionByZero) {
  auto data = Example();
  for (auto& point : data.points)
    point.coordinates = {0, 0};
  ASSERT_TRUE(WriteClassDistanceHtml(data, {}, OutputPath()).ok());
  const std::string html = Read(OutputPath());
  EXPECT_NE(html.find("Zero-distance class pairs: 3"), std::string::npos);
  EXPECT_NE(html.find("No positive class-pair distances"), std::string::npos);
  EXPECT_EQ(HistogramCount(html), 3);
  EXPECT_EQ(html.find("\"nan\""), std::string::npos);
}

TEST(ClassDistanceHtmlTest, SinglePositivePairConstantDistribution) {
  auto data = Example();
  data.points[4].target_token = 7;
  ASSERT_TRUE(WriteClassDistanceHtml(data, {}, OutputPath()).ok());
  const std::string html = Read(OutputPath());
  EXPECT_EQ(HistogramCount(html), 2);
  EXPECT_NE(html.find("<td>Minimum</td><td>5</td>"), std::string::npos);
  EXPECT_NE(html.find("<td>Maximum</td><td>5</td>"), std::string::npos);
}

TEST(ClassDistanceHtmlTest, EmptyDistributionHasExplicitNoDataMessage) {
  auto data = Example();
  for (auto& point : data.points)
    point.target_token = -1;
  ASSERT_TRUE(WriteClassDistanceHtml(data, {}, OutputPath()).ok());
  EXPECT_NE(Read(OutputPath()).find("No class-pair distances to plot"),
            std::string::npos);
  EXPECT_EQ(HistogramCount(Read(OutputPath())), 0);
}

TEST(ClassDistanceHtmlTest, RejectsMissingLabelsAndBadPaths) {
  const std::vector<std::string> labels(7);
  EXPECT_EQ(WriteClassDistanceHtml(Example(), labels, OutputPath()).code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(WriteClassDistanceHtml(Example(), {}, "").code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(WriteClassDistanceHtml(Example(), {},
                                   OutputPath() + "/missing/report.html")
                .code(),
            absl::StatusCode::kInternal);
}

}  // namespace
}  // namespace pluto::llm::memorize_general_facts
