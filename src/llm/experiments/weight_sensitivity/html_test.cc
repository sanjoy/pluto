#include "src/llm/experiments/weight_sensitivity/html.h"

#include <limits>
#include <string>

#include "gtest/gtest.h"

namespace pluto::llm::weight_sensitivity {
namespace {

SensitivityReport Example() {
  SensitivityReport report;
  report.checkpoint = "/checkpoints/step_10";
  report.corpus = "/corpus.txt";
  report.config.model_width = 16;
  report.expected_samples = 4;
  report.baseline.exact = {1, 1, 1, 0};
  report.baseline.scored_tokens = 40;
  report.baseline.token_errors = 1;
  report.planned_results = 2;
  report.complete = true;
  AblationResult result;
  result.target.name = "Attn 0 Q";
  result.target.checkpoint_index = 4;
  result.target.shape = {16, 16};
  result.target.rows = 16;
  result.target.columns = 16;
  result.target.row_stride = 48;
  result.target.tensor_elements = 768;
  result.noise_stddev = .02;
  result.seconds = 1.25;
  result.scores.exact = {1, 1, 0, 0};
  result.scores.scored_tokens = 40;
  result.scores.token_errors = 2;
  report.results.push_back(result);
  result.target.name = "Attn 0 K";
  result.target.offset = 16;
  result.scores.exact = {0, 0, 1, 1};
  result.scores.token_errors = 3;
  report.results.push_back(result);
  return report;
}

TEST(SensitivityHtmlTest, CountsNewFailuresSeparatelyAndSortsByImpact) {
  const auto rendered = RenderHtml(Example());
  ASSERT_TRUE(rendered.ok()) << rendered.status();
  EXPECT_NE(rendered->find("3 / 4 exact completions"), std::string::npos);
  EXPECT_NE(rendered->find("Complete: 2 / 2"), std::string::npos);
  // K breaks two previously-correct completions while fixing a baseline error.
  // Its all-wrong count ties Q, which breaks only one new completion.
  const size_t k = rendered->find("<td>Attn 0 K</td>");
  const size_t q = rendered->find("<td>Attn 0 Q</td>");
  ASSERT_NE(k, std::string::npos);
  ASSERT_NE(q, std::string::npos);
  EXPECT_LT(k, q);
  EXPECT_NE(rendered->find("<td data-sort=\"2\">2 / 4 (50%)</td>"
                           "<td data-sort=\"2\">2 / 4 (50%)</td>",
                           k),
            std::string::npos);
  EXPECT_NE(rendered->find("<td data-sort=\"1\">1 / 4 (25%)</td>"
                           "<td data-sort=\"2\">2 / 4 (50%)</td>",
                           q),
            std::string::npos);
  EXPECT_NE(rendered->find("16 &times; 16"), std::string::npos);
  EXPECT_NE(rendered->find("offset 16; rows 16; columns 16; row stride 48"),
            std::string::npos);
}

TEST(SensitivityHtmlTest, EscapesAllStringsAndHasNoExternalResources) {
  SensitivityReport report = Example();
  const std::string malicious = "</td><script>alert(\"x\")</script>&'";
  report.checkpoint = malicious;
  report.corpus = malicious;
  report.target_filter = malicious;
  report.results[0].target.name = malicious;
  const auto rendered = RenderHtml(report);
  ASSERT_TRUE(rendered.ok()) << rendered.status();
  EXPECT_EQ(rendered->find(malicious), std::string::npos);
  const std::string escaped =
      "&lt;/td&gt;&lt;script&gt;alert(&quot;x&quot;)"
      "&lt;/script&gt;&amp;&#39;";
  size_t occurrences = 0;
  for (size_t at = 0; (at = rendered->find(escaped, at)) != std::string::npos;
       at += escaped.size())
    ++occurrences;
  EXPECT_EQ(occurrences, 4u);
  EXPECT_EQ(rendered->find("<script src="), std::string::npos);
  EXPECT_EQ(rendered->find("<link "), std::string::npos);
  EXPECT_EQ(rendered->find("fetch("), std::string::npos);
  EXPECT_EQ(rendered->find("innerHTML"), std::string::npos);
  EXPECT_NE(rendered->find("<noscript>"), std::string::npos);
}

TEST(SensitivityHtmlTest, MarksPartialResultsEvenBeforeFirstTrial) {
  SensitivityReport report = Example();
  report.complete = false;
  report.results.clear();
  const auto rendered = RenderHtml(report);
  ASSERT_TRUE(rendered.ok()) << rendered.status();
  EXPECT_NE(rendered->find("Partial results: 0 / 2"), std::string::npos);
  EXPECT_NE(rendered->find("Unmeasured targets are not shown"),
            std::string::npos);
}

TEST(SensitivityHtmlTest, RejectsInvalidScoresAndProgress) {
  SensitivityReport report = Example();
  report.baseline.exact.clear();
  EXPECT_FALSE(RenderHtml(report).ok());
  report = Example();
  report.results[0].scores.exact.pop_back();
  EXPECT_FALSE(RenderHtml(report).ok());
  report = Example();
  report.results[0].scores.exact[0] = 2;
  EXPECT_FALSE(RenderHtml(report).ok());
  report = Example();
  report.results[0].scores.token_errors = 41;
  EXPECT_FALSE(RenderHtml(report).ok());
  report = Example();
  report.results[0].scores.nonfinite_rows = -1;
  EXPECT_FALSE(RenderHtml(report).ok());
  report = Example();
  report.results[0].scores.scored_tokens = 39;
  EXPECT_FALSE(RenderHtml(report).ok());
  report = Example();
  report.results.pop_back();
  EXPECT_FALSE(RenderHtml(report).ok());
  report = Example();
  report.planned_results = 1;
  EXPECT_FALSE(RenderHtml(report).ok());
}

TEST(SensitivityHtmlTest, RejectsInvalidNoiseTimingAndSlices) {
  for (double invalid : {-1.0, std::numeric_limits<double>::infinity(),
                         std::numeric_limits<double>::quiet_NaN()}) {
    SensitivityReport report = Example();
    report.results[0].noise_stddev = invalid;
    EXPECT_FALSE(RenderHtml(report).ok());
    report = Example();
    report.results[0].seconds = invalid;
    EXPECT_FALSE(RenderHtml(report).ok());
    report = Example();
    report.noise_scale = invalid;
    EXPECT_FALSE(RenderHtml(report).ok());
  }
  SensitivityReport report = Example();
  report.results[0].target.offset = std::numeric_limits<size_t>::max();
  EXPECT_FALSE(RenderHtml(report).ok());
  report = Example();
  report.results[0].target.rows = std::numeric_limits<size_t>::max();
  EXPECT_FALSE(RenderHtml(report).ok());
  report = Example();
  report.results[0].target.row_stride = 0;
  EXPECT_FALSE(RenderHtml(report).ok());
}

TEST(SensitivityHtmlTest, ExplainsAssumptionsAndKeepsTrialSeed) {
  SensitivityReport report = Example();
  report.results[0].seed = std::numeric_limits<uint64_t>::max();
  const auto rendered = RenderHtml(report);
  ASSERT_TRUE(rendered.ok()) << rendered.status();
  EXPECT_NE(rendered->find("18446744073709551615"), std::string::npos);
  EXPECT_NE(rendered->find("teacher-forced"), std::string::npos);
  EXPECT_NE(rendered->find("plus EOS"), std::string::npos);
  EXPECT_NE(rendered->find("weights are tied"), std::string::npos);
  EXPECT_NE(rendered->find("not unique semantic attributions"),
            std::string::npos);
  EXPECT_NE(rendered->find("zero-RMS"), std::string::npos);
}

}  // namespace
}  // namespace pluto::llm::weight_sensitivity
