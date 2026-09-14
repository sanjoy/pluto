#include "src/llm/experiments/weight_history/html.h"

#include <cstdint>
#include <ios>
#include <limits>
#include <locale>
#include <sstream>
#include <string>

#include "absl/status/status.h"
#include "gtest/gtest.h"
#include "src/llm/experiments/weight_history/history.h"

namespace pluto::llm::weight_history {
namespace {

History Example() {
  History history;
  history.directory = "/checkpoints/example";
  history.steps = {0, 10, 100};
  TensorHistory tensor;
  tensor.weight_id = 0;
  tensor.element_count = 4;
  Sample first;
  first.step = 0;
  first.rms = 0;
  first.l2 = 0;
  first.from_first_rms = 0;
  Sample second;
  second.step = 10;
  second.rms = 2;
  second.l2 = 4;
  second.from_first_rms = 2;
  second.delta_rms = 2;
  second.delta_l2 = 4;
  second.max_abs_delta = 2;
  second.changed_fraction = 1;
  // The previous norm is zero, so relative_l2 deliberately remains undefined.
  Sample third;
  third.step = 100;
  third.rms = 3;
  third.l2 = 6;
  third.from_first_rms = 3;
  third.delta_rms = 1;
  third.delta_l2 = 2;
  third.relative_l2 = 0.5;
  third.max_abs_delta = 1;
  third.changed_fraction = 1;
  tensor.samples = {first, second, third};
  history.tensors.push_back(tensor);
  tensor.weight_id = 7;
  history.tensors.push_back(tensor);
  return history;
}

std::string Render(const History& history) {
  std::ostringstream output;
  const absl::Status status = WriteHtml(output, history);
  EXPECT_TRUE(status.ok()) << status;
  return output.str();
}

size_t Count(const std::string& text, const std::string& needle) {
  size_t count = 0;
  size_t offset = 0;
  while ((offset = text.find(needle, offset)) != std::string::npos) {
    ++count;
    offset += needle.size();
  }
  return count;
}

TEST(WeightHistoryHtmlTest, IncludesOneStaticGraphAndTablePerTensor) {
  const std::string html = Render(Example());
  EXPECT_EQ(Count(html, "<section class=\"tensor\""), 2);
  EXPECT_EQ(Count(html, "<svg class=\"chart\""), 2);
  EXPECT_EQ(Count(html, "<table>"), 2);
  EXPECT_EQ(Count(html, "<script type=\"application/json\""), 2);
  EXPECT_NE(html.find("<h2>weight_0.bin</h2>"), std::string::npos);
  EXPECT_NE(html.find("<h2>weight_7.bin</h2>"), std::string::npos);
  EXPECT_NE(html.find("3 checkpoints &middot; 2 tensors"), std::string::npos);
  // Position step 10 at 10% of the numeric interval [0, 100], not at the
  // midpoint just because this is the middle checkpoint in the array.
  EXPECT_NE(html.find("points=\"151,20 880,115 \""), std::string::npos);
  EXPECT_NE(html.find("<title>Step 10: 2</title>"), std::string::npos);
  EXPECT_NE(html.find("<tr><td>0</td><td>N/A</td>"), std::string::npos);
  EXPECT_NE(html.find("<tr><td>10</td><td>2</td><td>4</td><td>N/A</td>"),
            std::string::npos);
}

TEST(WeightHistoryHtmlTest, SelfContainedAndEscapesUntrustedDirectory) {
  History history = Example();
  history.directory = "</code><script>alert(\"x\")</script>&'";
  const std::string html = Render(history);
  EXPECT_EQ(html.find(history.directory), std::string::npos);
  EXPECT_NE(html.find("&lt;/code&gt;&lt;script&gt;alert(&quot;x&quot;)"
                      "&lt;/script&gt;&amp;&#39;"),
            std::string::npos);
  EXPECT_EQ(html.find("<script src="), std::string::npos);
  EXPECT_EQ(html.find("<link "), std::string::npos);
  EXPECT_EQ(html.find("fetch("), std::string::npos);
  EXPECT_EQ(html.find("innerHTML"), std::string::npos);
  EXPECT_NE(html.find("<style>"), std::string::npos);
  EXPECT_NE(html.find("<noscript>"), std::string::npos);
  EXPECT_NE(html.find("value=\"from_first_rms\""), std::string::npos);
  EXPECT_NE(html.find("id=\"filter\""), std::string::npos);
}

TEST(WeightHistoryHtmlTest, SingleCheckpointHasNoFabricatedDelta) {
  History history = Example();
  history.steps.resize(1);
  for (TensorHistory& tensor : history.tensors)
    tensor.samples.resize(1);
  const std::string html = Render(history);
  EXPECT_EQ(Count(html, "No predecessor checkpoint to compare"), 2);
  EXPECT_EQ(html.find("<circle "), std::string::npos);
  EXPECT_NE(html.find("\"delta_rms\":null"), std::string::npos);
  EXPECT_NE(html.find("\"relative_l2\":null"), std::string::npos);
  // Magnitude metrics can still display the one snapshot at the midpoint.
  EXPECT_NE(html.find("\"step\":\"0\",\"x\":475"), std::string::npos);
}

TEST(WeightHistoryHtmlTest, ZeroChangesProduceFiniteVisiblePoints) {
  History history = Example();
  for (TensorHistory& tensor : history.tensors)
    for (size_t i = 1; i < tensor.samples.size(); ++i) {
      tensor.samples[i].delta_rms = 0;
      tensor.samples[i].delta_l2 = 0;
      tensor.samples[i].max_abs_delta = 0;
      tensor.samples[i].changed_fraction = 0;
    }
  const std::string html = Render(history);
  EXPECT_NE(html.find("points=\"151,210 880,210 \""), std::string::npos);
  EXPECT_NE(html.find("<title>Step 10: 0</title>"), std::string::npos);
  EXPECT_EQ(html.find("NaN"), std::string::npos);
  EXPECT_EQ(html.find("Infinity"), std::string::npos);
}

TEST(WeightHistoryHtmlTest, PreservesLargeAdjacentIntegerSteps) {
  History history = Example();
  const int64_t last = std::numeric_limits<int64_t>::max();
  history.steps = {last - 2, last - 1, last};
  for (TensorHistory& tensor : history.tensors)
    for (size_t i = 0; i < tensor.samples.size(); ++i)
      tensor.samples[i].step = history.steps[i];
  const std::string html = Render(history);
  EXPECT_NE(html.find("\"step\":\"9223372036854775805\",\"x\":70"),
            std::string::npos);
  EXPECT_NE(html.find("\"step\":\"9223372036854775806\",\"x\":475"),
            std::string::npos);
  EXPECT_NE(html.find("\"step\":\"9223372036854775807\",\"x\":880"),
            std::string::npos);
  EXPECT_NE(html.find("Step 9223372036854775806: 2"), std::string::npos);
}

class CommaDecimal final : public std::numpunct<char> {
 protected:
  char do_decimal_point() const override { return ','; }
};

TEST(WeightHistoryHtmlTest, UsesFullPrecisionAndDoesNotChangeCallerFormatting) {
  History history = Example();
  history.tensors[0].samples[1].delta_rms = 1.2345678901234567;
  std::ostringstream output;
  output.imbue(std::locale(std::locale::classic(), new CommaDecimal));
  output.precision(2);
  ASSERT_TRUE(WriteHtml(output, history).ok());
  EXPECT_NE(output.str().find("\"delta_rms\":1.2345678901234567"),
            std::string::npos);
  EXPECT_EQ(output.precision(), 2);
  EXPECT_EQ(
      std::use_facet<std::numpunct<char>>(output.getloc()).decimal_point(),
      ',');
}

TEST(WeightHistoryHtmlTest, RejectsInvalidMetricsWithoutWritingPartialReport) {
  for (double invalid : {-1.0, std::numeric_limits<double>::infinity(),
                         std::numeric_limits<double>::quiet_NaN()}) {
    History history = Example();
    history.tensors[0].samples[1].delta_rms = invalid;
    std::ostringstream output;
    EXPECT_EQ(WriteHtml(output, history).code(),
              absl::StatusCode::kInvalidArgument);
    EXPECT_TRUE(output.str().empty());
  }
}

TEST(WeightHistoryHtmlTest, RejectsMalformedAlignmentAndDeltaAvailability) {
  for (int variant = 0; variant < 9; ++variant) {
    History history = Example();
    switch (variant) {
      case 0:
        history.steps[1] = 0;
        break;
      case 1:
        history.tensors[1].weight_id = 0;
        break;
      case 2:
        history.tensors[0].samples[1].step = 11;
        break;
      case 3:
        history.tensors[0].samples[0].delta_rms = 0;
        break;
      case 4:
        history.tensors[0].samples[1].delta_l2.reset();
        break;
      case 5:
        history.tensors[0].samples[1].changed_fraction = 1.1;
        break;
      case 6:
        history.tensors[0].samples.clear();
        break;
      case 7:
        history.tensors[0].samples[1].relative_l2 = 0;
        break;
      case 8:
        history.tensors[0].samples[2].relative_l2.reset();
        break;
    }
    std::ostringstream output;
    EXPECT_EQ(WriteHtml(output, history).code(),
              absl::StatusCode::kInvalidArgument)
        << variant;
    EXPECT_TRUE(output.str().empty());
  }
}

TEST(WeightHistoryHtmlTest, EmptyHistoryAndBrokenOutputStream) {
  const std::string html = Render(History{});
  EXPECT_NE(html.find("No tensors to display."), std::string::npos);
  std::ostringstream output;
  output.setstate(std::ios::badbit);
  EXPECT_EQ(WriteHtml(output, Example()).code(), absl::StatusCode::kInternal);
}

}  // namespace
}  // namespace pluto::llm::weight_history
