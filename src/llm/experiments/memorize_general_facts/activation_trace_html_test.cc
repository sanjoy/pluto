#include "src/llm/experiments/memorize_general_facts/activation_trace_html.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <locale>
#include <sstream>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/types/span.h"
#include "gtest/gtest.h"

namespace pluto::llm::memorize_general_facts {
namespace {

ActivationTrace ExampleTrace() {
  ActivationTrace trace;
  trace.prompt = "The capital";
  trace.continuation = " of";
  trace.prompt_token_count = 2;
  trace.model_width = 16;
  trace.token_ids = {10, 20, 30};
  trace.token_texts = {"The", " capital", " of"};
  trace.boundaries = {{"Token + position embeddings", std::vector<float>(48)},
                      {"Transformer block 0", std::vector<float>(48)}};
  for (size_t i = 0; i < trace.boundaries[0].values.size(); ++i)
    trace.boundaries[0].values[i] = static_cast<float>(i % 8) / 8;
  return trace;
}

size_t Count(absl::string_view text, absl::string_view needle) {
  size_t count = 0;
  size_t position = 0;
  while ((position = text.find(needle, position)) != absl::string_view::npos) {
    ++count;
    position += needle.size();
  }
  return count;
}

TEST(ActivationTraceHtmlTest, SelfContainedEightPlanesPerTokenAndBoundary) {
  const auto trace = ExampleTrace();
  std::ostringstream output;
  const auto status = WriteActivationTraceHtml({trace}, output);
  ASSERT_TRUE(status.ok()) << status;
  const std::string html = output.str();
  EXPECT_EQ(html.find("<!doctype html>"), 0);
  EXPECT_NE(html.find("</body></html>"), std::string::npos);
  EXPECT_EQ(Count(html, "<svg class=\"plane\""), 3 * 2 * 8);
  EXPECT_EQ(Count(html, "<div class=\"planes\">"), 3 * 2);
  EXPECT_EQ(Count(html, "16 raw values"), 3 * 2);
  EXPECT_EQ(Count(html, "class=\"prompt\""), 2);
  EXPECT_EQ(Count(html, "class=\"generated\""), 1);
  EXPECT_NE(html.find("Position 2 · Token ID 30"), std::string::npos);
  EXPECT_NE(html.find("&quot; capital&quot;"), std::string::npos);
  EXPECT_NE(html.find("Transformer block 0"), std::string::npos);
  EXPECT_NE(html.find("Dimensions 14, 15"), std::string::npos);
  EXPECT_EQ(html.find("<script"), std::string::npos);
  EXPECT_EQ(html.find("<link"), std::string::npos);
  EXPECT_EQ(html.find("http://"), std::string::npos);
  EXPECT_EQ(html.find("https://"), std::string::npos);
  EXPECT_EQ(html.find("src="), std::string::npos);
}

TEST(ActivationTraceHtmlTest, UsesSharedSymmetricScaleAndCorrectDirections) {
  auto trace = ExampleTrace();
  auto& first = trace.boundaries[0].values;
  first[0] = 1;
  first[1] = -1;
  first[2] = -2;
  first[3] = 2;
  // The scale's largest magnitude is at a different boundary and token.
  trace.boundaries[1].values[47] = -4;
  std::ostringstream output;
  ASSERT_TRUE(WriteActivationTraceHtml({trace}, output).ok());
  const auto html = output.str();
  EXPECT_NE(html.find("Shared coordinate range: [−4, +4]"), std::string::npos);
  EXPECT_NE(html.find("Dimensions 0, 1: (1, -1)"), std::string::npos);
  EXPECT_NE(html.find("x2=\"70\" y2=\"70\""), std::string::npos);
  EXPECT_NE(html.find("x2=\"40\" y2=\"40\""), std::string::npos);
  EXPECT_NE(html.find("x2=\"60\" y2=\"100\""), std::string::npos);
  EXPECT_NE(html.find("[1, -1, -2, 2, 0.5, 0.625, 0.75, 0.875,"),
            std::string::npos);
}

TEST(ActivationTraceHtmlTest, ZeroVectorsRemainAtOrigin) {
  auto trace = ExampleTrace();
  for (auto& boundary : trace.boundaries)
    boundary.values.assign(48, 0);
  std::ostringstream output;
  ASSERT_TRUE(WriteActivationTraceHtml({trace}, output).ok());
  const auto html = output.str();
  EXPECT_NE(html.find("Shared coordinate range: [−1, +1]"), std::string::npos);
  EXPECT_EQ(Count(html, "x2=\"60\" y2=\"60\""), 3 * 2 * 8);
  EXPECT_EQ(html.find("marker-end="), std::string::npos);
  EXPECT_EQ(html.find("nan"), std::string::npos);
}

TEST(ActivationTraceHtmlTest, PreservesRawPrecisionAndTinyFiniteMagnitudes) {
  auto trace = ExampleTrace();
  for (auto& boundary : trace.boundaries)
    boundary.values.assign(48, 0);
  trace.boundaries[0].values[0] = std::numeric_limits<float>::denorm_min();
  std::ostringstream output;
  ASSERT_TRUE(WriteActivationTraceHtml({trace}, output).ok());
  const auto html = output.str();
  EXPECT_NE(html.find("1.40129846e-45"), std::string::npos);
  EXPECT_NE(html.find("x2=\"100\" y2=\"60\""), std::string::npos);
}

TEST(ActivationTraceHtmlTest, EscapesMarkupAndArbitraryTokenBytes) {
  auto trace = ExampleTrace();
  trace.prompt = "<script>alert('x')</script> & \"prompt\"";
  trace.continuation = "</dd><img src=x onerror=alert(1)>";
  trace.boundaries[0].name = "<svg onload='bad'> & \"boundary\"";
  trace.token_texts[0] = "<script>";
  trace.token_texts[1] = std::string("\0\xff\xc3", 3);
  trace.token_texts[2] = "\n\r\t\\ café ☀";
  std::ostringstream output;
  ASSERT_TRUE(WriteActivationTraceHtml({trace}, output).ok());
  const auto html = output.str();
  EXPECT_EQ(html.find("<script>"), std::string::npos);
  EXPECT_EQ(html.find("<img"), std::string::npos);
  EXPECT_EQ(html.find("<svg onload"), std::string::npos);
  EXPECT_NE(html.find("&lt;script&gt;alert(&#39;x&#39;)&lt;/script&gt; &amp; "
                      "&quot;prompt&quot;"),
            std::string::npos);
  EXPECT_NE(html.find("\\x00\\xFF\\xC3"), std::string::npos);
  EXPECT_NE(html.find("\\n\\r\\t\\\\ café ☀"), std::string::npos);
  EXPECT_EQ(html.find('\0'), std::string::npos);
  EXPECT_EQ(html.find('\xff'), std::string::npos);
}

TEST(ActivationTraceHtmlTest, EscapesInvalidUtf8Sequences) {
  auto trace = ExampleTrace();
  trace.token_texts[0] = "\xed\xa0\x80";          // UTF-8-encoded surrogate.
  trace.token_texts[1] = "\xf4\x90\x80\x80";      // Above Unicode range.
  trace.token_texts[2] = "\xc0\xaf\xe0\x80\x80";  // Overlong encodings.
  std::ostringstream output;
  ASSERT_TRUE(WriteActivationTraceHtml({trace}, output).ok());
  const auto html = output.str();
  EXPECT_NE(html.find("\\xED\\xA0\\x80"), std::string::npos);
  EXPECT_NE(html.find("\\xF4\\x90\\x80\\x80"), std::string::npos);
  EXPECT_NE(html.find("\\xC0\\xAF\\xE0\\x80\\x80"), std::string::npos);
}

TEST(ActivationTraceHtmlTest, PrintsMultiplePromptsInOneDocument) {
  const std::vector<ActivationTrace> traces = {ExampleTrace(), ExampleTrace()};
  std::ostringstream output;
  ASSERT_TRUE(WriteActivationTraceHtml(traces, output).ok());
  const auto html = output.str();
  EXPECT_EQ(Count(html, "<!doctype html>"), 1);
  EXPECT_EQ(Count(html, "<section>"), 2);
  EXPECT_EQ(Count(html, "id=\"arrow-tip\""), 1);
  EXPECT_NE(html.find("<h2>Prompt 1</h2>"), std::string::npos);
  EXPECT_NE(html.find("<h2>Prompt 2</h2>"), std::string::npos);
}

TEST(ActivationTraceHtmlTest, RejectsMalformedTraceBeforeWritingAnything) {
  std::vector<ActivationTrace> invalid;
  auto trace = ExampleTrace();
  trace.model_width = 32;
  invalid.push_back(trace);
  trace = ExampleTrace();
  trace.token_ids.clear();
  invalid.push_back(trace);
  trace = ExampleTrace();
  trace.token_texts.pop_back();
  invalid.push_back(trace);
  trace = ExampleTrace();
  trace.prompt_token_count = 4;
  invalid.push_back(trace);
  trace = ExampleTrace();
  trace.boundaries.clear();
  invalid.push_back(trace);
  trace = ExampleTrace();
  trace.boundaries[0].values.pop_back();
  invalid.push_back(trace);
  trace = ExampleTrace();
  trace.token_ids[0] = -1;
  invalid.push_back(trace);
  for (float value : {std::numeric_limits<float>::infinity(),
                      -std::numeric_limits<float>::infinity(),
                      std::numeric_limits<float>::quiet_NaN()}) {
    trace = ExampleTrace();
    trace.boundaries[1].values[1] = value;
    invalid.push_back(trace);
  }
  for (const auto& bad : invalid) {
    std::ostringstream output;
    EXPECT_EQ(WriteActivationTraceHtml({ExampleTrace(), bad}, output).code(),
              absl::StatusCode::kInvalidArgument);
    EXPECT_TRUE(output.str().empty());
  }
  std::ostringstream output;
  EXPECT_EQ(WriteActivationTraceHtml({}, output).code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(output.str().empty());
}

TEST(ActivationTraceHtmlTest, ReportsOutputStreamFailure) {
  std::ostringstream output;
  output.setstate(std::ios::badbit);
  EXPECT_EQ(WriteActivationTraceHtml({ExampleTrace()}, output).code(),
            absl::StatusCode::kInternal);
}

TEST(ActivationTraceHtmlTest, SvgCoordinatesAreLocaleIndependent) {
  class DecimalComma : public std::numpunct<char> {
   protected:
    char do_decimal_point() const override { return ','; }
  };
  auto trace = ExampleTrace();
  trace.boundaries[1].values[47] = 4;
  trace.boundaries[0].values[0] = 0.125f;
  std::ostringstream output;
  output.imbue(std::locale(std::locale::classic(), new DecimalComma));
  ASSERT_TRUE(WriteActivationTraceHtml({trace}, output).ok());
  EXPECT_NE(output.str().find("x2=\"61.25\""), std::string::npos);
  EXPECT_NE(output.str().find("Dimensions 0, 1: (0.125, 0.125)"),
            std::string::npos);
}

class ActivationTraceHtmlFileTest : public testing::Test {
 protected:
  void SetUp() override {
    std::string pattern =
        (std::filesystem::path(testing::TempDir()) / "activation-trace-XXXXXX")
            .string();
    ASSERT_NE(mkdtemp(pattern.data()), nullptr);
    directory_ = pattern;
  }
  void TearDown() override {
    if (directory_.empty())
      return;
    std::error_code error;
    std::filesystem::remove_all(directory_, error);
    EXPECT_FALSE(error) << error.message();
  }
  std::string Read(const std::filesystem::path& file) {
    std::ifstream input(file, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(input),
                       std::istreambuf_iterator<char>());
  }
  void ExpectNoTemporaryFiles() {
    for (const auto& entry : std::filesystem::directory_iterator(directory_))
      EXPECT_NE(
          entry.path().filename().string().find(".pluto-activation-trace-"), 0);
  }
  std::filesystem::path directory_;
};

TEST_F(ActivationTraceHtmlFileTest, CreatesAndAtomicallyReplacesReport) {
  const auto filename = directory_ / "trace.html";
  auto trace = ExampleTrace();
  ASSERT_TRUE(WriteActivationTraceHtmlFile({trace}, filename).ok());
  std::ostringstream expected;
  ASSERT_TRUE(WriteActivationTraceHtml({trace}, expected).ok());
  EXPECT_EQ(Read(filename), expected.str());
  trace.prompt = "A second prompt";
  ASSERT_TRUE(WriteActivationTraceHtmlFile({trace}, filename).ok());
  EXPECT_NE(Read(filename).find("A second prompt"), std::string::npos);
  EXPECT_EQ(Read(filename).find("The capital"), std::string::npos);
  ExpectNoTemporaryFiles();
}

TEST_F(ActivationTraceHtmlFileTest, InvalidInputPreservesExistingFile) {
  const auto filename = directory_ / "trace.html";
  ASSERT_TRUE(WriteActivationTraceHtmlFile({ExampleTrace()}, filename).ok());
  const auto previous = Read(filename);
  auto invalid = ExampleTrace();
  invalid.model_width = 32;
  EXPECT_EQ(WriteActivationTraceHtmlFile({invalid}, filename).code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(Read(filename), previous);
  ExpectNoTemporaryFiles();
}

TEST_F(ActivationTraceHtmlFileTest, ReportsMissingParentAndEmptyFilename) {
  EXPECT_EQ(WriteActivationTraceHtmlFile({ExampleTrace()}, {}).code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(WriteActivationTraceHtmlFile({ExampleTrace()},
                                         directory_ / "absent" / "trace.html")
                .code(),
            absl::StatusCode::kNotFound);
  EXPECT_FALSE(std::filesystem::exists(directory_ / "absent"));
  ExpectNoTemporaryFiles();
}

TEST_F(ActivationTraceHtmlFileTest, FailedPublicationRemovesOnlyItsTemporary) {
  const auto target = directory_ / "existing-directory";
  ASSERT_TRUE(std::filesystem::create_directory(target));
  const auto retained = target / "keep.txt";
  std::ofstream(retained) << "untouched";
  EXPECT_FALSE(WriteActivationTraceHtmlFile({ExampleTrace()}, target).ok());
  EXPECT_TRUE(std::filesystem::is_directory(target));
  EXPECT_EQ(Read(retained), "untouched");
  ExpectNoTemporaryFiles();
}

}  // namespace
}  // namespace pluto::llm::memorize_general_facts
