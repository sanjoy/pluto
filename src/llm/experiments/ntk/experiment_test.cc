#include "src/llm/experiments/ntk/experiment.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <string>
#include <system_error>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/escaping.h"
#include "absl/strings/str_cat.h"
#include "gtest/gtest.h"

namespace pluto::llm::ntk {
namespace {

// The report emits an ASCII JSON string containing a C-escaped byte string.
// These JSON strings use only quote/backslash escapes, so CUnescape decodes
// that first layer too. The second layer must recover the exact original path.
void ExpectPathRoundTrips(const std::string& json, absl::string_view key,
                          const std::string& original) {
  const std::string prefix = absl::StrCat("\"", key, "\":\"");
  const size_t field = json.find(prefix);
  ASSERT_NE(field, std::string::npos);
  const size_t start = field + prefix.size();
  size_t end = start;
  while (end < json.size() && json[end] != '"')
    end += json[end] == '\\' ? 2 : 1;
  ASSERT_LT(end, json.size());
  std::string escaped_bytes;
  ASSERT_TRUE(absl::CUnescape(
      absl::string_view(json).substr(start, end - start), &escaped_bytes));
  EXPECT_EQ(escaped_bytes, absl::CEscape(original));
  std::string decoded;
  ASSERT_TRUE(absl::CUnescape(escaped_bytes, &decoded));
  EXPECT_EQ(decoded, original);
}

TEST(NtkExperimentTest, WindowsPreserveGlobalTokenOffsetsAndHeldOutTargets) {
  const std::vector<int> corpus{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  auto examples = SelectCorpusExamples(corpus, WindowOptions{2, 1, 2, 3, 1});
  ASSERT_TRUE(examples.ok()) << examples.status();
  ASSERT_EQ(examples->size(), 3);
  EXPECT_EQ((*examples)[0].tokens, (std::vector<int>{1, 2}));
  EXPECT_EQ((*examples)[0].next_token, 3);
  EXPECT_EQ((*examples)[0].corpus_token_offset, 1);
  EXPECT_EQ((*examples)[1].tokens, (std::vector<int>{4, 5}));
  EXPECT_EQ((*examples)[1].next_token, 6);
  EXPECT_EQ((*examples)[1].split, "train");
  EXPECT_EQ((*examples)[2].tokens, (std::vector<int>{7, 8}));
  EXPECT_EQ((*examples)[2].next_token, 9);
  EXPECT_EQ((*examples)[2].corpus_token_offset, 7);
  EXPECT_EQ((*examples)[2].split, "eval");
}

TEST(NtkExperimentTest, RejectsOverlapInsufficientCorpusAndOverflow) {
  const std::vector<int> corpus{0, 1, 2};
  EXPECT_TRUE(SelectCorpusExamples(corpus, {1, 0, 2, 3, 0}).ok());
  for (const WindowOptions options :
       {WindowOptions{0, 1, 1, 2, 0}, WindowOptions{1, 0, 0, 1, 0},
        WindowOptions{1, 0, 2, 2, 0},
        WindowOptions{std::numeric_limits<size_t>::max(), 1, 1, 2, 0},
        WindowOptions{1, 0, std::numeric_limits<size_t>::max(), 3, 0}})
    EXPECT_EQ(SelectCorpusExamples(corpus, options).status().code(),
              absl::StatusCode::kInvalidArgument);
  for (const WindowOptions options :
       {WindowOptions{1, 1, 2, 3, 0}, WindowOptions{1, 0, 3, 4, 0},
        WindowOptions{1, 0, 1, 2, std::numeric_limits<size_t>::max()},
        WindowOptions{std::numeric_limits<size_t>::max(), 0, 1, 2, 0}})
    EXPECT_EQ(SelectCorpusExamples(corpus, options).status().code(),
              absl::StatusCode::kOutOfRange);
  EXPECT_EQ(SelectCorpusExamples({}, {1, 0, 1, 2, 0}).status().code(),
            absl::StatusCode::kOutOfRange);
}

TEST(NtkExperimentTest, DefaultOutputsAreTrainingOnlySortedAndUnique) {
  const std::vector<Example> examples{
      {"train", 0, {0}, 4, ""},
      {"train", 2, {0}, 2, ""},
      {"train", 4, {0}, 4, ""},
      {"eval", 6, {0}, 9, ""},
      {"prompt", std::nullopt, {0}, std::nullopt, ""}};
  auto ids = ResolveOutputTokens(examples, {}, 10);
  ASSERT_TRUE(ids.ok()) << ids.status();
  EXPECT_EQ(*ids, (std::vector<int>{2, 4}));
  auto explicit_ids = ResolveOutputTokens(examples, {9, 4, 2}, 10);
  ASSERT_TRUE(explicit_ids.ok()) << explicit_ids.status();
  EXPECT_EQ(*explicit_ids, (std::vector<int>{9, 4, 2}));
  for (const std::vector<int> invalid :
       {std::vector<int>{2}, {2, 4, 4}, {-1, 2, 4}, {2, 4, 10}})
    EXPECT_EQ(ResolveOutputTokens(examples, invalid, 10).status().code(),
              absl::StatusCode::kInvalidArgument);
  EXPECT_FALSE(ResolveOutputTokens({}, {}, 10).ok());
  EXPECT_FALSE(ResolveOutputTokens(examples, {}, 0).ok());
}

TEST(NtkExperimentTest, ParsesCompleteCommaSeparatedTokenIds) {
  auto ids = ParseOutputTokenIds(" 2 ,17, 3 ");
  ASSERT_TRUE(ids.ok()) << ids.status();
  EXPECT_EQ(*ids, (std::vector<int>{2, 17, 3}));
  auto empty = ParseOutputTokenIds("");
  ASSERT_TRUE(empty.ok());
  EXPECT_TRUE(empty->empty());
  for (const char* malformed :
       {" ", ",", "1,", ",1", "1,,2", "1x", "1 2", "-1", "2147483648", "1.0"})
    EXPECT_EQ(ParseOutputTokenIds(malformed).status().code(),
              absl::StatusCode::kInvalidArgument)
        << malformed;
}

ExperimentReport SmallReport() {
  ExperimentReport report;
  report.checkpoint = "checkpoint/step_0";
  report.tokenizer_directory = "tokenizer";
  report.corpus_path = "shakespeare.txt";
  report.compute_type = "fp16";
  report.windows = {1, 1, 2, 3, 0};
  report.dimensions = {8, 16, 4, 1, 8, 1, 8, 32};
  report.padding_token = 7;
  report.examples = {{"train", 0, {0, 1}, 2, "a\n\"b"},
                     {"eval", 3, {3, 4}, 4, "cd"}};
  report.output_tokens = {2, 3};
  report.output_token_text = {"x\"y", std::string("\x80", 1)};
  report.parameters = {{0, 3, 0}, {2, 2, 3}};
  report.parameter_count = 5;
  report.kernel = {4, 4, {4, 0, 1, 0, 0, 4, 0, 1, 1, 0, 4, 0, 0, 1, 0, 4}};
  report.initial_values = {1000, 999, 0, 2};
  report.ridge_predictions = {1, 0, -2, 2};
  return report;
}

TEST(NtkExperimentTest,
     ReportMapsRowsAndKeepsRawKernelAndNonzeroInitialFunction) {
  auto rendered = RenderReport(SmallReport());
  ASSERT_TRUE(rendered.ok()) << rendered.status();
  EXPECT_EQ(rendered->kernel_csv,
            "row,row_0,row_1,row_2,row_3\n0,4,0,1,0\n1,0,4,0,1\n"
            "2,1,0,4,0\n3,0,1,0,4\n");
  EXPECT_NE(rendered->metadata_json.find("\"weights_updated\":false"),
            std::string::npos);
  EXPECT_NE(rendered->metadata_json.find("\"initial_logits\":[1000,999,0,2]"),
            std::string::npos);
  EXPECT_NE(rendered->metadata_json.find(
                "\"row\":0,\"sample\":0,\"output_index\":0,\"flat_output_"
                "element\":18,\"output_token_id\":2,\"target\":1"),
            std::string::npos);
  EXPECT_NE(rendered->metadata_json.find(
                "\"row\":2,\"sample\":1,\"output_index\":0,\"flat_output_"
                "element\":18,\"output_token_id\":2,\"target\":null"),
            std::string::npos);
  EXPECT_NE(
      rendered->metadata_json.find("\"target_in_selected_outputs\":false"),
      std::string::npos);
  EXPECT_NE(rendered->metadata_json.find("NOT full-vocabulary probabilities"),
            std::string::npos);
  EXPECT_NE(
      rendered->metadata_json.find("\"initial_selected_softmax\":[0.731058"),
      std::string::npos);
  EXPECT_NE(rendered->metadata_json.find("\"gradient_descent_logits\":[]"),
            std::string::npos);
}

TEST(NtkExperimentTest,
     ReportEscapesArbitraryTokenBytesAndJsonControlCharacters) {
  auto report = SmallReport();
  report.checkpoint = "a\"b\\c\n\t";
  auto rendered = RenderReport(report);
  ASSERT_TRUE(rendered.ok()) << rendered.status();
  ExpectPathRoundTrips(rendered->metadata_json, "checkpoint_escaped_bytes",
                       report.checkpoint);
  // CEscape preserves a token's non-UTF-8 byte as an octal escape; JSON then
  // escapes that backslash. The file itself stays valid ASCII JSON.
  EXPECT_NE(rendered->metadata_json.find("\\\\200"), std::string::npos);
  for (unsigned char byte : rendered->metadata_json)
    EXPECT_LT(byte, 128);
  EXPECT_NE(rendered->predictions_csv.find("\"x\\\"\"y\""), std::string::npos);
}

TEST(NtkExperimentTest, OptionalGdPredictionsAreDistinctFromRidge) {
  auto report = SmallReport();
  report.kernel_steps = 2;
  report.gradient_descent_predictions = {0.1, 0.2, 0.3, 0.4};
  auto rendered = RenderReport(report);
  ASSERT_TRUE(rendered.ok()) << rendered.status();
  EXPECT_NE(rendered->metadata_json.find("\"kernel_steps\":2"),
            std::string::npos);
  EXPECT_NE(
      rendered->metadata_json.find("\"gradient_descent_logits\":[0.100000"),
      std::string::npos);
  EXPECT_NE(rendered->metadata_json.find("\"ridge_logits\":[1,0,-2,2]"),
            std::string::npos);
}

TEST(NtkExperimentTest, FilesystemPathByteEncodingIsExplicitAndReversible) {
  auto report = SmallReport();
  // These are UTF-8 bytes for e-acute, followed by an arbitrary Linux path
  // byte that is not valid UTF-8. Neither provenance value may be lost.
  report.corpus_path = std::string("\xc3\xa9\xff", 3);
  report.tokenizer_directory = std::string("\xff'\"\\\t", 5);
  auto rendered = RenderReport(report);
  ASSERT_TRUE(rendered.ok()) << rendered.status();
  ExpectPathRoundTrips(rendered->metadata_json, "corpus_path_escaped_bytes",
                       report.corpus_path);
  ExpectPathRoundTrips(rendered->metadata_json,
                       "tokenizer_directory_escaped_bytes",
                       report.tokenizer_directory);
}

TEST(NtkExperimentTest, RejectsMalformedReportsRatherThanWritingInvalidJson) {
  auto report = SmallReport();
  report.kernel.rows = 3;
  EXPECT_FALSE(RenderReport(report).ok());
  report = SmallReport();
  report.initial_values[0] = std::numeric_limits<double>::infinity();
  EXPECT_FALSE(RenderReport(report).ok());
  report = SmallReport();
  report.ridge_predictions.pop_back();
  EXPECT_FALSE(RenderReport(report).ok());
  report = SmallReport();
  report.kernel_steps = 1;
  EXPECT_FALSE(RenderReport(report).ok());
  report = SmallReport();
  report.output_tokens[1] = report.output_tokens[0];
  EXPECT_FALSE(RenderReport(report).ok());
  report = SmallReport();
  report.parameters[1].offset = 4;
  EXPECT_FALSE(RenderReport(report).ok());
  report = SmallReport();
  report.examples[0].tokens.clear();
  EXPECT_FALSE(RenderReport(report).ok());
  report = SmallReport();
  report.windows.train_examples = 2;
  EXPECT_FALSE(RenderReport(report).ok());
}

class NtkReportDirectoryTest : public testing::Test {
 protected:
  void SetUp() override {
    const std::string pattern =
        (std::filesystem::path(testing::TempDir()) / "ntk_report.XXXXXX")
            .string();
    std::vector<char> writable(pattern.begin(), pattern.end());
    writable.push_back('\0');
    const char* created = mkdtemp(writable.data());
    ASSERT_NE(created, nullptr);
    directory_ = created;
  }

  void TearDown() override {
    if (directory_.empty())
      return;
    std::error_code error;
    std::filesystem::remove_all(directory_, error);
    EXPECT_FALSE(error) << error.message();
  }

  std::filesystem::path directory_;
};

TEST_F(NtkReportDirectoryTest,
       CreatesThreeFilesAndNeverOverwritesExistingOutput) {
  auto report = RenderReport(SmallReport());
  ASSERT_TRUE(report.ok()) << report.status();
  const auto output = directory_ / "report";
  EXPECT_TRUE(CheckOutputDirectory(output).ok());
  ASSERT_TRUE(WriteReportDirectory(output, *report).ok());
  for (const char* name : {"report.json", "kernel.csv", "predictions.csv"})
    EXPECT_TRUE(std::filesystem::is_regular_file(output / name));
  EXPECT_EQ(CheckOutputDirectory(output).code(),
            absl::StatusCode::kAlreadyExists);
  EXPECT_EQ(WriteReportDirectory(output, *report).code(),
            absl::StatusCode::kAlreadyExists);
  std::ifstream input(output / "kernel.csv");
  const std::string contents{std::istreambuf_iterator<char>(input),
                             std::istreambuf_iterator<char>()};
  EXPECT_EQ(contents, report->kernel_csv);
}

TEST_F(NtkReportDirectoryTest, RejectsMissingParentsAndDanglingSymlinks) {
  EXPECT_FALSE(CheckOutputDirectory("").ok());
  EXPECT_FALSE(CheckOutputDirectory(directory_ / "missing" / "report").ok());
  std::error_code error;
  std::filesystem::create_symlink(directory_ / "nonexistent",
                                  directory_ / "link", error);
  ASSERT_FALSE(error) << error.message();
  EXPECT_EQ(CheckOutputDirectory(directory_ / "link").code(),
            absl::StatusCode::kAlreadyExists);
}

}  // namespace
}  // namespace pluto::llm::ntk
