#include "src/llm/experiments/path_kernel/report.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "absl/strings/escaping.h"
#include "absl/strings/str_cat.h"
#include "gtest/gtest.h"

namespace pluto::llm::path_kernel {
namespace {

TEST(PathKernelReportTest, QueryTokensAreIndependentOfTrainingLabels) {
  const std::vector<ntk::Example> examples{{"train", 0, {0}, 4, ""},
                                           {"train", 2, {0}, 2, ""},
                                           {"train", 4, {0}, 4, ""},
                                           {"eval", 6, {0}, 7, ""}};
  auto defaults = ResolveQueryTokens(examples, {}, 10);
  ASSERT_TRUE(defaults.ok()) << defaults.status();
  EXPECT_EQ(*defaults, (std::vector<int>{2, 4}));
  auto explicit_tokens = ResolveQueryTokens(examples, {9, 1}, 10);
  ASSERT_TRUE(explicit_tokens.ok()) << explicit_tokens.status();
  EXPECT_EQ(*explicit_tokens, (std::vector<int>{9, 1}));
  for (const std::vector<int> invalid : {std::vector<int>{1, 1}, {-1}, {10}})
    EXPECT_FALSE(ResolveQueryTokens(examples, invalid, 10).ok());
  EXPECT_FALSE(ResolveQueryTokens(examples, {1}, 0).ok());
}

ExperimentReport SmallReport() {
  ExperimentReport report;
  report.checkpoint = "checkpoint/step_15000";
  report.tokenizer_directory = "tokenizer";
  report.corpus_path = "shakespeare.txt";
  report.windows = {3, 1, 1, 2, 0};
  report.dimensions = {8, 16, 4, 1, 8, 1, 8, 32};
  report.padding_token = 7;
  report.steps = 1;
  report.learning_rate = 0.5;
  report.examples = {{"train", 0, {0}, 1, "first"},
                     {"train", 2, {2}, 1, "second"},
                     {"train", 4, {3}, 1, "third"},
                     {"eval", 6, {4}, 6, "held out"}};
  report.output_tokens = {1};
  report.output_token_text = {"token"};
  auto& result = report.result;
  result.parameters = {{0, 2, 0}};
  result.parameter_count = 2;
  result.initial_values = {1, 2, 3, 4};
  result.final_values = {0.125, 6.125, -5.875, 4.125};
  result.reconstructed_values = {0, 6, -6, 4};
  result.residual = {0.125, 0.125, 0.125, 0.125};
  result.final_training_losses = {0.5, 0.25, 0.75};
  result.path_kernel = {
      4, 4, {0.5, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, 0.5}};
  result.contributions = {4, 3, {1, -2, 0, 3, -1, 2, -4, -3, -2, 0, 0, 0}};
  StepResult step;
  step.step = 1;
  step.values_before = result.initial_values;
  step.values_after = result.final_values;
  step.training_losses = {2, 3, 4};
  step.tangent_kernel = {
      4, 4, {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}};
  step.contributions = result.contributions;
  step.predicted_delta = {-1, 4, -9, 0};
  step.residual = result.residual;
  result.steps.push_back(step);
  return report;
}

TEST(PathKernelReportTest, RetainsUnweightedKernelAndSignedReconstruction) {
  auto rendered = RenderReport(SmallReport());
  ASSERT_TRUE(rendered.ok()) << rendered.status();
  EXPECT_EQ(rendered->path_kernel_csv,
            "row,row_0,row_1,row_2,row_3\n0,0.5,0,0,0\n1,0,0.5,0,0\n"
            "2,0,0,0.5,0\n3,0,0,0,0.5\n");
  for (const char* expected :
       {"\"initial_logits\":[1,2,3,4]",
        "\"actual_final_logits\":[0.125,6.125,-5.875,4.125]",
        "\"reconstructed_logits\":[0,6,-6,4]",
        "\"residual\":[0.125,0.125,0.125,0.125]",
        "\"final_training_losses\":[0.5,0.25,0.75]",
        "\"training_losses_before\":[2,3,4]", "\"predicted_delta\":[-1,4,-9,0]",
        "\"flat_output_element\":1,\"output_token_id\":1",
        "mean full-vocabulary last-position cross entropy",
        "not a reconstruction of historical AdamW training"})
    EXPECT_NE(rendered->metadata_json.find(expected), std::string::npos)
        << expected;
  EXPECT_NE(rendered->contributions_csv.find("0,1,-2,\"second\"\n"),
            std::string::npos);
  EXPECT_NE(rendered->predictions_csv.find("0,0,1,\"token\",1,0.125,0,0.125\n"),
            std::string::npos);
}

TEST(PathKernelReportTest, ReportsQueryTokensThatOmitAllTrainingLabels) {
  auto report = SmallReport();
  report.output_tokens = {7};
  auto rendered = RenderReport(report);
  ASSERT_TRUE(rendered.ok()) << rendered.status();
  EXPECT_NE(rendered->metadata_json.find("\"output_token_ids\":[7]"),
            std::string::npos);
  EXPECT_NE(rendered->metadata_json.find("\"output_token_id\":7"),
            std::string::npos);
}

TEST(PathKernelReportTest, RanksSupportingAndOpposingSeparatelyAndOmitsZeros) {
  auto rendered = RenderReport(SmallReport());
  ASSERT_TRUE(rendered.ok()) << rendered.status();
  const auto& json = rendered->metadata_json;
  EXPECT_NE(
      json.find("\"supporting\":[{\"training_example\":0,\"contribution\":3,"
                "\"text_escaped_bytes\":\"first\"},{\"training_example\":2,"
                "\"contribution\":2,\"text_escaped_bytes\":\"third\"}]"),
      std::string::npos);
  EXPECT_NE(
      json.find("\"opposing\":[{\"training_example\":0,\"contribution\":-4,"
                "\"text_escaped_bytes\":\"first\"},{\"training_example\":1,"
                "\"contribution\":-3,\"text_escaped_bytes\":\"second\"},"
                "{\"training_example\":2,\"contribution\":-2,"
                "\"text_escaped_bytes\":\"third\"}]"),
      std::string::npos);
  EXPECT_NE(json.find("\"supporting\":[],\"opposing\":[]"), std::string::npos);
}

TEST(PathKernelReportTest, TiedContributionsKeepCorpusOrder) {
  auto report = SmallReport();
  report.result.contributions(0, 0) = 2;
  report.result.contributions(0, 1) = 2;
  auto rendered = RenderReport(report);
  ASSERT_TRUE(rendered.ok()) << rendered.status();
  EXPECT_NE(rendered->metadata_json.find(
                "\"supporting\":[{\"training_example\":0,\"contribution\":2,"
                "\"text_escaped_bytes\":\"first\"},{\"training_example\":1,"
                "\"contribution\":2,\"text_escaped_bytes\":\"second\"}]"),
            std::string::npos);
}

TEST(PathKernelReportTest, EscapedByteFieldsRoundTripPathsAndTokens) {
  auto report = SmallReport();
  report.corpus_path = std::string("\xc3\xa9\xff\"\\\n", 6);
  report.examples[0].text = "a\n\"b";
  report.output_token_text = {std::string("\x80", 1)};
  auto rendered = RenderReport(report);
  ASSERT_TRUE(rendered.ok()) << rendered.status();
  const auto& json = rendered->metadata_json;
  const std::string prefix = "\"corpus_path_escaped_bytes\":\"";
  const size_t start = json.find(prefix) + prefix.size();
  size_t end = start;
  while (end < json.size() && json[end] != '"')
    end += json[end] == '\\' ? 2 : 1;
  ASSERT_LT(end, json.size());
  std::string escaped, decoded;
  ASSERT_TRUE(absl::CUnescape(
      absl::string_view(json).substr(start, end - start), &escaped));
  ASSERT_TRUE(absl::CUnescape(escaped, &decoded));
  EXPECT_EQ(decoded, report.corpus_path);
  EXPECT_NE(json.find("\\\\200"), std::string::npos);
  for (unsigned char byte : json)
    EXPECT_LT(byte, 128);
  EXPECT_NE(rendered->contributions_csv.find("\"a\\n\\\"\"b\""),
            std::string::npos);
}

TEST(PathKernelReportTest, RejectsMalformedDimensionsAndNonfiniteValues) {
  auto report = SmallReport();
  report.result.path_kernel.rows = 3;
  EXPECT_FALSE(RenderReport(report).ok());
  report = SmallReport();
  report.result.contributions.columns = 2;
  EXPECT_FALSE(RenderReport(report).ok());
  report = SmallReport();
  report.result.steps[0].values_after.pop_back();
  EXPECT_FALSE(RenderReport(report).ok());
  report = SmallReport();
  report.result.steps[0].step = 0;
  EXPECT_FALSE(RenderReport(report).ok());
  report = SmallReport();
  report.result.final_training_losses.clear();
  EXPECT_FALSE(RenderReport(report).ok());
  report = SmallReport();
  report.result.steps[0].tangent_kernel.values[0] =
      std::numeric_limits<double>::quiet_NaN();
  EXPECT_FALSE(RenderReport(report).ok());
  report = SmallReport();
  report.learning_rate = std::numeric_limits<double>::infinity();
  EXPECT_FALSE(RenderReport(report).ok());
  report = SmallReport();
  report.result.residual[0] = std::numeric_limits<double>::infinity();
  EXPECT_FALSE(RenderReport(report).ok());
  report = SmallReport();
  report.result.parameters[0].offset = 1;
  EXPECT_FALSE(RenderReport(report).ok());
}

TEST(PathKernelReportTest, RejectsMissingTargetsInvalidTokensAndSplitOrder) {
  auto report = SmallReport();
  report.examples[0].next_token.reset();
  EXPECT_FALSE(RenderReport(report).ok());
  report = SmallReport();
  report.examples[0].tokens = {8};
  EXPECT_FALSE(RenderReport(report).ok());
  report = SmallReport();
  report.examples[1].split = "eval";
  EXPECT_FALSE(RenderReport(report).ok());
  report = SmallReport();
  report.output_token_text.clear();
  EXPECT_FALSE(RenderReport(report).ok());
  report = SmallReport();
  report.steps = 0;
  EXPECT_FALSE(RenderReport(report).ok());
}

TEST(PathKernelReportTest, WritesNewReportsButNeverOverwritesPaths) {
  const char* temporary = std::getenv("TEST_TMPDIR");
  std::string pattern = absl::StrCat(temporary ? temporary : "/tmp",
                                     "/path-kernel-report-XXXXXX");
  ASSERT_NE(mkdtemp(pattern.data()), nullptr);
  const std::filesystem::path parent(pattern);
  const auto destination = parent / "result";
  auto rendered = RenderReport(SmallReport());
  ASSERT_TRUE(rendered.ok()) << rendered.status();
  ASSERT_TRUE(WriteReportDirectory(destination, *rendered).ok());
  for (const auto& [name, expected] :
       std::vector<std::pair<std::string, std::string>>{
           {"report.json", rendered->metadata_json},
           {"path_kernel.csv", rendered->path_kernel_csv},
           {"contributions.csv", rendered->contributions_csv},
           {"predictions.csv", rendered->predictions_csv}}) {
    std::ifstream input(destination / name);
    const std::string actual((std::istreambuf_iterator<char>(input)), {});
    EXPECT_EQ(actual, expected);
  }
  EXPECT_EQ(WriteReportDirectory(destination, *rendered).code(),
            absl::StatusCode::kAlreadyExists);
  const auto file = parent / "existing-file";
  {
    std::ofstream output(file);
    output << "keep";
  }
  EXPECT_EQ(WriteReportDirectory(file, *rendered).code(),
            absl::StatusCode::kAlreadyExists);
  std::error_code error;
  const auto symlink = parent / "dangling-link";
  std::filesystem::create_symlink(parent / "missing", symlink, error);
  ASSERT_FALSE(error) << error.message();
  EXPECT_EQ(WriteReportDirectory(symlink, *rendered).code(),
            absl::StatusCode::kAlreadyExists);
  EXPECT_FALSE(
      WriteReportDirectory(parent / "missing" / "child", *rendered).ok());
  EXPECT_FALSE(WriteReportDirectory({}, *rendered).ok());
  std::filesystem::remove_all(parent, error);
  EXPECT_FALSE(error) << error.message();
}

}  // namespace
}  // namespace pluto::llm::path_kernel
