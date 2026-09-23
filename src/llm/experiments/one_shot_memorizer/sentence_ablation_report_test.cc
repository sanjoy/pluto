#include "src/llm/experiments/one_shot_memorizer/sentence_ablation_report.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <sstream>
#include <string>
#include <system_error>
#include <vector>

#include "gtest/gtest.h"

namespace pluto::llm::one_shot_memorizer {
namespace {

class SentenceAblationReportTest : public testing::Test {
 protected:
  void SetUp() override {
    const auto pattern = (std::filesystem::path(testing::TempDir()) /
                          "sentence_ablation_report.XXXXXX")
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
    // Only remove the unique fixture directory successfully created above.
    std::error_code error;
    std::filesystem::remove_all(directory_, error);
    EXPECT_FALSE(error) << error.message();
  }

  absl::StatusOr<ParameterDeltaReport> Report(size_t top_count = 1) {
    const std::vector<TensorSpec> layout{{.checkpoint_index = 0,
                                          .name = "embedding",
                                          .shape = {2, 2},
                                          .flat_offset = 0,
                                          .element_count = 4,
                                          .token_embedding = true},
                                         {.checkpoint_index = 1,
                                          .name = "block.0.qkv",
                                          .shape = {2, 6},
                                          .flat_offset = 4,
                                          .element_count = 12,
                                          .qkv_width = 2},
                                         {.checkpoint_index = 2,
                                          .name = "bias",
                                          .shape = {2},
                                          .flat_offset = 16,
                                          .element_count = 2}};
    const std::vector<float> embedding_before{0.0f, 2, 0, 0};
    const std::vector<float> embedding_after{-0.0f, 1, 0, 0};
    std::vector<float> qkv_before(12, 0), qkv_after(12, 0);
    qkv_before[3] = 5;
    qkv_after[3] = 3;
    qkv_before[11] = -1;
    qkv_after[11] = 1;
    const std::vector<float> bias_before{0, 4}, bias_after{0, 1};
    const std::vector<absl::Span<const float>> before{embedding_before,
                                                      qkv_before, bias_before};
    const std::vector<absl::Span<const float>> after{embedding_after, qkv_after,
                                                     bias_after};
    return CompareParameterValues(layout, before, after, top_count);
  }

  std::string Read(const char* name) {
    std::ifstream input(directory_ / name, std::ios::binary);
    EXPECT_TRUE(input.good());
    return std::string(std::istreambuf_iterator<char>(input),
                       std::istreambuf_iterator<char>());
  }

  void Put(const char* name, const std::string& contents) {
    std::ofstream output(directory_ / name, std::ios::binary);
    output << contents;
    ASSERT_TRUE(output.good());
  }

  std::filesystem::path directory_;
};

TEST_F(SentenceAblationReportTest, WritesAllChangedCoordinatesNotOnlyTopLists) {
  auto report = Report();
  ASSERT_TRUE(report.ok()) << report.status();
  ASSERT_EQ(report->total.bitwise_changed_count, 5u);
  ASSERT_EQ(report->total.numerically_changed_count, 4u);
  ASSERT_EQ(report->tensors[0].top_coordinates.size(), 1u);
  const auto status = WriteSentenceAblationReport(*report, directory_);
  ASSERT_TRUE(status.ok()) << status;
  const auto coordinates = Read("coordinates.tsv");
  // Signed-zero change is retained although not in this tensor's top-1 list.
  EXPECT_NE(coordinates.find("embedding\t0\t0\t0\t0\t0\t0\tnone\t\t0\t1\n"),
            std::string::npos);
  EXPECT_NE(coordinates.find("embedding\t0\t1\t1\t0\t1\t0\tnone\t\t1\t1\n"),
            std::string::npos);
  EXPECT_NE(coordinates.find("block.0.qkv\t1\t3\t7\t0\t3\t\tkey\t1\t2\t1\n"),
            std::string::npos);
  EXPECT_NE(
      coordinates.find("block.0.qkv\t1\t11\t15\t1\t5\t\tvalue\t1\t-2\t1\n"),
      std::string::npos);
  EXPECT_NE(coordinates.find("bias\t2\t1\t17\t1\t\t\tnone\t\t3\t1\n"),
            std::string::npos);
  std::istringstream lines(coordinates);
  std::string line;
  size_t records = 0;
  while (std::getline(lines, line))
    if (!line.empty() && line[0] != '#')
      ++records;
  EXPECT_EQ(records, 6u);  // Header plus five bitwise-changed coordinates.
  const auto table = Read("per_tensor.tsv");
  EXPECT_NE(table.find("0\tembedding\t2x2\t4\t2\t1\t"), std::string::npos);
  const auto html = Read("report.html");
  EXPECT_NE(html.find("baseline minus ablated"), std::string::npos);
  EXPECT_NE(html.find("not evidence of exclusive ownership"),
            std::string::npos);
  EXPECT_NE(html.find("<!doctype html>"), std::string::npos);
  EXPECT_EQ(html.find("<script src="), std::string::npos);
  EXPECT_EQ(html.find("<link"), std::string::npos);
}

TEST_F(SentenceAblationReportTest, EscapesHtmlAndTsvNames) {
  auto report = Report();
  ASSERT_TRUE(report.ok()) << report.status();
  report->tensors[0].tensor.name = "<script>\"x\"&'</script>\t\\\n";
  ASSERT_TRUE(WriteSentenceAblationReport(*report, directory_).ok());
  const auto html = Read("report.html");
  EXPECT_EQ(html.find("<script>"), std::string::npos);
  EXPECT_NE(html.find("&lt;script&gt;&quot;x&quot;&amp;&#39;&lt;/script&gt;"),
            std::string::npos);
  const auto table = Read("per_tensor.tsv");
  EXPECT_NE(table.find("<script>\"x\"&'</script>\\t\\\\\\n\t2x2"),
            std::string::npos);
}

TEST_F(SentenceAblationReportTest,
       SupportsNoTopCoordinatesAndZeroBaselineNorm) {
  const std::vector<TensorSpec> layout{
      {.name = "zero", .shape = {1}, .element_count = 1}};
  const std::vector<float> before{0}, after{-0.0f};
  const std::vector<absl::Span<const float>> before_tensors{before},
      after_tensors{after};
  auto report =
      CompareParameterValues(layout, before_tensors, after_tensors, 0);
  ASSERT_TRUE(report.ok()) << report.status();
  ASSERT_TRUE(WriteSentenceAblationReport(*report, directory_).ok());
  EXPECT_NE(Read("report.html").find("undefined"), std::string::npos);
  EXPECT_NE(
      Read("coordinates.tsv").find("zero\t0\t0\t0\t0\t\t\tnone\t\t0\t1\n"),
      std::string::npos);
}

TEST_F(SentenceAblationReportTest,
       RejectsMalformedReportsBeforeTruncatingFiles) {
  auto valid = Report();
  ASSERT_TRUE(valid.ok()) << valid.status();
  Put("per_tensor.tsv", "keep table");
  Put("coordinates.tsv", "keep coordinates");
  Put("report.html", "keep html");
  auto invalid = [&](const ParameterDeltaReport& report) {
    EXPECT_FALSE(WriteSentenceAblationReport(report, directory_).ok());
    EXPECT_EQ(Read("per_tensor.tsv"), "keep table");
    EXPECT_EQ(Read("coordinates.tsv"), "keep coordinates");
    EXPECT_EQ(Read("report.html"), "keep html");
  };
  auto malformed = *valid;
  malformed.tensors[0].deltas.pop_back();
  invalid(malformed);
  malformed = *valid;
  malformed.tensors[0].bitwise_changed.pop_back();
  invalid(malformed);
  malformed = *valid;
  malformed.tensors[0].deltas[0] = std::numeric_limits<double>::infinity();
  invalid(malformed);
  malformed = *valid;
  malformed.tensors[0].top_coordinates[0].coordinate.row = 99;
  invalid(malformed);
  malformed = *valid;
  ++malformed.total.bitwise_changed_count;
  invalid(malformed);
  malformed = *valid;
  malformed.tensors[0].summary.delta_l2 = -1;
  invalid(malformed);
}

TEST_F(SentenceAblationReportTest,
       RequiresExistingDirectoryAndRejectsOutputSymlinks) {
  auto report = Report();
  ASSERT_TRUE(report.ok()) << report.status();
  EXPECT_FALSE(
      WriteSentenceAblationReport(*report, directory_ / "missing").ok());
  EXPECT_FALSE(WriteSentenceAblationReport(*report, {}).ok());
  Put("not_a_directory", "keep");
  EXPECT_FALSE(
      WriteSentenceAblationReport(*report, directory_ / "not_a_directory")
          .ok());
  std::error_code error;
  std::filesystem::create_symlink(directory_ / "not_a_directory",
                                  directory_ / "coordinates.tsv", error);
  ASSERT_FALSE(error) << error.message();
  EXPECT_FALSE(WriteSentenceAblationReport(*report, directory_).ok());
  EXPECT_EQ(Read("not_a_directory"), "keep");
  EXPECT_FALSE(std::filesystem::exists(directory_ / "per_tensor.tsv", error));
  EXPECT_FALSE(error) << error.message();
}

TEST_F(SentenceAblationReportTest, ReplacesOnlyTheThreeNamedArtifacts) {
  auto report = Report();
  ASSERT_TRUE(report.ok()) << report.status();
  Put("per_tensor.tsv", "old table");
  Put("coordinates.tsv", "old coordinates");
  Put("report.html", "old html");
  Put("notes.txt", "unrelated user notes");
  ASSERT_TRUE(WriteSentenceAblationReport(*report, directory_).ok());
  EXPECT_NE(Read("per_tensor.tsv"), "old table");
  EXPECT_NE(Read("coordinates.tsv"), "old coordinates");
  EXPECT_NE(Read("report.html"), "old html");
  EXPECT_EQ(Read("notes.txt"), "unrelated user notes");
}

}  // namespace
}  // namespace pluto::llm::one_shot_memorizer
