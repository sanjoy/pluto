#include "src/llm/experiments/completion_trace/report.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "gtest/gtest.h"

namespace pluto::llm::completion_trace {
namespace {

TensorSnapshot Tensor(std::string scope, std::string name,
                      std::vector<int64_t> dimensions,
                      std::vector<float> values) {
  TensorSnapshot result{.scope = std::move(scope),
                        .name = std::move(name),
                        .data_type = DataType::FP32,
                        .dimensions = std::move(dimensions),
                        .values = std::move(values)};
  result.raw_bytes.resize(result.values.size() * sizeof(float));
  std::memcpy(result.raw_bytes.data(), result.values.data(),
              result.raw_bytes.size());
  return result;
}

ReportMetadata Metadata() {
  return {.checkpoint = "/checkpoints/<step>&one",
          .tokenizer = "/tokenizer",
          .corpus = "/dataset",
          .source_revision = "test-source-revision",
          .model_width = 2,
          .layers = 1,
          .heads = 1,
          .feed_forward_width = 4,
          .eos_token = 3,
          .original_ids = {10, 20, 30, 40},
          .token_text = {"A", " B", "<C>&\"'\n", "<EOS>"}};
}

CompletionExample Example() {
  CompletionStep step{
      .forward = {.prefix = {0, 1},
                  .activations = {Tensor("gpt2", "PositionEmbeddingLayer",
                                         {1, 2, 2},
                                         {1, -0.0f, 1.00000012f, -2.5f}),
                                  Tensor("gpt2", "LanguageModelingHeadLayer",
                                         {1, 2, 4},
                                         {-1, -2, -3, -4, 0, 1, 3, -4})},
                  .attention = {Tensor(
                      "gpt2/transformer_block_0/ResidualLayer/attention",
                      "AttentionLayer", {1, 1, 2, 2}, {1, 0, .25f, .75f})},
                  .next_logits = {0, 1, 3, -4}},
      .predicted = 2,
      .expected = 2};
  return {.corpus_line = 80,
          .sentence = "A B<C>&\"'\n",
          .prompt = {0, 1},
          .steps = {std::move(step)}};
}

std::string Read(const std::filesystem::path& path) {
  std::ifstream file(path);
  return {std::istreambuf_iterator<char>(file),
          std::istreambuf_iterator<char>()};
}

std::string Hex(absl::Span<const uint8_t> bytes) {
  std::ostringstream out;
  for (uint8_t byte : bytes)
    out << std::hex << std::setw(2) << std::setfill('0') << unsigned(byte);
  return out.str();
}

class CompletionReportTest : public testing::Test {
 protected:
  void SetUp() override {
    std::string pattern = testing::TempDir() + "/completion-report-XXXXXX";
    const char* created = mkdtemp(pattern.data());
    ASSERT_NE(created, nullptr);
    directory_ = created;
  }
  void TearDown() override {
    if (directory_.empty())
      return;
    // This directory was freshly created by this test, never supplied by a
    // caller. Removing its own generated artifacts cannot touch user reports.
    std::error_code error;
    std::filesystem::remove_all(directory_, error);
    EXPECT_FALSE(error) << error.message();
  }
  std::filesystem::path directory_;
};

TEST(CompletionReportCosineTest, KnownValuesAndUndefinedInputs) {
  EXPECT_EQ(CosineSimilarity({1, 0}, {0, 1}), 0.0);
  EXPECT_EQ(CosineSimilarity({1, 2}, {2, 4}), 1.0);
  EXPECT_EQ(CosineSimilarity({1, 2}, {-2, -4}), -1.0);
  EXPECT_FALSE(CosineSimilarity({}, {}));
  EXPECT_FALSE(CosineSimilarity({1}, {1, 2}));
  EXPECT_FALSE(CosineSimilarity({0, 0}, {1, 2}));
  EXPECT_FALSE(CosineSimilarity({std::numeric_limits<float>::infinity()}, {1}));
  EXPECT_FALSE(
      CosineSimilarity({1}, {std::numeric_limits<float>::quiet_NaN()}));
  const auto large = CosineSimilarity({1.0e30f, 1.0e30f}, {1.0e30f, 1.0e30f});
  ASSERT_TRUE(large);
  EXPECT_DOUBLE_EQ(*large, 1.0);
}

TEST_F(CompletionReportTest, WritesCompleteEscapedValuesAndRawBytes) {
  const auto example = Example();
  auto status = WriteReports(directory_, Metadata(), {&example, 1});
  ASSERT_TRUE(status.ok()) << status;
  const auto text = Read(directory_ / "report.txt");
  const auto html = Read(directory_ / "report.html");
  const auto vocabulary = Read(directory_ / "vocabulary.tsv");
  EXPECT_NE(text.find("test-source-revision"), std::string::npos);
  EXPECT_NE(text.find("1.00000012, -2.5"), std::string::npos);
  EXPECT_NE(text.find("dtype=FP32 shape=[1,2,2]"), std::string::npos);
  EXPECT_NE(text.find("Position 1 (input \"A\", compact ID 0)"),
            std::string::npos);
  EXPECT_NE(text.find("Position 2 (input \" B\", compact ID 1)"),
            std::string::npos);
  EXPECT_NE(text.find("raw storage bytes (hex, element order)"),
            std::string::npos);
  const auto& first = example.steps[0].forward.activations[0];
  EXPECT_NE(text.find(Hex(absl::MakeConstSpan(first.raw_bytes).subspan(4, 4))),
            std::string::npos);  // Exact signed-zero bytes.
  EXPECT_NE(html.find("&lt;C&gt;&amp;"), std::string::npos);
  EXPECT_NE(html.find("/checkpoints/&lt;step&gt;&amp;one"), std::string::npos);
  EXPECT_EQ(html.find("<C>"), std::string::npos);
  EXPECT_NE(html.find("<details>"), std::string::npos);
  EXPECT_NE(vocabulary.find("2\t30\t\"<C>&\\\"\\'\\n\""), std::string::npos);
  EXPECT_EQ(std::count(vocabulary.begin(), vocabulary.end(), '\n'), 5);
  const auto completions = Read(directory_ / "completions.tsv");
  EXPECT_NE(completions.find("80\t1\t3\t2\t30\t"), std::string::npos);
  EXPECT_NE(text.find("predicted NEXT token, not by the current input token"),
            std::string::npos);
  EXPECT_NE(text.find("Attention(qkv)*W_o+b_o"), std::string::npos);
}

TEST_F(CompletionReportTest, PrintsOnlyCausalAttentionEntriesAndAllHeads) {
  auto example = Example();
  auto& attention = example.steps[0].forward.attention[0];
  attention = Tensor(attention.scope, attention.name, {1, 2, 2, 2},
                     {1, 0, .25f, .75f, 1, 0, .6f, .4f});
  auto metadata = Metadata();
  metadata.heads = 2;
  ASSERT_TRUE(WriteReports(directory_, metadata, {&example, 1}).ok());
  const auto text = Read(directory_ / "report.txt");
  const auto start = text.find("AttentionLayer[0]/output0 dtype=FP32");
  ASSERT_NE(start, std::string::npos);
  const auto causal = text.substr(start);
  EXPECT_NE(causal.find("Head 0:"), std::string::npos);
  EXPECT_NE(causal.find("Head 1:"), std::string::npos);
  EXPECT_NE(causal.find("  [0] 0.25, 0.75\n"), std::string::npos);
  EXPECT_EQ(causal.find("  [0] 1, 0\n"), std::string::npos);
  EXPECT_NE(causal.find("  [0] 1\n"), std::string::npos);
  const auto rows = Read(directory_ / "query_attention.tsv");
  EXPECT_EQ(std::count(rows.begin(), rows.end(), '\n'), 5);
}

TEST_F(CompletionReportTest, AcceptsGrowingPrefixesWithConstantSiteShapes) {
  auto example = Example();
  auto step = example.steps[0];
  step.forward.prefix.push_back(step.predicted);
  step.forward.activations = {
      Tensor("gpt2", "PositionEmbeddingLayer", {1, 3, 2}, {1, 0, 2, 0, 3, 0}),
      Tensor("gpt2", "LanguageModelingHeadLayer", {1, 3, 4},
             {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 4})};
  step.forward.attention = {Tensor(example.steps[0].forward.attention[0].scope,
                                   "AttentionLayer", {1, 1, 3, 3},
                                   {1, 0, 0, .5f, .5f, 0, .25f, .25f, .5f})};
  step.forward.next_logits = {0, 1, 2, 4};
  step.predicted = 3;
  step.expected = 3;
  example.steps.push_back(std::move(step));
  auto status = WriteReports(directory_, Metadata(), {&example, 1});
  ASSERT_TRUE(status.ok()) << status;
  EXPECT_NE(Read(directory_ / "report.txt").find("Position 3 [generated]"),
            std::string::npos);
}

TEST_F(CompletionReportTest, RejectsChangingSiteWidthBeforeWritingFiles) {
  auto first = Example();
  auto second = Example();
  second.corpus_line = 406;
  second.steps[0].forward.activations[0] =
      Tensor("gpt2", "PositionEmbeddingLayer", {1, 2, 1}, {1, 2});
  const std::vector<CompletionExample> examples{first, second};
  const auto status = WriteReports(directory_, Metadata(), examples);
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_NE(status.message().find("non-prefix dimensions"), std::string::npos);
  EXPECT_TRUE(std::filesystem::is_empty(directory_));
}

TEST_F(CompletionReportTest, RejectsChangingAttentionHeadCount) {
  auto first = Example();
  auto second = Example();
  second.corpus_line = 406;
  auto& attention = second.steps[0].forward.attention[0];
  attention = Tensor(attention.scope, attention.name, {1, 2, 2, 2},
                     {1, 0, .25f, .75f, 1, 0, .25f, .75f});
  const auto status = WriteReports(directory_, Metadata(), {first, second});
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(std::filesystem::is_empty(directory_));
}

TEST_F(CompletionReportTest, ReportsBf16BitsAndRejectsChangingSiteDtype) {
  const auto first = Example();
  auto second = Example();
  second.corpus_line = 406;
  auto& tensor = second.steps[0].forward.activations[0];
  tensor.data_type = DataType::BF16;
  tensor.values = {1, -0.0f, 1, -2.5f};
  const std::vector<uint16_t> bits{0x3f80, 0x8000, 0x3f80, 0xc020};
  tensor.raw_bytes.resize(bits.size() * sizeof(uint16_t));
  std::memcpy(tensor.raw_bytes.data(), bits.data(), tensor.raw_bytes.size());
  EXPECT_EQ(WriteReports(directory_, Metadata(), {first, second}).code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(std::filesystem::is_empty(directory_));
  ASSERT_TRUE(WriteReports(directory_, Metadata(), {&second, 1}).ok());
  const auto text = Read(directory_ / "report.txt");
  EXPECT_NE(text.find("dtype=BF16 shape=[1,2,2]"), std::string::npos);
  EXPECT_NE(text.find(Hex(absl::MakeConstSpan(tensor.raw_bytes).subspan(2, 2))),
            std::string::npos);
}

TEST_F(CompletionReportTest, RejectsMalformedTokensShapesArgmaxAndAttention) {
  const auto metadata = Metadata();
  auto invalid = [&](const CompletionExample& example) {
    EXPECT_EQ(WriteReports(directory_, metadata, {&example, 1}).code(),
              absl::StatusCode::kInvalidArgument);
    EXPECT_TRUE(std::filesystem::is_empty(directory_));
  };
  auto example = Example();
  example.prompt[0] = -1;
  invalid(example);
  example = Example();
  example.steps[0].forward.prefix[1] = 2;
  invalid(example);
  example = Example();
  example.steps[0].predicted = 1;
  invalid(example);
  example = Example();
  example.steps[0].forward.next_logits[2] =
      std::numeric_limits<float>::infinity();
  invalid(example);
  example = Example();
  example.steps[0].forward.activations[0].raw_bytes.pop_back();
  invalid(example);
  example = Example();
  example.steps[0].forward.activations[0].dimensions[2] =
      std::numeric_limits<int64_t>::max();
  invalid(example);
  example = Example();
  example.steps[0].forward.attention[0].values[1] = .01f;
  invalid(example);
  example = Example();
  example.steps[0].forward.attention[0].values[3] = .25f;
  invalid(example);
  example = Example();
  example.steps[0].forward.next_logits = {0, 1, 2, 4};
  example.steps[0].predicted = 3;
  example.steps.push_back(example.steps[0]);
  invalid(example);
}

TEST_F(CompletionReportTest, AllowsCapturedSoftmaxReconstructionTolerance) {
  auto example = Example();
  // TraceForward permits this tiny FP32 normalization discrepancy. Reports
  // must preserve the observed value rather than impose a different limit.
  auto& attention = example.steps[0].forward.attention[0];
  attention.values[0] = 1.00005f;
  std::memcpy(attention.raw_bytes.data(), attention.values.data(),
              attention.raw_bytes.size());
  EXPECT_TRUE(WriteReports(directory_, Metadata(), {&example, 1}).ok());
}

TEST_F(CompletionReportTest, RefusesToOverwriteExistingArtifacts) {
  const auto example = Example();
  ASSERT_TRUE(WriteReports(directory_, Metadata(), {&example, 1}).ok());
  const auto before = Read(directory_ / "report.txt");
  EXPECT_EQ(WriteReports(directory_, Metadata(), {&example, 1}).code(),
            absl::StatusCode::kAlreadyExists);
  EXPECT_EQ(Read(directory_ / "report.txt"), before);
}

TEST_F(CompletionReportTest, CorrelationsGroupByPredictedTokenAtEqualStep) {
  auto first = Example();
  auto second = Example();
  auto third = Example();
  second.corpus_line = 406;
  third.corpus_line = 411;
  first.steps[0].forward.activations[0] =
      Tensor("gpt2", "PositionEmbeddingLayer", {1, 2, 2}, {0, 0, 1, 0});
  second.steps[0].forward.activations[0] =
      Tensor("gpt2", "PositionEmbeddingLayer", {1, 2, 2}, {0, 0, 0, 1});
  third.steps[0].forward.activations[0] =
      Tensor("gpt2", "PositionEmbeddingLayer", {1, 2, 2}, {0, 0, -1, 0});
  third.steps[0].forward.next_logits = {0, 3, 1, -4};
  third.steps[0].predicted = 1;
  const auto status =
      WriteReports(directory_, Metadata(), {first, second, third});
  ASSERT_TRUE(status.ok()) << status;
  const auto pairs = Read(directory_ / "correlations.tsv");
  EXPECT_NE(
      pairs.find("predicted_token_a\tpredicted_token_b\tsame_predicted_token"),
      std::string::npos);
  EXPECT_NE(pairs.find("\t1\t80\t406\t2\t2\t1\t0\t"), std::string::npos);
  EXPECT_NE(pairs.find("\t1\t80\t411\t2\t1\t0\t-1\t"), std::string::npos);
  EXPECT_NE(pairs.find("\t1\t406\t411\t2\t1\t0\t0\t"), std::string::npos);
  const auto summary = Read(directory_ / "correlation_summary.tsv");
  EXPECT_NE(summary.find(
                "gpt2/PositionEmbeddingLayer[0]/output0\t3\t1\t0\t2\t-0.5\t"),
            std::string::npos);
}

TEST_F(CompletionReportTest,
       WalkthroughShowsAllQueryFeaturesWithoutWideLogits) {
  auto example = Example();
  auto metadata = Metadata();
  metadata.model_width = 16;
  metadata.feed_forward_width = 64;
  auto feature = [](std::string scope, std::string name, int width,
                    float start) {
    std::vector<float> values(2 * width, -1111);
    std::iota(values.begin() + width, values.end(), start);
    return Tensor(std::move(scope), std::move(name), {1, 2, width},
                  std::move(values));
  };
  example.steps[0].forward.activations = {
      feature("gpt2", "PositionEmbeddingLayer", 16, 0),
      feature("gpt2/transformer_block_0/ResidualLayer/attention",
              "FullyConnectedLayer", 48, 100),
      feature("gpt2/transformer_block_0/ResidualLayer/mlp",
              "FullyConnectedLayer", 64, 200),
      example.steps[0].forward.activations.back()};
  ASSERT_TRUE(WriteReports(directory_, metadata, {&example, 1}).ok());
  const auto short_report = Read(directory_ / "query_walkthrough.txt");
  EXPECT_NE(short_report.find(
                "See report.txt for ALL positions, ALL vocabulary logits"),
            std::string::npos);
  EXPECT_NE(
      short_report.find(
          "Top next-token candidates (all 4 logical tokens in denominator)"),
      std::string::npos);
  EXPECT_NE(short_report.find("Last active input position 2 (\" B\")"),
            std::string::npos);
  EXPECT_NE(short_report.find("channels=16"), std::string::npos);
  EXPECT_NE(short_report.find("  [8] 8, 9, 10, 11, 12, 13, 14, 15\n"),
            std::string::npos);
  EXPECT_NE(
      short_report.find("  [40] 140, 141, 142, 143, 144, 145, 146, 147\n"),
      std::string::npos);
  EXPECT_NE(
      short_report.find("  [56] 256, 257, 258, 259, 260, 261, 262, 263\n"),
      std::string::npos);
  EXPECT_NE(short_report.find("x*W_qkv+b_qkv; columns are Q, then K, then V"),
            std::string::npos);
  EXPECT_NE(short_report.find("MLP expansion: normalized residual * W_1 + b_1"),
            std::string::npos);
  EXPECT_NE(short_report.find("key position 1 \"A\" (compact 0): 0.25\n"),
            std::string::npos);
  EXPECT_NE(short_report.find("key position 2 \" B\" (compact 1): 0.75\n"),
            std::string::npos);
  EXPECT_EQ(short_report.find("-1111"), std::string::npos);
  EXPECT_EQ(short_report.find("LanguageModelingHeadLayer"), std::string::npos);
  EXPECT_EQ(short_report.find("raw storage bytes (hex"), std::string::npos);
}

}  // namespace
}  // namespace pluto::llm::completion_trace
