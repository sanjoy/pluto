#include "src/llm/experiments/one_shot_memorizer/quadratic_probe_artifacts.h"

#include <bit>
#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

#include "gtest/gtest.h"

namespace pluto::llm::one_shot_memorizer {
namespace {

constexpr char kCoefficientHeader[] =
    "condition\tblock\ttensor\tflat_index\tfp32_master\n";
constexpr char kFailureHeader[] =
    "condition\tselection\tline_1based\tgroup\ttarget_position\t"
    "query_position\tpredicted_token\texpected_token\tgold_prefix_ids\n";

std::string Coefficients() {
  std::ostringstream out;
  out << kCoefficientHeader;
  // Reverse block/coordinate ordering checks that file order is not assumed.
  for (int block = 7; block >= 0; --block) {
    for (int index = 2431; index >= 0; --index)
      out << "quadratic_refit\t" << block << "\tW2\t" << index << '\t'
          << block * 3000 + index << '\n';
    for (int index = 15; index >= 0; --index)
      out << "quadratic_refit\t" << block << "\tb2\t" << index << '\t'
          << -block - index * 0.25 << '\n';
  }
  return out.str();
}

std::string ReplaceFirstRow(std::string text, const std::string& row) {
  const size_t begin = text.find('\n') + 1;
  const size_t end = text.find('\n', begin);
  text.replace(begin, end - begin, row);
  return text;
}

bool CoefficientsOk(const std::string& text) {
  std::istringstream input(text);
  return ReadQuadraticCoefficients(input).ok();
}

bool FailuresOk(const std::string& text) {
  std::istringstream input(text);
  return ReadQuadraticFailures(input).ok();
}

TEST(QuadraticArtifactsTest,
     ReadsAllCoefficientsInAnyOrderAndIgnoresValidControls) {
  std::istringstream input(Coefficients() +
                           "learned_w1_refit\t0\tW1\t1023\t1.25e-2\n"
                           "original_mlp_clone\t7\tb1\t63\t-0\n"
                           "initial_w1_raw\t3\tW2\t1023\t0\n"
                           "initial_w1_standardized\t2\tb2\t15\t0.5\n");
  const auto result = ReadQuadraticCoefficients(input);
  ASSERT_TRUE(result.ok()) << result.status();
  for (int block = 0; block < 8; ++block) {
    ASSERT_EQ((*result)[block].weights.size(), 2432u);
    ASSERT_EQ((*result)[block].bias.size(), 16u);
    for (int index = 0; index < 2432; ++index)
      EXPECT_FLOAT_EQ((*result)[block].weights[index], block * 3000 + index);
    for (int index = 0; index < 16; ++index)
      EXPECT_FLOAT_EQ((*result)[block].bias[index], -block - index * 0.25);
  }
}

TEST(QuadraticArtifactsTest, RequiresExactHeaderAndCompleteUniqueCoordinates) {
  const auto text = Coefficients();
  EXPECT_FALSE(CoefficientsOk(""));
  EXPECT_FALSE(CoefficientsOk(std::string(kCoefficientHeader)));
  EXPECT_FALSE(CoefficientsOk(" " + text));
  EXPECT_FALSE(
      CoefficientsOk(text.substr(0, text.rfind('\n', text.size() - 2) + 1)));
  EXPECT_FALSE(CoefficientsOk(text + "quadratic_refit\t0\tW2\t0\t1\n"));
  EXPECT_FALSE(CoefficientsOk(text + "quadratic_refit\t00\tb2\t00\t1\n"));
  // A full last row need not have a newline; a truncated numeric field cannot
  // pass.
  EXPECT_TRUE(CoefficientsOk(text.substr(0, text.size() - 1)));
  EXPECT_FALSE(CoefficientsOk(text + "quadratic_refit\t0\tb2\t0\t"));
}

TEST(QuadraticArtifactsTest,
     RejectsNonfiniteOverflowAndPartiallyParsedCoefficients) {
  const auto text = Coefficients();
  for (const std::string bad : {"nan", "inf", "-inf", "1e100", "1e-100", "",
                                " 1", "1 ", "1.5oops", "0x1p0", "+1"}) {
    SCOPED_TRACE(bad);
    EXPECT_FALSE(CoefficientsOk(
        ReplaceFirstRow(text, "quadratic_refit\t7\tW2\t2431\t" + bad)));
    // Ignored controls must not hide malformed or nonfinite coefficients.
    EXPECT_FALSE(
        CoefficientsOk(text + "learned_w1_refit\t0\tb2\t0\t" + bad + "\n"));
  }
  std::istringstream negative_zero(
      ReplaceFirstRow(text, "quadratic_refit\t7\tW2\t2431\t-0"));
  const auto result = ReadQuadraticCoefficients(negative_zero);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(std::bit_cast<uint32_t>((*result)[7].weights.back()), 0x80000000u);
}

TEST(QuadraticArtifactsTest, RejectsUnknownFieldsAndBadCoefficientIndices) {
  const auto text = Coefficients();
  for (const std::string row :
       {"unknown\t0\tW2\t0\t1", "quadratic_refit\t8\tW2\t0\t1",
        "quadratic_refit\t-1\tW2\t0\t1", "quadratic_refit\t0\tW1\t0\t1",
        "quadratic_refit\t0\tb1\t0\t1", "quadratic_refit\t0\tW3\t0\t1",
        "quadratic_refit\t0\tW2\t2432\t1", "quadratic_refit\t0\tb2\t16\t1",
        "quadratic_refit\t0\tW2\t9999999999999999999999\t1",
        "quadratic_refit\t0\tW2\t+1\t1", "quadratic_refit\t0\tW2\t1.0\t1",
        "quadratic_refit\t0\tW2\t 1\t1", "quadratic_refit\t0\tW2\t0\t1\textra",
        "learned_w1_refit\t0\tW2\t1024\t1", "learned_w1_refit\t0\tb1\t64\t1",
        "learned_w1_refit\t0\tb3\t0\t1", ""}) {
    SCOPED_TRACE(row);
    EXPECT_FALSE(CoefficientsOk(text + row + "\n"));
  }
}

TEST(QuadraticArtifactsTest, SelectsJointFailuresAndPreservesFileOrder) {
  std::istringstream input(
      std::string(kFailureHeader) +
      "quadratic_refit\t0\t1\theld\t5\t4\t1\t2\t0,1,2,3,4\n"
      "initial_w1_raw\tall\t1\theld\t5\t4\t1\t2\t0,1,2,3,4\n"
      "quadratic_refit\tall\t2\tfit\t6\t5\t7\t4474\t0,1,2,3,4,5\n"
      "quadratic_refit\tall\t1\theld\t5\t4\t4474\t2\t0,1,2,3,4\n");
  const auto result = ReadQuadraticFailures(input);
  ASSERT_TRUE(result.ok()) << result.status();
  ASSERT_EQ(result->size(), 2u);
  EXPECT_EQ((*result)[0].line, 2u);
  EXPECT_EQ((*result)[0].target_position, 6);
  EXPECT_EQ((*result)[0].predicted_token, 7);
  EXPECT_EQ((*result)[0].expected_token, 4474);
  EXPECT_EQ((*result)[0].prefix, (std::vector<int>{0, 1, 2, 3, 4, 5}));
  EXPECT_EQ((*result)[1].line, 1u);
  EXPECT_EQ((*result)[1].predicted_token, 4474);
}

TEST(QuadraticArtifactsTest, AllowsContextBoundaryAndDoesNotRequire28Failures) {
  std::ostringstream out;
  out << kFailureHeader
      << "quadratic_refit\tall\t1024\tfit\t1024\t1023\t0\t4474\t";
  for (int i = 0; i < 1024; ++i)
    out << (i == 0 ? "" : ",") << i;
  EXPECT_TRUE(FailuresOk(out.str()));
}

TEST(QuadraticArtifactsTest,
     RequiresFailureHeaderSelectionAndUniqueCorpusLines) {
  const std::string row =
      "quadratic_refit\tall\t1\theld\t5\t4\t1\t2\t0,1,2,3,4\n";
  EXPECT_FALSE(FailuresOk(""));
  EXPECT_FALSE(FailuresOk(kFailureHeader));
  EXPECT_FALSE(FailuresOk("bad header\n" + row));
  EXPECT_FALSE(FailuresOk(std::string(kFailureHeader) + row + row));
  EXPECT_FALSE(
      FailuresOk(std::string(kFailureHeader) + row +
                 "quadratic_refit\tall\t01\theld\t5\t4\t1\t2\t0,1,2,3,4\n"));
  EXPECT_FALSE(
      FailuresOk(std::string(kFailureHeader) +
                 "quadratic_refit\t7\t1\theld\t5\t4\t1\t2\t0,1,2,3,4\n"));
  EXPECT_FALSE(
      FailuresOk(std::string(kFailureHeader) + row + "quadratic_refit\tall\t"));
}

TEST(QuadraticArtifactsTest, RejectsMalformedFailureFieldsEvenOnIgnoredRows) {
  const std::string valid =
      std::string(kFailureHeader) +
      "quadratic_refit\tall\t1\theld\t5\t4\t1\t2\t0,1,2,3,4\n";
  const std::vector<std::string> fields{
      "quadratic_refit", "all", "2", "fit", "5", "4", "1", "2", "0,1,2,3,4"};
  const std::vector<std::pair<int, std::string>> invalid{
      {0, "unknown"},
      {1, "8"},
      {1, "-1"},
      {1, "All"},
      {2, "0"},
      {2, "1025"},
      {2, "99999999999999999999"},
      {2, " 2"},
      {3, "held"},
      {3, "train"},
      {4, "4"},
      {4, "1025"},
      {4, "6"},
      {5, "3"},
      {5, "5"},
      {6, "4475"},
      {6, "-1"},
      {6, "2"},
      {7, "4475"},
      {7, "1"},
      {8, "0,1,2,3"},
      {8, "0,1,2,3,4,5"},
      {8, "0,1,2,,4"},
      {8, "0,1,2,3,4,"},
      {8, "0,1,2,3,4475"},
      {8, "0,1,2,3,-1"},
      {8, "0,1,2,3, 4"},
      {8, "0,1,2,3,4oops"},
      {8, ""}};
  for (const auto& [index, bad] : invalid) {
    SCOPED_TRACE(index);
    SCOPED_TRACE(bad);
    for (bool ignored : {false, true}) {
      auto changed = fields;
      if (ignored)
        changed[0] = "learned_w1_refit";
      changed[index] = bad;
      std::ostringstream row;
      for (size_t i = 0; i < changed.size(); ++i)
        row << (i == 0 ? "" : "\t") << changed[i];
      EXPECT_FALSE(FailuresOk(valid + row.str() + "\n"));
    }
  }
  EXPECT_FALSE(FailuresOk(valid + "\n"));
  EXPECT_FALSE(FailuresOk(
      valid + "quadratic_refit\tall\t2\tfit\t5\t4\t1\t2\t0,1,2,3,4\textra\n"));
}

TEST(QuadraticArtifactsTest, RejectsFailedStreams) {
  std::istringstream coefficients(Coefficients());
  coefficients.setstate(std::ios::badbit);
  EXPECT_FALSE(ReadQuadraticCoefficients(coefficients).ok());
  std::istringstream failures(kFailureHeader);
  failures.setstate(std::ios::badbit);
  EXPECT_FALSE(ReadQuadraticFailures(failures).ok());
}

}  // namespace
}  // namespace pluto::llm::one_shot_memorizer
