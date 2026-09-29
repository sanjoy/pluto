#include "src/llm/qwen/embedding_algebra_expression.h"

#include <initializer_list>
#include <string>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "gtest/gtest.h"

namespace pluto::llm::qwen {
namespace {

void ExpectExpression(absl::string_view text,
                      std::initializer_list<AlgebraSymbol> terms,
                      bool raw = false) {
  SCOPED_TRACE(text);
  const auto result = ParseAlgebraExpression(text);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result->raw, raw);
  ASSERT_EQ(result->terms.size(), terms.size());
  size_t index = 0;
  for (const auto& term : terms) {
    EXPECT_EQ(result->terms[index].text, term.text) << index;
    EXPECT_EQ(result->terms[index].coefficient, term.coefficient) << index;
    ++index;
  }
}

TEST(EmbeddingAlgebraExpressionTest, ParsesAdditionAndSubtraction) {
  for (absl::string_view text :
       {"king - queen + boy", "king-queen+boy", "  king\t-\nqueen + boy\r\n"})
    ExpectExpression(text, {{"king", 1}, {"queen", -1}, {"boy", 1}});
  ExpectExpression("king", {{"king", 1}});
  ExpectExpression("king + king - king",
                   {{"king", 1}, {"king", 1}, {"king", -1}});
}

TEST(EmbeddingAlgebraExpressionTest, CombinesBinaryAndUnarySigns) {
  ExpectExpression("-king + +queen - -boy",
                   {{"king", -1}, {"queen", 1}, {"boy", 1}});
  ExpectExpression("+king+-queen--boy",
                   {{"king", 1}, {"queen", -1}, {"boy", 1}});
  ExpectExpression("king++queen-+boy",
                   {{"king", 1}, {"queen", 1}, {"boy", -1}});
}

TEST(EmbeddingAlgebraExpressionTest, SupportsRawPrefixAndSuffix) {
  for (absl::string_view text :
       {"raw king - queen + boy", "king - queen + boy raw",
        "raw king - queen + boy raw", " \traw\nking-queen+boy\nraw\r "})
    ExpectExpression(text, {{"king", 1}, {"queen", -1}, {"boy", 1}}, true);
  ExpectExpression("raw -king", {{"king", -1}}, true);
  ExpectExpression("-king raw", {{"king", -1}}, true);
  ExpectExpression("raw 'raw'", {{"raw", 1}}, true);
  ExpectExpression("\"raw\" raw", {{"raw", 1}}, true);
}

TEST(EmbeddingAlgebraExpressionTest, QuotingPreservesExactBytes) {
  ExpectExpression("\" king\" - ' queen ' + 'boy'",
                   {{" king", 1}, {" queen ", -1}, {"boy", 1}});
  ExpectExpression("'+' + \"-\" - '()*/'", {{"+", 1}, {"-", 1}, {"()*/", -1}});
  ExpectExpression("'raw' + \"raw\"", {{"raw", 1}, {"raw", 1}});
  ExpectExpression("'two words'", {{"two words", 1}});
  ExpectExpression("' ' + '\n'", {{" ", 1}, {"\n", 1}});
}

TEST(EmbeddingAlgebraExpressionTest, DecodesOnlySupportedQuotedEscapes) {
  ExpectExpression(R"("a\\b\/c\'d\"e\nf\rg\th")",
                   {{"a\\b/c'd\"e\nf\rg\th", 1}});
  ExpectExpression(R"('a\'b\"c')", {{"a'b\"c", 1}});
}

TEST(EmbeddingAlgebraExpressionTest, PreservesUtf8AndLiteralPunctuation) {
  ExpectExpression("王 - reine + garçon",
                   {{"王", 1}, {"reine", -1}, {"garçon", 1}});
  ExpectExpression(". + , - <|endoftext|>",
                   {{".", 1}, {",", 1}, {"<|endoftext|>", -1}});
  ExpectExpression("rawhide + draw", {{"rawhide", 1}, {"draw", 1}});
  ExpectExpression("RAW", {{"RAW", 1}});
}

TEST(EmbeddingAlgebraExpressionTest, RejectsMalformedGrammar) {
  for (absl::string_view text : {"",
                                 " \t\r\n",
                                 "raw",
                                 "raw raw",
                                 "+",
                                 "-",
                                 "raw +",
                                 "king +",
                                 "king -",
                                 "king - -",
                                 "--king",
                                 "++king",
                                 "king +++ queen",
                                 "king queen",
                                 "king'queen'",
                                 "'king'queen",
                                 "'king'\"queen\"",
                                 "raw king queen",
                                 "king raw queen",
                                 "king + raw",
                                 "king + + raw",
                                 "king * queen",
                                 "king/queen",
                                 "(king)",
                                 "king + (queen)",
                                 "\"\"",
                                 "''",
                                 "'king",
                                 "\"king",
                                 "'king\\",
                                 "'king\\q'",
                                 "'king\\u1234'",
                                 "king\\queen",
                                 "\\n",
                                 "king + ''"}) {
    SCOPED_TRACE(text);
    const auto result = ParseAlgebraExpression(text);
    ASSERT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), absl::StatusCode::kInvalidArgument);
    EXPECT_NE(result.status().message().find("embedding expression at byte"),
              absl::string_view::npos)
        << result.status();
  }
}

TEST(EmbeddingAlgebraExpressionTest, ReportsBytePositionForInvalidSyntax) {
  const auto result = ParseAlgebraExpression("king queen");
  ASSERT_FALSE(result.ok());
  EXPECT_NE(result.status().message().find("byte 5"), absl::string_view::npos);
  const auto escape = ParseAlgebraExpression("'king\\q'");
  ASSERT_FALSE(escape.ok());
  EXPECT_NE(escape.status().message().find("byte 5"), absl::string_view::npos);
}

TEST(EmbeddingAlgebraExpressionTest, EncodesLiteralBytesInTermOrder) {
  const auto expression =
      ParseAlgebraExpression("raw ' king' - queen + ' king'");
  ASSERT_TRUE(expression.ok()) << expression.status();
  std::vector<std::string> seen;
  const auto result = EncodeAlgebraSymbols(
      *expression,
      [&](absl::string_view text) -> absl::StatusOr<std::vector<int>> {
        seen.emplace_back(text);
        return std::vector<int>{text == " king" ? 7 : 0};
      });
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(*result, (std::vector<int>{7, 0, 7}));
  EXPECT_EQ(seen, (std::vector<std::string>{" king", "queen", " king"}));
}

TEST(EmbeddingAlgebraExpressionTest, RejectsZeroAndMultipleTokenSymbols) {
  for (size_t count : {0, 2, 5}) {
    SCOPED_TRACE(count);
    const AlgebraExpression expression{{{" a\"b\nc", 1}}, false};
    const auto result = EncodeAlgebraSymbols(
        expression,
        [count](absl::string_view) -> absl::StatusOr<std::vector<int>> {
          return std::vector<int>(count, 7);
        });
    ASSERT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), absl::StatusCode::kInvalidArgument);
    EXPECT_NE(result.status().message().find("\" a\\\"b\\nc\""),
              absl::string_view::npos)
        << result.status();
    EXPECT_NE(result.status().message().find(std::to_string(count) + " tokens"),
              absl::string_view::npos)
        << result.status();
  }
}

TEST(EmbeddingAlgebraExpressionTest, PropagatesTokenizerErrorsUnchanged) {
  const AlgebraExpression expression{{{"king", 1}, {"queen", -1}}, false};
  int calls = 0;
  const auto result = EncodeAlgebraSymbols(
      expression, [&](absl::string_view) -> absl::StatusOr<std::vector<int>> {
        ++calls;
        return absl::UnavailableError("test tokenizer failed");
      });
  ASSERT_FALSE(result.ok());
  EXPECT_EQ(result.status().code(), absl::StatusCode::kUnavailable);
  EXPECT_EQ(result.status().message(), "test tokenizer failed");
  EXPECT_EQ(calls, 1);
}

TEST(EmbeddingAlgebraExpressionTest, RejectsNegativeTokenIds) {
  const AlgebraExpression expression{{{"king", 1}}, false};
  const auto result = EncodeAlgebraSymbols(
      expression, [](absl::string_view) -> absl::StatusOr<std::vector<int>> {
        return std::vector<int>{-1};
      });
  ASSERT_FALSE(result.ok());
  EXPECT_EQ(result.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_NE(result.status().message().find("symbol \"king\""),
            absl::string_view::npos);
  EXPECT_NE(result.status().message().find("negative token ID"),
            absl::string_view::npos);
}

TEST(EmbeddingAlgebraExpressionTest, RejectsMissingTokenizerOrExpression) {
  EXPECT_EQ(EncodeAlgebraSymbols({{{"king", 1}}, false}, {}).status().code(),
            absl::StatusCode::kInvalidArgument);
  int calls = 0;
  const auto result = EncodeAlgebraSymbols(
      {}, [&](absl::string_view) -> absl::StatusOr<std::vector<int>> {
        ++calls;
        return std::vector<int>{7};
      });
  EXPECT_EQ(result.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(calls, 0);
}

}  // namespace
}  // namespace pluto::llm::qwen
