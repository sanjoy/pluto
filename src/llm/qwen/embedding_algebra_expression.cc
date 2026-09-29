#include "src/llm/qwen/embedding_algebra_expression.h"

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/ascii.h"
#include "absl/strings/escaping.h"
#include "absl/strings/str_cat.h"
#include "src/util/status_macros.h"

namespace pluto::llm::qwen {
namespace {

enum class Kind { kSymbol, kPlus, kMinus };

// Retains quoting and source position for modifier recognition and diagnostics.
struct Lexeme {
  Kind kind;
  std::string text;
  size_t offset;
  bool quoted = false;
};

absl::Status Invalid(size_t offset, absl::string_view message) {
  return absl::InvalidArgumentError(
      absl::StrCat("embedding expression at byte ", offset, ": ", message));
}

bool Unsupported(char character) {
  return character == '(' || character == ')' || character == '*' ||
         character == '/';
}

// Tokenize before interpreting signs, so quoted operators and the quoted word
// "raw" stay ordinary symbol bytes rather than becoming grammar elements.
absl::StatusOr<std::vector<Lexeme>> Tokenize(absl::string_view input) {
  std::vector<Lexeme> lexemes;
  size_t position = 0;
  while (position < input.size()) {
    const char character = input[position];
    if (absl::ascii_isspace(character)) {
      ++position;
      continue;
    }
    const size_t start = position++;
    if (character == '+' || character == '-') {
      lexemes.push_back({character == '+' ? Kind::kPlus : Kind::kMinus,
                         std::string(1, character), start});
      continue;
    }
    if (Unsupported(character))
      return Invalid(start,
                     "only addition and subtraction are supported; "
                     "quote punctuation to use it as a symbol");
    if (character == '\'' || character == '"') {
      std::string symbol;
      bool closed = false;
      while (position < input.size()) {
        char next = input[position++];
        if (next == character) {
          closed = true;
          break;
        }
        if (next == '\\') {
          if (position == input.size())
            return Invalid(position - 1, "unfinished quoted escape");
          next = input[position++];
          switch (next) {
            case '\\':
            case '/':
            case '\'':
            case '"':
              break;
            case 'n':
              next = '\n';
              break;
            case 'r':
              next = '\r';
              break;
            case 't':
              next = '\t';
              break;
            default:
              return Invalid(position - 2, "unknown quoted escape");
          }
        }
        symbol.push_back(next);
      }
      if (!closed)
        return Invalid(start, "unterminated quoted symbol");
      if (symbol.empty())
        return Invalid(start, "empty symbols cannot name an embedding");
      lexemes.push_back({Kind::kSymbol, std::move(symbol), start, true});
      continue;
    }
    // Bare names are byte strings, not identifiers: UTF-8 and punctuation are
    // allowed, but expressions with literal operators must use quoted symbols.
    if (character == '\\')
      return Invalid(start, "escapes require a quoted symbol");
    while (position < input.size()) {
      const char next = input[position];
      if (absl::ascii_isspace(next) || next == '+' || next == '-' ||
          next == '\'' || next == '"' || Unsupported(next))
        break;
      if (next == '\\')
        return Invalid(position, "escapes require a quoted symbol");
      ++position;
    }
    lexemes.push_back({Kind::kSymbol,
                       std::string(input.substr(start, position - start)),
                       start});
  }
  return lexemes;
}

bool IsRaw(const Lexeme& lexeme) {
  return lexeme.kind == Kind::kSymbol && !lexeme.quoted && lexeme.text == "raw";
}

int Sign(const Lexeme& lexeme) { return lexeme.kind == Kind::kMinus ? -1 : 1; }

}  // namespace

absl::StatusOr<AlgebraExpression> ParseAlgebraExpression(
    absl::string_view text) {
  ASSIGN_OR_RETURN(auto lexemes, Tokenize(text));
  AlgebraExpression expression;
  size_t first = 0;
  size_t end = lexemes.size();
  if (first < end && IsRaw(lexemes[first])) {
    expression.raw = true;
    ++first;
  }
  if (first < end && IsRaw(lexemes[end - 1])) {
    expression.raw = true;
    --end;
  }
  if (first == end)
    return Invalid(text.size(), "expected a symbol");

  size_t position = first;
  int binary_sign = 1;
  while (position < end) {
    int coefficient = binary_sign;
    if (lexemes[position].kind != Kind::kSymbol)
      coefficient *= Sign(lexemes[position++]);
    if (position == end)
      return Invalid(text.size(), "expected a symbol after the sign");
    if (lexemes[position].kind != Kind::kSymbol)
      return Invalid(lexemes[position].offset,
                     "expected a symbol; at most one unary sign is allowed");
    expression.terms.push_back(
        {std::move(lexemes[position].text), coefficient});
    ++position;
    if (position == end)
      break;
    if (lexemes[position].kind == Kind::kSymbol)
      return Invalid(lexemes[position].offset,
                     "expected '+' or '-' between symbols");
    binary_sign = Sign(lexemes[position++]);
    if (position == end)
      return Invalid(text.size(), "expected a symbol after the sign");
  }
  return expression;
}

absl::StatusOr<std::vector<int>> EncodeAlgebraSymbols(
    const AlgebraExpression& expression,
    const std::function<absl::StatusOr<std::vector<int>>(absl::string_view)>&
        encode) {
  if (!encode)
    return absl::InvalidArgumentError("missing embedding symbol tokenizer");
  if (expression.terms.empty())
    return absl::InvalidArgumentError("embedding expression has no symbols");
  std::vector<int> result;
  result.reserve(expression.terms.size());
  for (const auto& term : expression.terms) {
    ASSIGN_OR_RETURN(auto tokens, encode(term.text));
    if (tokens.size() != 1)
      return absl::InvalidArgumentError(absl::StrCat(
          "symbol \"", absl::CEscape(term.text), "\" tokenizes to ",
          tokens.size(),
          " tokens; each symbol must tokenize to exactly one embedding"));
    if (tokens.front() < 0)
      return absl::InvalidArgumentError(
          absl::StrCat("symbol \"", absl::CEscape(term.text),
                       "\" encodes to a negative token ID: ", tokens.front()));
    result.push_back(tokens.front());
  }
  return result;
}

}  // namespace pluto::llm::qwen
