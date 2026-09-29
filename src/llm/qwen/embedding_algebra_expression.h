#pragma once

#include <functional>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

namespace pluto::llm::qwen {

// One literal tokenizer input and its signed contribution to the result.
struct AlgebraSymbol {
  std::string text;  // Exact symbol bytes; no leading space is inserted.
  int coefficient;   // Either +1 or -1, including any unary sign.
};

// A vector sum, optionally requesting coordinates instead of nearest tokens.
struct AlgebraExpression {
  std::vector<AlgebraSymbol> terms;  // Contributions in expression order.
  bool raw = false;                  // Print the complete result vector.
};

// Parses [raw] [sign] symbol {sign [sign] symbol} [raw], where signs are +/−
// (ASCII '-' for minus). Whitespace outside symbols has no vector meaning.
// Bare symbols end at whitespace or operators; single/double quotes preserve
// spaces and punctuation. Quoted escapes are \\, \/, \', \", \n, \r, and \t.
// An unquoted first/last `raw` is a display modifier, not a vocabulary symbol;
// quote "raw" to make it a literal. Empty symbols, missing operators, repeated
// unary signs, parentheses, and multiplication/division are rejected. Whether
// each symbol actually encodes as one token is checked by the caller.
absl::StatusOr<AlgebraExpression> ParseAlgebraExpression(
    absl::string_view text);

// Encodes each literal independently, preserving its exact bytes and requiring
// exactly one nonnegative token ID. Returns IDs in term order, or a diagnostic
// that quotes the offending symbol; tokenizer failures propagate unchanged.
absl::StatusOr<std::vector<int>> EncodeAlgebraSymbols(
    const AlgebraExpression& expression,
    const std::function<absl::StatusOr<std::vector<int>>(absl::string_view)>&
        encode);

}  // namespace pluto::llm::qwen
