#include "src/llm/experiments/one_shot_memorizer/isolated_fact_selection.h"

#include <algorithm>
#include <cstddef>
#include <string>

#include "absl/status/status.h"
#include "absl/strings/ascii.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"

namespace pluto::llm::one_shot_memorizer {

absl::StatusOr<size_t> IsolatedFactSelection::CorpusIndex(
    size_t local_sample_index) const {
  if (local_sample_index >= corpus_indices.size())
    return absl::OutOfRangeError(
        absl::StrCat("isolated sample index ", local_sample_index,
                     " is outside ", corpus_indices.size(), " selected facts"));
  return corpus_indices[local_sample_index];
}

absl::StatusOr<IsolatedFactSelection> SelectIsolatedFacts(
    absl::Span<const std::string> corpus_lines,
    absl::Span<const std::string> line_numbers_1based) {
  if (line_numbers_1based.empty())
    return absl::InvalidArgumentError(
        "isolated fact selection must not be empty");
  IsolatedFactSelection result;
  result.corpus_indices.reserve(line_numbers_1based.size());
  for (const std::string& number : line_numbers_1based) {
    size_t line = 0;
    if (number.empty() ||
        !std::all_of(number.begin(), number.end(),
                     [](char ch) { return ch >= '0' && ch <= '9'; }) ||
        !absl::SimpleAtoi(number, &line) || line == 0 ||
        line > corpus_lines.size())
      return absl::InvalidArgumentError(
          absl::StrCat("invalid isolated corpus line number: ", number));
    const size_t index = line - 1;
    if (corpus_lines[index].find('\n') != std::string::npos)
      return absl::InvalidArgumentError(absl::StrCat(
          "selected corpus line ", line, " contains an embedded newline"));
    if (absl::StripAsciiWhitespace(corpus_lines[index]).empty())
      return absl::InvalidArgumentError(
          absl::StrCat("selected corpus line ", line, " is blank"));
    result.corpus_indices.push_back(index);
  }
  std::sort(result.corpus_indices.begin(), result.corpus_indices.end());
  if (std::adjacent_find(result.corpus_indices.begin(),
                         result.corpus_indices.end()) !=
      result.corpus_indices.end())
    return absl::InvalidArgumentError(
        "isolated corpus line numbers must be distinct");
  for (size_t index : result.corpus_indices) {
    if (!result.text.empty())
      result.text += '\n';
    result.text += corpus_lines[index];
  }
  return result;
}

}  // namespace pluto::llm::one_shot_memorizer
