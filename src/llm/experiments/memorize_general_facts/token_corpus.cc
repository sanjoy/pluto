#include "src/llm/experiments/memorize_general_facts/token_corpus.h"

#include <utility>

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/strings/ascii.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "src/llm/token_order.h"
#include "src/util/status_macros.h"

namespace pluto::llm::memorize_general_facts {
namespace {

// Match PaddedLineDataSetIterator exactly: strip one final LF and one CR per
// line, preserving every other byte (including leading spaces in GPT-2 text).
std::vector<absl::string_view> Lines(absl::string_view text) {
  if (text.empty())
    return {};
  if (text.back() == '\n')
    text.remove_suffix(1);
  std::vector<absl::string_view> lines = absl::StrSplit(text, '\n');
  for (auto& line : lines)
    if (!line.empty() && line.back() == '\r')
      line.remove_suffix(1);
  return lines;
}

}  // namespace

absl::StatusOr<std::vector<int32_t>> ParseTokenOrder(absl::string_view text,
                                                     int vocabulary_size) {
  std::vector<int32_t> order;
  for (absl::string_view word : absl::StrSplit(
           text, absl::ByAnyChar(" \t\r\n\v\f"), absl::SkipEmpty())) {
    int32_t id;
    if (!absl::SimpleAtoi(word, &id))
      return absl::InvalidArgumentError(
          absl::StrCat("token order contains an invalid int32 ID: ", word));
    order.push_back(id);
  }
  if (order.empty())
    return absl::InvalidArgumentError("token order file must not be empty");
  RETURN_IF_ERROR(ValidateTokenOrder(vocabulary_size, order));
  return order;
}

absl::StatusOr<std::vector<std::vector<int>>> ParseTokenCorpus(
    absl::string_view token_text, absl::Span<const size_t> expected_lengths,
    int vocabulary_size, int eos_token) {
  if (vocabulary_size <= 0 || eos_token < 0 || eos_token >= vocabulary_size)
    return absl::InvalidArgumentError("invalid token corpus vocabulary or EOS");
  const auto lines = Lines(token_text);
  if (lines.empty() || lines.size() != expected_lengths.size())
    return absl::InvalidArgumentError(
        "token corpus must have exactly one row per original sentence");
  std::vector<std::vector<int>> rows;
  rows.reserve(lines.size());
  for (size_t row = 0; row < lines.size(); ++row) {
    std::vector<int> ids;
    for (absl::string_view word : absl::StrSplit(
             lines[row], absl::ByAnyChar(" \t\r\v\f"), absl::SkipEmpty())) {
      int id;
      if (!absl::SimpleAtoi(word, &id) || id < 0 || id >= vocabulary_size ||
          id == eos_token)
        return absl::InvalidArgumentError(
            absl::StrCat("token corpus row ", row + 1,
                         " contains invalid or EOS token ID: ", word));
      ids.push_back(id);
    }
    if (ids.empty() || ids.size() != expected_lengths[row])
      return absl::InvalidArgumentError(
          absl::StrCat("token corpus row ", row + 1, " has ", ids.size(),
                       " tokens; expected ", expected_lengths[row]));
    rows.push_back(std::move(ids));
  }
  return rows;
}

absl::StatusOr<std::unique_ptr<TokenCorpusTokenizer>>
TokenCorpusTokenizer::Create(cuda::Executor& executor,
                             const tokenizer::Tokenizer& base,
                             absl::string_view original_corpus,
                             absl::string_view token_text, int eos_token) {
  const auto lines = Lines(original_corpus);
  std::vector<size_t> lengths;
  lengths.reserve(lines.size());
  for (absl::string_view line : lines) {
    if (absl::StripAsciiWhitespace(line).empty())
      return absl::InvalidArgumentError("original corpus has an empty line");
    ASSIGN_OR_RETURN(auto tokens, base.Encode(executor, line));
    lengths.push_back(tokens.size());
  }
  ASSIGN_OR_RETURN(auto rows, ParseTokenCorpus(token_text, lengths,
                                               base.vocab_size(), eos_token));
  absl::flat_hash_map<std::string, std::vector<int>> by_sentence;
  for (size_t i = 0; i < lines.size(); ++i) {
    auto [it, inserted] =
        by_sentence.try_emplace(std::string(lines[i]), rows[i]);
    if (!inserted && it->second != rows[i])
      return absl::InvalidArgumentError(
          "duplicate original sentence has conflicting token corpus rows");
  }
  return absl::WrapUnique(
      new TokenCorpusTokenizer(base.vocab_size(), std::move(by_sentence)));
}

TokenCorpusTokenizer::TokenCorpusTokenizer(
    int vocabulary_size,
    absl::flat_hash_map<std::string, std::vector<int>> rows)
    : vocabulary_size_(vocabulary_size), rows_(std::move(rows)) {}

absl::StatusOr<cuda::PageLockedHostArray<int>> TokenCorpusTokenizer::Encode(
    cuda::Executor& executor, absl::string_view text) const {
  const auto found = rows_.find(text);
  if (found == rows_.end())
    return absl::InvalidArgumentError(
        "token corpus adapter only accepts exact original corpus sentences");
  return cuda::PageLockedHostArray<int>::CopyFrom(executor, found->second);
}

}  // namespace pluto::llm::memorize_general_facts
