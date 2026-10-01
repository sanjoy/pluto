#include "src/llm/experiments/finite_state_machine/tokenizer.h"

#include <cstddef>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/escaping.h"
#include "absl/strings/str_cat.h"

namespace pluto::llm::fsm {
namespace {

bool IsDigit(char value) { return value >= '0' && value <= '9'; }

absl::Status InvalidText(absl::string_view text, size_t compact_offset,
                         absl::string_view reason) {
  // Tokenization ignores spaces, but diagnostics still identify the original
  // input byte. The extra scan occurs only when the input is invalid.
  size_t offset = 0;
  size_t nonspace_count = 0;
  for (; offset < text.size(); ++offset) {
    if (text[offset] == ' ')
      continue;
    if (nonspace_count++ == compact_offset)
      break;
  }
  return absl::InvalidArgumentError(
      absl::StrCat(reason, " at byte ", offset, ": \"",
                   absl::CEscape(text.substr(offset, 12)), "\""));
}

}  // namespace

absl::StatusOr<cuda::PageLockedHostArray<int>> FsmTokenizer::Encode(
    cuda::Executor& executor, absl::string_view text) const {
  // Spaces only improve readability: removing them before lexing also handles
  // spellings such as "0 0 0" and "E R R" consistently.
  const absl::string_view original_text = text;
  std::string compact_text;
  compact_text.reserve(text.size());
  for (const char value : text)
    if (value != ' ')
      compact_text.push_back(value);
  text = compact_text;

  std::vector<int> tokens;
  tokens.reserve(text.size());
  size_t offset = 0;
  bool in_output = false;
  while (offset < text.size()) {
    const char value = text[offset];
    if (IsDigit(value)) {
      if (text.size() - offset < 3 || !IsDigit(text[offset + 1]) ||
          !IsDigit(text[offset + 2])) {
        return InvalidText(original_text, offset,
                           "state token requires three digits");
      }
      tokens.push_back((value - '0') * 100 + (text[offset + 1] - '0') * 10 +
                       text[offset + 2] - '0');
      offset += 3;
    } else if ((text == "ERR" || in_output) &&
               text.substr(offset, 3) == "ERR") {
      tokens.push_back(kErrorToken);
      offset += 3;
    } else if (value >= 'A' && value <= 'Z') {
      tokens.push_back(kLetterOffset + value - 'A');
      ++offset;
    } else if (value == ';') {
      tokens.push_back(kSemicolonToken);
      ++offset;
    } else if (value == '>') {
      tokens.push_back(kOutputSeparatorToken);
      in_output = true;
      ++offset;
    } else {
      return InvalidText(original_text, offset, "unsupported FSM character");
    }
  }
  return cuda::PageLockedHostArray<int>::CopyFrom(executor, tokens);
}

absl::StatusOr<std::string> FsmTokenizer::Decode(
    absl::Span<const int> token_ids) const {
  std::string text;
  for (size_t index = 0; index < token_ids.size(); ++index) {
    const int token = token_ids[index];
    if (token < 0 || token >= kVocabularySize) {
      return absl::InvalidArgumentError(
          absl::StrCat("FSM token ID ", token, " at index ", index,
                       " is outside [0, ", kVocabularySize, ")"));
    }
    if (token < kStateCount) {
      text.push_back('0' + token / 100);
      text.push_back('0' + (token / 10) % 10);
      text.push_back('0' + token % 10);
    } else if (token < kSemicolonToken) {
      text.push_back('A' + token - kLetterOffset);
    } else if (token == kSemicolonToken) {
      text.push_back(';');
    } else if (token == kOutputSeparatorToken) {
      text.push_back('>');
    } else {
      text += "ERR";
    }
  }
  return text;
}

}  // namespace pluto::llm::fsm
