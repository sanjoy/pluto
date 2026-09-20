#include "src/dataset/plain_text_tokenizer.h"

#include <cstddef>
#include <string>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "src/util/status_macros.h"

namespace pluto::tokenizer {

absl::StatusOr<cuda::PageLockedHostArray<int>> PlainTextTokenizer::Encode(
    cuda::Executor& executor, absl::string_view text) const {
  // The output length is known exactly, so fill pinned storage directly rather
  // than building a pageable temporary vector and copying it afterward.
  ASSIGN_OR_RETURN(auto tokens, cuda::PageLockedHostArray<int>::Allocate(
                                    executor, text.size()));
  for (size_t index = 0; index < text.size(); ++index)
    tokens[index] = static_cast<unsigned char>(text[index]);
  return tokens;
}

absl::StatusOr<std::string> PlainTextTokenizer::Decode(
    absl::Span<const int> tokens) const {
  std::string text;
  text.reserve(tokens.size());
  for (int token : tokens) {
    // Signed IDs are shared with GPT-2 and the dataset schema. Reject negative
    // values explicitly; converting them to char would silently corrupt input.
    if (token < 0 || token >= kVocabSize)
      return absl::InvalidArgumentError(
          absl::StrCat("plain-text token is outside [0, 255]: ", token));
    text.push_back(static_cast<char>(token));
  }
  return text;
}

}  // namespace pluto::tokenizer
