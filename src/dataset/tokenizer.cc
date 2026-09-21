#include "src/dataset/tokenizer.h"

#include "absl/status/status.h"
#include "absl/strings/escaping.h"
#include "absl/strings/str_cat.h"
#include "src/util/status_macros.h"

namespace pluto::tokenizer {

Tokenizer::~Tokenizer() = default;

absl::StatusOr<cuda::PageLockedHostArray<int>> Tokenizer::EncodeWithVocabulary(
    cuda::Executor& executor, absl::string_view text,
    absl::Span<const uint8_t> token_is_allowed) const {
  if (vocab_size() <= 0 ||
      token_is_allowed.size() != static_cast<size_t>(vocab_size()))
    return absl::InvalidArgumentError("vocabulary mask has an invalid size");
  ASSIGN_OR_RETURN(auto tokens, Encode(executor, text));
  for (int token : tokens) {
    if (token < 0 || token >= vocab_size())
      return absl::InvalidArgumentError(
          "tokenizer returned an invalid token ID");
    if (!token_is_allowed[token])
      return absl::InvalidArgumentError(
          absl::StrCat("substring \"", absl::Utf8SafeCEscape(text),
                       "\" could not be encoded using the compact vocabulary"));
  }
  return tokens;
}

}  // namespace pluto::tokenizer
