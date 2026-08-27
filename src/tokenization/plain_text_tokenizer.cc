#include "src/tokenization/plain_text_tokenizer.h"

#include <cstdint>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"

namespace pluto::tokenization {

std::vector<uint32_t> PlainTextTokenizer::Encode(
    absl::string_view text) const {
  std::vector<uint32_t> tokens;
  tokens.reserve(text.size());
  for (unsigned char byte : text) tokens.push_back(byte);
  return tokens;
}

absl::StatusOr<std::string> PlainTextTokenizer::Decode(
    absl::Span<const uint32_t> tokens) const {
  std::string text;
  text.reserve(tokens.size());
  for (uint32_t token : tokens) {
    if (token >= kVocabSize) {
      return absl::InvalidArgumentError(
          absl::StrCat("plain-text token is outside [0, 255]: ", token));
    }
    text.push_back(static_cast<char>(token));
  }
  return text;
}

}  // namespace pluto::tokenization
