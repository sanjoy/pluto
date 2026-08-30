#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"

namespace pluto::tokenization {

// A stateless byte tokenizer for training directly from plain-text corpora.
//
// Unlike the GPT-2 tokenizer, this class needs no downloaded vocabulary. Each
// byte maps to the identically numbered token, so it is deterministic, handles
// arbitrary UTF-8 byte sequences, and always round-trips. Its deliberately
// small 256-token vocabulary also keeps tiny language-model tests inexpensive.
class PlainTextTokenizer final {
 public:
  static constexpr uint32_t kVocabSize = 256;

  std::vector<uint32_t> Encode(absl::string_view text) const;
  absl::StatusOr<std::string> Decode(
      absl::Span<const uint32_t> tokens) const;

  constexpr uint32_t vocab_size() const { return kVocabSize; }
};

}  // namespace pluto::tokenization
