#pragma once

#include <string>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "src/cuda/executor.h"
#include "src/cuda/page_locked_host_array.h"
#include "src/dataset/detokenizer.h"
#include "src/dataset/tokenizer.h"

namespace pluto::tokenizer {

// A stateless byte tokenizer and detokenizer for plain-text corpora.
//
// No downloaded vocabulary is needed: each byte maps to the identically
// numbered token, including NUL and bytes that are not valid UTF-8. Signed char
// never changes a byte's ID. The 256-token vocabulary keeps small tests cheap;
// every possible input byte string round-trips without adding special tokens.
class PlainTextTokenizer final : public Tokenizer, public Detokenizer {
 public:
  static constexpr int kVocabSize = 256;

  absl::StatusOr<cuda::PageLockedHostArray<int>> Encode(
      cuda::Executor& executor, absl::string_view text) const override;
  absl::StatusOr<std::string> Decode(
      absl::Span<const int> tokens) const override;

  int vocab_size() const override { return kVocabSize; }
};

}  // namespace pluto::tokenizer
