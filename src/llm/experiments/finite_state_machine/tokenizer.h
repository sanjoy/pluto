#pragma once

#include <string>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "src/cuda/executor.h"
#include "src/cuda/page_locked_host_array.h"
#include "src/dataset/detokenizer.h"
#include "src/dataset/tokenizer.h"

namespace pluto::llm::fsm {

inline constexpr int kStateCount = 1000;
inline constexpr int kLetterOffset = kStateCount;
inline constexpr int kSemicolonToken = kLetterOffset + 26;
inline constexpr int kOutputSeparatorToken = kSemicolonToken + 1;
inline constexpr int kErrorToken = kOutputSeparatorToken + 1;
inline constexpr int kVocabularySize = kErrorToken + 1;

// The complete FSM vocabulary is 000..999, A..Z, ';', '>', and ERR. State
// tokens preserve all three digits, including leading zeroes. Numeric runs
// are grouped into triples, so adjacent state tokens also round-trip; leftover
// one- or two-digit groups are rejected. ASCII spaces are ignored everywhere,
// including within a state or ERR; other whitespace and unsupported bytes are
// invalid. Decoding returns the canonical spelling without spaces.
//
// ERR is one token anywhere after '>' or when encoding the standalone string
// "ERR". Before '>', its letters remain E, R, R. Thus the answer can be either
// one final state or a full execution trace such as ">000 093 ERR". Full
// sentences and prompts ending at any complete token are both supported.
//
// No BOS, EOS, or padding token is added. Encoding is lexical; the dataset
// validates the transition grammar and executes the FSM to check its answer.
class FsmTokenizer final : public tokenizer::Tokenizer,
                           public tokenizer::Detokenizer {
 public:
  absl::StatusOr<cuda::PageLockedHostArray<int>> Encode(
      cuda::Executor& executor, absl::string_view text) const override;

  absl::StatusOr<std::string> Decode(
      absl::Span<const int> token_ids) const override;

  int vocab_size() const override { return kVocabularySize; }
};

}  // namespace pluto::llm::fsm
