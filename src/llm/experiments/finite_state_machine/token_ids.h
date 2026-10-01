#pragma once

namespace pluto::llm::fsm {

// Shared vocabulary numbering for CPU-only constructions and GPU training.
// No BOS, EOS, padding, or other hidden vocabulary entries are included.
inline constexpr int kStateCount = 1000;
inline constexpr int kLetterOffset = kStateCount;
inline constexpr int kSemicolonToken = kLetterOffset + 26;
inline constexpr int kOutputSeparatorToken = kSemicolonToken + 1;
inline constexpr int kErrorToken = kOutputSeparatorToken + 1;
inline constexpr int kVocabularySize = kErrorToken + 1;

}  // namespace pluto::llm::fsm
