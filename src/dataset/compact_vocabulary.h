#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "src/cuda/executor.h"
#include "src/cuda/page_locked_host_array.h"
#include "src/dataset/tokenizer.h"

namespace pluto::tokenizer {

// Bidirectional, CPU-only lookup tables. Compact IDs are contiguous and follow
// sorted original IDs. Original IDs not in the compact vocabulary map to -1.
// EOS is always included, even when it does not occur in the corpus.
struct CompactVocabularyMapping {
  std::vector<int> compact_to_original;
  std::vector<int> original_to_compact;
  int original_eos_id = -1;
};

// Tokenizes each corpus line independently. Accepts LF/CRLF and an optional
// final newline; rejects empty/whitespace-only lines and empty token sequences.
// Includes all input tokens (including prompt-only tokens) and EOS. Neither
// the corpus nor the tokenizer is retained. The executor is needed only for
// the base tokenizer's temporary encoded arrays. EOS is explicit because the
// abstract Tokenizer does not define a special ID.
absl::StatusOr<CompactVocabularyMapping> BuildCompactVocabularyMapping(
    cuda::Executor& executor, const Tokenizer& original,
    absl::string_view corpus_text, int original_eos_id);

// Encodes with the retained vocabulary, then remaps to contiguous compact IDs.
// GPT-2 preserves the original tokenization when all its tokens are retained;
// otherwise it finds an exact alternative using retained smaller token pieces.
// The referenced original tokenizer must outlive this wrapper. Encoded arrays
// retain the usual executor lifetime requirements of Tokenizer.
class CompactVocabularyTokenizer final : public Tokenizer {
 public:
  // Owns the mapping, which may come from the helper above or another source.
  // Validates both directions, sorted original IDs, and EOS membership. Does
  // not encode any text or allocate CUDA memory; no corpus or executor is
  // needed.
  static absl::StatusOr<std::unique_ptr<CompactVocabularyTokenizer>> Create(
      const Tokenizer& original, CompactVocabularyMapping mapping);

  // Restores a canonical saved mapping without encoding text or allocating
  // CUDA memory. Rejects malformed files and a different original vocabulary
  // size. The caller must still verify the original tokenizer's identity.
  static absl::StatusOr<std::unique_ptr<CompactVocabularyTokenizer>>
  LoadFromFile(const Tokenizer& original, const std::filesystem::path& path);

  // Delegates vocabulary-aware encoding to the original tokenizer. Substrings
  // without a permitted encoding are reported as text, not just numeric IDs.
  // Returns fresh storage, leaving shared/cached source arrays untouched.
  absl::StatusOr<cuda::PageLockedHostArray<int>> Encode(
      cuda::Executor& executor, absl::string_view text) const override;

  int vocab_size() const override {
    return static_cast<int>(mapping_.compact_to_original.size());
  }
  int eos_token_id() const {
    return mapping_.original_to_compact[mapping_.original_eos_id];
  }
  int original_vocab_size() const {
    return static_cast<int>(mapping_.original_to_compact.size());
  }
  int original_eos_token_id() const { return mapping_.original_eos_id; }
  absl::Span<const int> original_token_ids() const {
    return absl::MakeConstSpan(mapping_.compact_to_original);
  }
  absl::StatusOr<int> OriginalId(int compact_id) const;
  absl::StatusOr<int> CompactId(int original_id) const;

  // Canonical ASCII TSV, LF endings, including the final LF:
  // compact_vocabulary_v1
  // original_vocab_size\t<N>
  // original_eos_token\t<E>
  // compact_vocab_size\t<V>
  // compact_id\toriginal_id
  // 0\t<first sorted original ID>
  // ...
  // Existing files must match exactly and are never silently replaced.
  // Callers still need their separate corpus/tokenizer content-hash checks;
  // different corpora can legitimately induce the same ID mapping.
  absl::Status SaveToFile(const std::filesystem::path& path) const;
  absl::Status ValidateFile(const std::filesystem::path& path) const;

 private:
  CompactVocabularyTokenizer(const Tokenizer& original,
                             CompactVocabularyMapping mapping);
  std::string CanonicalText() const;

  const Tokenizer& original_;
  CompactVocabularyMapping mapping_;
  // Immutable membership mask in the original ID space, reused on every call.
  std::vector<uint8_t> token_is_allowed_;
};

}  // namespace pluto::tokenizer
