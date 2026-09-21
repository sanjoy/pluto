#pragma once

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

namespace pluto::llm::memorize_general_facts {

// Experiment-local ID remapping; the original tokenization is unchanged.
// The referenced original tokenizer must outlive this wrapper. Encoded arrays
// retain the usual executor lifetime requirements of tokenizer::Tokenizer.
class CompactVocabularyTokenizer final : public tokenizer::Tokenizer {
 public:
  // Tokenizes each corpus line independently, with the same LF/CRLF and blank
  // line rules as PaddedLineDataSetIterator. Includes every input token (also
  // prompt-only tokens) and EOS. Compact IDs follow sorted original IDs.
  static absl::StatusOr<std::unique_ptr<CompactVocabularyTokenizer>> Create(
      cuda::Executor& executor, const tokenizer::Tokenizer& original,
      absl::string_view corpus_text, int original_eos_id);

  // Rejects tokens absent from the discovered vocabulary. Returns fresh
  // storage, leaving any shared/cached original-tokenizer arrays untouched.
  absl::StatusOr<cuda::PageLockedHostArray<int>> Encode(
      cuda::Executor& executor, absl::string_view text) const override;

  int vocab_size() const override {
    return static_cast<int>(original_token_ids_.size());
  }
  int eos_token_id() const { return eos_token_id_; }
  int original_vocab_size() const {
    return static_cast<int>(original_to_compact_.size());
  }
  int original_eos_token_id() const { return original_eos_id_; }
  absl::Span<const int> original_token_ids() const {
    return absl::MakeConstSpan(original_token_ids_);
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
  CompactVocabularyTokenizer(const tokenizer::Tokenizer& original,
                             std::vector<int> original_token_ids,
                             std::vector<int> original_to_compact,
                             int original_eos_id, int eos_token_id);
  std::string CanonicalText() const;

  const tokenizer::Tokenizer& original_;
  std::vector<int> original_token_ids_;
  std::vector<int> original_to_compact_;
  int original_eos_id_;
  int eos_token_id_;
};

}  // namespace pluto::llm::memorize_general_facts
