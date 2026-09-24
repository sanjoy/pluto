#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "src/dataset/tokenizer.h"

namespace pluto::llm::memorize_general_facts {

// Parses one whitespace-separated compact-ID row per original corpus sentence.
// Row lengths must be unchanged; EOS is supplied by the dataset, never by a
// row. This validator performs no CUDA work and does not reinterpret IDs as
// text.
absl::StatusOr<std::vector<std::vector<int>>> ParseTokenCorpus(
    absl::string_view token_text, absl::Span<const size_t> expected_lengths,
    int vocabulary_size, int eos_token);

// A research-only tokenizer adapter: exact original sentence bytes select an
// already-tokenized experimental row. This deliberately does not support
// arbitrary prompt text or claim that decoding preserves the original bytes.
// The fixed base vocabulary is built from the ORIGINAL corpus, so every trial
// has identical embedding dimensions, ID meanings, and initialization.
class TokenCorpusTokenizer final : public tokenizer::Tokenizer {
 public:
  static absl::StatusOr<std::unique_ptr<TokenCorpusTokenizer>> Create(
      cuda::Executor& executor, const tokenizer::Tokenizer& base,
      absl::string_view original_corpus, absl::string_view token_text,
      int eos_token);

  absl::StatusOr<cuda::PageLockedHostArray<int>> Encode(
      cuda::Executor& executor, absl::string_view text) const override;
  int vocab_size() const override { return vocabulary_size_; }

 private:
  TokenCorpusTokenizer(int vocabulary_size,
                       absl::flat_hash_map<std::string, std::vector<int>> rows);

  int vocabulary_size_;
  absl::flat_hash_map<std::string, std::vector<int>> rows_;
};

}  // namespace pluto::llm::memorize_general_facts
