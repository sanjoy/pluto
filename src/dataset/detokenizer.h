#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include <utility>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "src/dataset/gpt2_tokenizer_vocabulary.h"

namespace pluto::tokenizer {

// Inverse of Gpt2Tokenizer for token-id sequences from the same model.
class Gpt2Detokenizer final {
 public:
  static absl::StatusOr<std::unique_ptr<Gpt2Detokenizer>> Load(
      const std::filesystem::path& directory);

  // Returns InvalidArgument when any id is outside the loaded vocabulary.
  absl::StatusOr<std::string> Decode(absl::Span<const int> token_ids) const;

  int vocab_size() const { return model_->vocab_size(); }
  int eos_token_id() const { return model_->eos_token_id(); }

 private:
  explicit Gpt2Detokenizer(
      std::shared_ptr<const internal::Gpt2TokenizerVocabulary> model)
      : model_(std::move(model)) {}

  std::shared_ptr<const internal::Gpt2TokenizerVocabulary> model_;
};

}  // namespace pluto::tokenizer
