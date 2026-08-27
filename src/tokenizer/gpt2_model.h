#ifndef PLUTO_SRC_TOKENIZER_GPT2_MODEL_H_
#define PLUTO_SRC_TOKENIZER_GPT2_MODEL_H_

#include <array>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

namespace pluto::tokenizer::internal {

// Immutable vocabulary shared by the encoder and decoder implementations.
//
// The loader intentionally understands only the Hugging Face tokenizer.json
// representation used by openai-community/gpt2: a byte-level BPE model whose
// `vocab` is a string-to-id object and whose `merges` are ordered string pairs.
class Gpt2Model final {
 public:
  static absl::StatusOr<std::shared_ptr<const Gpt2Model>> Load(
      const std::filesystem::path& directory);

  const absl::flat_hash_map<std::string, int>& encoder() const {
    return encoder_;
  }
  const std::vector<std::string>& decoder() const { return decoder_; }
  const absl::flat_hash_map<std::string, int>& merge_ranks() const {
    return merge_ranks_;
  }
  const std::array<std::string, 256>& byte_encoder() const {
    return byte_encoder_;
  }
  const std::array<int, 512>& byte_decoder() const { return byte_decoder_; }

  int vocab_size() const { return static_cast<int>(decoder_.size()); }
  int eos_token_id() const { return eos_token_id_; }
  absl::string_view eos_token() const { return eos_token_; }

 private:
  Gpt2Model() = default;

  absl::flat_hash_map<std::string, int> encoder_;
  std::vector<std::string> decoder_;
  absl::flat_hash_map<std::string, int> merge_ranks_;
  std::array<std::string, 256> byte_encoder_;
  std::array<int, 512> byte_decoder_;
  std::string eos_token_;
  int eos_token_id_ = -1;
};

// BPE pairs are stored as `left + NUL + right`. GPT-2's byte-to-Unicode
// alphabet never contains NUL, so the representation is unambiguous and avoids
// allocating a pair object in the hottest lookup path.
std::string MergeKey(absl::string_view left, absl::string_view right);

// Decodes one UTF-8 scalar and removes it from `input`.
absl::StatusOr<uint32_t> ConsumeUtf8(absl::string_view* input);

}  // namespace pluto::tokenizer::internal

#endif  // PLUTO_SRC_TOKENIZER_GPT2_MODEL_H_
