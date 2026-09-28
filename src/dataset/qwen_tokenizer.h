#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "src/cuda/page_locked_host_array.h"
#include "src/dataset/detokenizer.h"
#include "src/dataset/tokenizer.h"

namespace pluto::tokenizer {

// Native Qwen3.8 byte-level BPE with NFC normalization and the checkpoint's
// added tokens. Load the official tokenizer.json beside the model weights.
// Instances support concurrent encoding and decoding. Unlike GPT-2, encoding
// normalizes canonically equivalent Unicode strings to the same tokens.
class QwenTokenizer final : public Tokenizer, public Detokenizer {
 public:
  static absl::StatusOr<std::unique_ptr<QwenTokenizer>> Load(
      const std::filesystem::path& directory);
  ~QwenTokenizer() override;

  // CPU-only overload is useful during prompt preparation and verification.
  absl::StatusOr<std::vector<int>> Encode(absl::string_view text) const;
  absl::StatusOr<cuda::PageLockedHostArray<int>> Encode(
      cuda::Executor& executor, absl::string_view text) const override;

  // Returns exact bytes, including added token spellings. Like Detokenizer,
  // individual tokens may decode to incomplete UTF-8; concatenate before
  // rendering. The vocabulary includes added tokens, but not model padding.
  absl::StatusOr<std::string> Decode(
      absl::Span<const int> token_ids) const override;
  int vocab_size() const override;
  int eos_token_id() const;

  // Official single-user, text-only chat template with add_generation_prompt.
  // Thinking mode uses the template's default xhigh reasoning instruction.
  static std::string ChatPrompt(absl::string_view text,
                                bool enable_thinking = false);

 private:
  struct Impl;
  explicit QwenTokenizer(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace pluto::tokenizer
