#include "src/tokenization/detokenizer.h"

#include <filesystem>
#include <memory>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "src/tokenization/gpt2_model.h"

namespace pluto::tokenizer {

absl::StatusOr<std::unique_ptr<Gpt2Detokenizer>> Gpt2Detokenizer::Load(
    const std::filesystem::path& directory) {
  auto model = internal::Gpt2Model::Load(directory);
  if (!model.ok()) return model.status();
  return std::unique_ptr<Gpt2Detokenizer>(
      new Gpt2Detokenizer(std::move(*model)));
}

absl::StatusOr<std::string> Gpt2Detokenizer::Decode(
    absl::Span<const int> token_ids) const {
  std::string byte_encoded;
  for (const int id : token_ids) {
    if (id < 0 || id >= model_->vocab_size()) {
      return absl::InvalidArgumentError("token id is outside the vocabulary");
    }
    byte_encoded.append(model_->decoder()[id]);
  }

  std::string decoded;
  decoded.reserve(byte_encoded.size());
  absl::string_view remaining = byte_encoded;
  while (!remaining.empty()) {
    auto code_point = internal::ConsumeUtf8(&remaining);
    if (!code_point.ok()) return code_point.status();
    if (*code_point >= model_->byte_decoder().size() ||
        model_->byte_decoder()[*code_point] < 0) {
      return absl::DataLossError("token contains a non-GPT-2 byte scalar");
    }
    decoded.push_back(static_cast<char>(model_->byte_decoder()[*code_point]));
  }
  return decoded;
}

}  // namespace pluto::tokenizer
