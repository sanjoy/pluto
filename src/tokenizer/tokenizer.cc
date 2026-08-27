#include "src/tokenizer/tokenizer.h"

#include <algorithm>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "re2/re2.h"
#include "re2/stringpiece.h"
#include "src/tokenizer/gpt2_model.h"

namespace pluto::tokenizer {
namespace {

constexpr size_t kMaximumCacheEntries = 65536;

// This is GPT-2's pre-tokenization expression, minus its final whitespace
// branch. Whitespace is handled immediately below so we can reproduce the
// regex package's negative-lookahead behavior using RE2, which deliberately
// does not implement lookaround.
const RE2& TokenPattern() {
  static const auto* pattern = new RE2(
      R"((?:'(?:s|t|re|ve|m|ll|d)| ?\p{L}+| ?\p{N}+| ?[^\p{L}\p{N}\p{Z}\x{0009}-\x{000D}\x{0085}]+))");
  return *pattern;
}

const RE2& WhitespacePattern() {
  static const auto* pattern =
      new RE2(R"((?:[\p{Z}\x{0009}-\x{000D}\x{0085}]+))");
  return *pattern;
}

absl::StatusOr<std::vector<absl::string_view>> PreTokenize(
    absl::string_view text) {
  if (!TokenPattern().ok() || !WhitespacePattern().ok()) {
    return absl::InternalError("GPT-2 pre-tokenizer regex failed to compile");
  }

  std::vector<absl::string_view> pieces;
  absl::string_view remaining = text;
  while (!remaining.empty()) {
    re2::StringPiece candidate(remaining.data(), remaining.size());
    if (RE2::Consume(&candidate, TokenPattern())) {
      const size_t consumed = remaining.size() - candidate.size();
      pieces.push_back(remaining.substr(0, consumed));
      remaining.remove_prefix(consumed);
      continue;
    }

    candidate = re2::StringPiece(remaining.data(), remaining.size());
    if (!RE2::Consume(&candidate, WhitespacePattern())) {
      return absl::InvalidArgumentError(
          absl::StrCat("input is not valid UTF-8 near byte ",
                       text.size() - remaining.size()));
    }
    size_t consumed = remaining.size() - candidate.size();
    // For a run like "   word", the original pattern emits "  " and leaves
    // the last ASCII space to become the prefix of " word".
    if (!candidate.empty() && consumed > 1 && remaining[consumed - 1] == ' ') {
      --consumed;
    }
    pieces.push_back(remaining.substr(0, consumed));
    remaining.remove_prefix(consumed);
  }
  return pieces;
}

std::string ByteEncode(absl::string_view text,
                       const internal::Gpt2Model& model) {
  std::string encoded;
  encoded.reserve(text.size() * 2);
  for (const unsigned char byte : text) encoded.append(model.byte_encoder()[byte]);
  return encoded;
}

absl::StatusOr<std::vector<std::string>> SplitUtf8(absl::string_view text) {
  std::vector<std::string> symbols;
  while (!text.empty()) {
    const char* begin = text.data();
    auto code_point = internal::ConsumeUtf8(&text);
    if (!code_point.ok()) return code_point.status();
    symbols.emplace_back(begin, static_cast<size_t>(text.data() - begin));
  }
  return symbols;
}

}  // namespace

absl::StatusOr<std::unique_ptr<Gpt2Tokenizer>> Gpt2Tokenizer::Load(
    const std::filesystem::path& directory) {
  auto model = internal::Gpt2Model::Load(directory);
  if (!model.ok()) return model.status();
  return std::unique_ptr<Gpt2Tokenizer>(new Gpt2Tokenizer(std::move(*model)));
}

absl::StatusOr<std::vector<int>> Gpt2Tokenizer::ApplyBpe(
    std::string token) const {
  {
    absl::MutexLock lock(cache_mutex_);
    const auto cached = cache_.find(token);
    if (cached != cache_.end()) return cached->second;
  }

  auto split = SplitUtf8(token);
  if (!split.ok()) return split.status();
  std::vector<std::string> symbols = std::move(*split);

  while (symbols.size() > 1) {
    int best_rank = std::numeric_limits<int>::max();
    std::string best_left;
    std::string best_right;
    for (size_t i = 0; i + 1 < symbols.size(); ++i) {
      const auto rank =
          model_->merge_ranks().find(internal::MergeKey(symbols[i], symbols[i + 1]));
      if (rank != model_->merge_ranks().end() && rank->second < best_rank) {
        best_rank = rank->second;
        best_left = symbols[i];
        best_right = symbols[i + 1];
      }
    }
    if (best_rank == std::numeric_limits<int>::max()) break;

    std::vector<std::string> merged;
    merged.reserve(symbols.size());
    for (size_t i = 0; i < symbols.size();) {
      if (i + 1 < symbols.size() && symbols[i] == best_left &&
          symbols[i + 1] == best_right) {
        merged.push_back(absl::StrCat(symbols[i], symbols[i + 1]));
        i += 2;
      } else {
        merged.push_back(std::move(symbols[i++]));
      }
    }
    symbols = std::move(merged);
  }

  std::vector<int> ids;
  ids.reserve(symbols.size());
  for (const std::string& symbol : symbols) {
    const auto id = model_->encoder().find(symbol);
    if (id == model_->encoder().end()) {
      return absl::DataLossError("BPE produced a token absent from the vocabulary");
    }
    ids.push_back(id->second);
  }

  {
    absl::MutexLock lock(cache_mutex_);
    if (cache_.size() < kMaximumCacheEntries) cache_.emplace(std::move(token), ids);
  }
  return ids;
}

absl::Status Gpt2Tokenizer::EncodeOrdinary(absl::string_view text,
                                           std::vector<int>* output) const {
  auto pieces = PreTokenize(text);
  if (!pieces.ok()) return pieces.status();
  for (absl::string_view piece : *pieces) {
    auto ids = ApplyBpe(ByteEncode(piece, *model_));
    if (!ids.ok()) return ids.status();
    output->insert(output->end(), ids->begin(), ids->end());
  }
  return absl::OkStatus();
}

absl::StatusOr<std::vector<int>> Gpt2Tokenizer::Encode(
    absl::string_view text) const {
  std::vector<int> output;
  size_t begin = 0;
  while (begin < text.size()) {
    const size_t special = text.find(model_->eos_token(), begin);
    const size_t end = special == absl::string_view::npos ? text.size() : special;
    auto status = EncodeOrdinary(text.substr(begin, end - begin), &output);
    if (!status.ok()) return status;
    if (special == absl::string_view::npos) break;
    output.push_back(model_->eos_token_id());
    begin = special + model_->eos_token().size();
  }
  return output;
}

}  // namespace pluto::tokenizer
