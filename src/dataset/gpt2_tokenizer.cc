#include "src/dataset/gpt2_tokenizer.h"

#include <algorithm>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/escaping.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "re2/re2.h"
#include "re2/stringpiece.h"
#include "src/cuda/page_locked_host_array.h"
#include "src/dataset/gpt2_tokenizer_vocabulary.h"
#include "src/util/status_macros.h"

namespace pluto::tokenizer {
namespace {

constexpr size_t kMaximumCacheEntries = 65536;

// This is GPT-2's pre-tokenization expression, minus its final whitespace
// branch. Whitespace is handled immediately below so we can reproduce the
// regex package's negative-lookahead behavior using RE2, which deliberately
// does not implement lookaround. RE2 permits surrogate code points in a
// negated UTF-8 character class, so exclude Unicode category Cs explicitly.
// Surrogates are not Unicode scalar values: rejecting them during this same
// regex scan avoids a separate UTF-8 validation pass over every input byte.
const RE2& TokenPattern() {
  static const auto* pattern = new RE2(
      R"((?:'(?:s|t|re|ve|m|ll|d)| ?\p{L}+| ?\p{N}+| ?[^\p{L}\p{N}\p{Z}\p{Cs}\x{0009}-\x{000D}\x{0085}]+))");
  return *pattern;
}

const RE2& WhitespacePattern() {
  static const auto* pattern =
      new RE2(R"((?:[\p{Z}\x{0009}-\x{000D}\x{0085}]+))");
  return *pattern;
}

absl::StatusOr<std::vector<absl::string_view>> PreTokenize(
    absl::string_view text) {
  if (!TokenPattern().ok() || !WhitespacePattern().ok())
    return absl::InternalError("GPT-2 pre-tokenizer regex failed to compile");

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
    if (!candidate.empty() && consumed > 1 && remaining[consumed - 1] == ' ')
      --consumed;
    pieces.push_back(remaining.substr(0, consumed));
    remaining.remove_prefix(consumed);
  }
  return pieces;
}

std::string ByteEncode(absl::string_view text,
                       const internal::Gpt2TokenizerVocabulary& model) {
  std::string encoded;
  encoded.reserve(text.size() * 2);
  for (const unsigned char byte : text)
    encoded.append(model.byte_encoder()[byte]);
  return encoded;
}

absl::StatusOr<std::vector<std::string>> SplitUtf8(absl::string_view text) {
  std::vector<std::string> symbols;
  while (!text.empty()) {
    const char* begin = text.data();
    auto code_point = internal::ConsumeUtf8(&text);
    if (!code_point.ok())
      return code_point.status();
    symbols.emplace_back(begin, static_cast<size_t>(text.data() - begin));
  }
  return symbols;
}

absl::Status UnencodableSubstring(absl::string_view piece, size_t byte_offset) {
  return absl::InvalidArgumentError(absl::StrCat(
      "substring \"", absl::Utf8SafeCEscape(piece),
      "\" could not be encoded using the compact vocabulary at byte ",
      byte_offset));
}

absl::Status AppendVocabularySegmentation(
    absl::string_view piece, size_t byte_offset,
    absl::Span<const uint8_t> token_is_allowed,
    const internal::Gpt2TokenizerVocabulary& model,
    size_t maximum_encoded_token_length, std::vector<int>* output) {
  struct Suffix {
    size_t token_count = std::numeric_limits<size_t>::max();
    size_t next = 0;
    int token_id = -1;
  };
  std::vector<Suffix> suffixes(piece.size() + 1);
  suffixes.back().token_count = 0;
  std::string candidate;
  candidate.reserve(maximum_encoded_token_length);
  for (size_t begin = piece.size(); begin > 0;) {
    --begin;
    candidate.clear();
    // Each raw byte expands to at least one encoded byte, so this bound keeps
    // candidate enumeration linear in input length for a fixed vocabulary.
    const size_t limit =
        begin + std::min(piece.size() - begin, maximum_encoded_token_length);
    for (size_t end = begin; end < limit; ++end) {
      candidate.append(
          model.byte_encoder()[static_cast<unsigned char>(piece[end])]);
      if (candidate.size() > maximum_encoded_token_length)
        break;
      const Suffix& following = suffixes[end + 1];
      if (following.token_id < 0 && end + 1 != piece.size())
        continue;
      const auto token = model.encoder().find(candidate);
      if (token == model.encoder().end() || !token_is_allowed[token->second])
        continue;
      const size_t count = following.token_count + 1;
      Suffix& best = suffixes[begin];
      // Minimize token count; equal-cost paths prefer a longer first token,
      // then a lower vocabulary ID. The suffix has already made the same
      // deterministic choice, so unordered-map iteration cannot affect output.
      if (count < best.token_count ||
          (count == best.token_count &&
           (end + 1 > best.next ||
            (end + 1 == best.next && token->second < best.token_id)))) {
        best = {count, end + 1, token->second};
      }
    }
  }
  if (suffixes.front().token_id < 0)
    return UnencodableSubstring(piece, byte_offset);
  for (size_t begin = 0; begin < piece.size(); begin = suffixes[begin].next)
    output->push_back(suffixes[begin].token_id);
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<std::unique_ptr<Gpt2Tokenizer>> Gpt2Tokenizer::Load(
    const std::filesystem::path& directory) {
  ASSIGN_OR_RETURN(auto model,
                   internal::Gpt2TokenizerVocabulary::Load(directory));
  size_t maximum_encoded_token_length = 0;
  for (const auto& token : model->decoder())
    maximum_encoded_token_length =
        std::max(maximum_encoded_token_length, token.size());
  return absl::WrapUnique(
      new Gpt2Tokenizer(std::move(model), maximum_encoded_token_length));
}

absl::StatusOr<std::vector<int>> Gpt2Tokenizer::ApplyBpe(
    std::string token) const {
  {
    absl::MutexLock lock(cache_mutex_);
    const auto cached = cache_.find(token);
    if (cached != cache_.end())
      return cached->second;
  }

  ASSIGN_OR_RETURN(std::vector<std::string> symbols, SplitUtf8(token));

  while (symbols.size() > 1) {
    int best_rank = std::numeric_limits<int>::max();
    std::string best_left;
    std::string best_right;
    for (size_t i = 0; i + 1 < symbols.size(); ++i) {
      const auto rank = model_->merge_ranks().find(
          internal::MergeKey(symbols[i], symbols[i + 1]));
      if (rank != model_->merge_ranks().end() && rank->second < best_rank) {
        best_rank = rank->second;
        best_left = symbols[i];
        best_right = symbols[i + 1];
      }
    }
    if (best_rank == std::numeric_limits<int>::max())
      break;

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
      return absl::DataLossError(
          "BPE produced a token absent from the vocabulary");
    }
    ids.push_back(id->second);
  }

  {
    absl::MutexLock lock(cache_mutex_);
    if (cache_.size() < kMaximumCacheEntries)
      cache_.emplace(std::move(token), ids);
  }
  return ids;
}

absl::Status Gpt2Tokenizer::EncodeOrdinary(absl::string_view text,
                                           std::vector<int>* output) const {
  ASSIGN_OR_RETURN(auto pieces, PreTokenize(text));
  for (absl::string_view piece : pieces) {
    ASSIGN_OR_RETURN(auto ids, ApplyBpe(ByteEncode(piece, *model_)));
    output->insert(output->end(), ids.begin(), ids.end());
  }
  return absl::OkStatus();
}

absl::Status Gpt2Tokenizer::EncodeOrdinaryWithVocabulary(
    absl::string_view text, size_t byte_offset,
    absl::Span<const uint8_t> token_is_allowed,
    std::vector<int>* output) const {
  ASSIGN_OR_RETURN(auto pieces, PreTokenize(text));
  for (absl::string_view piece : pieces) {
    ASSIGN_OR_RETURN(auto ids, ApplyBpe(ByteEncode(piece, *model_)));
    if (std::all_of(ids.begin(), ids.end(), [token_is_allowed](int id) {
          return token_is_allowed[id];
        })) {
      // Training-corpus tokens must retain their exact original segmentation
      // so existing compact corpora and checkpoint IDs remain compatible.
      output->insert(output->end(), ids.begin(), ids.end());
      continue;
    }
    // Search the whole pretoken, including across original BPE boundaries.
    // Only terminal vocabulary entries must be allowed: their intermediate
    // BPE merges and individual UTF-8 bytes need not be retained. Unlike the
    // unrestricted BPE result, this mask-dependent result is never cached.
    RETURN_IF_ERROR(AppendVocabularySegmentation(
        piece, byte_offset + static_cast<size_t>(piece.data() - text.data()),
        token_is_allowed, *model_, maximum_encoded_token_length_, output));
  }
  return absl::OkStatus();
}

absl::StatusOr<cuda::PageLockedHostArray<int>> Gpt2Tokenizer::Encode(
    cuda::Executor& executor, absl::string_view text) const {
  std::vector<int> output;
  size_t begin = 0;
  while (begin < text.size()) {
    const size_t special = text.find(model_->eos_token(), begin);
    const size_t end =
        special == absl::string_view::npos ? text.size() : special;
    RETURN_IF_ERROR(EncodeOrdinary(text.substr(begin, end - begin), &output));
    if (special == absl::string_view::npos)
      break;
    output.push_back(model_->eos_token_id());
    begin = special + model_->eos_token().size();
  }
  return cuda::PageLockedHostArray<int>::CopyFrom(executor, output);
}

absl::StatusOr<cuda::PageLockedHostArray<int>>
Gpt2Tokenizer::EncodeWithVocabulary(
    cuda::Executor& executor, absl::string_view text,
    absl::Span<const uint8_t> token_is_allowed) const {
  if (token_is_allowed.size() != static_cast<size_t>(vocab_size())) {
    return absl::InvalidArgumentError(
        "allowed-token mask size must equal the tokenizer vocabulary size");
  }
  std::vector<int> output;
  size_t begin = 0;
  while (begin < text.size()) {
    const size_t special = text.find(model_->eos_token(), begin);
    const size_t end =
        special == absl::string_view::npos ? text.size() : special;
    RETURN_IF_ERROR(EncodeOrdinaryWithVocabulary(
        text.substr(begin, end - begin), begin, token_is_allowed, &output));
    if (special == absl::string_view::npos)
      break;
    // EOS is atomic even when its literal text could be split into byte tokens.
    if (!token_is_allowed[model_->eos_token_id()])
      return UnencodableSubstring(model_->eos_token(), special);
    output.push_back(model_->eos_token_id());
    begin = special + model_->eos_token().size();
  }
  return cuda::PageLockedHostArray<int>::CopyFrom(executor, output);
}

}  // namespace pluto::tokenizer
