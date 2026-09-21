#include "src/dataset/compact_vocabulary.h"

#include <charconv>
#include <cstddef>
#include <fstream>
#include <system_error>
#include <utility>

#include "absl/memory/memory.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "src/util/status_macros.h"

namespace pluto::tokenizer {
namespace {

absl::StatusOr<std::string> ReadMappingLine(std::istream& input) {
  std::string line;
  char character;
  while (input.get(character)) {
    if (character == '\n')
      return line;
    // Every canonical line contains at most a field name and one int, or
    // two ints. Bound storage even when a corrupt file has no newline.
    if (line.size() >= 64)
      return absl::DataLossError("compact vocabulary line is too long");
    line.push_back(character);
  }
  return absl::DataLossError("compact vocabulary is truncated or unreadable");
}

absl::StatusOr<int> ParseMappingNumber(absl::string_view text) {
  if (text.empty() || text.front() < '0' || text.front() > '9' ||
      (text.size() > 1 && text.front() == '0'))
    return absl::DataLossError("invalid compact vocabulary integer");
  int value;
  const auto result =
      std::from_chars(text.data(), text.data() + text.size(), value);
  if (result.ec != std::errc() || result.ptr != text.data() + text.size())
    return absl::DataLossError("invalid compact vocabulary integer");
  return value;
}

}  // namespace

absl::StatusOr<CompactVocabularyMapping> BuildCompactVocabularyMapping(
    cuda::Executor& executor, const Tokenizer& original,
    absl::string_view corpus_text, int original_eos_id) {
  const int original_size = original.vocab_size();
  if (original_size <= 0 || original_eos_id < 0 ||
      original_eos_id >= original_size)
    return absl::InvalidArgumentError("EOS is outside the original vocabulary");
  if (corpus_text.empty())
    return absl::InvalidArgumentError("the line corpus must not be empty");
  if (corpus_text.back() == '\n')
    corpus_text.remove_suffix(1);
  const std::vector<absl::string_view> lines =
      absl::StrSplit(corpus_text, '\n');
  std::vector<int> reverse(static_cast<size_t>(original_size), -1);
  reverse[original_eos_id] = 0;
  for (size_t line_index = 0; line_index < lines.size(); ++line_index) {
    absl::string_view line = lines[line_index];
    if (!line.empty() && line.back() == '\r')
      line.remove_suffix(1);
    if (absl::StripAsciiWhitespace(line).empty())
      return absl::InvalidArgumentError(
          absl::StrCat("line ", line_index + 1, " is empty"));
    ASSIGN_OR_RETURN(auto tokens, original.Encode(executor, line));
    if (tokens.empty())
      return absl::InvalidArgumentError(
          absl::StrCat("line ", line_index + 1, " has no token IDs"));
    for (int token : tokens) {
      if (token < 0 || token >= original_size)
        return absl::InvalidArgumentError(
            absl::StrCat("line ", line_index + 1,
                         " contains an invalid original token ID: ", token));
      reverse[token] = 0;
    }
  }
  std::vector<int> originals;
  for (int token = 0; token < original_size; ++token) {
    if (reverse[token] < 0)
      continue;
    reverse[token] = static_cast<int>(originals.size());
    originals.push_back(token);
  }
  return CompactVocabularyMapping{std::move(originals), std::move(reverse),
                                  original_eos_id};
}

absl::StatusOr<std::unique_ptr<CompactVocabularyTokenizer>>
CompactVocabularyTokenizer::Create(const Tokenizer& original,
                                   CompactVocabularyMapping mapping) {
  const int original_size = original.vocab_size();
  if (original_size <= 0 ||
      mapping.original_to_compact.size() != static_cast<size_t>(original_size))
    return absl::InvalidArgumentError(
        "reverse mapping must match the original vocabulary size");
  if (mapping.compact_to_original.empty() ||
      mapping.compact_to_original.size() > static_cast<size_t>(original_size))
    return absl::InvalidArgumentError("invalid compact vocabulary size");
  if (mapping.original_eos_id < 0 || mapping.original_eos_id >= original_size)
    return absl::InvalidArgumentError("EOS is outside the original vocabulary");

  // Validate the forward map before using its values as reverse-map indices.
  // Sorting keeps compact IDs and serialized mappings canonical.
  int previous_original = -1;
  for (size_t compact = 0; compact < mapping.compact_to_original.size();
       ++compact) {
    const int token = mapping.compact_to_original[compact];
    if (token <= previous_original || token >= original_size)
      return absl::InvalidArgumentError(
          "compact-to-original mapping must contain sorted unique valid IDs");
    if (mapping.original_to_compact[token] != static_cast<int>(compact))
      return absl::InvalidArgumentError("vocabulary mappings are not inverses");
    previous_original = token;
  }
  // Check inactive entries too: they must be -1, not aliases of active IDs.
  const int compact_size = static_cast<int>(mapping.compact_to_original.size());
  for (int token = 0; token < original_size; ++token) {
    const int compact = mapping.original_to_compact[token];
    if (compact == -1)
      continue;
    if (compact < 0 || compact >= compact_size ||
        mapping.compact_to_original[compact] != token)
      return absl::InvalidArgumentError("vocabulary mappings are not inverses");
  }
  if (mapping.original_to_compact[mapping.original_eos_id] == -1)
    return absl::InvalidArgumentError(
        "EOS is absent from the compact vocabulary");
  return absl::WrapUnique(
      new CompactVocabularyTokenizer(original, std::move(mapping)));
}

absl::StatusOr<std::unique_ptr<CompactVocabularyTokenizer>>
CompactVocabularyTokenizer::LoadFromFile(const Tokenizer& original,
                                         const std::filesystem::path& path) {
  if (path.empty())
    return absl::InvalidArgumentError("compact vocabulary path is empty");
  const int original_size = original.vocab_size();
  if (original_size <= 0)
    return absl::InvalidArgumentError("invalid original vocabulary size");
  std::error_code error;
  const bool exists = std::filesystem::exists(path, error);
  if (error)
    return absl::InternalError(absl::StrCat(
        "cannot inspect compact vocabulary path: ", error.message()));
  if (!exists)
    return absl::NotFoundError(
        absl::StrCat("compact vocabulary file is missing: ", path.string()));
  if (!std::filesystem::is_regular_file(path, error)) {
    if (error)
      return absl::InternalError(absl::StrCat(
          "cannot inspect compact vocabulary file: ", error.message()));
    return absl::FailedPreconditionError(
        "compact vocabulary path is not a regular file");
  }
  std::ifstream input(path, std::ios::binary);
  if (!input)
    return absl::InternalError(
        absl::StrCat("cannot open compact vocabulary file: ", path.string()));
  ASSIGN_OR_RETURN(const auto version, ReadMappingLine(input));
  if (version != "compact_vocabulary_v1")
    return absl::DataLossError("unsupported compact vocabulary version");
  auto read_field = [&](absl::string_view name) -> absl::StatusOr<int> {
    ASSIGN_OR_RETURN(const auto line, ReadMappingLine(input));
    const absl::string_view view(line);
    if (view.size() <= name.size() || view.substr(0, name.size()) != name ||
        view[name.size()] != '\t')
      return absl::DataLossError(
          absl::StrCat("invalid compact vocabulary field: ", name));
    return ParseMappingNumber(view.substr(name.size() + 1));
  };
  ASSIGN_OR_RETURN(const int saved_original_size,
                   read_field("original_vocab_size"));
  if (saved_original_size <= 0)
    return absl::DataLossError("invalid saved original vocabulary size");
  if (saved_original_size != original_size)
    return absl::FailedPreconditionError(
        "compact vocabulary does not match the original vocabulary size");
  ASSIGN_OR_RETURN(const int original_eos_id, read_field("original_eos_token"));
  ASSIGN_OR_RETURN(const int compact_size, read_field("compact_vocab_size"));
  if (compact_size <= 0 || compact_size > original_size)
    return absl::DataLossError("invalid saved compact vocabulary size");
  ASSIGN_OR_RETURN(const auto header, ReadMappingLine(input));
  if (header != "compact_id\toriginal_id")
    return absl::DataLossError("invalid compact vocabulary table header");

  CompactVocabularyMapping mapping;
  mapping.original_eos_id = original_eos_id;
  mapping.original_to_compact.assign(static_cast<size_t>(original_size), -1);
  mapping.compact_to_original.reserve(static_cast<size_t>(compact_size));
  for (int compact_id = 0; compact_id < compact_size; ++compact_id) {
    ASSIGN_OR_RETURN(const auto line, ReadMappingLine(input));
    const absl::string_view view(line);
    const size_t separator = view.find('\t');
    if (separator == absl::string_view::npos)
      return absl::DataLossError("invalid compact vocabulary row");
    ASSIGN_OR_RETURN(const int saved_compact_id,
                     ParseMappingNumber(view.substr(0, separator)));
    ASSIGN_OR_RETURN(const int original_id,
                     ParseMappingNumber(view.substr(separator + 1)));
    if (saved_compact_id != compact_id)
      return absl::DataLossError("compact vocabulary IDs are not contiguous");
    if (original_id >= original_size)
      return absl::DataLossError("original token ID is outside the vocabulary");
    mapping.compact_to_original.push_back(original_id);
    mapping.original_to_compact[original_id] = compact_id;
  }
  char extra;
  if (input.get(extra) || !input.eof() || input.bad())
    return absl::DataLossError(
        "compact vocabulary has trailing or unreadable data");
  auto compact = Create(original, std::move(mapping));
  if (!compact.ok())
    return absl::DataLossError(absl::StrCat(
        "invalid saved compact vocabulary: ", compact.status().message()));
  return compact;
}

CompactVocabularyTokenizer::CompactVocabularyTokenizer(
    const Tokenizer& original, CompactVocabularyMapping mapping)
    : original_(original),
      mapping_(std::move(mapping)),
      token_is_allowed_(mapping_.original_to_compact.size(), 0) {
  for (int original_id : mapping_.compact_to_original)
    token_is_allowed_[original_id] = 1;
}

absl::StatusOr<int> CompactVocabularyTokenizer::OriginalId(
    int compact_id) const {
  if (compact_id < 0 || compact_id >= vocab_size())
    return absl::InvalidArgumentError(absl::StrCat(
        "compact token ID is outside the vocabulary: ", compact_id));
  return mapping_.compact_to_original[compact_id];
}

absl::StatusOr<int> CompactVocabularyTokenizer::CompactId(
    int original_id) const {
  if (original_id < 0 || original_id >= original_vocab_size())
    return absl::InvalidArgumentError(absl::StrCat(
        "original token ID is outside the vocabulary: ", original_id));
  const int compact_id = mapping_.original_to_compact[original_id];
  if (compact_id < 0)
    return absl::InvalidArgumentError(absl::StrCat(
        "original token ID is absent from the compact vocabulary: ",
        original_id));
  return compact_id;
}

absl::StatusOr<cuda::PageLockedHostArray<int>>
CompactVocabularyTokenizer::Encode(cuda::Executor& executor,
                                   absl::string_view text) const {
  ASSIGN_OR_RETURN(auto original_tokens, original_.EncodeWithVocabulary(
                                             executor, text, token_is_allowed_));
  ASSIGN_OR_RETURN(auto compact_tokens,
                   cuda::PageLockedHostArray<int>::Allocate(
                       executor, original_tokens.size()));
  for (size_t index = 0; index < original_tokens.size(); ++index) {
    ASSIGN_OR_RETURN(compact_tokens[index], CompactId(original_tokens[index]));
  }
  return compact_tokens;
}

std::string CompactVocabularyTokenizer::CanonicalText() const {
  std::string text = absl::StrCat(
      "compact_vocabulary_v1\noriginal_vocab_size\t", original_vocab_size(),
      "\noriginal_eos_token\t", original_eos_token_id(),
      "\ncompact_vocab_size\t", vocab_size(), "\ncompact_id\toriginal_id\n");
  for (size_t index = 0; index < mapping_.compact_to_original.size(); ++index)
    absl::StrAppend(&text, index, "\t", mapping_.compact_to_original[index],
                    "\n");
  return text;
}

absl::Status CompactVocabularyTokenizer::SaveToFile(
    const std::filesystem::path& path) const {
  if (path.empty())
    return absl::InvalidArgumentError("compact vocabulary path is empty");
  std::error_code error;
  const bool exists = std::filesystem::exists(path, error);
  if (error)
    return absl::InternalError(absl::StrCat(
        "cannot inspect compact vocabulary path: ", error.message()));
  if (exists)
    return ValidateFile(path);
  std::ofstream output(path, std::ios::binary);
  if (!output)
    return absl::InternalError(
        absl::StrCat("cannot create compact vocabulary file: ", path.string()));
  output << CanonicalText();
  output.close();
  if (!output)
    return absl::DataLossError(
        absl::StrCat("cannot write compact vocabulary file: ", path.string()));
  return absl::OkStatus();
}

absl::Status CompactVocabularyTokenizer::ValidateFile(
    const std::filesystem::path& path) const {
  if (path.empty())
    return absl::InvalidArgumentError("compact vocabulary path is empty");
  std::error_code error;
  const bool exists = std::filesystem::exists(path, error);
  if (error)
    return absl::InternalError(absl::StrCat(
        "cannot inspect compact vocabulary path: ", error.message()));
  if (!exists)
    return absl::NotFoundError(
        absl::StrCat("compact vocabulary file is missing: ", path.string()));
  if (!std::filesystem::is_regular_file(path, error)) {
    if (error)
      return absl::InternalError(absl::StrCat(
          "cannot inspect compact vocabulary file: ", error.message()));
    return absl::FailedPreconditionError(
        "compact vocabulary path is not a regular file");
  }
  const std::string expected = CanonicalText();
  const auto size = std::filesystem::file_size(path, error);
  if (error)
    return absl::InternalError(absl::StrCat(
        "cannot read compact vocabulary file size: ", error.message()));
  if (size != expected.size())
    return absl::DataLossError(
        "compact vocabulary file has an unexpected size");
  std::ifstream input(path, std::ios::binary);
  if (!input)
    return absl::InternalError(
        absl::StrCat("cannot open compact vocabulary file: ", path.string()));
  std::string actual(expected.size(), '\0');
  input.read(actual.data(), static_cast<std::streamsize>(actual.size()));
  if (input.gcount() != static_cast<std::streamsize>(actual.size()) ||
      input.bad())
    return absl::DataLossError("cannot read the complete compact vocabulary");
  char extra;
  if (input.get(extra) || !input.eof())
    return absl::DataLossError("compact vocabulary changed while reading");
  if (actual != expected)
    return absl::DataLossError(
        "compact vocabulary does not match the corpus/tokenizer ID mapping");
  return absl::OkStatus();
}

}  // namespace pluto::tokenizer
