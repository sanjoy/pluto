#include "src/dataset/gpt2_tokenizer_vocabulary.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <limits>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "src/util/status_macros.h"

namespace pluto::tokenizer::internal {
namespace {

absl::Status JsonError(size_t position, absl::string_view message) {
  return absl::InvalidArgumentError(
      absl::StrCat("invalid tokenizer.json at byte ", position, ": ", message));
}

void AppendUtf8(uint32_t code_point, std::string* output) {
  if (code_point <= 0x7f) {
    output->push_back(static_cast<char>(code_point));
  } else if (code_point <= 0x7ff) {
    output->push_back(static_cast<char>(0xc0 | (code_point >> 6)));
    output->push_back(static_cast<char>(0x80 | (code_point & 0x3f)));
  } else if (code_point <= 0xffff) {
    output->push_back(static_cast<char>(0xe0 | (code_point >> 12)));
    output->push_back(static_cast<char>(0x80 | ((code_point >> 6) & 0x3f)));
    output->push_back(static_cast<char>(0x80 | (code_point & 0x3f)));
  } else {
    output->push_back(static_cast<char>(0xf0 | (code_point >> 18)));
    output->push_back(static_cast<char>(0x80 | ((code_point >> 12) & 0x3f)));
    output->push_back(static_cast<char>(0x80 | ((code_point >> 6) & 0x3f)));
    output->push_back(static_cast<char>(0x80 | (code_point & 0x3f)));
  }
}

class JsonCursor {
 public:
  explicit JsonCursor(absl::string_view input) : input_(input) {}

  size_t position() const { return position_; }

  void SkipWhitespace() {
    while (position_ < input_.size() &&
           std::isspace(static_cast<unsigned char>(input_[position_]))) {
      ++position_;
    }
  }

  char Peek() {
    SkipWhitespace();
    return position_ < input_.size() ? input_[position_] : '\0';
  }

  bool Consume(char expected) {
    SkipWhitespace();
    if (position_ >= input_.size() || input_[position_] != expected)
      return false;
    ++position_;
    return true;
  }

  absl::Status Expect(char expected) {
    if (Consume(expected))
      return absl::OkStatus();
    return JsonError(position_,
                     absl::StrCat("expected '", std::string(1, expected), "'"));
  }

  absl::StatusOr<std::string> ParseString() {
    SkipWhitespace();
    if (position_ >= input_.size() || input_[position_++] != '"')
      return JsonError(position_, "expected string");

    std::string result;
    while (position_ < input_.size()) {
      const unsigned char byte = input_[position_++];
      if (byte == '"')
        return result;
      if (byte < 0x20)
        return JsonError(position_ - 1, "control byte in string");
      if (byte != '\\') {
        result.push_back(static_cast<char>(byte));
        continue;
      }

      if (position_ >= input_.size())
        return JsonError(position_, "truncated string escape");
      const char escape = input_[position_++];
      switch (escape) {
        case '"':
          result.push_back('"');
          break;
        case '\\':
          result.push_back('\\');
          break;
        case '/':
          result.push_back('/');
          break;
        case 'b':
          result.push_back('\b');
          break;
        case 'f':
          result.push_back('\f');
          break;
        case 'n':
          result.push_back('\n');
          break;
        case 'r':
          result.push_back('\r');
          break;
        case 't':
          result.push_back('\t');
          break;
        case 'u': {
          ASSIGN_OR_RETURN(uint32_t scalar, ParseHexQuad());
          if (scalar >= 0xd800 && scalar <= 0xdbff) {
            if (position_ + 2 > input_.size() || input_[position_] != '\\' ||
                input_[position_ + 1] != 'u') {
              return JsonError(position_,
                               "high surrogate without low surrogate");
            }
            position_ += 2;
            ASSIGN_OR_RETURN(uint32_t low, ParseHexQuad());
            if (low < 0xdc00 || low > 0xdfff)
              return JsonError(position_, "invalid low surrogate");
            scalar = 0x10000 + ((scalar - 0xd800) << 10) + (low - 0xdc00);
          } else if (scalar >= 0xdc00 && scalar <= 0xdfff) {
            return JsonError(position_, "unexpected low surrogate");
          }
          AppendUtf8(scalar, &result);
          break;
        }
        default:
          return JsonError(position_ - 1, "unknown string escape");
      }
    }
    return JsonError(position_, "unterminated string");
  }

  absl::StatusOr<int> ParseNonnegativeInt() {
    SkipWhitespace();
    const size_t begin = position_;
    while (position_ < input_.size() &&
           std::isdigit(static_cast<unsigned char>(input_[position_]))) {
      ++position_;
    }
    int value = 0;
    if (begin == position_ ||
        !absl::SimpleAtoi(input_.substr(begin, position_ - begin), &value)) {
      return JsonError(begin, "expected nonnegative integer");
    }
    return value;
  }

  absl::Status SkipValue() {
    switch (Peek()) {
      case '"': {
        auto value = ParseString();
        return value.ok() ? absl::OkStatus() : value.status();
      }
      case '{': {
        ++position_;
        if (Consume('}'))
          return absl::OkStatus();
        while (true) {
          auto key = ParseString();
          if (!key.ok())
            return key.status();
          RETURN_IF_ERROR(Expect(':'));
          RETURN_IF_ERROR(SkipValue());
          if (Consume('}'))
            return absl::OkStatus();
          RETURN_IF_ERROR(Expect(','));
        }
      }
      case '[': {
        ++position_;
        if (Consume(']'))
          return absl::OkStatus();
        while (true) {
          RETURN_IF_ERROR(SkipValue());
          if (Consume(']'))
            return absl::OkStatus();
          RETURN_IF_ERROR(Expect(','));
        }
      }
      default: {
        const size_t begin = position_;
        while (position_ < input_.size()) {
          const char c = input_[position_];
          if (c == ',' || c == ']' || c == '}' ||
              std::isspace(static_cast<unsigned char>(c))) {
            break;
          }
          ++position_;
        }
        if (begin == position_)
          return JsonError(begin, "expected value");
        return absl::OkStatus();
      }
    }
  }

 private:
  absl::StatusOr<uint32_t> ParseHexQuad() {
    if (position_ + 4 > input_.size())
      return JsonError(position_, "truncated Unicode escape");
    uint32_t value = 0;
    for (int i = 0; i < 4; ++i) {
      const char c = input_[position_++];
      value <<= 4;
      if (c >= '0' && c <= '9')
        value += c - '0';
      else if (c >= 'a' && c <= 'f')
        value += c - 'a' + 10;
      else if (c >= 'A' && c <= 'F')
        value += c - 'A' + 10;
      else
        return JsonError(position_ - 1, "invalid hexadecimal digit");
    }
    return value;
  }

  absl::string_view input_;
  size_t position_ = 0;
};

absl::Status ParseVocabulary(JsonCursor* cursor,
                             absl::flat_hash_map<std::string, int>* vocab) {
  RETURN_IF_ERROR(cursor->Expect('{'));
  if (cursor->Consume('}'))
    return absl::OkStatus();
  while (true) {
    ASSIGN_OR_RETURN(auto token, cursor->ParseString());
    RETURN_IF_ERROR(cursor->Expect(':'));
    ASSIGN_OR_RETURN(int id, cursor->ParseNonnegativeInt());
    if (!vocab->emplace(std::move(token), id).second)
      return JsonError(cursor->position(), "duplicate vocabulary token");
    if (cursor->Consume('}'))
      return absl::OkStatus();
    RETURN_IF_ERROR(cursor->Expect(','));
  }
}

absl::Status ParseMerges(JsonCursor* cursor,
                         absl::flat_hash_map<std::string, int>* ranks) {
  RETURN_IF_ERROR(cursor->Expect('['));
  if (cursor->Consume(']'))
    return absl::OkStatus();

  int rank = 0;
  while (true) {
    std::string left;
    std::string right;
    if (cursor->Peek() == '[') {
      cursor->Consume('[');
      ASSIGN_OR_RETURN(left, cursor->ParseString());
      RETURN_IF_ERROR(cursor->Expect(','));
      ASSIGN_OR_RETURN(right, cursor->ParseString());
      RETURN_IF_ERROR(cursor->Expect(']'));
    } else {
      // Older tokenizer.json files encode a merge as one space-separated
      // string. GPT-2 alphabet symbols themselves never contain ASCII space.
      ASSIGN_OR_RETURN(auto merge, cursor->ParseString());
      const size_t separator = merge.find(' ');
      if (separator == std::string::npos)
        return JsonError(cursor->position(), "merge is not a token pair");
      left = merge.substr(0, separator);
      right = merge.substr(separator + 1);
    }
    ranks->emplace(MergeKey(left, right), rank++);

    if (cursor->Consume(']'))
      return absl::OkStatus();
    RETURN_IF_ERROR(cursor->Expect(','));
  }
}

absl::Status ParseModel(JsonCursor* cursor,
                        absl::flat_hash_map<std::string, int>* vocab,
                        absl::flat_hash_map<std::string, int>* ranks) {
  RETURN_IF_ERROR(cursor->Expect('{'));
  if (cursor->Consume('}'))
    return absl::OkStatus();
  while (true) {
    ASSIGN_OR_RETURN(auto key, cursor->ParseString());
    RETURN_IF_ERROR(cursor->Expect(':'));
    if (key == "vocab")
      RETURN_IF_ERROR(ParseVocabulary(cursor, vocab));
    else if (key == "merges")
      RETURN_IF_ERROR(ParseMerges(cursor, ranks));
    else
      RETURN_IF_ERROR(cursor->SkipValue());
    if (cursor->Consume('}'))
      return absl::OkStatus();
    RETURN_IF_ERROR(cursor->Expect(','));
  }
}

absl::Status ParseTokenizerJson(absl::string_view json,
                                absl::flat_hash_map<std::string, int>* vocab,
                                absl::flat_hash_map<std::string, int>* ranks) {
  JsonCursor cursor(json);
  RETURN_IF_ERROR(cursor.Expect('{'));
  if (cursor.Consume('}'))
    return JsonError(0, "missing model");

  bool found_model = false;
  while (true) {
    ASSIGN_OR_RETURN(auto key, cursor.ParseString());
    RETURN_IF_ERROR(cursor.Expect(':'));
    if (key == "model") {
      RETURN_IF_ERROR(ParseModel(&cursor, vocab, ranks));
      found_model = true;
    } else {
      RETURN_IF_ERROR(cursor.SkipValue());
    }
    if (cursor.Consume('}'))
      break;
    RETURN_IF_ERROR(cursor.Expect(','));
  }
  if (!found_model || vocab->empty() || ranks->empty())
    return JsonError(cursor.position(), "missing byte-level BPE model");
  return absl::OkStatus();
}

absl::StatusOr<std::string> ReadFile(const std::filesystem::path& path) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream)
    return absl::NotFoundError(absl::StrCat("cannot open ", path.string()));
  const std::streamoff size = stream.tellg();
  if (size < 0)
    return absl::InternalError(absl::StrCat("cannot size ", path.string()));
  std::string contents(static_cast<size_t>(size), '\0');
  stream.seekg(0);
  if (!contents.empty() && !stream.read(contents.data(), size))
    return absl::DataLossError(absl::StrCat("cannot read ", path.string()));
  return contents;
}

std::array<std::string, 256> MakeByteEncoder() {
  std::array<bool, 256> direct{};
  for (int byte = 33; byte <= 126; ++byte)
    direct[byte] = true;
  for (int byte = 161; byte <= 172; ++byte)
    direct[byte] = true;
  for (int byte = 174; byte <= 255; ++byte)
    direct[byte] = true;

  std::array<std::string, 256> encoder;
  uint32_t replacement = 256;
  for (int byte = 0; byte < 256; ++byte)
    AppendUtf8(direct[byte] ? byte : replacement++, &encoder[byte]);
  return encoder;
}

struct ModelCache {
  absl::Mutex mutex;
  absl::flat_hash_map<std::string, std::weak_ptr<const Gpt2TokenizerVocabulary>>
      models;
};

ModelCache& SharedModelCache() {
  // Deliberately leaked to avoid static-destruction ordering problems. Models
  // are immutable; weak_ptr entries do not extend caller-owned lifetimes.
  static auto* cache = new ModelCache;
  return *cache;
}

}  // namespace

std::string MergeKey(absl::string_view left, absl::string_view right) {
  std::string key;
  key.reserve(left.size() + 1 + right.size());
  key.append(left.data(), left.size());
  key.push_back('\0');
  key.append(right.data(), right.size());
  return key;
}

absl::StatusOr<uint32_t> ConsumeUtf8(absl::string_view* input) {
  if (input->empty())
    return absl::OutOfRangeError("end of UTF-8 input");
  const auto first = static_cast<unsigned char>((*input)[0]);
  int length = 0;
  uint32_t value = 0;
  if (first < 0x80) {
    length = 1;
    value = first;
  } else if ((first & 0xe0) == 0xc0) {
    length = 2;
    value = first & 0x1f;
  } else if ((first & 0xf0) == 0xe0) {
    length = 3;
    value = first & 0x0f;
  } else if ((first & 0xf8) == 0xf0) {
    length = 4;
    value = first & 0x07;
  } else {
    return absl::InvalidArgumentError("invalid UTF-8 leading byte");
  }
  if (input->size() < static_cast<size_t>(length))
    return absl::InvalidArgumentError("truncated UTF-8 scalar");
  for (int i = 1; i < length; ++i) {
    const auto byte = static_cast<unsigned char>((*input)[i]);
    if ((byte & 0xc0) != 0x80)
      return absl::InvalidArgumentError("invalid UTF-8 continuation byte");
    value = (value << 6) | (byte & 0x3f);
  }
  static constexpr uint32_t kMinimum[] = {0, 0, 0x80, 0x800, 0x10000};
  if (value < kMinimum[length] || value > 0x10ffff ||
      (value >= 0xd800 && value <= 0xdfff)) {
    return absl::InvalidArgumentError("invalid UTF-8 scalar value");
  }
  input->remove_prefix(length);
  return value;
}

absl::StatusOr<std::shared_ptr<const Gpt2TokenizerVocabulary>>
Gpt2TokenizerVocabulary::Load(const std::filesystem::path& directory) {
  const std::string cache_key =
      std::filesystem::absolute(directory).lexically_normal().string();
  ModelCache& cache = SharedModelCache();
  {
    absl::MutexLock lock(cache.mutex);
    const auto found = cache.models.find(cache_key);
    if (found != cache.models.end()) {
      if (std::shared_ptr<const Gpt2TokenizerVocabulary> model =
              found->second.lock())
        return model;
    }
  }

  ASSIGN_OR_RETURN(auto json, ReadFile(directory / "tokenizer.json"));

  std::shared_ptr<Gpt2TokenizerVocabulary> model(new Gpt2TokenizerVocabulary);
  RETURN_IF_ERROR(
      ParseTokenizerJson(json, &model->encoder_, &model->merge_ranks_));

  int maximum_id = -1;
  for (const auto& [token, id] : model->encoder_)
    maximum_id = std::max(maximum_id, id);
  model->decoder_.resize(static_cast<size_t>(maximum_id) + 1);
  std::vector<bool> seen(model->decoder_.size());
  for (const auto& [token, id] : model->encoder_) {
    if (id < 0 || id > maximum_id || seen[id]) {
      return absl::InvalidArgumentError(
          "vocabulary ids are invalid or duplicated");
    }
    seen[id] = true;
    model->decoder_[id] = token;
  }
  if (std::find(seen.begin(), seen.end(), false) != seen.end())
    return absl::InvalidArgumentError("vocabulary ids are not contiguous");

  model->eos_token_ = "<|endoftext|>";
  const auto eos = model->encoder_.find(model->eos_token_);
  if (eos == model->encoder_.end())
    return absl::InvalidArgumentError("GPT-2 EOS token is missing");
  model->eos_token_id_ = eos->second;

  model->byte_encoder_ = MakeByteEncoder();
  model->byte_decoder_.fill(-1);
  for (int byte = 0; byte < 256; ++byte) {
    absl::string_view encoded = model->byte_encoder_[byte];
    auto code_point = ConsumeUtf8(&encoded);
    if (!code_point.ok() || !encoded.empty() ||
        *code_point >= model->byte_decoder_.size()) {
      return absl::InternalError("invalid generated byte alphabet");
    }
    model->byte_decoder_[*code_point] = byte;
  }

  std::shared_ptr<const Gpt2TokenizerVocabulary> immutable_model =
      std::move(model);
  {
    absl::MutexLock lock(cache.mutex);
    cache.models[cache_key] = immutable_model;
  }
  return immutable_model;
}

}  // namespace pluto::tokenizer::internal
