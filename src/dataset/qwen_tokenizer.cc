#include "src/dataset/qwen_tokenizer.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <queue>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "re2/re2.h"
#include "src/dataset/gpt2_tokenizer_vocabulary.h"
#include "src/util/status_macros.h"
#include "utf8proc.h"

namespace pluto::tokenizer {
namespace {

constexpr absl::string_view kPreTokenizeRegex =
    R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)";
constexpr size_t kMaximumCacheEntries = 65536;
constexpr size_t kMaximumTokenizerBytes = 64 * 1024 * 1024;

// A bounded streaming reader avoids materializing a second 248k-token JSON
// object tree. Vocabulary keys are decoded directly into their owning map.
class JsonCursor {
 public:
  explicit JsonCursor(absl::string_view input) : input_(input) {}

  absl::Status Error(absl::string_view message) const {
    return absl::InvalidArgumentError(absl::StrCat(
        "invalid Qwen tokenizer.json at byte ", position_, ": ", message));
  }
  char Peek() {
    while (position_ < input_.size() &&
           (input_[position_] == ' ' || input_[position_] == '\t' ||
            input_[position_] == '\r' || input_[position_] == '\n'))
      ++position_;
    return position_ == input_.size() ? '\0' : input_[position_];
  }
  bool Consume(char c) {
    if (Peek() != c)
      return false;
    ++position_;
    return true;
  }
  bool AtEnd() {
    Peek();
    return position_ == input_.size();
  }
  absl::Status Expect(char c) {
    return Consume(c) ? absl::OkStatus() : Error("unexpected JSON delimiter");
  }
  absl::Status Literal(absl::string_view value) {
    Peek();
    if (input_.substr(position_, value.size()) != value)
      return Error(absl::StrCat("expected ", value));
    position_ += value.size();
    return absl::OkStatus();
  }
  absl::StatusOr<int> Integer() {
    Peek();
    const size_t begin = position_;
    while (position_ < input_.size() && input_[position_] >= '0' &&
           input_[position_] <= '9')
      ++position_;
    int value;
    if (position_ == begin || (position_ > begin + 1 && input_[begin] == '0') ||
        !absl::SimpleAtoi(input_.substr(begin, position_ - begin), &value))
      return Error("expected nonnegative integer");
    return value;
  }
  absl::StatusOr<std::string> String() {
    RETURN_IF_ERROR(Expect('"'));
    std::string value;
    while (position_ < input_.size()) {
      const unsigned char byte = input_[position_++];
      if (byte == '"')
        return value;
      if (byte < 0x20)
        return Error("unescaped control character");
      if (byte != '\\') {
        value.push_back(byte);
        continue;
      }
      if (position_ == input_.size())
        return Error("incomplete string escape");
      switch (input_[position_++]) {
        case '"':
          value.push_back('"');
          break;
        case '\\':
          value.push_back('\\');
          break;
        case '/':
          value.push_back('/');
          break;
        case 'b':
          value.push_back('\b');
          break;
        case 'f':
          value.push_back('\f');
          break;
        case 'n':
          value.push_back('\n');
          break;
        case 'r':
          value.push_back('\r');
          break;
        case 't':
          value.push_back('\t');
          break;
        case 'u': {
          ASSIGN_OR_RETURN(uint32_t cp, HexQuad());
          if (cp >= 0xd800 && cp <= 0xdbff) {
            if (input_.substr(position_, 2) != "\\u")
              return Error("high surrogate without a low surrogate");
            position_ += 2;
            ASSIGN_OR_RETURN(uint32_t low, HexQuad());
            if (low < 0xdc00 || low > 0xdfff)
              return Error("invalid low surrogate");
            cp = 0x10000 + ((cp - 0xd800) << 10) + low - 0xdc00;
          } else if (cp >= 0xdc00 && cp <= 0xdfff) {
            return Error("unpaired low surrogate");
          }
          utf8proc_uint8_t bytes[4];
          const int count = utf8proc_encode_char(cp, bytes);
          value.append(reinterpret_cast<const char*>(bytes), count);
          break;
        }
        default:
          return Error("invalid string escape");
      }
    }
    return Error("unterminated string");
  }
  template <typename Visitor>
  absl::Status Object(Visitor visitor) {
    RETURN_IF_ERROR(Expect('{'));
    if (Consume('}'))
      return absl::OkStatus();
    do {
      ASSIGN_OR_RETURN(auto key, String());
      RETURN_IF_ERROR(Expect(':'));
      RETURN_IF_ERROR(visitor(key));
      if (Consume('}'))
        return absl::OkStatus();
      RETURN_IF_ERROR(Expect(','));
    } while (true);
  }
  template <typename Visitor>
  absl::Status Array(Visitor visitor) {
    RETURN_IF_ERROR(Expect('['));
    if (Consume(']'))
      return absl::OkStatus();
    do {
      RETURN_IF_ERROR(visitor());
      if (Consume(']'))
        return absl::OkStatus();
      RETURN_IF_ERROR(Expect(','));
    } while (true);
  }
  absl::Status Skip(int depth = 0) {
    if (depth > 64)
      return Error("JSON nesting exceeds 64 levels");
    switch (Peek()) {
      case '{':
        return Object([&](absl::string_view) { return Skip(depth + 1); });
      case '[':
        return Array([&] { return Skip(depth + 1); });
      case '"': {
        auto value = String();
        return value.status();
      }
      case 't':
        return Literal("true");
      case 'f':
        return Literal("false");
      case 'n':
        return Literal("null");
      default: {
        const auto digit = [&] {
          return position_ < input_.size() && input_[position_] >= '0' &&
                 input_[position_] <= '9';
        };
        if (position_ < input_.size() && input_[position_] == '-')
          ++position_;
        if (!digit())
          return Error("expected JSON value");
        if (input_[position_++] != '0')
          while (digit())
            ++position_;
        if (position_ < input_.size() && input_[position_] == '.') {
          ++position_;
          if (!digit())
            return Error("missing fractional digits");
          while (digit())
            ++position_;
        }
        if (position_ < input_.size() &&
            (input_[position_] == 'e' || input_[position_] == 'E')) {
          ++position_;
          if (position_ < input_.size() &&
              (input_[position_] == '+' || input_[position_] == '-'))
            ++position_;
          if (!digit())
            return Error("missing exponent digits");
          while (digit())
            ++position_;
        }
        return absl::OkStatus();
      }
    }
  }
  absl::Status StringEquals(absl::string_view expected) {
    ASSIGN_OR_RETURN(auto value, String());
    return value == expected ? absl::OkStatus()
                             : Error(absl::StrCat("expected ", expected));
  }

 private:
  absl::StatusOr<uint32_t> HexQuad() {
    if (input_.size() - position_ < 4)
      return Error("incomplete Unicode escape");
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
        return Error("invalid Unicode escape");
    }
    return value;
  }
  absl::string_view input_;
  size_t position_ = 0;
};

struct AddedToken {
  std::string text;
  int id = -1;
};

absl::Status ReadAddedTokens(JsonCursor* json, std::vector<AddedToken>* added) {
  return json->Array([&]() -> absl::Status {
    AddedToken token;
    int flags = 0;
    RETURN_IF_ERROR(json->Object([&](absl::string_view key) -> absl::Status {
      if (key == "content") {
      ASSIGN_OR_RETURN(token.text, json->String());
      } else if (key == "id") {
      ASSIGN_OR_RETURN(token.id, json->Integer());
      } else if (key == "normalized" || key == "lstrip" || key == "rstrip" ||
                 key == "single_word") {
        RETURN_IF_ERROR(json->Literal("false"));
        flags |= key == "normalized" ? 1
                 : key == "lstrip"   ? 2
                 : key == "rstrip"   ? 4
                                     : 8;
      } else {
        RETURN_IF_ERROR(json->Skip());
      }
      return absl::OkStatus();
    }));
    if (token.id < 0 || token.text.empty() || flags != 15)
      return json->Error("added token is missing content, id, or match flags");
    added->push_back(std::move(token));
    return absl::OkStatus();
  });
}

absl::Status ReadModel(JsonCursor* json,
                       absl::flat_hash_map<std::string, int>* vocabulary,
                       absl::flat_hash_map<std::string, int>* ranks) {
  bool found_type = false, found_vocabulary = false, found_merges = false;
  RETURN_IF_ERROR(json->Object([&](absl::string_view key) -> absl::Status {
    if (key == "type") {
      RETURN_IF_ERROR(json->StringEquals("BPE"));
      found_type = true;
    } else if (key == "vocab") {
      found_vocabulary = true;
      RETURN_IF_ERROR(json->Object([&](absl::string_view text) -> absl::Status {
      ASSIGN_OR_RETURN(int id, json->Integer());
        if (!vocabulary->emplace(text, id).second)
          return json->Error("duplicate vocabulary entry");
        return absl::OkStatus();
      }));
    } else if (key == "merges") {
      found_merges = true;
      int rank = 0;
      RETURN_IF_ERROR(json->Array([&]() -> absl::Status {
        std::string left, right;
        if (json->Consume('[')) {
      ASSIGN_OR_RETURN(left, json->String());
          RETURN_IF_ERROR(json->Expect(','));
      ASSIGN_OR_RETURN(right, json->String());
          RETURN_IF_ERROR(json->Expect(']'));
        } else {
      ASSIGN_OR_RETURN(auto pair, json->String());
          const auto separator = pair.find(' ');
          if (separator == std::string::npos)
            return json->Error("invalid BPE merge");
          left = pair.substr(0, separator);
          right = pair.substr(separator + 1);
        }
        if (left.empty() || right.empty() ||
            !ranks->emplace(internal::MergeKey(left, right), rank++).second)
          return json->Error("empty or duplicated BPE merge");
        return absl::OkStatus();
      }));
    } else if (key == "dropout" || key == "unk_token") {
      RETURN_IF_ERROR(json->Literal("null"));
    } else if (key == "continuing_subword_prefix" ||
               key == "end_of_word_suffix") {
      RETURN_IF_ERROR(json->StringEquals(""));
    } else if (key == "byte_fallback" || key == "ignore_merges" ||
               key == "fuse_unk") {
      RETURN_IF_ERROR(json->Literal("false"));
    } else {
      RETURN_IF_ERROR(json->Skip());
    }
    return absl::OkStatus();
  }));
  return found_type && found_vocabulary && found_merges
             ? absl::OkStatus()
             : json->Error("missing BPE type, vocabulary, or merges");
}

absl::Status ReadByteLevel(JsonCursor* json) {
  bool found_type = false;
  int flags = 0;
  RETURN_IF_ERROR(json->Object([&](absl::string_view key) -> absl::Status {
    if (key == "type") {
      RETURN_IF_ERROR(json->StringEquals("ByteLevel"));
      found_type = true;
    } else if (key == "add_prefix_space" || key == "use_regex" ||
               key == "trim_offsets") {
      RETURN_IF_ERROR(json->Literal("false"));
      flags |= key == "add_prefix_space" ? 1 : key == "use_regex" ? 2 : 4;
    } else {
      RETURN_IF_ERROR(json->Skip());
    }
    return absl::OkStatus();
  }));
  return found_type && flags == 7
             ? absl::OkStatus()
             : json->Error("missing ByteLevel configuration");
}

absl::Status ReadPreTokenizer(JsonCursor* json) {
  int stages = 0;
  bool found_pattern = false, found_type = false;
  bool found_split = false, found_behavior = false, found_invert = false;
  RETURN_IF_ERROR(json->Object([&](absl::string_view key) -> absl::Status {
    if (key == "type") {
      RETURN_IF_ERROR(json->StringEquals("Sequence"));
      found_type = true;
    } else if (key == "pretokenizers") {
      RETURN_IF_ERROR(json->Array([&]() -> absl::Status {
        if (++stages == 2)
          return ReadByteLevel(json);
        if (stages != 1)
          return json->Error("unexpected pre-tokenizer stage");
        return json->Object([&](absl::string_view field) -> absl::Status {
          if (field == "type") {
            RETURN_IF_ERROR(json->StringEquals("Split"));
            found_split = true;
            return absl::OkStatus();
          }
          if (field == "behavior") {
            RETURN_IF_ERROR(json->StringEquals("Isolated"));
            found_behavior = true;
            return absl::OkStatus();
          }
          if (field == "invert") {
            RETURN_IF_ERROR(json->Literal("false"));
            found_invert = true;
            return absl::OkStatus();
          }
          if (field == "pattern")
            return json->Object([&](absl::string_view kind) -> absl::Status {
              if (kind != "Regex")
                return json->Error("expected a regular expression");
              RETURN_IF_ERROR(json->StringEquals(kPreTokenizeRegex));
              found_pattern = true;
              return absl::OkStatus();
            });
          return json->Skip();
        });
      }));
    } else {
      RETURN_IF_ERROR(json->Skip());
    }
    return absl::OkStatus();
  }));
  return found_type && found_pattern && found_split && found_behavior &&
                 found_invert && stages == 2
             ? absl::OkStatus()
             : json->Error("missing Qwen3.8 pre-tokenizer configuration");
}

std::array<std::string, 256> MakeByteEncoder() {
  std::array<std::string, 256> encoder;
  uint32_t replacement = 256;
  for (int byte = 0; byte < 256; ++byte) {
    const bool direct = (byte >= 33 && byte <= 126) ||
                        (byte >= 161 && byte <= 172) || byte >= 174;
    utf8proc_uint8_t encoded[4];
    const int size =
        utf8proc_encode_char(direct ? byte : replacement++, encoded);
    encoder[byte].assign(reinterpret_cast<char*>(encoded), size);
  }
  return encoder;
}

absl::StatusOr<std::string> NormalizeNfc(absl::string_view text) {
  if (text.empty())
    return std::string();
  utf8proc_uint8_t* normalized = nullptr;
  const auto size = utf8proc_map(
      reinterpret_cast<const utf8proc_uint8_t*>(text.data()), text.size(),
      &normalized,
      static_cast<utf8proc_option_t>(UTF8PROC_STABLE | UTF8PROC_COMPOSE));
  if (size < 0)
    return absl::InvalidArgumentError(
        absl::StrCat("Qwen NFC normalization: ", utf8proc_errmsg(size)));
  std::string result(reinterpret_cast<const char*>(normalized), size);
  std::free(normalized);
  return result;
}

// RE2 excludes lookahead and has ASCII-only \s. Spell Unicode White_Space
// explicitly, and implement the final \s+(?!\S) branch below. All preceding
// alternatives retain their exact order from the checkpoint's expression.
const RE2& TokenPattern() {
  static const auto* pattern = new RE2(
      R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\p{Z}\x{0009}-\x{000D}\x{0085}\p{L}\p{M}\p{N}]+[\r\n]*|[\p{Z}\x{0009}-\x{000D}\x{0085}]*[\r\n]+)");
  return *pattern;
}
const RE2& WhitespacePattern() {
  static const auto* pattern = new RE2(R"([\p{Z}\x{0009}-\x{000D}\x{0085}]+)");
  return *pattern;
}

absl::StatusOr<std::vector<absl::string_view>> PreTokenize(
    absl::string_view text) {
  std::vector<absl::string_view> pieces;
  while (!text.empty()) {
    absl::string_view remaining = text;
    if (!RE2::Consume(&remaining, TokenPattern())) {
      if (!RE2::Consume(&remaining, WhitespacePattern()))
        return absl::InternalError("Qwen pre-tokenizer did not consume UTF-8");
      size_t consumed = text.size() - remaining.size();
      if (!remaining.empty()) {
        size_t final_scalar = consumed - 1;
        while ((static_cast<unsigned char>(text[final_scalar]) & 0xc0) == 0x80)
          --final_scalar;
        // A following nonspace makes the negative lookahead backtrack by one
        // complete Unicode scalar. A single whitespace uses the final \s+.
        if (final_scalar > 0)
          consumed = final_scalar;
      }
      remaining = text.substr(consumed);
    }
    pieces.push_back(text.substr(0, text.size() - remaining.size()));
    text = remaining;
  }
  return pieces;
}

bool IsPythonWhitespace(uint32_t cp) {
  return (cp >= 0x09 && cp <= 0x0d) || (cp >= 0x1c && cp <= 0x20) ||
         cp == 0x85 || cp == 0xa0 || cp == 0x1680 ||
         (cp >= 0x2000 && cp <= 0x200a) || cp == 0x2028 || cp == 0x2029 ||
         cp == 0x202f || cp == 0x205f || cp == 0x3000;
}

absl::string_view TrimChatContent(absl::string_view text) {
  absl::string_view remaining = text;
  size_t begin = text.size(), end = 0;
  while (!remaining.empty()) {
    const size_t offset = text.size() - remaining.size();
    auto cp = internal::ConsumeUtf8(&remaining);
    // Encode reports malformed UTF-8. Preserve it here rather than silently
    // changing malformed user input while formatting the chat template.
    if (!cp.ok())
      return text;
    if (!IsPythonWhitespace(*cp)) {
      begin = std::min(begin, offset);
      end = text.size() - remaining.size();
    }
  }
  return end == 0 ? absl::string_view() : text.substr(begin, end - begin);
}

}  // namespace

struct QwenTokenizer::Impl {
  absl::flat_hash_map<std::string, int> vocabulary;
  absl::flat_hash_map<std::string, int> ranks;
  std::vector<AddedToken> added;
  std::vector<std::string> decoded;
  std::array<std::string, 256> byte_encoder = MakeByteEncoder();
  int eos = -1;
  mutable absl::Mutex cache_mutex;
  mutable absl::flat_hash_map<std::string, std::vector<int>> cache
      ABSL_GUARDED_BY(cache_mutex);

  absl::StatusOr<std::vector<int>> Bpe(absl::string_view piece) const {
    {
      absl::MutexLock lock(cache_mutex);
      const auto found = cache.find(piece);
      if (found != cache.end())
        return found->second;
    }
    struct Symbol {
      std::string text;
      int previous, next;
      size_t version = 0;
    };
    struct Merge {
      int rank, left, right;
      size_t left_version, right_version;
      bool operator<(const Merge& other) const {
        return rank != other.rank ? rank > other.rank : left > other.left;
      }
    };
    std::vector<Symbol> symbols;
    symbols.reserve(piece.size());
    for (unsigned char byte : piece) {
      const int index = static_cast<int>(symbols.size());
      symbols.push_back({byte_encoder[byte], index - 1, index + 1});
    }
    if (symbols.empty())
      return std::vector<int>();
    symbols.back().next = -1;
    std::priority_queue<Merge> merges;
    const auto queue_pair = [&](int left) {
      if (left < 0 || symbols[left].next < 0)
        return;
      const int right = symbols[left].next;
      const auto found = ranks.find(
          internal::MergeKey(symbols[left].text, symbols[right].text));
      if (found != ranks.end())
        merges.push({found->second, left, right, symbols[left].version,
                     symbols[right].version});
    };
    for (int i = 0; i + 1 < static_cast<int>(symbols.size()); ++i)
      queue_pair(i);
    while (!merges.empty()) {
      const Merge merge = merges.top();
      merges.pop();
      Symbol& left = symbols[merge.left];
      Symbol& right = symbols[merge.right];
      if (left.next != merge.right || left.version != merge.left_version ||
          right.version != merge.right_version)
        continue;
      left.text.append(right.text);
      left.next = right.next;
      ++left.version;
      right.next = -1;
      ++right.version;
      if (left.next >= 0)
        symbols[left.next].previous = merge.left;
      queue_pair(left.previous);
      queue_pair(merge.left);
    }
    std::vector<int> ids;
    for (int index = 0; index >= 0; index = symbols[index].next) {
      const auto found = vocabulary.find(symbols[index].text);
      if (found == vocabulary.end())
        return absl::DataLossError("Qwen BPE result is absent from vocabulary");
      ids.push_back(found->second);
    }
    {
      absl::MutexLock lock(cache_mutex);
      if (cache.size() < kMaximumCacheEntries && piece.size() <= 4096)
        cache.emplace(piece, ids);
    }
    return ids;
  }
  absl::Status EncodeOrdinary(absl::string_view text,
                              std::vector<int>* output) const {
    ASSIGN_OR_RETURN(auto normalized, NormalizeNfc(text));
    ASSIGN_OR_RETURN(auto pieces, PreTokenize(normalized));
    for (absl::string_view piece : pieces) {
      ASSIGN_OR_RETURN(auto ids, Bpe(piece));
      output->insert(output->end(), ids.begin(), ids.end());
    }
    return absl::OkStatus();
  }
};

QwenTokenizer::QwenTokenizer(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
QwenTokenizer::~QwenTokenizer() = default;

absl::StatusOr<std::unique_ptr<QwenTokenizer>> QwenTokenizer::Load(
    const std::filesystem::path& directory) {
  const auto path = directory / "tokenizer.json";
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream)
    return absl::NotFoundError(absl::StrCat("cannot open ", path.string()));
  const std::streamoff size = stream.tellg();
  if (size <= 0 || static_cast<uint64_t>(size) > kMaximumTokenizerBytes)
    return absl::InvalidArgumentError("Qwen tokenizer.json size is invalid");
  std::string contents(size, '\0');
  stream.seekg(0);
  if (!stream.read(contents.data(), size))
    return absl::DataLossError("cannot read Qwen tokenizer.json");

  auto model = absl::make_unique<Impl>();
  JsonCursor json(contents);
  bool found_normalizer = false, found_pre_tokenizer = false;
  bool found_decoder = false;
  RETURN_IF_ERROR(json.Object([&](absl::string_view key) -> absl::Status {
    if (key == "model")
      return ReadModel(&json, &model->vocabulary, &model->ranks);
    if (key == "added_tokens")
      return ReadAddedTokens(&json, &model->added);
    if (key == "normalizer") {
      return json.Object([&](absl::string_view field) -> absl::Status {
        if (field == "type") {
          RETURN_IF_ERROR(json.StringEquals("NFC"));
          found_normalizer = true;
          return absl::OkStatus();
        }
        return json.Skip();
      });
    }
    if (key == "pre_tokenizer") {
      RETURN_IF_ERROR(ReadPreTokenizer(&json));
      found_pre_tokenizer = true;
      return absl::OkStatus();
    }
    if (key == "decoder") {
      RETURN_IF_ERROR(ReadByteLevel(&json));
      found_decoder = true;
      return absl::OkStatus();
    }
    if (key == "post_processor")
      return ReadByteLevel(&json);
    if (key == "padding" || key == "truncation")
      return json.Literal("null");
    return json.Skip();
  }));
  if (!json.AtEnd() || !found_normalizer || !found_pre_tokenizer ||
      !found_decoder || model->vocabulary.empty())
    return json.Error("incomplete Qwen3.8 tokenizer configuration");

  const size_t count = model->vocabulary.size() + model->added.size();
  model->decoded.resize(count);
  std::vector<bool> seen(count);
  std::array<int, 512> byte_decoder;
  byte_decoder.fill(-1);
  for (int byte = 0; byte < 256; ++byte) {
    absl::string_view encoded = model->byte_encoder[byte];
    ASSIGN_OR_RETURN(auto cp, internal::ConsumeUtf8(&encoded));
    byte_decoder[cp] = byte;
    if (!model->vocabulary.contains(model->byte_encoder[byte]))
      return json.Error("vocabulary is missing a byte token");
  }
  for (const auto& [token, id] : model->vocabulary) {
    if (id < 0 || static_cast<size_t>(id) >= count || seen[id])
      return json.Error("invalid or duplicate vocabulary id");
    seen[id] = true;
    absl::string_view encoded = token;
    while (!encoded.empty()) {
      ASSIGN_OR_RETURN(auto cp, internal::ConsumeUtf8(&encoded));
      if (cp >= byte_decoder.size() || byte_decoder[cp] < 0)
        return json.Error("vocabulary contains a non-byte scalar");
      model->decoded[id].push_back(static_cast<char>(byte_decoder[cp]));
    }
  }
  absl::flat_hash_map<std::string, int> added_spellings;
  for (const AddedToken& token : model->added) {
    if (static_cast<size_t>(token.id) >= count || seen[token.id] ||
        !added_spellings.emplace(token.text, token.id).second)
      return json.Error("invalid or duplicate added token");
    seen[token.id] = true;
    model->decoded[token.id] = token.text;
    if (token.text == "<|im_end|>")
      model->eos = token.id;
  }
  if (model->eos < 0 ||
      std::find(seen.begin(), seen.end(), false) != seen.end())
    return json.Error("missing EOS token or non-contiguous vocabulary");
  if (!TokenPattern().ok() || !WhitespacePattern().ok())
    return absl::InternalError("Qwen regular expression failed to compile");
  return absl::WrapUnique(new QwenTokenizer(std::move(model)));
}

absl::StatusOr<std::vector<int>> QwenTokenizer::Encode(
    absl::string_view text) const {
  std::vector<int> ids;
  while (!text.empty()) {
    size_t first = text.size();
    const AddedToken* match = nullptr;
    for (const AddedToken& token : impl_->added) {
      const size_t position = text.find(token.text);
      if (position < first || (position == first && match &&
                               token.text.size() > match->text.size())) {
        first = position;
        match = &token;
      }
    }
    RETURN_IF_ERROR(impl_->EncodeOrdinary(text.substr(0, first), &ids));
    if (!match)
      break;
    ids.push_back(match->id);
    text.remove_prefix(first + match->text.size());
  }
  return ids;
}

absl::StatusOr<cuda::PageLockedHostArray<int>> QwenTokenizer::Encode(
    cuda::Executor& executor, absl::string_view text) const {
  ASSIGN_OR_RETURN(auto ids, Encode(text));
  return cuda::PageLockedHostArray<int>::CopyFrom(executor, ids);
}

absl::StatusOr<std::string> QwenTokenizer::Decode(
    absl::Span<const int> token_ids) const {
  std::string text;
  for (int id : token_ids) {
    if (id < 0 || static_cast<size_t>(id) >= impl_->decoded.size())
      return absl::InvalidArgumentError("Qwen token id is outside vocabulary");
    text.append(impl_->decoded[id]);
  }
  return text;
}
int QwenTokenizer::vocab_size() const { return impl_->decoded.size(); }
int QwenTokenizer::eos_token_id() const { return impl_->eos; }

std::string QwenTokenizer::ChatPrompt(absl::string_view text,
                                      bool enable_thinking) {
  std::string prompt;
  if (enable_thinking) {
    prompt =
        "<|im_start|>system\nReasoning effort is set to xhigh. Please think "
        "carefully through the task, validate key assumptions, consider "
        "plausible "
        "alternatives, and prioritize correctness, consistency, and clarity in "
        "the final answer.<|im_end|>\n";
  }
  absl::StrAppend(&prompt, "<|im_start|>user\n", TrimChatContent(text),
                  "<|im_end|>\n<|im_start|>assistant\n<think>\n");
  if (!enable_thinking)
    prompt.append("\n</think>\n\n");
  return prompt;
}

}  // namespace pluto::tokenizer
