#include "src/llm/qwen/checkpoint.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "src/util/status_macros.h"

namespace pluto::llm::qwen {
namespace {

constexpr size_t kMaxJsonBytes = 64 * 1024 * 1024;
constexpr size_t kMaxJsonNodes = 1000000;

absl::Status Bad(absl::string_view message) {
  return absl::InvalidArgumentError(message);
}

// Exact integer spelling is retained so file offsets never pass through a
// floating-point value. Limits bound recursion and allocation for bad files.
struct Json {
  enum class Type { kNull, kBool, kNumber, kString, kArray, kObject };
  Type type = Type::kNull;
  std::string scalar;
  std::vector<Json> array;
  std::map<std::string, Json, std::less<>> object;

  const Json* Find(absl::string_view key) const {
    auto it = object.find(std::string(key));
    return it == object.end() ? nullptr : &it->second;
  }
};

void AppendUtf8(uint32_t code, std::string* output) {
  if (code <= 0x7f) {
    output->push_back(static_cast<char>(code));
  } else if (code <= 0x7ff) {
    output->push_back(static_cast<char>(0xc0 | (code >> 6)));
    output->push_back(static_cast<char>(0x80 | (code & 0x3f)));
  } else if (code <= 0xffff) {
    output->push_back(static_cast<char>(0xe0 | (code >> 12)));
    output->push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3f)));
    output->push_back(static_cast<char>(0x80 | (code & 0x3f)));
  } else {
    output->push_back(static_cast<char>(0xf0 | (code >> 18)));
    output->push_back(static_cast<char>(0x80 | ((code >> 12) & 0x3f)));
    output->push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3f)));
    output->push_back(static_cast<char>(0x80 | (code & 0x3f)));
  }
}

class JsonParser {
 public:
  explicit JsonParser(absl::string_view input) : input_(input) {}

  absl::StatusOr<Json> Parse() {
    if (input_.size() > kMaxJsonBytes)
      return Bad("JSON exceeds the 64 MiB metadata limit");
    ASSIGN_OR_RETURN(Json value, Value(0));
    Space();
    if (position_ != input_.size())
      return Error("trailing content");
    return value;
  }

 private:
  absl::Status Error(absl::string_view detail) const {
    return Bad(absl::StrCat("invalid JSON at byte ", position_, ": ", detail));
  }

  void Space() {
    while (position_ < input_.size() &&
           (input_[position_] == ' ' || input_[position_] == '\n' ||
            input_[position_] == '\r' || input_[position_] == '\t'))
      ++position_;
  }

  bool Consume(char c) {
    Space();
    if (position_ >= input_.size() || input_[position_] != c)
      return false;
    ++position_;
    return true;
  }

  absl::StatusOr<uint32_t> Hex() {
    if (input_.size() - position_ < 4)
      return Error("truncated Unicode escape");
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

  absl::StatusOr<std::string> String() {
    if (!Consume('"'))
      return Error("expected string");
    std::string result;
    while (position_ < input_.size()) {
      const uint8_t c = input_[position_++];
      if (c == '"')
        return result;
      if (c < 0x20)
        return Error("control byte in string");
      if (c >= 0x80) {
        const int extra = c >= 0xc2 && c <= 0xdf   ? 1
                          : c >= 0xe0 && c <= 0xef ? 2
                          : c >= 0xf0 && c <= 0xf4 ? 3
                                                   : -1;
        if (extra < 0 || input_.size() - position_ < size_t(extra))
          return Error("invalid UTF-8");
        uint32_t code = c & ((1u << (6 - extra)) - 1);
        for (int i = 0; i < extra; ++i) {
          const uint8_t next = input_[position_++];
          if ((next & 0xc0) != 0x80)
            return Error("invalid UTF-8");
          code = (code << 6) | (next & 0x3f);
        }
        if (code < (extra == 1   ? 0x80u
                    : extra == 2 ? 0x800u
                                 : 0x10000u) ||
            code > 0x10ffff || (code >= 0xd800 && code <= 0xdfff))
          return Error("invalid UTF-8 scalar");
        AppendUtf8(code, &result);
        continue;
      }
      if (c != '\\') {
        result.push_back(static_cast<char>(c));
        continue;
      }
      if (position_ == input_.size())
        return Error("truncated escape");
      switch (input_[position_++]) {
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
          ASSIGN_OR_RETURN(uint32_t code, Hex());
          if (code >= 0xd800 && code <= 0xdbff) {
            if (input_.size() - position_ < 2 || input_[position_] != '\\' ||
                input_[position_ + 1] != 'u')
              return Error("missing low surrogate");
            position_ += 2;
            ASSIGN_OR_RETURN(uint32_t low, Hex());
            if (low < 0xdc00 || low > 0xdfff)
              return Error("invalid low surrogate");
            code = 0x10000 + ((code - 0xd800) << 10) + low - 0xdc00;
          } else if (code >= 0xdc00 && code <= 0xdfff) {
            return Error("unexpected low surrogate");
          }
          AppendUtf8(code, &result);
          break;
        }
        default:
          return Error("invalid escape");
      }
    }
    return Error("unterminated string");
  }

  bool Digit() const {
    return position_ < input_.size() && input_[position_] >= '0' &&
           input_[position_] <= '9';
  }

  absl::StatusOr<Json> Value(int depth) {
    if (depth > 64 || ++nodes_ > kMaxJsonNodes)
      return Error("JSON nesting or value count exceeds limit");
    Space();
    if (position_ == input_.size())
      return Error("expected value");
    Json result;
    const char c = input_[position_];
    if (c == '{') {
      result.type = Json::Type::kObject;
      ++position_;
      if (Consume('}'))
        return result;
      while (true) {
        ASSIGN_OR_RETURN(auto key, String());
        if (!Consume(':'))
          return Error("expected colon");
        ASSIGN_OR_RETURN(auto value, Value(depth + 1));
        if (!result.object.emplace(std::move(key), std::move(value)).second)
          return Error("duplicate object key");
        if (Consume('}'))
          return result;
        if (!Consume(','))
          return Error("expected comma");
      }
    }
    if (c == '[') {
      result.type = Json::Type::kArray;
      ++position_;
      if (Consume(']'))
        return result;
      while (true) {
        ASSIGN_OR_RETURN(auto value, Value(depth + 1));
        result.array.push_back(std::move(value));
        if (Consume(']'))
          return result;
        if (!Consume(','))
          return Error("expected comma");
      }
    }
    if (c == '"') {
      result.type = Json::Type::kString;
      ASSIGN_OR_RETURN(result.scalar, String());
      return result;
    }
    for (absl::string_view literal : {"true", "false", "null"}) {
      if (input_.substr(position_, literal.size()) == literal) {
        position_ += literal.size();
        result.type = literal == "null" ? Json::Type::kNull : Json::Type::kBool;
        result.scalar = std::string(literal);
        return result;
      }
    }
    result.type = Json::Type::kNumber;
    const size_t start = position_;
    if (c == '-')
      ++position_;
    if (!Digit())
      return Error("expected number");
    if (input_[position_] == '0')
      ++position_;
    else
      while (Digit())
        ++position_;
    if (position_ < input_.size() && input_[position_] == '.') {
      ++position_;
      if (!Digit())
        return Error("expected fractional digits");
      while (Digit())
        ++position_;
    }
    if (position_ < input_.size() &&
        (input_[position_] == 'e' || input_[position_] == 'E')) {
      ++position_;
      if (position_ < input_.size() &&
          (input_[position_] == '+' || input_[position_] == '-'))
        ++position_;
      if (!Digit())
        return Error("expected exponent digits");
      while (Digit())
        ++position_;
    }
    result.scalar = std::string(input_.substr(start, position_ - start));
    return result;
  }

  absl::string_view input_;
  size_t position_ = 0;
  size_t nodes_ = 0;
};

absl::StatusOr<const Json*> Field(const Json& object, absl::string_view name,
                                  Json::Type type) {
  const Json* value = object.Find(name);
  if (object.type != Json::Type::kObject || value == nullptr ||
      value->type != type)
    return Bad(absl::StrCat("missing or invalid field: ", name));
  return value;
}

absl::StatusOr<uint64_t> Unsigned(const Json& value) {
  uint64_t number;
  if (value.type != Json::Type::kNumber || value.scalar.empty() ||
      !std::all_of(value.scalar.begin(), value.scalar.end(),
                   [](char c) { return c >= '0' && c <= '9'; }) ||
      !absl::SimpleAtoi(value.scalar, &number))
    return Bad("expected an unsigned 64-bit integer");
  return number;
}

absl::StatusOr<int> Integer(const Json& object, absl::string_view name,
                            bool allow_zero = false) {
  ASSIGN_OR_RETURN(const Json* value, Field(object, name, Json::Type::kNumber));
  ASSIGN_OR_RETURN(uint64_t number, Unsigned(*value));
  if (number > std::numeric_limits<int>::max() || (!allow_zero && number == 0))
    return Bad(absl::StrCat("invalid positive integer field: ", name));
  return static_cast<int>(number);
}

absl::StatusOr<double> Number(const Json& object, absl::string_view name) {
  ASSIGN_OR_RETURN(const Json* value, Field(object, name, Json::Type::kNumber));
  double number;
  if (!absl::SimpleAtod(value->scalar, &number) || !std::isfinite(number))
    return Bad(absl::StrCat("invalid finite number field: ", name));
  return number;
}

absl::StatusOr<bool> Boolean(const Json& object, absl::string_view name) {
  ASSIGN_OR_RETURN(const Json* value, Field(object, name, Json::Type::kBool));
  return value->scalar == "true";
}

absl::Status RequireString(const Json& object, absl::string_view name,
                           absl::string_view expected) {
  ASSIGN_OR_RETURN(const Json* value, Field(object, name, Json::Type::kString));
  if (value->scalar != expected)
    return Bad(absl::StrCat("unsupported ", name, ": ", value->scalar));
  return absl::OkStatus();
}

absl::Status IoError(absl::string_view operation,
                     const std::filesystem::path& path, int error) {
  const std::string message =
      absl::StrCat(operation, " ", path.string(), ": ", std::strerror(error));
  return error == ENOENT ? absl::NotFoundError(message)
                         : absl::InternalError(message);
}

struct Mapping {
  const uint8_t* data = nullptr;
  size_t size = 0;
  ~Mapping() {
    if (data != nullptr)
      munmap(const_cast<uint8_t*>(data), size);
  }

  static absl::StatusOr<std::unique_ptr<Mapping>> Open(
      const std::filesystem::path& path) {
    if (path.native().find('\0') != std::string::npos)
      return Bad("file path contains a NUL byte");
    const int fd = open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NONBLOCK);
    if (fd < 0)
      return IoError("cannot open", path, errno);
    struct stat info;
    if (fstat(fd, &info) != 0) {
      const int error = errno;
      close(fd);
      return IoError("cannot stat", path, error);
    }
    if (!S_ISREG(info.st_mode) || info.st_size <= 0 ||
        static_cast<uint64_t>(info.st_size) >
            std::numeric_limits<size_t>::max()) {
      close(fd);
      return Bad(
          absl::StrCat("expected nonempty regular file: ", path.string()));
    }
    auto mapping = absl::make_unique<Mapping>();
    mapping->size = static_cast<size_t>(info.st_size);
    void* data = mmap(nullptr, mapping->size, PROT_READ, MAP_PRIVATE, fd, 0);
    const int error = errno;
    close(fd);
    if (data == MAP_FAILED)
      return IoError("cannot mmap", path, error);
    mapping->data = static_cast<const uint8_t*>(data);
    return mapping;
  }
};

absl::StatusOr<Json> ReadJson(const std::filesystem::path& path) {
  ASSIGN_OR_RETURN(auto mapping, Mapping::Open(path));
  return JsonParser(
             absl::string_view(reinterpret_cast<const char*>(mapping->data),
                               mapping->size))
      .Parse();
}

struct Shard {
  std::unique_ptr<Mapping> mapping;
  std::map<std::string, TensorView, std::less<>> tensors;
};

absl::StatusOr<std::unique_ptr<Shard>> OpenShard(
    const std::filesystem::path& path) {
  ASSIGN_OR_RETURN(auto mapping, Mapping::Open(path));
  if (mapping->size < 8)
    return Bad("truncated safetensors header length");
  uint64_t header_size = 0;
  for (int i = 0; i < 8; ++i)
    header_size |= uint64_t(mapping->data[i]) << (i * 8);
  if (header_size < 2 || header_size > kMaxJsonBytes ||
      header_size > mapping->size - 8 || mapping->data[8] != '{')
    return Bad("invalid safetensors header length or prefix");
  ASSIGN_OR_RETURN(auto header,
                   JsonParser(absl::string_view(reinterpret_cast<const char*>(
                                                    mapping->data + 8),
                                                header_size))
                       .Parse());
  if (header.type != Json::Type::kObject)
    return Bad("safetensors header must be an object");
  const size_t data_start = 8 + static_cast<size_t>(header_size);
  const size_t data_size = mapping->size - data_start;
  auto shard = absl::make_unique<Shard>();
  std::vector<std::pair<uint64_t, uint64_t>> ranges;
  for (const auto& [name, entry] : header.object) {
    if (name == "__metadata__") {
      if (entry.type != Json::Type::kObject)
        return Bad("safetensors metadata must be an object");
      for (const auto& [key, value] : entry.object)
        if (value.type != Json::Type::kString)
          return Bad("safetensors metadata values must be strings");
      continue;
    }
    ASSIGN_OR_RETURN(const Json* dtype,
                     Field(entry, "dtype", Json::Type::kString));
    TensorView tensor;
    uint64_t item_size;
    if (dtype->scalar == "BF16") {
      tensor.dtype = TensorDType::kBF16;
      item_size = 2;
    } else if (dtype->scalar == "F32") {
      tensor.dtype = TensorDType::kF32;
      item_size = 4;
    } else if (dtype->scalar == "F8_E4M3") {
      tensor.dtype = TensorDType::kF8E4M3;
      item_size = 1;
    } else {
      return Bad(
          absl::StrCat("unsupported safetensors dtype: ", dtype->scalar));
    }
    ASSIGN_OR_RETURN(const Json* shape,
                     Field(entry, "shape", Json::Type::kArray));
    if (shape->array.size() > 64)
      return Bad("tensor rank exceeds 64");
    bool empty = false;
    for (const Json& dimension : shape->array) {
      ASSIGN_OR_RETURN(uint64_t extent, Unsigned(dimension));
      if (extent > std::numeric_limits<int64_t>::max())
        return Bad("tensor dimension exceeds int64");
      tensor.shape.push_back(static_cast<int64_t>(extent));
      empty |= extent == 0;
    }
    uint64_t bytes = empty ? 0 : item_size;
    for (int64_t dimension : tensor.shape) {
      if (!empty && bytes > std::numeric_limits<uint64_t>::max() / dimension)
        return Bad("tensor byte count overflow");
      bytes *= static_cast<uint64_t>(dimension);
    }
    ASSIGN_OR_RETURN(const Json* offsets,
                     Field(entry, "data_offsets", Json::Type::kArray));
    if (offsets->array.size() != 2)
      return Bad("tensor data_offsets must contain two integers");
    ASSIGN_OR_RETURN(uint64_t start, Unsigned(offsets->array[0]));
    ASSIGN_OR_RETURN(uint64_t end, Unsigned(offsets->array[1]));
    if (start > end || end > data_size || end - start != bytes)
      return Bad(absl::StrCat("invalid tensor byte range: ", name));
    ranges.emplace_back(start, end);
    tensor.bytes = absl::Span<const uint8_t>(mapping->data + data_start + start,
                                             static_cast<size_t>(bytes));
    shard->tensors.emplace(name, std::move(tensor));
  }
  std::sort(ranges.begin(), ranges.end());
  uint64_t next = 0;
  for (const auto& [start, end] : ranges) {
    if (start != next)
      return Bad("safetensors data has a hole or overlap");
    next = end;
  }
  if (next != data_size)
    return Bad("unindexed trailing safetensors data");
  shard->mapping = std::move(mapping);
  return shard;
}

bool ValidShardName(absl::string_view name) {
  return !name.empty() && name != "." && name != ".." &&
         name.find('/') == absl::string_view::npos &&
         name.find('\\') == absl::string_view::npos &&
         name.find('\0') == absl::string_view::npos && name.size() > 12 &&
         name.substr(name.size() - 12) == ".safetensors";
}

absl::StatusOr<Config> ConfigFromJson(const Json& root) {
  RETURN_IF_ERROR(RequireString(root, "model_type", "qwen3_5"));
  ASSIGN_OR_RETURN(const Json* text,
                   Field(root, "text_config", Json::Type::kObject));
  RETURN_IF_ERROR(RequireString(*text, "model_type", "qwen3_5_text"));
  RETURN_IF_ERROR(RequireString(*text, "hidden_act", "silu"));
  RETURN_IF_ERROR(RequireString(*text, "dtype", "bfloat16"));
  Config config;
#define QWEN_INT(field) ASSIGN_OR_RETURN(config.field, Integer(*text, #field))
  QWEN_INT(hidden_size);
  QWEN_INT(intermediate_size);
  QWEN_INT(num_hidden_layers);
  QWEN_INT(num_attention_heads);
  QWEN_INT(num_key_value_heads);
  QWEN_INT(head_dim);
  QWEN_INT(vocab_size);
  QWEN_INT(max_position_embeddings);
  QWEN_INT(linear_conv_kernel_dim);
  QWEN_INT(linear_key_head_dim);
  QWEN_INT(linear_num_key_heads);
  QWEN_INT(linear_num_value_heads);
  QWEN_INT(linear_value_head_dim);
  QWEN_INT(full_attention_interval);
#undef QWEN_INT
  ASSIGN_OR_RETURN(config.bos_token_id, Integer(*text, "bos_token_id", true));
  ASSIGN_OR_RETURN(config.eos_token_id, Integer(*text, "eos_token_id", true));
  ASSIGN_OR_RETURN(config.rms_norm_eps, Number(*text, "rms_norm_eps"));
  ASSIGN_OR_RETURN(config.attn_output_gate, Boolean(*text, "attn_output_gate"));
  ASSIGN_OR_RETURN(config.tie_word_embeddings,
                   Boolean(*text, "tie_word_embeddings"));
  ASSIGN_OR_RETURN(const Json* output_gate,
                   Field(*text, "output_gate_type", Json::Type::kString));
  config.output_gate_type = output_gate->scalar;
  if (config.output_gate_type != "swish")
    return Bad("unsupported attention output gate type");
  if (!config.attn_output_gate || config.tie_word_embeddings)
    return Bad("decoder requires output gates and untied word embeddings");
  ASSIGN_OR_RETURN(bool bias, Boolean(*text, "attention_bias"));
  ASSIGN_OR_RETURN(double dropout, Number(*text, "attention_dropout"));
  if (bias || dropout != 0)
    return Bad(
        "only attention_bias=false and attention_dropout=0 are supported");
  ASSIGN_OR_RETURN(const Json* layers,
                   Field(*text, "layer_types", Json::Type::kArray));
  if (layers->array.size() != static_cast<size_t>(config.num_hidden_layers))
    return Bad("layer_types length differs from num_hidden_layers");
  for (const Json& layer : layers->array) {
    if (layer.type != Json::Type::kString)
      return Bad("layer_types entries must be strings");
    if (layer.scalar == "linear_attention")
      config.layer_types.push_back(LayerType::kLinearAttention);
    else if (layer.scalar == "full_attention")
      config.layer_types.push_back(LayerType::kFullAttention);
    else
      return Bad(absl::StrCat("unsupported layer type: ", layer.scalar));
  }
  ASSIGN_OR_RETURN(const Json* rope,
                   Field(*text, "rope_parameters", Json::Type::kObject));
  RETURN_IF_ERROR(RequireString(*rope, "rope_type", "default"));
  ASSIGN_OR_RETURN(config.rope_theta, Number(*rope, "rope_theta"));
  ASSIGN_OR_RETURN(config.partial_rotary_factor,
                   Number(*rope, "partial_rotary_factor"));
  ASSIGN_OR_RETURN(config.mrope_interleaved,
                   Boolean(*rope, "mrope_interleaved"));
  ASSIGN_OR_RETURN(const Json* sections,
                   Field(*rope, "mrope_section", Json::Type::kArray));
  if (sections->array.size() != config.mrope_section.size())
    return Bad("mrope_section must have three entries");
  int64_t section_sum = 0;
  for (size_t i = 0; i < config.mrope_section.size(); ++i) {
    ASSIGN_OR_RETURN(uint64_t value, Unsigned(sections->array[i]));
    if (value > std::numeric_limits<int>::max())
      return Bad("mrope_section exceeds int32");
    config.mrope_section[i] = static_cast<int>(value);
    section_sum += static_cast<int64_t>(value);
  }
  if (text->Find("partial_rotary_factor") != nullptr) {
    ASSIGN_OR_RETURN(double outer, Number(*text, "partial_rotary_factor"));
    if (outer != config.partial_rotary_factor)
      return Bad("partial_rotary_factor fields disagree");
  }
  ASSIGN_OR_RETURN(const Json* quant,
                   Field(root, "quantization_config", Json::Type::kObject));
  RETURN_IF_ERROR(RequireString(*quant, "quant_method", "fp8"));
  RETURN_IF_ERROR(RequireString(*quant, "fmt", "e4m3"));
  RETURN_IF_ERROR(RequireString(*quant, "activation_scheme", "dynamic"));
  ASSIGN_OR_RETURN(const Json* blocks,
                   Field(*quant, "weight_block_size", Json::Type::kArray));
  if (blocks->array.size() != config.weight_block_size.size())
    return Bad("weight_block_size must have two entries");
  for (size_t i = 0; i < config.weight_block_size.size(); ++i) {
    ASSIGN_OR_RETURN(uint64_t value, Unsigned(blocks->array[i]));
    if (value == 0 || value > std::numeric_limits<int>::max())
      return Bad("weight_block_size must be positive int32");
    config.weight_block_size[i] = static_cast<int>(value);
  }
  if (config.weight_block_size != std::array<int, 2>{128, 128})
    return Bad("only 128x128 FP8 weight blocks are supported");
  if (config.num_attention_heads % config.num_key_value_heads != 0 ||
      config.linear_num_value_heads % config.linear_num_key_heads != 0)
    return Bad("attention query/value heads must divide into key heads");
  if (config.bos_token_id >= config.vocab_size ||
      config.eos_token_id >= config.vocab_size)
    return Bad("special token ID exceeds vocabulary");
  const double rotary_dim = config.head_dim * config.partial_rotary_factor;
  if (config.rms_norm_eps <= 0 || config.rope_theta <= 0 ||
      config.partial_rotary_factor <= 0 || config.partial_rotary_factor > 1 ||
      rotary_dim != std::floor(rotary_dim) ||
      rotary_dim != static_cast<double>(section_sum * 2))
    return Bad("invalid normalization epsilon or rotary dimensions");
  // These dimensions become signed indexing expressions in the decoder.
  const int64_t attention_width =
      int64_t(config.num_attention_heads) * config.head_dim;
  const uint64_t linear_width =
      2 * uint64_t(config.linear_num_key_heads) * config.linear_key_head_dim +
      uint64_t(config.linear_num_value_heads) * config.linear_value_head_dim;
  if (attention_width > std::numeric_limits<int>::max() / 2 ||
      linear_width > std::numeric_limits<int>::max())
    return Bad("attention dimensions exceed supported indexing range");
  return config;
}

}  // namespace

struct SafetensorsCheckpoint::Impl {
  std::filesystem::path directory;
  std::map<std::string, std::string, std::less<>> weight_map;
  std::map<std::string, std::unique_ptr<Shard>, std::less<>> shards;
  std::vector<std::string> names;
};

SafetensorsCheckpoint::SafetensorsCheckpoint(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

SafetensorsCheckpoint::~SafetensorsCheckpoint() = default;

absl::StatusOr<std::unique_ptr<SafetensorsCheckpoint>>
SafetensorsCheckpoint::Open(const std::filesystem::path& directory) {
  auto impl = absl::make_unique<Impl>();
  impl->directory = directory;
  auto index = ReadJson(directory / "model.safetensors.index.json");
  if (index.ok()) {
    ASSIGN_OR_RETURN(const Json* weights,
                     Field(*index, "weight_map", Json::Type::kObject));
    if (weights->object.empty())
      return Bad("checkpoint weight_map is empty");
    for (const auto& [name, file] : weights->object) {
      if (file.type != Json::Type::kString || !ValidShardName(file.scalar))
        return Bad(absl::StrCat("unsafe or invalid shard filename for ", name));
      impl->weight_map.emplace(name, file.scalar);
      impl->names.push_back(name);
    }
  } else if (index.status().code() == absl::StatusCode::kNotFound) {
    ASSIGN_OR_RETURN(auto shard, OpenShard(directory / "model.safetensors"));
    for (const auto& [name, tensor] : shard->tensors) {
      impl->weight_map.emplace(name, "model.safetensors");
      impl->names.push_back(name);
    }
    impl->shards.emplace("model.safetensors", std::move(shard));
  } else {
    return index.status();
  }
  return absl::WrapUnique(new SafetensorsCheckpoint(std::move(impl)));
}

absl::StatusOr<TensorView> SafetensorsCheckpoint::Tensor(
    absl::string_view name) {
  const auto weight = impl_->weight_map.find(std::string(name));
  if (weight == impl_->weight_map.end())
    return absl::NotFoundError(
        absl::StrCat("checkpoint tensor not found: ", name));
  auto shard = impl_->shards.find(weight->second);
  if (shard == impl_->shards.end()) {
    ASSIGN_OR_RETURN(auto opened, OpenShard(impl_->directory / weight->second));
    // Validate the whole shard against the index before exposing any tensor.
    for (const auto& [tensor_name, tensor] : opened->tensors) {
      const auto expected = impl_->weight_map.find(tensor_name);
      if (expected == impl_->weight_map.end() ||
          expected->second != weight->second)
        return Bad(absl::StrCat("shard contains tensor absent from its index: ",
                                tensor_name));
    }
    for (const auto& [tensor_name, filename] : impl_->weight_map)
      if (filename == weight->second &&
          opened->tensors.find(tensor_name) == opened->tensors.end())
        return Bad(
            absl::StrCat("indexed tensor missing from shard: ", tensor_name));
    shard = impl_->shards.emplace(weight->second, std::move(opened)).first;
  }
  const auto tensor = shard->second->tensors.find(weight->first);
  if (tensor == shard->second->tensors.end())
    return Bad(absl::StrCat("indexed tensor missing from shard: ", name));
  return tensor->second;
}

const std::vector<std::string>& SafetensorsCheckpoint::tensor_names() const {
  return impl_->names;
}

absl::StatusOr<Config> ParseConfig(absl::string_view json) {
  ASSIGN_OR_RETURN(auto parsed, JsonParser(json).Parse());
  return ConfigFromJson(parsed);
}

absl::StatusOr<Config> LoadConfig(const std::filesystem::path& directory) {
  ASSIGN_OR_RETURN(auto parsed, ReadJson(directory / "config.json"));
  return ConfigFromJson(parsed);
}

}  // namespace pluto::llm::qwen
