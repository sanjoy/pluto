#include "src/parquet/fineweb_parquet_reader.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "snappy.h"
#include "src/util/status_macros.h"

namespace pluto::parquet {
namespace {

constexpr absl::string_view kParquetMagic = "PAR1";

enum class CompactType : uint8_t {
  kStop = 0,
  kBooleanTrue = 1,
  kBooleanFalse = 2,
  kByte = 3,
  kI16 = 4,
  kI32 = 5,
  kI64 = 6,
  kDouble = 7,
  kBinary = 8,
  kList = 9,
  kSet = 10,
  kMap = 11,
  kStruct = 12,
};

struct FieldHeader {
  int16_t id = 0;
  CompactType type = CompactType::kStop;
};

struct ListHeader {
  uint32_t size = 0;
  CompactType type = CompactType::kStop;
};

// Minimal reader for Thrift's compact protocol. Parquet uses this protocol for
// both the file footer and every page header. Bounds checks live here so the
// format-specific parsers below can stay simple and cannot walk off corrupt
// input.
class CompactReader {
 public:
  explicit CompactReader(absl::Span<const uint8_t> input) : input_(input) {}

  size_t position() const { return position_; }

  absl::StatusOr<uint8_t> ReadByte() {
    if (position_ >= input_.size()) {
      return absl::DataLossError("truncated compact-protocol value");
    }
    return input_[position_++];
  }

  absl::StatusOr<uint64_t> ReadVarint() {
    uint64_t value = 0;
    for (int shift = 0; shift < 64; shift += 7) {
      ASSIGN_OR_RETURN(uint8_t byte, ReadByte());
      if (shift == 63 && (byte & 0xfe) != 0) {
        return absl::DataLossError("compact-protocol varint overflow");
      }
      value |= static_cast<uint64_t>(byte & 0x7f) << shift;
      if ((byte & 0x80) == 0) return value;
    }
    return absl::DataLossError("unterminated compact-protocol varint");
  }

  absl::StatusOr<int64_t> ReadSigned() {
    ASSIGN_OR_RETURN(uint64_t encoded, ReadVarint());
    return static_cast<int64_t>(encoded >> 1) ^
           -static_cast<int64_t>(encoded & 1);
  }

  absl::StatusOr<int32_t> ReadI32() {
    ASSIGN_OR_RETURN(int64_t value, ReadSigned());
    if (value < std::numeric_limits<int32_t>::min() ||
        value > std::numeric_limits<int32_t>::max()) {
      return absl::DataLossError("compact-protocol i32 overflow");
    }
    return static_cast<int32_t>(value);
  }

  absl::StatusOr<int64_t> ReadI64() { return ReadSigned(); }

  absl::StatusOr<absl::string_view> ReadBinary() {
    ASSIGN_OR_RETURN(uint64_t length, ReadVarint());
    if (length > input_.size() - position_) {
      return absl::DataLossError("truncated compact-protocol binary");
    }
    const char* data = reinterpret_cast<const char*>(input_.data() + position_);
    position_ += static_cast<size_t>(length);
    return absl::string_view(data, static_cast<size_t>(length));
  }

  absl::StatusOr<FieldHeader> ReadFieldHeader(int16_t* previous_id) {
    ASSIGN_OR_RETURN(uint8_t encoded, ReadByte());
    const auto type = static_cast<CompactType>(encoded & 0x0f);
    if (type == CompactType::kStop) return FieldHeader{0, type};

    int16_t id = 0;
    const int16_t delta = encoded >> 4;
    if (delta == 0) {
      ASSIGN_OR_RETURN(int64_t explicit_id, ReadSigned());
      if (explicit_id < std::numeric_limits<int16_t>::min() ||
          explicit_id > std::numeric_limits<int16_t>::max()) {
        return absl::DataLossError("compact-protocol field id overflow");
      }
      id = static_cast<int16_t>(explicit_id);
    } else {
      id = *previous_id + delta;
    }
    *previous_id = id;
    return FieldHeader{id, type};
  }

  absl::StatusOr<ListHeader> ReadListHeader() {
    ASSIGN_OR_RETURN(uint8_t encoded, ReadByte());
    uint64_t size = encoded >> 4;
    if (size == 15) {
      ASSIGN_OR_RETURN(size, ReadVarint());
    }
    if (size > std::numeric_limits<uint32_t>::max()) {
      return absl::ResourceExhaustedError("compact-protocol list is too large");
    }
    return ListHeader{static_cast<uint32_t>(size),
                      static_cast<CompactType>(encoded & 0x0f)};
  }

  absl::Status Skip(CompactType type, bool collection_value = false) {
    switch (type) {
      case CompactType::kStop:
      case CompactType::kBooleanTrue:
      case CompactType::kBooleanFalse:
        if (collection_value && type != CompactType::kStop) {
          auto ignored = ReadByte();
          if (!ignored.ok()) return ignored.status();
        }
        return absl::OkStatus();
      case CompactType::kByte: {
        auto ignored = ReadByte();
        return ignored.ok() ? absl::OkStatus() : ignored.status();
      }
      case CompactType::kI16:
      case CompactType::kI32:
      case CompactType::kI64: {
        auto ignored = ReadVarint();
        return ignored.ok() ? absl::OkStatus() : ignored.status();
      }
      case CompactType::kDouble:
        return SkipBytes(8);
      case CompactType::kBinary: {
        auto ignored = ReadBinary();
        return ignored.ok() ? absl::OkStatus() : ignored.status();
      }
      case CompactType::kList:
      case CompactType::kSet: {
        ASSIGN_OR_RETURN(auto list, ReadListHeader());
        for (uint32_t i = 0; i < list.size; ++i) {
          RETURN_IF_ERROR(Skip(list.type, true));
        }
        return absl::OkStatus();
      }
      case CompactType::kMap: {
        ASSIGN_OR_RETURN(uint64_t size, ReadVarint());
        if (size == 0) return absl::OkStatus();
        ASSIGN_OR_RETURN(uint8_t types, ReadByte());
        const auto key_type = static_cast<CompactType>(types >> 4);
        const auto value_type = static_cast<CompactType>(types & 0x0f);
        for (uint64_t i = 0; i < size; ++i) {
          RETURN_IF_ERROR(Skip(key_type, true));
          RETURN_IF_ERROR(Skip(value_type, true));
        }
        return absl::OkStatus();
      }
      case CompactType::kStruct: {
        int16_t previous_id = 0;
        while (true) {
          ASSIGN_OR_RETURN(auto field, ReadFieldHeader(&previous_id));
          if (field.type == CompactType::kStop) return absl::OkStatus();
          RETURN_IF_ERROR(Skip(field.type));
        }
      }
    }
    return absl::DataLossError("unknown compact-protocol type");
  }

 private:
  absl::Status SkipBytes(size_t count) {
    if (count > input_.size() - position_) {
      return absl::DataLossError("truncated compact-protocol fixed value");
    }
    position_ += count;
    return absl::OkStatus();
  }

  absl::Span<const uint8_t> input_;
  size_t position_ = 0;
};

enum class PhysicalType : int32_t {
  kInt64 = 2,
  kDouble = 5,
  kByteArray = 6,
};

enum class CompressionCodec : int32_t { kSnappy = 1 };

enum class Encoding : int32_t {
  kPlain = 0,
  kPlainDictionary = 2,
  kRle = 3,
  kRleDictionary = 8,
};

enum class PageType : int32_t {
  kDataPage = 0,
  kIndexPage = 1,
  kDictionaryPage = 2,
  kDataPageV2 = 3,
};

struct ColumnChunk {
  std::string name;
  PhysicalType type = PhysicalType::kByteArray;
  CompressionCodec codec = CompressionCodec::kSnappy;
  int64_t num_values = 0;
  int64_t total_compressed_size = 0;
  int64_t data_page_offset = -1;
  int64_t dictionary_page_offset = -1;
};

struct RowGroup {
  int64_t first_row = 0;
  int64_t num_rows = 0;
  std::vector<ColumnChunk> columns;
};

struct FileMetadata {
  int64_t num_rows = 0;
  std::vector<RowGroup> row_groups;
};

struct DataPageHeader {
  int32_t num_values = -1;
  Encoding encoding = Encoding::kPlain;
};

struct DictionaryPageHeader {
  int32_t num_values = -1;
  Encoding encoding = Encoding::kPlain;
};

struct PageHeader {
  PageType type = PageType::kIndexPage;
  int32_t uncompressed_size = -1;
  int32_t compressed_size = -1;
  DataPageHeader data;
  DictionaryPageHeader dictionary;
};

absl::Status ExpectWireType(CompactType actual, CompactType expected,
                            absl::string_view field) {
  if (actual == expected) return absl::OkStatus();
  return absl::DataLossError(absl::StrCat("wrong wire type for ", field));
}

absl::Status ParseColumnMetadata(CompactReader* reader, ColumnChunk* column) {
  int16_t previous_id = 0;
  while (true) {
    ASSIGN_OR_RETURN(auto field, reader->ReadFieldHeader(&previous_id));
    if (field.type == CompactType::kStop) return absl::OkStatus();

    switch (field.id) {
      case 1: {
        RETURN_IF_ERROR(
            ExpectWireType(field.type, CompactType::kI32, "physical type"));
        ASSIGN_OR_RETURN(int32_t value, reader->ReadI32());
        column->type = static_cast<PhysicalType>(value);
        break;
      }
      case 3: {
        RETURN_IF_ERROR(
            ExpectWireType(field.type, CompactType::kList, "column path"));
        ASSIGN_OR_RETURN(auto list, reader->ReadListHeader());
        if (list.type != CompactType::kBinary || list.size != 1) {
          return absl::UnimplementedError(
              "only flat one-component columns are supported");
        }
        ASSIGN_OR_RETURN(absl::string_view name, reader->ReadBinary());
        column->name.assign(name.data(), name.size());
        break;
      }
      case 4: {
        RETURN_IF_ERROR(
            ExpectWireType(field.type, CompactType::kI32, "compression codec"));
        ASSIGN_OR_RETURN(int32_t value, reader->ReadI32());
        column->codec = static_cast<CompressionCodec>(value);
        break;
      }
      case 5: {
        ASSIGN_OR_RETURN(column->num_values, reader->ReadI64());
        break;
      }
      case 7: {
        ASSIGN_OR_RETURN(column->total_compressed_size, reader->ReadI64());
        break;
      }
      case 9: {
        ASSIGN_OR_RETURN(column->data_page_offset, reader->ReadI64());
        break;
      }
      case 11: {
        ASSIGN_OR_RETURN(column->dictionary_page_offset, reader->ReadI64());
        break;
      }
      default:
        RETURN_IF_ERROR(reader->Skip(field.type));
    }
  }
}

absl::Status ParseColumnChunk(CompactReader* reader, ColumnChunk* column) {
  int16_t previous_id = 0;
  while (true) {
    ASSIGN_OR_RETURN(auto field, reader->ReadFieldHeader(&previous_id));
    if (field.type == CompactType::kStop) return absl::OkStatus();
    if (field.id == 3) {
      RETURN_IF_ERROR(
          ExpectWireType(field.type, CompactType::kStruct, "column metadata"));
      RETURN_IF_ERROR(ParseColumnMetadata(reader, column));
    } else {
      RETURN_IF_ERROR(reader->Skip(field.type));
    }
  }
}

absl::Status ParseRowGroup(CompactReader* reader, RowGroup* row_group) {
  int16_t previous_id = 0;
  while (true) {
    ASSIGN_OR_RETURN(auto field, reader->ReadFieldHeader(&previous_id));
    if (field.type == CompactType::kStop) return absl::OkStatus();
    if (field.id == 1) {
      ASSIGN_OR_RETURN(auto list, reader->ReadListHeader());
      if (list.type != CompactType::kStruct) {
        return absl::DataLossError("row-group columns are not structs");
      }
      row_group->columns.resize(list.size);
      for (ColumnChunk& column : row_group->columns) {
        RETURN_IF_ERROR(ParseColumnChunk(reader, &column));
      }
    } else if (field.id == 3) {
      ASSIGN_OR_RETURN(row_group->num_rows, reader->ReadI64());
    } else {
      RETURN_IF_ERROR(reader->Skip(field.type));
    }
  }
}

absl::StatusOr<FileMetadata> ParseFileMetadata(
    absl::Span<const uint8_t> footer) {
  CompactReader reader(footer);
  FileMetadata metadata;
  int16_t previous_id = 0;
  while (true) {
    ASSIGN_OR_RETURN(auto field, reader.ReadFieldHeader(&previous_id));
    if (field.type == CompactType::kStop) break;
    if (field.id == 3) {
      ASSIGN_OR_RETURN(metadata.num_rows, reader.ReadI64());
    } else if (field.id == 4) {
      ASSIGN_OR_RETURN(auto list, reader.ReadListHeader());
      if (list.type != CompactType::kStruct) {
        return absl::DataLossError("row groups are not structs");
      }
      metadata.row_groups.resize(list.size);
      for (RowGroup& row_group : metadata.row_groups) {
        RETURN_IF_ERROR(ParseRowGroup(&reader, &row_group));
      }
    } else {
      RETURN_IF_ERROR(reader.Skip(field.type));
    }
  }
  return metadata;
}

absl::Status ParseDataPageHeader(CompactReader* reader,
                                 DataPageHeader* header) {
  int16_t previous_id = 0;
  while (true) {
    ASSIGN_OR_RETURN(auto field, reader->ReadFieldHeader(&previous_id));
    if (field.type == CompactType::kStop) return absl::OkStatus();
    if (field.id == 1 || field.id == 2) {
      ASSIGN_OR_RETURN(int32_t value, reader->ReadI32());
      if (field.id == 1)
        header->num_values = value;
      else
        header->encoding = static_cast<Encoding>(value);
    } else {
      RETURN_IF_ERROR(reader->Skip(field.type));
    }
  }
}

absl::Status ParseDictionaryPageHeader(CompactReader* reader,
                                       DictionaryPageHeader* header) {
  int16_t previous_id = 0;
  while (true) {
    ASSIGN_OR_RETURN(auto field, reader->ReadFieldHeader(&previous_id));
    if (field.type == CompactType::kStop) return absl::OkStatus();
    if (field.id == 1 || field.id == 2) {
      ASSIGN_OR_RETURN(int32_t value, reader->ReadI32());
      if (field.id == 1)
        header->num_values = value;
      else
        header->encoding = static_cast<Encoding>(value);
    } else {
      RETURN_IF_ERROR(reader->Skip(field.type));
    }
  }
}

absl::StatusOr<PageHeader> ParsePageHeader(absl::Span<const uint8_t> input,
                                           size_t* bytes_read) {
  CompactReader reader(input);
  PageHeader header;
  int16_t previous_id = 0;
  while (true) {
    ASSIGN_OR_RETURN(auto field, reader.ReadFieldHeader(&previous_id));
    if (field.type == CompactType::kStop) break;
    if (field.id >= 1 && field.id <= 3) {
      ASSIGN_OR_RETURN(int32_t value, reader.ReadI32());
      if (field.id == 1)
        header.type = static_cast<PageType>(value);
      else if (field.id == 2)
        header.uncompressed_size = value;
      else
        header.compressed_size = value;
    } else if (field.id == 5) {
      RETURN_IF_ERROR(ParseDataPageHeader(&reader, &header.data));
    } else if (field.id == 7) {
      RETURN_IF_ERROR(ParseDictionaryPageHeader(&reader, &header.dictionary));
    } else {
      RETURN_IF_ERROR(reader.Skip(field.type));
    }
  }
  if (header.uncompressed_size < 0 || header.compressed_size < 0) {
    return absl::DataLossError("page header has invalid sizes");
  }
  *bytes_read = reader.position();
  return header;
}

uint32_t LoadLittle32(absl::Span<const uint8_t> bytes) {
  return static_cast<uint32_t>(bytes[0]) |
         (static_cast<uint32_t>(bytes[1]) << 8) |
         (static_cast<uint32_t>(bytes[2]) << 16) |
         (static_cast<uint32_t>(bytes[3]) << 24);
}

uint64_t LoadLittle64(absl::Span<const uint8_t> bytes) {
  uint64_t value = 0;
  for (int index = 7; index >= 0; --index) value = (value << 8) | bytes[index];
  return value;
}

absl::StatusOr<std::string> DecompressSnappy(
    absl::Span<const uint8_t> compressed, size_t expected_size) {
  size_t actual_size = 0;
  const char* data = reinterpret_cast<const char*>(compressed.data());
  if (!snappy::GetUncompressedLength(data, compressed.size(), &actual_size) ||
      actual_size != expected_size) {
    return absl::DataLossError("Snappy page has an invalid uncompressed size");
  }
  std::string output(actual_size, '\0');
  if (!snappy::RawUncompress(data, compressed.size(), output.data())) {
    return absl::DataLossError("Snappy page decompression failed");
  }
  return output;
}

struct ColumnData {
  explicit ColumnData(PhysicalType physical_type) : type(physical_type) {}

  size_t size() const {
    switch (type) {
      case PhysicalType::kByteArray:
        return strings.size();
      case PhysicalType::kDouble:
        return doubles.size();
      case PhysicalType::kInt64:
        return integers.size();
    }
    return 0;
  }

  PhysicalType type;
  std::vector<std::string> strings;
  std::vector<double> doubles;
  std::vector<int64_t> integers;
};

absl::Status AppendPlain(absl::Span<const uint8_t> bytes, size_t count,
                         ColumnData* output, size_t* consumed) {
  size_t position = 0;
  for (size_t index = 0; index < count; ++index) {
    switch (output->type) {
      case PhysicalType::kByteArray: {
        if (bytes.size() - position < 4) {
          return absl::DataLossError("truncated PLAIN byte-array length");
        }
        const uint32_t length = LoadLittle32(bytes.subspan(position, 4));
        position += 4;
        if (length > bytes.size() - position) {
          return absl::DataLossError("truncated PLAIN byte array");
        }
        output->strings.emplace_back(
            reinterpret_cast<const char*>(bytes.data() + position), length);
        position += length;
        break;
      }
      case PhysicalType::kDouble: {
        if (bytes.size() - position < 8) {
          return absl::DataLossError("truncated PLAIN double");
        }
        const uint64_t bits = LoadLittle64(bytes.subspan(position, 8));
        double value = 0;
        std::memcpy(&value, &bits, sizeof(value));
        output->doubles.push_back(value);
        position += 8;
        break;
      }
      case PhysicalType::kInt64: {
        if (bytes.size() - position < 8) {
          return absl::DataLossError("truncated PLAIN int64");
        }
        const uint64_t bits = LoadLittle64(bytes.subspan(position, 8));
        int64_t value = 0;
        std::memcpy(&value, &bits, sizeof(value));
        output->integers.push_back(value);
        position += 8;
        break;
      }
    }
  }
  *consumed = position;
  return absl::OkStatus();
}

absl::StatusOr<uint64_t> ReadUnsignedVarint(absl::Span<const uint8_t> bytes,
                                            size_t* position) {
  uint64_t value = 0;
  for (int shift = 0; shift < 64; shift += 7) {
    if (*position >= bytes.size()) {
      return absl::DataLossError("truncated RLE/bit-packed header");
    }
    const uint8_t byte = bytes[(*position)++];
    value |= static_cast<uint64_t>(byte & 0x7f) << shift;
    if ((byte & 0x80) == 0) return value;
  }
  return absl::DataLossError("RLE/bit-packed header overflow");
}

absl::StatusOr<std::vector<uint32_t>> DecodeHybrid(
    absl::Span<const uint8_t> bytes, int bit_width, size_t value_count) {
  if (bit_width < 0 || bit_width > 32) {
    return absl::DataLossError("invalid RLE/bit-packed bit width");
  }
  std::vector<uint32_t> values;
  values.reserve(value_count);
  size_t position = 0;
  while (values.size() < value_count) {
    ASSIGN_OR_RETURN(uint64_t header, ReadUnsignedVarint(bytes, &position));
    if ((header & 1) == 0) {
      const uint64_t run_length = header >> 1;
      const size_t byte_width = (bit_width + 7) / 8;
      if (run_length == 0 || byte_width > bytes.size() - position) {
        return absl::DataLossError("invalid RLE run");
      }
      uint32_t value = 0;
      for (size_t i = 0; i < byte_width; ++i) {
        value |= static_cast<uint32_t>(bytes[position++]) << (8 * i);
      }
      const size_t append =
          std::min<uint64_t>(run_length, value_count - values.size());
      values.insert(values.end(), append, value);
    } else {
      const uint64_t groups = header >> 1;
      if (groups == 0 || groups > std::numeric_limits<size_t>::max() / 8) {
        return absl::DataLossError("invalid bit-packed run");
      }
      const size_t run_values = static_cast<size_t>(groups) * 8;
      const size_t packed_bytes =
          (run_values * static_cast<size_t>(bit_width) + 7) / 8;
      if (packed_bytes > bytes.size() - position) {
        return absl::DataLossError("truncated bit-packed run");
      }
      for (size_t i = 0; i < run_values && values.size() < value_count; ++i) {
        uint32_t value = 0;
        const size_t start_bit = i * static_cast<size_t>(bit_width);
        for (int bit = 0; bit < bit_width; ++bit) {
          const size_t source_bit = start_bit + bit;
          value |= ((bytes[position + source_bit / 8] >> (source_bit % 8)) & 1u)
                   << bit;
        }
        values.push_back(value);
      }
      position += packed_bytes;
    }
  }
  return values;
}

absl::Status AppendDictionaryValue(const ColumnData& dictionary, uint32_t index,
                                   ColumnData* output) {
  if (index >= dictionary.size()) {
    return absl::DataLossError("dictionary index is out of range");
  }
  switch (output->type) {
    case PhysicalType::kByteArray:
      output->strings.push_back(dictionary.strings[index]);
      break;
    case PhysicalType::kDouble:
      output->doubles.push_back(dictionary.doubles[index]);
      break;
    case PhysicalType::kInt64:
      output->integers.push_back(dictionary.integers[index]);
      break;
  }
  return absl::OkStatus();
}

absl::Status DecodeDataPage(absl::Span<const uint8_t> bytes,
                            const DataPageHeader& header,
                            const ColumnData& dictionary, ColumnData* output) {
  if (header.num_values < 0)
    return absl::DataLossError("invalid data-page count");
  const size_t count = static_cast<size_t>(header.num_values);
  if (bytes.size() < 4) return absl::DataLossError("missing definition levels");

  // Every column is optional in the Arrow schema, so max definition level is
  // one. The inspected shards contain no nulls; rejecting nulls lets the public
  // record type remain compact and avoids optional branches in downstream code.
  const uint32_t definition_size = LoadLittle32(bytes.first(4));
  if (definition_size > bytes.size() - 4) {
    return absl::DataLossError("truncated definition levels");
  }
  ASSIGN_OR_RETURN(auto definitions,
                   DecodeHybrid(bytes.subspan(4, definition_size), 1, count));
  if (std::find(definitions.begin(), definitions.end(), 0) !=
      definitions.end()) {
    return absl::UnimplementedError("null FineWeb values are not supported");
  }
  bytes = bytes.subspan(4 + definition_size);

  if (header.encoding == Encoding::kRleDictionary ||
      header.encoding == Encoding::kPlainDictionary) {
    if (bytes.empty()) return absl::DataLossError("missing dictionary indexes");
    const int bit_width = bytes[0];
    ASSIGN_OR_RETURN(auto indexes,
                     DecodeHybrid(bytes.subspan(1), bit_width, count));
    for (const uint32_t index : indexes) {
      RETURN_IF_ERROR(AppendDictionaryValue(dictionary, index, output));
    }
    return absl::OkStatus();
  }
  if (header.encoding == Encoding::kPlain) {
    size_t consumed = 0;
    return AppendPlain(bytes, count, output, &consumed);
  }
  return absl::UnimplementedError("unsupported FineWeb data-page encoding");
}

constexpr std::array<absl::string_view, 10> kColumnNames = {
    "text",      "id",       "dump",           "url",
    "file_path", "language", "language_score", "token_count",
    "score",     "int_score"};
constexpr std::array<PhysicalType, 10> kColumnTypes = {
    PhysicalType::kByteArray, PhysicalType::kByteArray,
    PhysicalType::kByteArray, PhysicalType::kByteArray,
    PhysicalType::kByteArray, PhysicalType::kByteArray,
    PhysicalType::kDouble,    PhysicalType::kInt64,
    PhysicalType::kDouble,    PhysicalType::kInt64};

absl::Status ValidateMetadata(FileMetadata* metadata, int64_t file_size) {
  int64_t first_row = 0;
  for (RowGroup& row_group : metadata->row_groups) {
    row_group.first_row = first_row;
    if (row_group.num_rows <= 0 ||
        row_group.num_rows > std::numeric_limits<int64_t>::max() - first_row) {
      return absl::DataLossError("invalid row-group size");
    }
    first_row += row_group.num_rows;
    if (row_group.columns.size() != kColumnNames.size()) {
      return absl::UnimplementedError(
          "file does not have the FineWeb 10-column schema");
    }
    for (size_t i = 0; i < row_group.columns.size(); ++i) {
      const ColumnChunk& column = row_group.columns[i];
      if (column.name != kColumnNames[i] || column.type != kColumnTypes[i]) {
        return absl::UnimplementedError(
            "file does not match the FineWeb column schema");
      }
      if (column.codec != CompressionCodec::kSnappy) {
        return absl::UnimplementedError(
            "only Snappy FineWeb columns are supported");
      }
      if (column.num_values != row_group.num_rows ||
          column.total_compressed_size <= 0 || column.data_page_offset < 0) {
        return absl::DataLossError("invalid FineWeb column metadata");
      }
      const int64_t start = column.dictionary_page_offset >= 0
                                ? column.dictionary_page_offset
                                : column.data_page_offset;
      if (start < 4 || start > file_size ||
          column.total_compressed_size > file_size - start) {
        return absl::DataLossError("FineWeb column range is outside the file");
      }
    }
  }
  if (first_row != metadata->num_rows) {
    return absl::DataLossError("footer row count does not match row groups");
  }
  return absl::OkStatus();
}

absl::Status ValidateRange(int64_t first, size_t count, int64_t num_rows) {
  if (first < 0 || first > num_rows) {
    return absl::OutOfRangeError("first row is outside the file");
  }
  if (count > static_cast<uint64_t>(num_rows - first)) {
    return absl::OutOfRangeError("requested rows extend past end of file");
  }
  return absl::OkStatus();
}

}  // namespace

struct FineWebParquetReader::Impl {
  ~Impl() {
    if (file_descriptor >= 0) close(file_descriptor);
  }

  absl::StatusOr<std::string> ReadRange(int64_t offset, size_t size) const {
    if (offset < 0 || size > static_cast<uint64_t>(file_size - offset)) {
      return absl::OutOfRangeError("read range is outside the Parquet file");
    }
    std::string output(size, '\0');
    size_t done = 0;
    while (done < size) {
      const ssize_t result = pread(file_descriptor, output.data() + done,
                                   size - done, offset + done);
      if (result < 0) {
        if (errno == EINTR) continue;
        return absl::ErrnoToStatus(errno, "pread failed");
      }
      if (result == 0)
        return absl::DataLossError("unexpected end of Parquet file");
      done += static_cast<size_t>(result);
    }
    return output;
  }

  absl::StatusOr<ColumnData> DecodeColumn(const RowGroup& row_group,
                                          size_t column_index) const {
    const ColumnChunk& column = row_group.columns[column_index];
    const int64_t start = column.dictionary_page_offset >= 0
                              ? column.dictionary_page_offset
                              : column.data_page_offset;
    ASSIGN_OR_RETURN(
        auto bytes,
        ReadRange(start, static_cast<size_t>(column.total_compressed_size)));

    const auto* raw = reinterpret_cast<const uint8_t*>(bytes.data());
    absl::Span<const uint8_t> remaining(raw, bytes.size());
    ColumnData dictionary(column.type);
    ColumnData output(column.type);
    while (output.size() < static_cast<size_t>(row_group.num_rows)) {
      size_t header_size = 0;
      ASSIGN_OR_RETURN(auto page, ParsePageHeader(remaining, &header_size));
      if (header_size > remaining.size() ||
          static_cast<size_t>(page.compressed_size) >
              remaining.size() - header_size) {
        return absl::DataLossError("page extends beyond column chunk");
      }
      const auto compressed = remaining.subspan(
          header_size, static_cast<size_t>(page.compressed_size));
      ASSIGN_OR_RETURN(
          auto uncompressed,
          DecompressSnappy(compressed,
                           static_cast<size_t>(page.uncompressed_size)));
      const auto* page_data =
          reinterpret_cast<const uint8_t*>(uncompressed.data());
      const absl::Span<const uint8_t> body(page_data, uncompressed.size());

      if (page.type == PageType::kDictionaryPage) {
        if (page.dictionary.encoding != Encoding::kPlain ||
            page.dictionary.num_values < 0 || dictionary.size() != 0) {
          return absl::UnimplementedError("unsupported dictionary page");
        }
        size_t consumed = 0;
        RETURN_IF_ERROR(AppendPlain(body, page.dictionary.num_values,
                                    &dictionary, &consumed));
        if (consumed != body.size()) {
          return absl::DataLossError("dictionary page has trailing bytes");
        }
      } else if (page.type == PageType::kDataPage) {
        RETURN_IF_ERROR(DecodeDataPage(body, page.data, dictionary, &output));
      } else if (page.type == PageType::kDataPageV2) {
        return absl::UnimplementedError("data-page v2 is not supported");
      }
      remaining = remaining.subspan(header_size + page.compressed_size);
    }
    if (output.size() != static_cast<size_t>(row_group.num_rows)) {
      return absl::DataLossError("decoded column row count is wrong");
    }
    return output;
  }

  int file_descriptor = -1;
  int64_t file_size = 0;
  FileMetadata metadata;
};

FineWebParquetReader::FineWebParquetReader(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

FineWebParquetReader::~FineWebParquetReader() = default;

absl::StatusOr<std::unique_ptr<FineWebParquetReader>>
FineWebParquetReader::Open(const std::filesystem::path& path) {
  std::unique_ptr<Impl> impl(new Impl);
  impl->file_descriptor = open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (impl->file_descriptor < 0) {
    return absl::ErrnoToStatus(errno,
                               absl::StrCat("cannot open ", path.string()));
  }

  struct stat attributes {};
  if (fstat(impl->file_descriptor, &attributes) != 0) {
    return absl::ErrnoToStatus(errno,
                               absl::StrCat("cannot stat ", path.string()));
  }
  impl->file_size = attributes.st_size;
  if (impl->file_size < 12)
    return absl::DataLossError("file is too short for Parquet");

  ASSIGN_OR_RETURN(auto prefix, impl->ReadRange(0, 4));
  ASSIGN_OR_RETURN(auto trailer, impl->ReadRange(impl->file_size - 8, 8));
  if (prefix != kParquetMagic || trailer.substr(4) != kParquetMagic) {
    return absl::DataLossError("Parquet magic bytes are missing");
  }

  const auto* trailer_bytes = reinterpret_cast<const uint8_t*>(trailer.data());
  const uint32_t footer_size =
      LoadLittle32(absl::Span<const uint8_t>(trailer_bytes, 4));
  if (footer_size > static_cast<uint64_t>(impl->file_size - 12)) {
    return absl::DataLossError("Parquet footer size is invalid");
  }
  ASSIGN_OR_RETURN(
      auto footer,
      impl->ReadRange(impl->file_size - 8 - footer_size, footer_size));
  const auto* footer_bytes = reinterpret_cast<const uint8_t*>(footer.data());
  ASSIGN_OR_RETURN(impl->metadata, ParseFileMetadata(absl::Span<const uint8_t>(
                                       footer_bytes, footer.size())));
  RETURN_IF_ERROR(ValidateMetadata(&impl->metadata, impl->file_size));

  return std::unique_ptr<FineWebParquetReader>(
      new FineWebParquetReader(std::move(impl)));
}

int64_t FineWebParquetReader::num_rows() const {
  return impl_->metadata.num_rows;
}

size_t FineWebParquetReader::num_row_groups() const {
  return impl_->metadata.row_groups.size();
}

absl::StatusOr<std::vector<std::string>> FineWebParquetReader::ReadTextRows(
    int64_t first, size_t count) const {
  RETURN_IF_ERROR(ValidateRange(first, count, num_rows()));
  std::vector<std::string> records;
  if (count == 0) return records;
  records.reserve(count);
  const int64_t end = first + static_cast<int64_t>(count);
  for (const RowGroup& row_group : impl_->metadata.row_groups) {
    const int64_t group_end = row_group.first_row + row_group.num_rows;
    if (group_end <= first) continue;
    if (row_group.first_row >= end) break;
    ASSIGN_OR_RETURN(auto column, impl_->DecodeColumn(row_group, 0));
    const size_t local_begin = static_cast<size_t>(
        std::max(first, row_group.first_row) - row_group.first_row);
    const size_t local_end =
        static_cast<size_t>(std::min(end, group_end) - row_group.first_row);
    for (size_t i = local_begin; i < local_end; ++i) {
      records.push_back(std::move(column.strings[i]));
    }
  }
  return records;
}

absl::StatusOr<std::vector<FineWebRecord>> FineWebParquetReader::ReadRows(
    int64_t first, size_t count) const {
  RETURN_IF_ERROR(ValidateRange(first, count, num_rows()));
  std::vector<FineWebRecord> records;
  if (count == 0) return records;
  records.reserve(count);
  const int64_t end = first + static_cast<int64_t>(count);
  for (const RowGroup& row_group : impl_->metadata.row_groups) {
    const int64_t group_end = row_group.first_row + row_group.num_rows;
    if (group_end <= first) continue;
    if (row_group.first_row >= end) break;

    std::vector<ColumnData> columns;
    columns.reserve(kColumnNames.size());
    for (size_t column = 0; column < kColumnNames.size(); ++column) {
      ASSIGN_OR_RETURN(auto decoded, impl_->DecodeColumn(row_group, column));
      columns.push_back(std::move(decoded));
    }
    const size_t local_begin = static_cast<size_t>(
        std::max(first, row_group.first_row) - row_group.first_row);
    const size_t local_end =
        static_cast<size_t>(std::min(end, group_end) - row_group.first_row);
    for (size_t i = local_begin; i < local_end; ++i) {
      records.push_back(FineWebRecord{
          std::move(columns[0].strings[i]), std::move(columns[1].strings[i]),
          std::move(columns[2].strings[i]), std::move(columns[3].strings[i]),
          std::move(columns[4].strings[i]), std::move(columns[5].strings[i]),
          columns[6].doubles[i], columns[7].integers[i], columns[8].doubles[i],
          columns[9].integers[i]});
    }
  }
  return records;
}

}  // namespace pluto::parquet
