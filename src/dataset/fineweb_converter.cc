#include "src/dataset/fineweb_converter.h"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "src/common/status_macros.h"
#include "src/parquet/fineweb_parquet_reader.h"
#include "src/dataset/document_file.h"

namespace pluto::tokenized {
namespace {

absl::Status WithContext(const absl::Status& status,
                         const std::string& context) {
  return absl::Status(status.code(),
                      absl::StrCat(context, ": ", status.message()));
}

}  // namespace

absl::Status ConvertFineWebParquetFile(
    const std::filesystem::path& input_path,
    const std::filesystem::path& output_path,
    const tokenizer::Gpt2Tokenizer& tokenizer,
    FineWebConversionOptions options) {
  if (options.batch_size == 0) {
    return absl::InvalidArgumentError("conversion batch size must be positive");
  }

  auto reader = parquet::FineWebParquetReader::Open(input_path);
  if (!reader.ok()) {
    return WithContext(reader.status(),
                       absl::StrCat("cannot read ", input_path.string()));
  }
  const int64_t row_count = (*reader)->num_rows();
  if (row_count < 0 ||
      static_cast<uint64_t>(row_count) > std::numeric_limits<uint32_t>::max()) {
    return absl::ResourceExhaustedError(
        "tokenized-document files support at most UINT32_MAX documents");
  }

  auto writer =
      DocumentFileWriter::Create(output_path, static_cast<uint32_t>(row_count));
  if (!writer.ok()) {
    return WithContext(writer.status(),
                       absl::StrCat("cannot create ", output_path.string()));
  }

  int64_t first = 0;
  while (first < row_count) {
    const size_t remaining = static_cast<size_t>(row_count - first);
    const size_t count = std::min(remaining, options.batch_size);
    auto texts = (*reader)->ReadTextRows(first, count);
    if (!texts.ok()) {
      return WithContext(texts.status(),
                         absl::StrCat("cannot read rows beginning at ", first,
                                      " from ", input_path.string()));
    }

    for (const std::string& text : *texts) {
      auto encoded = tokenizer.Encode(text);
      if (!encoded.ok()) {
        return WithContext(encoded.status(),
                           absl::StrCat("cannot tokenize document ",
                                        (*writer)->documents_written(),
                                        " from ", input_path.string()));
      }

      std::vector<uint16_t> token_ids;
      token_ids.reserve(encoded->size());
      for (const int token_id : *encoded) {
        if (token_id < 0 || token_id > std::numeric_limits<uint16_t>::max()) {
          return absl::FailedPreconditionError(absl::StrCat(
              "token id ", token_id, " cannot be stored as uint16"));
        }
        token_ids.push_back(static_cast<uint16_t>(token_id));
      }
      RETURN_IF_ERROR((*writer)->AddDocument(token_ids));
    }
    first += static_cast<int64_t>(count);
  }

  const absl::Status status = (*writer)->Close();
  if (!status.ok()) {
    return WithContext(status,
                       absl::StrCat("cannot finalize ", output_path.string()));
  }
  return absl::OkStatus();
}

}  // namespace pluto::tokenized
