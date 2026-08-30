#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "absl/status/statusor.h"

namespace pluto::parquet {

// One row in the FineWeb-Edu 10BT sample schema.
struct FineWebRecord {
  std::string text;
  std::string id;
  std::string dump;
  std::string url;
  std::string file_path;
  std::string language;
  double language_score = 0;
  int64_t token_count = 0;
  double score = 0;
  int64_t int_score = 0;
};

// A deliberately narrow, low-overhead reader for the FineWeb-Edu sample
// shards. It is not a general-purpose Parquet implementation.
//
// Supported physical layout:
//   * the exact flat FineWebRecord schema above;
//   * Parquet data-page v1, dictionary or PLAIN values;
//   * RLE definition levels with no null values;
//   * Snappy-compressed BYTE_ARRAY, DOUBLE, and INT64 columns.
//
// Open() parses only the footer. Reads use pread(), so independent calls do not
// share a mutable file position and may safely run concurrently. ReadTextRows()
// projects only the large `text` column and is the preferred fast path for ML
// input pipelines.
class FineWebParquetReader final {
 public:
  static absl::StatusOr<std::unique_ptr<FineWebParquetReader>> Open(
      const std::filesystem::path& path);
  ~FineWebParquetReader();

  FineWebParquetReader(const FineWebParquetReader&) = delete;
  FineWebParquetReader& operator=(const FineWebParquetReader&) = delete;

  int64_t num_rows() const;
  size_t num_row_groups() const;

  // Reads exactly `count` rows starting at the zero-based row index `first`.
  // Crossing row-group boundaries is supported. Out-of-range requests fail.
  absl::StatusOr<std::vector<FineWebRecord>> ReadRows(int64_t first,
                                                       size_t count) const;

  // Projection optimized for tokenizer pipelines: other columns are never read
  // or decompressed.
  absl::StatusOr<std::vector<std::string>> ReadTextRows(int64_t first,
                                                         size_t count) const;

 private:
  struct Impl;
  explicit FineWebParquetReader(std::unique_ptr<Impl> impl);

  std::unique_ptr<Impl> impl_;
};

}  // namespace pluto::parquet
