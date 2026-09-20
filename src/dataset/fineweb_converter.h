#pragma once

#include <cstddef>
#include <filesystem>

#include "absl/status/status.h"
#include "src/cuda/executor.h"
#include "src/dataset/tokenizer.h"

namespace pluto::tokenized {

struct FineWebConversionOptions {
  // FineWeb shards use 1,000-row groups. Matching that size amortizes Parquet
  // setup while bounding the amount of decoded text held in memory.
  size_t batch_size = 1000;
};

// Projects the text column from a FineWeb Parquet shard, encodes each document
// with the supplied tokenizer, and writes a tokenized-document file. The output
// is published atomically, so an error cannot leave a file that looks complete.
absl::Status ConvertFineWebParquetFile(cuda::Executor& executor,
                                       const std::filesystem::path& input_path,
                                       const std::filesystem::path& output_path,
                                       const tokenizer::Tokenizer& tokenizer,
                                       FineWebConversionOptions options = {});

}  // namespace pluto::tokenized
