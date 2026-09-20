#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"

namespace pluto::tokenized {

// Reader for the compact tokenized-document format:
//
//   uint32_le document_count
//   uint32_le document_lengths[document_count]
//   uint16_le tokens[sum(document_lengths)]
//
// Open() validates the complete shape of the file and builds byte offsets once.
// ReadDocument() uses pread(), so calls do not share a seek position and may
// run concurrently.
class DocumentFileReader final {
 public:
  static absl::StatusOr<std::unique_ptr<DocumentFileReader>> Open(
      const std::filesystem::path& path);
  ~DocumentFileReader();

  DocumentFileReader(const DocumentFileReader&) = delete;
  DocumentFileReader& operator=(const DocumentFileReader&) = delete;

  uint32_t num_documents() const;
  absl::StatusOr<uint32_t> document_length(uint32_t index) const;
  absl::StatusOr<std::vector<uint16_t>> ReadDocument(uint32_t index) const;

 private:
  struct Impl;
  explicit DocumentFileReader(std::unique_ptr<Impl> impl);

  std::unique_ptr<Impl> impl_;
};

// Streaming writer for the same format.
//
// The caller declares the document count up front, but documents themselves are
// streamed directly to disk. Only the uint32 length table and a 1 MiB token
// buffer remain in memory. Close() fills the header and atomically renames a
// temporary file over the requested destination; abandoning the writer removes
// the temporary file without publishing a partial result.
class DocumentFileWriter final {
 public:
  static absl::StatusOr<std::unique_ptr<DocumentFileWriter>> Create(
      const std::filesystem::path& path, uint32_t num_documents);
  ~DocumentFileWriter();

  DocumentFileWriter(const DocumentFileWriter&) = delete;
  DocumentFileWriter& operator=(const DocumentFileWriter&) = delete;

  absl::Status AddDocument(absl::Span<const uint16_t> token_ids);
  absl::Status Close();

  uint32_t expected_documents() const;
  uint32_t documents_written() const;

 private:
  struct Impl;
  explicit DocumentFileWriter(std::unique_ptr<Impl> impl);

  std::unique_ptr<Impl> impl_;
};

}  // namespace pluto::tokenized
