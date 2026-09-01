#include "src/dataset/document_file.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
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
#include "absl/types/span.h"
#include "src/util/status_macros.h"

namespace pluto::tokenized {
namespace {

constexpr size_t kPayloadBufferSize = 1 << 20;

uint32_t LoadLittle32(const uint8_t* bytes) {
  return static_cast<uint32_t>(bytes[0]) |
         (static_cast<uint32_t>(bytes[1]) << 8) |
         (static_cast<uint32_t>(bytes[2]) << 16) |
         (static_cast<uint32_t>(bytes[3]) << 24);
}

uint16_t LoadLittle16(const uint8_t* bytes) {
  return static_cast<uint16_t>(bytes[0]) |
         (static_cast<uint16_t>(bytes[1]) << 8);
}

void StoreLittle32(uint32_t value, uint8_t* output) {
  output[0] = static_cast<uint8_t>(value);
  output[1] = static_cast<uint8_t>(value >> 8);
  output[2] = static_cast<uint8_t>(value >> 16);
  output[3] = static_cast<uint8_t>(value >> 24);
}

absl::Status ReadExactly(int file_descriptor, uint64_t offset,
                         absl::Span<uint8_t> output) {
  size_t done = 0;
  while (done < output.size()) {
    const ssize_t result =
        pread(file_descriptor, output.data() + done, output.size() - done,
              static_cast<off_t>(offset + done));
    if (result < 0) {
      if (errno == EINTR) continue;
      return absl::ErrnoToStatus(errno, "pread failed");
    }
    if (result == 0) return absl::DataLossError("unexpected end of file");
    done += static_cast<size_t>(result);
  }
  return absl::OkStatus();
}

absl::Status WriteExactly(int file_descriptor, absl::Span<const uint8_t> data) {
  size_t done = 0;
  while (done < data.size()) {
    const ssize_t result =
        write(file_descriptor, data.data() + done, data.size() - done);
    if (result < 0) {
      if (errno == EINTR) continue;
      return absl::ErrnoToStatus(errno, "write failed");
    }
    if (result == 0) {
      return absl::InternalError("write made no progress");
    }
    done += static_cast<size_t>(result);
  }
  return absl::OkStatus();
}

absl::Status PwriteExactly(int file_descriptor, uint64_t offset,
                           absl::Span<const uint8_t> data) {
  size_t done = 0;
  while (done < data.size()) {
    const ssize_t result =
        pwrite(file_descriptor, data.data() + done, data.size() - done,
               static_cast<off_t>(offset + done));
    if (result < 0) {
      if (errno == EINTR) continue;
      return absl::ErrnoToStatus(errno, "pwrite failed");
    }
    if (result == 0) {
      return absl::InternalError("pwrite made no progress");
    }
    done += static_cast<size_t>(result);
  }
  return absl::OkStatus();
}

}  // namespace

struct DocumentFileReader::Impl {
  ~Impl() {
    if (file_descriptor >= 0) close(file_descriptor);
  }

  int file_descriptor = -1;
  uint64_t file_size = 0;
  std::vector<uint32_t> lengths;
  std::vector<uint64_t> offsets;
};

DocumentFileReader::DocumentFileReader(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

DocumentFileReader::~DocumentFileReader() = default;

absl::StatusOr<std::unique_ptr<DocumentFileReader>> DocumentFileReader::Open(
    const std::filesystem::path& path) {
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
  if (attributes.st_size < 4) {
    return absl::DataLossError(
        "tokenized-document file is shorter than its header");
  }
  impl->file_size = static_cast<uint64_t>(attributes.st_size);

  uint8_t count_bytes[4];
  RETURN_IF_ERROR(ReadExactly(impl->file_descriptor, 0, count_bytes));
  const uint32_t document_count = LoadLittle32(count_bytes);
  const uint64_t header_size = 4 + 4 * static_cast<uint64_t>(document_count);
  if (header_size > impl->file_size) {
    return absl::DataLossError(
        "document-length table extends past end of file");
  }

  std::vector<uint8_t> encoded_lengths(4 * static_cast<size_t>(document_count));
  RETURN_IF_ERROR(
      ReadExactly(impl->file_descriptor, 4, absl::MakeSpan(encoded_lengths)));
  impl->lengths.resize(document_count);
  impl->offsets.resize(static_cast<size_t>(document_count) + 1);
  impl->offsets[0] = header_size;
  for (uint32_t index = 0; index < document_count; ++index) {
    impl->lengths[index] = LoadLittle32(encoded_lengths.data() + 4 * index);
    const uint64_t token_bytes =
        2 * static_cast<uint64_t>(impl->lengths[index]);
    if (token_bytes > impl->file_size - impl->offsets[index]) {
      return absl::DataLossError(
          "document token payload extends past end of file");
    }
    impl->offsets[index + 1] = impl->offsets[index] + token_bytes;
  }
  if (impl->offsets.back() != impl->file_size) {
    return absl::DataLossError("tokenized-document file has trailing bytes");
  }

  return std::unique_ptr<DocumentFileReader>(
      new DocumentFileReader(std::move(impl)));
}

uint32_t DocumentFileReader::num_documents() const {
  return static_cast<uint32_t>(impl_->lengths.size());
}

absl::StatusOr<uint32_t> DocumentFileReader::document_length(
    uint32_t index) const {
  if (index >= impl_->lengths.size()) {
    return absl::OutOfRangeError("document index is outside the file");
  }
  return impl_->lengths[index];
}

absl::StatusOr<std::vector<uint16_t>> DocumentFileReader::ReadDocument(
    uint32_t index) const {
  ASSIGN_OR_RETURN(uint32_t length, document_length(index));
  std::vector<uint8_t> encoded(2 * static_cast<size_t>(length));
  RETURN_IF_ERROR(ReadExactly(impl_->file_descriptor, impl_->offsets[index],
                              absl::MakeSpan(encoded)));

  std::vector<uint16_t> tokens(length);
  for (size_t token = 0; token < tokens.size(); ++token) {
    tokens[token] = LoadLittle16(encoded.data() + 2 * token);
  }
  return tokens;
}

struct DocumentFileWriter::Impl {
  ~Impl() {
    if (file_descriptor >= 0) close(file_descriptor);
    if (!published && !temporary_path.empty()) unlink(temporary_path.c_str());
  }

  absl::Status FlushPayload() {
    if (payload.empty()) return absl::OkStatus();
    auto status = WriteExactly(file_descriptor, payload);
    if (status.ok()) payload.clear();
    return status;
  }

  int file_descriptor = -1;
  std::filesystem::path final_path;
  std::filesystem::path temporary_path;
  uint32_t expected_documents = 0;
  std::vector<uint32_t> lengths;
  std::vector<uint8_t> payload;
  bool finalization_started = false;
  bool published = false;
};

DocumentFileWriter::DocumentFileWriter(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

DocumentFileWriter::~DocumentFileWriter() = default;

absl::StatusOr<std::unique_ptr<DocumentFileWriter>> DocumentFileWriter::Create(
    const std::filesystem::path& path, uint32_t num_documents) {
  std::unique_ptr<Impl> impl(new Impl);
  impl->final_path = path;
  impl->expected_documents = num_documents;
  impl->lengths.reserve(num_documents);
  impl->payload.reserve(kPayloadBufferSize);

  std::string temporary = absl::StrCat(path.string(), ".tmp.XXXXXX");
  std::vector<char> writable_name(temporary.begin(), temporary.end());
  writable_name.push_back('\0');
  impl->file_descriptor = mkstemp(writable_name.data());
  if (impl->file_descriptor < 0) {
    return absl::ErrnoToStatus(
        errno,
        absl::StrCat("cannot create temporary file for ", path.string()));
  }
  impl->temporary_path = writable_name.data();
  if (fcntl(impl->file_descriptor, F_SETFD, FD_CLOEXEC) != 0) {
    return absl::ErrnoToStatus(errno,
                               "cannot mark temporary file close-on-exec");
  }

  const uint64_t header_size = 4 + 4 * static_cast<uint64_t>(num_documents);
  if (header_size > static_cast<uint64_t>(std::numeric_limits<off_t>::max())) {
    return absl::ResourceExhaustedError("document-length table is too large");
  }
  if (ftruncate(impl->file_descriptor, static_cast<off_t>(header_size)) != 0 ||
      lseek(impl->file_descriptor, static_cast<off_t>(header_size), SEEK_SET) <
          0) {
    return absl::ErrnoToStatus(errno, "cannot reserve document-length table");
  }

  return std::unique_ptr<DocumentFileWriter>(
      new DocumentFileWriter(std::move(impl)));
}

absl::Status DocumentFileWriter::AddDocument(
    absl::Span<const uint16_t> token_ids) {
  if (impl_->finalization_started) {
    return absl::FailedPreconditionError("writer is already being finalized");
  }
  if (impl_->lengths.size() >= impl_->expected_documents) {
    return absl::OutOfRangeError("more documents supplied than declared");
  }
  if (token_ids.size() > std::numeric_limits<uint32_t>::max()) {
    return absl::ResourceExhaustedError("document has too many tokens");
  }

  impl_->lengths.push_back(static_cast<uint32_t>(token_ids.size()));
  for (const uint16_t token : token_ids) {
    if (impl_->payload.size() + 2 > kPayloadBufferSize) {
      RETURN_IF_ERROR(impl_->FlushPayload());
    }
    impl_->payload.push_back(static_cast<uint8_t>(token));
    impl_->payload.push_back(static_cast<uint8_t>(token >> 8));
  }
  return absl::OkStatus();
}

absl::Status DocumentFileWriter::Close() {
  if (impl_->finalization_started) {
    return absl::FailedPreconditionError("writer has already been finalized");
  }
  impl_->finalization_started = true;
  if (impl_->lengths.size() != impl_->expected_documents) {
    return absl::FailedPreconditionError(
        absl::StrCat("expected ", impl_->expected_documents, " documents, got ",
                     impl_->lengths.size()));
  }

  RETURN_IF_ERROR(impl_->FlushPayload());
  std::vector<uint8_t> header(4 + 4 * impl_->lengths.size());
  StoreLittle32(impl_->expected_documents, header.data());
  for (size_t index = 0; index < impl_->lengths.size(); ++index) {
    StoreLittle32(impl_->lengths[index], header.data() + 4 + 4 * index);
  }
  RETURN_IF_ERROR(PwriteExactly(impl_->file_descriptor, 0, header));
  if (close(impl_->file_descriptor) != 0) {
    impl_->file_descriptor = -1;
    return absl::ErrnoToStatus(errno, "cannot close tokenized-document file");
  }
  impl_->file_descriptor = -1;
  if (rename(impl_->temporary_path.c_str(), impl_->final_path.c_str()) != 0) {
    return absl::ErrnoToStatus(errno, "cannot publish tokenized-document file");
  }
  impl_->published = true;
  return absl::OkStatus();
}

uint32_t DocumentFileWriter::expected_documents() const {
  return impl_->expected_documents;
}

uint32_t DocumentFileWriter::documents_written() const {
  return static_cast<uint32_t>(impl_->lengths.size());
}

}  // namespace pluto::tokenized
