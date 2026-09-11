#include <cuda_runtime_api.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <bit>
#include <cerrno>
#include <cstring>
#include <limits>

#include "absl/strings/str_cat.h"
#include "ai-slop/weight_analysis/token_trace_probe.h"
#include "src/util/status_macros.h"

namespace pluto::weight_analysis {
namespace {
class FileDescriptor final {
 public:
  explicit FileDescriptor(int value) : value_(value) {}
  ~FileDescriptor() { close(value_); }
  int get() const { return value_; }
  FileDescriptor(const FileDescriptor&) = delete;
  FileDescriptor& operator=(const FileDescriptor&) = delete;

 private:
  int value_;
};
}  // namespace

absl::Status ValidateTokenIds(absl::Span<const int32_t> ids, int vocabulary,
                              int context_length) {
  if (vocabulary <= 0 || context_length <= 0 || ids.empty() ||
      ids.size() > static_cast<size_t>(context_length)) {
    return absl::InvalidArgumentError("invalid token count or model limits");
  }
  for (size_t i = 0; i < ids.size(); ++i) {
    if (ids[i] < 0 || ids[i] >= vocabulary) {
      return absl::InvalidArgumentError(
          absl::StrCat("token ID outside logical vocabulary at index ", i));
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<cuda::PageLockedHostArray<int32_t>> ReadTokenIds(
    cuda::Executor& executor, const std::filesystem::path& path, int vocabulary,
    int context_length) {
  if constexpr (std::endian::native != std::endian::little)
    return absl::UnimplementedError("token files require little-endian host");
  if (vocabulary <= 0 || context_length <= 0)
    return absl::InvalidArgumentError("invalid model limits");
  // O_NONBLOCK also prevents accidentally blocking forever on a FIFO before
  // fstat can reject it. O_NOFOLLOW refuses a symlink as the final component.
  const int raw =
      open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK);
  if (raw < 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("cannot open token file: ", std::strerror(errno)));
  }
  const FileDescriptor fd(raw);
  struct stat before {};
  if (fstat(fd.get(), &before) != 0 || !S_ISREG(before.st_mode) ||
      before.st_size <= 0 || before.st_size % sizeof(int32_t) != 0 ||
      static_cast<uint64_t>(before.st_size) >
          static_cast<uint64_t>(context_length) * sizeof(int32_t)) {
    return absl::InvalidArgumentError(
        "token file must be regular, nonempty, and contain 1..context int32s");
  }
  ASSIGN_OR_RETURN(auto ids, cuda::PageLockedHostArray<int32_t>::Allocate(
                                 executor, before.st_size / sizeof(int32_t)));
  size_t done = 0;
  while (done < ids.size_bytes()) {
    auto* bytes = reinterpret_cast<uint8_t*>(ids.data());
    const ssize_t got = read(fd.get(), bytes + done, ids.size_bytes() - done);
    if (got < 0 && errno == EINTR)
      continue;
    if (got <= 0)
      return absl::DataLossError("short read of token file");
    done += got;
  }
  struct stat after {};
  struct stat named {};
  if (fstat(fd.get(), &after) != 0 || lstat(path.c_str(), &named) != 0 ||
      !S_ISREG(named.st_mode) || before.st_dev != named.st_dev ||
      before.st_ino != named.st_ino || before.st_size != after.st_size ||
      before.st_mtim.tv_sec != after.st_mtim.tv_sec ||
      before.st_mtim.tv_nsec != after.st_mtim.tv_nsec ||
      before.st_ctim.tv_sec != after.st_ctim.tv_sec ||
      before.st_ctim.tv_nsec != after.st_ctim.tv_nsec) {
    return absl::DataLossError("token file changed during reading");
  }
  RETURN_IF_ERROR(ValidateTokenIds(ids.span(), vocabulary, context_length));
  return ids;
}

absl::StatusOr<cuda::PageLockedHostArray<uint8_t>> ReadSelectedRow(
    cuda::Executor& executor, const cuda::Buffer& buffer, int row, int width,
    size_t element_bytes) {
  if (&buffer.executor() != &executor || row < 0 || width <= 0 ||
      (element_bytes != 2 && element_bytes != 4) ||
      static_cast<size_t>(width) >
          std::numeric_limits<size_t>::max() / element_bytes) {
    return absl::InvalidArgumentError("invalid row shape, type, or executor");
  }
  const size_t row_bytes = static_cast<size_t>(width) * element_bytes;
  // Division before multiplication means even a huge row index cannot wrap.
  if (buffer.size_bytes() % row_bytes != 0 ||
      static_cast<size_t>(row) >= buffer.size_bytes() / row_bytes) {
    return absl::InvalidArgumentError("selected row outside device matrix");
  }
  const size_t offset = static_cast<size_t>(row) * row_bytes;
  ASSIGN_OR_RETURN(auto host, cuda::PageLockedHostArray<uint8_t>::Allocate(
                                  executor, row_bytes));
  const auto* source = static_cast<const uint8_t*>(buffer.data()) + offset;
  RETURN_IF_ERROR(cuda::CudaStatus(
      cudaMemcpyAsync(host.data(), source, row_bytes, cudaMemcpyDeviceToHost,
                      executor.stream()),
      "read selected native row"));
  RETURN_IF_ERROR(executor.Synchronize());
  return host;
}

}  // namespace pluto::weight_analysis
