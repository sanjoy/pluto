#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <string>
#include <system_error>
#include <utility>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "src/llm/experiments/weight_history/history.h"
#include "src/llm/experiments/weight_history/html.h"
#include "src/util/status_macros.h"

ABSL_FLAG(std::string, checkpoint, "",
          "Directory containing step_<number> checkpoint directories");
ABSL_FLAG(std::string, output, "",
          "New self-contained HTML report; existing files are never replaced");

namespace pluto::llm::weight_history {
namespace {

// Check before scanning potentially large checkpoints. lstat also rejects a
// dangling output symlink; the later exclusive open closes the check/open race.
absl::Status CheckOutput(const std::string& output) {
  struct stat info;
  if (lstat(output.c_str(), &info) == 0)
    return absl::AlreadyExistsError(
        absl::StrCat("output already exists: ", output));
  if (errno != ENOENT)
    return absl::ErrnoToStatus(errno,
                               absl::StrCat("cannot inspect output: ", output));
  auto parent = std::filesystem::path(output).parent_path();
  if (parent.empty())
    parent = ".";
  std::error_code error;
  const auto status = std::filesystem::status(parent, error);
  if (error)
    return absl::InvalidArgumentError(
        absl::StrCat("cannot inspect output directory ", parent.string(), ": ",
                     error.message()));
  if (!std::filesystem::is_directory(status))
    return absl::InvalidArgumentError(
        absl::StrCat("output parent is not a directory: ", parent.string()));
  return absl::OkStatus();
}

// Never truncate a preexisting report (or accidentally overwrite a checkpoint).
// Analysis and rendering finish before opening this file. A disk/write failure
// may leave a partial new report, but cannot damage an existing destination.
absl::Status WriteNewFile(const std::string& path,
                          const std::string& contents) {
  const int fd =
      open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0666);
  if (fd < 0)
    return absl::ErrnoToStatus(errno,
                               absl::StrCat("cannot create output: ", path));
  size_t offset = 0;
  while (offset < contents.size()) {
    // Bound individual writes even if a very large history produces a report
    // exceeding the operating system's maximum single-write count.
    const size_t count = std::min(contents.size() - offset, size_t{1} << 30);
    const ssize_t written = write(fd, contents.data() + offset, count);
    if (written < 0 && errno == EINTR)
      continue;
    if (written <= 0) {
      const int saved_errno = written < 0 ? errno : EIO;
      close(fd);
      return absl::ErrnoToStatus(
          saved_errno,
          absl::StrCat("failed writing output (partial file may remain): ",
                       path));
    }
    offset += static_cast<size_t>(written);
  }
  // Do not retry close on EINTR: on Linux the descriptor has already closed.
  if (close(fd) != 0)
    return absl::ErrnoToStatus(errno,
                               absl::StrCat("failed closing output: ", path));
  return absl::OkStatus();
}

absl::Status Run() {
  const std::string checkpoint = absl::GetFlag(FLAGS_checkpoint);
  const std::string output = absl::GetFlag(FLAGS_output);
  if (checkpoint.empty() || output.empty())
    return absl::InvalidArgumentError("--checkpoint and --output are required");
  RETURN_IF_ERROR(CheckOutput(output));
  ASSIGN_OR_RETURN(auto history, AnalyzeDirectory(
                                     checkpoint,
                                     [](size_t completed, size_t total) {
                                       std::cout << "Analyzed weight tensors: "
                                                 << completed << '/' << total
                                                 << std::endl;
                                     }));
  std::ostringstream report;
  RETURN_IF_ERROR(WriteHtml(report, history));
  RETURN_IF_ERROR(WriteNewFile(output, std::move(report).str()));
  std::cout << "Wrote " << output << " (" << history.tensors.size()
            << " weight tensors, " << history.steps.size() << " checkpoints)\n";
  return absl::OkStatus();
}

}  // namespace
}  // namespace pluto::llm::weight_history

int main(int argc, char** argv) {
  const auto positional = absl::ParseCommandLine(argc, argv);
  if (positional.size() != 1) {
    std::cerr << "Use --checkpoint=<directory> and --output=<file.html>; "
                 "positional arguments are not accepted.\n";
    return 1;
  }
  const auto status = pluto::llm::weight_history::Run();
  if (!status.ok()) {
    std::cerr << status << '\n';
    return 1;
  }
  return 0;
}
