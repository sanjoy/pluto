#include <algorithm>
#include <atomic>
#include <cstddef>
#include <filesystem>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "src/common/status_macros.h"
#include "src/tokenization/fineweb_converter.h"
#include "src/tokenization/tokenizer.h"

ABSL_FLAG(std::string, input_dir, "",
          "Directory containing FineWeb .parquet shards.");
ABSL_FLAG(std::string, output_dir, "",
          "Directory in which matching .tokenized files are written.");
ABSL_FLAG(std::string, tokenizer_dir, "",
          "Directory containing the saved GPT-2 tokenizer.json.");
ABSL_FLAG(
    int, jobs, 0,
    "Number of shards to convert concurrently; zero uses available CPUs.");
ABSL_FLAG(int, batch_size, 1000,
          "Number of Parquet records decoded per batch.");
ABSL_FLAG(bool, overwrite, false, "Replace existing .tokenized shard outputs.");

namespace pluto::tokenized {
namespace {

struct WorkItem {
  std::filesystem::path input;
  std::filesystem::path output;
};

absl::StatusOr<std::vector<std::filesystem::path>> FindParquetShards(
    const std::filesystem::path& directory) {
  std::error_code error;
  if (!std::filesystem::is_directory(directory, error)) {
    return absl::InvalidArgumentError(
        absl::StrCat("input is not a directory: ", directory.string()));
  }

  std::vector<std::filesystem::path> shards;
  std::filesystem::directory_iterator iterator(directory, error);
  const std::filesystem::directory_iterator end;
  while (!error && iterator != end) {
    std::error_code entry_error;
    if (iterator->is_regular_file(entry_error) && !entry_error &&
        iterator->path().extension() == ".parquet") {
      shards.push_back(iterator->path());
    }
    iterator.increment(error);
  }
  if (error) {
    return absl::ErrnoToStatus(
        error.value(), absl::StrCat("cannot enumerate ", directory.string()));
  }
  std::sort(shards.begin(), shards.end());
  if (shards.empty()) {
    return absl::NotFoundError(
        absl::StrCat("no .parquet files in ", directory.string()));
  }
  return shards;
}

absl::Status RunConversion() {
  const std::filesystem::path input_dir = absl::GetFlag(FLAGS_input_dir);
  const std::filesystem::path output_dir = absl::GetFlag(FLAGS_output_dir);
  const std::filesystem::path tokenizer_dir =
      absl::GetFlag(FLAGS_tokenizer_dir);
  const int requested_jobs = absl::GetFlag(FLAGS_jobs);
  const int requested_batch_size = absl::GetFlag(FLAGS_batch_size);
  if (input_dir.empty() || output_dir.empty() || tokenizer_dir.empty()) {
    return absl::InvalidArgumentError(
        "--input_dir, --output_dir, and --tokenizer_dir are required");
  }
  if (requested_jobs < 0) {
    return absl::InvalidArgumentError("--jobs cannot be negative");
  }
  if (requested_batch_size <= 0) {
    return absl::InvalidArgumentError("--batch_size must be positive");
  }

  ASSIGN_OR_RETURN(auto shards, FindParquetShards(input_dir));

  std::error_code error;
  std::filesystem::create_directories(output_dir, error);
  if (error) {
    return absl::ErrnoToStatus(
        error.value(), absl::StrCat("cannot create ", output_dir.string()));
  }

  const bool overwrite = absl::GetFlag(FLAGS_overwrite);
  size_t skipped = 0;
  std::vector<WorkItem> work;
  for (const std::filesystem::path& input : shards) {
    std::filesystem::path output = output_dir / input.stem();
    output += ".tokenized";
    error.clear();
    const bool exists = std::filesystem::exists(output, error);
    if (error) {
      return absl::ErrnoToStatus(
          error.value(), absl::StrCat("cannot inspect ", output.string()));
    }
    if (exists && !overwrite) {
      ++skipped;
    } else {
      work.push_back({input, output});
    }
  }

  if (work.empty()) {
    std::cout << "Nothing to do; " << skipped
              << " output file(s) already exist.\n";
    return absl::OkStatus();
  }

  const unsigned int detected_cpus = std::thread::hardware_concurrency();
  size_t worker_count =
      requested_jobs == 0
          ? std::max<size_t>(1, static_cast<size_t>(detected_cpus))
          : static_cast<size_t>(requested_jobs);
  worker_count = std::min(worker_count, work.size());

  // Load one encoder per worker. The parsed immutable GPT-2 model is cached and
  // shared, while each encoder keeps an independent BPE cache to avoid lock
  // contention during CPU-heavy shard conversion.
  std::vector<std::unique_ptr<tokenizer::Gpt2Tokenizer>> encoders;
  encoders.reserve(worker_count);
  for (size_t worker = 0; worker < worker_count; ++worker) {
    ASSIGN_OR_RETURN(auto encoder,
                     tokenizer::Gpt2Tokenizer::Load(tokenizer_dir));
    encoders.push_back(std::move(encoder));
  }

  std::cout << "Converting " << work.size() << " shard(s) with " << worker_count
            << " worker(s)";
  if (skipped != 0) std::cout << "; skipping " << skipped << " existing";
  std::cout << ".\n";

  std::atomic<size_t> next{0};
  std::atomic<bool> stop{false};
  std::mutex mutex;
  absl::Status failure;
  std::vector<std::thread> workers;
  workers.reserve(worker_count);
  for (size_t worker = 0; worker < worker_count; ++worker) {
    workers.emplace_back([&, worker] {
      while (!stop.load(std::memory_order_relaxed)) {
        const size_t index = next.fetch_add(1, std::memory_order_relaxed);
        if (index >= work.size()) return;
        const WorkItem& item = work[index];
        {
          std::lock_guard<std::mutex> lock(mutex);
          std::cout << "Starting " << item.input.filename().string() << "\n";
        }

        FineWebConversionOptions options;
        options.batch_size = static_cast<size_t>(requested_batch_size);
        const absl::Status status = ConvertFineWebParquetFile(
            item.input, item.output, *encoders[worker], options);
        if (!status.ok()) {
          std::lock_guard<std::mutex> lock(mutex);
          if (failure.ok()) {
            failure = absl::Status(status.code(),
                                   absl::StrCat(item.input.filename().string(),
                                                ": ", status.message()));
          }
          stop.store(true, std::memory_order_relaxed);
          return;
        }
        {
          std::lock_guard<std::mutex> lock(mutex);
          std::cout << "Finished " << item.output.filename().string() << "\n";
        }
      }
    });
  }
  for (std::thread& worker : workers) worker.join();
  return failure;
}

}  // namespace
}  // namespace pluto::tokenized

int main(int argc, char* argv[]) {
  const std::vector<char*> positional = absl::ParseCommandLine(argc, argv);
  if (positional.size() != 1) {
    std::cerr << "Unexpected positional arguments. Use --help for usage.\n";
    return 2;
  }
  const absl::Status status = pluto::tokenized::RunConversion();
  if (!status.ok()) {
    std::cerr << status << "\n";
    return 1;
  }
  return 0;
}
