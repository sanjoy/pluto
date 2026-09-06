#include "src/llm/checkpoint.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <fstream>
#include <limits>
#include <string>
#include <system_error>
#include <unordered_set>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "src/cuda/buffer.h"
#include "src/cuda/executor.h"
#include "src/util/status_macros.h"

namespace pluto::llm {
namespace {

constexpr char kWeightPrefix[] = "weight_";
constexpr char kWeightSuffix[] = ".bin";
constexpr char kStepPrefix[] = "step_";

absl::Status FileSystemError(const char* operation,
                             const std::filesystem::path& path,
                             const std::error_code& error) {
  return absl::InternalError(
      absl::StrCat(operation, " ", path.string(), ": ", error.message()));
}

std::filesystem::path WeightPath(const std::filesystem::path& directory,
                                 size_t index) {
  return directory / absl::StrCat(kWeightPrefix, index, kWeightSuffix);
}

bool IsWeightFile(const std::filesystem::path& path) {
  const std::string name = path.filename().string();
  const size_t prefix_size = sizeof(kWeightPrefix) - 1;
  const size_t suffix_size = sizeof(kWeightSuffix) - 1;
  if (name.size() <= prefix_size + suffix_size ||
      name.compare(0, prefix_size, kWeightPrefix) != 0 ||
      name.compare(name.size() - suffix_size, suffix_size, kWeightSuffix) !=
          0) {
    return false;
  }
  return std::all_of(
      name.begin() + prefix_size, name.end() - suffix_size,
      [](char character) { return character >= '0' && character <= '9'; });
}

absl::StatusOr<int> ParseCheckpointStep(
    const std::filesystem::path& directory) {
  std::filesystem::path name_path = directory;
  while (name_path.filename().empty() && name_path.has_parent_path() &&
         name_path != name_path.root_path()) {
    name_path = name_path.parent_path();
  }
  const std::string name = name_path.filename().string();
  constexpr size_t prefix_size = sizeof(kStepPrefix) - 1;
  if (name.size() <= prefix_size ||
      name.compare(0, prefix_size, kStepPrefix) != 0 ||
      !std::all_of(name.begin() + prefix_size, name.end(), [](char character) {
        return character >= '0' && character <= '9';
      })) {
    return absl::InvalidArgumentError(absl::StrCat(
        "checkpoint directory must be named step_N: ", directory.string()));
  }
  int step;
  if (!absl::SimpleAtoi(name.substr(prefix_size), &step)) {
    return absl::OutOfRangeError(
        absl::StrCat("checkpoint step is too large: ", name));
  }
  return step;
}

absl::Status EnsureWriteDirectory(const std::filesystem::path& directory) {
  if (directory.empty()) {
    return absl::InvalidArgumentError("checkpoint directory must not be empty");
  }
  std::error_code error;
  const bool exists = std::filesystem::exists(directory, error);
  if (error) return FileSystemError("cannot inspect", directory, error);
  if (exists) {
    if (!std::filesystem::is_directory(directory, error)) {
      if (error) return FileSystemError("cannot inspect", directory, error);
      return absl::FailedPreconditionError(absl::StrCat(
          "checkpoint path is not a directory: ", directory.string()));
    }
    return absl::OkStatus();
  }
  if (!std::filesystem::create_directories(directory, error) || error) {
    return FileSystemError("cannot create checkpoint directory", directory,
                           error);
  }
  return absl::OkStatus();
}

absl::Status ValidateReadDirectory(const std::filesystem::path& directory) {
  if (directory.empty()) {
    return absl::InvalidArgumentError("checkpoint directory must not be empty");
  }
  std::error_code error;
  const bool exists = std::filesystem::exists(directory, error);
  if (error) return FileSystemError("cannot inspect", directory, error);
  if (!exists) {
    return absl::NotFoundError(absl::StrCat(
        "checkpoint directory does not exist: ", directory.string()));
  }
  if (!std::filesystem::is_directory(directory, error)) {
    if (error) return FileSystemError("cannot inspect", directory, error);
    return absl::FailedPreconditionError(absl::StrCat(
        "checkpoint path is not a directory: ", directory.string()));
  }
  return absl::OkStatus();
}

std::vector<const Buffer*> UniqueWeights(const Layer& layer) {
  std::unordered_set<const void*> seen;
  std::vector<const Buffer*> result;
  for (const Buffer& weight : layer.weights()) {
    const void* identity = weight.data();
    if (identity == nullptr || seen.insert(identity).second) {
      result.push_back(&weight);
    }
  }
  return result;
}

std::vector<Buffer*> UniqueWeights(Layer& layer) {
  std::unordered_set<void*> seen;
  std::vector<Buffer*> result;
  for (Buffer& weight : layer.weights()) {
    void* identity = weight.data();
    if (identity == nullptr || seen.insert(identity).second) {
      result.push_back(&weight);
    }
  }
  return result;
}

template <class BufferPointer>
absl::Status ValidateWeights(cuda::Executor& executor,
                             const std::vector<BufferPointer>& weights) {
  for (size_t index = 0; index < weights.size(); ++index) {
    if (&weights[index]->executor() != &executor) {
      return absl::InvalidArgumentError(absl::StrCat(
          "checkpoint weight ", index, " belongs to a different executor"));
    }
    if (weights[index]->size_bytes() >
        static_cast<size_t>(std::numeric_limits<std::streamsize>::max())) {
      return absl::ResourceExhaustedError(absl::StrCat(
          "checkpoint weight ", index, " is too large for file I/O"));
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<size_t> CountWeightFiles(
    const std::filesystem::path& directory) {
  std::error_code error;
  std::filesystem::directory_iterator iterator(directory, error);
  if (error) return FileSystemError("cannot list", directory, error);

  size_t count = 0;
  const std::filesystem::directory_iterator end;
  while (iterator != end) {
    const std::filesystem::directory_entry& entry = *iterator;
    if (IsWeightFile(entry.path())) {
      const bool regular = entry.is_regular_file(error);
      if (error) return FileSystemError("cannot inspect", entry.path(), error);
      if (regular) ++count;
    }
    iterator.increment(error);
    if (error) return FileSystemError("cannot list", directory, error);
  }
  return count;
}

absl::Status RemoveStaleWeightFiles(
    const std::filesystem::path& directory,
    const std::unordered_set<std::string>& retained_names) {
  std::error_code error;
  std::filesystem::directory_iterator iterator(directory, error);
  if (error) return FileSystemError("cannot list", directory, error);

  const std::filesystem::directory_iterator end;
  while (iterator != end) {
    const std::filesystem::directory_entry& entry = *iterator;
    const std::string name = entry.path().filename().string();
    if (IsWeightFile(entry.path()) &&
        retained_names.find(name) == retained_names.end()) {
      const bool regular = entry.is_regular_file(error);
      if (error) return FileSystemError("cannot inspect", entry.path(), error);
      if (regular && !std::filesystem::remove(entry.path(), error)) {
        if (error) {
          return FileSystemError("cannot remove stale checkpoint weight",
                                 entry.path(), error);
        }
      }
    }
    iterator.increment(error);
    if (error) return FileSystemError("cannot list", directory, error);
  }
  return absl::OkStatus();
}

absl::StatusOr<std::vector<char>> ReadWeightFile(
    const std::filesystem::path& path, size_t expected_size, size_t index) {
  std::error_code error;
  const bool exists = std::filesystem::exists(path, error);
  if (error) return FileSystemError("cannot inspect", path, error);
  if (!exists) {
    return absl::NotFoundError(
        absl::StrCat("checkpoint weight file is missing: ", path.string()));
  }
  const bool regular = std::filesystem::is_regular_file(path, error);
  if (error) return FileSystemError("cannot inspect", path, error);
  if (!regular) {
    return absl::DataLossError(absl::StrCat(
        "checkpoint weight is not a regular file: ", path.string()));
  }
  const uintmax_t actual_size = std::filesystem::file_size(path, error);
  if (error) return FileSystemError("cannot determine size of", path, error);
  if (actual_size != expected_size) {
    return absl::DataLossError(
        absl::StrCat("checkpoint weight ", index, " has ", actual_size,
                     " bytes; expected ", expected_size));
  }

  std::ifstream input(path, std::ios::binary);
  if (!input) {
    return absl::InternalError(
        absl::StrCat("cannot open checkpoint weight: ", path.string()));
  }
  std::vector<char> contents(expected_size);
  if (expected_size != 0) {
    input.read(contents.data(), static_cast<std::streamsize>(expected_size));
    if (!input) {
      return absl::DataLossError(
          absl::StrCat("cannot read checkpoint weight: ", path.string()));
    }
  }
  if (input.peek() != std::ifstream::traits_type::eof()) {
    return absl::DataLossError(absl::StrCat(
        "checkpoint weight changed while reading: ", path.string()));
  }
  return contents;
}

absl::StatusOr<std::vector<CheckpointInfo>> FindCheckpoints(
    const std::filesystem::path& parent_directory) {
  RETURN_IF_ERROR(ValidateReadDirectory(parent_directory));

  std::error_code error;
  std::filesystem::directory_iterator iterator(parent_directory, error);
  if (error) return FileSystemError("cannot list", parent_directory, error);

  std::vector<CheckpointInfo> checkpoints;
  const std::filesystem::directory_iterator end;
  while (iterator != end) {
    const std::filesystem::directory_entry& entry = *iterator;
    const bool is_directory = entry.is_directory(error);
    if (error) return FileSystemError("cannot inspect", entry.path(), error);
    if (is_directory) {
      absl::StatusOr<int> step = ParseCheckpointStep(entry.path());
      if (step.ok()) {
        checkpoints.push_back(
            CheckpointInfo{.directory = entry.path(), .step = *step});
      } else if (step.status().code() != absl::StatusCode::kInvalidArgument) {
        return step.status();
      }
    }
    iterator.increment(error);
    if (error) return FileSystemError("cannot list", parent_directory, error);
  }

  if (checkpoints.empty()) {
    return absl::NotFoundError(
        absl::StrCat("no step_N checkpoint directories found in ",
                     parent_directory.string()));
  }
  std::sort(checkpoints.begin(), checkpoints.end(),
            [](const CheckpointInfo& left, const CheckpointInfo& right) {
              return left.step > right.step;
            });
  return checkpoints;
}

bool IsMalformedCheckpoint(const absl::Status& status) {
  return status.code() == absl::StatusCode::kDataLoss ||
         status.code() == absl::StatusCode::kNotFound ||
         status.code() == absl::StatusCode::kFailedPrecondition;
}

}  // namespace

absl::StatusOr<CheckpointInfo> InspectCheckpointDirectory(
    const std::filesystem::path& directory) {
  RETURN_IF_ERROR(ValidateReadDirectory(directory));
  ASSIGN_OR_RETURN(const int step, ParseCheckpointStep(directory));
  return CheckpointInfo{.directory = directory, .step = step};
}

absl::StatusOr<CheckpointInfo> FindLatestCheckpoint(
    const std::filesystem::path& parent_directory) {
  ASSIGN_OR_RETURN(auto checkpoints, FindCheckpoints(parent_directory));
  return std::move(checkpoints.front());
}

absl::Status WriteToDirectory(cuda::Executor& executor, const Layer& layer,
                              const std::filesystem::path& directory) {
  RETURN_IF_ERROR(EnsureWriteDirectory(directory));
  const std::vector<const Buffer*> weights = UniqueWeights(layer);
  RETURN_IF_ERROR(ValidateWeights(executor, weights));

  std::vector<std::vector<char>> host_weights;
  host_weights.reserve(weights.size());
  for (const Buffer* weight : weights) {
    host_weights.emplace_back(weight->size_bytes());
    if (weight->size_bytes() == 0) continue;
    RETURN_IF_ERROR(cuda::CudaStatus(
        cudaMemcpyAsync(host_weights.back().data(), weight->data(),
                        weight->size_bytes(), cudaMemcpyDeviceToHost,
                        executor.stream()),
        "cudaMemcpyAsync(checkpoint write)"));
  }
  RETURN_IF_ERROR(executor.Synchronize());

  std::unordered_set<std::string> retained_names;
  for (size_t index = 0; index < weights.size(); ++index) {
    const std::filesystem::path path = WeightPath(directory, index);
    retained_names.insert(path.filename().string());
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) {
      return absl::InternalError(
          absl::StrCat("cannot open checkpoint weight: ", path.string()));
    }
    if (!host_weights[index].empty()) {
      output.write(host_weights[index].data(),
                   static_cast<std::streamsize>(host_weights[index].size()));
    }
    output.close();
    if (!output) {
      return absl::InternalError(
          absl::StrCat("cannot write checkpoint weight: ", path.string()));
    }
  }
  return RemoveStaleWeightFiles(directory, retained_names);
}

absl::Status ReadFromDirectory(cuda::Executor& executor, Layer& layer,
                               const std::filesystem::path& directory) {
  RETURN_IF_ERROR(ValidateReadDirectory(directory));
  const std::vector<Buffer*> weights = UniqueWeights(layer);
  RETURN_IF_ERROR(ValidateWeights(executor, weights));
  ASSIGN_OR_RETURN(const size_t file_count, CountWeightFiles(directory));
  if (file_count < weights.size()) {
    return absl::DataLossError(absl::StrCat(
        "checkpoint contains ", file_count,
        " weight files; layer requires at least ", weights.size()));
  }

  std::vector<std::vector<char>> host_weights;
  host_weights.reserve(weights.size());
  for (size_t index = 0; index < weights.size(); ++index) {
    ASSIGN_OR_RETURN(auto contents,
                     ReadWeightFile(WeightPath(directory, index),
                                    weights[index]->size_bytes(), index));
    host_weights.push_back(std::move(contents));
  }

  for (size_t index = 0; index < weights.size(); ++index) {
    if (weights[index]->size_bytes() == 0) continue;
    RETURN_IF_ERROR(cuda::CudaStatus(
        cudaMemcpyAsync(weights[index]->data(), host_weights[index].data(),
                        weights[index]->size_bytes(), cudaMemcpyHostToDevice,
                        executor.stream()),
        "cudaMemcpyAsync(checkpoint read)"));
  }
  return executor.Synchronize();
}

absl::StatusOr<CheckpointInfo> ReadLatestCheckpoint(
    cuda::Executor& executor, Layer& layer,
    const std::filesystem::path& parent_directory,
    absl::FunctionRef<void(const CheckpointInfo&, const absl::Status&)>
        on_malformed_checkpoint) {
  ASSIGN_OR_RETURN(auto checkpoints, FindCheckpoints(parent_directory));
  absl::Status newest_error;
  for (const CheckpointInfo& checkpoint : checkpoints) {
    absl::Status status =
        ReadFromDirectory(executor, layer, checkpoint.directory);
    if (status.ok()) return checkpoint;
    if (!IsMalformedCheckpoint(status)) return status;
    if (newest_error.ok()) newest_error = status;
    on_malformed_checkpoint(checkpoint, status);
  }
  return newest_error;
}

}  // namespace pluto::llm
