#include "src/llm/experiments/weight_history/history.h"

#include <algorithm>
#include <bit>
#include <charconv>
#include <cmath>
#include <fstream>
#include <limits>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "src/util/status_macros.h"

namespace pluto::llm::weight_history {
namespace {

static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559);

struct WeightFile {
  int64_t id;
  std::filesystem::path path;
  size_t elements;
};

struct Checkpoint {
  int64_t step;
  std::filesystem::path path;
  std::vector<WeightFile> weights;
};

absl::Status FilesystemError(const std::filesystem::path& path,
                             const std::error_code& error) {
  return absl::InternalError(
      absl::StrCat("Cannot inspect ", path.string(), ": ", error.message()));
}

// Nonmatching names (including .tar.gz archives) are unrelated entries.
// Matching but overflowing decimal IDs are errors, not silently skipped.
absl::StatusOr<std::optional<int64_t>> ParseId(absl::string_view name,
                                               absl::string_view prefix,
                                               absl::string_view suffix) {
  if (name.size() <= prefix.size() + suffix.size() ||
      !name.starts_with(prefix) || !name.ends_with(suffix))
    return std::nullopt;
  const auto digits =
      name.substr(prefix.size(), name.size() - prefix.size() - suffix.size());
  if (!std::all_of(digits.begin(), digits.end(),
                   [](char c) { return c >= '0' && c <= '9'; }))
    return std::nullopt;
  int64_t id;
  const auto parsed =
      std::from_chars(digits.data(), digits.data() + digits.size(), id);
  if (parsed.ec != std::errc())
    return absl::OutOfRangeError(
        absl::StrCat("Numeric ID overflows int64: ", name));
  return id;
}

absl::StatusOr<std::vector<WeightFile>> DiscoverWeights(
    const std::filesystem::path& directory) {
  std::error_code error;
  std::filesystem::directory_iterator iterator(directory, error);
  if (error)
    return FilesystemError(directory, error);
  std::vector<WeightFile> weights;
  while (iterator != std::filesystem::directory_iterator()) {
    const auto path = iterator->path();
    ASSIGN_OR_RETURN(auto id,
                     ParseId(path.filename().string(), "weight_", ".bin"));
    if (id.has_value()) {
      const bool regular = iterator->is_regular_file(error);
      if (error)
        return FilesystemError(path, error);
      if (!regular)
        return absl::InvalidArgumentError(
            absl::StrCat("Weight is not a regular file: ", path.string()));
      const uintmax_t bytes = iterator->file_size(error);
      if (error)
        return FilesystemError(path, error);
      if (bytes == 0 || bytes % sizeof(float) != 0)
        return absl::DataLossError(absl::StrCat(
            "Weight must contain a nonempty array of FP32 values: ",
            path.string()));
      const uintmax_t elements = bytes / sizeof(float);
      if (elements > std::vector<float>().max_size())
        return absl::ResourceExhaustedError(
            absl::StrCat("Weight is too large to address: ", path.string()));
      weights.push_back({*id, path, static_cast<size_t>(elements)});
    }
    iterator.increment(error);
    if (error)
      return FilesystemError(directory, error);
  }
  if (weights.empty())
    return absl::NotFoundError(
        absl::StrCat("No weight_<number>.bin files in ", directory.string()));
  std::sort(
      weights.begin(), weights.end(),
      [](const WeightFile& a, const WeightFile& b) { return a.id < b.id; });
  for (size_t index = 1; index < weights.size(); ++index)
    if (weights[index - 1].id == weights[index].id)
      return absl::InvalidArgumentError(
          absl::StrCat("Duplicate numeric weight ID in ", directory.string(),
                       ": ", weights[index - 1].path.filename().string(),
                       " and ", weights[index].path.filename().string()));
  return weights;
}

absl::StatusOr<std::vector<Checkpoint>> Discover(
    const std::filesystem::path& directory) {
  std::error_code error;
  const bool exists = std::filesystem::is_directory(directory, error);
  if (error)
    return FilesystemError(directory, error);
  if (!exists)
    return absl::InvalidArgumentError(absl::StrCat(
        "Not a checkpoint parent directory: ", directory.string()));
  std::filesystem::directory_iterator iterator(directory, error);
  if (error)
    return FilesystemError(directory, error);
  std::vector<Checkpoint> checkpoints;
  while (iterator != std::filesystem::directory_iterator()) {
    const auto path = iterator->path();
    ASSIGN_OR_RETURN(auto step, ParseId(path.filename().string(), "step_", ""));
    if (step.has_value()) {
      const bool child_directory = iterator->is_directory(error);
      if (error)
        return FilesystemError(path, error);
      if (child_directory)
        checkpoints.push_back({*step, path, {}});
    }
    iterator.increment(error);
    if (error)
      return FilesystemError(directory, error);
  }
  if (checkpoints.empty())
    return absl::NotFoundError(absl::StrCat(
        "No step_<number> checkpoint directories in ", directory.string()));
  std::sort(
      checkpoints.begin(), checkpoints.end(),
      [](const Checkpoint& a, const Checkpoint& b) { return a.step < b.step; });
  for (size_t index = 1; index < checkpoints.size(); ++index)
    if (checkpoints[index - 1].step == checkpoints[index].step)
      return absl::InvalidArgumentError(absl::StrCat(
          "Duplicate checkpoint step: ", checkpoints[index - 1].path.string(),
          " and ", checkpoints[index].path.string()));

  // Validate every manifest before expensive tensor reads. Missing or newly
  // added weights must not silently create histories with different meanings.
  for (auto& checkpoint : checkpoints) {
    ASSIGN_OR_RETURN(checkpoint.weights, DiscoverWeights(checkpoint.path));
    const auto& baseline = checkpoints.front().weights;
    if (checkpoint.weights.size() != baseline.size())
      return absl::DataLossError(absl::StrCat("Weight index set differs in ",
                                              checkpoint.path.string()));
    for (size_t index = 0; index < baseline.size(); ++index) {
      if (checkpoint.weights[index].id != baseline[index].id)
        return absl::DataLossError(absl::StrCat("Weight index set differs in ",
                                                checkpoint.path.string()));
      if (checkpoint.weights[index].elements != baseline[index].elements)
        return absl::DataLossError(
            absl::StrCat("Weight byte size changed: ",
                         checkpoint.weights[index].path.string()));
    }
  }
  return checkpoints;
}

absl::StatusOr<TensorHistory> AnalyzeTensor(
    const std::vector<Checkpoint>& checkpoints, size_t weight_index,
    size_t chunk_elements) {
  const auto& baseline = checkpoints.front().weights[weight_index];
  const size_t count = baseline.elements;
  // Keep precisely two full tensors, not one tensor per checkpoint. The
  // previous tensor is overwritten chunk-by-chunk after computing both deltas.
  std::vector<float> first(count);
  std::vector<float> previous(count);
  std::vector<float> chunk(std::min(count, chunk_elements));
  TensorHistory history{baseline.id, count, {}};
  history.samples.reserve(checkpoints.size());
  double previous_l2 = 0;
  for (size_t snapshot = 0; snapshot < checkpoints.size(); ++snapshot) {
    const auto& checkpoint = checkpoints[snapshot];
    const auto& file = checkpoint.weights[weight_index];
    std::ifstream input(file.path, std::ios::binary);
    if (!input)
      return absl::NotFoundError(
          absl::StrCat("Cannot open weight: ", file.path.string()));
    double squared_norm = 0;
    double squared_delta = 0;
    double squared_from_first = 0;
    double max_abs_delta = 0;
    size_t changed = 0;
    for (size_t offset = 0; offset < count;) {
      const size_t length = std::min(chunk.size(), count - offset);
      const auto bytes = static_cast<std::streamsize>(length * sizeof(float));
      if (!input.read(reinterpret_cast<char*>(chunk.data()), bytes))
        return absl::DataLossError(
            absl::StrCat("Short or failed weight read: ", file.path.string()));
      for (size_t index = 0; index < length; ++index) {
        const float value = chunk[index];
        if (!std::isfinite(value))
          return absl::DataLossError(
              absl::StrCat("Nonfinite FP32 value in ", file.path.string(),
                           " at element ", offset + index));
        // Convert BEFORE subtracting or squaring: even two finite floats can
        // overflow a float subtraction. FP64 safely covers FP32 squared sums.
        const double current = value;
        squared_norm += current * current;
        if (snapshot == 0) {
          first[offset + index] = value;
        } else {
          const double delta =
              current - static_cast<double>(previous[offset + index]);
          const double from_first =
              current - static_cast<double>(first[offset + index]);
          squared_delta += delta * delta;
          squared_from_first += from_first * from_first;
          max_abs_delta = std::max(max_abs_delta, std::abs(delta));
          changed += value != previous[offset + index];
        }
        previous[offset + index] = value;
      }
      offset += length;
    }
    // Detect a file growing/shrinking between manifest inspection and reading.
    // Same-size concurrent overwrites cannot be detected: snapshots must be
    // quiescent, as documented by the public API.
    if (input.peek() != std::ifstream::traits_type::eof() || input.bad())
      return absl::DataLossError(absl::StrCat(
          "Weight changed size or failed during read: ", file.path.string()));
    Sample sample{};
    sample.step = checkpoint.step;
    sample.rms = std::sqrt(squared_norm / static_cast<double>(count));
    sample.l2 = std::sqrt(squared_norm);
    sample.from_first_rms =
        std::sqrt(squared_from_first / static_cast<double>(count));
    if (snapshot != 0) {
      sample.delta_l2 = std::sqrt(squared_delta);
      sample.delta_rms = std::sqrt(squared_delta / static_cast<double>(count));
      if (previous_l2 != 0)
        sample.relative_l2 = *sample.delta_l2 / previous_l2;
      sample.max_abs_delta = max_abs_delta;
      sample.changed_fraction =
          static_cast<double>(changed) / static_cast<double>(count);
    }
    previous_l2 = sample.l2;
    history.samples.push_back(std::move(sample));
  }
  return history;
}

}  // namespace

absl::StatusOr<History> AnalyzeDirectory(
    const std::filesystem::path& directory,
    absl::FunctionRef<void(size_t, size_t)> progress, size_t chunk_elements) {
  if constexpr (std::endian::native != std::endian::little)
    return absl::UnimplementedError(
        "This tool reads little-endian FP32 checkpoints");
  if (directory.empty())
    return absl::InvalidArgumentError("Checkpoint parent directory is empty");
  if (chunk_elements == 0 ||
      chunk_elements >
          static_cast<size_t>(std::numeric_limits<std::streamsize>::max()) /
              sizeof(float))
    return absl::InvalidArgumentError(
        "chunk_elements must be positive and fit file I/O");
  std::error_code error;
  const auto absolute = std::filesystem::absolute(directory, error);
  if (error)
    return FilesystemError(directory, error);
  ASSIGN_OR_RETURN(auto checkpoints, Discover(absolute));
  History history;
  history.directory = absolute.string();
  for (const auto& checkpoint : checkpoints)
    history.steps.push_back(checkpoint.step);
  const size_t tensors = checkpoints.front().weights.size();
  history.tensors.reserve(tensors);
  for (size_t index = 0; index < tensors; ++index) {
    ASSIGN_OR_RETURN(auto tensor,
                     AnalyzeTensor(checkpoints, index, chunk_elements));
    history.tensors.push_back(std::move(tensor));
    progress(index + 1, tensors);
  }
  return history;
}

}  // namespace pluto::llm::weight_history
