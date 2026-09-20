#include "src/llm/experiments/memorize_general_facts/checkpoint_validation.h"

#include <algorithm>
#include <string>
#include <system_error>

#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_cat.h"

namespace pluto::llm::memorize_general_facts {
namespace {

bool IsNumberedWeight(const std::string& name) {
  constexpr size_t kPrefixSize = sizeof("weight_") - 1;
  constexpr size_t kSuffixSize = sizeof(".bin") - 1;
  return name.size() > kPrefixSize + kSuffixSize &&
         name.compare(0, kPrefixSize, "weight_") == 0 &&
         name.compare(name.size() - kSuffixSize, kSuffixSize, ".bin") == 0 &&
         std::all_of(name.begin() + kPrefixSize, name.end() - kSuffixSize,
                     [](char c) { return c >= '0' && c <= '9'; });
}

absl::Status FileSystemError(const std::filesystem::path& directory,
                             const std::error_code& error) {
  return absl::InternalError(absl::StrCat(
      "cannot inspect checkpoint ", directory.string(), ": ", error.message()));
}

}  // namespace

absl::Status ValidateExactCheckpointFiles(
    const std::filesystem::path& directory, size_t unique_weight_count) {
  if (directory.empty())
    return absl::InvalidArgumentError("checkpoint directory must not be empty");
  std::error_code error;
  const bool exists = std::filesystem::exists(directory, error);
  if (error) return FileSystemError(directory, error);
  if (!exists)
    return absl::NotFoundError(
        absl::StrCat("checkpoint does not exist: ", directory.string()));
  const bool is_directory = std::filesystem::is_directory(directory, error);
  if (error) return FileSystemError(directory, error);
  if (!is_directory)
    return absl::FailedPreconditionError(
        absl::StrCat("checkpoint is not a directory: ", directory.string()));

  absl::flat_hash_set<std::string> missing;
  for (size_t index = 0; index < unique_weight_count; ++index)
    missing.insert(absl::StrCat("weight_", index, ".bin"));
  std::filesystem::directory_iterator iterator(directory, error);
  if (error) return FileSystemError(directory, error);
  const std::filesystem::directory_iterator end;
  while (iterator != end) {
    const std::string name = iterator->path().filename().string();
    if (IsNumberedWeight(name)) {
      const bool regular = iterator->is_regular_file(error);
      if (error) return FileSystemError(directory, error);
      if (!regular)
        return absl::DataLossError(
            absl::StrCat("checkpoint weight is not a regular file: ", name));
      // Checking canonical names, not only a count, rejects a hole replaced by
      // a higher index or a zero-padded alias such as weight_00.bin.
      if (missing.erase(name) == 0)
        return absl::DataLossError(
            absl::StrCat("unexpected checkpoint weight ", name,
                         "; model requires exactly ", unique_weight_count,
                         " unique weights (check the requested model depth)"));
    }
    iterator.increment(error);
    if (error) return FileSystemError(directory, error);
  }
  if (!missing.empty()) {
    const auto first = std::min_element(missing.begin(), missing.end());
    return absl::DataLossError(absl::StrCat(
        "checkpoint is missing ", missing.size(), " weight files, including ",
        *first, "; model requires exactly ", unique_weight_count,
        " unique weights"));
  }
  return absl::OkStatus();
}

}  // namespace pluto::llm::memorize_general_facts
