#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "nlohmann/json.hpp"

namespace pluto::llm::discretized::generator {

// Structured in-memory generator values. This representation never becomes a
// capture or model file; generated runtime code has no dependency on it.
using Json = nlohmann::json;

absl::StatusOr<std::string> ReadFile(const std::filesystem::path& path);
absl::Status WriteFile(const std::filesystem::path& path,
                       absl::string_view data);
// Deterministic, human-readable key/value report with indented subsections.
std::string TextReport(const Json& value);

std::string Sha256(absl::string_view data);
absl::StatusOr<std::string> Sha256File(const std::filesystem::path& path);
absl::StatusOr<std::filesystem::path> FindExecutable(absl::string_view name);

// Launch directly, without interpreting paths or arguments as shell syntax.
absl::Status RunProcess(const std::vector<std::string>& arguments);

}  // namespace pluto::llm::discretized::generator
