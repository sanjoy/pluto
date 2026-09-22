#pragma once

#include <filesystem>
#include <map>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "src/llm/experiments/memorize_general_facts/discretized_model/generator_model.h"

namespace pluto::llm::discretized::generator {

// Relative generated file names and their complete contents. Keeping rendering
// separate from publication lets callers format and inspect an output
// atomically.
using FileMap = std::map<std::string, std::string>;

std::string Literal(absl::string_view bytes);
std::string TokenName(absl::string_view bytes, int compact_id, int eos_token);
std::string Source(
    absl::string_view body, absl::string_view header = "\"tables.h\"",
    absl::string_view description = "", bool vocabulary = false,
    absl::string_view name_space = "pluto::llm::discretized::gen::internal",
    const std::vector<std::string>& extra_headers = {});
std::string TransitionObject(absl::string_view body,
                             absl::string_view interface,
                             absl::string_view factory);

// Validate the source model and render CPU-only C++ with separate inference,
// prompt-encoding, and test-only corpus-verification targets.
absl::StatusOr<FileMap> RenderModel(const SymbolicModel& model,
                                    bool include_state_index = false,
                                    bool compact_transitions = false);

// Publish a fresh directory atomically. Existing paths, including dangling
// symlinks and empty directories, are never replaced.
absl::Status PublishFiles(const FileMap& files,
                          const std::filesystem::path& destination);
absl::Status EmitModel(const SymbolicModel& model,
                       const std::filesystem::path& destination,
                       bool include_state_index = false,
                       bool compact_transitions = false);

}  // namespace pluto::llm::discretized::generator
