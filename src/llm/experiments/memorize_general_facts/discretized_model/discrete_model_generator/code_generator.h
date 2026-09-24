#pragma once

#include <filesystem>
#include <map>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "src/llm/experiments/memorize_general_facts/discretized_model/discrete_model_generator/captured_model.h"
#include "src/llm/experiments/memorize_general_facts/discretized_model/discrete_model_generator/captured_state_vectors.h"

namespace pluto::llm::discretized::generator {

// Relative generated file names and their complete contents. Keeping rendering
// separate from publication lets callers format and inspect an output
// atomically.
using FileMap = std::map<std::string, std::string>;

// Validate the source model and render CPU-only C++ with separate inference,
// prompt-encoding, and test-only corpus-verification targets.
// When supplied, vectors must contain exactly the original states identified
// by model.states[*].members. They populate print_state, not inference logic.
absl::StatusOr<FileMap> RenderModel(
    const CapturedModel& model, bool include_state_index = false,
    bool compact_transitions = false,
    const CapturedStateVectors* vectors = nullptr);

// Publish a fresh directory atomically. Existing paths, including dangling
// symlinks and empty directories, are never replaced.
absl::Status PublishFiles(const FileMap& files,
                          const std::filesystem::path& destination);
absl::Status EmitModel(const CapturedModel& model,
                       const std::filesystem::path& destination,
                       bool include_state_index = false,
                       bool compact_transitions = false,
                       const CapturedStateVectors* vectors = nullptr);

}  // namespace pluto::llm::discretized::generator
