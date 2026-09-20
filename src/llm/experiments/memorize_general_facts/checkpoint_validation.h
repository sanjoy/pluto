#pragma once

#include <cstddef>
#include <filesystem>

#include "absl/status/status.h"

namespace pluto::llm::memorize_general_facts {

// Require exactly weight_0.bin through weight_(unique_weight_count - 1).bin.
// The caller must count unique weight allocations in the actual model, so a
// tied embedding/head contributes once. Other, non-weight metadata is allowed.
// File sizes and contents are still checked by ReadFromDirectory afterwards.
//
// This stricter experiment-only check prevents a shallower verification model
// from silently loading a larger model's prefix. The generic checkpoint reader
// intentionally permits prefixes for activation generators; that behavior is
// inappropriate when certifying the complete trained model. No GPU work or
// checkpoint mutation occurs here.
absl::Status ValidateExactCheckpointFiles(
    const std::filesystem::path& directory, size_t unique_weight_count);

}  // namespace pluto::llm::memorize_general_facts
