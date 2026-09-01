#pragma once

#include <filesystem>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "src/cuda/executor.h"
#include "src/llm/layer.h"

namespace pluto::llm {

struct CheckpointInfo {
  std::filesystem::path directory;
  int step;
};

// Validates that directory exists and is named step_N, then returns N.
absl::StatusOr<CheckpointInfo> InspectCheckpointDirectory(
    const std::filesystem::path& directory);

// Finds the direct child directory named step_N with the numerically largest N.
// Unrelated entries are ignored. Returns NotFound when no checkpoint exists.
absl::StatusOr<CheckpointInfo> FindLatestCheckpoint(
    const std::filesystem::path& parent_directory);

// Writes the layer's unique weight buffers as weight_0.bin, weight_1.bin, ...
// in stable weights() order. Tied buffers are written only on first occurrence.
// Existing checkpoint weight files are replaced; unrelated files are retained.
absl::Status WriteToDirectory(cuda::Executor& executor, const Layer& layer,
                              const std::filesystem::path& directory);

// Restores the layer's unique weight buffers from directory. The complete file
// count and every byte size are checked before any device buffer is modified.
// The caller must supply the same layer topology used to write the checkpoint.
absl::Status ReadFromDirectory(cuda::Executor& executor, Layer& layer,
                               const std::filesystem::path& directory);

}  // namespace pluto::llm
