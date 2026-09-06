#pragma once

#include <filesystem>

#include "absl/functional/function_ref.h"
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

// Restores the layer's unique weight buffers from the corresponding
// weight_0.bin, weight_1.bin, ... prefix in directory. Every required file and
// byte size is checked before any device buffer is modified. Additional
// higher-index weight files are ignored, allowing a prefix layer such as an
// activation generator to read weights from a complete-model checkpoint.
absl::Status ReadFromDirectory(cuda::Executor& executor, Layer& layer,
                               const std::filesystem::path& directory);

// Tries step_N children from newest to oldest and restores the first valid
// checkpoint. Malformed or concurrently removed candidates invoke
// on_malformed_checkpoint before the previous step is tried. Other failures,
// such as an executor mismatch or CUDA error, are returned immediately.
absl::StatusOr<CheckpointInfo> ReadLatestCheckpoint(
    cuda::Executor& executor, Layer& layer,
    const std::filesystem::path& parent_directory,
    absl::FunctionRef<void(const CheckpointInfo&, const absl::Status&)>
        on_malformed_checkpoint);

}  // namespace pluto::llm
