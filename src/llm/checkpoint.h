#pragma once

#include <filesystem>

#include "absl/status/status.h"
#include "src/cuda/executor.h"
#include "src/llm/layer.h"

namespace pluto::llm {

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
