#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "src/cuda/buffer.h"
#include "src/cuda/executor.h"
#include "src/cuda/page_locked_host_array.h"

namespace pluto::weight_analysis {

// Validate an exact generated prefix. There is no tokenizer, decoding,
// normalization, automatic BOS, or token replacement in this path.
absl::Status ValidateTokenIds(absl::Span<const int32_t> ids, int vocabulary,
                              int context_length);

// Read a nonempty, regular, non-symlink file of little-endian int32 token IDs.
// The small, bounded file is loaded directly into pinned memory for H2D upload.
// Its allocation belongs to executor, which must outlive the returned array.
// Stat checks detect ordinary concurrent replacement/changes; the experiment
// runner separately hashes this file and all checkpoint files before/after.
absl::StatusOr<cuda::PageLockedHostArray<int32_t>> ReadTokenIds(
    cuda::Executor& executor, const std::filesystem::path& path, int vocabulary,
    int context_length);

// Copy exactly one matrix row, not a large prefix of vocabulary logits. Shape,
// row bounds, byte arithmetic and executor ownership are validated before
// queuing anything. A successful return is synchronized and ready to inspect.
absl::StatusOr<cuda::PageLockedHostArray<uint8_t>> ReadSelectedRow(
    cuda::Executor& executor, const cuda::Buffer& buffer, int row, int width,
    size_t element_bytes);

}  // namespace pluto::weight_analysis
