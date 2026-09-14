#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "absl/functional/function_ref.h"
#include "absl/status/statusor.h"

namespace pluto::llm::weight_history {

// One saved snapshot of one tensor. Reductions use double precision even
// though checkpoint elements are FP32. Deltas compare to the previous SAVED
// checkpoint, not necessarily the previous optimizer step.
struct Sample {
  int64_t step;
  double rms;
  double l2;
  double from_first_rms;
  // Absent at the first checkpoint, because there is no observed predecessor.
  std::optional<double> delta_rms;
  std::optional<double> delta_l2;
  // delta_l2 / predecessor.l2; absent whenever the predecessor norm is zero.
  std::optional<double> relative_l2;
  std::optional<double> max_abs_delta;
  // Fraction of numerically unequal coordinates; +0 and -0 compare equal.
  std::optional<double> changed_fraction;
};

struct TensorHistory {
  int64_t weight_id;
  size_t element_count;
  std::vector<Sample> samples;
};

struct History {
  std::string directory;
  std::vector<int64_t> steps;
  std::vector<TensorHistory> tensors;
};

// Reads raw FP32 weight_<number>.bin files from direct step_<number>
// subdirectories. Both indices are sorted numerically; archives/unrelated
// entries are ignored, but duplicate numeric aliases and malformed snapshots
// are errors. All checkpoints must have identical nonempty weight-index sets
// and matching nonzero tensor byte sizes. Sparse weight-index sets are valid.
//
// File contents must remain unchanged throughout analysis. Each file is read
// once, in chunks. Only the first and previous values of the CURRENT tensor
// are retained, plus one chunk and the compact report: no model/GPU is loaded.
// Memory is approximately twice the largest tensor plus chunk_elements * 4.
// No tensor shapes or semantic layer names are inferred from the raw files.
//
// progress(completed_tensors, total_tensors) runs after each completed tensor.
// chunk_elements must be positive; it is exposed for bounded I/O and tests.
absl::StatusOr<History> AnalyzeDirectory(
    const std::filesystem::path& directory,
    absl::FunctionRef<void(size_t, size_t)> progress,
    size_t chunk_elements = 262144);

}  // namespace pluto::llm::weight_history
