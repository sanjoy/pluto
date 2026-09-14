#pragma once

#include <cstdint>
#include <filesystem>
#include <iosfwd>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "src/llm/experiments/mlp_automaton/graph.h"

namespace pluto::llm::mlp_automaton {

struct Checkpoint {
  std::filesystem::path directory;
  int64_t step;
};

struct CheckpointSelection {
  bool is_history;
  std::vector<Checkpoint> checkpoints;
};

// A directory containing weight_0.bin denotes one checkpoint, regardless of
// its name. Otherwise, discover direct step_<decimal integer> subdirectories
// and sort numerically. Archives, files, and unrelated names are ignored.
// Numeric aliases and overflowing step IDs are errors. Tensor validation is
// deliberately left to the checkpoint loader. step is unused in single mode.
absl::StatusOr<CheckpointSelection> DiscoverCheckpoints(
    const std::filesystem::path& directory);

struct CheckpointRange {
  int64_t first_step;
  int64_t last_step;
  std::vector<int> mlp_blocks;
};

struct PathHistory {
  std::string bytes;
  std::vector<CheckpointRange> ranges;
};

// Retains full membership histories, but reports only strings sampled in at
// least one checkpoint. Keeping unsampled histories permits a word sampled
// late in training to report its earlier memberships as well.
//
// Ranges combine identical memberships at consecutive ANALYZED checkpoints;
// they make no claim about unsaved/intermediate steps. Absence at an analyzed
// checkpoint breaks a range even when the same block set reappears later.
class HistoryAccumulator {
 public:
  // Steps must be nonnegative and strictly increasing. Both lists use exact
  // bytes, contain unique nonempty texts, and have sorted, unique, nonnegative
  // block IDs. Samples must be present in complete_paths with identical block
  // IDs. Validation failure leaves the accumulator unchanged.
  absl::Status AddCheckpoint(int64_t step,
                             absl::Span<const CombinedPath> complete_paths,
                             absl::Span<const CombinedPath> sampled_paths);

  // Returns sampled histories sorted by exact bytes. May be called repeatedly
  // and does not prevent adding more checkpoints.
  std::vector<PathHistory> Finish() const;

 private:
  struct Entry {
    std::vector<CheckpointRange> ranges;
    int64_t last_seen_step = -1;
    bool sampled = false;
  };
  absl::flat_hash_map<std::string, Entry> entries_;
  int64_t last_step_ = -1;
};

// Text output quotes C-escaped token bytes, followed by indented lines such
// as "Chkpt 100 - 300 — block 3,5,6". Singleton ranges use "Chkpt 100".
absl::Status WriteHistoryText(std::ostream& output,
                              absl::Span<const PathHistory> histories);

// JSON preserves exact bytes in bytes_hex and includes the analyzed step list
// so readers can distinguish observed checkpoints from gaps in save cadence.
absl::Status WriteHistoryJson(std::ostream& output,
                              absl::Span<const PathHistory> histories,
                              absl::Span<const int64_t> analyzed_steps);

}  // namespace pluto::llm::mlp_automaton
