#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>

#include "absl/status/statusor.h"
#include "src/llm/experiments/memorize_general_facts/discretized_model/discrete_model_generator/captured_model.h"
#include "src/llm/experiments/memorize_general_facts/discretized_model/discrete_model_generator/state_vector_hints.h"

namespace pluto::llm::discretized::generator {

// Exhausting a bounded candidate shortlist never proves global minimality.
struct CompactionOptions {
  // Neighbors per ordering coordinate (or neighboring IDs without hints).
  int neighbors = 4;
  int max_passes = 10;
  std::optional<int64_t> max_attempts;
  int64_t exhaustive_pair_limit = 100000;
  std::function<void(const CompactionProgress&)> progress;
  // Cancellation is checked only between complete, atomic compaction trials.
  std::function<bool()> interrupted;
};

// Incremental symbolic congruence closure. Rejected trials roll back every
// index; successful trials preserve boundaries and distinct vocabulary labels.
class StateCompactor {
 public:
  // Hints are copied if supplied; they affect ordering, never correctness.
  static absl::StatusOr<std::unique_ptr<StateCompactor>> Create(
      const CapturedModel& model,
      const StateVectorHints* vector_hints = nullptr);
  ~StateCompactor();
  // Tries a seed pair and every forced downstream combination. Returns false
  // without changing the partition if the required outputs would conflict.
  absl::StatusOr<bool> TryCompact(int first_id, int second_id);
  CapturedModel Export() const;
  int RootForState(int state_id) const;

  struct Impl;
  const Impl& impl() const { return *impl_; }

 private:
  explicit StateCompactor(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

// Produces a new captured model; optional vectors only prioritize trial pairs.
// Each nearest-candidate pass is followed by exact pointwise compaction: inputs
// of the same MLP/head with equal current outputs are interchangeable. This
// respects the same attempt budget and never combines different boundaries.
// SIGINT and cooperative cancellation return Cancelled between atomic trials.
// Ordinary search must run before MLP input/output symbols are paired.
absl::StatusOr<CapturedModel> CompactModel(
    const CapturedModel& model, const CompactionOptions& options = {},
    const StateVectorHints* vector_hints = nullptr);

// Renames within-boundary state IDs to expose identity/affine pointwise maps.
// Does not combine states or modify supported transitions.
// Rejects a model whose MLP input/output symbols have already been paired.
absl::StatusOr<CapturedModel> RelabelMlpOutputs(const CapturedModel& model);

}  // namespace pluto::llm::discretized::generator
