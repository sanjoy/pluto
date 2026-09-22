#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "src/llm/experiments/memorize_general_facts/discretized_model/generator_model.h"

namespace pluto::llm::discretized::generator {

// Checks all externally supplied shapes, IDs, boundary types, and memberships.
// Validation does not claim that the supplied model completes its corpus.
absl::Status ValidateModel(const SymbolicModel& model);
// Interns exact BF16 boundary vectors from an in-memory execution trace.
absl::StatusOr<SymbolicModel> BuildModel(
    const ModelMetadata& metadata, const std::vector<ExecutionSample>& samples,
    int expected_samples = -1);
absl::StatusOr<SymbolicModel> RestoreMembership(
    const SymbolicModel& model, const SymbolicModel& original_model);
absl::StatusOr<VerificationResult> EvaluateModel(const SymbolicModel& model);
absl::StatusOr<int> PredictNext(const SymbolicModel& model,
                                const std::vector<int>& tokens);

// Search limits are explicit: exhausting a shortlist never proves minimality.
struct ReductionOptions {
  int neighbors = 4;
  int max_passes = 10;
  std::optional<int64_t> max_attempts;
  int64_t exhaustive_pair_limit = 100000;
  std::function<void(const ReductionProgress&)> progress;
  // Optional cooperative cancellation, checked only between complete trials.
  std::function<bool()> interrupted;
};

// Incremental congruence closure. A rejected merge rolls back every index;
// fixed vocabulary labels and distinct residual boundaries can never merge.
class QuotientReducer {
 public:
  static absl::StatusOr<std::unique_ptr<QuotientReducer>> Create(
      const SymbolicModel& model);
  ~QuotientReducer();
  absl::StatusOr<bool> TryMerge(int first_id, int second_id);
  SymbolicModel Export() const;
  int RootForState(int state_id) const;

  struct Impl;
  const Impl& impl() const { return *impl_; }

 private:
  explicit QuotientReducer(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

// SIGINT and optional cancellation return Cancelled between atomic trials.
absl::StatusOr<SymbolicModel> ReduceModel(const SymbolicModel& model,
                                          const ReductionOptions& options = {});

}  // namespace pluto::llm::discretized::generator
