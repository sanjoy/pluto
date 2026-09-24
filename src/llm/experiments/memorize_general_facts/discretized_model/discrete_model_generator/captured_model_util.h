#pragma once

#include <cstdint>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "src/llm/experiments/memorize_general_facts/discretized_model/discrete_model_generator/captured_model.h"
#include "src/llm/experiments/memorize_general_facts/discretized_model/discrete_model_generator/state_vector_hints.h"

namespace pluto::llm::discretized::generator {

// Temporary native observations consumed by BuildModel. Vectors are indexed
// [residual boundary][real token position][channel], with no padded rows.
struct ExecutionSample {
  std::vector<int> tokens;
  std::vector<int> predictions;
  std::vector<std::vector<std::vector<uint16_t>>> boundaries;
  auto operator<=>(const ExecutionSample&) const = default;
};

// Checks all externally supplied shapes, IDs, boundary types, and memberships.
// Validation does not claim that the supplied model completes its corpus.
absl::Status ValidateModel(const CapturedModel& model);
// Interns exact BF16 boundary vectors into purely symbolic observations. When
// requested, vector_hints receives every original state's exact vector on
// success only. Singleton original memberships survive subsequent renaming.
absl::StatusOr<CapturedModel> BuildModel(
    const ModelMetadata& metadata, const std::vector<ExecutionSample>& samples,
    int expected_samples = -1, StateVectorHints* vector_hints = nullptr);
absl::StatusOr<CapturedModel> RestoreMembership(
    const CapturedModel& model, const CapturedModel& original_model);
absl::StatusOr<VerificationResult> EvaluateModel(const CapturedModel& model);
absl::StatusOr<int> PredictNext(const CapturedModel& model,
                                const std::vector<int>& tokens);

}  // namespace pluto::llm::discretized::generator
