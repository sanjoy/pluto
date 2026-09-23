#pragma once

#include "absl/status/statusor.h"

namespace pluto::llm::one_shot_memorizer {

// Hyperparameters of a two-update, zero-decay Adam replay. The first rate
// cancels from the comparison; both component runs retain both clock slots.
struct TwoStepAdamConfig {
  double beta1 = 0.9f;     // First-moment retention, promoted from FP32.
  double beta2 = 0.99f;    // Second-moment retention, promoted from FP32.
  double epsilon = 1e-8f;  // Added after the bias-corrected RMS square root.
  double second_rate = 0;  // Learning rate for the second update.
};

// Coordinate-wise difference between Adam([a,b]) and the summed parameter
// changes of Adam([a,0]) and Adam([0,b]), starting with zero moments. The two
// signed terms can cancel; their norms are not additive fractions of an effect.
struct TwoStepAdamInteraction {
  double a_denominator = 0;      // A-only bias-corrected RMS plus epsilon.
  double b_denominator = 0;      // B-only bias-corrected RMS plus epsilon.
  double joint_denominator = 0;  // Shared-history RMS plus epsilon.
  double a_term = 0;             // Effect of rescaling A's surviving momentum.
  double b_term = 0;             // Effect of rescaling B's new gradient.
  double difference = 0;         // Frozen joint weights minus summed weights.
};

// Closed form over ideal real arithmetic, evaluated in FP64. This is not a
// bitwise emulator of production FP32 Adam or of the final FP32 SUM cast.
// Inputs are fixed gradient coordinates, not weights or recomputed gradients.
// Rejects nonfinite inputs, invalid hyperparameters, and nonfinite results.
absl::StatusOr<TwoStepAdamInteraction> ExplainTwoStepAdamInteraction(
    float a, float b, const TwoStepAdamConfig& config);

}  // namespace pluto::llm::one_shot_memorizer
