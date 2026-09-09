#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <random>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"

namespace pluto::weight_analysis {

// Small, CPU-only specifications of production Generate/Predict behavior.
// They do not tokenize, train, mutate weights, or select a candidate word.
absl::Status ValidateGenerationOptions(int steps, double temperature, int seed);

struct PredictionWindow {
  size_t start;
  size_t length;
  size_t output_row;
};

// Crop only the model input, never the retained token history. The final
// context_length IDs start at absolute position zero in the forward pass;
// unused suffix rows repeat the last input ID, exactly as production Predict.
// The output span is caller-owned (pinned storage in the actual GPU binary).
absl::StatusOr<PredictionWindow> FillPredictionInput(
    absl::Span<const int> history, int context_length, int vocab_size,
    absl::Span<int> output);

struct SamplingDecision {
  int token_id;
  float raw_logit;
  int raw_rank;  // One-based descending logit order; lower ID wins exact ties.
  int raw_argmax_token_id;
  double sampled_probability;
  double uniform;
  double cdf_lower;
  double cdf_upper;
  std::array<uint32_t, 2> rng_words;
};

// EXACT production sampling order: subtract the maximum in FP32, divide by
// the double temperature, exp in double, then std::discrete_distribution<int>.
// The actual selected ID always comes from that standard distribution.
//
// On a CLONE of the initial RNG, generate_canonical<double,53> reconstructs the
// uniform. A second clone records its two MT19937 words. Their final states
// must both match the actual distribution's final RNG state. The chosen ID
// must equal lower_bound of the reconstructed normalized CDF. Fail closed if
// a C++ library changes either convention, rather than guess its random draw.
absl::StatusOr<SamplingDecision> SampleProductionLogits(
    absl::Span<const float> logits, double temperature, std::mt19937& random);

}  // namespace pluto::weight_analysis
