#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"

namespace pluto::llm::one_shot_memorizer {

inline constexpr size_t kProjectedReluWindow = 9;
inline constexpr size_t kProjectedReluPrompt = 5;

// Three actual FP64 neural weights per memory unit, not a retained text key.
struct ProjectedReluUnit {
  double negative_bias = 0;  // -h_i in ReLU(h - h_i).
  double positive_bias = 0;  // +h_i in ReLU(-h + h_i).
  double output_weight = 0;  // Scalar next-token ID, including possible EOS.
};

// CPU-only, wide/high-precision construction, NOT the learned GPT-2
// architecture. Project nine suffix IDs (left padding -1) to one integer h.
// Every unit runs u_i = ReLU(1-ReLU(h-h_i)-ReLU(h_i-h)); output = sum_i
// u_i*output_weight_i. At integer h these are exact zero/one indicators.
// Construction proves unique hashes only for distinct CORPUS keys. An unseen
// suffix can alias a stored hash and will be accepted: this does NOT provide
// the full-key ReLU memory's unsupported-context guarantee. No original keys
// are retained for inference.
struct ProjectedReluMemory {
  int vocabulary_size = 0;  // IDs lie in [0,vocabulary_size), at most 65536.
  int eos_token_id = -1;    // EOS may be a target, never an input token.
  uint32_t projection_attempts = 0;  // Builder draws tried, including success.
  // Integer FP64 weights for the nine suffix coordinates.
  std::array<double, kProjectedReluWindow> projection{};
  std::vector<ProjectedReluUnit> units;  // Strict ascending order of +h_i.
};

// Fixed protocol: prompt five, suffix nine, SplitMix64 seed 0; draw nine
// integer coefficients in [1,2^20], retry up to 256 times after ANY
// distinct-key hash collision, including equal-target collisions. Merge
// identical key/target pairs and reject conflicting labels. No checkpoints or
// fitted activations.
absl::StatusOr<ProjectedReluMemory> BuildProjectedReluMemory(
    const std::vector<std::vector<int>>& sentences, int vocabulary_size,
    int eos_token_id);

// Checks the selected seed-0 draw, finite integer weights, ranges proving exact
// FP64 arithmetic, opposite biases, valid target IDs and sorted distinct
// hashes. Original text keys are absent, so this cannot certify that earlier
// draws collided or validate unseen text keys from the intentionally lossy
// projection.
absl::Status ValidateProjectedReluMemory(const ProjectedReluMemory& model);

// Validates all input IDs and weights, then evaluates EVERY unit's ReLUs and
// output contribution. No map/search selects a unit. NotFound means no hash
// match. Success does NOT prove the full suffix appeared in the corpus.
absl::StatusOr<int> ProjectedReluNextToken(const ProjectedReluMemory& model,
                                           absl::Span<const int> prefix);

// Validates once, feeds back its own predictions, includes EOS on termination.
// A positive exhausted limit returns the partial continuation without EOS.
absl::StatusOr<std::vector<int>> ProjectedReluGreedyContinuation(
    const ProjectedReluMemory& model, absl::Span<const int> prefix,
    size_t max_new_tokens);

// Portable little-endian numeric weights: explicit opposite FP64 biases,
// output weights and projection coefficients, plus fixed-shape metadata.
// Loader rejects truncation, trailing bytes, bad shapes and invalid weights;
// dimensions are checked against the byte extent before allocating memory.
absl::StatusOr<std::string> SerializeProjectedReluMemory(
    const ProjectedReluMemory& model);
absl::StatusOr<ProjectedReluMemory> DeserializeProjectedReluMemory(
    absl::string_view bytes);

}  // namespace pluto::llm::one_shot_memorizer
