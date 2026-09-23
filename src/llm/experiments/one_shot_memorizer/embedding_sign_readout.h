#pragma once

#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"

namespace pluto::llm::one_shot_memorizer {

// Label-free geometric readout of one row-major embedding matrix or update.
struct EmbeddingSignScores {
  std::vector<double> mean;       // Mean embedding/update vector over all rows.
  std::vector<double> scores;     // Per-row dot product with that mean.
  std::vector<int> selected_ids;  // Ascending row IDs with score strictly < 0.
};

// The delta rule is frozen for prospective single-fact experiments. The
// trained-table-only variants are separate exploratory diagnostics.
struct EmbeddingSignReadout {
  EmbeddingSignScores delta_fp32;    // Score E_trained - E_initial.
  EmbeddingSignScores delta_bf16;    // Round endpoints to BF16, then subtract.
  EmbeddingSignScores trained_fp32;  // Score E_trained without initialization.
  EmbeddingSignScores trained_bf16;  // Score BF16-rounded E_trained.
};

// Applies row dot mean_rows < 0 to each of the four matrices above. Inputs are
// finite FP32 arrays of the same nonempty [vocabulary_size, width] shape. CPU
// calculations use double precision; BF16 variants round each FP32 endpoint
// to nearest, ties to even. Overflowing BF16 conversions are rejected.
//
// Neither labels nor corpus text enter decoding. For one exploratory
// single-fact run, negative rows matched the distinct supervised suffix/EOS
// token set, NOT its order, repetitions, or the prompt. The hypothesis was
// discovered after inspecting labeled groups; this is not a universal fact
// decoder. Non-target vocabulary rows still participate in softmax training.
// Results depend on parameterization: compensating common shifts of token
// and position embeddings can preserve inference while changing these scores.
absl::StatusOr<EmbeddingSignReadout> ComputeEmbeddingSignReadout(
    absl::Span<const float> initial, absl::Span<const float> trained,
    int width);

}  // namespace pluto::llm::one_shot_memorizer
