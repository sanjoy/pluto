#pragma once

#include <array>
#include <cstddef>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"

namespace pluto::llm::one_shot_memorizer {

// One output vector from each attention intervention. Bit zero chooses donor
// routing (Q+K); bit one chooses donor values. Each vector must have the same
// width and contain finite numbers. Outputs may already include GPU rounding.
using AttentionFactorialCorners = std::array<absl::Span<const double>, 4>;

// Finite differences anchored at the recipient corner. The three components
// sum to total (up to ordinary double arithmetic). Their norms are NOT additive
// ownership fractions: directions can oppose each other, and interaction is
// generally nonzero. This calculation does not run or reconstruct attention.
struct AttentionFactorialDecomposition {
  std::vector<double> total;        // Both donor families minus recipient.
  std::vector<double> routing;      // Donor Q+K only minus recipient.
  std::vector<double> values;       // Donor V only minus recipient.
  std::vector<double> interaction;  // Joint change minus separate changes.
};

absl::StatusOr<AttentionFactorialDecomposition> DecomposeAttentionFactorial(
    const AttentionFactorialCorners& corners);

// Ideal real-arithmetic pV accounting, distinct from GPU finite differences.
// Input probabilities are normalized query rows, and values are row-major
// [probability_count,width]. shared_prefix_length identifies exactly equal
// leading value rows; mismatches are rejected rather than approximately
// treated as shared. No clipping or renormalization is performed.
struct IdealAttentionDecomposition {
  std::vector<double> routing;      // (p_donor-p_recipient) V_recipient.
  std::vector<double> values;       // p_recipient (V_donor-V_recipient).
  std::vector<double> interaction;  // Product of both changes.
  std::vector<double> shared_prefix_routing;  // Routing on equal value rows.
};

absl::StatusOr<IdealAttentionDecomposition> DecomposeIdealAttention(
    absl::Span<const float> recipient_probabilities,
    absl::Span<const float> donor_probabilities,
    absl::Span<const float> recipient_values,
    absl::Span<const float> donor_values, size_t width,
    size_t shared_prefix_length);

}  // namespace pluto::llm::one_shot_memorizer
