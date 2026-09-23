#pragma once

#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"

namespace pluto::llm::one_shot_memorizer {

// Diagnostic accounting for one observed residual boundary, not a causal
// effect or a prediction made at that boundary. Boundary zero accounts for
// x_initial; later boundaries account for x_current - x_previous. Every term
// uses the SAME direction and normalization from the final observed state.
struct ResidualMarginContribution {
  std::vector<double> dimension_contributions;  // a[i] * x_or_delta[i].
  double total = 0;  // Sum across dimensions for this boundary.
};

struct TraceReadoutAttribution {
  std::vector<double> embedding_difference;  // Effective E_target - E_rival.
  std::vector<double> direction;  // center(gamma * embedding_difference) / s.
  double final_mean = 0;
  double final_variance = 0;  // Population variance; denominator is width.
  double normalization_denominator = 0;  // s = sqrt(final_variance + epsilon).
  std::vector<ResidualMarginContribution> boundaries;  // Same order as input.
  double beta_contribution = 0;  // beta dot embedding_difference.
  double ideal_margin = 0;  // Literal real-valued final LayerNorm/head margin.
  double accounted_margin = 0;  // Sum of all boundary totals plus beta.
  // accounted - ideal: arithmetic closure error, not a model effect. Zero in
  // exact arithmetic; retained explicitly rather than adjusting a boundary.
  double accounting_residual = 0;
  // Actual final-normalized values times effective embedding differences.
  // These include beta already; do NOT add beta_contribution a second time.
  std::vector<double> actual_dimension_contributions;
  double actual_normalized_margin = 0;  // Sum of actual dimension terms.
  // actual_normalized - ideal. Includes FP32 normalization/statistic/affine
  // arithmetic and BF16 output rounding, not just a BF16 conversion in
  // isolation.
  double normalization_residual = 0;
};

// CPU-only accounting for ONE query position and one target/rival pair. Rows
// are its ordered captured residuals: token+position, then post-attention and
// post-MLP boundaries, ending at the actual final LayerNorm input. Every row
// and all five parameter/output vectors must have the same nonzero width.
// Embeddings and actual_normalized_output are the effective physical BF16
// values expanded to floats, when the model uses BF16; gamma/beta remain FP32.
// The caller supplies those effective values, not unrounded master embeddings.
//
// The direction depends on x_final, so earlier boundary totals are NOT early
// next-token predictions or causal ablation effects. Ideal normalization and
// all accounting use double arithmetic. A GPU head margin can differ from
// actual_normalized_margin through its FP32 dot-product/subtraction arithmetic;
// compare it separately rather than hiding it in the normalization residual.
// Rejects empty/mismatched/nonfinite inputs and nonpositive/nonfinite epsilon.
absl::StatusOr<TraceReadoutAttribution> ComputeTraceReadoutAttribution(
    absl::Span<const absl::Span<const float>> residual_rows,
    absl::Span<const float> gamma, absl::Span<const float> beta,
    absl::Span<const float> target_embedding,
    absl::Span<const float> rival_embedding,
    absl::Span<const float> actual_normalized_output, double epsilon = 1e-5);

// Descriptive accounting of one dense projection along a fixed direction.
// These terms explain the observed arithmetic, not causal effects or which
// facts a neuron owns. In particular, changing a neuron can change downstream
// nonlinear operations and the final readout direction itself.
struct DenseProjectionReadoutAttribution {
  // dot(W2[j, :], direction), also retained when GELU[j] is zero.
  std::vector<double> neuron_directions;
  // GELU[j] * dot(W2[j, :], direction), retaining each term's sign.
  std::vector<double> neuron_contributions;
  double bias_contribution = 0;       // dot(FP32 bias, direction).
  double ideal_directional_sum = 0;   // Sum of neuron terms plus bias.
  double actual_directional_sum = 0;  // dot(captured output, direction).
  // Actual minus ideal. Includes projection accumulation, bias addition, and
  // output rounding; this is not a simulated isolated BF16 rounding error.
  double rounding_residual = 0;
};

// CPU-only accounting for ONE position's dense GELU-to-residual projection.
// gelu_values has [hidden] elements, row_major_weights is [hidden, width], and
// bias, actual_output, and direction each have [width] elements. For BF16
// models, supply the effective BF16 GELU/weight operands and captured BF16
// output expanded to floats, not the unrounded FP32 master weights. Bias is
// the original FP32 parameter. All accounting uses double arithmetic and the
// SAME supplied direction; no layer is rerun or neuron removed here.
// Rejects empty/mismatched/nonfinite inputs and overflowing arithmetic.
absl::StatusOr<DenseProjectionReadoutAttribution>
ComputeDenseProjectionReadoutAttribution(
    absl::Span<const float> gelu_values,
    absl::Span<const float> row_major_weights, absl::Span<const float> bias,
    absl::Span<const float> actual_output, absl::Span<const double> direction);

}  // namespace pluto::llm::one_shot_memorizer
