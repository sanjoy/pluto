#pragma once

#include <cstddef>
#include <functional>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"

namespace pluto::llm::one_shot_memorizer {

// These outcomes describe a bounded numerical experiment, not whether any
// architecture could represent the corpus. Only kVerifiedPositiveMargin is a
// successful construction, and its certificate is for real-valued CPU scores;
// callers must additionally test the actual rounded GPU implementation.
enum class MarginProjectionOutcome {
  kVerifiedPositiveMargin,
  kBoundedNoPositiveMargin,
  kBoundedMarginBelowTolerance,
  kRoundLimit,
  kCutLimit,
  kSolverLimit,
  kSolverFailure,
  kNumericalMismatch,
};

const char* MarginProjectionOutcomeName(MarginProjectionOutcome outcome);

struct MarginProjectionProgress {
  int round = 0;  // Latest LP attempt; zero precedes the first solve.
  // Source of the reported coefficients/measurements. May lag round after a
  // timeout or numerical failure; zero is the initial zero projection.
  int checked_round = 0;
  size_t cut_count = 0;
  size_t correct_count = 0;   // Top-1 over the entire supplied vocabulary.
  double minimum_margin = 0;  // Minimum target-minus-best-rival score.
  double lp_objective = 0;    // Last checked candidate's LP objective, or zero.
  int solver_status = -1;     // Latest attempt's lp_solve status; initially -1.
  std::string status;         // Human-readable solver/termination description.
  double elapsed_seconds = 0;
};

struct MarginProjectionOptions {
  double coefficient_bound = 1;  // Every weight and bias lies in [-B, B].
  double margin_cap = 1;         // Upper bound on the maximized common margin.
  double acceptance_tolerance = 1e-7;  // Require every margin > this value.
  // Constrain each output coefficient vector to sum to zero; decoder rows must
  // be centered. This removes score-invisible common mode, but with a finite
  // coefficient box it is an additional restriction on the searched model.
  bool center_coefficients = false;
  // Use dual simplex in both phases, instead of lp_solve's dual/primal mix.
  // This changes search strategy only, not the bounded feasible region.
  bool dual_simplex = false;
  int max_rounds = 30;
  size_t max_new_cuts = 1024;  // Worst unrepresented row/rival pairs per round.
  size_t max_total_cuts = 30000;
  int per_solve_timeout_seconds = 60;  // Positive; zero is not unlimited.
  std::function<void(const MarginProjectionProgress&)> progress;
};

struct MarginProjectionResult {
  MarginProjectionOutcome outcome = MarginProjectionOutcome::kSolverFailure;
  int input_dim = 0;
  int output_dim = 0;
  std::vector<double> weights;        // Row-major [input_dim, output_dim].
  std::vector<double> biases;         // [output_dim].
  MarginProjectionProgress progress;  // Describes this returned candidate.
};

// Constructs W,b from frozen features, incoming residuals and token labels;
// neither teacher MLP outputs nor the original projection are inputs. For each
// row i and EVERY other vocabulary class j, require
//   (decoder[y_i]-decoder[j]) dot (residual[i]+features[i]*W+b) >= delta.
// The score rows already encode any desired centered LayerNorm scale/embedding
// product. This linear score rule requires zero final LayerNorm bias.
//
// Maximizes delta with a simplex LP and deterministic cutting planes, starting
// from W=b=0. Delta has no restrictive lower bound. Each candidate is checked
// independently against every row and every vocabulary class, including tokens
// absent from labels. Expected limits/numerical failures are explicit outcomes,
// not infeasibility claims. A numerical restricted-LP optimum <= 0 is reported
// as kBoundedNoPositiveMargin, scoped only to this coefficient bound and model.
// A positive restricted optimum <= acceptance_tolerance instead produces
// kBoundedMarginBelowTolerance: that is insufficient robustness, not a failure
// of strict class separation or a solver numerical error.
//
// Matrices are row-major features[n,input_dim], residuals[n,output_dim], and
// decoder_rows[vocab_size,output_dim]; labels[n] must be in [0,vocab_size).
absl::StatusOr<MarginProjectionResult> FitMarginProjection(
    absl::Span<const float> features, int input_dim,
    absl::Span<const float> residuals, absl::Span<const int> labels,
    absl::Span<const double> decoder_rows, int output_dim, int vocab_size,
    const MarginProjectionOptions& options = {});

}  // namespace pluto::llm::one_shot_memorizer
