#include "src/llm/experiments/one_shot_memorizer/margin_projection.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <utility>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "lp_lib.h"
#include "src/util/status_macros.h"

namespace pluto::llm::one_shot_memorizer {
namespace {

using CutKey = std::pair<size_t, int>;  // Observed row and competing token.
using Clock = std::chrono::steady_clock;

struct Violation {
  size_t row;
  int rival;
  double margin;
};

struct Evaluation {
  size_t correct_count = 0;
  double minimum_margin = std::numeric_limits<double>::infinity();
  std::vector<Violation> new_cuts;
};

absl::Status Validate(absl::Span<const float> features, int input_dim,
                      absl::Span<const float> residuals,
                      absl::Span<const int> labels,
                      absl::Span<const double> decoder, int output_dim,
                      int vocab_size, const MarginProjectionOptions& options) {
  if (input_dim <= 0 || output_dim <= 0 || vocab_size < 2 || labels.empty())
    return absl::InvalidArgumentError(
        "margin projection needs positive dimensions, rows and >=2 classes");
  const size_t p = input_dim, q = output_dim, n = labels.size();
  if (n > std::numeric_limits<size_t>::max() / p ||
      n > std::numeric_limits<size_t>::max() / q ||
      static_cast<size_t>(vocab_size) >
          std::numeric_limits<size_t>::max() / q ||
      q >= static_cast<size_t>(std::numeric_limits<int>::max()) - 1 ||
      p > (static_cast<size_t>(std::numeric_limits<int>::max()) - q - 1) / q)
    return absl::OutOfRangeError("margin projection dimensions overflow");
  if (features.size() != n * p || residuals.size() != n * q ||
      decoder.size() != static_cast<size_t>(vocab_size) * q)
    return absl::InvalidArgumentError("malformed margin projection matrices");
  if (!std::isfinite(options.coefficient_bound) ||
      options.coefficient_bound <= 0 || !std::isfinite(options.margin_cap) ||
      options.margin_cap <= 0 || !std::isfinite(options.acceptance_tolerance) ||
      options.acceptance_tolerance <= 0 ||
      options.acceptance_tolerance >= options.margin_cap ||
      options.max_rounds <= 0 ||
      options.max_rounds == std::numeric_limits<int>::max() ||
      options.max_new_cuts == 0 || options.max_total_cuts == 0 ||
      options.max_total_cuts >
          static_cast<size_t>(std::numeric_limits<int>::max()) ||
      options.per_solve_timeout_seconds <= 0)
    return absl::InvalidArgumentError("invalid margin projection options");
  for (const auto values : {features, residuals})
    for (float value : values)
      if (!std::isfinite(value))
        return absl::InvalidArgumentError(
            "margin projection observations must be finite");
  for (double value : decoder)
    if (!std::isfinite(value))
      return absl::InvalidArgumentError(
          "margin projection decoder must be finite");
  if (options.center_coefficients)
    for (int token = 0; token < vocab_size; ++token) {
      double sum = 0, absolute_sum = 0;
      for (size_t dim = 0; dim < q; ++dim) {
        const double value = decoder[static_cast<size_t>(token) * q + dim];
        sum += value;
        absolute_sum += std::abs(value);
      }
      if (!std::isfinite(absolute_sum) ||
          std::abs(sum) > 1e-12 * std::max(1.0, absolute_sum))
        return absl::InvalidArgumentError(
            "centered coefficients require centered decoder score rows");
    }
  for (int label : labels)
    if (label < 0 || label >= vocab_size)
      return absl::InvalidArgumentError("margin projection label out of range");
  return absl::OkStatus();
}

// This oracle does not inspect LP rows or solver residuals. It directly runs
// the proposed affine map and compares its label against the WHOLE decoder.
// Only the worst missing constraint for each failing row becomes a new cut;
// positive but suboptimal margins need not be optimized further for our goal.
absl::StatusOr<Evaluation> Evaluate(
    const MarginProjectionResult& model, absl::Span<const float> features,
    absl::Span<const float> residuals, absl::Span<const int> labels,
    absl::Span<const double> decoder, int vocab_size, double tolerance,
    const absl::flat_hash_set<CutKey>& existing_cuts) {
  Evaluation result;
  const size_t p = model.input_dim, q = model.output_dim;
  std::vector<double> activation(q);
  for (size_t row = 0; row < labels.size(); ++row) {
    for (size_t dim = 0; dim < q; ++dim)
      activation[dim] = residuals[row * q + dim] + model.biases[dim];
    for (size_t feature = 0; feature < p; ++feature)
      for (size_t dim = 0; dim < q; ++dim)
        activation[dim] +=
            features[row * p + feature] * model.weights[feature * q + dim];
    double target_score = 0;
    for (size_t dim = 0; dim < q; ++dim)
      target_score +=
          decoder[static_cast<size_t>(labels[row]) * q + dim] * activation[dim];
    if (!std::isfinite(target_score))
      return absl::OutOfRangeError("margin projection score overflowed");
    double best_rival_score = -std::numeric_limits<double>::infinity();
    int rival = -1;
    for (int token = 0; token < vocab_size; ++token) {
      if (token == labels[row])
        continue;
      double score = 0;
      for (size_t dim = 0; dim < q; ++dim)
        score +=
            decoder[static_cast<size_t>(token) * q + dim] * activation[dim];
      if (!std::isfinite(score))
        return absl::OutOfRangeError("margin projection score overflowed");
      // Scanning in ascending token order makes exact ties deterministic.
      if (score > best_rival_score) {
        best_rival_score = score;
        rival = token;
      }
    }
    const double margin = target_score - best_rival_score;
    if (!std::isfinite(margin))
      return absl::OutOfRangeError("margin projection margin overflowed");
    result.minimum_margin = std::min(result.minimum_margin, margin);
    if (margin > 0 || (margin == 0 && labels[row] < rival))
      ++result.correct_count;
    if (margin <= tolerance && !existing_cuts.contains({row, rival}))
      result.new_cuts.push_back({row, rival, margin});
  }
  std::sort(result.new_cuts.begin(), result.new_cuts.end(),
            [](const Violation& left, const Violation& right) {
              if (left.margin != right.margin)
                return left.margin < right.margin;
              if (left.row != right.row)
                return left.row < right.row;
              return left.rival < right.rival;
            });
  return result;
}

// The coefficient ordering is all row-major W entries, then b, then delta.
// lp_solve columns are one-based, whereas its sparse row arrays are zero-based.
absl::Status AddCut(lprec* lp, const Violation& cut,
                    absl::Span<const float> features, int input_dim,
                    absl::Span<const float> residuals,
                    absl::Span<const int> labels,
                    absl::Span<const double> decoder, int output_dim) {
  const size_t p = input_dim, q = output_dim;
  std::vector<double> difference(q);
  double residual_margin = 0;
  for (size_t dim = 0; dim < q; ++dim) {
    difference[dim] = decoder[static_cast<size_t>(labels[cut.row]) * q + dim] -
                      decoder[static_cast<size_t>(cut.rival) * q + dim];
    residual_margin += difference[dim] * residuals[cut.row * q + dim];
  }
  std::vector<REAL> values;
  std::vector<int> columns;
  values.reserve(p * q + q + 1);
  columns.reserve(p * q + q + 1);
  const double solver_infinity = get_infinite(lp);
  auto append = [&](double value, size_t column) -> absl::Status {
    if (!std::isfinite(value) || std::abs(value) >= solver_infinity)
      return absl::OutOfRangeError(
          "margin constraint exceeds the solver's finite numeric range");
    if (value != 0) {
      values.push_back(value);
      columns.push_back(static_cast<int>(column));
    }
    return absl::OkStatus();
  };
  for (size_t feature = 0; feature < p; ++feature)
    for (size_t dim = 0; dim < q; ++dim)
      RETURN_IF_ERROR(append(features[cut.row * p + feature] * difference[dim],
                             feature * q + dim + 1));
  for (size_t dim = 0; dim < q; ++dim)
    RETURN_IF_ERROR(append(difference[dim], p * q + dim + 1));
  RETURN_IF_ERROR(append(-1, p * q + q + 1));
  if (!std::isfinite(residual_margin) ||
      std::abs(residual_margin) >= solver_infinity)
    return absl::OutOfRangeError(
        "margin constraint RHS exceeds the solver's finite numeric range");
  if (!add_constraintex(lp, static_cast<int>(values.size()), values.data(),
                        columns.data(), GE, -residual_margin))
    return absl::InternalError("lp_solve failed to append margin constraint");
  return absl::OkStatus();
}

// An OPTIMAL status is not itself a numerical certificate. Before using its
// objective as an upper bound, check variable boxes, optional gauge equations,
// and every constraint actually present in the restricted LP. This tolerance
// is only for roundoff in LP feasibility; success still requires the separate
// all-class oracle to exceed the caller's acceptance_tolerance.
bool ValidateLpCandidate(const MarginProjectionResult& candidate,
                         absl::Span<const REAL> variables,
                         absl::Span<const float> features,
                         absl::Span<const float> residuals,
                         absl::Span<const int> labels,
                         absl::Span<const double> decoder,
                         const absl::flat_hash_set<CutKey>& cuts,
                         const MarginProjectionOptions& options) {
  const double objective = candidate.progress.lp_objective;
  const double delta = variables.back();
  if (std::abs(objective - delta) >
          1e-8 * std::max({1.0, std::abs(objective), std::abs(delta)}) ||
      delta > options.margin_cap + 1e-8 * std::max(1.0, options.margin_cap))
    return false;
  const double bound_tolerance =
      1e-8 * std::max(1.0, options.coefficient_bound);
  for (size_t index = 0; index + 1 < variables.size(); ++index)
    if (std::abs(variables[index]) >
        options.coefficient_bound + bound_tolerance)
      return false;
  const size_t p = candidate.input_dim, q = candidate.output_dim;
  if (options.center_coefficients)
    for (size_t feature = 0; feature <= p; ++feature) {
      double sum = 0, absolute_sum = 0;
      for (size_t dim = 0; dim < q; ++dim) {
        const double value = feature == p
                                 ? candidate.biases[dim]
                                 : candidate.weights[feature * q + dim];
        sum += value;
        absolute_sum += std::abs(value);
      }
      if (std::abs(sum) > 1e-8 * std::max(1.0, absolute_sum))
        return false;
    }
  std::vector<double> activation(q);
  for (const auto& [row, rival] : cuts) {
    for (size_t dim = 0; dim < q; ++dim) {
      activation[dim] = residuals[row * q + dim] + candidate.biases[dim];
      for (size_t feature = 0; feature < p; ++feature)
        activation[dim] +=
            features[row * p + feature] * candidate.weights[feature * q + dim];
    }
    double margin = 0, absolute_sum = 0;
    for (size_t dim = 0; dim < q; ++dim) {
      const double term = (decoder[static_cast<size_t>(labels[row]) * q + dim] -
                           decoder[static_cast<size_t>(rival) * q + dim]) *
                          activation[dim];
      margin += term;
      absolute_sum += std::abs(term);
    }
    if (!std::isfinite(margin) || !std::isfinite(absolute_sum) ||
        margin < delta - 1e-8 * std::max({1.0, absolute_sum, std::abs(delta)}))
      return false;
  }
  return true;
}

}  // namespace

const char* MarginProjectionOutcomeName(MarginProjectionOutcome outcome) {
  switch (outcome) {
    case MarginProjectionOutcome::kVerifiedPositiveMargin:
      return "verified_positive_margin";
    case MarginProjectionOutcome::kBoundedNoPositiveMargin:
      return "bounded_no_positive_margin";
    case MarginProjectionOutcome::kBoundedMarginBelowTolerance:
      return "bounded_margin_below_tolerance";
    case MarginProjectionOutcome::kRoundLimit:
      return "round_limit";
    case MarginProjectionOutcome::kCutLimit:
      return "cut_limit";
    case MarginProjectionOutcome::kSolverLimit:
      return "solver_limit";
    case MarginProjectionOutcome::kSolverFailure:
      return "solver_failure";
    case MarginProjectionOutcome::kNumericalMismatch:
      return "numerical_mismatch";
  }
  return "unknown";
}

absl::StatusOr<MarginProjectionResult> FitMarginProjection(
    absl::Span<const float> features, int input_dim,
    absl::Span<const float> residuals, absl::Span<const int> labels,
    absl::Span<const double> decoder_rows, int output_dim, int vocab_size,
    const MarginProjectionOptions& options) {
  RETURN_IF_ERROR(Validate(features, input_dim, residuals, labels, decoder_rows,
                           output_dim, vocab_size, options));
  const auto started = Clock::now();
  const size_t weight_count = static_cast<size_t>(input_dim) * output_dim;
  const int variable_count = static_cast<int>(weight_count + output_dim + 1);
  const size_t gauge_rows = options.center_coefficients ? input_dim + 1 : 0;
  const size_t available_rows =
      static_cast<size_t>(std::numeric_limits<int>::max()) - variable_count;
  if (gauge_rows > available_rows ||
      options.max_total_cuts > available_rows - gauge_rows)
    return absl::OutOfRangeError("margin LP row/column count overflows");
  // This C library needs its own destructor, so a custom-deleter unique_ptr
  // owns the LP. It uses simplex, not gradient descent or iterative SGD.
  std::unique_ptr<lprec, void (*)(lprec*)> lp(make_lp(0, variable_count),
                                              delete_lp);
  if (!lp)
    return absl::ResourceExhaustedError("lp_solve could not allocate a model");
  const double infinity = get_infinite(lp.get());
  if (options.coefficient_bound >= infinity || options.margin_cap >= infinity)
    return absl::InvalidArgumentError(
        "margin projection bounds exceed the solver's finite numeric range");
  MarginProjectionResult result;
  result.input_dim = input_dim;
  result.output_dim = output_dim;
  result.weights.resize(weight_count, 0);
  result.biases.resize(output_dim, 0);
  absl::flat_hash_set<CutKey> cuts;
  auto report = [&](std::string status) {
    result.progress.status = std::move(status);
    result.progress.elapsed_seconds =
        std::chrono::duration<double>(Clock::now() - started).count();
    if (options.progress)
      options.progress(result.progress);
  };
  auto finish = [&](MarginProjectionOutcome outcome) {
    result.outcome = outcome;
    report(MarginProjectionOutcomeName(outcome));
    return result;
  };
  ASSIGN_OR_RETURN(auto evaluation,
                   Evaluate(result, features, residuals, labels, decoder_rows,
                            vocab_size, options.acceptance_tolerance, cuts));
  result.progress.correct_count = evaluation.correct_count;
  result.progress.minimum_margin = evaluation.minimum_margin;
  report("checked_initial_zero_projection");
  if (evaluation.minimum_margin > options.acceptance_tolerance)
    return finish(MarginProjectionOutcome::kVerifiedPositiveMargin);

  set_verbose(lp.get(), NEUTRAL);
  set_maxim(lp.get());
  set_timeout(lp.get(), options.per_solve_timeout_seconds);
  if (options.dual_simplex)
    set_simplextype(lp.get(), SIMPLEX_DUAL_DUAL);
  if (!set_obj(lp.get(), variable_count, 1) ||
      !set_unbounded(lp.get(), variable_count) ||
      !set_upbo(lp.get(), variable_count, options.margin_cap))
    return absl::InternalError("lp_solve could not set margin objective");
  for (int column = 1; column < variable_count; ++column)
    if (!set_bounds(lp.get(), column, -options.coefficient_bound,
                    options.coefficient_bound))
      return absl::InternalError("lp_solve could not set coefficient bounds");
  if (options.center_coefficients) {
    std::vector<REAL> values(output_dim, 1);
    std::vector<int> columns(output_dim);
    for (int feature = 0; feature <= input_dim; ++feature) {
      for (int dim = 0; dim < output_dim; ++dim)
        columns[dim] = feature * output_dim + dim + 1;
      if (!add_constraintex(lp.get(), output_dim, values.data(), columns.data(),
                            EQ, 0))
        return absl::InternalError("lp_solve could not add centering equality");
    }
  }

  std::vector<REAL> variables(variable_count);
  for (int round = 1; round <= options.max_rounds; ++round) {
    if (cuts.size() == options.max_total_cuts)
      return finish(MarginProjectionOutcome::kCutLimit);
    if (evaluation.new_cuts.empty())
      return finish(MarginProjectionOutcome::kNumericalMismatch);
    const size_t add_count =
        std::min({options.max_new_cuts, evaluation.new_cuts.size(),
                  options.max_total_cuts - cuts.size()});
    for (size_t index = 0; index < add_count; ++index) {
      const auto& cut = evaluation.new_cuts[index];
      RETURN_IF_ERROR(AddCut(lp.get(), cut, features, input_dim, residuals,
                             labels, decoder_rows, output_dim));
      cuts.insert({cut.row, cut.rival});
    }
    result.progress.round = round;
    result.progress.cut_count = cuts.size();
    report("solving_restricted_lp");
    // Appending rows after solve is supported. Do not re-enter add-row mode:
    // lp_solve permits that optimization only while initially building an LP.
    const int status = solve(lp.get());
    result.progress.solver_status = status;
    if (status != OPTIMAL) {
      // get_variables is documented as valid only after a successful solve.
      // Preserve the last checked candidate instead of reading timeout debris.
      report(get_statustext(lp.get(), status));
      if (status == TIMEOUT || status == USERABORT || status == SUBOPTIMAL)
        return finish(MarginProjectionOutcome::kSolverLimit);
      return finish(MarginProjectionOutcome::kSolverFailure);
    }
    if (!get_variables(lp.get(), variables.data()))
      return finish(MarginProjectionOutcome::kSolverFailure);
    for (double value : variables)
      if (!std::isfinite(value))
        return finish(MarginProjectionOutcome::kNumericalMismatch);
    const double objective = get_objective(lp.get());
    if (!std::isfinite(objective))
      return finish(MarginProjectionOutcome::kNumericalMismatch);
    MarginProjectionResult candidate = result;
    candidate.progress.lp_objective = objective;
    // Clamp roundoff excursions to the declared box. ValidateLpCandidate
    // rejects larger excursions and checks the resulting gauge/LP residuals.
    for (size_t index = 0; index < weight_count; ++index)
      candidate.weights[index] =
          std::clamp(variables[index], -options.coefficient_bound,
                     options.coefficient_bound);
    for (int dim = 0; dim < output_dim; ++dim)
      candidate.biases[dim] =
          std::clamp(variables[weight_count + dim], -options.coefficient_bound,
                     options.coefficient_bound);
    if (!ValidateLpCandidate(candidate, variables, features, residuals, labels,
                             decoder_rows, cuts, options))
      return finish(MarginProjectionOutcome::kNumericalMismatch);
    result = std::move(candidate);
    ASSIGN_OR_RETURN(evaluation,
                     Evaluate(result, features, residuals, labels, decoder_rows,
                              vocab_size, options.acceptance_tolerance, cuts));
    result.progress.correct_count = evaluation.correct_count;
    result.progress.minimum_margin = evaluation.minimum_margin;
    result.progress.checked_round = round;
    report("checked_candidate_against_all_classes");
    if (evaluation.minimum_margin > options.acceptance_tolerance)
      return finish(MarginProjectionOutcome::kVerifiedPositiveMargin);
    // The restricted LP drops constraints, so its optimum is an upper bound
    // on the full bounded problem. This is a numerical finding, not an exact
    // infeasibility proof, and is never inferred from timeout/error statuses.
    if (result.progress.lp_objective <= 0)
      return finish(MarginProjectionOutcome::kBoundedNoPositiveMargin);
    if (result.progress.lp_objective <= options.acceptance_tolerance)
      return finish(MarginProjectionOutcome::kBoundedMarginBelowTolerance);
  }
  return finish(MarginProjectionOutcome::kRoundLimit);
}

}  // namespace pluto::llm::one_shot_memorizer
