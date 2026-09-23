#include "src/llm/experiments/one_shot_memorizer/associative_readout.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <utility>

#include "absl/status/status.h"
#include "src/util/status_macros.h"

namespace pluto::llm::one_shot_memorizer {
namespace {

absl::Status ValidateOptions(int width, int vocabulary_size,
                             const ReadoutOptions& options) {
  if (width <= 0 || vocabulary_size <= 0)
    return absl::InvalidArgumentError("Width and vocabulary must be positive");
  if (!std::isfinite(options.ridge) || options.ridge <= 0)
    return absl::InvalidArgumentError("Ridge must be finite and positive");
  if (options.mode != ReadoutMode::kMeanDot &&
      options.mode != ReadoutMode::kNearestCentroid &&
      options.mode != ReadoutMode::kSharedCovarianceLda)
    return absl::InvalidArgumentError("Unknown readout mode");
  if (static_cast<size_t>(vocabulary_size) >
          std::numeric_limits<size_t>::max() / width ||
      static_cast<size_t>(width) > std::numeric_limits<size_t>::max() / width)
    return absl::OutOfRangeError("Readout dimensions overflow size_t");
  return absl::OkStatus();
}

absl::StatusOr<std::vector<double>> PrepareRows(
    absl::Span<const float> features, absl::Span<const int> labels, int width,
    int vocabulary_size, const ReadoutOptions& options) {
  RETURN_IF_ERROR(ValidateOptions(width, vocabulary_size, options));
  if (labels.empty() ||
      labels.size() > std::numeric_limits<size_t>::max() / width ||
      features.size() != labels.size() * width)
    return absl::InvalidArgumentError("Features must be nonempty [n, width]");
  for (int label : labels)
    if (label < 0 || label >= vocabulary_size)
      return absl::InvalidArgumentError("Label is outside the vocabulary");
  std::vector<double> rows(features.begin(), features.end());
  for (size_t i = 0; i < labels.size(); ++i) {
    double norm2 = 0;
    for (int j = 0; j < width; ++j) {
      const double value = rows[i * width + j];
      if (!std::isfinite(value))
        return absl::InvalidArgumentError("Features must be finite");
      norm2 += value * value;
    }
    if (options.normalize_inputs) {
      if (norm2 == 0 || !std::isfinite(norm2))
        return absl::InvalidArgumentError(
            "Cannot normalize zero/nonfinite norm");
      const double norm = std::sqrt(norm2);
      for (int j = 0; j < width; ++j)
        rows[i * width + j] /= norm;
    }
  }
  return rows;
}

struct Statistics {
  int width;
  int vocabulary_size;
  size_t n;
  std::vector<double> rows;
  std::vector<double> means;
  std::vector<size_t> counts;
  std::vector<int> seen;
  std::vector<double> scatter;
};

absl::StatusOr<Statistics> ComputeStatistics(absl::Span<const float> features,
                                             absl::Span<const int> labels,
                                             int width, int vocabulary_size,
                                             const ReadoutOptions& options) {
  auto rows = PrepareRows(features, labels, width, vocabulary_size, options);
  if (!rows.ok())
    return rows.status();
  Statistics s{
      width,
      vocabulary_size,
      labels.size(),
      std::move(*rows),
      std::vector<double>(static_cast<size_t>(vocabulary_size) * width),
      std::vector<size_t>(vocabulary_size),
      {},
      {}};
  for (size_t i = 0; i < s.n; ++i) {
    const size_t offset = static_cast<size_t>(labels[i]) * width;
    ++s.counts[labels[i]];
    for (int j = 0; j < width; ++j)
      s.means[offset + j] += s.rows[i * width + j];
  }
  for (int token = 0; token < vocabulary_size; ++token) {
    if (s.counts[token] == 0)
      continue;
    s.seen.push_back(token);
    for (int j = 0; j < width; ++j)
      s.means[static_cast<size_t>(token) * width + j] /= s.counts[token];
  }
  if (options.mode == ReadoutMode::kSharedCovarianceLda) {
    s.scatter.resize(static_cast<size_t>(width) * width);
    std::vector<double> residual(width);
    // Center before forming outer products; avoid cancellation of large raw
    // second moments against squared means.
    for (size_t i = 0; i < s.n; ++i) {
      for (int j = 0; j < width; ++j)
        residual[j] = s.rows[i * width + j] -
                      s.means[static_cast<size_t>(labels[i]) * width + j];
      for (int j = 0; j < width; ++j)
        for (int k = 0; k < width; ++k)
          s.scatter[static_cast<size_t>(j) * width + k] +=
              residual[j] * residual[k];
    }
  }
  return s;
}

// Cholesky factorization and triangular solves, all in double precision. The
// returned decoder is still stored as float weights and biases.
absl::StatusOr<std::vector<double>> CovarianceInverse(const Statistics& s,
                                                      size_t denominator,
                                                      double ridge) {
  const size_t d = s.width;
  if (denominator == 0)
    return absl::InvalidArgumentError(
        "Covariance needs a positive sample count");
  std::vector<double> lower(d * d);
  for (size_t i = 0; i < d; ++i) {
    for (size_t j = 0; j <= i; ++j) {
      double value = s.scatter[i * d + j] / denominator + (i == j ? ridge : 0);
      for (size_t k = 0; k < j; ++k)
        value -= lower[i * d + k] * lower[j * d + k];
      if (!std::isfinite(value) || (i == j && value <= 0))
        return absl::FailedPreconditionError(
            "Regularized covariance is not numerically positive definite");
      lower[i * d + j] = i == j ? std::sqrt(value) : value / lower[j * d + j];
    }
  }
  std::vector<double> inverse(d * d), forward(d), backward(d);
  for (size_t column = 0; column < d; ++column) {
    for (size_t i = 0; i < d; ++i) {
      double value = i == column ? 1.0 : 0.0;
      for (size_t j = 0; j < i; ++j)
        value -= lower[i * d + j] * forward[j];
      forward[i] = value / lower[i * d + i];
    }
    for (size_t i = d; i-- > 0;) {
      double value = forward[i];
      for (size_t j = i + 1; j < d; ++j)
        value -= lower[j * d + i] * backward[j];
      backward[i] = value / lower[i * d + i];
      if (!std::isfinite(backward[i]))
        return absl::OutOfRangeError("Covariance inverse overflowed");
      inverse[i * d + column] = backward[i];
    }
  }
  for (size_t i = 0; i < d; ++i)
    for (size_t j = 0; j < i; ++j)
      inverse[i * d + j] = inverse[j * d + i] =
          (inverse[i * d + j] + inverse[j * d + i]) / 2;
  return inverse;
}

double Dot(const double* a, const double* b, int width) {
  double sum = 0;
  for (int j = 0; j < width; ++j)
    sum += a[j] * b[j];
  return sum;
}

void MakeClassWeights(const double* mean, int width, ReadoutMode mode,
                      const std::vector<double>& inverse, double* weights,
                      double* bias) {
  for (int j = 0; j < width; ++j) {
    weights[j] =
        mode == ReadoutMode::kSharedCovarianceLda
            ? Dot(&inverse[static_cast<size_t>(j) * width], mean, width)
            : mean[j];
  }
  *bias = mode == ReadoutMode::kMeanDot ? 0 : -0.5 * Dot(mean, weights, width);
}

absl::Status ValidateModel(const AssociativeReadout& model) {
  RETURN_IF_ERROR(
      ValidateOptions(model.width, model.vocabulary_size, model.options));
  const size_t v = model.vocabulary_size;
  if (model.training_sample_count == 0 || model.class_counts.size() != v ||
      model.biases.size() != v || model.weights.size() != v * model.width ||
      model.seen_classes.empty())
    return absl::InvalidArgumentError(
        "Inconsistent readout dimensions or counts");
  size_t n = 0, seen = 0, singletons = 0;
  for (size_t token = 0; token < v; ++token) {
    if (model.class_counts[token] > std::numeric_limits<size_t>::max() - n)
      return absl::InvalidArgumentError("Readout class count overflow");
    n += model.class_counts[token];
    singletons += model.class_counts[token] == 1;
    if (model.class_counts[token] != 0) {
      if (seen >= model.seen_classes.size() ||
          model.seen_classes[seen++] != static_cast<int>(token) ||
          !std::isfinite(model.biases[token]))
        return absl::InvalidArgumentError("Invalid active classes or bias");
    } else if (model.biases[token] != -std::numeric_limits<float>::infinity()) {
      return absl::InvalidArgumentError(
          "Unseen class must have negative infinity bias");
    }
    for (int j = 0; j < model.width; ++j) {
      const float weight = model.weights[token * model.width + j];
      if (!std::isfinite(weight) ||
          (model.class_counts[token] == 0 && weight != 0))
        return absl::InvalidArgumentError("Invalid decoder weight");
    }
  }
  if (n != model.training_sample_count || seen != model.seen_classes.size() ||
      singletons != model.singleton_class_count)
    return absl::InvalidArgumentError("Readout summary counts disagree");
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<AssociativeReadout> BuildReadout(
    absl::Span<const float> features, absl::Span<const int> labels, int width,
    int vocabulary_size, const ReadoutOptions& options) {
  auto stats =
      ComputeStatistics(features, labels, width, vocabulary_size, options);
  if (!stats.ok())
    return stats.status();
  std::vector<double> inverse;
  if (options.mode == ReadoutMode::kSharedCovarianceLda) {
    auto result = CovarianceInverse(*stats, stats->n, options.ridge);
    if (!result.ok())
      return result.status();
    inverse = std::move(*result);
  }
  AssociativeReadout model;
  model.width = width;
  model.vocabulary_size = vocabulary_size;
  model.options = options;
  model.training_sample_count = stats->n;
  model.class_counts = stats->counts;
  model.seen_classes = stats->seen;
  model.weights.resize(stats->means.size());
  model.biases.assign(vocabulary_size, -std::numeric_limits<float>::infinity());
  std::vector<double> class_weights(width);
  for (int token : stats->seen) {
    model.singleton_class_count += stats->counts[token] == 1;
    double bias;
    const size_t offset = static_cast<size_t>(token) * width;
    MakeClassWeights(&stats->means[offset], width, options.mode, inverse,
                     class_weights.data(), &bias);
    model.biases[token] = static_cast<float>(bias);
    if (!std::isfinite(model.biases[token]))
      return absl::OutOfRangeError("Fitted float bias overflowed");
    for (int j = 0; j < width; ++j) {
      model.weights[offset + j] = static_cast<float>(class_weights[j]);
      if (!std::isfinite(model.weights[offset + j]))
        return absl::OutOfRangeError("Fitted float weight overflowed");
    }
  }
  return model;
}

absl::StatusOr<ReadoutEvaluation> EvaluateReadout(
    const AssociativeReadout& model, absl::Span<const float> features,
    absl::Span<const int> labels) {
  RETURN_IF_ERROR(ValidateModel(model));
  auto rows = PrepareRows(features, labels, model.width, model.vocabulary_size,
                          model.options);
  if (!rows.ok())
    return rows.status();
  ReadoutEvaluation evaluation;
  evaluation.sample_count = labels.size();
  evaluation.seen_class_count = model.seen_classes.size();
  evaluation.predictions.reserve(labels.size());
  for (size_t i = 0; i < labels.size(); ++i) {
    int best_token = -1;
    double best_score = -std::numeric_limits<double>::infinity();
    for (int token : model.seen_classes) {
      double score = model.biases[token];
      for (int j = 0; j < model.width; ++j)
        score += (*rows)[i * model.width + j] *
                 model.weights[static_cast<size_t>(token) * model.width + j];
      if (score > best_score) {
        best_score = score;
        best_token = token;
      }
    }
    evaluation.predictions.push_back(best_token);
    evaluation.correct_count += best_token == labels[i];
    evaluation.singleton_target_count += model.class_counts[labels[i]] == 1;
    evaluation.unseen_target_count += model.class_counts[labels[i]] == 0;
  }
  evaluation.accuracy =
      static_cast<double>(evaluation.correct_count) / labels.size();
  return evaluation;
}

absl::StatusOr<ReadoutEvaluation> EvaluateReadoutLeaveOneOut(
    absl::Span<const float> features, absl::Span<const int> labels, int width,
    int vocabulary_size, const ReadoutOptions& options) {
  auto stats =
      ComputeStatistics(features, labels, width, vocabulary_size, options);
  if (!stats.ok())
    return stats.status();
  if (stats->n < 2)
    return absl::InvalidArgumentError(
        "Leave-one-out requires at least two samples");
  std::vector<double> inverse;
  if (options.mode == ReadoutMode::kSharedCovarianceLda) {
    // A = S/(n-1) + ridge I is shared across all leave-one-out fits. Removing
    // row x of a class with m>1 gives S' = S-m/(m-1)*(x-mu)(x-mu)^T.
    auto result = CovarianceInverse(*stats, stats->n - 1, options.ridge);
    if (!result.ok())
      return result.status();
    inverse = std::move(*result);
  }
  std::vector<double> base_weights(stats->means.size()),
      base_bias(vocabulary_size);
  for (int token : stats->seen) {
    const size_t offset = static_cast<size_t>(token) * width;
    MakeClassWeights(&stats->means[offset], width, options.mode, inverse,
                     &base_weights[offset], &base_bias[token]);
  }
  ReadoutEvaluation evaluation;
  evaluation.sample_count = stats->n;
  evaluation.seen_class_count = stats->seen.size();
  evaluation.leave_one_out = true;
  evaluation.predictions.reserve(stats->n);
  std::vector<double> residual(width), u(width), new_mean(width),
      new_weights(width);
  for (size_t i = 0; i < stats->n; ++i) {
    const int target = labels[i];
    const size_t count = stats->counts[target];
    const double* x = &stats->rows[i * width];
    const double* mean = &stats->means[static_cast<size_t>(target) * width];
    double gamma = 0, x_dot_u = 0, new_bias = 0;
    if (count > 1) {
      for (int j = 0; j < width; ++j) {
        residual[j] = x[j] - mean[j];
        new_mean[j] = mean[j] - residual[j] / (count - 1);
      }
      if (options.mode == ReadoutMode::kSharedCovarianceLda) {
        for (int j = 0; j < width; ++j)
          u[j] = Dot(&inverse[static_cast<size_t>(j) * width], residual.data(),
                     width);
        const double a =
            static_cast<double>(count) / (count - 1) / (stats->n - 1);
        const double denominator =
            1.0 - a * Dot(residual.data(), u.data(), width);
        if (!std::isfinite(denominator) || denominator <= 0)
          return absl::FailedPreconditionError(
              "Leave-one-out covariance downdate is not numerically positive "
              "definite");
        gamma = a / denominator;
        x_dot_u = Dot(x, u.data(), width);
      }
      MakeClassWeights(new_mean.data(), width, options.mode, inverse,
                       new_weights.data(), &new_bias);
    }
    int best_token = -1;
    double best_score = -std::numeric_limits<double>::infinity();
    for (int token : stats->seen) {
      if (token == target && count == 1)
        continue;
      const size_t offset = static_cast<size_t>(token) * width;
      const double* candidate_mean =
          token == target ? new_mean.data() : &stats->means[offset];
      const double* weights =
          token == target ? new_weights.data() : &base_weights[offset];
      double score = Dot(x, weights, width) +
                     (token == target ? new_bias : base_bias[token]);
      if (gamma != 0) {
        const double projection = Dot(candidate_mean, u.data(), width);
        score += gamma * (projection * x_dot_u - 0.5 * projection * projection);
      }
      if (!std::isfinite(score))
        return absl::OutOfRangeError("Leave-one-out decoding score overflowed");
      if (score > best_score) {
        best_score = score;
        best_token = token;
      }
    }
    evaluation.predictions.push_back(best_token);
    evaluation.correct_count += best_token == target;
    evaluation.singleton_target_count += count == 1;
    evaluation.unseen_target_count += count == 1;
  }
  evaluation.accuracy =
      static_cast<double>(evaluation.correct_count) / stats->n;
  return evaluation;
}

absl::StatusOr<ReadoutEvaluation> EvaluateReadoutLeaveGroupOut(
    absl::Span<const float> features, absl::Span<const int> labels,
    absl::Span<const int> group_ids, int width, int vocabulary_size,
    const ReadoutOptions& options) {
  // Validate the full input before constructing or evaluating any folds.
  auto validated =
      PrepareRows(features, labels, width, vocabulary_size, options);
  if (!validated.ok())
    return validated.status();
  if (group_ids.size() != labels.size())
    return absl::InvalidArgumentError("Group IDs must match the sample count");
  std::map<int, std::vector<size_t>> groups;
  std::vector<size_t> full_counts(vocabulary_size);
  for (size_t i = 0; i < labels.size(); ++i) {
    if (group_ids[i] < 0)
      return absl::InvalidArgumentError("Group IDs must be nonnegative");
    groups[group_ids[i]].push_back(i);
    ++full_counts[labels[i]];
  }
  if (groups.size() < 2)
    return absl::InvalidArgumentError(
        "Leave-group-out requires at least two groups");
  ReadoutEvaluation evaluation;
  evaluation.sample_count = labels.size();
  evaluation.leave_group_out = true;
  evaluation.predictions.resize(labels.size(), -1);
  for (size_t count : full_counts)
    evaluation.seen_class_count += count != 0;
  for (int label : labels)
    evaluation.singleton_target_count += full_counts[label] == 1;
  for (const auto& [held_group, held_rows] : groups) {
    std::vector<float> train_x, query_x;
    std::vector<int> train_y, query_y;
    train_x.reserve(features.size() - held_rows.size() * width);
    train_y.reserve(labels.size() - held_rows.size());
    query_x.reserve(held_rows.size() * width);
    query_y.reserve(held_rows.size());
    for (size_t i = 0; i < labels.size(); ++i) {
      auto& x = group_ids[i] == held_group ? query_x : train_x;
      auto& y = group_ids[i] == held_group ? query_y : train_y;
      x.insert(x.end(), features.begin() + i * width,
               features.begin() + (i + 1) * width);
      y.push_back(labels[i]);
    }
    auto readout =
        BuildReadout(train_x, train_y, width, vocabulary_size, options);
    if (!readout.ok())
      return readout.status();
    auto fold = EvaluateReadout(*readout, query_x, query_y);
    if (!fold.ok())
      return fold.status();
    evaluation.correct_count += fold->correct_count;
    evaluation.unseen_target_count += fold->unseen_target_count;
    for (size_t j = 0; j < held_rows.size(); ++j)
      evaluation.predictions[held_rows[j]] = fold->predictions[j];
  }
  evaluation.accuracy =
      static_cast<double>(evaluation.correct_count) / labels.size();
  return evaluation;
}

}  // namespace pluto::llm::one_shot_memorizer
