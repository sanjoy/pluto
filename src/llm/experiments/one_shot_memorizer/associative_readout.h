#pragma once

#include <cstddef>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"

namespace pluto::llm::one_shot_memorizer {

enum class ReadoutMode { kMeanDot, kNearestCentroid, kSharedCovarianceLda };

struct ReadoutOptions {
  ReadoutMode mode = ReadoutMode::kMeanDot;
  double ridge = 1e-3;
  // Normalize each input vector before both fitting and prediction. This is
  // feature normalization, not cosine normalization of class prototypes.
  bool normalize_inputs = false;
};

// A conditional decoder fitted to supplied hidden vectors and next-token
// labels. It neither constructs the features from the corpus nor recovers
// the original model's weights. Class means and a small linear solve suffice;
// there are no gradient updates and no frequency prior.
struct AssociativeReadout {
  int width = 0;
  int vocabulary_size = 0;
  ReadoutOptions options;
  size_t training_sample_count = 0;
  size_t singleton_class_count = 0;
  std::vector<size_t> class_counts;
  std::vector<int> seen_classes;  // Sorted; only these classes compete.
  std::vector<float> weights;     // Row-major [vocabulary_size, width].
  std::vector<float> biases;      // Unseen classes have -infinity bias.
};

struct ReadoutEvaluation {
  size_t sample_count = 0;
  size_t correct_count = 0;
  size_t seen_class_count = 0;  // Classes in the complete training sample.
  size_t singleton_target_count = 0;
  size_t unseen_target_count = 0;
  bool leave_one_out = false;
  bool leave_group_out = false;
  double accuracy = 0.0;
  std::vector<int> predictions;
};

// Inputs are row-major [labels.size(), width], finite, and nonzero if input
// normalization is requested. Mean-dot uses w_c=mu_c, b_c=0. Nearest-centroid
// uses w_c=mu_c, b_c=-||mu_c||^2/2. LDA uses w_c=Sigma^-1 mu_c and
// b_c=-mu_c^T Sigma^-1 mu_c/2, with pooled within-class covariance
// Sigma=(sum_i (x_i-mu_label_i)(x_i-mu_label_i)^T)/n + ridge*I.
// This is the maximum-likelihood /n normalization, not / (n-class_count).
// Positive finite ridge is required. Unseen classes are disabled explicitly.
absl::StatusOr<AssociativeReadout> BuildReadout(
    absl::Span<const float> features, absl::Span<const int> labels, int width,
    int vocabulary_size, const ReadoutOptions& options = {});

// Ties select the smallest seen token ID. Unseen targets are counted as
// misses. singleton_target_count uses this model's training class counts.
absl::StatusOr<ReadoutEvaluation> EvaluateReadout(
    const AssociativeReadout& model, absl::Span<const float> features,
    absl::Span<const int> labels);

// Refit mathematically after excluding the query's own training row, for
// every row. Class means, covariance, and normalization denominator all
// exclude that row. A singleton target disappears and counts as an unseen
// miss. Uses sufficient-statistic updates; LDA uses a rank-one inverse update.
// Requires at least two samples. No held-out features enter a fitted mean.
absl::StatusOr<ReadoutEvaluation> EvaluateReadoutLeaveOneOut(
    absl::Span<const float> features, absl::Span<const int> labels, int width,
    int vocabulary_size, const ReadoutOptions& options = {});

// Exclude every row of one group, fit on the remaining groups, and score that
// group's rows. Group IDs must be nonnegative, match labels.size(), and name
// at least two distinct groups; IDs need not be contiguous or ordered.
// Predictions retain original row order. unseen_target_count uses each fold's
// training support; singleton_target_count and seen_class_count refer to the
// full corpus. This excludes groups from readout fitting only, not backbone
// training. Performs a complete non-GD refit for each group.
absl::StatusOr<ReadoutEvaluation> EvaluateReadoutLeaveGroupOut(
    absl::Span<const float> features, absl::Span<const int> labels,
    absl::Span<const int> group_ids, int width, int vocabulary_size,
    const ReadoutOptions& options = {});

}  // namespace pluto::llm::one_shot_memorizer
