#pragma once

#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"

namespace pluto::llm::one_shot_memorizer {

// Exponential subset search is intentionally bounded to modest token sets.
inline constexpr int kMaxPositionPathNodes = 16;

enum class PositionPathScoreMode {
  kPositionAware,  // Use a different edge matrix for each position.
  kFixedPosition,  // Repeat one supplied edge matrix at every position.
  kMeanPositions,  // Average supplied matrices, then repeat the mean.
};

// Produces [n, n, n+1] scores for SolvePositionPaths. Fixed-position mode takes
// just [1, n, n+1]; the other modes take [n, n, n+1]. Mean mode averages
// SCORES, not probabilities: averaging log probabilities yields an unnormalized
// geometric-mean score after exponentiation. All entries must be finite;
// malformed modes/shapes and intermediate arithmetic overflow are rejected.
absl::StatusOr<std::vector<double>> PreparePositionPathScores(
    int node_count, absl::Span<const double> scores,
    PositionPathScoreMode mode);

// One ordering of distinct node indices, followed by an implicit terminal.
struct PositionPath {
  double score;  // Sum of position-dependent edges, including terminal.
  std::vector<int>
      nodes;  // Permutation of [0, node_count), excluding terminal.
};

// Returns the best and runner-up distinct Hamiltonian paths, in descending
// score order (only one path exists when node_count is one). Scores have shape
// [node_count, node_count, node_count + 1]: [position, source, destination].
// Destination node_count denotes a terminal, allowed ONLY after the last node.
// Every ordinary node occurs exactly once and all starting nodes have score
// zero. Ties are broken deterministically by the search's fixed index order.
//
// This is a generic CPU solver: it never reads tokens, labels, or a corpus.
// Applying it to a set of suffix tokens ASSUMES no repetitions; neither token
// multiplicities nor missing tokens can be inferred by this search. Self edges
// and terminal edges before the final position do not participate in a path.
// All supplied scores must be finite, including unused entries. Rejects bad
// shapes/counts and arithmetic overflow rather than silently losing a path.
// Time O(2^n * n^2), space O(2^n * n), with n <= kMaxPositionPathNodes.
absl::StatusOr<std::vector<PositionPath>> SolvePositionPaths(
    int node_count, absl::Span<const double> scores);

}  // namespace pluto::llm::one_shot_memorizer
