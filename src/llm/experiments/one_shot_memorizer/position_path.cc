#include "src/llm/experiments/one_shot_memorizer/position_path.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>

#include "absl/status/status.h"

namespace pluto::llm::one_shot_memorizer {
namespace {

struct Candidate {
  // Only this internal sentinel is nonfinite; valid path sums must be finite.
  double score = -std::numeric_limits<double>::infinity();
  // Twice the predecessor node plus its rank (0 or 1); root has no predecessor.
  uint16_t previous = std::numeric_limits<uint16_t>::max();
};
using TwoBest = std::array<Candidate, 2>;

void Insert(TwoBest& best, Candidate candidate) {
  if (candidate.score > best[0].score) {
    best[1] = best[0];
    best[0] = candidate;
  } else if (candidate.score > best[1].score) {
    best[1] = candidate;
  }
}

}  // namespace

absl::StatusOr<std::vector<double>> PreparePositionPathScores(
    int node_count, absl::Span<const double> scores,
    PositionPathScoreMode mode) {
  const int n = node_count;
  if (n < 1 || n > kMaxPositionPathNodes)
    return absl::InvalidArgumentError("node_count must be in [1, 16]");
  if (mode != PositionPathScoreMode::kPositionAware &&
      mode != PositionPathScoreMode::kFixedPosition &&
      mode != PositionPathScoreMode::kMeanPositions)
    return absl::InvalidArgumentError("unknown position score mode");
  const size_t plane = static_cast<size_t>(n) * (n + 1);
  const int input_positions =
      mode == PositionPathScoreMode::kFixedPosition ? 1 : n;
  if (scores.size() != plane * input_positions)
    return absl::InvalidArgumentError("score shape differs from selected mode");
  for (double score : scores)
    if (!std::isfinite(score))
      return absl::InvalidArgumentError("all edge scores must be finite");
  if (mode == PositionPathScoreMode::kPositionAware)
    return std::vector<double>(scores.begin(), scores.end());
  std::vector<double> result(plane * n);
  for (size_t edge = 0; edge < plane; ++edge) {
    double sum = 0;
    for (int position = 0; position < input_positions; ++position) {
      sum += scores[position * plane + edge];
      if (!std::isfinite(sum))
        return absl::OutOfRangeError("mean edge score overflowed");
    }
    const double value = sum / input_positions;
    for (int position = 0; position < n; ++position)
      result[position * plane + edge] = value;
  }
  return result;
}

absl::StatusOr<std::vector<PositionPath>> SolvePositionPaths(
    int node_count, absl::Span<const double> scores) {
  const int n = node_count;
  if (n < 1 || n > kMaxPositionPathNodes)
    return absl::InvalidArgumentError("node_count must be in [1, 16]");
  if (scores.size() != static_cast<size_t>(n) * n * (n + 1))
    return absl::InvalidArgumentError("scores must have shape [n, n, n + 1]");
  for (double score : scores)
    if (!std::isfinite(score))
      return absl::InvalidArgumentError("all edge scores must be finite");
  const auto edge = [&](int position, int source, int destination) {
    return scores[(position * n + source) * (n + 1) + destination];
  };
  const unsigned mask_count = 1u << n;
  std::vector<TwoBest> table(static_cast<size_t>(mask_count) * n);
  for (int first = 0; first < n; ++first)
    table[static_cast<size_t>(1u << first) * n + first][0].score = 0;

  // A state records the two best distinct paths using exactly mask and ending
  // at last. Future scores depend only on this state, so two prefixes suffice
  // to recover the global two best paths. A path has exactly one predecessor,
  // preventing the same ordering from occupying both ranks under a score tie.
  for (unsigned mask = 1; mask < mask_count; ++mask) {
    const int position = std::popcount(mask) - 1;
    if (position == n - 1)
      continue;
    for (int last = 0; last < n; ++last) {
      if (!(mask & (1u << last)))
        continue;
      for (int rank = 0; rank < 2; ++rank) {
        const Candidate current =
            table[static_cast<size_t>(mask) * n + last][rank];
        if (!std::isfinite(current.score))
          continue;
        for (int next = 0; next < n; ++next) {
          if (mask & (1u << next))
            continue;
          const double score = current.score + edge(position, last, next);
          if (!std::isfinite(score))
            return absl::OutOfRangeError("path score overflowed");
          const unsigned next_mask = mask | (1u << next);
          Insert(table[static_cast<size_t>(next_mask) * n + next],
                 {score, static_cast<uint16_t>(2 * last + rank)});
        }
      }
    }
  }
  const unsigned all = mask_count - 1;
  TwoBest endings;
  for (int last = 0; last < n; ++last)
    for (int rank = 0; rank < 2; ++rank) {
      const Candidate current =
          table[static_cast<size_t>(all) * n + last][rank];
      if (!std::isfinite(current.score))
        continue;
      const double score = current.score + edge(n - 1, last, n);
      if (!std::isfinite(score))
        return absl::OutOfRangeError("terminal path score overflowed");
      Insert(endings, {score, static_cast<uint16_t>(2 * last + rank)});
    }

  std::vector<PositionPath> result;
  for (const Candidate ending : endings) {
    if (!std::isfinite(ending.score))
      continue;
    PositionPath path{ending.score, {}};
    unsigned mask = all;
    uint16_t link = ending.previous;
    while (mask) {
      const int last = link / 2;
      const int rank = link % 2;
      path.nodes.push_back(last);
      link = table[static_cast<size_t>(mask) * n + last][rank].previous;
      mask ^= 1u << last;
    }
    std::reverse(path.nodes.begin(), path.nodes.end());
    result.push_back(std::move(path));
  }
  return result;
}

}  // namespace pluto::llm::one_shot_memorizer
