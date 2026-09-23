#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"

namespace pluto::llm::one_shot_memorizer {

struct ContextAnalysisOptions {
  int prompt_tokens = 5;
  int eos_token = 50256;
};

struct ContextWindowStats {
  int window = 0;
  int64_t contexts = 0;
  // Contexts associated with more than one distinct supervised next token.
  int64_t conflicting_contexts = 0;
  int64_t targets = 0;
  // Sum over contexts of (occurrence count - largest next-token count).
  int64_t irreducible_top1_errors = 0;
};

struct ContextPrediction {
  // Most frequent next token, breaking ties with the smallest token ID.
  int token = -1;
  double probability = 0;
  int64_t occurrences = 0;
  int64_t distinct_next_tokens = 0;
};

// An empirical next-token model built only from supervised corpus targets.
// The key is the final min(window, history.size()) tokens. Key length retains
// the sentence's left boundary when its history is shorter than the window.
// No backoff, smoothing, position, or sentence identity enters the prediction.
class ContextModel {
 public:
  // Sentences contain original text token IDs, without EOS or padding. Score
  // positions [prompt_tokens, sentence.size()], including one terminal EOS;
  // tokens within the supplied prompt are context but are never targets.
  // Requires a nonempty corpus, positive prompt_tokens, nonnegative window,
  // and at least prompt_tokens tokens in every sentence. All tokens must be
  // nonnegative, and eos_token is reserved and must not appear in sentences.
  static absl::StatusOr<ContextModel> Build(
      absl::Span<const std::vector<int>> sentences, int window,
      ContextAnalysisOptions options = {});

  // Histories may be empty and may contain any nonnegative token IDs.
  // Returns NotFound for an unseen context, including insufficient history
  // when that shorter context was never observed. Does not silently back off.
  absl::StatusOr<ContextPrediction> Predict(
      absl::Span<const int> history) const;

  // Empirical probability of token after history; zero for a next token not
  // observed after a known context, and NotFound for an unknown context.
  absl::StatusOr<double> Probability(absl::Span<const int> history,
                                     int token) const;

  int window() const { return stats_.window; }
  const ContextWindowStats& stats() const { return stats_; }

 private:
  struct ContextCounts {
    absl::flat_hash_map<int, int64_t> next_tokens;
    ContextPrediction prediction;
  };

  explicit ContextModel(int window) { stats_.window = window; }

  absl::StatusOr<const ContextCounts*> Find(
      absl::Span<const int> history) const;

  absl::flat_hash_map<std::vector<int>, ContextCounts> counts_;
  ContextWindowStats stats_;
};

struct TargetContextStats {
  size_t sentence_index = 0;
  // Zero-based next-token index; also the number of preceding text tokens.
  // An index equal to sentence.size() denotes the terminal EOS prediction.
  int target_index = 0;
  int target_token = -1;
  // First window with just one distinct next token for this target's context.
  // -1 means no tested window suffices, even with the complete prefix.
  int shortest_sufficient_window = -1;
  // First window identifying a single supervised occurrence, not merely a
  // sentence. Repeated occurrences with the same target can be sufficient
  // without being unique. -1 means no tested window identifies one occurrence.
  int shortest_unique_window = -1;
};

struct ContextAnalysis {
  int longest_prefix = 0;
  int shortest_exact_window = -1;
  // One entry for each window from zero through longest_prefix, inclusive.
  std::vector<ContextWindowStats> windows;
  // Corpus order, then increasing target_index within each sentence.
  std::vector<TargetContextStats> targets;
};

// Measures what suffix-window mechanisms can represent on supervised corpus
// positions; it does not identify the mechanism used by a trained model.
// Shortest windows include the left-boundary distinction: a prefix of length
// n may first become sufficient at window n+1, when longer matching histories
// gain another token. Thus a window is not always an observed suffix length.
absl::StatusOr<ContextAnalysis> AnalyzeContexts(
    absl::Span<const std::vector<int>> sentences,
    ContextAnalysisOptions options = {});

}  // namespace pluto::llm::one_shot_memorizer
