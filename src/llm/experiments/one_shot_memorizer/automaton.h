#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"

namespace pluto::llm::one_shot_memorizer {

// One nonzero of token-specific sparse matrix T_token: T_token[source,target]
// equals 1. Each source has at most one transition for any given token.
struct Transition {
  int token;
  size_t target;
};

struct State {
  // The terminal-vector entry: occurrences of the empty accepted suffix.
  uint64_t terminal_count = 0;
  // Total accepted suffix occurrences, including the empty suffix.
  uint64_t suffix_count = 0;
  // Sorted by token; EOS is represented by terminal_count, never by an edge.
  std::vector<Transition> transitions;
};

// A sparse weighted acyclic automaton constructed directly from a corpus.
// Starting from one-hot row h at initial_state, h <- h * T_token consumes a
// token. h * terminal_count gives the exact multiplicity of the consumed
// sentence; no embeddings, gradient updates, or pretrained weights are used.
// State IDs are canonical for a corpus multiset, with target < source on every
// transition. States merge only for exactly equal weighted right languages:
// this does not claim minimum dimension among all weighted linear models.
struct Model {
  int vocabulary_size = 0;
  int eos_token_id = -1;
  size_t initial_state = 0;
  size_t trie_state_count = 0;
  uint64_t sentence_count = 0;
  std::vector<State> states;
};

struct TokenProbability {
  int token;
  // Number of corpus sentence occurrences accepting this next token.
  uint64_t count;
  double probability;
};

// Sentences contain in-range token IDs excluding EOS. Sentence boundaries add
// an implicit explicit-EOS prediction. Empty sentences and duplicates are
// retained; an empty corpus, invalid IDs, and EOS in sentences are rejected.
// Uses a prefix trie and exact bottom-up weighted-right-language minimization.
absl::StatusOr<Model> BuildModel(const std::vector<std::vector<int>>& sentences,
                                 int vocabulary_size, int eos_token_id);

// Checks bounds, acyclicity, sorted deterministic transitions, count sums,
// overflow, reachability, and corpus metadata. Useful after deserialization.
absl::Status ValidateModel(const Model& model);

// Prefixes exclude EOS. An unseen prefix returns NotFound; invalid token IDs
// or an EOS in the prefix return InvalidArgument. Each query validates model.
absl::StatusOr<size_t> StateForPrefix(const Model& model,
                                      absl::Span<const int> prefix);

// Returns only positive probabilities, sorted by token ID. Non-EOS token t
// has probability suffix_count[target(t)] / suffix_count[current]; EOS has
// probability terminal_count[current] / suffix_count[current].
absl::StatusOr<std::vector<TokenProbability>> NextTokenProbabilities(
    const Model& model, absl::Span<const int> prefix);

// Returns generated tokens only, including EOS if reached within the positive
// limit. Chooses maximum next-token count, breaking ties by smaller token ID.
// Multiple corpus suffixes under a prefix can make exact recall ambiguous.
absl::StatusOr<std::vector<int>> GreedyContinuation(
    const Model& model, absl::Span<const int> prefix, size_t max_new_tokens);

}  // namespace pluto::llm::one_shot_memorizer
