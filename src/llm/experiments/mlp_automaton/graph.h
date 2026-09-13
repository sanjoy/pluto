#pragma once

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"

namespace pluto::llm::mlp_automaton {

struct Edge {
  int source;
  int target;
  double probability;
};

// Nodes are tokenizer IDs, including isolated tokens and special tokens. A
// strict threshold >= 0.5 permits at most one outgoing edge per softmax row.
// Token pieces are arbitrary bytes: an individual GPT-2 token need not be valid
// UTF-8, even when the complete input text is valid UTF-8.
struct Graph {
  double threshold = 0.75;
  std::vector<std::string> token_bytes;
  std::vector<Edge> edges;
  int eos_token_id = -1;
};

enum class Termination { kNoEdge, kCycle, kEndOfSequence, kTokenLimit };

struct Path {
  // A cycle includes its closing, repeated token. The token limit includes the
  // initial token, and EOS may appear as the last token but is never followed.
  std::vector<int> tokens;
  Termination termination;
};

absl::string_view TerminationName(Termination termination);

// Checks ID bounds, unique sources, the threshold, and finite probabilities in
// (threshold, 1]. An EOS node may have an edge; traversal simply does not use
// it.
absl::Status ValidateGraph(const Graph& graph);

absl::StatusOr<Path> Walk(const Graph& graph, int start, size_t max_tokens);

// Uniformly chooses distinct starting nodes that have an outgoing edge,
// excluding EOS. Returns at most count paths (fewer when the graph has fewer
// eligible starts). Traversal itself is deterministic because outdegree <= 1.
// The seed determines ordering reproducibly, independently of edge-list order.
absl::StatusOr<std::vector<Path>> SamplePaths(const Graph& graph, size_t count,
                                              size_t max_tokens, uint64_t seed);

// Keeps paths whose complete concatenated token bytes occur as a contiguous,
// case-sensitive substring of training_text. No trimming, UTF-8 conversion, or
// word-boundary requirement is applied. Empty decoded strings never qualify.
// Preserves order, token IDs and termination reasons; the graph is unchanged.
// The caller supplies only the training portion, excluding any held-out text.
absl::StatusOr<std::vector<Path>> FilterPathsInCorpus(
    const Graph& graph, absl::Span<const Path> paths,
    absl::string_view training_text);

// JSON contains every vocabulary node, not just nodes participating in edges.
// bytes_hex is the lossless representation; bytes_escaped is an ASCII-only
// display using C-style byte escapes. Neither assumes token bytes are UTF-8.
absl::Status WriteGraphJson(std::ostream& output, const Graph& graph);

// Writes concatenated token bytes as well as token IDs and the stopping reason.
// Paths are diagnostic token continuations, not necessarily linguistic words or
// evidence that the full model has memorized a particular training substring.
absl::Status WritePathsJson(std::ostream& output, const Graph& graph,
                            absl::Span<const Path> paths);

}  // namespace pluto::llm::mlp_automaton
