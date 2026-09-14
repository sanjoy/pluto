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

// A block's full transition graph and corpus-filtered sampled paths. Keeping
// graphs separate prevents accidentally stitching together different MLPs.
struct BlockPaths {
  int mlp_block;
  Graph graph;
  std::vector<Path> paths;
};

// One exact decoded byte string, deduplicated across paths/tokenizations.
// Block IDs are unique, sorted and zero-based. Whitespace remains significant.
struct CombinedPath {
  std::string bytes;
  std::vector<int> mlp_blocks;
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

// Combines the sampled texts, then checks every starting token in every
// supplied graph for each text's membership. Thus block attribution does not
// depend on that block also having sampled the text. Membership means a full
// Walk ending under the same max_tokens limit (not just a prefix of one).
// Results are sorted by exact bytes, independently of block/sample order.
// Input paths must be full walks in their block under that same token limit;
// block IDs must be nonnegative and unique. An empty input returns no results.
absl::StatusOr<std::vector<CombinedPath>> CombinePaths(
    absl::Span<const BlockPaths> blocks, size_t max_tokens);

// Collects full Walks from every starting token in every supplied graph,
// including isolated tokens and EOS. Unlike CombinePaths, this does not select
// candidates from BlockPaths::paths; that field is ignored. Empty decoded
// strings are omitted, but empty token pieces within a nonempty walk are kept.
// Membership and sorting use exact decoded bytes, not a particular
// tokenization. Block IDs must be nonnegative and unique, and max_tokens must
// be positive. No corpus filtering or cross-block edge stitching is performed.
// Complete membership is useful for checkpoint histories: an unsampled path
// must not look like a path that disappeared during training.
absl::StatusOr<std::vector<CombinedPath>> CollectAllPaths(
    absl::Span<const BlockPaths> blocks, size_t max_tokens);

// Writes a lossless bytes_hex, ASCII bytes_escaped, and mlp_blocks array for
// each combined path. No single tokenization represents all possible blocks.
absl::Status WriteCombinedPathsJson(std::ostream& output,
                                    absl::Span<const CombinedPath> paths);

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
