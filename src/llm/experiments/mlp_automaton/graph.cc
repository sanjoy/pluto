#include "src/llm/experiments/mlp_automaton/graph.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <locale>
#include <ostream>
#include <random>
#include <sstream>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_cat.h"
#include "src/util/status_macros.h"

namespace pluto::llm::mlp_automaton {
namespace {

constexpr char kHex[] = "0123456789abcdef";

std::string Hex(absl::string_view bytes) {
  std::string result;
  result.reserve(2 * bytes.size());
  for (unsigned char c : bytes) {
    result.push_back(kHex[c >> 4]);
    result.push_back(kHex[c & 15]);
  }
  return result;
}

// Keep display text ASCII-only. For example, a literal backslash followed by
// 'n' is displayed as "\\\\n", while a newline is displayed as "\\n".
std::string EscapedBytes(absl::string_view bytes) {
  std::string result;
  for (unsigned char c : bytes) {
    switch (c) {
      case '\\':
        result += "\\\\";
        break;
      case '\n':
        result += "\\n";
        break;
      case '\r':
        result += "\\r";
        break;
      case '\t':
        result += "\\t";
        break;
      default:
        if (c >= 32 && c < 127) {
          result.push_back(c);
        } else {
          result += "\\x";
          result.push_back(kHex[c >> 4]);
          result.push_back(kHex[c & 15]);
        }
    }
  }
  return result;
}

// Only called with ASCII display text, hex, or our fixed schema strings.
std::string Quote(absl::string_view value) {
  std::string result = "\"";
  for (char c : value) {
    if (c == '\\' || c == '"')
      result.push_back('\\');
    result.push_back(c);
  }
  result.push_back('"');
  return result;
}

// Do not inherit a caller's locale, precision, or fixed/scientific formatting:
// JSON numbers require a period, and probabilities should round-trip exactly.
std::string Number(double number) {
  std::ostringstream output;
  output.imbue(std::locale::classic());
  output << std::setprecision(std::numeric_limits<double>::max_digits10)
         << number;
  return output.str();
}

std::vector<int> Successors(const Graph& graph) {
  std::vector<int> successors(graph.token_bytes.size(), -1);
  for (const Edge& edge : graph.edges)
    successors[edge.source] = edge.target;
  return successors;
}

// A generation stamp avoids clearing a vocabulary-sized visited array between
// short sampled paths. Repeated nodes are detected before following an edge
// again, so even a graph made entirely of cycles has bounded traversal cost.
Path WalkUnchecked(const Graph& graph, const std::vector<int>& successors,
                   int start, size_t max_tokens, size_t stamp,
                   std::vector<size_t>& visited) {
  Path path;
  int current = start;
  path.tokens.push_back(current);
  visited[current] = stamp;
  while (true) {
    if (current == graph.eos_token_id) {
      path.termination = Termination::kEndOfSequence;
      return path;
    }
    if (successors[current] == -1) {
      path.termination = Termination::kNoEdge;
      return path;
    }
    if (path.tokens.size() >= max_tokens) {
      path.termination = Termination::kTokenLimit;
      return path;
    }
    current = successors[current];
    path.tokens.push_back(current);
    if (visited[current] == stamp) {
      path.termination = Termination::kCycle;
      return path;
    }
    visited[current] = stamp;
  }
}

// A validated Walk records its starting token followed by each traversed
// edge's target. A singleton is only a vocabulary lookup, not evidence of a
// learned continuation. Checking path length (not just outgoing degree) also
// excludes EOS starts and walks stopped before an edge by max_tokens=1.
bool HasTransition(const Path& path) { return path.tokens.size() >= 2; }

std::string DecodedBytes(const Graph& graph, const Path& path) {
  std::string bytes;
  for (int token : path.tokens)
    bytes += graph.token_bytes[token];
  return bytes;
}

void SortCombinedPaths(std::vector<CombinedPath>& paths) {
  for (auto& path : paths)
    std::sort(path.mlp_blocks.begin(), path.mlp_blocks.end());
  std::sort(paths.begin(), paths.end(),
            [](const CombinedPath& a, const CombinedPath& b) {
              return a.bytes < b.bytes;
            });
}

// Rejection sampling removes modulo bias without depending on the
// implementation-specific mapping of std::uniform_int_distribution.
uint64_t UniformBelow(std::mt19937_64& random, uint64_t bound) {
  const uint64_t cutoff = -bound % bound;
  uint64_t value;
  do {
    value = random();
  } while (value < cutoff);
  return value % bound;
}

absl::Status StreamStatus(const std::ostream& output) {
  if (!output.good())
    return absl::DataLossError("Failed writing JSON output");
  return absl::OkStatus();
}

}  // namespace

absl::string_view TerminationName(Termination termination) {
  switch (termination) {
    case Termination::kNoEdge:
      return "no_edge";
    case Termination::kCycle:
      return "cycle";
    case Termination::kEndOfSequence:
      return "end_of_sequence";
    case Termination::kTokenLimit:
      return "token_limit";
  }
  return "invalid";
}

absl::Status ValidateGraph(const Graph& graph) {
  if (!std::isfinite(graph.threshold) || graph.threshold < 0.5 ||
      graph.threshold >= 1.0) {
    return absl::InvalidArgumentError("Graph threshold must be in [0.5, 1)");
  }
  if (graph.token_bytes.empty() ||
      graph.token_bytes.size() > std::numeric_limits<int>::max()) {
    return absl::InvalidArgumentError(
        "Vocabulary size must be in [1, INT_MAX]");
  }
  const int size = static_cast<int>(graph.token_bytes.size());
  if (graph.eos_token_id < -1 || graph.eos_token_id >= size)
    return absl::InvalidArgumentError("EOS token is outside the vocabulary");
  std::vector<bool> has_edge(size, false);
  for (const Edge& edge : graph.edges) {
    if (edge.source < 0 || edge.source >= size || edge.target < 0 ||
        edge.target >= size) {
      return absl::InvalidArgumentError("Edge token is outside the vocabulary");
    }
    if (!std::isfinite(edge.probability) ||
        edge.probability <= graph.threshold || edge.probability > 1.0) {
      return absl::InvalidArgumentError(
          "Edge probability must be finite, > threshold, and <= 1");
    }
    if (has_edge[edge.source]) {
      return absl::InvalidArgumentError(
          absl::StrCat("Multiple outgoing edges for token ", edge.source));
    }
    has_edge[edge.source] = true;
  }
  return absl::OkStatus();
}

absl::StatusOr<Path> Walk(const Graph& graph, int start, size_t max_tokens) {
  RETURN_IF_ERROR(ValidateGraph(graph));
  if (start < 0 || static_cast<size_t>(start) >= graph.token_bytes.size())
    return absl::InvalidArgumentError("Starting token is outside vocabulary");
  if (max_tokens == 0)
    return absl::InvalidArgumentError("max_tokens must be positive");
  std::vector<size_t> visited(graph.token_bytes.size(), 0);
  return WalkUnchecked(graph, Successors(graph), start, max_tokens, 1, visited);
}

absl::StatusOr<std::vector<Path>> SamplePaths(const Graph& graph, size_t count,
                                              size_t max_tokens,
                                              uint64_t seed) {
  RETURN_IF_ERROR(ValidateGraph(graph));
  if (max_tokens == 0)
    return absl::InvalidArgumentError("max_tokens must be positive");
  const std::vector<int> successors = Successors(graph);
  std::vector<int> starts;
  for (size_t i = 0; i < successors.size(); ++i)
    if (successors[i] != -1 && static_cast<int>(i) != graph.eos_token_id)
      starts.push_back(static_cast<int>(i));
  count = std::min(count, starts.size());
  std::mt19937_64 random(seed);
  std::vector<size_t> visited(graph.token_bytes.size(), 0);
  std::vector<Path> paths;
  paths.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    const size_t selected = i + UniformBelow(random, starts.size() - i);
    std::swap(starts[i], starts[selected]);
    paths.push_back(WalkUnchecked(graph, successors, starts[i], max_tokens,
                                  i + 1, visited));
  }
  return paths;
}

absl::StatusOr<std::vector<Path>> FilterPathsInCorpus(
    const Graph& graph, absl::Span<const Path> paths,
    absl::string_view training_text) {
  RETURN_IF_ERROR(ValidateGraph(graph));
  std::vector<Path> matches;
  for (const Path& path : paths) {
    if (path.tokens.empty() || TerminationName(path.termination) == "invalid")
      return absl::InvalidArgumentError(
          "Path must have tokens and termination");
    std::string bytes;
    for (int token : path.tokens) {
      if (token < 0 || static_cast<size_t>(token) >= graph.token_bytes.size())
        return absl::InvalidArgumentError("Path token is outside vocabulary");
      bytes += graph.token_bytes[token];
    }
    // Use the whole byte string, including any closing cycle token. Matching
    // individual edges (or a shorter prefix) would admit nonexistent phrases.
    if (!bytes.empty() && training_text.find(bytes) != absl::string_view::npos)
      matches.push_back(path);
  }
  return matches;
}

absl::StatusOr<std::vector<CombinedPath>> CombinePaths(
    absl::Span<const BlockPaths> blocks, size_t max_tokens) {
  if (max_tokens == 0)
    return absl::InvalidArgumentError("max_tokens must be positive");
  absl::flat_hash_set<int> block_ids;
  absl::flat_hash_map<std::string, size_t> index;
  std::vector<CombinedPath> combined;
  // Gather candidate strings from the caller's filtered samples. Validate
  // full walks here so malformed paths cannot invent a combined candidate.
  // Explicit zero-edge samples remain diagnostics, not continuation candidates.
  for (const BlockPaths& block : blocks) {
    if (block.mlp_block < 0 || !block_ids.insert(block.mlp_block).second)
      return absl::InvalidArgumentError(
          "MLP block IDs must be nonnegative and unique");
    RETURN_IF_ERROR(ValidateGraph(block.graph));
    const auto successors = Successors(block.graph);
    std::vector<size_t> visited(block.graph.token_bytes.size(), 0);
    for (size_t i = 0; i < block.paths.size(); ++i) {
      const Path& path = block.paths[i];
      if (path.tokens.empty() || path.tokens.front() < 0 ||
          static_cast<size_t>(path.tokens.front()) >= successors.size())
        return absl::InvalidArgumentError("Invalid combined path start");
      const Path expected =
          WalkUnchecked(block.graph, successors, path.tokens.front(),
                        max_tokens, i + 1, visited);
      if (path.tokens != expected.tokens ||
          path.termination != expected.termination)
        return absl::InvalidArgumentError(
            "Combined candidates must be full walks under max_tokens");
      if (!HasTransition(path))
        continue;
      std::string bytes = DecodedBytes(block.graph, path);
      if (!bytes.empty() && index.emplace(bytes, combined.size()).second)
        combined.push_back({std::move(bytes), {}});
    }
  }
  if (combined.empty())
    return combined;

  // Enumerate each block once, sharing the successor table and visited
  // scratch across walks. Only texts already sampled in some block are kept.
  // Looking solely at each block's random samples would miss valid memberships.
  for (const BlockPaths& block : blocks) {
    const auto successors = Successors(block.graph);
    std::vector<size_t> visited(block.graph.token_bytes.size(), 0);
    std::vector<bool> found(combined.size(), false);
    for (size_t start = 0; start < successors.size(); ++start) {
      const Path path =
          WalkUnchecked(block.graph, successors, static_cast<int>(start),
                        max_tokens, start + 1, visited);
      if (!HasTransition(path))
        continue;
      std::string bytes = DecodedBytes(block.graph, path);
      const auto entry = index.find(bytes);
      if (entry == index.end() || found[entry->second])
        continue;
      found[entry->second] = true;
      combined[entry->second].mlp_blocks.push_back(block.mlp_block);
    }
  }
  SortCombinedPaths(combined);
  return combined;
}

absl::StatusOr<std::vector<CombinedPath>> CollectAllPaths(
    absl::Span<const BlockPaths> blocks, size_t max_tokens) {
  if (max_tokens == 0)
    return absl::InvalidArgumentError("max_tokens must be positive");
  absl::flat_hash_set<int> block_ids;
  absl::flat_hash_map<std::string, size_t> index;
  std::vector<CombinedPath> combined;
  for (const BlockPaths& block : blocks) {
    if (block.mlp_block < 0 || !block_ids.insert(block.mlp_block).second)
      return absl::InvalidArgumentError(
          "MLP block IDs must be nonnegative and unique");
    RETURN_IF_ERROR(ValidateGraph(block.graph));
    const auto successors = Successors(block.graph);
    std::vector<size_t> visited(block.graph.token_bytes.size(), 0);
    // Reuse one successor table and generation-stamped visited array per
    // block. Calling public Walk for every token would rebuild and clear
    // vocabulary-sized scratch for every (usually very short) walk.
    for (size_t start = 0; start < successors.size(); ++start) {
      const Path path =
          WalkUnchecked(block.graph, successors, static_cast<int>(start),
                        max_tokens, start + 1, visited);
      if (!HasTransition(path))
        continue;
      std::string bytes = DecodedBytes(block.graph, path);
      if (bytes.empty())
        continue;
      auto [entry, inserted] = index.emplace(bytes, combined.size());
      if (inserted)
        combined.push_back({std::move(bytes), {}});
      auto& ids = combined[entry->second].mlp_blocks;
      // All walks from a block are contiguous, and block IDs are unique.
      // Different starts/tokenizations can spell the same text within a
      // block; only the first such walk adds membership.
      if (ids.empty() || ids.back() != block.mlp_block)
        ids.push_back(block.mlp_block);
    }
  }
  SortCombinedPaths(combined);
  return combined;
}

absl::Status WriteCombinedPathsJson(std::ostream& output,
                                    absl::Span<const CombinedPath> paths) {
  // Validate before writing any bytes, as with the other JSON serializers.
  absl::flat_hash_set<absl::string_view> seen;
  for (const auto& path : paths) {
    if (path.bytes.empty() || path.mlp_blocks.empty() ||
        !seen.insert(path.bytes).second)
      return absl::InvalidArgumentError(
          "Combined paths must have unique nonempty text and block IDs");
    int previous = -1;
    for (int block : path.mlp_blocks) {
      if (block <= previous)
        return absl::InvalidArgumentError(
            "Combined block IDs must be nonnegative, unique and sorted");
      previous = block;
    }
  }
  output << "{\n  \"format\": \"pluto.mlp_automaton.combined_paths.v1\",\n"
         << "  \"paths\": [\n";
  for (size_t i = 0; i < paths.size(); ++i) {
    const auto& path = paths[i];
    output << "    {\"bytes_hex\": " << Quote(Hex(path.bytes))
           << ", \"bytes_escaped\": " << Quote(EscapedBytes(path.bytes))
           << ", \"mlp_blocks\": [";
    for (size_t j = 0; j < path.mlp_blocks.size(); ++j) {
      if (j != 0)
        output << ", ";
      output << std::to_string(path.mlp_blocks[j]);
    }
    output << "]}" << (i + 1 == paths.size() ? "\n" : ",\n");
  }
  output << "  ]\n}\n";
  return StreamStatus(output);
}

absl::Status WriteGraphJson(std::ostream& output, const Graph& graph) {
  RETURN_IF_ERROR(ValidateGraph(graph));
  output << "{\n  \"format\": \"pluto.mlp_automaton.v1\",\n"
         << "  \"threshold\": " << Number(graph.threshold) << ",\n"
         << "  \"comparison\": \"strictly_greater\",\n"
         << "  \"eos_token_id\": " << std::to_string(graph.eos_token_id)
         << ",\n  \"nodes\": [\n";
  for (size_t i = 0; i < graph.token_bytes.size(); ++i) {
    output << "    {\"id\": " << std::to_string(i)
           << ", \"bytes_hex\": " << Quote(Hex(graph.token_bytes[i]))
           << ", \"bytes_escaped\": "
           << Quote(EscapedBytes(graph.token_bytes[i])) << "}"
           << (i + 1 == graph.token_bytes.size() ? "\n" : ",\n");
  }
  output << "  ],\n  \"edges\": [\n";
  for (size_t i = 0; i < graph.edges.size(); ++i) {
    const Edge& edge = graph.edges[i];
    output << "    {\"source\": " << std::to_string(edge.source)
           << ", \"target\": " << std::to_string(edge.target)
           << ", \"probability\": " << Number(edge.probability) << "}"
           << (i + 1 == graph.edges.size() ? "\n" : ",\n");
  }
  output << "  ]\n}\n";
  return StreamStatus(output);
}

absl::Status WritePathsJson(std::ostream& output, const Graph& graph,
                            absl::Span<const Path> paths) {
  RETURN_IF_ERROR(ValidateGraph(graph));
  // Validate before writing, so invalid paths do not leave a partial document.
  for (const Path& path : paths) {
    if (path.tokens.empty() || TerminationName(path.termination) == "invalid") {
      return absl::InvalidArgumentError(
          "Path must have tokens and termination");
    }
    for (int token : path.tokens)
      if (token < 0 || static_cast<size_t>(token) >= graph.token_bytes.size())
        return absl::InvalidArgumentError("Path token is outside vocabulary");
  }
  output << "{\n  \"format\": \"pluto.mlp_automaton.paths.v1\",\n"
         << "  \"paths\": [\n";
  for (size_t i = 0; i < paths.size(); ++i) {
    const Path& path = paths[i];
    std::string bytes;
    output << "    {\"tokens\": [";
    for (size_t j = 0; j < path.tokens.size(); ++j) {
      if (j != 0)
        output << ", ";
      output << std::to_string(path.tokens[j]);
      bytes += graph.token_bytes[path.tokens[j]];
    }
    output << "], \"termination\": " << Quote(TerminationName(path.termination))
           << ", \"bytes_hex\": " << Quote(Hex(bytes))
           << ", \"bytes_escaped\": " << Quote(EscapedBytes(bytes)) << "}"
           << (i + 1 == paths.size() ? "\n" : ",\n");
  }
  output << "  ]\n}\n";
  return StreamStatus(output);
}

}  // namespace pluto::llm::mlp_automaton
