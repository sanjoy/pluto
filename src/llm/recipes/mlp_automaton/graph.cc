#include "src/llm/recipes/mlp_automaton/graph.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <locale>
#include <ostream>
#include <random>
#include <sstream>
#include <utility>

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
