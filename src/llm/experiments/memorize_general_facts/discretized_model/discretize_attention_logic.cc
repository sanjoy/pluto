#include "src/llm/experiments/memorize_general_facts/discretized_model/discretize_attention_logic.h"

#include <algorithm>
#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "src/llm/experiments/memorize_general_facts/discretized_model/naming_utils.h"
#include "src/util/status_macros.h"

namespace pluto::llm::discretized::generator {
namespace {

using absl::StrCat;
using Lines = std::vector<std::string>;

absl::StatusOr<int> State(int value) {
  if (value < 0)
    return absl::InvalidArgumentError(
        "attention symbols and outputs must be nonnegative int32 integers");
  return value;
}

}  // namespace

absl::StatusOr<AttentionProgram> BuildAttention(
    absl::Span<const AttentionTransition> rows) {
  std::map<std::vector<int>, int> table;
  for (const auto& row : rows) {
    if (row.prefix.empty())
      return absl::InvalidArgumentError(
          "attention history must be a nonempty sequence");
    std::vector<int> key;
    for (int symbol : row.prefix) {
      ASSIGN_OR_RETURN(auto id, State(symbol));
      key.push_back(id);
    }
    ASSIGN_OR_RETURN(auto output, State(row.output));
    auto [it, inserted] = table.emplace(std::move(key), output);
    if (!inserted && it->second != output)
      return absl::InvalidArgumentError(
          "conflicting attention outputs for the same history");
  }

  std::vector<std::map<int, int>> children(1);
  std::vector<std::optional<int>> outputs(1);
  for (const auto& [key, output] : table) {
    int node = 0;
    for (int symbol : key) {
      auto found = children[node].find(symbol);
      if (found == children[node].end()) {
        int next = children.size();
        children[node][symbol] = next;
        children.emplace_back();
        outputs.emplace_back();
        node = next;
      } else {
        node = found->second;
      }
    }
    outputs[node] = output;
  }
  // Children always follow their parents. Interning bottom-up preserves both
  // acceptance at each prefix and every possible future labeled transition.
  std::map<AttentionNode, int> canonical;
  std::vector<AttentionNode> nodes;
  std::vector<int> mapped(children.size());
  for (size_t index = children.size(); index-- > 0;) {
    AttentionNode signature{outputs[index], {}};
    for (const auto& [symbol, child] : children[index])
      signature.edges.emplace_back(symbol, mapped[child]);
    auto [it, inserted] = canonical.emplace(signature, nodes.size());
    if (inserted)
      nodes.push_back(std::move(signature));
    mapped[index] = it->second;
  }
  return AttentionProgram{std::move(nodes), mapped[0]};
}

std::optional<int> EvaluateAttention(const AttentionProgram& program,
                                     absl::Span<const int> history) {
  int node = program.root;
  for (int symbol : history) {
    if (symbol < 0)
      return std::nullopt;
    const auto& edges = program.nodes[node].edges;
    auto found = std::lower_bound(edges.begin(), edges.end(),
                                  std::pair<int, int>{symbol, -1});
    if (found == edges.end() || found->first != symbol)
      return std::nullopt;
    node = found->second;
  }
  return program.nodes[node].output;
}

absl::StatusOr<std::string> RenderAttention(
    absl::string_view name, absl::Span<const AttentionTransition> rows,
    int chunk_size, absl::string_view strategy) {
  RETURN_IF_ERROR(ValidateTransitionFunctionName(name));
  if (chunk_size < 1)
    return absl::InvalidArgumentError("chunk_size must be a positive integer");
  if (strategy != "hybrid" && strategy != "control_flow")
    return absl::InvalidArgumentError(
        "strategy must be hybrid or control_flow");
  ASSIGN_OR_RETURN(auto program, BuildAttention(rows));
  std::vector<std::vector<int>> parents(program.nodes.size());
  for (size_t source = 0; source < program.nodes.size(); ++source)
    for (const auto& [symbol, target] : program.nodes[source].edges)
      parents[target].push_back(source);
  std::set<int> starts;
  std::map<int, std::vector<int>> entries;
  for (size_t node = 0; node < parents.size(); ++node) {
    int chunk = node / chunk_size;
    const auto& incoming = parents[node];
    bool entry = static_cast<int>(node) == program.root ||
                 std::any_of(incoming.begin(), incoming.end(), [&](int parent) {
                   return parent / chunk_size != chunk;
                 });
    if (entry)
      entries[chunk].push_back(node);
    if (entry || incoming.size() != 1 ||
        program.nodes[incoming[0]].edges.size() != 1)
      starts.insert(node);
  }
  const std::string result_type = StrCat(name, "Step");
  const std::string done = StrCat("k", name, "Done");
  Lines lines = {
      "// Exact ordered-history recognizer; unknown paths are rejected.",
      "// Control labels share identical suffix programs, not neural states.",
      "// MATCH checks one symbol; ending before it returns this prefix's "
      "output.",
      "// No hash, nearest-state fallback, corpus ID, or answer cache is used.",
      "namespace {",
      StrCat("constexpr std::uint32_t ", done, " = 0xffffffffu;"),
      StrCat("struct ", result_type,
             " { std::uint32_t next; std::optional<DiscreteHiddenState> "
             "result; };"),
      "#define PLUTO_ATTN_END(output) \\",
      StrCat("  do { if (position == history.size()) return {", done,
             ", {DiscreteHiddenState{output}}}; } while (false)"),
      "#define PLUTO_ATTN_MORE() \\",
      StrCat("  do { if (position == history.size()) return {", done,
             ", {}}; } while (false)"),
      "#define PLUTO_ATTN_MATCH(symbol, output) \\",
      StrCat("  do { PLUTO_ATTN_END(output); if (history[position++].value != "
             "symbol) return {",
             done, ", {}}; } while (false)"),
      "#define PLUTO_ATTN_SKIP(symbol) \\",
      StrCat("  do { PLUTO_ATTN_MORE(); if (history[position++].value != "
             "symbol) return {",
             done, ", {}}; } while (false)"),
  };
  auto jump = [&](int target, int chunk) {
    return target / chunk_size == chunk ? StrCat("goto n", target, ";")
                                        : StrCat("return {", target, "u, {}};");
  };
  const size_t sequence_insert = lines.size();
  using Pattern = std::vector<std::pair<int, int>>;
  std::map<Pattern, int> sequence_index;
  std::vector<Pattern> sequences;
  std::set<int> visited;
  for (const auto& [chunk, entry_nodes] : entries) {
    lines.push_back(
        StrCat(result_type, " ", name, "Part", chunk, "(std::uint32_t node,"));
    lines.emplace_back(
        "    [[maybe_unused]] absl::Span<const DiscreteHiddenState> history,");
    lines.emplace_back("    [[maybe_unused]] std::size_t& position) {");
    lines.emplace_back("  switch (node) {");
    for (auto it = entry_nodes.rbegin(); it != entry_nodes.rend(); ++it)
      lines.push_back(StrCat("    case ", *it, "u: goto n", *it, ";"));
    lines.push_back(StrCat("    default: return {", done, ", {}};"));
    lines.emplace_back("  }");
    for (auto it = starts.rbegin(); it != starts.rend(); ++it) {
      int first = *it;
      if (first / chunk_size != chunk)
        continue;
      lines.push_back(StrCat("n", first, ":"));
      int node_id = first;
      for (;;) {
        Pattern run;
        std::vector<int> run_nodes;
        int target = node_id;
        if (strategy == "hybrid") {
          for (;;) {
            const auto& candidate = program.nodes[target];
            if (!candidate.output.has_value() || candidate.edges.size() != 1)
              break;
            auto [symbol, following] = candidate.edges[0];
            run.emplace_back(symbol, *candidate.output);
            run_nodes.push_back(target);
            target = following;
            if (starts.contains(target) || target / chunk_size != chunk)
              break;
          }
        }
        if (run.size() >= 4) {
          auto [found, inserted] =
              sequence_index.emplace(run, sequences.size());
          if (inserted)
            sequences.push_back(std::move(run));
          for (int consumed : run_nodes)
            if (!visited.insert(consumed).second)
              return absl::InternalError("attention sequence emitted twice");
          lines.push_back(StrCat("  PLUTO_ATTN_RUN(k", name, "Sequence",
                                 found->second, ");"));
          if (!starts.contains(target) && target / chunk_size == chunk) {
            node_id = target;
            continue;
          }
          lines.push_back(StrCat("  ", jump(target, chunk)));
          break;
        }
        if (!visited.insert(node_id).second)
          return absl::InternalError(
              "attention control-flow block emitted twice");
        const auto& node = program.nodes[node_id];
        if (node.edges.empty()) {
          if (node.output.has_value())
            lines.push_back(StrCat("  PLUTO_ATTN_END(", *node.output, ");"));
          lines.push_back(StrCat("  return {", done, ", {}};"));
          break;
        }
        if (node.edges.size() == 1) {
          auto [symbol, next] = node.edges[0];
          lines.push_back(node.output.has_value()
                              ? StrCat("  PLUTO_ATTN_MATCH(", symbol, ", ",
                                       *node.output, ");")
                              : StrCat("  PLUTO_ATTN_SKIP(", symbol, ");"));
          if (!starts.contains(next) && next / chunk_size == chunk) {
            node_id = next;
            continue;
          }
          lines.push_back(StrCat("  ", jump(next, chunk)));
          break;
        }
        lines.push_back(node.output.has_value()
                            ? StrCat("  PLUTO_ATTN_END(", *node.output, ");")
                            : "  PLUTO_ATTN_MORE();");
        lines.emplace_back("  switch (history[position++].value) {");
        // Match Python's first-encounter ordering, rather than target-ID order.
        std::vector<std::pair<int, std::vector<int>>> targets;
        for (auto [symbol, next] : node.edges) {
          auto found =
              std::find_if(targets.begin(), targets.end(),
                           [&](const auto& row) { return row.first == next; });
          if (found == targets.end())
            targets.push_back({next, {symbol}});
          else
            found->second.push_back(symbol);
        }
        for (const auto& [next, symbols] : targets) {
          Lines cases;
          for (int symbol : symbols)
            cases.push_back(StrCat("case ", symbol, ":"));
          lines.push_back(StrCat("    ", absl::StrJoin(cases, " ")));
          lines.push_back(StrCat("      ", jump(next, chunk)));
        }
        lines.push_back(StrCat("    default: return {", done, ", {}};"));
        lines.emplace_back("  }");
        break;
      }
    }
    lines.emplace_back("}");
  }
  if (visited.size() != program.nodes.size())
    return absl::InternalError("attention layout omitted a reachable node");
  int sequence_bits = 16;
  for (const auto& pattern : sequences) {
    for (auto [symbol, output] : pattern)
      if (symbol > 65535 || output > 65535)
        sequence_bits = 32;
  }
  if (!sequences.empty()) {
    std::string sequence_type = StrCat(name, "MatchStep");
    Lines sequence_lines = {
        StrCat("struct ", sequence_type, " { std::uint", sequence_bits,
               "_t symbol, output; };"),
        "[[gnu::noinline]]",
        StrCat(result_type, " ", name, "MatchSequence(const ", sequence_type,
               "* steps,"),
        "    std::size_t count, absl::Span<const DiscreteHiddenState> history, "
        "std::size_t& position) {",
        "  for (std::size_t index = 0; index < count; ++index) {",
        StrCat(
            "    if (position == history.size()) return {", done,
            ", {DiscreteHiddenState{static_cast<int>(steps[index].output)}}};"),
        StrCat("    if (history[position++].value != "
               "static_cast<int>(steps[index].symbol)) return {",
               done, ", {}};"),
        "  }",
        "  return {0u, {}};  // The literal run matched; continue at its "
        "shared tail.",
        "}",
        "#define PLUTO_ATTN_RUN(steps) \\",
        StrCat("  do { auto matched = ", name,
               "MatchSequence(steps, sizeof(steps) / sizeof(steps[0]), "
               "history, position); if (matched.next == ",
               done, ") return matched; } while (false)"),
    };
    for (size_t index = 0; index < sequences.size(); ++index) {
      sequence_lines.push_back(StrCat("constexpr ", sequence_type, " k", name,
                                      "Sequence", index, "[] = {"));
      for (auto [symbol, output] : sequences[index])
        sequence_lines.push_back(StrCat("  {", symbol, "u, ", output, "u},"));
      sequence_lines.emplace_back("};");
    }
    lines.insert(lines.begin() + sequence_insert, sequence_lines.begin(),
                 sequence_lines.end());
  }
  Lines footer = {
      "#undef PLUTO_ATTN_END",
      "#undef PLUTO_ATTN_MORE",
      "#undef PLUTO_ATTN_MATCH",
      "#undef PLUTO_ATTN_SKIP",
      "#undef PLUTO_ATTN_RUN",
      "}  // namespace",
      "",
      StrCat("std::optional<DiscreteHiddenState> ", name,
             "(absl::Span<const DiscreteHiddenState> history) {"),
      "  std::size_t position = 0;",
      StrCat("  std::uint32_t node = ", program.root, "u;"),
      "  for (;;) {",
      StrCat("    ", result_type, " step;"),
      StrCat("    switch (node / ", chunk_size, "u) {"),
  };
  lines.insert(lines.end(), footer.begin(), footer.end());
  for (const auto& [chunk, unused] : entries)
    lines.push_back(StrCat("      case ", chunk, "u: step = ", name, "Part",
                           chunk, "(node, history, position); break;"));
  footer = {"      default: return {};",
            "    }",
            StrCat("    if (step.next == ", done, ") return step.result;"),
            "    node = step.next;",
            "  }",
            "}"};
  lines.insert(lines.end(), footer.begin(), footer.end());
  return StrCat(absl::StrJoin(lines, "\n"), "\n");
}

}  // namespace pluto::llm::discretized::generator
