#include "src/llm/experiments/memorize_general_facts/discretized_model/discrete_model_generator/discretize_position_embedding.h"

#include <algorithm>
#include <bit>
#include <cstdint>
#include <map>
#include <set>
#include <tuple>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_replace.h"
#include "src/util/status_macros.h"

namespace pluto::llm::discretized::generator {
namespace {

using absl::StrCat;
using Lines = std::vector<std::string>;
using Mapping = std::map<int, int>;
using EntryMapping = std::map<std::pair<int, int>, int>;

absl::StatusOr<EntryMapping> MakeEntryMapping(
    absl::Span<const EntryTransition> rows) {
  EntryMapping result;
  for (const auto& row : rows) {
    if (row.token < 0 || row.output < 0)
      return absl::InvalidArgumentError(
          "entry IDs must be nonnegative and fit int");
    if (row.position < 0)
      return absl::InvalidArgumentError(
          "entry position must be nonnegative and fit int32_t");
    std::pair<int, int> key{row.token, row.position};
    int output = row.output;
    auto [it, inserted] = result.emplace(key, output);
    if (!inserted && it->second != output)
      return absl::InvalidArgumentError("conflicting entry mapping");
  }
  return result;
}

std::string Name(int value, const TokenNames* names) {
  return names == nullptr ? StrCat(value) : names->find(value)->second;
}

absl::StatusOr<std::string> RenderEntry(absl::string_view name,
                                        absl::Span<const EntryTransition> rows,
                                        const TokenNames& token_names) {
  RETURN_IF_ERROR(ValidateTransitionFunctionName(name));
  ASSIGN_OR_RETURN(auto entry, MakeEntryMapping(rows));
  for (const auto& [key, output] : entry)
    if (!token_names.contains(key.first))
      return absl::InvalidArgumentError("missing named entry token");
  Lines begin = {StrCat("std::optional<DiscreteHiddenState> ", name,
                        "(DiscreteToken token, int32_t position) {")};
  if (entry.empty()) {
    begin.insert(begin.end(),
                 {"  (void)token;", "  (void)position;", "  return {};", "}"});
    return JoinCppLines(begin);
  }
  std::map<int, Mapping> tokens;
  std::set<int> state_set;
  int max_position = 0;
  for (const auto& [key, output] : entry) {
    tokens[key.first][key.second] = output;
    state_set.insert(output);
    max_position = std::max(max_position, key.second);
  }
  int low = tokens.begin()->first, high = tokens.rbegin()->first;
  std::vector<int> states(state_set.begin(), state_set.end());
  Mapping state_index;
  for (size_t index = 0; index < states.size(); ++index)
    state_index[states[index]] = index;
  int64_t position_bits = static_cast<int64_t>(max_position) + 1;
  int state_bits =
      std::max(1, static_cast<int>(std::bit_width(states.size() - 1)));
  if (position_bits + state_bits > 64 ||
      static_cast<int64_t>(high) - low + 1 >
          std::max<int64_t>(64, 4 * tokens.size())) {
    Lines lines = begin;
    lines.emplace_back("  switch (token.value) {");
    for (const auto& [token, positions] : tokens) {
      lines.push_back(
          StrCat("    case ", Name(token, &token_names), ".value:"));
      lines.emplace_back("      switch (position) {");
      for (auto [position, output] : positions)
        lines.push_back(StrCat("        case ", position,
                               ": return {DiscreteHiddenState{", output,
                               "}};"));
      lines.insert(lines.end(), {"        default: return {};", "      }"});
    }
    lines.insert(lines.end(), {"    default: return {};", "  }", "}"});
    return JoinCppLines(lines);
  }
  std::vector<uint64_t> patterns{0};
  std::map<uint64_t, size_t> pattern_index{{0, 0}};
  std::vector<size_t> indices;
  std::vector<std::tuple<int, int, int>> exceptions;
  for (int64_t token = low; token <= high; ++token) {
    auto found = tokens.find(token);
    if (found == tokens.end()) {
      indices.push_back(0);
      continue;
    }
    const auto& positions = found->second;
    Mapping counts;
    for (auto [position, output] : positions)
      ++counts[output];
    int default_state = -1, default_count = -1;
    for (auto [state, count] : counts)
      if (count > default_count) {
        default_state = state;
        default_count = count;
      }
    uint64_t mask = 0;
    for (auto [position, output] : positions)
      mask |= uint64_t{1} << position;
    uint64_t packed =
        (static_cast<uint64_t>(state_index[default_state]) << position_bits) |
        mask;
    auto [pattern, inserted] = pattern_index.emplace(packed, patterns.size());
    if (inserted)
      patterns.push_back(packed);
    indices.push_back(pattern->second);
    for (auto [position, output] : positions)
      if (output != default_state)
        exceptions.emplace_back(token, position, output);
  }
  int word_bytes = position_bits + state_bits <= 32 ? 4 : 8;
  std::string word_type = word_bytes == 4 ? "uint32_t" : "uint64_t";
  int index_bytes =
      patterns.size() <= 256 ? 1 : (patterns.size() <= 65536 ? 2 : 4);
  std::string index_type = StrCat("uint", index_bytes * 8, "_t");
  const char* suffix = word_bytes == 4 ? "u" : "ull";
  Lines lines = begin;
  lines.push_back(StrCat("  if (token.value < ", low, " || token.value > ",
                         high, " || position < 0 || position > ", max_position,
                         ") return {};"));
  Lines values;
  for (size_t index : indices)
    values.push_back(StrCat(index));
  AppendCppArray(lines, "kTokenPatterns", index_type, values);
  values.clear();
  for (uint64_t pattern : patterns)
    values.push_back(absl::StrFormat("0x%x%s", pattern, suffix));
  AppendCppArray(lines, "kPatterns", word_type, values, 8);
  lines.push_back(StrCat("  const ", word_type,
                         " packed = kPatterns[kTokenPatterns[token.value - ",
                         low, "]];"));
  lines.push_back(StrCat("  if ((packed & (", word_type,
                         "{1} << position)) == 0) return {};"));
  if (!exceptions.empty()) {
    lines.emplace_back("  switch (token.value) {");
    std::map<int, std::vector<std::pair<int, int>>> grouped;
    for (auto [token, position, output] : exceptions)
      grouped[token].emplace_back(position, output);
    for (const auto& [token, positions] : grouped) {
      lines.push_back(
          StrCat("    case ", Name(token, &token_names), ".value:"));
      for (auto [position, output] : positions)
        lines.push_back(StrCat("      if (position == ", position,
                               ") return {DiscreteHiddenState{", output,
                               "}};"));
      lines.emplace_back("      break;");
    }
    lines.insert(lines.end(), {"    default: break;", "  }"});
  }
  if (states.size() ==
      static_cast<size_t>(states.back()) - states.front() + 1) {
    lines.push_back(StrCat("  return {DiscreteHiddenState{", states.front(),
                           " + static_cast<int>(packed >> ", position_bits,
                           ")}};"));
  } else {
    values.clear();
    for (int state : states)
      values.push_back(StrCat(state));
    AppendCppArray(lines, "kStates", "int", values);
    lines.push_back(StrCat("  return {DiscreteHiddenState{kStates[packed >> ",
                           position_bits, "]}};"));
  }
  lines.emplace_back("}");
  return JoinCppLines(lines);
}

std::string PlainPositionEmbedding(const CapturedPositionEmbedding& embedding,
                                   absl::string_view name,
                                   const TokenNames& token_names) {
  auto transitions = embedding.transitions;
  std::sort(transitions.begin(), transitions.end());
  std::vector<std::string> rows;
  for (const auto& row : transitions) {
    const int token = row.token, position = row.position, state = row.output;
    rows.push_back(absl::StrCat("{", token_names.at(token), ", ", position,
                                ", {", state, "}}"));
  }
  std::string body =
      "struct EntryRow { DiscreteToken token; int32_t position; "
      "DiscreteHiddenState state; };\n";
  body += CppArray("EntryRow", "kRows", rows);
  absl::StrAppend(
      &body,
      "std::optional<DiscreteHiddenState> Lookup(DiscreteToken token,\n"
      "                                                      int32_t position) "
      "{\n"
      "              size_t first = 0;\n"
      "              size_t last =",
      rows.size(),
      ";\n"
      "                         while (first < last) {\n"
      "                           const size_t middle = first + (last - first) "
      "/ 2;\n"
      "                           const auto& row = kRows[middle];\n"
      "                           if (row.token < token || (row.token == token "
      "&& row.position < position))\n"
      "                             first = middle + 1;\n"
      "                           else\n"
      "                             last = middle;\n"
      "                         }\n"
      "  if (first ==",
      rows.size(),
      " || kRows[first].token != token ||\n"
      "      kRows[first].position != position)\n"
      "    return {};\n"
      "  return {kRows[first].state};\n"
      "}\n");
  return absl::StrReplaceAll(body,
                             {{" Lookup(", absl::StrCat(" ", name, "(")}});
}

}  // namespace

absl::StatusOr<SerializedCppProgram> RenderPositionEmbedding(
    const CapturedPositionEmbedding& embedding, absl::string_view name,
    const TokenNames& token_names, bool compact) {
  RETURN_IF_ERROR(ValidateTransitionFunctionName(name));
  ASSIGN_OR_RETURN(auto mapping, MakeEntryMapping(embedding.transitions));
  for (const auto& [key, output] : mapping)
    if (!token_names.contains(key.first))
      return absl::InvalidArgumentError("missing named entry token");
  if (!compact)
    return SerializedCppProgram{
        PlainPositionEmbedding(embedding, name, token_names)};
  ASSIGN_OR_RETURN(auto source,
                   RenderEntry(name, embedding.transitions, token_names));
  return SerializedCppProgram{std::move(source)};
}

}  // namespace pluto::llm::discretized::generator
