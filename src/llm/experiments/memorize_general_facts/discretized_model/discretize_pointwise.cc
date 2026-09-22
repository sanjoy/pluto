#include "src/llm/experiments/memorize_general_facts/discretized_model/discretize_pointwise.h"

#include <algorithm>
#include <bit>
#include <cstdint>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <tuple>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "src/util/status_macros.h"

namespace pluto::llm::discretized::generator {
namespace {

using absl::StrCat;
using Lines = std::vector<std::string>;
using Mapping = std::map<int, int>;
using EntryMapping = std::map<std::pair<int, int>, int>;
constexpr int kMaxId = std::numeric_limits<int32_t>::max();

absl::StatusOr<Mapping> MakeMapping(absl::Span<const StateTransition> rows) {
  Mapping result;
  for (const auto& row : rows) {
    if (row.input < 0 || row.output < 0)
      return absl::InvalidArgumentError(
          "pointwise IDs must be nonnegative and fit int");
    int source = row.input, output = row.output;
    auto [it, inserted] = result.emplace(source, output);
    if (!inserted && it->second != output)
      return absl::InvalidArgumentError("conflicting pointwise mapping");
  }
  return result;
}

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

void Array(Lines& lines, absl::string_view name, absl::string_view type,
           const Lines& values, size_t columns = 12) {
  lines.push_back(StrCat("  static constexpr ", type, " ", name, "[] = {"));
  for (size_t offset = 0; offset < values.size(); offset += columns) {
    auto end = std::min(values.size(), offset + columns);
    lines.push_back(StrCat(
        "    ",
        absl::StrJoin(values.begin() + offset, values.begin() + end, ", "),
        ","));
  }
  lines.emplace_back("  };");
}

RenderedTransition Finish(const Lines& lines, TransitionStatistics stats) {
  std::string body = StrCat(absl::StrJoin(lines, "\n"), "\n");
  stats.source_bytes = body.size();
  return {std::move(body), std::move(stats)};
}

std::string AffineExpression(int source, int output, const TokenNames* names) {
  if (names != nullptr)
    return StrCat(Name(output, names), ".value + (state.value - ", source, ")");
  int64_t delta = static_cast<int64_t>(output) - source;
  if (delta == 0)
    return "state.value";
  return StrCat("state.value ", delta > 0 ? "+ " : "- ",
                delta > 0 ? delta : -delta);
}

std::set<int> Keys(const Mapping& mapping) {
  std::set<int> result;
  for (auto [input, output] : mapping)
    result.insert(input);
  return result;
}

std::set<int> Values(const Mapping& mapping) {
  std::set<int> result;
  for (auto [input, output] : mapping)
    result.insert(output);
  return result;
}

}  // namespace

absl::StatusOr<std::optional<int>> EvaluatePointwise(
    absl::Span<const StateTransition> rows, int state) {
  ASSIGN_OR_RETURN(auto mapping, MakeMapping(rows));
  auto found = mapping.find(state);
  if (found == mapping.end())
    return std::optional<int>{};
  return std::optional<int>{found->second};
}

absl::StatusOr<std::optional<int>> EvaluateEntry(
    absl::Span<const EntryTransition> rows, int token, int position) {
  ASSIGN_OR_RETURN(auto mapping, MakeEntryMapping(rows));
  auto found = mapping.find({token, position});
  if (found == mapping.end())
    return std::optional<int>{};
  return std::optional<int>{found->second};
}

absl::StatusOr<SymbolicModel> RelabelMlpOutputs(const SymbolicModel& model) {
  if (model.metadata.layers < 0 || model.metadata.vocab_size <= 0)
    return absl::InvalidArgumentError("malformed model for MLP relabeling");
  int layers = model.metadata.layers, vocab_size = model.metadata.vocab_size;
  if (static_cast<size_t>(layers) != model.transformers.size() ||
      layers > kMaxId / 2)
    return absl::InvalidArgumentError("inconsistent layer count");
  std::map<int, SymbolicState> states;
  std::map<int, std::set<int>> stage_states;
  Mapping renaming;
  for (const auto& row : model.states) {
    if (row.id < 0 || row.boundary < 0)
      return absl::InvalidArgumentError("malformed state for MLP relabeling");
    int id = row.id, stage = row.boundary;
    if (stage > 2 * layers)
      return absl::InvalidArgumentError("state stage exceeds model boundaries");
    if (!states.emplace(id, row).second)
      return absl::InvalidArgumentError("duplicate state ID");
    stage_states[stage].insert(id);
    renaming[id] = id;
  }
  std::vector<Mapping> mlp;
  for (const auto& transformer : model.transformers) {
    ASSIGN_OR_RETURN(auto mapping, MakeMapping(transformer.mlp));
    for (auto [input, output] : mapping)
      if (!states.contains(input) || !states.contains(output))
        return absl::InvalidArgumentError("MLP references missing state");
    mlp.push_back(std::move(mapping));
  }
  ASSIGN_OR_RETURN(auto head, MakeMapping(model.language_modeling_head));
  for (auto [input, output] : head)
    if (!states.contains(input))
      return absl::InvalidArgumentError(
          "language modeling head references missing state");
  ASSIGN_OR_RETURN(auto entry, MakeEntryMapping(model.entry));
  for (const auto& [key, output] : entry)
    if (!states.contains(output))
      return absl::InvalidArgumentError("entry references missing state");
  for (const auto& transformer : model.transformers) {
    ASSIGN_OR_RETURN(auto program, BuildAttention(transformer.attention));
    for (const auto& row : transformer.attention) {
      if (!states.contains(row.output))
        return absl::InvalidArgumentError("attention references missing state");
      for (int input : row.prefix)
        if (!states.contains(input))
          return absl::InvalidArgumentError(
              "attention references missing state");
    }
  }
  RelabelStatistics stats;
  for (int layer = 0; layer < layers; ++layer) {
    auto inputs = Keys(mlp[layer]), outputs = Values(mlp[layer]);
    if (inputs.empty() || inputs.size() != outputs.size() ||
        inputs.size() !=
            static_cast<size_t>(*inputs.rbegin()) - *inputs.begin() + 1 ||
        outputs.size() !=
            static_cast<size_t>(*outputs.rbegin()) - *outputs.begin() + 1 ||
        inputs != stage_states[2 * layer + 1] ||
        outputs != stage_states[2 * layer + 2]) {
      stats.skipped_layers.push_back(layer);
      continue;
    }
    stats.eligible_layers.push_back(layer);
    for (auto [input, output] : mlp[layer])
      renaming[output] =
          static_cast<int64_t>(*outputs.begin()) + input - *inputs.begin();
  }
  if (layers > 0) {
    const auto& before = stage_states[2 * layers - 1];
    const auto& after = stage_states[2 * layers];
    const auto labels = Values(head);
    bool complete = !before.empty() && Keys(mlp.back()) == before &&
                    Values(mlp.back()) == after &&
                    before.size() == after.size() && Keys(head) == after &&
                    labels.size() == after.size() &&
                    std::all_of(labels.begin(), labels.end(),
                                [&](int token) { return token < vocab_size; });
    int64_t prefix_max = vocab_size - 1;
    for (const auto& [state, row] : states)
      if (row.boundary < 2 * layers - 1)
        prefix_max = std::max(prefix_max, static_cast<int64_t>(state));
    int64_t attention_base = prefix_max + 1;
    int64_t final_base = attention_base + vocab_size;
    if (complete && final_base + vocab_size - 1 <= kMaxId) {
      for (auto [state, output] : mlp.back())
        renaming[state] = attention_base + head[output];
      for (auto [state, token] : head)
        renaming[state] = final_base + token;
      stats.vocabulary_aligned_final_boundaries = true;
      stats.last_attention_base = attention_base;
      stats.final_state_base = final_base;
      stats.reserved_range_size = vocab_size;
    }
  }
  SymbolicModel result = model;
  for (auto& row : result.states)
    row.id = renaming.at(row.id);
  std::sort(result.states.begin(), result.states.end(),
            [](const SymbolicState& left, const SymbolicState& right) {
              return std::pair{left.boundary, left.id} <
                     std::pair{right.boundary, right.id};
            });
  for (auto& row : result.entry)
    row.output = renaming.at(row.output);
  std::sort(result.entry.begin(), result.entry.end());
  for (auto& transformer : result.transformers) {
    for (auto& row : transformer.attention) {
      for (int& state : row.prefix)
        state = renaming.at(state);
      row.output = renaming.at(row.output);
    }
    std::sort(transformer.attention.begin(), transformer.attention.end());
    for (auto& row : transformer.mlp) {
      row.input = renaming.at(row.input);
      row.output = renaming.at(row.output);
    }
    std::sort(transformer.mlp.begin(), transformer.mlp.end());
  }
  for (auto& row : result.language_modeling_head)
    row.input = renaming.at(row.input);
  std::sort(result.language_modeling_head.begin(),
            result.language_modeling_head.end());
  result.state_relabeling.clear();
  for (auto [old_id, new_id] : renaming) {
    stats.changed_states += old_id != new_id;
    result.state_relabeling.push_back(
        {old_id, new_id, states.at(old_id).boundary});
  }
  result.stats.pointwise_relabeling = std::move(stats);
  return result;
}

absl::StatusOr<RenderedTransition> RenderPointwise(
    absl::string_view name, absl::Span<const StateTransition> rows,
    const TokenNames* token_names) {
  RETURN_IF_ERROR(ValidateTransitionFunctionName(name));
  ASSIGN_OR_RETURN(auto mapping, MakeMapping(rows));
  if (token_names != nullptr)
    for (auto [input, output] : mapping)
      if (!token_names->contains(output))
        return absl::InvalidArgumentError("missing named pointwise output");
  Lines begin = {StrCat("std::optional<DiscreteHiddenState> ", name,
                        "(DiscreteHiddenState state) {")};
  if (mapping.empty()) {
    begin.insert(begin.end(), {"  (void)state;", "  return {};", "}"});
    return Finish(begin, {.representation = TransitionRepresentation::kEmpty,
                          .rows = 0,
                          .table_bytes = 0});
  }
  int low = mapping.begin()->first, high = mapping.rbegin()->first;
  const std::string guard = StrCat("  if (state.value < ", low,
                                   " || state.value > ", high, ") return {};");
  int64_t first_delta = static_cast<int64_t>(mapping.begin()->second) - low;
  bool affine =
      std::all_of(mapping.begin(), mapping.end(), [&](const auto& row) {
        return static_cast<int64_t>(row.second) - row.first == first_delta;
      });
  bool dense = mapping.size() == static_cast<size_t>(high) - low + 1;
  auto add_named_comment = [&](Lines& lines) {
    if (token_names != nullptr)
      lines.emplace_back(
          "  // State labels encode vocabulary IDs; no neural-head linearity "
          "is implied.");
  };
  auto affine_return = [&] {
    return StrCat("  return {DiscreteHiddenState{",
                  AffineExpression(low, mapping.begin()->second, token_names),
                  "}};");
  };
  if (affine && dense) {
    Lines lines = begin;
    lines.push_back(guard);
    add_named_comment(lines);
    lines.insert(lines.end(), {affine_return(), "}"});
    return Finish(lines,
                  {.representation = TransitionRepresentation::kGuardedAffine,
                   .rows = mapping.size(),
                   .table_bytes = 0,
                   .affine_ranges = 1,
                   .named_anchor = token_names != nullptr});
  }
  std::optional<RenderedTransition> affine_candidate;
  int64_t support_bytes = (static_cast<int64_t>(high) - low + 8) / 8;
  if (affine &&
      support_bytes <= std::min<int64_t>(1'048'576, mapping.size() * 8)) {
    std::vector<uint8_t> support(support_bytes);
    for (auto [state, output] : mapping) {
      int index = state - low;
      support[index / 8] |= 1 << (index % 8);
    }
    Lines lines = begin;
    lines.push_back(guard);
    add_named_comment(lines);
    Lines values;
    for (uint8_t value : support)
      values.push_back(absl::StrFormat("0x%02xu", value));
    Array(lines, "kSupport", "uint8_t", values, 16);
    lines.push_back(
        StrCat("  const uint32_t offset = state.value - ", low, ";"));
    lines.emplace_back(
        "  if ((kSupport[offset >> 3] & (uint32_t{1} << (offset & 7u))) == 0) "
        "return {};");
    lines.insert(lines.end(), {affine_return(), "}"});
    affine_candidate = Finish(
        lines,
        {.representation = TransitionRepresentation::kSparseAffineSupportMask,
         .rows = mapping.size(),
         .table_bytes = support_bytes,
         .supported_span = static_cast<int64_t>(high) - low + 1,
         .named_anchor = token_names != nullptr});
  }
  struct Run {
    int first, last;
    int64_t delta;
  };
  std::vector<Run> runs;
  for (auto [source, output] : mapping) {
    int64_t delta = static_cast<int64_t>(output) - source;
    if (token_names == nullptr && !runs.empty() &&
        static_cast<int64_t>(source) ==
            static_cast<int64_t>(runs.back().last) + 1 &&
        delta == runs.back().delta)
      runs.back().last = source;
    else
      runs.push_back({source, source, delta});
  }
  Lines branches = begin;
  branches.push_back(guard);
  Mapping singletons;
  int affine_ranges = 0;
  for (auto [first, last, delta] : runs) {
    if (last - first >= 2) {
      ++affine_ranges;
      Lines conditions;
      if (first != 0)
        conditions.push_back(StrCat("state.value >= ", first));
      if (last < kMaxId)
        conditions.push_back(StrCat("state.value <= ", last));
      std::string condition =
          conditions.empty() ? "true" : absl::StrJoin(conditions, " && ");
      branches.push_back(
          StrCat("  if (", condition, ") return {DiscreteHiddenState{",
                 AffineExpression(first, first + delta, nullptr), "}};"));
    } else {
      for (int64_t state = first; state <= last; ++state)
        singletons[state] = mapping.find(state)->second;
    }
  }
  if (!singletons.empty()) {
    branches.emplace_back("  switch (state.value) {");
    for (auto [source, output] : singletons) {
      std::string expression =
          token_names != nullptr ? StrCat("static_cast<DiscreteHiddenState>(",
                                          Name(output, token_names), ")")
                                 : StrCat("DiscreteHiddenState{", output, "}");
      branches.push_back(
          StrCat("    case ", source, ": return {", expression, "};"));
    }
    branches.insert(branches.end(), {"    default: return {};", "  }"});
  } else {
    branches.emplace_back("  return {};");
  }
  branches.emplace_back("}");
  auto branch = Finish(
      branches,
      {.representation = TransitionRepresentation::kAffineRangesAndSwitch,
       .rows = mapping.size(),
       .table_bytes = 0,
       .affine_ranges = affine_ranges,
       .switch_cases = singletons.size()});
  if (affine_candidate.has_value() &&
      affine_candidate->source.size() < branch.source.size())
    return *affine_candidate;
  if (dense) {
    bool small =
        std::all_of(mapping.begin(), mapping.end(),
                    [](const auto& row) { return row.second <= 65535; });
    Lines vector = begin;
    vector.push_back(guard);
    Lines values;
    for (auto [input, output] : mapping)
      values.push_back(token_names != nullptr
                           ? StrCat(Name(output, token_names), ".value")
                           : StrCat(output));
    Array(vector, "kOutputs", small ? "uint16_t" : "int", values);
    vector.push_back(StrCat(
        "  return {DiscreteHiddenState{kOutputs[state.value - ", low, "]}};"));
    vector.emplace_back("}");
    auto candidate =
        Finish(vector,
               {.representation = TransitionRepresentation::kGuardedOutputArray,
                .rows = mapping.size(),
                .table_bytes = mapping.size() * (small ? 2 : 4),
                .named_outputs = token_names != nullptr});
    if (candidate.source.size() < branch.source.size())
      return candidate;
  }
  return branch;
}

absl::StatusOr<RenderedTransition> RenderEntry(
    absl::string_view name, absl::Span<const EntryTransition> rows,
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
    return Finish(begin, {.representation = TransitionRepresentation::kEmpty,
                          .rows = 0,
                          .table_bytes = 0});
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
    return Finish(
        lines,
        {.representation = TransitionRepresentation::kExactTokenPositionSwitch,
         .rows = entry.size(),
         .table_bytes = 0,
         .tokens = tokens.size()});
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
  Array(lines, "kTokenPatterns", index_type, values);
  values.clear();
  for (uint64_t pattern : patterns)
    values.push_back(absl::StrFormat("0x%x%s", pattern, suffix));
  Array(lines, "kPatterns", word_type, values, 8);
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
  size_t state_bytes = 0;
  if (states.size() ==
      static_cast<size_t>(states.back()) - states.front() + 1) {
    lines.push_back(StrCat("  return {DiscreteHiddenState{", states.front(),
                           " + static_cast<int>(packed >> ", position_bits,
                           ")}};"));
  } else {
    values.clear();
    for (int state : states)
      values.push_back(StrCat(state));
    Array(lines, "kStates", "int", values);
    state_bytes = 4 * states.size();
    lines.push_back(StrCat("  return {DiscreteHiddenState{kStates[packed >> ",
                           position_bits, "]}};"));
  }
  lines.emplace_back("}");
  return Finish(
      lines, {.representation =
                  TransitionRepresentation::kPackedSupportPatternsAndExceptions,
              .rows = entry.size(),
              .table_bytes = index_bytes * indices.size() +
                             word_bytes * patterns.size() + state_bytes,
              .tokens = tokens.size(),
              .patterns = patterns.size(),
              .position_bits = position_bits,
              .token_only_defaults = tokens.size(),
              .position_exceptions = exceptions.size(),
              .named_exception_tokens = true});
}

}  // namespace pluto::llm::discretized::generator
