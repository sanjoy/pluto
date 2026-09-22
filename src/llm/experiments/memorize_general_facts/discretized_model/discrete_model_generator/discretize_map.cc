#include "src/llm/experiments/memorize_general_facts/discretized_model/discrete_model_generator/discretize_map.h"

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
#include "absl/strings/str_replace.h"
#include "src/llm/experiments/memorize_general_facts/discretized_model/discrete_model_generator/utils.h"
#include "src/util/status_macros.h"

namespace pluto::llm::discretized::generator {
namespace {

using absl::StrCat;
using Lines = std::vector<std::string>;
using Mapping = std::map<int, int>;
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

std::string Name(int value, const TokenNames* names) {
  return names == nullptr ? StrCat(value) : names->find(value)->second;
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

absl::StatusOr<std::string> RenderCompactMap(
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
    return JoinCppLines(begin);
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
    return JoinCppLines(lines);
  }
  std::optional<std::string> affine_candidate;
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
    AppendCppArray(lines, "kSupport", "uint8_t", values, 16);
    lines.push_back(
        StrCat("  const uint32_t offset = state.value - ", low, ";"));
    lines.emplace_back(
        "  if ((kSupport[offset >> 3] & (uint32_t{1} << (offset & 7u))) == 0) "
        "return {};");
    lines.insert(lines.end(), {affine_return(), "}"});
    affine_candidate = JoinCppLines(lines);
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
  for (auto [first, last, delta] : runs) {
    if (last - first >= 2) {
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
  auto branch = JoinCppLines(branches);
  if (affine_candidate.has_value() && affine_candidate->size() < branch.size())
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
    AppendCppArray(vector, "kOutputs", small ? "uint16_t" : "int", values);
    vector.push_back(StrCat(
        "  return {DiscreteHiddenState{kOutputs[state.value - ", low, "]}};"));
    vector.emplace_back("}");
    auto candidate = JoinCppLines(vector);
    if (candidate.size() < branch.size())
      return candidate;
  }
  return branch;
}

std::string PlainMap(const CapturedMap& map, absl::string_view name,
                     const TokenNames* token_names) {
  auto rows = map.transitions;
  std::sort(rows.begin(), rows.end());
  std::vector<std::string> values;
  for (const auto& row : rows) {
    const int input = row.input, output = row.output;
    values.push_back(token_names
                         ? absl::StrCat("{{", input,
                                        "}, static_cast<DiscreteHiddenState>(",
                                        token_names->at(output), ")}")
                         : absl::StrCat("{{", input, "}, {", output, "}}"));
  }
  std::string body =
      "struct StateRow { DiscreteHiddenState input; DiscreteHiddenState "
      "output; };\n";
  body += CppArray("StateRow", "kRows", values);
  absl::StrAppend(
      &body,
      "std::optional<DiscreteHiddenState> Lookup(\n"
      "                                   DiscreteHiddenState state) {\n"
      "                                 size_t first = 0;\n"
      "                                 size_t last =",
      rows.size(),
      ";\n"
      "                                     while (first < last) {\n"
      "                                       const size_t middle = first + "
      "(last - first) / 2;\n"
      "                                       if (kRows[middle].input < "
      "state)\n"
      "                                         first = middle + 1;\n"
      "                                       else\n"
      "                                         last = middle;\n"
      "                                     }\n"
      "  if (first ==",
      rows.size(),
      " || kRows[first].input != state)\n"
      "    return {};\n"
      "  return {kRows[first].output};\n"
      "}\n");
  return absl::StrReplaceAll(body,
                             {{" Lookup(", absl::StrCat(" ", name, "(")}});
}
}  // namespace

absl::StatusOr<SerializedCppProgram> RenderMap(const CapturedMap& map,
                                               absl::string_view name,
                                               const TokenNames* token_names,
                                               bool compact) {
  RETURN_IF_ERROR(ValidateTransitionFunctionName(name));
  ASSIGN_OR_RETURN(auto mapping, MakeMapping(map.transitions));
  if (token_names != nullptr)
    for (const auto& [input, output] : mapping)
      if (!token_names->contains(output))
        return absl::InvalidArgumentError("missing named pointwise output");
  if (!compact)
    return SerializedCppProgram{PlainMap(map, name, token_names)};
  ASSIGN_OR_RETURN(auto source,
                   RenderCompactMap(name, map.transitions, token_names));
  return SerializedCppProgram{std::move(source)};
}

}  // namespace pluto::llm::discretized::generator
