#include "src/llm/experiments/memorize_general_facts/discretized_model/discrete_model_generator/mlp_pair_compaction.h"

#include <algorithm>
#include <cstdint>
#include <map>
#include <set>
#include <vector>

#include "absl/strings/str_cat.h"
#include "src/llm/experiments/memorize_general_facts/discretized_model/discrete_model_generator/captured_model_util.h"
#include "src/util/status_macros.h"

namespace pluto::llm::discretized::generator {

absl::StatusOr<CapturedModel> CompactMlpPairs(const CapturedModel& model) {
  RETURN_IF_ERROR(ValidateModel(model));
  const bool any_shared = std::any_of(
      model.states.begin(), model.states.end(), [](const CapturedState& state) {
        return state.shared_boundary.has_value();
      });
  if (any_shared) {
    for (const auto& state : model.states) {
      if (state.boundary != 0 && !state.shared_boundary)
        return absl::FailedPreconditionError(
            "MLP pair compaction does not accept a partially paired model");
      // A shared ID cannot reconstruct either side's original identity. Even
      // the idempotent path must preserve a complete existing membership map.
      if (!state.members ||
          (state.shared_boundary && state.members->size() < 2))
        return absl::FailedPreconditionError(
            "paired models require complete original-state membership");
    }
    RETURN_IF_ERROR(EvaluateModel(model).status());
    return model;
  }

  std::vector<std::set<int>> boundaries(2 * model.metadata.layers + 1);
  std::map<int, CapturedState> states;
  for (const auto& state : model.states) {
    boundaries[state.boundary].insert(state.id);
    states.emplace(state.id, state);
  }
  // Complete provenance is essential: both original vectors must survive an
  // x/y identification even though the generated model uses only one symbol.
  // Untouched captures may omit explicit singleton memberships; a model that
  // has already changed its IDs cannot safely infer missing original IDs.
  const bool implicit_singletons = model.stats.state_compactions == 0 &&
                                   model.state_relabeling.empty() &&
                                   !model.stats.pointwise_relabeling;
  for (auto& [id, state] : states)
    if (!state.members) {
      if (!implicit_singletons)
        return absl::FailedPreconditionError(
            "MLP pair compaction requires complete original-state membership");
      state.members = std::vector<int>{id};
    }

  std::map<int, int> replacement;
  for (int layer = 0; layer < model.metadata.layers; ++layer) {
    const auto& rows = model.transformers[layer].mlp.transitions;
    std::set<int> inputs, outputs;
    for (const auto& row : rows) {
      inputs.insert(row.input);
      outputs.insert(row.output);
    }
    if (inputs != boundaries[2 * layer + 1] ||
        outputs != boundaries[2 * layer + 2] || outputs.size() != rows.size())
      return absl::FailedPreconditionError(absl::StrCat(
          "MLP ", layer,
          " must be a complete bijection before MLP pair compaction"));
    for (const auto& row : rows) {
      // Keep x's ID and let y use it too. This is a bijective change of the
      // next boundary's alphabet, not the claim that the two neural vectors
      // were numerically equal. The intervening layer is still represented.
      auto& input = states.at(row.input);
      const auto& output = states.at(row.output);
      input.shared_boundary = 2 * layer + 2;
      input.members->insert(input.members->end(), output.members->begin(),
                            output.members->end());
      std::sort(input.members->begin(), input.members->end());
      replacement.emplace(row.output, row.input);
    }
  }
  RETURN_IF_ERROR(EvaluateModel(model).status());
  auto rename = [&](int state) {
    const auto found = replacement.find(state);
    return found == replacement.end() ? state : found->second;
  };
  CapturedModel result = model;
  result.states.clear();
  int64_t membership_count = 0;
  for (const auto& original : model.states)
    if (!replacement.contains(original.id)) {
      result.states.push_back(states.at(original.id));
      membership_count += result.states.back().members->size();
    }
  for (auto& row : result.position_embedding.transitions)
    row.output = rename(row.output);
  for (auto& transformer : result.transformers) {
    for (auto& row : transformer.attention.transitions) {
      for (int& input : row.prefix)
        input = rename(input);
      row.output = rename(row.output);
    }
    std::sort(transformer.attention.transitions.begin(),
              transformer.attention.transitions.end());
    for (auto& row : transformer.mlp.transitions) {
      row.input = rename(row.input);
      row.output = rename(row.output);
    }
    std::sort(transformer.mlp.transitions.begin(),
              transformer.mlp.transitions.end());
  }
  for (auto& row : result.language_modeling_head.transitions)
    row.input = rename(row.input);  // The output is a fixed vocabulary ID.
  std::sort(result.language_modeling_head.transitions.begin(),
            result.language_modeling_head.transitions.end());
  result.state_relabeling.clear();
  result.stats.pointwise_relabeling.reset();
  result.stats.mlp_pair_compactions += replacement.size();
  result.stats.states = result.states.size();
  result.stats.states_per_stage.assign(boundaries.size(), 0);
  for (const auto& state : result.states) {
    ++result.stats.states_per_stage[state.boundary];
    if (state.shared_boundary)
      ++result.stats.states_per_stage[*state.shared_boundary];
  }
  result.stats.membership_complete = true;
  result.stats.membership_original_states = membership_count;
  RETURN_IF_ERROR(ValidateModel(result));
  ASSIGN_OR_RETURN(result.stats.verification, EvaluateModel(result));
  return result;
}

}  // namespace pluto::llm::discretized::generator
