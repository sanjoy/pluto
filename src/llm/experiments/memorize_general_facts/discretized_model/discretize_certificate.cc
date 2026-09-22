#include "src/llm/experiments/memorize_general_facts/discretized_model/discretize_certificate.h"

#include <algorithm>
#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <utility>
#include <vector>

#include "src/llm/experiments/memorize_general_facts/discretized_model/generator_model_validation.h"
#include "src/util/status_macros.h"

namespace pluto::llm::discretized::generator {
namespace {
using StateTable = std::map<int, int>;
using AttentionTable = std::map<std::vector<int>, int>;

int64_t PairCount(const std::vector<int>& states) {
  const int64_t count = states.size();
  return count * (count - 1) / 2;
}

std::optional<std::pair<int, int>> DuplicateOutput(const StateTable& table) {
  std::map<int, int> seen;
  for (const auto& [state, output] : table) {
    auto [found, inserted] = seen.emplace(output, state);
    if (!inserted)
      return std::pair(found->second, state);
  }
  return std::nullopt;
}

bool Covers(const StateTable& table, const std::vector<int>& states) {
  if (table.size() != states.size())
    return false;
  for (int state : states)
    if (!table.contains(state))
      return false;
  return true;
}
}  // namespace

absl::StatusOr<CertificateResult> CertifyCompaction(
    const SymbolicModel& model) {
  RETURN_IF_ERROR(internal::ValidateTables(model, false));
  const int layers = model.metadata.layers;
  std::vector<std::vector<int>> by_stage(2 * layers + 1);
  for (const auto& row : model.states)
    by_stage[row.boundary].push_back(row.id);
  int64_t pairs = 0;
  for (auto& states : by_stage) {
    std::sort(states.begin(), states.end());
    pairs += PairCount(states);
  }
  CertificateResult report{.states = static_cast<int64_t>(model.states.size()),
                           .boundaries = static_cast<int64_t>(by_stage.size()),
                           .same_boundary_pairs = pairs};
  auto inconclusive = [&](const std::string& reason, int stage,
                          std::optional<std::pair<int, int>> pair =
                              std::nullopt) {
    report.reason = reason;
    report.unresolved_stage = stage;
    if (pair)
      report.unresolved_pair = std::array<int, 2>{pair->first, pair->second};
    return report;
  };
  auto proven = [&](int stage, const std::string& argument) {
    const int64_t count = PairCount(by_stage[stage]);
    report.proven_pairs += count;
    report.stages.push_back(
        {.stage = stage,
         .states = static_cast<int64_t>(by_stage[stage].size()),
         .pairs = count,
         .argument = argument});
  };
  StateTable head;
  for (const auto& row : model.language_modeling_head)
    head.emplace(row.input, row.output);
  const int final = by_stage.size() - 1;
  if (!Covers(head, by_stage[final]))
    return inconclusive("not every final state has a fixed readout label",
                        final);
  if (auto duplicate = DuplicateOutput(head))
    return inconclusive("two final states have the same fixed token label",
                        final, duplicate);
  proven(final, "distinct fixed token labels");
  for (int layer = layers - 1; layer >= 0; --layer) {
    int stage = 2 * layer + 1;
    StateTable mlp;
    for (const auto& row : model.transformers[layer].mlp)
      mlp.emplace(row.input, row.output);
    if (!Covers(mlp, by_stage[stage]))
      return inconclusive("MLP table does not cover every input state", stage);
    if (auto duplicate = DuplicateOutput(mlp))
      return inconclusive("MLP is not injective into distinguishable outputs",
                          stage, duplicate);
    proven(stage,
           "injective MLP into already-distinguishable downstream states");
    --stage;
    AttentionTable table;
    for (const auto& row : model.transformers[layer].attention)
      table.emplace(row.prefix, row.output);
    std::map<int, std::vector<const AttentionTable::value_type*>> uses;
    for (const auto& row : table) {
      const std::set<int> unique(row.first.begin(), row.first.end());
      for (int state : unique)
        uses[state].push_back(&row);
    }
    const auto& states = by_stage[stage];
    for (size_t first_index = 0; first_index < states.size(); ++first_index)
      for (size_t second_index = first_index + 1; second_index < states.size();
           ++second_index) {
        const int first = states[first_index], second = states[second_index];
        AttentionTable rewritten;
        bool witnessed = false;
        ++report.attention_pairs_checked;
        // Only histories containing first change under first->second. Check
        // both unchanged histories and other rewritten histories; either can
        // witness a contradiction. Length and order always remain significant.
        for (const auto* row : uses[first]) {
          auto normalized = row->first;
          std::replace(normalized.begin(), normalized.end(), first, second);
          auto changed = rewritten.find(normalized);
          auto original = table.find(normalized);
          const int other = changed != rewritten.end() ? changed->second
                            : original != table.end()  ? original->second
                                                       : -1;
          if (other >= 0 && other != row->second) {
            witnessed = true;
            break;
          }
          rewritten.emplace(std::move(normalized), row->second);
        }
        if (!witnessed)
          return inconclusive(
              "state compaction has no contradictory attention-key "
              "collision",
              stage, std::pair(first, second));
      }
    proven(stage,
           "every pair collides on already-distinguishable attention outputs");
  }
  report.status = CertificateStatus::kProven;
  report.pairwise_compaction_complete = true;
  return report;
}
}  // namespace pluto::llm::discretized::generator
