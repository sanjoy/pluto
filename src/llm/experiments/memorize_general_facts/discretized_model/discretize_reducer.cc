#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <deque>
#include <map>
#include <mutex>
#include <numeric>
#include <set>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/memory/memory.h"
#include "src/llm/experiments/memorize_general_facts/discretized_model/discretize_core.h"
#include "src/llm/experiments/memorize_general_facts/discretized_model/generator_model_validation.h"
#include "src/util/status_macros.h"

namespace pluto::llm::discretized::generator {
namespace {
double Distance(const Json& first, const Json& second) {
  // BF16 values are search coordinates only. Runtime transitions remain exact
  // integers, and even distant states can merge in the exhaustive phase.
  long double sum = 0;
  for (size_t index = 0; index < first.size(); ++index) {
    const double a = std::bit_cast<float>(first[index].get<uint32_t>() << 16);
    const double b = std::bit_cast<float>(second[index].get<uint32_t>() << 16);
    sum += (a - b) * (a - b);
  }
  return static_cast<double>(sum);
}

volatile std::sig_atomic_t interrupted = 0;
void Interrupt(int) { interrupted = 1; }

// The signal handler only records intent. All rollback work happens
// synchronously after TryMerge has committed or completely restored its state.
class CooperativeInterrupt {
 public:
  CooperativeInterrupt()
      : lock_(Mutex()), previous_(std::signal(SIGINT, Interrupt)) {
    interrupted = 0;
  }
  ~CooperativeInterrupt() { std::signal(SIGINT, previous_); }
  bool Requested() const { return interrupted != 0; }

 private:
  static std::mutex& Mutex() {
    static std::mutex mutex;
    return mutex;
  }
  std::unique_lock<std::mutex> lock_;
  void (*previous_)(int);
};
}  // namespace

struct QuotientReducer::Impl {
  struct Term {
    int stage;
    std::vector<int> inputs;
    int output;
  };
  struct Undo {
    enum Kind { kDictionary, kTerm, kUses, kUnion } kind;
    int first = 0, second = 0, size = 0, label = 0;
    std::vector<int> values;
  };

  explicit Impl(const Json& source) : model(source), states(source["states"]) {
    const int count = states.size();
    parent.resize(count);
    std::iota(parent.begin(), parent.end(), 0);
    size.assign(count, 1);
    label.assign(count, -1);
    uses.resize(count);
    for (int i = 0; i < count; ++i) {
      index[states[i]["id"].get<int>()] = i;
      stage.push_back(states[i]["stage"].get<int>());
    }
    for (int layer = 0; layer < model["layers"].get<int>(); ++layer) {
      for (const auto& row : model["attention"][layer])
        AddTerm(2 * layer + 1, row[0].get<std::vector<int>>(), row[1]);
      for (const auto& row : model["mlp"][layer])
        AddTerm(2 * layer + 2, {row[0]}, row[1]);
    }
    for (const auto& row : model["language_modeling_head"])
      label[index.at(row[0].get<int>())] = row[1];
  }

  void AddTerm(int boundary, const std::vector<int>& inputs, int output) {
    std::vector<int> arguments;
    for (int state : inputs)
      arguments.push_back(index.at(state));
    std::vector<int> signature = {boundary};
    signature.insert(signature.end(), arguments.begin(), arguments.end());
    const int term = terms.size();
    terms.push_back({boundary, arguments, index.at(output)});
    term_signatures.push_back(signature);
    signatures.emplace(signature, term);
    for (int argument : arguments)
      uses[argument].insert(term);
  }

  int Find(int value) const {
    // No path compression: union by size bounds depth while allowing rollback.
    while (parent[value] != value)
      value = parent[value];
    return value;
  }
  std::vector<int> Roots(int boundary = -1) const {
    std::vector<int> result;
    for (int i = 0; i < static_cast<int>(parent.size()); ++i)
      if (parent[i] == i && (boundary < 0 || stage[i] == boundary))
        result.push_back(i);
    return result;
  }
  int TerminalLabel(int root) const {
    const int final = 2 * model["layers"].get<int>();
    if (stage[root] == final)
      return label[root];
    if (stage[root] == final - 1)
      for (int term : uses[root])
        if (int token = label[Find(terms[term].output)]; token >= 0)
          return token;
    return -1;
  }

  void Rollback(const std::vector<Undo>& undo) {
    for (auto iterator = undo.rbegin(); iterator != undo.rend(); ++iterator) {
      const auto& operation = *iterator;
      switch (operation.kind) {
        case Undo::kDictionary:
          if (operation.first < 0)
            signatures.erase(operation.values);
          else
            signatures[operation.values] = operation.first;
          break;
        case Undo::kTerm:
          term_signatures[operation.first] = operation.values;
          break;
        case Undo::kUses:
          for (int term : operation.values)
            uses[operation.first].erase(term);
          break;
        case Undo::kUnion:
          parent[operation.second] = operation.second;
          size[operation.first] = operation.size;
          label[operation.first] = operation.label;
          break;
      }
    }
  }

  bool Merge(int first, int second) {
    first = Find(first);
    second = Find(second);
    if (first == second)
      return true;
    const std::pair<int, int> pair = std::minmax(first, second);
    if (rejected_pairs.contains(pair)) {
      ++cached_rejections;
      return false;
    }
    const int seed_stage = stage[first];
    const Json seed_ids =
        Json::array({states[first]["id"], states[second]["id"]});
    const double distance =
        std::sqrt(Distance(states[first]["bits"], states[second]["bits"]));
    ++attempted;
    std::deque<std::pair<int, int>> pending{{first, second}};
    std::vector<Undo> undo;
    int count = 0;
    while (!pending.empty()) {
      auto [a, b] = pending.front();
      pending.pop_front();
      a = Find(a);
      b = Find(b);
      if (a == b)
        continue;
      const int a_label = TerminalLabel(a), b_label = TerminalLabel(b);
      if (a_label >= 0 && b_label >= 0 && a_label != b_label) {
        Rollback(undo);
        rejected_pairs.insert(pair);
        return false;
      }
      if (std::pair(size[a], -a) < std::pair(size[b], -b))
        std::swap(a, b);
      std::vector<int> affected(uses[b].begin(), uses[b].end());
      for (int term : affected) {
        const auto& signature = term_signatures[term];
        auto found = signatures.find(signature);
        if (found != signatures.end() && found->second == term) {
          undo.push_back({Undo::kDictionary, term, 0, 0, 0, signature});
          signatures.erase(found);
        }
      }
      undo.push_back({Undo::kUnion, a, b, size[a], label[a], {}});
      parent[b] = a;
      size[a] += size[b];
      if (label[a] < 0)
        label[a] = label[b];
      std::vector<int> added;
      for (int term : uses[b])
        if (uses[a].insert(term).second)
          added.push_back(term);
      undo.push_back({Undo::kUses, a, 0, 0, 0, std::move(added)});
      ++count;
      for (int term : affected) {
        std::vector<int> signature{terms[term].stage};
        for (int argument : terms[term].inputs)
          signature.push_back(Find(argument));
        undo.push_back({Undo::kTerm, term, 0, 0, 0, term_signatures[term]});
        term_signatures[term] = signature;
        auto other = signatures.find(signature);
        if (other == signatures.end()) {
          undo.push_back({Undo::kDictionary, -1, 0, 0, 0, signature});
          signatures.emplace(std::move(signature), term);
        } else {
          // Chase an implication toward fixed token labels promptly. A BFS
          // could otherwise expand thousands of doomed intermediate unions.
          pending.emplace_front(terms[term].output,
                                terms[other->second].output);
        }
      }
    }
    if (count) {
      ++accepted;
      unions += count;
      accepted_merges.push_back({{"stage", seed_stage},
                                 {"seed_ids", seed_ids},
                                 {"euclidean_distance", distance},
                                 {"induced_unions", count - 1}});
    }
    return true;
  }

  Json model;
  Json states;
  absl::flat_hash_map<int, int> index;
  std::vector<int> parent, size, label, stage;
  std::vector<std::set<int>> uses;
  std::vector<Term> terms;
  std::map<std::vector<int>, int> signatures;
  std::vector<std::vector<int>> term_signatures;
  std::set<std::pair<int, int>> rejected_pairs;
  int64_t attempted = 0, accepted = 0, unions = 0, cached_rejections = 0;
  Json accepted_merges = Json::array();
};

QuotientReducer::QuotientReducer(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
QuotientReducer::~QuotientReducer() = default;
absl::StatusOr<std::unique_ptr<QuotientReducer>> QuotientReducer::Create(
    const Json& model) {
  RETURN_IF_ERROR(internal::ValidateTables(model, false));
  return absl::WrapUnique(new QuotientReducer(absl::make_unique<Impl>(model)));
}
absl::StatusOr<bool> QuotientReducer::TryMerge(int first_id, int second_id) {
  auto first = impl_->index.find(first_id),
       second = impl_->index.find(second_id);
  if (first == impl_->index.end() || second == impl_->index.end())
    return absl::InvalidArgumentError("unknown discrete state ID");
  if (impl_->stage[first->second] != impl_->stage[second->second])
    return absl::InvalidArgumentError(
        "states from different boundaries cannot merge");
  return impl_->Merge(first->second, second->second);
}
int QuotientReducer::RootForState(int state_id) const {
  auto found = impl_->index.find(state_id);
  return found == impl_->index.end() ? -1 : impl_->Find(found->second);
}

Json QuotientReducer::Export() const {
  const auto& r = *impl_;
  auto roots = r.Roots();
  std::sort(roots.begin(), roots.end(), [&](int a, int b) {
    return std::pair(r.stage[a], r.states[a]["id"].get<int>()) <
           std::pair(r.stage[b], r.states[b]["id"].get<int>());
  });
  absl::flat_hash_map<int, int> renumber;
  for (int i = 0; i < static_cast<int>(roots.size()); ++i)
    renumber[roots[i]] = r.model["vocab_size"].get<int>() + i;
  auto state_id = [&](int old) { return renumber.at(r.Find(r.index.at(old))); };
  Json result;
  for (const char* key :
       {"schema", "width", "layers", "vocab_size", "eos_token", "prompt_tokens",
        "vocabulary", "samples"})
    if (r.model.contains(key))
      result[key] = r.model[key];
  const Json previous = r.model.value("stats", Json::object());
  bool complete = previous.value("state_unions", int64_t{0}) == 0;
  if (!complete)
    complete =
        std::all_of(r.states.begin(), r.states.end(),
                    [](const Json& row) { return row.contains("members"); });
  std::map<int, std::vector<int>> members;
  if (complete)
    for (int i = 0; i < static_cast<int>(r.states.size()); ++i) {
      auto& group = members[r.Find(i)];
      const auto& row = r.states[i];
      if (row.contains("members"))
        for (const auto& member : row["members"])
          group.push_back(member.get<int>());
      else
        group.push_back(row["id"].get<int>());
    }
  result["states"] = Json::array();
  int64_t original_count = 0;
  for (int root : roots) {
    Json row = {{"id", renumber.at(root)},
                {"stage", r.stage[root]},
                {"bits", r.states[root]["bits"]}};
    if (complete) {
      auto& group = members[root];
      std::sort(group.begin(), group.end());
      row["members"] = group;
      row["member_count"] = group.size();
      original_count += group.size();
    }
    result["states"].push_back(std::move(row));
  }
  result["entry"] = Json::array();
  for (const auto& row : r.model["entry"])
    result["entry"].push_back({row[0], row[1], state_id(row[2])});
  result["attention"] = Json::array();
  result["mlp"] = Json::array();
  const int layers = r.model["layers"];
  for (int layer = 0; layer < layers; ++layer) {
    std::map<std::vector<int>, int> attention;
    std::map<int, int> mlp;
    for (const auto& row : r.model["attention"][layer]) {
      std::vector<int> key;
      for (const auto& state : row[0])
        key.push_back(state_id(state));
      attention[key] = state_id(row[1]);
    }
    for (const auto& row : r.model["mlp"][layer])
      mlp[state_id(row[0])] = state_id(row[1]);
    Json attention_rows = Json::array(), mlp_rows = Json::array();
    for (const auto& [key, output] : attention)
      attention_rows.push_back({key, output});
    for (const auto& [key, output] : mlp)
      mlp_rows.push_back({key, output});
    result["attention"].push_back(std::move(attention_rows));
    result["mlp"].push_back(std::move(mlp_rows));
  }
  std::map<int, int> head;
  for (const auto& row : r.model["language_modeling_head"])
    head[state_id(row[0])] = row[1];
  result["language_modeling_head"] = Json::array();
  for (const auto& [key, output] : head)
    result["language_modeling_head"].push_back({key, output});
  result["stats"] = previous;
  auto& stats = result["stats"];
  stats["states"] = roots.size();
  stats["membership_complete"] = complete;
  stats["membership_original_states"] =
      complete ? Json(original_count) : Json(nullptr);
  std::vector<int> counts(2 * layers + 1);
  for (int root : roots)
    ++counts[r.stage[root]];
  stats["states_per_stage"] = counts;
  stats["attempted_seeds"] =
      previous.value("attempted_seeds", int64_t{0}) + r.attempted;
  stats["accepted_seeds"] =
      previous.value("accepted_seeds", int64_t{0}) + r.accepted;
  stats["state_unions"] = previous.value("state_unions", int64_t{0}) + r.unions;
  stats["cached_rejections"] =
      previous.value("cached_rejections", int64_t{0}) + r.cached_rejections;
  stats["accepted_merges"] = previous.value("accepted_merges", Json::array());
  for (const auto& merge : r.accepted_merges)
    stats["accepted_merges"].push_back(merge);
  return result;
}

namespace {
using Candidate = std::tuple<double, int, int>;
std::vector<Candidate> CandidatePairs(const QuotientReducer::Impl& r, int stage,
                                      int neighbors, bool exhaustive) {
  auto roots = r.Roots(stage);
  std::vector<int> labels;
  std::map<int, std::vector<int>> groups;
  for (int i = 0; i < static_cast<int>(roots.size()); ++i) {
    labels.push_back(r.TerminalLabel(roots[i]));
    groups[labels.back()].push_back(i);
  }
  std::set<std::pair<int, int>> pairs;
  if (exhaustive) {
    for (const auto& [label, group] : groups)
      for (int i = 0; i < static_cast<int>(group.size()); ++i)
        for (int j = i + 1; j < static_cast<int>(group.size()); ++j)
          pairs.insert(std::minmax(group[i], group[j]));
    for (int first : groups[-1])
      for (int second = 0; second < static_cast<int>(roots.size()); ++second)
        if (labels[second] != -1)
          pairs.insert(std::minmax(first, second));
  } else {
    // Same dependency-free coordinate-window shortlist used by the Python
    // Bazel generator. It is deliberately approximate, unlike the final sweep.
    for (const auto& [label, group] : groups) {
      std::vector<int> pool = group;
      if (label == -1) {
        pool.resize(roots.size());
        std::iota(pool.begin(), pool.end(), 0);
      }
      const std::set<int> selected(group.begin(), group.end());
      for (int dimension = 0;
           dimension < std::min(4, r.model["width"].get<int>()); ++dimension) {
        auto coordinate = [&](int i) {
          return std::bit_cast<float>(
              r.states[roots[i]]["bits"][dimension].get<uint32_t>() << 16);
        };
        std::sort(pool.begin(), pool.end(), [&](int a, int b) {
          return std::pair(coordinate(a), roots[a]) <
                 std::pair(coordinate(b), roots[b]);
        });
        for (int rank = 0; rank < static_cast<int>(pool.size()); ++rank) {
          if (!selected.contains(pool[rank]))
            continue;
          const int begin = std::max(0, rank - neighbors),
                    end = std::min<int64_t>(pool.size(),
                                            int64_t{rank} + 1 + neighbors);
          for (int j = begin; j < end; ++j)
            if (pool[rank] != pool[j])
              pairs.insert(std::minmax(pool[rank], pool[j]));
        }
      }
    }
  }
  std::vector<Candidate> result;
  for (auto [first, second] : pairs)
    result.emplace_back(Distance(r.states[roots[first]]["bits"],
                                 r.states[roots[second]]["bits"]),
                        r.states[roots[first]]["id"].get<int>(),
                        r.states[roots[second]]["id"].get<int>());
  std::sort(result.begin(), result.end());
  return result;
}
int64_t EligiblePairCount(const QuotientReducer::Impl& r) {
  int64_t total = 0;
  for (int stage = 0; stage <= 2 * r.model["layers"].get<int>(); ++stage) {
    std::map<int, int64_t> counts;
    for (int root : r.Roots(stage))
      ++counts[r.TerminalLabel(root)];
    const int64_t unknown = counts[-1];
    counts.erase(-1);
    total += unknown * (unknown - 1) / 2;
    for (const auto& [label, count] : counts)
      total += unknown * count + count * (count - 1) / 2;
  }
  return total;
}
}  // namespace

absl::StatusOr<Json> ReduceModel(const Json& model,
                                 const ReductionOptions& options) {
  RETURN_IF_ERROR(ValidateModel(model));
  if (options.neighbors < 1 || options.max_passes < 1 ||
      options.exhaustive_pair_limit < 0 ||
      (options.max_attempts && *options.max_attempts < 0))
    return absl::InvalidArgumentError("invalid reduction search limits");
  ASSIGN_OR_RETURN(auto reducer, QuotientReducer::Create(model));
  const auto& r = reducer->impl();
  CooperativeInterrupt interrupt;
  using Clock = std::chrono::steady_clock;
  const auto start = Clock::now();
  auto last_report = start;
  int64_t last_attempt = 0;
  Json history = Json::array();
  auto elapsed = [](auto from) {
    return std::chrono::duration<double>(Clock::now() - from).count();
  };
  auto check_interrupt = [&]() -> absl::Status {
    if (!interrupt.Requested() &&
        (!options.interrupted || !options.interrupted()))
      return absl::OkStatus();
    return absl::CancelledError(
        "reduction interrupted at a safe trial boundary");
  };
  auto report = [&](const std::string& phase,
                    int pass) -> absl::StatusOr<Json> {
    RETURN_IF_ERROR(check_interrupt());
    auto roots = r.Roots();
    std::vector<int> counts(2 * model["layers"].get<int>() + 1);
    for (int root : roots)
      ++counts[r.stage[root]];
    Json info = {{"phase", phase},           {"pass", pass},
                 {"states", roots.size()},   {"states_per_stage", counts},
                 {"attempted", r.attempted}, {"accepted", r.accepted},
                 {"unions", r.unions},       {"seconds", elapsed(start)}};
    if (options.progress)
      options.progress(info);
    last_attempt = r.attempted;
    last_report = Clock::now();
    return info;
  };
  auto report_due = [&] {
    return r.attempted >= last_attempt + 1000 || elapsed(last_report) >= 10;
  };
  auto run_pass = [&](bool exhaustive, int pass) -> absl::StatusOr<bool> {
    const std::string phase = exhaustive ? "exhaustive" : "nearest";
    bool budget = false;
    for (int stage = 0; stage <= 2 * model["layers"].get<int>(); ++stage) {
      for (const auto& [distance, first, second] :
           CandidatePairs(r, stage, options.neighbors, exhaustive)) {
        RETURN_IF_ERROR(check_interrupt());
        if (options.max_attempts && r.attempted >= *options.max_attempts) {
          budget = true;
          break;
        }
        if (reducer->RootForState(first) != reducer->RootForState(second))
          RETURN_IF_ERROR(reducer->TryMerge(first, second).status());
        RETURN_IF_ERROR(check_interrupt());
        if (report_due())
          RETURN_IF_ERROR(report(phase, pass).status());
      }
      RETURN_IF_ERROR(report(phase, pass).status());
      if (budget)
        break;
    }
    ASSIGN_OR_RETURN(auto info, report(phase + "_pass_complete", pass));
    history.push_back(std::move(info));
    return budget;
  };
  std::string stop = "pass_limit";
  bool pairwise_irreducible = false;
  for (int pass = 1; pass <= options.max_passes; ++pass) {
    const int64_t previous = r.unions;
    RETURN_IF_ERROR(report("nearest", pass).status());
    ASSIGN_OR_RETURN(bool budget, run_pass(false, pass));
    if (budget) {
      stop = "attempt_limit";
      break;
    }
    if (r.unions == previous) {
      stop = "nearest_candidates_exhausted";
      break;
    }
  }
  const int64_t remaining = EligiblePairCount(r);
  if (remaining <= options.exhaustive_pair_limit && stop != "attempt_limit") {
    for (int sweep = 1;; ++sweep) {
      const int64_t previous = r.unions;
      ASSIGN_OR_RETURN(bool budget, run_pass(true, sweep));
      if (budget) {
        stop = "attempt_limit";
        break;
      }
      if (r.unions == previous) {
        pairwise_irreducible = true;
        stop = "no_compatible_pair";
        break;
      }
    }
  }
  Json result = reducer->Export();
  ASSIGN_OR_RETURN(result["stats"]["verification"], EvaluateModel(result));
  result["stats"]["search"] = {
      {"stopping_reason", stop},
      {"pairwise_irreducible", pairwise_irreducible},
      {"global_minimum_proven", false},
      {"nearest_neighbors", options.neighbors},
      {"exhaustive_pair_limit", options.exhaustive_pair_limit},
      {"remaining_pairs_before_sweep", remaining},
      {"history", history},
      {"seconds", elapsed(start)}};
  RETURN_IF_ERROR(check_interrupt());
  return result;
}
}  // namespace pluto::llm::discretized::generator
