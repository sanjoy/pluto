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
double Distance(const std::vector<uint16_t>& first,
                const std::vector<uint16_t>& second) {
  // BF16 values are search coordinates only. Runtime transitions remain exact
  // integers, and even distant states can be compacted in the exhaustive phase.
  long double sum = 0;
  for (size_t index = 0; index < first.size(); ++index) {
    const double a = std::bit_cast<float>(uint32_t(first[index]) << 16);
    const double b = std::bit_cast<float>(uint32_t(second[index]) << 16);
    sum += (a - b) * (a - b);
  }
  return static_cast<double>(sum);
}

volatile std::sig_atomic_t interrupted = 0;
void Interrupt(int) { interrupted = 1; }

// The signal handler only records intent. All rollback work happens
// synchronously after TryCompact has committed or completely restored its
// state.
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

struct StateCompactor::Impl {
  struct Term {
    int stage;
    std::vector<int> inputs;
    int output;
  };
  struct Undo {
    enum Kind { kDictionary, kTerm, kUses, kCompaction } kind;
    int first = 0, second = 0, size = 0, label = 0;
    std::vector<int> values;
  };

  explicit Impl(const SymbolicModel& source)
      : model(source), states(source.states) {
    const int count = states.size();
    parent.resize(count);
    std::iota(parent.begin(), parent.end(), 0);
    size.assign(count, 1);
    label.assign(count, -1);
    uses.resize(count);
    for (int i = 0; i < count; ++i) {
      index[states[i].id] = i;
      stage.push_back(states[i].boundary);
    }
    for (int layer = 0; layer < model.metadata.layers; ++layer) {
      for (const auto& row : model.transformers[layer].attention)
        AddTerm(2 * layer + 1, row.prefix, row.output);
      for (const auto& row : model.transformers[layer].mlp)
        AddTerm(2 * layer + 2, {row.input}, row.output);
    }
    for (const auto& row : model.language_modeling_head)
      label[index.at(row.input)] = row.output;
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
    // Attach smaller classes to larger ones to bound depth without path
    // compression, so every change can be rolled back.
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
    const int final = 2 * model.metadata.layers;
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
        case Undo::kCompaction:
          parent[operation.second] = operation.second;
          size[operation.first] = operation.size;
          label[operation.first] = operation.label;
          break;
      }
    }
  }

  bool Compact(int first, int second) {
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
    const std::array<int, 2> seed_ids = {states[first].id, states[second].id};
    const double distance =
        std::sqrt(Distance(states[first].bits, states[second].bits));
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
      undo.push_back({Undo::kCompaction, a, b, size[a], label[a], {}});
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
          // could otherwise expand thousands of doomed intermediate
          // compactions.
          pending.emplace_front(terms[term].output,
                                terms[other->second].output);
        }
      }
    }
    if (count) {
      ++accepted;
      compactions += count;
      accepted_compactions.push_back(
          {seed_stage, seed_ids, distance, count - 1});
    }
    return true;
  }

  SymbolicModel model;
  std::vector<SymbolicState> states;
  absl::flat_hash_map<int, int> index;
  std::vector<int> parent, size, label, stage;
  std::vector<std::set<int>> uses;
  std::vector<Term> terms;
  std::map<std::vector<int>, int> signatures;
  std::vector<std::vector<int>> term_signatures;
  std::set<std::pair<int, int>> rejected_pairs;
  int64_t attempted = 0, accepted = 0, compactions = 0, cached_rejections = 0;
  std::vector<CompactionRecord> accepted_compactions;
};

StateCompactor::StateCompactor(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
StateCompactor::~StateCompactor() = default;
absl::StatusOr<std::unique_ptr<StateCompactor>> StateCompactor::Create(
    const SymbolicModel& model) {
  RETURN_IF_ERROR(internal::ValidateTables(model, false));
  return absl::WrapUnique(new StateCompactor(absl::make_unique<Impl>(model)));
}
absl::StatusOr<bool> StateCompactor::TryCompact(int first_id, int second_id) {
  auto first = impl_->index.find(first_id),
       second = impl_->index.find(second_id);
  if (first == impl_->index.end() || second == impl_->index.end())
    return absl::InvalidArgumentError("unknown discrete state ID");
  if (impl_->stage[first->second] != impl_->stage[second->second])
    return absl::InvalidArgumentError(
        "compaction cannot identify states from different boundaries");
  return impl_->Compact(first->second, second->second);
}
int StateCompactor::RootForState(int state_id) const {
  auto found = impl_->index.find(state_id);
  return found == impl_->index.end() ? -1 : impl_->Find(found->second);
}

SymbolicModel StateCompactor::Export() const {
  const auto& r = *impl_;
  auto roots = r.Roots();
  std::sort(roots.begin(), roots.end(), [&](int a, int b) {
    return std::pair(r.stage[a], r.states[a].id) <
           std::pair(r.stage[b], r.states[b].id);
  });
  absl::flat_hash_map<int, int> renumber;
  for (int i = 0; i < static_cast<int>(roots.size()); ++i)
    renumber[roots[i]] = r.model.metadata.vocab_size + i;
  auto state_id = [&](int old) { return renumber.at(r.Find(r.index.at(old))); };
  SymbolicModel result;
  result.metadata = r.model.metadata;
  result.samples = r.model.samples;
  const ModelStatistics& previous = r.model.stats;
  bool complete = previous.state_compactions == 0;
  if (!complete)
    complete = std::all_of(
        r.states.begin(), r.states.end(),
        [](const SymbolicState& row) { return row.members.has_value(); });
  std::map<int, std::vector<int>> members;
  if (complete)
    for (int i = 0; i < static_cast<int>(r.states.size()); ++i) {
      auto& group = members[r.Find(i)];
      const auto& row = r.states[i];
      if (row.members)
        group.insert(group.end(), row.members->begin(), row.members->end());
      else
        group.push_back(row.id);
    }
  int64_t original_count = 0;
  for (int root : roots) {
    SymbolicState row{renumber.at(root), r.stage[root], r.states[root].bits,
                      std::nullopt};
    if (complete) {
      auto& group = members[root];
      std::sort(group.begin(), group.end());
      row.members = group;
      original_count += group.size();
    }
    result.states.push_back(std::move(row));
  }
  for (const auto& row : r.model.entry)
    result.entry.push_back({row.token, row.position, state_id(row.output)});
  const int layers = r.model.metadata.layers;
  result.transformers.resize(layers);
  for (int layer = 0; layer < layers; ++layer) {
    std::map<std::vector<int>, int> attention;
    std::map<int, int> mlp;
    for (const auto& row : r.model.transformers[layer].attention) {
      std::vector<int> key;
      for (int state : row.prefix)
        key.push_back(state_id(state));
      attention[key] = state_id(row.output);
    }
    for (const auto& row : r.model.transformers[layer].mlp)
      mlp[state_id(row.input)] = state_id(row.output);
    for (const auto& [key, output] : attention)
      result.transformers[layer].attention.push_back({key, output});
    for (const auto& [key, output] : mlp)
      result.transformers[layer].mlp.push_back({key, output});
  }
  std::map<int, int> head;
  for (const auto& row : r.model.language_modeling_head)
    head[state_id(row.input)] = row.output;
  for (const auto& [key, output] : head)
    result.language_modeling_head.push_back({key, output});
  result.stats = previous;
  auto& stats = result.stats;
  stats.states = roots.size();
  stats.membership_complete = complete;
  stats.membership_original_states =
      complete ? std::optional(original_count) : std::nullopt;
  std::vector<int> counts(2 * layers + 1);
  for (int root : roots)
    ++counts[r.stage[root]];
  stats.states_per_stage = counts;
  stats.attempted_seeds += r.attempted;
  stats.accepted_seeds += r.accepted;
  stats.state_compactions += r.compactions;
  stats.cached_rejections += r.cached_rejections;
  stats.accepted_compactions.insert(stats.accepted_compactions.end(),
                                    r.accepted_compactions.begin(),
                                    r.accepted_compactions.end());
  return result;
}

namespace {
using Candidate = std::tuple<double, int, int>;
std::vector<Candidate> CandidatePairs(const StateCompactor::Impl& r, int stage,
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
      for (int dimension = 0; dimension < std::min(4, r.model.metadata.width);
           ++dimension) {
        auto coordinate = [&](int i) {
          return std::bit_cast<float>(
              uint32_t(r.states[roots[i]].bits[dimension]) << 16);
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
    result.emplace_back(
        Distance(r.states[roots[first]].bits, r.states[roots[second]].bits),
        r.states[roots[first]].id, r.states[roots[second]].id);
  std::sort(result.begin(), result.end());
  return result;
}
int64_t EligiblePairCount(const StateCompactor::Impl& r) {
  int64_t total = 0;
  for (int stage = 0; stage <= 2 * r.model.metadata.layers; ++stage) {
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

absl::StatusOr<SymbolicModel> CompactModel(const SymbolicModel& model,
                                           const CompactionOptions& options) {
  RETURN_IF_ERROR(ValidateModel(model));
  if (options.neighbors < 1 || options.max_passes < 1 ||
      options.exhaustive_pair_limit < 0 ||
      (options.max_attempts && *options.max_attempts < 0))
    return absl::InvalidArgumentError("invalid compaction search limits");
  ASSIGN_OR_RETURN(auto compactor, StateCompactor::Create(model));
  const auto& r = compactor->impl();
  CooperativeInterrupt interrupt;
  using Clock = std::chrono::steady_clock;
  const auto start = Clock::now();
  auto last_report = start;
  int64_t last_attempt = 0;
  std::vector<CompactionProgress> history;
  auto elapsed = [](auto from) {
    return std::chrono::duration<double>(Clock::now() - from).count();
  };
  auto check_interrupt = [&]() -> absl::Status {
    if (!interrupt.Requested() &&
        (!options.interrupted || !options.interrupted()))
      return absl::OkStatus();
    return absl::CancelledError(
        "compaction interrupted at a safe trial boundary");
  };
  auto report = [&](CompactionPhase phase,
                    int pass) -> absl::StatusOr<CompactionProgress> {
    RETURN_IF_ERROR(check_interrupt());
    auto roots = r.Roots();
    std::vector<int> counts(2 * model.metadata.layers + 1);
    for (int root : roots)
      ++counts[r.stage[root]];
    CompactionProgress info{
        phase,         pass,          static_cast<int64_t>(roots.size()),
        counts,        r.attempted,   r.accepted,
        r.compactions, elapsed(start)};
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
    const CompactionPhase phase =
        exhaustive ? CompactionPhase::kExhaustive : CompactionPhase::kNearest;
    bool budget = false;
    for (int stage = 0; stage <= 2 * model.metadata.layers; ++stage) {
      for (const auto& [distance, first, second] :
           CandidatePairs(r, stage, options.neighbors, exhaustive)) {
        RETURN_IF_ERROR(check_interrupt());
        if (options.max_attempts && r.attempted >= *options.max_attempts) {
          budget = true;
          break;
        }
        if (compactor->RootForState(first) != compactor->RootForState(second))
          RETURN_IF_ERROR(compactor->TryCompact(first, second).status());
        RETURN_IF_ERROR(check_interrupt());
        if (report_due())
          RETURN_IF_ERROR(report(phase, pass).status());
      }
      RETURN_IF_ERROR(report(phase, pass).status());
      if (budget)
        break;
    }
    ASSIGN_OR_RETURN(
        auto info, report(exhaustive ? CompactionPhase::kExhaustivePassComplete
                                     : CompactionPhase::kNearestPassComplete,
                          pass));
    history.push_back(std::move(info));
    return budget;
  };
  CompactionStoppingReason stop = CompactionStoppingReason::kPassLimit;
  bool pairwise_compaction_complete = false;
  for (int pass = 1; pass <= options.max_passes; ++pass) {
    const int64_t previous = r.compactions;
    RETURN_IF_ERROR(report(CompactionPhase::kNearest, pass).status());
    ASSIGN_OR_RETURN(bool budget, run_pass(false, pass));
    if (budget) {
      stop = CompactionStoppingReason::kAttemptLimit;
      break;
    }
    if (r.compactions == previous) {
      stop = CompactionStoppingReason::kNearestCandidatesExhausted;
      break;
    }
  }
  const int64_t remaining = EligiblePairCount(r);
  if (remaining <= options.exhaustive_pair_limit &&
      stop != CompactionStoppingReason::kAttemptLimit) {
    for (int sweep = 1;; ++sweep) {
      const int64_t previous = r.compactions;
      ASSIGN_OR_RETURN(bool budget, run_pass(true, sweep));
      if (budget) {
        stop = CompactionStoppingReason::kAttemptLimit;
        break;
      }
      if (r.compactions == previous) {
        pairwise_compaction_complete = true;
        stop = CompactionStoppingReason::kNoCompatiblePair;
        break;
      }
    }
  }
  SymbolicModel result = compactor->Export();
  ASSIGN_OR_RETURN(result.stats.verification, EvaluateModel(result));
  result.stats.compaction_search =
      CompactionSearchStatistics{stop,
                                 pairwise_compaction_complete,
                                 false,
                                 options.neighbors,
                                 options.exhaustive_pair_limit,
                                 remaining,
                                 history,
                                 elapsed(start)};
  RETURN_IF_ERROR(check_interrupt());
  return result;
}
}  // namespace pluto::llm::discretized::generator
