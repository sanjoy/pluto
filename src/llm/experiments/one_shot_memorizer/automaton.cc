#include "src/llm/experiments/one_shot_memorizer/automaton.h"

#include <algorithm>
#include <limits>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_cat.h"
#include "src/util/status_macros.h"

namespace pluto::llm::one_shot_memorizer {
namespace {

struct TrieState {
  uint64_t terminal_count = 0;
  absl::flat_hash_map<int, size_t> children;
};

// Equality compares the complete signature, not just a hash: hash collisions
// never combine unequal right languages. Child IDs already identify exact
// weighted right languages because the construction proceeds bottom-up.
using Signature = std::pair<uint64_t, std::vector<std::pair<int, size_t>>>;

absl::Status ValidateVocabulary(int vocabulary_size, int eos_token_id) {
  if (vocabulary_size <= 0)
    return absl::InvalidArgumentError("vocabulary_size must be positive");
  if (eos_token_id < 0 || eos_token_id >= vocabulary_size)
    return absl::InvalidArgumentError("EOS token ID is outside the vocabulary");
  return absl::OkStatus();
}

absl::Status ValidateToken(int token, int vocabulary_size, int eos_token_id) {
  if (token < 0 || token >= vocabulary_size)
    return absl::InvalidArgumentError(
        absl::StrCat("Token ID outside the vocabulary: ", token));
  if (token == eos_token_id)
    return absl::InvalidArgumentError(
        "EOS must be represented by the sentence boundary, not an input token");
  return absl::OkStatus();
}

absl::Status AddCount(uint64_t count, uint64_t* total) {
  if (count > std::numeric_limits<uint64_t>::max() - *total)
    return absl::OutOfRangeError("Sentence count overflows uint64_t");
  *total += count;
  return absl::OkStatus();
}

absl::StatusOr<size_t> StateForPrefixUnchecked(const Model& model,
                                               absl::Span<const int> prefix) {
  // Validate all tokens even if an earlier valid token has no matching edge.
  for (int token : prefix) {
    RETURN_IF_ERROR(
        ValidateToken(token, model.vocabulary_size, model.eos_token_id));
  }
  size_t state_id = model.initial_state;
  for (int token : prefix) {
    const auto& transitions = model.states[state_id].transitions;
    const auto edge =
        std::lower_bound(transitions.begin(), transitions.end(), token,
                         [](const Transition& transition, int value) {
                           return transition.token < value;
                         });
    if (edge == transitions.end() || edge->token != token)
      return absl::NotFoundError("Prefix does not occur in the corpus");
    state_id = edge->target;
  }
  return state_id;
}

std::vector<TokenProbability> ProbabilitiesAtState(const Model& model,
                                                   size_t state_id) {
  const State& state = model.states[state_id];
  std::vector<TokenProbability> result;
  result.reserve(state.transitions.size() + (state.terminal_count != 0));
  for (const Transition& transition : state.transitions) {
    const uint64_t count = model.states[transition.target].suffix_count;
    result.push_back({transition.token, count,
                      static_cast<double>(count) / state.suffix_count});
  }
  if (state.terminal_count != 0) {
    const auto position =
        std::lower_bound(result.begin(), result.end(), model.eos_token_id,
                         [](const TokenProbability& item, int token) {
                           return item.token < token;
                         });
    result.insert(position, {model.eos_token_id, state.terminal_count,
                             static_cast<double>(state.terminal_count) /
                                 state.suffix_count});
  }
  return result;
}

}  // namespace

absl::StatusOr<Model> BuildModel(const std::vector<std::vector<int>>& sentences,
                                 int vocabulary_size, int eos_token_id) {
  RETURN_IF_ERROR(ValidateVocabulary(vocabulary_size, eos_token_id));
  if (sentences.empty())
    return absl::InvalidArgumentError("Corpus must contain a sentence");

  std::vector<TrieState> trie(1);
  for (const auto& sentence : sentences) {
    size_t current = 0;
    for (int token : sentence) {
      RETURN_IF_ERROR(ValidateToken(token, vocabulary_size, eos_token_id));
      auto child = trie[current].children.find(token);
      if (child == trie[current].children.end()) {
        const size_t next = trie.size();
        trie[current].children.emplace(token, next);
        trie.emplace_back();
        current = next;
      } else {
        current = child->second;
      }
    }
    RETURN_IF_ERROR(AddCount(1, &trie[current].terminal_count));
  }

  // Traverse sorted children to make postorder (and therefore interned state
  // IDs) independent of input order or hash-table iteration. An explicit work
  // list handles deep sentences without consuming the C++ call stack.
  std::vector<size_t> order;
  std::vector<size_t> pending = {0};
  order.reserve(trie.size());
  while (!pending.empty()) {
    const size_t current = pending.back();
    pending.pop_back();
    order.push_back(current);
    std::vector<std::pair<int, size_t>> children(trie[current].children.begin(),
                                                 trie[current].children.end());
    std::sort(children.begin(), children.end());
    for (auto child = children.rbegin(); child != children.rend(); ++child)
      pending.push_back(child->second);
  }

  Model model;
  model.vocabulary_size = vocabulary_size;
  model.eos_token_id = eos_token_id;
  model.trie_state_count = trie.size();
  absl::flat_hash_map<Signature, size_t> interned;
  std::vector<size_t> minimized(trie.size());
  for (auto it = order.rbegin(); it != order.rend(); ++it) {
    const TrieState& node = trie[*it];
    Signature signature;
    signature.first = node.terminal_count;
    signature.second.reserve(node.children.size());
    for (const auto& [token, target] : node.children)
      signature.second.emplace_back(token, minimized[target]);
    std::sort(signature.second.begin(), signature.second.end());
    const auto existing = interned.find(signature);
    if (existing != interned.end()) {
      minimized[*it] = existing->second;
      continue;
    }
    State state;
    state.terminal_count = node.terminal_count;
    state.suffix_count = node.terminal_count;
    state.transitions.reserve(signature.second.size());
    for (const auto& [token, target] : signature.second) {
      state.transitions.push_back({token, target});
      RETURN_IF_ERROR(
          AddCount(model.states[target].suffix_count, &state.suffix_count));
    }
    const size_t id = model.states.size();
    interned.emplace(std::move(signature), id);
    model.states.push_back(std::move(state));
    minimized[*it] = id;
  }
  model.initial_state = minimized[0];
  model.sentence_count = model.states[model.initial_state].suffix_count;
  return model;
}

absl::Status ValidateModel(const Model& model) {
  RETURN_IF_ERROR(
      ValidateVocabulary(model.vocabulary_size, model.eos_token_id));
  if (model.states.empty() || model.initial_state >= model.states.size())
    return absl::InvalidArgumentError("Model must have a valid initial state");
  if (model.trie_state_count < model.states.size())
    return absl::InvalidArgumentError("Trie state count is too small");
  if (model.sentence_count == 0 ||
      model.sentence_count != model.states[model.initial_state].suffix_count)
    return absl::InvalidArgumentError("Sentence count disagrees with root");

  for (size_t source = 0; source < model.states.size(); ++source) {
    const State& state = model.states[source];
    uint64_t count = state.terminal_count;
    int previous_token = -1;
    for (const Transition& edge : state.transitions) {
      RETURN_IF_ERROR(
          ValidateToken(edge.token, model.vocabulary_size, model.eos_token_id));
      if (edge.token <= previous_token)
        return absl::InvalidArgumentError(
            "Transitions must have unique tokens in sorted order");
      if (edge.target >= source)
        return absl::InvalidArgumentError(
            "Transition targets must precede their source state");
      previous_token = edge.token;
      RETURN_IF_ERROR(AddCount(model.states[edge.target].suffix_count, &count));
    }
    if (count == 0 || count != state.suffix_count)
      return absl::InvalidArgumentError("State suffix count is inconsistent");
  }

  // Descending IDs visit all predecessors before a state, so reachability
  // needs no recursive traversal or duplicate-entry queue.
  std::vector<bool> reached(model.states.size(), false);
  reached[model.initial_state] = true;
  for (size_t id = model.states.size(); id-- > 0;) {
    if (!reached[id])
      return absl::InvalidArgumentError("Model contains an unreachable state");
    for (const Transition& edge : model.states[id].transitions)
      reached[edge.target] = true;
  }
  return absl::OkStatus();
}

absl::StatusOr<size_t> StateForPrefix(const Model& model,
                                      absl::Span<const int> prefix) {
  RETURN_IF_ERROR(ValidateModel(model));
  return StateForPrefixUnchecked(model, prefix);
}

absl::StatusOr<std::vector<TokenProbability>> NextTokenProbabilities(
    const Model& model, absl::Span<const int> prefix) {
  RETURN_IF_ERROR(ValidateModel(model));
  ASSIGN_OR_RETURN(size_t state, StateForPrefixUnchecked(model, prefix));
  return ProbabilitiesAtState(model, state);
}

absl::StatusOr<std::vector<int>> GreedyContinuation(
    const Model& model, absl::Span<const int> prefix, size_t max_new_tokens) {
  RETURN_IF_ERROR(ValidateModel(model));
  if (max_new_tokens == 0)
    return absl::InvalidArgumentError("max_new_tokens must be positive");
  ASSIGN_OR_RETURN(size_t state_id, StateForPrefixUnchecked(model, prefix));
  std::vector<int> result;
  while (result.size() < max_new_tokens) {
    const State& state = model.states[state_id];
    int best_token = model.eos_token_id;
    uint64_t best_count = state.terminal_count;
    size_t next_state = state_id;
    for (const Transition& transition : state.transitions) {
      const uint64_t count = model.states[transition.target].suffix_count;
      if (count > best_count ||
          (count == best_count && transition.token < best_token)) {
        best_token = transition.token;
        best_count = count;
        next_state = transition.target;
      }
    }
    result.push_back(best_token);
    if (best_token == model.eos_token_id)
      break;
    state_id = next_state;
  }
  return result;
}

}  // namespace pluto::llm::one_shot_memorizer
