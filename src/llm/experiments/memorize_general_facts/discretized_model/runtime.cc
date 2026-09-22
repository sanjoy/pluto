#include "src/llm/experiments/memorize_general_facts/discretized_model/runtime.h"

#include <algorithm>
#include <limits>
#include <tuple>

#include "absl/strings/str_cat.h"

namespace pluto::llm::discretized {
namespace {

int Compare(absl::Span<const StateId> left, absl::Span<const StateId> right) {
  const size_t common = std::min(left.size(), right.size());
  for (size_t i = 0; i < common; ++i) {
    if (left[i] < right[i])
      return -1;
    if (left[i] > right[i])
      return 1;
  }
  return left.size() < right.size() ? -1 : left.size() > right.size();
}

absl::Status CheckPrompt(const Model& model, absl::Span<const TokenId> tokens) {
  if (tokens.empty() || tokens.size() > model.context_length)
    return absl::InvalidArgumentError(
        "prompt must contain 1..context_length tokens");
  if (model.attention.size() != model.mlp.size())
    return absl::InvalidArgumentError("attention and MLP block counts differ");
  for (TokenId token : tokens)
    if (token < 0 || static_cast<size_t>(token) >= model.vocabulary.size())
      return absl::InvalidArgumentError(
          "prompt contains an invalid compact token ID");
  return absl::OkStatus();
}

absl::StatusOr<StateId> LookupState(StateTable table, StateId key,
                                    absl::string_view boundary, size_t block) {
  if (table.function != nullptr) {
    const auto result = table.function(key);
    if (result.has_value())
      return *result;
    return absl::NotFoundError(absl::StrCat(
        "unsupported ", boundary, " state at block ", block, ": ", key));
  }
  auto row = std::lower_bound(
      table.rows.begin(), table.rows.end(), key,
      [](const StateRow& r, StateId k) { return r.input < k; });
  if (row == table.rows.end() || row->input != key)
    return absl::NotFoundError(absl::StrCat(
        "unsupported ", boundary, " state at block ", block, ": ", key));
  return row->output;
}

absl::Status CheckStateTable(StateTable table, bool snap, size_t vocabulary) {
  if (table.function != nullptr)
    return table.rows.empty()
               ? absl::OkStatus()
               : absl::InvalidArgumentError("ambiguous state table/function");
  bool first = true;
  StateId previous = 0;
  for (const StateRow& row : table.rows) {
    if ((!first && row.input <= previous) || row.input < vocabulary ||
        (snap ? row.output >= vocabulary : row.output < vocabulary))
      return absl::InvalidArgumentError("invalid or unsorted state table");
    first = false;
    previous = row.input;
  }
  return absl::OkStatus();
}

}  // namespace

absl::Status ValidateModel(const Model& model) {
  if (model.context_length == 0 || model.prompt_tokens == 0 ||
      model.prompt_tokens > model.context_length || model.vocabulary.empty() ||
      model.vocabulary.size() >
          static_cast<size_t>(std::numeric_limits<TokenId>::max()) ||
      model.eos_token < 0 ||
      static_cast<size_t>(model.eos_token) >= model.vocabulary.size() ||
      model.attention.size() != model.mlp.size() ||
      (model.entry.empty() && model.entry_function == nullptr) ||
      (model.snap.rows.empty() && model.snap.function == nullptr))
    return absl::InvalidArgumentError("invalid integer model dimensions");
  if (!model.entry.empty() && model.entry_function != nullptr)
    return absl::InvalidArgumentError("ambiguous entry table/function");
  const EntryRow* previous = nullptr;
  for (const EntryRow& row : model.entry) {
    if (row.token < 0 ||
        static_cast<size_t>(row.token) >= model.vocabulary.size() ||
        row.position >= model.context_length ||
        row.state < model.vocabulary.size() ||
        (previous != nullptr &&
         std::tie(row.token, row.position) <=
             std::tie(previous->token, previous->position)))
      return absl::InvalidArgumentError("invalid or unsorted entry table");
    previous = &row;
  }
  for (size_t block = 0; block < model.attention.size(); ++block) {
    const auto& table = model.attention[block];
    if (table.function != nullptr &&
        (!table.rows.empty() || !table.keys.empty()))
      return absl::InvalidArgumentError("ambiguous attention table/function");
    absl::Span<const StateId> last;
    for (const AttentionRow& row : table.rows) {
      if (row.length == 0 || row.length > model.context_length ||
          row.offset > table.keys.size() ||
          row.length > table.keys.size() - row.offset ||
          row.output < model.vocabulary.size())
        return absl::InvalidArgumentError("invalid attention key storage");
      const auto key = table.keys.subspan(row.offset, row.length);
      if ((!last.empty() && Compare(last, key) >= 0) ||
          std::any_of(key.begin(), key.end(),
                      [&](StateId id) { return id < model.vocabulary.size(); }))
        return absl::InvalidArgumentError("invalid or unsorted attention keys");
      last = key;
    }
    auto status =
        CheckStateTable(model.mlp[block], false, model.vocabulary.size());
    if (!status.ok())
      return status;
  }
  return CheckStateTable(model.snap, true, model.vocabulary.size());
}

absl::StatusOr<TokenId> PredictNext(const Model& model,
                                    absl::Span<const TokenId> tokens) {
  auto status = CheckPrompt(model, tokens);
  if (!status.ok())
    return status;
  std::vector<StateId> states;
  states.reserve(tokens.size());
  for (size_t position = 0; position < tokens.size(); ++position) {
    if (model.entry_function != nullptr) {
      const auto result = model.entry_function(tokens[position], position);
      if (!result.has_value())
        return absl::NotFoundError(
            absl::StrCat("unsupported token/position entry: ", tokens[position],
                         "/", position));
      if (*result < model.vocabulary.size())
        return absl::DataLossError("entry function produced an invalid state");
      states.push_back(*result);
      continue;
    }
    const auto key =
        std::make_pair(tokens[position], static_cast<uint32_t>(position));
    auto row =
        std::lower_bound(model.entry.begin(), model.entry.end(), key,
                         [](const EntryRow& r, const auto& k) {
                           return std::make_pair(r.token, r.position) < k;
                         });
    if (row == model.entry.end() || row->token != key.first ||
        row->position != key.second)
      return absl::NotFoundError(absl::StrCat(
          "unsupported token/position entry: ", key.first, "/", key.second));
    states.push_back(row->state);
  }
  for (size_t block = 0; block < model.attention.size(); ++block) {
    const auto& attention = model.attention[block];
    std::vector<StateId> next(states.size());
    for (size_t position = 0; position < states.size(); ++position) {
      const auto key = absl::MakeConstSpan(states).first(position + 1);
      TransitionResult attended;
      if (attention.function != nullptr) {
        attended = attention.function(key);
      } else {
        auto row = std::lower_bound(
            attention.rows.begin(), attention.rows.end(), key,
            [&](const AttentionRow& r, absl::Span<const StateId> k) {
              return Compare(attention.keys.subspan(r.offset, r.length), k) < 0;
            });
        if (row != attention.rows.end() &&
            Compare(attention.keys.subspan(row->offset, row->length), key) == 0)
          attended = row->output;
      }
      if (!attended.has_value())
        return absl::NotFoundError(
            absl::StrCat("unsupported attention history at block ", block,
                         ", position ", position));
      if (*attended < model.vocabulary.size())
        return absl::DataLossError(
            "attention function produced an invalid state");
      auto output = LookupState(model.mlp[block], *attended, "MLP", block);
      if (!output.ok())
        return output.status();
      if (*output < model.vocabulary.size())
        return absl::DataLossError("MLP function produced an invalid state");
      next[position] = *output;
    }
    states.swap(next);
  }
  auto token =
      LookupState(model.snap, states.back(), "snap", model.attention.size());
  if (!token.ok())
    return token.status();
  if (*token >= model.vocabulary.size())
    return absl::DataLossError("snap produced an invalid compact token ID");
  return static_cast<TokenId>(*token);
}

absl::StatusOr<std::vector<TokenId>> Generate(const Model& model,
                                              absl::Span<const TokenId> prompt,
                                              size_t max_new_tokens) {
  auto status = CheckPrompt(model, prompt);
  if (!status.ok())
    return status;
  std::vector<TokenId> prefix(prompt.begin(), prompt.end());
  std::vector<TokenId> generated;
  for (size_t i = 0; i < max_new_tokens && prefix.size() < model.context_length;
       ++i) {
    auto token = PredictNext(model, prefix);
    if (!token.ok())
      return token.status();
    generated.push_back(*token);
    if (*token == model.eos_token)
      break;
    prefix.push_back(*token);
  }
  return generated;
}

absl::StatusOr<std::string> Decode(const Model& model,
                                   absl::Span<const TokenId> tokens) {
  std::string result;
  for (TokenId token : tokens) {
    if (token < 0 || static_cast<size_t>(token) >= model.vocabulary.size())
      return absl::InvalidArgumentError(
          "cannot decode invalid compact token ID");
    result.append(model.vocabulary[token].bytes.data(),
                  model.vocabulary[token].bytes.size());
  }
  return result;
}

}  // namespace pluto::llm::discretized
