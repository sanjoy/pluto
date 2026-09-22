#include "src/llm/experiments/memorize_general_facts/discretized_model/runtime.h"

#include <limits>

#include "absl/strings/str_cat.h"
#include "src/util/status_macros.h"

namespace pluto::llm::discretized {
namespace {

absl::Status CheckPrompt(const Model& model, absl::Span<const TokenId> tokens) {
  RETURN_IF_ERROR(ValidateModel(model));
  if (tokens.empty() || tokens.size() > model.context_length)
    return absl::InvalidArgumentError(
        "prompt must contain 1..context_length tokens");
  for (TokenId token : tokens)
    if (token < 0 || static_cast<size_t>(token) >= model.vocabulary.size())
      return absl::InvalidArgumentError(
          "prompt contains an invalid compact token ID");
  return absl::OkStatus();
}

absl::StatusOr<StateId> LookupState(StateTable table, StateId key,
                                    absl::string_view boundary, size_t block) {
  const auto result = table.function(key);
  if (result.output.has_value())
    return *result.output;
  return absl::NotFoundError(absl::StrCat(
      "unsupported ", boundary, " state at block ", block, ": ", key));
}

}  // namespace

absl::Status ValidateModel(const Model& model) {
  if (model.context_length == 0 || model.prompt_tokens == 0 ||
      model.prompt_tokens > model.context_length || model.vocabulary.empty() ||
      model.vocabulary.size() >
          static_cast<size_t>(std::numeric_limits<TokenId>::max()) ||
      model.eos_token < 0 ||
      static_cast<size_t>(model.eos_token) >= model.vocabulary.size() ||
      model.attention.size() != model.mlp.size())
    return absl::InvalidArgumentError("invalid integer model dimensions");
  if (model.entry_function == nullptr || model.snap.function == nullptr)
    return absl::InvalidArgumentError("missing entry or snap lookup function");
  for (size_t block = 0; block < model.attention.size(); ++block)
    if (model.attention[block].function == nullptr ||
        model.mlp[block].function == nullptr)
      return absl::InvalidArgumentError(absl::StrCat(
          "missing attention or MLP lookup function at block ", block));
  return absl::OkStatus();
}

absl::StatusOr<TokenId> PredictNext(const Model& model,
                                    absl::Span<const TokenId> tokens) {
  RETURN_IF_ERROR(CheckPrompt(model, tokens));
  std::vector<StateId> states;
  states.reserve(tokens.size());
  for (size_t position = 0; position < tokens.size(); ++position) {
    const auto result = model.entry_function(tokens[position], position);
    if (!result.output.has_value())
      return absl::NotFoundError(
          absl::StrCat("unsupported token/position entry: ", tokens[position],
                       "/", position));
    if (*result.output < model.vocabulary.size())
      return absl::DataLossError("entry function produced an invalid state");
    states.push_back(*result.output);
  }
  for (size_t block = 0; block < model.attention.size(); ++block) {
    const auto& attention = model.attention[block];
    std::vector<StateId> next(states.size());
    for (size_t position = 0; position < states.size(); ++position) {
      const auto key = absl::MakeConstSpan(states).first(position + 1);
      const auto attended = attention.function(key);
      if (!attended.output.has_value())
        return absl::NotFoundError(
            absl::StrCat("unsupported attention history at block ", block,
                         ", position ", position));
      if (*attended.output < model.vocabulary.size())
        return absl::DataLossError(
            "attention function produced an invalid state");
      auto output =
          LookupState(model.mlp[block], *attended.output, "MLP", block);
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
