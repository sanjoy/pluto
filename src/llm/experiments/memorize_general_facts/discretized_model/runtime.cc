#include "src/llm/experiments/memorize_general_facts/discretized_model/runtime.h"

#include <limits>

#include "absl/strings/str_cat.h"
#include "src/util/status_macros.h"

namespace pluto::llm::discretized {
namespace {

absl::Status CheckPrompt(const DiscreteModel& model,
                         absl::Span<const DiscreteToken> tokens) {
  RETURN_IF_ERROR(ValidateModel(model));
  if (tokens.empty() || tokens.size() > model.context_length)
    return absl::InvalidArgumentError(
        "prompt must contain 1..context_length tokens");
  for (DiscreteToken token : tokens)
    if (token.value < 0 ||
        static_cast<size_t>(token.value) >= model.vocabulary.size())
      return absl::InvalidArgumentError(
          "prompt contains an invalid compact token ID");
  return absl::OkStatus();
}

absl::StatusOr<DiscreteHiddenState> LookupState(Map& map,
                                                DiscreteHiddenState key,
                                                absl::string_view boundary,
                                                size_t block) {
  const auto result = map(key);
  if (result.has_value())
    return *result;
  return absl::NotFoundError(absl::StrCat(
      "unsupported ", boundary, " state at block ", block, ": ", key.value));
}

}  // namespace

absl::Status ValidateModel(const DiscreteModel& model) {
  if (model.context_length == 0 ||
      model.context_length >
          static_cast<uint32_t>(std::numeric_limits<int32_t>::max()) ||
      model.prompt_token_count == 0 ||
      model.prompt_token_count > model.context_length ||
      model.vocabulary.empty() ||
      model.vocabulary.size() >
          static_cast<size_t>(std::numeric_limits<int>::max()) ||
      model.eos_token.value < 0 ||
      static_cast<size_t>(model.eos_token.value) >= model.vocabulary.size())
    return absl::InvalidArgumentError("invalid integer model dimensions");
  return absl::OkStatus();
}

absl::StatusOr<DiscreteToken> PredictNext(
    const DiscreteModel& model, absl::Span<const DiscreteToken> tokens) {
  RETURN_IF_ERROR(CheckPrompt(model, tokens));
  std::vector<DiscreteHiddenState> states;
  states.reserve(tokens.size());
  for (size_t position = 0; position < tokens.size(); ++position) {
    const auto result = model.position_embedding(
        tokens[position], static_cast<int32_t>(position));
    if (!result.has_value())
      return absl::NotFoundError(absl::StrCat(
          "unsupported token/position embedding: ", tokens[position].value, "/",
          position));
    if (result->value < 0 ||
        static_cast<size_t>(result->value) < model.vocabulary.size())
      return absl::DataLossError(
          "position embedding produced an invalid state");
    states.push_back(*result);
  }
  for (size_t block = 0; block < model.transformers.size(); ++block) {
    const auto& transformer = model.transformers[block];
    std::vector<DiscreteHiddenState> next(states.size());
    for (size_t position = 0; position < states.size(); ++position) {
      const auto key = absl::MakeConstSpan(states).first(position + 1);
      const auto attended = transformer.attention(key);
      if (!attended.has_value())
        return absl::NotFoundError(
            absl::StrCat("unsupported attention history at block ", block,
                         ", position ", position));
      if (attended->value < 0 ||
          static_cast<size_t>(attended->value) < model.vocabulary.size())
        return absl::DataLossError(
            "attention function produced an invalid state");
      auto output = LookupState(transformer.mlp, *attended, "MLP", block);
      if (!output.ok())
        return output.status();
      if (output->value < 0 ||
          static_cast<size_t>(output->value) < model.vocabulary.size())
        return absl::DataLossError("MLP function produced an invalid state");
      next[position] = *output;
    }
    states.swap(next);
  }
  auto token = LookupState(model.language_modeling_head, states.back(),
                           "language modeling head", model.transformers.size());
  if (!token.ok())
    return token.status();
  if (token->value < 0 ||
      static_cast<size_t>(token->value) >= model.vocabulary.size())
    return absl::DataLossError(
        "language modeling head produced an invalid compact token ID");
  return static_cast<DiscreteToken>(*token);
}

absl::StatusOr<std::vector<DiscreteToken>> Generate(
    const DiscreteModel& model, absl::Span<const DiscreteToken> prompt,
    size_t max_new_tokens) {
  auto status = CheckPrompt(model, prompt);
  if (!status.ok())
    return status;
  std::vector<DiscreteToken> prefix(prompt.begin(), prompt.end());
  std::vector<DiscreteToken> generated;
  // A full input window still predicts its next token, including EOS. If that
  // token is not EOS, appending it stops the loop before any overlong
  // evaluation.
  for (size_t i = 0;
       i < max_new_tokens && prefix.size() <= model.context_length; ++i) {
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

absl::StatusOr<std::string> Decode(const DiscreteModel& model,
                                   absl::Span<const DiscreteToken> tokens) {
  std::string result;
  for (DiscreteToken token : tokens) {
    if (token.value < 0 ||
        static_cast<size_t>(token.value) >= model.vocabulary.size())
      return absl::InvalidArgumentError(
          "cannot decode invalid compact token ID");
    result.append(model.vocabulary[token.value].bytes.data(),
                  model.vocabulary[token.value].bytes.size());
  }
  return result;
}

}  // namespace pluto::llm::discretized
