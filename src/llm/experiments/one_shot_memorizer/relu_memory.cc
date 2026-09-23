#include "src/llm/experiments/one_shot_memorizer/relu_memory.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <utility>

#include "absl/strings/str_cat.h"
#include "src/util/status_macros.h"

namespace pluto::llm::one_shot_memorizer {
namespace {

absl::Status ValidateMetadata(const ReluMemory& model) {
  if (model.vocabulary_size <= 0 || model.vocabulary_size > 65536)
    return absl::InvalidArgumentError("Vocabulary size must be in [1, 65536]");
  if (model.eos_token_id < 0 || model.eos_token_id >= model.vocabulary_size)
    return absl::InvalidArgumentError("EOS token ID is outside the vocabulary");
  if (model.prompt_tokens == 0)
    return absl::InvalidArgumentError("prompt_tokens must be positive");
  if (model.context_window == 0 ||
      model.context_window > std::numeric_limits<size_t>::max() / 2)
    return absl::InvalidArgumentError(
        "Context window must be positive and fit");
  return absl::OkStatus();
}

absl::Status ValidateToken(int token, const ReluMemory& model) {
  if (token < 0 || token >= model.vocabulary_size)
    return absl::InvalidArgumentError(
        absl::StrCat("Token ID outside the vocabulary: ", token));
  if (token == model.eos_token_id)
    return absl::InvalidArgumentError("Input tokens must exclude EOS");
  return absl::OkStatus();
}

absl::Status ValidatePrefix(const ReluMemory& model,
                            absl::Span<const int> prefix) {
  if (prefix.size() < model.prompt_tokens)
    return absl::InvalidArgumentError("Prefix is shorter than prompt_tokens");
  for (int token : prefix)
    RETURN_IF_ERROR(ValidateToken(token, model));
  return absl::OkStatus();
}

std::vector<int> MakeKey(absl::Span<const int> prefix, size_t window) {
  std::vector<int> key(window, -1);
  const size_t retained = std::min(window, prefix.size());
  std::copy(prefix.end() - retained, prefix.end(), key.end() - retained);
  return key;
}

ReluMemoryCode TokenCode(int token) {
  ReluMemoryCode code;
  for (size_t bit = 0; bit < code.size(); ++bit)
    code[bit] = (static_cast<unsigned>(token) & (1u << bit)) ? 1.0f : -1.0f;
  return code;
}

absl::StatusOr<int> DecodeUnchecked(const ReluMemory& model,
                                    const ReluMemoryCode& output) {
  bool exact_code = true;
  unsigned code_token = 0;
  for (size_t bit = 0; bit < output.size(); ++bit) {
    if (!std::isfinite(output[bit]))
      return absl::InvalidArgumentError("Output code must be finite");
    if (output[bit] == 1.0f)
      code_token |= 1u << bit;
    else if (output[bit] != -1.0f)
      exact_code = false;
  }
  if (exact_code) {
    if (code_token >= static_cast<unsigned>(model.vocabulary_size))
      return absl::InvalidArgumentError(
          "Output code is outside the vocabulary");
    // dot(c_a,c_b) = 16-2*Hamming(a,b), so exact codes win with margin >=2.
    return static_cast<int>(code_token);
  }
  int best_token = 0;
  float best_score = -std::numeric_limits<float>::infinity();
  for (int token = 0; token < model.vocabulary_size; ++token) {
    float score = 0.0f;
    for (size_t bit = 0; bit < output.size(); ++bit)
      score += output[bit] *
               ((static_cast<unsigned>(token) & (1u << bit)) ? 1.0f : -1.0f);
    if (!std::isfinite(score))
      return absl::OutOfRangeError("Float32 decoding score overflowed");
    if (score > best_score) {
      best_score = score;
      best_token = token;
    }
  }
  return best_token;
}

absl::StatusOr<ReluMemoryCode> EvaluateUnchecked(const ReluMemory& model,
                                                 absl::Span<const int> prefix) {
  const auto key = MakeKey(prefix, model.context_window);
  ReluMemoryCode output = {};
  bool active = false;
  // Evaluate actual ReLU arithmetic for every unit. No map or target labels
  // from construction are retained at inference.
  for (const ReluMemoryUnit& unit : model.units) {
    float distance = 0.0f;
    for (size_t j = 0; j < key.size(); ++j) {
      const float query = static_cast<float>(key[j]);
      const float positive = std::max(0.0f, query + unit.input_biases[2 * j]);
      const float negative =
          std::max(0.0f, -query + unit.input_biases[2 * j + 1]);
      distance += positive + negative;
    }
    const float hidden = std::max(0.0f, 1.0f - distance);
    // Zero output contributions may be skipped; the nonlinear activation
    // itself is always computed for every stored key.
    if (hidden > 0.0f) {
      active = true;
      for (size_t bit = 0; bit < output.size(); ++bit)
        output[bit] += hidden * unit.output_code[bit];
    }
  }
  if (!active)
    return absl::NotFoundError("No ReLU memory unit activates for this suffix");
  return output;
}

absl::StatusOr<uint64_t> MultiplyCount(uint64_t a, uint64_t b) {
  if (a != 0 && b > std::numeric_limits<uint64_t>::max() / a)
    return absl::OutOfRangeError("ReLU memory storage count overflowed");
  return a * b;
}

}  // namespace

absl::StatusOr<ReluMemory> BuildReluMemory(
    const std::vector<std::vector<int>>& sentences, int vocabulary_size,
    int eos_token_id, size_t prompt_tokens, size_t context_window) {
  ReluMemory model;
  model.vocabulary_size = vocabulary_size;
  model.eos_token_id = eos_token_id;
  model.prompt_tokens = prompt_tokens;
  model.context_window = context_window;
  RETURN_IF_ERROR(ValidateMetadata(model));
  if (sentences.empty())
    return absl::InvalidArgumentError("Corpus must contain a sentence");
  std::map<std::vector<int>, int> targets;
  for (const auto& sentence : sentences) {
    RETURN_IF_ERROR(ValidatePrefix(model, sentence));
    for (size_t position = prompt_tokens; position <= sentence.size();
         ++position) {
      auto key = MakeKey(absl::Span<const int>(sentence).first(position),
                         context_window);
      const int target =
          position == sentence.size() ? eos_token_id : sentence[position];
      const auto [existing, inserted] = targets.emplace(std::move(key), target);
      if (!inserted && existing->second != target)
        return absl::InvalidArgumentError(absl::StrCat(
            "Conflicting next-token labels for context window ", context_window,
            ": ", existing->second, " and ", target));
    }
  }
  model.units.reserve(targets.size());
  for (const auto& [key, target] : targets) {
    ReluMemoryUnit unit;
    unit.input_biases.reserve(2 * context_window);
    for (int token : key) {
      unit.input_biases.push_back(-static_cast<float>(token));
      unit.input_biases.push_back(static_cast<float>(token));
    }
    unit.output_code = TokenCode(target);
    model.units.push_back(std::move(unit));
  }
  return model;
}

absl::Status ValidateReluMemory(const ReluMemory& model) {
  RETURN_IF_ERROR(ValidateMetadata(model));
  if (model.units.empty())
    return absl::InvalidArgumentError("ReLU memory must contain a unit");
  const ReluMemoryUnit* previous = nullptr;
  for (const ReluMemoryUnit& unit : model.units) {
    if (unit.input_biases.size() != 2 * model.context_window)
      return absl::InvalidArgumentError("Unit has the wrong bias dimension");
    bool token_seen = false;
    size_t tokens = 0;
    int order = 0;
    for (size_t j = 0; j < model.context_window; ++j) {
      const float negative = unit.input_biases[2 * j];
      const float key = unit.input_biases[2 * j + 1];
      if (!std::isfinite(key) || !std::isfinite(negative) || negative != -key ||
          std::trunc(key) != key || key < -1.0f ||
          key >= static_cast<float>(model.vocabulary_size) ||
          key == static_cast<float>(model.eos_token_id))
        return absl::InvalidArgumentError(
            "Unit has invalid integer key biases");
      if (key == -1.0f) {
        if (token_seen)
          return absl::InvalidArgumentError("Padding must precede key tokens");
      } else {
        token_seen = true;
        ++tokens;
      }
      if (previous != nullptr && order == 0) {
        const float previous_key = previous->input_biases[2 * j + 1];
        if (previous_key < key)
          order = -1;
        if (previous_key > key)
          order = 1;
      }
    }
    if (tokens < std::min(model.prompt_tokens, model.context_window))
      return absl::InvalidArgumentError(
          "Unit key is shorter than prompt_tokens");
    if (previous != nullptr && order != -1)
      return absl::InvalidArgumentError("Unit keys must be unique and sorted");
    for (float value : unit.output_code)
      if (value != -1.0f && value != 1.0f)
        return absl::InvalidArgumentError("Unit output weights must be +/-1");
    auto token = DecodeUnchecked(model, unit.output_code);
    if (!token.ok())
      return token.status();
    previous = &unit;
  }
  return absl::OkStatus();
}

absl::StatusOr<ReluMemoryCode> EvaluateReluMemory(
    const ReluMemory& model, absl::Span<const int> prefix) {
  RETURN_IF_ERROR(ValidateReluMemory(model));
  RETURN_IF_ERROR(ValidatePrefix(model, prefix));
  return EvaluateUnchecked(model, prefix);
}

absl::StatusOr<int> DecodeReluMemoryOutput(const ReluMemory& model,
                                           const ReluMemoryCode& output) {
  RETURN_IF_ERROR(ValidateMetadata(model));
  return DecodeUnchecked(model, output);
}

absl::StatusOr<int> ReluMemoryNextToken(const ReluMemory& model,
                                        absl::Span<const int> prefix) {
  auto output = EvaluateReluMemory(model, prefix);
  if (!output.ok())
    return output.status();
  return DecodeUnchecked(model, *output);
}

absl::StatusOr<std::vector<int>> ReluMemoryGreedyContinuation(
    const ReluMemory& model, absl::Span<const int> prefix,
    size_t max_new_tokens) {
  RETURN_IF_ERROR(ValidateReluMemory(model));
  RETURN_IF_ERROR(ValidatePrefix(model, prefix));
  if (max_new_tokens == 0)
    return absl::InvalidArgumentError("max_new_tokens must be positive");
  std::vector<int> context(prefix.begin(), prefix.end());
  std::vector<int> generated;
  for (size_t step = 0; step < max_new_tokens; ++step) {
    auto output = EvaluateUnchecked(model, context);
    if (!output.ok())
      return output.status();
    auto token = DecodeUnchecked(model, *output);
    if (!token.ok())
      return token.status();
    generated.push_back(*token);
    if (*token == model.eos_token_id)
      return generated;
    context.push_back(*token);
  }
  return generated;
}

absl::StatusOr<ReluMemoryStorageCounts> GetReluMemoryStorageCounts(
    const ReluMemory& model) {
  RETURN_IF_ERROR(ValidateReluMemory(model));
  ReluMemoryStorageCounts counts;
  counts.second_hidden_units = model.units.size();
  auto first = MultiplyCount(model.units.size(), 2 * model.context_window);
  if (!first.ok())
    return first.status();
  counts.first_hidden_units = *first;
  auto output = MultiplyCount(model.units.size(), kReluMemoryCodeWidth);
  if (!output.ok())
    return output.status();
  if (*output > std::numeric_limits<uint64_t>::max() - *first)
    return absl::OutOfRangeError("ReLU memory storage count overflowed");
  counts.data_dependent_float_count = *first + *output;
  auto bytes = MultiplyCount(counts.data_dependent_float_count, sizeof(float));
  if (!bytes.ok())
    return bytes.status();
  counts.data_dependent_bytes = *bytes;
  auto fixed_weights = MultiplyCount(*first, 2);
  if (!fixed_weights.ok())
    return fixed_weights.status();
  counts.fixed_sparse_weight_count = *fixed_weights;
  counts.fixed_bias_count = model.units.size();
  counts.fixed_decoder_weight_count =
      static_cast<uint64_t>(model.vocabulary_size) * kReluMemoryCodeWidth;
  return counts;
}

}  // namespace pluto::llm::one_shot_memorizer
