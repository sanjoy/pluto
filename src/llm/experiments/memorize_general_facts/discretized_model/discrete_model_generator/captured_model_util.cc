#include "src/llm/experiments/memorize_general_facts/discretized_model/discrete_model_generator/captured_model_util.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_cat.h"
#include "src/llm/experiments/memorize_general_facts/discretized_model/discrete_model_generator/generator_model_validation.h"
#include "src/util/status_macros.h"

namespace pluto::llm::discretized::generator {
namespace {
absl::Status Error(const std::string& message) {
  return absl::InvalidArgumentError(message);
}

absl::Status ValidateMetadata(const ModelMetadata& m, bool full) {
  if (m.layers < 0 || m.layers > 1024 || m.width < 1 || m.vocab_size < 1)
    return Error("invalid model dimensions");
  if (!full)
    return absl::OkStatus();
  if (m.prompt_tokens < 1 || m.prompt_tokens > 1024 || m.eos_token < 0 ||
      m.eos_token >= m.vocab_size ||
      m.vocabulary.size() != static_cast<size_t>(m.vocab_size))
    return Error("invalid model vocabulary or prompt metadata");
  absl::flat_hash_set<int> originals;
  for (const auto& token : m.vocabulary) {
    if (token.original_id < 0 || !originals.insert(token.original_id).second)
      return Error("invalid or duplicate original vocabulary ID");
    if (token.bytes.empty())
      return Error("vocabulary token bytes must not be empty");
  }
  return absl::OkStatus();
}

template <class Map, class Key>
absl::Status Put(Map& table, const Key& key, int value, const char* name) {
  auto [iterator, inserted] = table.emplace(key, value);
  if (!inserted && iterator->second != value)
    return Error(absl::StrCat("conflicting ", name, " key"));
  return absl::OkStatus();
}

// Ordered lookup indexes keep evaluation independent of compaction internals.
struct IntegerModel {
  std::map<std::pair<int, int>, int> entry;
  std::vector<std::map<std::vector<int>, int>> attention;
  std::vector<std::map<int, int>> mlp;
  std::map<int, int> language_modeling_head;

  explicit IntegerModel(const CapturedModel& model) {
    attention.resize(model.metadata.layers);
    mlp.resize(model.metadata.layers);
    for (const auto& row : model.position_embedding.transitions)
      entry.emplace(std::pair{row.token, row.position}, row.output);
    for (int layer = 0; layer < model.metadata.layers; ++layer) {
      for (const auto& row : model.transformers[layer].attention.transitions)
        attention[layer].emplace(row.prefix, row.output);
      for (const auto& row : model.transformers[layer].mlp.transitions)
        mlp[layer].emplace(row.input, row.output);
    }
    for (const auto& row : model.language_modeling_head.transitions)
      language_modeling_head.emplace(row.input, row.output);
  }

  absl::StatusOr<int> Predict(const std::vector<int>& tokens) const {
    if (tokens.empty())
      return Error("cannot predict from an empty prefix");
    if (tokens.size() > 1024)
      return Error("prediction prefix exceeds the 1024-token context");
    auto undefined = [] {
      return Error("undefined discrete transition for prefix");
    };
    std::vector<int> current;
    for (int position = 0; position < static_cast<int>(tokens.size());
         ++position) {
      auto row = entry.find({tokens[position], position});
      if (row == entry.end())
        return undefined();
      current.push_back(row->second);
    }
    for (size_t layer = 0; layer < attention.size(); ++layer) {
      std::vector<int> history, next;
      for (int state : current) {
        history.push_back(state);
        auto attention_row = attention[layer].find(history);
        if (attention_row == attention[layer].end())
          return undefined();
        auto mlp_row = mlp[layer].find(attention_row->second);
        if (mlp_row == mlp[layer].end())
          return undefined();
        next.push_back(mlp_row->second);
      }
      current = std::move(next);
    }
    auto row = language_modeling_head.find(current.back());
    if (row == language_modeling_head.end())
      return undefined();
    return row->second;
  }
};
}  // namespace

namespace internal {
absl::Status ValidateTables(const CapturedModel& model, bool full) {
  RETURN_IF_ERROR(ValidateMetadata(model.metadata, full));
  const int layers = model.metadata.layers, vocab = model.metadata.vocab_size;
  if (model.states.size() > static_cast<size_t>(INT32_MAX - vocab))
    return Error("too many discrete states");
  absl::flat_hash_map<int, int> states;
  absl::flat_hash_set<int> members;
  std::vector<int> stage_sizes(2 * layers + 1);
  for (const auto& row : model.states) {
    if (row.id < vocab || row.boundary < 0 || row.boundary > 2 * layers)
      return Error("invalid state ID or boundary");
    if (!states.emplace(row.id, row.boundary).second)
      return Error("duplicate state ID");
    ++stage_sizes[row.boundary];
    if (row.members) {
      if (row.members->empty())
        return Error("original-state membership must not be empty");
      for (int member : *row.members)
        if (member < vocab || !members.insert(member).second)
          return Error("invalid or duplicate original-state member");
    }
  }
  if (std::find(stage_sizes.begin(), stage_sizes.end(), 0) != stage_sizes.end())
    return Error("every boundary must contain at least one state");
  auto state_at = [&](int value, int stage) -> absl::Status {
    auto found = states.find(value);
    if (found == states.end() || found->second != stage)
      return Error(absl::StrCat("state ", value,
                                " does not belong to boundary ", stage));
    return absl::OkStatus();
  };
  if (full && !model.state_relabeling.empty()) {
    if (model.state_relabeling.size() != states.size())
      return Error("state relabeling must cover every state");
    absl::flat_hash_set<int> old_ids, new_ids;
    for (const auto& row : model.state_relabeling) {
      if (row.old_id < vocab || row.boundary < 0 || row.boundary > 2 * layers)
        return Error("invalid state relabeling ID or boundary");
      RETURN_IF_ERROR(state_at(row.new_id, row.boundary));
      if (!old_ids.insert(row.old_id).second ||
          !new_ids.insert(row.new_id).second)
        return Error("state relabeling must be a bijection within boundaries");
    }
  }
  std::set<std::pair<int, int>> entry_keys;
  for (const auto& row : model.position_embedding.transitions) {
    if (row.token < 0 || row.token >= vocab || row.position < 0 ||
        row.position >= 1024)
      return Error("invalid entry token or position");
    RETURN_IF_ERROR(state_at(row.output, 0));
    if (!entry_keys.emplace(row.token, row.position).second)
      return Error("duplicate entry key");
  }
  if (entry_keys.empty())
    return Error("entry table must not be empty");
  if (model.transformers.size() != static_cast<size_t>(layers))
    return Error("transformer count disagrees with metadata");
  for (int layer = 0; layer < layers; ++layer) {
    std::set<std::vector<int>> attention_keys;
    absl::flat_hash_set<int> mlp_keys;
    for (const auto& row : model.transformers[layer].attention.transitions) {
      if (row.prefix.empty() || row.prefix.size() > 1024)
        return Error("invalid attention history length");
      for (int state : row.prefix)
        RETURN_IF_ERROR(state_at(state, 2 * layer));
      RETURN_IF_ERROR(state_at(row.output, 2 * layer + 1));
      if (!attention_keys.insert(row.prefix).second)
        return Error("duplicate attention key");
    }
    for (const auto& row : model.transformers[layer].mlp.transitions) {
      RETURN_IF_ERROR(state_at(row.input, 2 * layer + 1));
      RETURN_IF_ERROR(state_at(row.output, 2 * layer + 2));
      if (!mlp_keys.insert(row.input).second)
        return Error("duplicate MLP key");
    }
  }
  absl::flat_hash_set<int> head_keys;
  for (const auto& row : model.language_modeling_head.transitions) {
    RETURN_IF_ERROR(state_at(row.input, 2 * layers));
    if (row.output < 0 || row.output >= vocab)
      return Error("invalid language modeling head token");
    if (!head_keys.insert(row.input).second)
      return Error("duplicate language modeling head key");
  }
  if (full) {
    if (head_keys.empty() || model.samples.empty())
      return Error("missing readout or verification samples");
    for (const auto& sample : model.samples) {
      if (sample.tokens.size() <
              static_cast<size_t>(model.metadata.prompt_tokens) ||
          sample.tokens.size() >= 1024)
        return Error("verification sample cannot fit prompt and EOS");
      for (size_t position = 0; position < sample.tokens.size(); ++position) {
        int token = sample.tokens[position];
        if (token < 0 || token >= vocab)
          return Error("invalid verification sample token");
        // Generation stops at its first EOS; an EOS inside the required suffix
        // would prevent verification of all remaining tokens and the final EOS.
        if (position >= static_cast<size_t>(model.metadata.prompt_tokens) &&
            token == model.metadata.eos_token)
          return Error("verification sample suffix must not contain EOS");
      }
    }
  }
  const auto& stats = model.stats;
  if (stats.state_compactions < 0 || stats.attempted_seeds < 0 ||
      stats.accepted_seeds < 0 || stats.cached_rejections < 0)
    return Error("compaction counters must not be negative");
  int64_t induced_total = 0;
  for (const auto& compaction : stats.accepted_compactions) {
    if (compaction.euclidean_distance &&
        (!std::isfinite(*compaction.euclidean_distance) ||
         *compaction.euclidean_distance < 0))
      return Error(
          "accepted compaction distance must be finite and nonnegative");
    if (compaction.induced_compactions < 0 ||
        compaction.induced_compactions > INT64_MAX - induced_total)
      return Error("induced compaction count is invalid or overflows");
    induced_total += compaction.induced_compactions;
  }
  return absl::OkStatus();
}
}  // namespace internal

absl::Status ValidateModel(const CapturedModel& model) {
  return internal::ValidateTables(model, true);
}

absl::StatusOr<CapturedModel> BuildModel(
    const ModelMetadata& metadata,
    const std::vector<ExecutionSample>& captured_samples, int expected_samples,
    StateVectorHints* vector_hints) {
  RETURN_IF_ERROR(ValidateMetadata(metadata, true));
  if (metadata.layers < 1)
    return Error("execution capture must contain at least one transformer");
  const int width = metadata.width, layers = metadata.layers,
            vocab = metadata.vocab_size, prompt = metadata.prompt_tokens,
            eos = metadata.eos_token;
  CapturedModel result;
  StateVectorHints hints;
  result.metadata = metadata;
  std::vector<std::map<std::vector<uint16_t>, int>> intern(2 * layers + 1);
  std::vector<std::map<std::vector<int>, int>> attention(layers);
  std::vector<std::map<int, int>> mlp(layers);
  std::map<std::pair<int, int>, int> entry;
  std::map<int, int> head;
  int64_t targets = 0;
  int sample_number = 0;
  for (const auto& sample : captured_samples) {
    ++sample_number;
    const auto& tokens = sample.tokens;
    if (tokens.size() < static_cast<size_t>(prompt) || tokens.size() >= 1024)
      return Error("invalid token sequence length");
    for (int token : tokens)
      if (token < 0 || token >= vocab)
        return Error("invalid captured token");
    if (std::find(tokens.begin() + prompt, tokens.end(), eos) != tokens.end())
      return Error("execution trace sample suffix must not contain EOS");
    if (sample.predictions.size() != tokens.size())
      return Error("prediction count must match tokens");
    for (int token : sample.predictions)
      if (token < 0 || token >= vocab)
        return Error("invalid captured prediction");
    for (int position = prompt - 1; position < static_cast<int>(tokens.size());
         ++position)
      if (sample.predictions[position] !=
          (position + 1 == static_cast<int>(tokens.size())
               ? eos
               : tokens[position + 1]))
        return Error(absl::StrCat(
            "reference suffix/EOS is incorrect at sample ", sample_number));
    if (sample.boundaries.size() != intern.size())
      return Error("captured residual boundary count disagrees with model");
    std::vector<std::vector<int>> encoded(intern.size());
    for (int stage = 0; stage < static_cast<int>(intern.size()); ++stage) {
      const auto& vectors = sample.boundaries[stage];
      if (vectors.size() != tokens.size())
        return Error("captured boundary row count disagrees with tokens");
      for (const auto& bits : vectors) {
        if (bits.size() != static_cast<size_t>(width))
          return Error("captured vector width disagrees with model");
        for (uint16_t word : bits)
          if ((word & 0x7f80) == 0x7f80)
            return Error("invalid or nonfinite BF16 boundary vector");
        if (result.states.size() >= static_cast<size_t>(INT32_MAX - vocab))
          return Error("too many discrete states");
        auto [iterator, inserted] =
            intern[stage].emplace(bits, vocab + result.states.size());
        if (inserted) {
          result.states.push_back({iterator->second, stage, std::nullopt});
          if (vector_hints != nullptr)
            hints.emplace(iterator->second, bits);
        }
        encoded[stage].push_back(iterator->second);
      }
    }
    for (int position = 0; position < static_cast<int>(tokens.size());
         ++position) {
      RETURN_IF_ERROR(Put(entry, std::pair{tokens[position], position},
                          encoded[0][position], "entry"));
      for (int layer = 0; layer < layers; ++layer) {
        std::vector<int> prefix(encoded[2 * layer].begin(),
                                encoded[2 * layer].begin() + position + 1);
        RETURN_IF_ERROR(Put(attention[layer], prefix,
                            encoded[2 * layer + 1][position], "attention"));
        RETURN_IF_ERROR(Put(mlp[layer], encoded[2 * layer + 1][position],
                            encoded[2 * layer + 2][position], "MLP"));
      }
      if (position >= prompt - 1)
        RETURN_IF_ERROR(Put(head, encoded.back()[position],
                            sample.predictions[position],
                            "language modeling head"));
    }
    targets += tokens.size() - prompt + 1;
    result.samples.push_back({tokens});
  }
  if (result.samples.empty() ||
      (expected_samples >= 0 &&
       result.samples.size() != static_cast<size_t>(expected_samples)))
    return Error(
        absl::StrCat("unexpected sample count: ", result.samples.size()));
  for (const auto& [key, output] : entry)
    result.position_embedding.transitions.push_back(
        {key.first, key.second, output});
  result.transformers.resize(layers);
  for (int layer = 0; layer < layers; ++layer) {
    for (const auto& [key, output] : attention[layer])
      result.transformers[layer].attention.transitions.push_back({key, output});
    for (const auto& [key, output] : mlp[layer])
      result.transformers[layer].mlp.transitions.push_back({key, output});
  }
  for (const auto& [key, output] : head)
    result.language_modeling_head.transitions.push_back({key, output});
  result.stats.captured_samples = result.samples.size();
  result.stats.scored_targets = targets;
  result.stats.exact_states = result.states.size();
  result.stats.states = result.states.size();
  result.stats.states_per_stage.resize(2 * layers + 1);
  for (const auto& state : result.states)
    ++result.stats.states_per_stage[state.boundary];
  ASSIGN_OR_RETURN(result.stats.verification, EvaluateModel(result));
  if (vector_hints != nullptr)
    *vector_hints = std::move(hints);
  return result;
}

absl::StatusOr<int> PredictNext(const CapturedModel& model,
                                const std::vector<int>& tokens) {
  RETURN_IF_ERROR(ValidateModel(model));
  return IntegerModel(model).Predict(tokens);
}

absl::StatusOr<VerificationResult> EvaluateModel(const CapturedModel& model) {
  RETURN_IF_ERROR(ValidateModel(model));
  IntegerModel runtime(model);
  const int prompt = model.metadata.prompt_tokens,
            eos = model.metadata.eos_token;
  int64_t targets = 0;
  int number = 0;
  for (const auto& sample : model.samples) {
    ++number;
    auto expected = sample.tokens;
    std::vector<int> prefix(expected.begin(), expected.begin() + prompt);
    expected.push_back(eos);
    for (int position = prompt; position < static_cast<int>(expected.size());
         ++position) {
      ASSIGN_OR_RETURN(int predicted, runtime.Predict(prefix));
      ++targets;
      if (predicted != expected[position])
        return Error(absl::StrCat("sample ", number, ", token ", prefix.size(),
                                  ": expected ", expected[position], ", got ",
                                  predicted));
      if (predicted != eos)
        prefix.push_back(predicted);
    }
  }
  return VerificationResult{
      .samples = static_cast<int64_t>(model.samples.size()),
      .targets = targets,
      .errors = 0,
      .explicit_eos = static_cast<int64_t>(model.samples.size())};
}

absl::StatusOr<CapturedModel> RestoreMembership(
    const CapturedModel& model, const CapturedModel& original_model) {
  RETURN_IF_ERROR(ValidateModel(model));
  RETURN_IF_ERROR(ValidateModel(original_model));
  if (model.metadata != original_model.metadata)
    return Error("membership source disagrees on model metadata");
  IntegerModel quotient(model);
  absl::flat_hash_map<int, int> mapping;
  auto missing = [] {
    return Error("model is not a complete quotient of the membership source");
  };
  for (const auto& row : original_model.position_embedding.transitions) {
    auto found = quotient.entry.find({row.token, row.position});
    if (found == quotient.entry.end())
      return missing();
    RETURN_IF_ERROR(
        Put(mapping, row.output, found->second, "state membership"));
  }
  for (int layer = 0; layer < model.metadata.layers; ++layer) {
    for (const auto& row :
         original_model.transformers[layer].attention.transitions) {
      std::vector<int> transformed;
      for (int input : row.prefix) {
        auto found = mapping.find(input);
        if (found == mapping.end())
          return missing();
        transformed.push_back(found->second);
      }
      auto found = quotient.attention[layer].find(transformed);
      if (found == quotient.attention[layer].end())
        return missing();
      RETURN_IF_ERROR(
          Put(mapping, row.output, found->second, "state membership"));
    }
    for (const auto& row : original_model.transformers[layer].mlp.transitions) {
      auto source = mapping.find(row.input);
      if (source == mapping.end())
        return missing();
      auto found = quotient.mlp[layer].find(source->second);
      if (found == quotient.mlp[layer].end())
        return missing();
      RETURN_IF_ERROR(
          Put(mapping, row.output, found->second, "state membership"));
    }
  }
  for (const auto& row : original_model.language_modeling_head.transitions) {
    auto source = mapping.find(row.input);
    if (source == mapping.end())
      return missing();
    auto found = quotient.language_modeling_head.find(source->second);
    if (found == quotient.language_modeling_head.end())
      return missing();
    if (found->second != row.output)
      return Error("membership source disagrees on a required token label");
  }
  std::map<int, std::vector<int>> members;
  absl::flat_hash_map<int, int> stages;
  for (const auto& row : model.states) {
    members[row.id] = {};
    stages[row.id] = row.boundary;
  }
  for (const auto& row : original_model.states) {
    auto found = mapping.find(row.id);
    if (found == mapping.end() || !stages.contains(found->second) ||
        stages[found->second] != row.boundary)
      return Error("membership state is missing or crosses a boundary");
    auto& group = members[found->second];
    if (row.members)
      group.insert(group.end(), row.members->begin(), row.members->end());
    else
      group.push_back(row.id);
  }
  CapturedModel result = model;
  int64_t count = 0;
  for (auto& row : result.states) {
    auto& group = members[row.id];
    if (group.empty())
      return Error("quotient contains a state without an original member");
    std::sort(group.begin(), group.end());
    count += group.size();
    row.members = group;
  }
  result.stats.membership_complete = true;
  result.stats.membership_original_states = count;
  return result;
}
}  // namespace pluto::llm::discretized::generator
