#include "src/llm/experiments/memorize_general_facts/discretized_model/discretize_core.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_cat.h"
#include "src/llm/experiments/memorize_general_facts/discretized_model/generator_model_validation.h"
#include "src/util/status_macros.h"

namespace pluto::llm::discretized::generator {
namespace {
absl::Status Error(const std::string& message) {
  return absl::InvalidArgumentError(message);
}

const Json& Field(const Json& object, const char* key) {
  static const Json missing;
  if (!object.is_object())
    return missing;
  auto iterator = object.find(key);
  return iterator == object.end() ? missing : *iterator;
}

absl::Status Integer(const Json& value, const std::string& name,
                     int64_t minimum = 0,
                     int64_t maximum = std::numeric_limits<int32_t>::max()) {
  if (!value.is_number_integer() ||
      (value.is_number_unsigned() &&
       value.get<uint64_t>() > uint64_t(maximum)) ||
      (!value.is_number_unsigned() &&
       (value.get<int64_t>() < minimum || value.get<int64_t>() > maximum)) ||
      (value.is_number_unsigned() && value.get<uint64_t>() < uint64_t(minimum)))
    return Error(absl::StrCat(name, " must be an integer in [", minimum, ", ",
                              maximum, "]"));
  return absl::OkStatus();
}

absl::Status Array(const Json& value, const std::string& name,
                   int64_t size = -1) {
  if (!value.is_array() ||
      (size >= 0 && value.size() != static_cast<size_t>(size)))
    return Error(
        absl::StrCat(name, " must be an array",
                     size < 0 ? "" : absl::StrCat(" of length ", size)));
  return absl::OkStatus();
}

absl::Status Vocabulary(const Json& vocabulary, int size) {
  RETURN_IF_ERROR(Array(vocabulary, "vocabulary", size));
  absl::flat_hash_set<int> originals;
  for (const auto& row : vocabulary) {
    RETURN_IF_ERROR(Integer(Field(row, "original_id"), "original_id"));
    if (!originals.insert(row["original_id"].get<int>()).second)
      return Error("duplicate original vocabulary ID");
    const auto& hex = Field(row, "hex");
    if (!hex.is_string())
      return Error("vocabulary bytes require nonempty canonical lowercase hex");
    const auto& bytes = hex.get_ref<const std::string&>();
    if (bytes.empty() || bytes.size() % 2 != 0 ||
        !std::all_of(bytes.begin(), bytes.end(), [](char ch) {
          return (ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f');
        }))
      return Error("vocabulary bytes require nonempty canonical lowercase hex");
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

// Ordered maps give the same stable table ordering as the original format.
struct IntegerModel {
  std::map<std::pair<int, int>, int> entry;
  std::vector<std::map<std::vector<int>, int>> attention;
  std::vector<std::map<int, int>> mlp;
  std::map<int, int> language_modeling_head;

  explicit IntegerModel(const Json& model) {
    const int layers = model["layers"];
    attention.resize(layers);
    mlp.resize(layers);
    for (const auto& row : model["entry"])
      entry.emplace(std::pair<int, int>{row[0], row[1]}, row[2]);
    for (int layer = 0; layer < layers; ++layer) {
      for (const auto& row : model["attention"][layer])
        attention[layer].emplace(row[0].get<std::vector<int>>(), row[1]);
      for (const auto& row : model["mlp"][layer])
        mlp[layer].emplace(row[0], row[1]);
    }
    for (const auto& row : model["language_modeling_head"])
      language_modeling_head.emplace(row[0], row[1]);
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
      std::vector<int> history;
      std::vector<int> next;
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
absl::Status ValidateTables(const Json& model, bool full) {
  if (!model.is_object())
    return Error("expected integer model schema 1");
  RETURN_IF_ERROR(Integer(Field(model, "schema"), "schema", 1, 1));
  RETURN_IF_ERROR(Integer(Field(model, "layers"), "layers", 0, 1024));
  RETURN_IF_ERROR(Integer(Field(model, "width"), "width", 1));
  RETURN_IF_ERROR(Integer(Field(model, "vocab_size"), "vocab_size", 1));
  const int layers = model["layers"], width = model["width"],
            vocab = model["vocab_size"];
  if (full) {
    RETURN_IF_ERROR(
        Integer(Field(model, "prompt_tokens"), "prompt_tokens", 1, 1024));
    RETURN_IF_ERROR(
        Integer(Field(model, "eos_token"), "eos_token", 0, vocab - 1));
    RETURN_IF_ERROR(Vocabulary(Field(model, "vocabulary"), vocab));
  }
  RETURN_IF_ERROR(Array(Field(model, "states"), "states"));
  absl::flat_hash_map<int, int> states;
  absl::flat_hash_set<int> members;
  std::vector<int> stage_sizes(2 * layers + 1);
  for (const auto& row : model["states"]) {
    RETURN_IF_ERROR(Integer(Field(row, "id"), "state ID", vocab));
    RETURN_IF_ERROR(Integer(Field(row, "stage"), "state stage", 0, 2 * layers));
    RETURN_IF_ERROR(Array(Field(row, "bits"), "state bits", width));
    const int state = row["id"], stage = row["stage"];
    if (!states.emplace(state, stage).second)
      return Error("duplicate state ID");
    ++stage_sizes[stage];
    for (const auto& bits : row["bits"]) {
      RETURN_IF_ERROR(Integer(bits, "state bits", 0, 65535));
      if ((bits.get<int>() & 0x7f80) == 0x7f80)
        return Error("nonfinite BF16 boundary vector");
    }
    if (row.contains("members")) {
      RETURN_IF_ERROR(Array(row["members"], "original-state members"));
      RETURN_IF_ERROR(Integer(Field(row, "member_count"), "member_count", 1));
      if (row["members"].empty() ||
          row["member_count"] != row["members"].size())
        return Error("invalid original-state membership count");
      for (const auto& member : row["members"]) {
        RETURN_IF_ERROR(Integer(member, "original state ID", vocab));
        if (!members.insert(member.get<int>()).second)
          return Error("original state belongs to multiple classes");
      }
    }
  }
  if (std::find(stage_sizes.begin(), stage_sizes.end(), 0) != stage_sizes.end())
    return Error("every boundary must contain at least one state");
  auto state_at = [&](const Json& value, int stage) -> absl::Status {
    RETURN_IF_ERROR(Integer(value, "referenced state", vocab));
    auto found = states.find(value.get<int>());
    if (found == states.end() || found->second != stage)
      return Error(absl::StrCat("state ", value.get<int>(),
                                " does not belong to boundary ", stage));
    return absl::OkStatus();
  };
  if (full && model.contains("state_relabeling")) {
    RETURN_IF_ERROR(
        Array(model["state_relabeling"], "state_relabeling", states.size()));
    absl::flat_hash_set<int> old_ids, new_ids;
    for (const auto& row : model["state_relabeling"]) {
      RETURN_IF_ERROR(Array(row, "state relabeling row", 3));
      RETURN_IF_ERROR(Integer(row[0], "original relabeled state ID", vocab));
      RETURN_IF_ERROR(Integer(row[2], "relabeling boundary", 0, 2 * layers));
      RETURN_IF_ERROR(state_at(row[1], row[2].get<int>()));
      if (!old_ids.insert(row[0].get<int>()).second ||
          !new_ids.insert(row[1].get<int>()).second)
        return Error("state relabeling must be a bijection within boundaries");
    }
  }
  RETURN_IF_ERROR(Array(Field(model, "entry"), "entry"));
  std::set<std::pair<int, int>> entry_keys;
  for (const auto& row : model["entry"]) {
    RETURN_IF_ERROR(Array(row, "entry row", 3));
    RETURN_IF_ERROR(Integer(row[0], "entry token", 0, vocab - 1));
    RETURN_IF_ERROR(Integer(row[1], "entry position", 0, 1023));
    RETURN_IF_ERROR(state_at(row[2], 0));
    if (!entry_keys.emplace(row[0].get<int>(), row[1].get<int>()).second)
      return Error("duplicate entry key");
  }
  if (entry_keys.empty())
    return Error("entry table must not be empty");
  RETURN_IF_ERROR(Array(Field(model, "attention"), "attention", layers));
  RETURN_IF_ERROR(Array(Field(model, "mlp"), "MLP", layers));
  for (int layer = 0; layer < layers; ++layer) {
    RETURN_IF_ERROR(Array(model["attention"][layer], "attention table"));
    RETURN_IF_ERROR(Array(model["mlp"][layer], "MLP table"));
    std::set<std::vector<int>> attention_keys;
    absl::flat_hash_set<int> mlp_keys;
    for (const auto& row : model["attention"][layer]) {
      RETURN_IF_ERROR(Array(row, "attention row", 2));
      RETURN_IF_ERROR(Array(row[0], "attention prefix"));
      if (row[0].empty() || row[0].size() > 1024)
        return Error("invalid attention history length");
      for (const auto& state : row[0])
        RETURN_IF_ERROR(state_at(state, 2 * layer));
      RETURN_IF_ERROR(state_at(row[1], 2 * layer + 1));
      if (!attention_keys.insert(row[0].get<std::vector<int>>()).second)
        return Error("duplicate attention key");
    }
    for (const auto& row : model["mlp"][layer]) {
      RETURN_IF_ERROR(Array(row, "MLP row", 2));
      RETURN_IF_ERROR(state_at(row[0], 2 * layer + 1));
      RETURN_IF_ERROR(state_at(row[1], 2 * layer + 2));
      if (!mlp_keys.insert(row[0].get<int>()).second)
        return Error("duplicate MLP key");
    }
  }
  RETURN_IF_ERROR(
      Array(Field(model, "language_modeling_head"), "language modeling head"));
  absl::flat_hash_set<int> head_keys;
  for (const auto& row : model["language_modeling_head"]) {
    RETURN_IF_ERROR(Array(row, "language modeling head row", 2));
    RETURN_IF_ERROR(state_at(row[0], 2 * layers));
    RETURN_IF_ERROR(
        Integer(row[1], "language modeling head token", 0, vocab - 1));
    if (!head_keys.insert(row[0].get<int>()).second)
      return Error("duplicate language modeling head key");
  }
  if (full) {
    RETURN_IF_ERROR(Array(Field(model, "samples"), "samples"));
    if (head_keys.empty() || model["samples"].empty())
      return Error("missing readout or verification samples");
    for (const auto& sample : model["samples"]) {
      RETURN_IF_ERROR(Array(Field(sample, "tokens"), "sample tokens"));
      if (sample["tokens"].size() < model["prompt_tokens"].get<size_t>() ||
          sample["tokens"].size() >= 1024)
        return Error("verification sample cannot fit prompt and EOS");
      for (size_t position = 0; position < sample["tokens"].size();
           ++position) {
        const auto& token = sample["tokens"][position];
        RETURN_IF_ERROR(Integer(token, "sample token", 0, vocab - 1));
        // Greedy generation terminates on its first EOS. An EOS inside the
        // required suffix would make suffix-plus-final-EOS verification
        // impossible, even if repeatedly predicting EOS matched the table.
        if (position >= model["prompt_tokens"].get<size_t>() &&
            token == model["eos_token"])
          return Error("verification sample suffix must not contain EOS");
      }
    }
  }
  if (model.contains("stats") && !model["stats"].is_object())
    return Error("stats must be an object");
  if (model.contains("stats")) {
    const auto& stats = model["stats"];
    for (const char* key : {"state_unions", "attempted_seeds", "accepted_seeds",
                            "cached_rejections"})
      if (stats.contains(key))
        RETURN_IF_ERROR(Integer(stats[key], key, 0, INT64_MAX));
    if (stats.contains("accepted_merges")) {
      RETURN_IF_ERROR(Array(stats["accepted_merges"], "accepted_merges"));
      int64_t induced_total = 0;
      for (const auto& merge : stats["accepted_merges"]) {
        const auto& distance = Field(merge, "euclidean_distance");
        if (!distance.is_number() || !std::isfinite(distance.get<double>()) ||
            distance.get<double>() < 0)
          return Error(
              "accepted merge distance must be finite and nonnegative");
        RETURN_IF_ERROR(Integer(Field(merge, "induced_unions"),
                                "induced_unions", 0, INT64_MAX));
        const auto induced = merge["induced_unions"].get<int64_t>();
        if (induced > INT64_MAX - induced_total)
          return Error("accepted merge induced union count overflows int64");
        induced_total += induced;
      }
    }
    if (stats.contains("search")) {
      const auto& search = stats["search"];
      if (!search.is_object())
        return Error("search statistics must be an object");
      if (search.contains("pairwise_irreducible") &&
          !search["pairwise_irreducible"].is_boolean())
        return Error("pairwise_irreducible must be a boolean");
    }
  }
  return absl::OkStatus();
}
}  // namespace internal

absl::Status ValidateModel(const Json& model) {
  return internal::ValidateTables(model, true);
}

absl::StatusOr<Json> BuildModel(const Json& header,
                                const std::vector<Json>& captured_samples,
                                int expected_samples) {
  if (!header.is_object())
    return Error("execution trace must begin with metadata");
  RETURN_IF_ERROR(Integer(Field(header, "schema"), "schema", 1, 1));
  RETURN_IF_ERROR(Integer(Field(header, "width"), "width", 1));
  RETURN_IF_ERROR(Integer(Field(header, "layers"), "layers", 1, 1024));
  RETURN_IF_ERROR(Integer(Field(header, "vocab_size"), "vocab_size", 1));
  RETURN_IF_ERROR(
      Integer(Field(header, "prompt_tokens"), "prompt_tokens", 1, 1024));
  const int width = header["width"], layers = header["layers"],
            vocab = header["vocab_size"], prompt = header["prompt_tokens"];
  RETURN_IF_ERROR(
      Integer(Field(header, "eos_token"), "eos_token", 0, vocab - 1));
  const int eos = header["eos_token"];
  RETURN_IF_ERROR(Vocabulary(Field(header, "vocabulary"), vocab));
  Json states = Json::array(), samples = Json::array();
  std::vector<std::map<std::vector<int>, int>> intern(2 * layers + 1),
      attention(layers);
  std::vector<std::map<int, int>> mlp(layers);
  std::map<std::pair<int, int>, int> entry;
  std::map<int, int> head;
  int64_t targets = 0;
  int sample_number = 0;
  for (const auto& sample : captured_samples) {
    ++sample_number;
    if (!sample.is_object())
      return Error("execution trace sample must be an object");
    RETURN_IF_ERROR(Array(Field(sample, "tokens"), "tokens"));
    const auto& token_json = sample["tokens"];
    if (token_json.size() < static_cast<size_t>(prompt) ||
        token_json.size() >= 1024)
      return Error("invalid token sequence length");
    for (const auto& token : token_json)
      RETURN_IF_ERROR(Integer(token, "token", 0, vocab - 1));
    auto tokens = token_json.get<std::vector<int>>();
    if (std::find(tokens.begin() + prompt, tokens.end(), eos) != tokens.end())
      return Error("execution trace sample suffix must not contain EOS");
    RETURN_IF_ERROR(
        Array(Field(sample, "predictions"), "predictions", tokens.size()));
    for (const auto& token : sample["predictions"])
      RETURN_IF_ERROR(Integer(token, "prediction", 0, vocab - 1));
    auto predictions = sample["predictions"].get<std::vector<int>>();
    for (int position = prompt - 1; position < static_cast<int>(tokens.size());
         ++position)
      if (predictions[position] !=
          (position + 1 == static_cast<int>(tokens.size())
               ? eos
               : tokens[position + 1]))
        return Error(absl::StrCat(
            "reference suffix/EOS is incorrect at sample ", sample_number));
    RETURN_IF_ERROR(
        Array(Field(sample, "boundaries"), "boundaries", intern.size()));
    std::vector<std::vector<int>> encoded(intern.size());
    for (int stage = 0; stage < static_cast<int>(intern.size()); ++stage) {
      const auto& vectors = sample["boundaries"][stage];
      RETURN_IF_ERROR(Array(vectors, "boundary vectors", tokens.size()));
      for (const auto& words : vectors) {
        RETURN_IF_ERROR(Array(words, "boundary vector", width));
        for (const auto& word : words) {
          RETURN_IF_ERROR(Integer(word, "BF16 bits", 0, 65535));
          if ((word.get<int>() & 0x7f80) == 0x7f80)
            return Error("invalid or nonfinite BF16 boundary vector");
        }
        auto bits = words.get<std::vector<int>>();
        if (states.size() >= static_cast<size_t>(INT32_MAX - vocab))
          return Error("too many discrete states");
        auto [iterator, inserted] =
            intern[stage].emplace(bits, vocab + states.size());
        if (inserted)
          states.push_back(
              {{"id", iterator->second}, {"stage", stage}, {"bits", bits}});
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
                            predictions[position], "language modeling head"));
    }
    targets += tokens.size() - prompt + 1;
    samples.push_back({{"tokens", tokens}});
  }
  if (samples.empty() ||
      (expected_samples >= 0 &&
       samples.size() != static_cast<size_t>(expected_samples)))
    return Error(absl::StrCat("unexpected sample count: ", samples.size()));
  Json result;
  for (const char* key : {"schema", "width", "layers", "vocab_size",
                          "eos_token", "prompt_tokens", "vocabulary"})
    result[key] = header[key];
  result["states"] = std::move(states);
  result["samples"] = std::move(samples);
  result["entry"] = Json::array();
  for (const auto& [key, output] : entry)
    result["entry"].push_back({key.first, key.second, output});
  result["attention"] = Json::array();
  result["mlp"] = Json::array();
  for (int layer = 0; layer < layers; ++layer) {
    Json attention_rows = Json::array(), mlp_rows = Json::array();
    for (const auto& [key, output] : attention[layer])
      attention_rows.push_back({key, output});
    for (const auto& [key, output] : mlp[layer])
      mlp_rows.push_back({key, output});
    result["attention"].push_back(std::move(attention_rows));
    result["mlp"].push_back(std::move(mlp_rows));
  }
  result["language_modeling_head"] = Json::array();
  for (const auto& [key, output] : head)
    result["language_modeling_head"].push_back({key, output});
  result["stats"] = {{"captured_samples", result["samples"].size()},
                     {"scored_targets", targets},
                     {"exact_states", result["states"].size()}};
  ASSIGN_OR_RETURN(result["stats"]["verification"], EvaluateModel(result));
  return result;
}

absl::StatusOr<int> PredictNext(const Json& model,
                                const std::vector<int>& tokens) {
  RETURN_IF_ERROR(ValidateModel(model));
  return IntegerModel(model).Predict(tokens);
}

absl::StatusOr<Json> EvaluateModel(const Json& model) {
  RETURN_IF_ERROR(ValidateModel(model));
  IntegerModel runtime(model);
  const int prompt = model["prompt_tokens"], eos = model["eos_token"];
  int64_t targets = 0;
  int number = 0;
  for (const auto& sample : model["samples"]) {
    ++number;
    auto expected = sample["tokens"].get<std::vector<int>>();
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
  return Json{{"samples", model["samples"].size()},
              {"targets", targets},
              {"errors", 0},
              {"explicit_eos", model["samples"].size()}};
}

absl::StatusOr<Json> RestoreMembership(const Json& model,
                                       const Json& original_model) {
  RETURN_IF_ERROR(ValidateModel(model));
  RETURN_IF_ERROR(ValidateModel(original_model));
  for (const char* key : {"schema", "width", "layers", "vocab_size",
                          "eos_token", "prompt_tokens", "vocabulary"})
    if (model[key] != original_model[key])
      return Error(absl::StrCat("membership source disagrees on ", key));
  IntegerModel quotient(model);
  absl::flat_hash_map<int, int> mapping;
  auto missing = [] {
    return Error("model is not a complete quotient of the membership source");
  };
  for (const auto& row : original_model["entry"]) {
    auto found = quotient.entry.find({row[0], row[1]});
    if (found == quotient.entry.end())
      return missing();
    RETURN_IF_ERROR(
        Put(mapping, row[2].get<int>(), found->second, "state membership"));
  }
  for (int layer = 0; layer < model["layers"].get<int>(); ++layer) {
    for (const auto& row : original_model["attention"][layer]) {
      std::vector<int> transformed;
      for (const auto& input : row[0]) {
        auto found = mapping.find(input.get<int>());
        if (found == mapping.end())
          return missing();
        transformed.push_back(found->second);
      }
      auto found = quotient.attention[layer].find(transformed);
      if (found == quotient.attention[layer].end())
        return missing();
      RETURN_IF_ERROR(
          Put(mapping, row[1].get<int>(), found->second, "state membership"));
    }
    for (const auto& row : original_model["mlp"][layer]) {
      auto source = mapping.find(row[0].get<int>());
      if (source == mapping.end())
        return missing();
      auto found = quotient.mlp[layer].find(source->second);
      if (found == quotient.mlp[layer].end())
        return missing();
      RETURN_IF_ERROR(
          Put(mapping, row[1].get<int>(), found->second, "state membership"));
    }
  }
  for (const auto& row : original_model["language_modeling_head"]) {
    auto source = mapping.find(row[0].get<int>());
    if (source == mapping.end())
      return missing();
    auto found = quotient.language_modeling_head.find(source->second);
    if (found == quotient.language_modeling_head.end())
      return missing();
    if (found->second != row[1].get<int>())
      return Error("membership source disagrees on a required token label");
  }
  std::map<int, std::vector<int>> members;
  absl::flat_hash_map<int, int> stages;
  for (const auto& row : model["states"]) {
    members[row["id"].get<int>()] = {};
    stages[row["id"].get<int>()] = row["stage"];
  }
  for (const auto& row : original_model["states"]) {
    auto found = mapping.find(row["id"].get<int>());
    if (found == mapping.end() || !stages.contains(found->second) ||
        stages[found->second] != row["stage"].get<int>())
      return Error("membership state is missing or crosses a boundary");
    auto& group = members[found->second];
    if (row.contains("members"))
      for (const auto& member : row["members"])
        group.push_back(member.get<int>());
    else
      group.push_back(row["id"].get<int>());
  }
  Json result = model;
  int64_t count = 0;
  for (auto& row : result["states"]) {
    auto& group = members[row["id"].get<int>()];
    if (group.empty())
      return Error("quotient contains a state without an original member");
    std::sort(group.begin(), group.end());
    count += group.size();
    row["members"] = group;
    row["member_count"] = group.size();
  }
  result["stats"]["membership_complete"] = true;
  result["stats"]["membership_original_states"] = count;
  return result;
}
}  // namespace pluto::llm::discretized::generator
