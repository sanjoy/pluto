#include "src/llm/experiments/memorize_general_facts/discretized_model/discrete_model_generator/discretize_transition_tests.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "src/llm/experiments/memorize_general_facts/discretized_model/discrete_model_generator/captured_model_util.h"
#include "src/util/status_macros.h"

namespace pluto::llm::discretized::generator {
namespace {

constexpr size_t kProbeLimit = 64;
constexpr int kMax = std::numeric_limits<int32_t>::max();
constexpr int kMin = std::numeric_limits<int32_t>::min();

struct ReplaySample {
  size_t token_offset;
  size_t state_offset;
  size_t length;
};

// Every expected activation comes from the source dictionaries, not from the
// production renderer or any compact transition evaluator. This intentionally
// repeats the simple model semantics to catch compensating compiler errors.
struct Replay {
  std::vector<int> tokens;
  std::vector<int> states;
  std::vector<ReplaySample> samples;
};

absl::StatusOr<Replay> ReplaySamples(const CapturedModel& model) {
  std::map<std::pair<int, int>, int> entry;
  for (const auto& row : model.position_embedding.transitions)
    entry[{row.token, row.position}] = row.output;
  const int layers = model.metadata.layers;
  std::vector<std::map<std::vector<int>, int>> attention(layers);
  std::vector<std::map<int, int>> mlp(layers);
  for (int block = 0; block < layers; ++block) {
    for (const auto& row : model.transformers[block].attention.transitions)
      attention[block][row.prefix] = row.output;
    for (const auto& row : model.transformers[block].mlp.transitions)
      mlp[block][row.input] = row.output;
  }
  std::map<int, int> head;
  for (const auto& row : model.language_modeling_head.transitions)
    head[row.input] = row.output;
  Replay replay;
  size_t number = 0;
  for (const auto& sample : model.samples) {
    const auto& original = sample.tokens;
    replay.samples.push_back(
        {replay.tokens.size(), replay.states.size(), original.size()});
    replay.tokens.insert(replay.tokens.end(), original.begin(), original.end());
    std::vector<int> boundary;
    for (size_t position = 0; position < original.size(); ++position) {
      auto row = entry.find({original[position], position});
      if (row == entry.end())
        return absl::InvalidArgumentError(
            absl::StrCat("sample ", number, ": unsupported source entry"));
      boundary.push_back(row->second);
    }
    replay.states.insert(replay.states.end(), boundary.begin(), boundary.end());
    for (int block = 0; block < layers; ++block) {
      std::vector<int> prefix, next;
      for (int state : boundary) {
        prefix.push_back(state);
        auto row = attention[block].find(prefix);
        if (row == attention[block].end())
          return absl::InvalidArgumentError(absl::StrCat(
              "sample ", number, ": unsupported source attention"));
        next.push_back(row->second);
      }
      boundary = std::move(next);
      replay.states.insert(replay.states.end(), boundary.begin(),
                           boundary.end());
      for (int& state : boundary) {
        auto row = mlp[block].find(state);
        if (row == mlp[block].end())
          return absl::InvalidArgumentError(
              absl::StrCat("sample ", number, ": unsupported source MLP"));
        state = row->second;
      }
      replay.states.insert(replay.states.end(), boundary.begin(),
                           boundary.end());
    }
    for (size_t position =
             static_cast<size_t>(model.metadata.prompt_tokens) - 1;
         position < original.size(); ++position) {
      const int target = position + 1 < original.size()
                             ? original[position + 1]
                             : model.metadata.eos_token;
      auto row = head.find(boundary[position]);
      if (row == head.end() || row->second != target)
        return absl::InvalidArgumentError(
            absl::StrCat("sample ", number, ", position ", position,
                         ": source readout disagrees with sample target"));
    }
    ++number;
  }
  return replay;
}

template <class T>
std::vector<T> EvenlySpaced(const std::vector<T>& values) {
  if (values.size() <= kProbeLimit)
    return values;
  std::vector<T> result;
  for (size_t i = 0; i < kProbeLimit; ++i)
    result.push_back(values[i * (values.size() - 1) / (kProbeLimit - 1)]);
  return result;
}

struct AttentionProbe {
  std::vector<int> prefix;
  std::optional<int> output;
};

std::vector<AttentionProbe> AttentionProbes(
    const std::vector<AttentionTransition>& rows) {
  std::map<std::vector<int>, int> table;
  for (const auto& row : rows)
    table[row.prefix] = row.output;
  std::vector<std::vector<int>> keys;
  for (const auto& [prefix, ignored] : table)
    keys.push_back(prefix);
  std::vector<AttentionProbe> probes;
  std::set<std::vector<int>> seen;
  auto add = [&](const std::vector<int>& prefix) {
    if (probes.size() >= kProbeLimit || !seen.insert(prefix).second)
      return;
    auto it = table.find(prefix);
    probes.push_back({prefix, it == table.end()
                                  ? std::nullopt
                                  : std::optional<int>(it->second)});
  };
  add({});
  add({0});
  add({-1});
  add({kMax});
  if (!keys.empty()) {
    auto longest = *std::max_element(
        keys.begin(), keys.end(),
        [](const auto& a, const auto& b) { return a.size() < b.size(); });
    longest.push_back(longest.back());
    add(longest);
    longest.pop_back();
    longest.pop_back();
    add(longest);
    std::set<int> domain;
    for (const auto& key : keys)
      domain.insert(key.begin(), key.end());
    std::map<int, int> successor;
    for (auto it = domain.begin(); it != domain.end(); ++it) {
      auto next = std::next(it);
      successor[*it] = next == domain.end() ? *domain.begin() : *next;
    }
    for (const auto& prefix : EvenlySpaced(keys)) {
      add(std::vector<int>(prefix.begin(), prefix.end() - 1));
      auto extended = prefix;
      extended.push_back(prefix.back());
      add(extended);
      add(std::vector<int>(prefix.begin() + 1, prefix.end()));
      add(std::vector<int>(prefix.rbegin(), prefix.rend()));
      for (size_t position :
           std::set<size_t>{0, prefix.size() / 2, prefix.size() - 1})
        for (int state : {successor[prefix[position]], 0, -1, kMax}) {
          auto mutated = prefix;
          mutated[position] = state;
          add(mutated);
        }
      if (probes.size() >= kProbeLimit)
        break;
    }
  }
  return probes;
}

struct StateProbe {
  int state;
  std::optional<int> output;
};

std::vector<StateProbe> PointwiseProbes(
    const std::vector<StateTransition>& rows) {
  std::map<int, int> table;
  for (const auto& row : rows)
    table[row.input] = row.output;
  std::set<int> seen;
  std::vector<StateProbe> probes;
  auto add = [&](int64_t state) {
    if (state < kMin || state > kMax || probes.size() >= kProbeLimit ||
        !seen.insert(state).second)
      return;
    auto it = table.find(state);
    probes.push_back(
        {static_cast<int>(state),
         it == table.end() ? std::nullopt : std::optional<int>(it->second)});
  };
  add(0);
  add(-1);
  add(kMax);
  if (!table.empty()) {
    add(int64_t(table.begin()->first) - 1);
    add(int64_t(table.rbegin()->first) + 1);
    std::vector<int> keys;
    for (const auto& [state, ignored] : table)
      keys.push_back(state);
    for (int64_t state : EvenlySpaced(keys)) {
      add(state);
      add(state - 1);
      add(state + 1);
    }
  }
  return probes;
}

struct EntryProbe {
  int token;
  int position;
  std::optional<int> output;
};

std::vector<EntryProbe> EntryProbes(const CapturedModel& model) {
  std::map<std::pair<int, int>, int> table;
  for (const auto& row : model.position_embedding.transitions)
    table[{row.token, row.position}] = row.output;
  std::vector<EntryProbe> probes;
  std::set<std::pair<int, int>> seen;
  auto add = [&](int64_t token, int64_t position) {
    if (token < kMin || token > kMax || position < kMin || position > kMax ||
        probes.size() >= kProbeLimit)
      return;
    const std::pair<int, int> key(token, position);
    if (!seen.insert(key).second)
      return;
    auto it = table.find(key);
    probes.push_back(
        {key.first, key.second,
         it == table.end() ? std::nullopt : std::optional<int>(it->second)});
  };
  for (int token : {-1, model.metadata.vocab_size, kMax})
    add(token, 0);
  std::vector<std::pair<int, int>> keys;
  for (const auto& [key, ignored] : table)
    keys.push_back(key);
  for (const auto& [token, position] : EvenlySpaced(keys)) {
    for (int64_t candidate :
         {int64_t(position), int64_t(position) - 1, int64_t(position) + 1,
          int64_t(1024), int64_t(kMin), int64_t(kMax)})
      add(token, candidate);
    add(int64_t(token) - 1, position);
    add(int64_t(token) + 1, position);
  }
  return probes;
}

std::string Array(absl::string_view type, absl::string_view name,
                  std::vector<std::string> values, size_t per_line = 1) {
  if (values.empty())
    values.push_back("{}");
  std::string result = absl::StrCat("constexpr ", type, " ", name, "[] = {\n");
  for (size_t i = 0; i < values.size(); ++i) {
    if (i % per_line == 0)
      result += "  ";
    absl::StrAppend(&result, values[i], ",");
    result +=
        i % per_line == per_line - 1 || i + 1 == values.size() ? "\n" : " ";
  }
  return result + "};\n";
}

std::string Result(std::optional<int> output,
                   const std::vector<std::string>* names = nullptr) {
  if (!output)
    return "{std::nullopt}";
  if (names)
    return absl::StrCat("{static_cast<DiscreteHiddenState>(", (*names)[*output],
                        ")}");
  return absl::StrCat("{DiscreteHiddenState{", *output, "}}");
}

}  // namespace

absl::StatusOr<std::string> RenderTransitionTest(
    const CapturedModel& model, const std::vector<std::string>& token_names) {
  RETURN_IF_ERROR(ValidateModel(model));
  if (token_names.size() != static_cast<size_t>(model.metadata.vocab_size))
    return absl::InvalidArgumentError(
        "token_names must cover the compact vocabulary");
  ASSIGN_OR_RETURN(auto replay, ReplaySamples(model));
  const int layers = model.metadata.layers;
  std::vector<std::string> attention_keys, attention_rows, pointwise_rows;
  for (int block = 0; block < layers; ++block) {
    for (const auto& probe :
         AttentionProbes(model.transformers[block].attention.transitions)) {
      attention_rows.push_back(
          absl::StrCat("{", block, ", ", attention_keys.size(), ", ",
                       probe.prefix.size(), ", ", Result(probe.output), "}"));
      for (int state : probe.prefix)
        attention_keys.push_back(absl::StrCat("{", state, "}"));
    }
    for (const auto& probe :
         PointwiseProbes(model.transformers[block].mlp.transitions))
      pointwise_rows.push_back(absl::StrCat("{", block, ", {", probe.state,
                                            "}, ", Result(probe.output), "}"));
  }
  std::string body = R"cpp(// Generated independent boundary fixtures;
                           // test-only, never inference input.
                           // Expectations replay source transitions
                           // independently of compiled control flow. Histories
                           // are reconstructed from boundary vectors at real
                           // positions only.
#include <cstddef>
#include "model.h"
#include "vocabulary_tokens.h"
#include "gtest/gtest.h"

                           namespace pluto::llm::discretized::gen {
                           namespace {
                           namespace vocab = internal::vocab;
  )cpp";
  absl::StrAppend(&body, "constexpr size_t kLayers = ", layers,
                  ";\nconstexpr size_t kSampleCount = ", replay.samples.size(),
                  ";\n");
  body += R"cpp(struct Sample {
                  size_t token_offset;
                  size_t state_offset;
                  size_t length;
                };
                struct LanguageModelingHeadRow {
                  DiscreteHiddenState input;
                  DiscreteHiddenState output;
                };
                struct AttentionProbe {
                  size_t block;
                  size_t offset;
                  size_t length;
                  std::optional<DiscreteHiddenState> expected;
                };
                struct PointwiseProbe {
                  size_t block;
                  DiscreteHiddenState input;
                  std::optional<DiscreteHiddenState> expected;
                };
                struct StateProbe {
                  DiscreteHiddenState input;
                  std::optional<DiscreteHiddenState> expected;
                };
                struct EntryProbe {
                  DiscreteToken token;
                  int32_t position;
                  std::optional<DiscreteHiddenState> expected;
                };
  )cpp";
  std::vector<std::string> values;
  for (int token : replay.tokens)
    values.push_back(token_names[token]);
  body += Array("DiscreteToken", "kSampleTokens", values, 4);
  values.clear();
  for (int state : replay.states)
    values.push_back(absl::StrCat("{", state, "}"));
  body += Array("DiscreteHiddenState", "kExpectedStates", values, 16);
  values.clear();
  for (const auto& sample : replay.samples)
    values.push_back(absl::StrCat("{", sample.token_offset, ", ",
                                  sample.state_offset, ", ", sample.length,
                                  "}"));
  body += Array("Sample", "kSamples", values);
  std::map<int, int> head;
  for (const auto& row : model.language_modeling_head.transitions)
    head[row.input] = row.output;
  values.clear();
  for (const auto& [state, token] : head)
    values.push_back(absl::StrCat("{{", state,
                                  "}, static_cast<DiscreteHiddenState>(",
                                  token_names[token], ")}"));
  body += Array("LanguageModelingHeadRow", "kLanguageModelingHeadRows", values);
  body +=
      Array("DiscreteHiddenState", "kAttentionProbeKeys", attention_keys, 16);
  body += Array("AttentionProbe", "kAttentionProbes", attention_rows);
  body += Array("PointwiseProbe", "kMlpProbes", pointwise_rows);
  values.clear();
  const auto head_probes =
      PointwiseProbes(model.language_modeling_head.transitions);
  for (const auto& probe : head_probes)
    values.push_back(absl::StrCat("{{", probe.state, "}, ",
                                  Result(probe.output, &token_names), "}"));
  body += Array("StateProbe", "kLanguageModelingHeadProbes", values);
  values.clear();
  const auto entry_probes = EntryProbes(model);
  for (const auto& probe : entry_probes) {
    const auto token =
        probe.token >= 0 && size_t(probe.token) < token_names.size()
            ? token_names[probe.token]
            : absl::StrCat("DiscreteToken{", probe.token, "}");
    values.push_back(absl::StrCat("{", token, ", ", probe.position, ", ",
                                  Result(probe.output), "}"));
  }
  body += Array("EntryProbe", "kEntryProbes", values);
  body += R"cpp(
    void ExpectTransition(std::optional<DiscreteHiddenState> actual,
                          std::optional<DiscreteHiddenState> expected) {
      EXPECT_EQ(actual, expected);
    }

    TEST(GeneratedTransitionBoundaries, EverySamplePositionAtEveryBoundary) {
      const auto& model = GeneratedModel();
      ASSERT_EQ(model.transformers.size(), kLayers);
      for (size_t index = 0; index < kSampleCount; ++index) {
        SCOPED_TRACE(::testing::Message() << "sample " << index);
        const auto& sample = kSamples[index];
        const auto* tokens = kSampleTokens + sample.token_offset;
        const auto* expected = kExpectedStates + sample.state_offset;
        for (size_t position = 0; position < sample.length; ++position) {
          SCOPED_TRACE(::testing::Message() << "entry position " << position);
          ExpectTransition(
              model.position_embedding(tokens[position],
                                       static_cast<int32_t>(position)),
              {expected[position]});
        }
        for (size_t block = 0; block < kLayers; ++block) {
          SCOPED_TRACE(::testing::Message() << "block " << block);
          const auto* input = expected + (2 * block) * sample.length;
          const auto* after_attention = input + sample.length;
          const auto* after_mlp = after_attention + sample.length;
          for (size_t position = 0; position < sample.length; ++position) {
            SCOPED_TRACE(::testing::Message() << "position " << position);
            // SOURCE expectations, not prior callback outputs: compensating
            // mistakes fail.
            ExpectTransition(model.transformers[block].attention(
                                 absl::MakeConstSpan(input, position + 1)),
                             {after_attention[position]});
            ExpectTransition(
                model.transformers[block].mlp(after_attention[position]),
                {after_mlp[position]});
          }
        }
        const auto* final_states = expected + (2 * kLayers) * sample.length;
  )cpp";
  absl::StrAppend(
      &body, "    for (size_t position = ", model.metadata.prompt_tokens - 1,
      "; position < sample.length; ++position) {\n",
      "      SCOPED_TRACE(::testing::Message() << \"language modeling head "
      "position \" << position);\n",
      "      const DiscreteToken target = position + 1 < sample.length ? "
      "tokens[position + 1] : ",
      token_names[model.metadata.eos_token], ";\n",
      "      "
      "ExpectTransition(model.language_modeling_head(final_states[position]),"
      "\n",
      "                       {static_cast<DiscreteHiddenState>(target)});\n   "
      " }\n  }\n}\n");
  absl::StrAppend(
      &body, R"cpp(
        TEST(GeneratedTransitionBoundaries,
             EverySourceLanguageModelingHeadConstraint) {
          const auto& model = GeneratedModel();
  for (size_t index = 0; index <)cpp",
      head.size(), R"cpp(; ++index) {
                           const auto& row = kLanguageModelingHeadRows[index];
                           SCOPED_TRACE(::testing::Message()
                                        << "language modeling head state "
                                        << row.input.value);
                           ExpectTransition(model.language_modeling_head(row.input), {row.output});
                         }
                         }
                         TEST(GeneratedTransitionBoundaries, ExactAttentionDomainMutationProbes) {
                           const auto& model = GeneratedModel();
                           ASSERT_EQ(model.transformers.size(), kLayers);
  for (size_t index = 0; index <)cpp",
      attention_rows.size(),
      R"cpp(; ++index) {
              const auto& probe = kAttentionProbes[index];
              SCOPED_TRACE(::testing::Message()
                           << "probe " << index << " block " << probe.block);
              ExpectTransition(
                  model.transformers[probe.block].attention(absl::MakeConstSpan(
                      kAttentionProbeKeys + probe.offset, probe.length)),
                  probe.expected);
            }
            }
            TEST(GeneratedTransitionBoundaries, EntryAndPointwiseDomainProbes) {
              const auto& model = GeneratedModel();
              ASSERT_EQ(model.transformers.size(), kLayers);
  for (size_t index = 0; index <)cpp",
      entry_probes.size(),
      R"cpp(; ++index) {
              const auto& probe = kEntryProbes[index];
              SCOPED_TRACE(::testing::Message() << "entry probe " << index);
              ExpectTransition(
                  model.position_embedding(probe.token, probe.position),
                  probe.expected);
            }
  for (size_t index = 0; index <)cpp",
      pointwise_rows.size(),
      R"cpp(; ++index) {
              const auto& probe = kMlpProbes[index];
              SCOPED_TRACE(::testing::Message() << "MLP probe " << index
                                                << " block " << probe.block);
              ExpectTransition(model.transformers[probe.block].mlp(probe.input),
                               probe.expected);
            }
  for (size_t index = 0; index <)cpp",
      head_probes.size(),
      R"cpp(; ++index) {
              const auto& probe = kLanguageModelingHeadProbes[index];
              SCOPED_TRACE(::testing::Message()
                           << "language modeling head probe " << index);
              ExpectTransition(model.language_modeling_head(probe.input), probe.expected);
            }
            }
            }  // namespace
            }  // namespace pluto::llm::discretized::gen
      )cpp");
  return body;
}

}  // namespace pluto::llm::discretized::generator
