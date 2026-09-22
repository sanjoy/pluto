#!/usr/bin/env python3
"""Independent, test-only fixtures for compiled integer boundary callbacks.

Expectations come exclusively from the source JSON transition dictionaries.
Each real sample position is stored once per residual boundary. Attention
prefixes are reconstructed from these vectors, avoiding a second copy of every
complete history. This module has no dependency on the production emitter or
its control-flow construction algorithms.
"""


_UINT32_MAX = 2**32 - 1
_INT32_MAX = 2**31 - 1
_PROBE_LIMIT = 64
_RUNTIME = "src/llm/experiments/memorize_general_facts/discretized_model/runtime.h"


def _replay_samples(model):
    """Return tokens, sample offsets, and sample-major boundary vectors."""
    entry = {(token, position): state for token, position, state in model["entry"]}
    attention = [{tuple(prefix): output for prefix, output in rows}
                 for rows in model["attention"]]
    mlp = [dict(rows) for rows in model["mlp"]]
    snap = dict(model["snap"])
    tokens, states, samples = [], [], []
    for number, sample in enumerate(model["samples"]):
        original = sample["tokens"]
        samples.append((len(tokens), len(states), len(original)))
        tokens.extend(original)
        try:
            boundary = [entry[token, position]
                        for position, token in enumerate(original)]
            states.extend(boundary)
            for block in range(model["layers"]):
                boundary = [attention[block][tuple(boundary[:position + 1])]
                            for position in range(len(original))]
                states.extend(boundary)
                boundary = [mlp[block][state] for state in boundary]
                states.extend(boundary)
            # Early prompt positions need not have readout constraints.
            for position in range(model["prompt_tokens"] - 1, len(original)):
                expected = (original[position + 1] if position + 1 < len(original)
                            else model["eos_token"])
                if snap[boundary[position]] != expected:
                    raise ValueError(f"sample {number}, position {position}: "
                                     "source readout disagrees with sample target")
        except KeyError as error:
            raise ValueError(f"sample {number}: unsupported source transition "
                             f"{error}") from error
    return tokens, states, samples


def _evenly_spaced(values, limit):
    if len(values) <= limit:
        return values
    return [values[i * (len(values) - 1) // (limit - 1)] for i in range(limit)]


def _attention_probes(rows, limit=_PROBE_LIMIT):
    """Mutate source histories, then label every probe by exact dictionary lookup.

    Mutations can still be supported; those outputs are checked too. Neither
    shape nor the presence of known state IDs is assumed to imply membership.
    """
    table = {tuple(prefix): output for prefix, output in rows}
    keys = sorted(table)
    probes = {}

    def add(prefix):
        if len(probes) < limit and prefix not in probes:
            probes[prefix] = (prefix in table, table.get(prefix, 0))

    add(())
    add((0,))
    add((_UINT32_MAX,))
    if keys:
        longest = max(keys, key=len)
        add(longest + (longest[-1],))
        add(longest[:-1])
        domain = sorted({state for prefix in keys for state in prefix})
        successor = {state: domain[(index + 1) % len(domain)]
                     for index, state in enumerate(domain)}
        for prefix in _evenly_spaced(keys, limit):
            add(prefix[:-1])
            add(prefix + (prefix[-1],))
            add(prefix[1:])
            add(prefix[::-1])
            for position in sorted({0, len(prefix) // 2, len(prefix) - 1}):
                for state in (successor[prefix[position]], 0, _UINT32_MAX):
                    add(prefix[:position] + (state,) + prefix[position + 1:])
            if len(probes) >= limit:
                break
    return [(prefix, supported, output)
            for prefix, (supported, output) in probes.items()]


def _pointwise_probes(rows, limit=_PROBE_LIMIT):
    """Probe edges and sparse-domain holes without assuming contiguous IDs."""
    table = dict(rows)
    probes = {}

    def add(state):
        if 0 <= state <= _UINT32_MAX and len(probes) < limit:
            probes.setdefault(state, (state in table, table.get(state, 0)))

    add(0)
    add(_UINT32_MAX)
    keys = sorted(table)
    if keys:
        add(keys[0] - 1)
        add(keys[-1] + 1)
        for state in _evenly_spaced(keys, limit):
            for candidate in (state, state - 1, state + 1):
                add(candidate)
    return [(state, supported, output)
            for state, (supported, output) in probes.items()]


def _entry_probes(model, limit=_PROBE_LIMIT):
    table = {(token, position): state for token, position, state in model["entry"]}
    probes = {}

    def add(token, position):
        if (-2**31 <= token <= _INT32_MAX and 0 <= position <= _UINT32_MAX
                and len(probes) < limit):
            key = (token, position)
            probes.setdefault(key, (key in table, table.get(key, 0)))

    for token in (-1, model["vocab_size"], _INT32_MAX):
        add(token, 0)
    for token, position in _evenly_spaced(sorted(table), limit):
        for candidate in (position, position - 1, position + 1, 1024, _UINT32_MAX):
            add(token, candidate)
        add(token - 1, position)
        add(token + 1, position)
    return [(token, position, supported, output)
            for (token, position), (supported, output) in probes.items()]


def _array(typename, name, values, per_line=1):
    values = list(values) or ["{}"]
    lines = ["  " + ", ".join(values[start:start + per_line]) + ","
             for start in range(0, len(values), per_line)]
    return f"constexpr {typename} {name}[] = {{\n" + "\n".join(lines) + "\n};\n"


def _result(supported, output):
    return f"{{StateId{{{output}}}}}" if supported else "{std::nullopt}"


def render_transition_test(model, token_names):
    """Render a standalone gtest source; token_names are C++ token expressions.

    The caller must put this source in a dedicated cc_test, never the model or
    CLI libraries. It needs runtime, generated model, vocabulary_tokens, and
    gtest_main dependencies. The source JSON is expected to be schema-validated
    by the caller; replay additionally rejects missing transitions/bad targets.
    """
    if len(token_names) != model["vocab_size"]:
        raise ValueError("token_names must cover the compact vocabulary")
    tokens, states, samples = _replay_samples(model)
    layers = model["layers"]
    attention_keys, attention_rows, pointwise_rows = [], [], []
    for block in range(layers):
        for prefix, supported, output in _attention_probes(model["attention"][block]):
            attention_rows.append(
                f"{{{block}, {len(attention_keys)}, {len(prefix)}, "
                f"{_result(supported, output)}}}")
            attention_keys.extend(prefix)
        pointwise_rows.extend(
            f"{{{block}, {state}, {_result(supported, output)}}}"
            for state, supported, output in _pointwise_probes(model["mlp"][block]))
    snap_probes = _pointwise_probes(model["snap"])
    entry_probes = _entry_probes(model)
    body = f'''// Generated independent boundary fixtures; test-only, never inference input.
// Expectations replay the source JSON, independently of compiled control flow.
// Histories are reconstructed from boundary vectors at real positions only.
#include <cstddef>
#include "{_RUNTIME}"
#include "vocabulary_tokens.h"
#include "gtest/gtest.h"

namespace pluto::llm::discretized {{
namespace {{
constexpr size_t kLayers = {layers};
constexpr size_t kSampleCount = {len(samples)};
struct Sample {{ size_t token_offset; size_t state_offset; size_t length; }};
// Independent readout expectations; not part of the production model interface.
struct SnapRow {{ StateId input; StateId output; }};
struct AttentionProbe {{
  size_t block; size_t offset; size_t length; TransitionResult expected;
}};
struct PointwiseProbe {{ size_t block; StateId input; TransitionResult expected; }};
struct StateProbe {{ StateId input; TransitionResult expected; }};
struct EntryProbe {{ TokenId token; uint32_t position; TransitionResult expected; }};
'''
    body += _array("TokenId", "kSampleTokens", (token_names[t] for t in tokens), 4)
    body += _array("StateId", "kExpectedStates", map(str, states), 16)
    body += _array("Sample", "kSamples", (f"{{{a}, {b}, {c}}}" for a, b, c in samples))
    body += _array("SnapRow", "kSnapRows",
                   (f"{{{state}, {token_names[token]}}}" for state, token in sorted(model["snap"])))
    body += _array("StateId", "kAttentionProbeKeys", map(str, attention_keys), 16)
    body += _array("AttentionProbe", "kAttentionProbes", attention_rows)
    body += _array("PointwiseProbe", "kMlpProbes", pointwise_rows)
    body += _array("StateProbe", "kSnapProbes",
                   (f"{{{state}, {_result(supported, token_names[output] if supported else 0)}}}"
                    for state, supported, output in snap_probes))
    body += _array("EntryProbe", "kEntryProbes",
                   (f"{{{token_names[token] if 0 <= token < len(token_names) else token}, "
                    f"{position}, {_result(supported, output)}}}"
                    for token, position, supported, output in entry_probes))
    body += f'''
void ExpectTransition(TransitionResult actual, TransitionResult expected) {{
  EXPECT_EQ(actual.output, expected.output);
}}

TEST(GeneratedTransitionBoundaries, EverySamplePositionAtEveryBoundary) {{
  const auto& model = GeneratedModel();
  ASSERT_NE(model.entry_function, nullptr);
  ASSERT_NE(model.snap.function, nullptr);
  ASSERT_EQ(model.attention.size(), kLayers);
  ASSERT_EQ(model.mlp.size(), kLayers);
  for (size_t block = 0; block < kLayers; ++block) {{
    ASSERT_NE(model.attention[block].function, nullptr);
    ASSERT_NE(model.mlp[block].function, nullptr);
  }}
  for (size_t index = 0; index < kSampleCount; ++index) {{
    SCOPED_TRACE(::testing::Message() << "sample " << index);
    const auto& sample = kSamples[index];
    const auto* tokens = kSampleTokens + sample.token_offset;
    const auto* expected = kExpectedStates + sample.state_offset;
    for (size_t position = 0; position < sample.length; ++position) {{
      SCOPED_TRACE(::testing::Message() << "entry position " << position);
      ExpectTransition(model.entry_function(tokens[position], position),
                       {{expected[position]}});
    }}
    for (size_t block = 0; block < kLayers; ++block) {{
      SCOPED_TRACE(::testing::Message() << "block " << block);
      const auto* input = expected + (2 * block) * sample.length;
      const auto* after_attention = input + sample.length;
      const auto* after_mlp = after_attention + sample.length;
      for (size_t position = 0; position < sample.length; ++position) {{
        SCOPED_TRACE(::testing::Message() << "position " << position);
        // Both callbacks receive SOURCE expectations, never prior callback outputs.
        // Two compensating boundary mistakes therefore cannot pass this test.
        ExpectTransition(model.attention[block].function(
                             absl::MakeConstSpan(input, position + 1)),
                         {{after_attention[position]}});
        ExpectTransition(model.mlp[block].function(after_attention[position]),
                         {{after_mlp[position]}});
      }}
    }}
    const auto* final_states = expected + (2 * kLayers) * sample.length;
    for (size_t position = {model['prompt_tokens'] - 1}; position < sample.length;
         ++position) {{
      SCOPED_TRACE(::testing::Message() << "snap position " << position);
      const StateId target = position + 1 < sample.length
          ? tokens[position + 1] : {token_names[model['eos_token']]};
      ExpectTransition(model.snap.function(final_states[position]), {{target}});
    }}
  }}
}}

TEST(GeneratedTransitionBoundaries, EverySourceSnapConstraint) {{
  const auto& model = GeneratedModel();
  ASSERT_NE(model.snap.function, nullptr);
  for (size_t index = 0; index < {len(model['snap'])}; ++index) {{
    const auto& row = kSnapRows[index];
    SCOPED_TRACE(::testing::Message() << "snap state " << row.input);
    ExpectTransition(model.snap.function(row.input), {{row.output}});
  }}
}}

TEST(GeneratedTransitionBoundaries, ExactAttentionDomainMutationProbes) {{
  const auto& model = GeneratedModel();
  ASSERT_EQ(model.attention.size(), kLayers);
  for (size_t index = 0; index < {len(attention_rows)}; ++index) {{
    const auto& probe = kAttentionProbes[index];
    SCOPED_TRACE(::testing::Message() << "probe " << index << " block " << probe.block);
    ASSERT_NE(model.attention[probe.block].function, nullptr);
    ExpectTransition(model.attention[probe.block].function(absl::MakeConstSpan(
                         kAttentionProbeKeys + probe.offset, probe.length)),
                     probe.expected);
  }}
}}

TEST(GeneratedTransitionBoundaries, EntryAndPointwiseDomainProbes) {{
  const auto& model = GeneratedModel();
  ASSERT_NE(model.entry_function, nullptr);
  ASSERT_NE(model.snap.function, nullptr);
  ASSERT_EQ(model.mlp.size(), kLayers);
  for (size_t index = 0; index < {len(entry_probes)}; ++index) {{
    const auto& probe = kEntryProbes[index];
    SCOPED_TRACE(::testing::Message() << "entry probe " << index);
    ExpectTransition(model.entry_function(probe.token, probe.position), probe.expected);
  }}
  for (size_t index = 0; index < {len(pointwise_rows)}; ++index) {{
    const auto& probe = kMlpProbes[index];
    SCOPED_TRACE(::testing::Message() << "MLP probe " << index << " block " << probe.block);
    ASSERT_NE(model.mlp[probe.block].function, nullptr);
    ExpectTransition(model.mlp[probe.block].function(probe.input), probe.expected);
  }}
  for (size_t index = 0; index < {len(snap_probes)}; ++index) {{
    const auto& probe = kSnapProbes[index];
    SCOPED_TRACE(::testing::Message() << "snap probe " << index);
    ExpectTransition(model.snap.function(probe.input), probe.expected);
  }}
}}
}}  // namespace
}}  // namespace pluto::llm::discretized
'''
    return body
