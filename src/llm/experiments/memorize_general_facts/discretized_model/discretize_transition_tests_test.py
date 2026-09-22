#!/usr/bin/env python3
"""Small independent examples for test-only compiled-transition fixtures."""

import copy
import re
import unittest

from discretize_transition_tests import (
    _attention_probes, _entry_probes, _pointwise_probes, _replay_samples, _result,
    render_transition_test,
)


def fixture():
    # IDs intentionally have gaps, and some valid rows are not sample-reachable.
    return {
        "layers": 2, "vocab_size": 4, "eos_token": 3, "prompt_tokens": 1,
        "samples": [{"tokens": [0, 1]}, {"tokens": [1]}],
        "entry": [[0, 0, 10], [1, 0, 30], [1, 1, 20], [2, 1, 25]],
        "attention": [
            [[[10], 100], [[10, 20], 110], [[10, 25], 115], [[30], 120]],
            [[[1000], 2000], [[1000, 1010], 2010],
             [[1000, 1015], 2015], [[1020], 2020]],
        ],
        "mlp": [
            [[100, 1000], [110, 1010], [115, 1015], [120, 1020]],
            [[2000, 3000], [2010, 3010], [2015, 3015], [2020, 3020]],
        ],
        "language_modeling_head": [[3000, 1], [3010, 3], [3015, 0], [3020, 3]],
    }


_NAMES = ["vocab::kA_0", "vocab::kB_1", "vocab::kC_2", "vocab::kEos_3"]


class TransitionFixturesTest(unittest.TestCase):
    def test_replay_preserves_every_boundary_and_real_position(self):
        model = fixture()
        before = copy.deepcopy(model)
        tokens, states, samples = _replay_samples(model)
        self.assertEqual(tokens, [0, 1, 1])
        self.assertEqual(samples, [(0, 0, 2), (2, 10, 1)])
        self.assertEqual(states, [10, 20, 100, 110, 1000, 1010,
                                  2000, 2010, 3000, 3010,
                                  30, 120, 1020, 2020, 3020])
        self.assertEqual(len(states), len(tokens) * (2 * model["layers"] + 1))
        self.assertEqual(model, before)

    def test_source_tables_determine_expectations_even_after_state_relabeling(self):
        model = fixture()
        model["mlp"][0][0][1] = 1005
        for prefix, _ in model["attention"][1]:
            prefix[:] = [1005 if state == 1000 else state for state in prefix]
        _, states, _ = _replay_samples(model)
        self.assertEqual(states[4], 1005)
        self.assertEqual(states[6:], [2000, 2010, 3000, 3010,
                                     30, 120, 1020, 2020, 3020])

    def test_all_language_modeling_head_constraints_including_unreachable_ones_are_emitted(self):
        source = render_transition_test(fixture(), _NAMES)
        for state, token in fixture()["language_modeling_head"]:
            self.assertIn(f"{{{{{state}}}, static_cast<DiscreteHiddenState>({_NAMES[token]})}}", source)
        self.assertIn("EverySourceLanguageModelingHeadConstraint", source)
        self.assertIn("model.language_modeling_head(row.input), {row.output}", source)

    def test_optional_expectations_preserve_supported_zero(self):
        self.assertEqual(_result(True, 0), "{DiscreteHiddenState{0}}")
        self.assertEqual(_result(True, _NAMES[0]),
                         "{static_cast<DiscreteHiddenState>(vocab::kA_0)}")
        self.assertEqual(_result(False, 0), "{std::nullopt}")
        source = render_transition_test(fixture(), _NAMES)
        self.assertIn("EXPECT_EQ(actual, expected);", source)
        self.assertNotIn(".supported", source)

    def test_prompt_only_states_need_not_have_language_modeling_head_constraints(self):
        model = fixture()
        model["prompt_tokens"] = 2
        model["samples"] = model["samples"][:1]
        model["language_modeling_head"] = [row for row in model["language_modeling_head"] if row[0] != 3000]
        _, states, _ = _replay_samples(model)
        self.assertEqual(states[-2:], [3000, 3010])
        source = render_transition_test(model, _NAMES)
        self.assertIn("for (size_t position = 1; position < sample.length;", source)

    def test_missing_source_rows_fail_before_cpp_generation(self):
        for location in ("entry", "attention", "mlp", "language_modeling_head"):
            with self.subTest(location=location):
                model = fixture()
                if location in ("attention", "mlp"):
                    model[location][0] = model[location][0][1:]
                else:
                    model[location] = model[location][1:]
                with self.assertRaisesRegex(ValueError, "unsupported source transition"):
                    render_transition_test(model, _NAMES)

    def test_wrong_source_readout_is_rejected(self):
        model = fixture()
        model["language_modeling_head"][0][1] = 0
        with self.assertRaisesRegex(ValueError, "source readout disagrees"):
            render_transition_test(model, _NAMES)

    def test_attention_probe_labels_use_exact_full_prefix_membership(self):
        for rows in fixture()["attention"]:
            expected = {tuple(prefix): output for prefix, output in rows}
            probes = _attention_probes(rows)
            self.assertLessEqual(len(probes), 64)
            self.assertEqual(probes[0], ((), False, 0))
            self.assertEqual(len({prefix for prefix, _, _ in probes}), len(probes))
            supported = 0
            known_state_rejections = 0
            domain = {state for prefix in expected for state in prefix}
            for prefix, present, output in probes:
                self.assertEqual(present, prefix in expected)
                self.assertEqual(output, expected.get(prefix, 0))
                supported += present
                known_state_rejections += bool(prefix) and not present and all(
                    state in domain for state in prefix)
            self.assertGreater(supported, 0)
            self.assertGreater(known_state_rejections, 0)
            self.assertTrue(any(len(prefix) > 2 for prefix, _, _ in probes))
            self.assertEqual(probes, _attention_probes(list(reversed(rows))))

    def test_probe_limits_and_sparse_domains(self):
        rows = [[state, state + 1] for state in range(10, 2000, 10)]
        probes = _pointwise_probes(rows)
        self.assertEqual(len(probes), 64)
        self.assertIn((0, False, 0), probes)
        self.assertIn((2**31 - 1, False, 0), probes)
        self.assertIn((-1, False, 0), probes)
        self.assertIn((9, False, 0), probes)
        self.assertIn((10, True, 11), probes)
        self.assertIn((11, False, 0), probes)
        self.assertEqual(probes, _pointwise_probes(list(reversed(rows))))
        expected = dict(rows)
        for state, supported, output in probes:
            self.assertEqual(supported, state in expected)
            self.assertEqual(output, expected.get(state, 0))

    def test_int_endpoint_is_not_assumed_unsupported(self):
        rows = [[2**31 - 1, 7], [0, 9]]
        self.assertIn((2**31 - 1, True, 7), _pointwise_probes(rows))
        self.assertIn((0, True, 9), _pointwise_probes(rows))
        attention = [[[2**31 - 1], 7], [[0], 9]]
        self.assertIn(((2**31 - 1,), True, 7), _attention_probes(attention))
        self.assertIn(((0,), True, 9), _attention_probes(attention))

    def test_entry_probes_use_both_token_and_position(self):
        model = fixture()
        expected = {(token, position): output
                    for token, position, output in model["entry"]}
        probes = _entry_probes(model)
        self.assertIn((-1, 0, False, 0), probes)
        self.assertIn((4, 0, False, 0), probes)
        self.assertIn((0, 2**31 - 1, False, 0), probes)
        self.assertIn((0, -(2**31), False, 0), probes)
        self.assertIn((0, -1, False, 0), probes)
        self.assertIn((0, 1, False, 0), probes)
        self.assertIn((1, 1, True, 20), probes)
        for token, position, supported, output in probes:
            self.assertEqual(supported, (token, position) in expected)
            self.assertEqual(output, expected.get((token, position), 0))

    def test_generated_cpp_uses_independent_inputs_and_compact_stage_vectors(self):
        source = render_transition_test(fixture(), _NAMES)
        values = re.search(r"constexpr DiscreteHiddenState kExpectedStates\[\] = \{(.*?)\};",
                           source, re.S).group(1)
        states = [int(value) for value in re.findall(r"\{(-?\d+)\}", values)]
        self.assertEqual(states, _replay_samples(fixture())[1])
        self.assertIn("absl::MakeConstSpan(input, position + 1)", source)
        self.assertIn("model.transformers[block].mlp(after_attention[position])", source)
        self.assertIn("model.transformers[block].attention(", source)
        self.assertIn("model.position_embedding(tokens[position],", source)
        self.assertIn("static_cast<int32_t>(position)", source)
        self.assertNotIn("TransitionResult", source)
        self.assertNotIn(".function", source)
        self.assertNotIn("entry_function", source)
        self.assertIn("input = expected + (2 * block) * sample.length", source)
        self.assertNotIn("vocab::vocab::", source)
        # The fixture cannot populate production tables or execute autoregression.
        self.assertNotIn("PredictNext(", source)
        self.assertNotIn("Generate(", source)
        self.assertNotIn("GeneratedEntry(", source)
        self.assertNotIn("#include \"tables.h\"", source)
        self.assertNotIn("kKeys[]", source)
        self.assertEqual(source.count("TEST(GeneratedTransitionBoundaries,"), 4)

    def test_generation_is_deterministic_and_does_not_change_the_model(self):
        model = fixture()
        before = copy.deepcopy(model)
        source = render_transition_test(model, _NAMES)
        self.assertEqual(model, before)
        model["entry"].reverse()
        model["language_modeling_head"].reverse()
        for rows in model["attention"] + model["mlp"]:
            rows.reverse()
        self.assertEqual(render_transition_test(model, _NAMES), source)

    def test_names_must_cover_vocabulary(self):
        with self.assertRaisesRegex(ValueError, "cover the compact vocabulary"):
            render_transition_test(fixture(), _NAMES[:2])


if __name__ == "__main__":
    unittest.main()
