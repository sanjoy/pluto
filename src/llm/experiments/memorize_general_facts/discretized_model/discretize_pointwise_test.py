#!/usr/bin/env python3
"""Verify boundary-preserving renaming and compiled generated partial functions."""

import copy
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from discretize_core import evaluate_model
from discretize_pointwise import (evaluate_entry, evaluate_pointwise, relabel_mlp_outputs,
                                  render_entry, render_pointwise)


def fixture():
    return {"schema": 1, "width": 1, "layers": 2, "vocab_size": 3,
            "eos_token": 2, "prompt_tokens": 1,
            "vocabulary": [{"original_id": i, "hex": format(65 + i, "02x")} for i in range(3)],
            "states": [{"id": 100 + i, "stage": i // 2, "bits": [100 + i],
                        "members": [1000 + i], "member_count": 1} for i in range(10)],
            "entry": [[0, 0, 100], [1, 0, 101], [1, 1, 101]],
            "attention": [[[[100], 102], [[100, 101], 103], [[101], 103]],
                          [[[105], 106], [[105, 104], 107], [[104], 107]]],
            "mlp": [[[102, 105], [103, 104]], [[106, 109], [107, 108]]],
            "snap": [[108, 2], [109, 1]], "samples": [{"tokens": [0, 1]}, {"tokens": [1]}],
            "stats": {"membership_complete": True}}


class PointwiseTest(unittest.TestCase):
    def test_relabel_preserves_all_boundaries_metadata_and_continuations(self):
        original = fixture()
        before = copy.deepcopy(original)
        model, mapping_rows = relabel_mlp_outputs(original)
        self.assertEqual(original, before)
        self.assertEqual(evaluate_model(model), evaluate_model(original))
        self.assertEqual(model["mlp"], [[[102, 104], [103, 105]], [[107, 110], [108, 111]]])
        self.assertEqual(len(mapping_rows), len(original["states"]))
        after = {row["id"]: row for row in model["states"]}
        old = {row["id"]: row for row in original["states"]}
        for source, target, stage in mapping_rows:
            self.assertEqual(stage, after[target]["stage"])
            self.assertEqual({key: value for key, value in old[source].items() if key != "id"},
                             {key: value for key, value in after[target].items() if key != "id"})
        self.assertEqual(model["stats"]["pointwise_relabeling"]["changed_states"], 6)
        self.assertTrue(model["stats"]["pointwise_relabeling"]["vocabulary_aligned_final_boundaries"])
        self.assertEqual(model["snap"], [[110, 1], [111, 2]])
        again, _ = relabel_mlp_outputs(model)
        self.assertEqual(again["mlp"], model["mlp"])
        self.assertEqual(again["attention"], model["attention"])

    def test_nonbijective_or_noncontiguous_layers_are_not_relabelled(self):
        model = fixture()
        model["mlp"][0][1][1] = model["mlp"][0][0][1]
        renamed, _ = relabel_mlp_outputs(model)
        self.assertIn(0, renamed["stats"]["pointwise_relabeling"]["skipped_layers"])
        self.assertEqual(renamed["mlp"][0], model["mlp"][0])

    def test_final_alignment_requires_complete_bijections_and_disjoint_ranges(self):
        for defect in ("incomplete", "nonbijective"):
            model = fixture()
            if defect == "incomplete":
                model["snap"].pop()
            else:
                model["snap"][1][1] = model["snap"][0][1]
            renamed, _ = relabel_mlp_outputs(model)
            self.assertFalse(renamed["stats"]["pointwise_relabeling"]["vocabulary_aligned_final_boundaries"])
        model, _ = relabel_mlp_outputs(fixture())
        stats = model["stats"]["pointwise_relabeling"]
        self.assertEqual(stats["final_state_base"] - stats["last_attention_base"], model["vocab_size"])
        for state in model["states"]:
            if state["stage"] < 3:
                self.assertLess(state["id"], stats["last_attention_base"])
            else:
                base = stats["last_attention_base"] if state["stage"] == 3 else stats["final_state_base"]
                self.assertTrue(base <= state["id"] < base + model["vocab_size"])

    def test_final_alignment_does_not_exceed_signed_id_range(self):
        model = fixture()
        model["states"].append({"id": 2147483646, "stage": 0})
        renamed, _ = relabel_mlp_outputs(model)
        self.assertFalse(renamed["stats"]["pointwise_relabeling"]
                         ["vocabulary_aligned_final_boundaries"])
        self.assertLessEqual(max(row["id"] for row in renamed["states"]), 2147483647)

    def test_large_sparse_affine_named_map_uses_exact_bitset(self):
        rows = [[1000 + i, i] for i in range(128) if i % 3 != 1]
        body, stats = render_pointwise("Snap", rows, {i: f"vocab::Token{i}" for i in range(128)})
        self.assertEqual(stats["representation"], "sparse_affine_support_mask")
        self.assertEqual(stats["table_bytes"], 16)
        self.assertIn("vocab::Token0", body)
        self.assertIn("no neural-head linearity is implied", body)
        self.assertNotIn("kOutputs", body)
        self.assertLess(stats["source_bytes"], 1000)

    def test_evaluators_distinguish_unsupported_zero(self):
        self.assertEqual(evaluate_pointwise([[3, 0]], 3), (0, True))
        self.assertEqual(evaluate_pointwise([[3, 0]], 4), (0, False))
        self.assertEqual(evaluate_entry([[1, 2, 0]], 1, 2), (0, True))
        self.assertEqual(evaluate_entry([[1, 2, 0]], 1, 3), (0, False))
        with self.assertRaises(ValueError):
            render_pointwise("Bad", [[1, 3], [1, 4]])
        with self.assertRaises(ValueError):
            render_entry("Bad", [[1, 2, 3], [1, 2, 4]], {1: "vocab::B"})

    def test_affine_function_is_guarded_and_named_snap_stays_named(self):
        body, stats = render_pointwise("Mlp", [[10, 20], [11, 21], [12, 22]])
        self.assertEqual(stats["representation"], "guarded_affine")
        self.assertEqual(stats["table_bytes"], 0)
        self.assertIn("state.value < 10 || state.value > 12", body)
        self.assertIn("return {DiscreteHiddenState{state.value + 10}};", body)
        named, stats = render_pointwise("Snap", [[10 + i, (i * 7) % 20] for i in range(20)],
                                        {i: f"vocab::Token{i}" for i in range(20)})
        self.assertIn("vocab::Token19", named)
        self.assertEqual(stats["representation"], "guarded_output_array")
        self.assertEqual(stats["table_bytes"], 40)

    def test_entry_keeps_position_support_and_named_exceptions(self):
        rows = [[0, 0, 20], [0, 2, 20], [0, 3, 21], [1, 1, 21], [3, 0, 20]]
        body, stats = render_entry("Entry", rows, {0: "vocab::A", 1: "vocab::B", 3: "vocab::D"})
        self.assertEqual(stats["position_exceptions"], 1)
        self.assertIn("case vocab::A.value:", body)
        self.assertIn("position == 3", body)
        self.assertIn("position > 3", body)
        self.assertIn("return {};", body)
        self.assertEqual(stats["source_bytes"], len(body.encode()))

    def test_generator_rejects_unrepresentable_ids_and_positions(self):
        for bad in (-1, 2147483648, 4294967295, True, 1.5):
            with self.subTest(bad=bad):
                for rows in ([[bad, 0]], [[0, bad]]):
                    with self.assertRaisesRegex(ValueError, "IDs"):
                        render_pointwise("Bad", rows)
                for rows in ([[bad, 0, 0]], [[0, 0, bad]]):
                    with self.assertRaisesRegex(ValueError, "IDs"):
                        render_entry("Bad", rows, {0: "vocab::A"})
        for position in (-1, 4294967296, True, 1.5):
            with self.assertRaisesRegex(ValueError, "position"):
                render_entry("Bad", [[0, position, 0]], {0: "vocab::A"})

    @unittest.skipUnless(shutil.which("c++"), "C++ compiler unavailable")
    def test_compiled_functions_match_exact_partial_domains(self):
        declarations = ["#include <cstdint>", "#include <optional>", "#include <compare>",
                        "struct DiscreteToken;",
                        "struct DiscreteHiddenState { int value=0; constexpr auto operator<=>(const DiscreteHiddenState&) const = default; constexpr explicit operator DiscreteToken() const; };",
                        "struct DiscreteToken { int value=0; constexpr auto operator<=>(const DiscreteToken&) const = default; constexpr explicit operator DiscreteHiddenState() const { return {value}; } };",
                        "constexpr DiscreteHiddenState::operator DiscreteToken() const { return {value}; }",
                        "struct TransitionResult { std::optional<DiscreteHiddenState> output; };",
                        "namespace vocab { constexpr DiscreteToken A{0}, B{1}, C{2}, D{3}, Far{1000000000}, Max{2147483647}; }"]
        declarations.append("namespace vocab {" + " ".join(f"constexpr DiscreteToken Token{i}{{{i}}};" for i in range(128)) + "}")
        checks = []
        pointwise = {
            "Affine": [[5, 25], [6, 26], [7, 27]],
            "Holes": [[5, 25], [6, 26], [7, 27], [8, 40], [10, 50]],
            "Irregular": [[i, (i * 7) % 13] for i in range(13)],
            "Named": [[i, i % 3] for i in range(15)],
            "SparseAffine": [[20 + i, 100 + i] for i in range(100) if i % 3 != 1],
            "NamedSparseAffine": [[i, i] for i in range(128) if i % 3 != 1],
            "MaxSigned": [[2147483642 + i, i] for i in range(6) if i != 2],
            "EntireSignedSpan": [[0, 0], [2147483647, 2147483647]],
            "HighOutputs": [[i, 2147483647 - (i * 7) % 13] for i in range(13)],
            "HighAffineOutputs": [[0, 2147483645], [1, 2147483646], [2, 2147483647]],
            "ZeroBasedRanges": [[i, i] for i in range(8)] + [[100, 200]],
            "Empty": []}
        names = {0: "vocab::A", 1: "vocab::B", 2: "vocab::C", 3: "vocab::D",
                 1000000000: "vocab::Far", 2147483647: "vocab::Max"}
        for name, rows in pointwise.items():
            token_names = (names if name == "Named" else
                           {i: f"vocab::Token{i}" for i in range(128)} if name == "NamedSparseAffine" else None)
            declarations.append(render_pointwise(name, rows, token_names)[0])
            for state in [-2147483648] + list(range(-2, 130)) + list(range(2147483640, 2147483648)):
                output, supported = evaluate_pointwise(rows, state)
                checks.append(f"{{ auto r={name}(DiscreteHiddenState{{{state}}}); if(r.output.value_or(DiscreteHiddenState{{0}}).value!={output} || r.output.has_value()!={str(supported).lower()}) return 1; }}")
        entries = {
            "Entry": [[0, 0, 20], [0, 2, 20], [0, 3, 21], [1, 1, 21], [3, 0, 20]],
            "WideMask": [[0, 31, 20], [1, 31, 21]],
            "Sparse": [[0, 0, 20], [1000000000, 0, 21]],
            "WidePosition": [[0, 100, 20]],
            "ZeroEntry": [[0, 0, 0]],
            "MaxEntry": [[0, 0, 2147483647], [1, 0, 2147483646]],
            "MaxToken": [[2147483647, 0, 2147483647]],
            "SparseStates": [[0, 0, 0], [1, 0, 2147483647]],
            "EmptyEntry": []}
        for name, rows in entries.items():
            declarations.append(render_entry(name, rows, names)[0])
            for token in [-2147483648, -1, 0, 1, 2, 3, 4, 1000000000, 2147483647]:
                for position in [0, 1, 2, 3, 4, 31, 32, 63, 64, 100, 101, 4294967295]:
                    output, supported = evaluate_entry(rows, token, position)
                    checks.append(f"{{ auto r={name}(DiscreteToken{{{token}}},{position}u); if(r.output.value_or(DiscreteHiddenState{{0}}).value!={output} || r.output.has_value()!={str(supported).lower()}) return 2; }}")
        program = "\n".join(declarations) + "\nint main(){\n" + "\n".join(checks) + "\nreturn 0;}\n"
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source, binary = directory / "pointwise.cc", directory / "pointwise"
            source.write_text(program)
            subprocess.run(["c++", "-std=c++20", "-O1", "-Wall", "-Wextra", "-Werror", "-fsanitize=undefined",
                            str(source), "-o", str(binary)], check=True, capture_output=True, text=True)
            subprocess.run([str(binary)], check=True)


if __name__ == "__main__":
    unittest.main()
