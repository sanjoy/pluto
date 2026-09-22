#!/usr/bin/env python3
"""Independent local witnesses must justify every irreducibility claim."""

import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest

from discretize_certificate import CertificateError, certify_model, main


def model_with_distinct_outputs(size=2):
    vocabulary = max(size, 3)
    first = vocabulary
    middle = first + size
    final = middle + size
    return {
        "schema": 1, "width": 1, "layers": 1, "vocab_size": vocabulary,
        "states": [{"id": first + stage * size + index, "stage": stage, "bits": [0]}
                   for stage in range(3) for index in range(size)],
        "entry": [[index, 0, first + index] for index in range(size)],
        "attention": [[[[first + index], middle + index] for index in range(size)]],
        "mlp": [[[middle + index, final + index] for index in range(size)]],
        "snap": [[final + index, index] for index in range(size)],
    }


class DiscretizeCertificateTest(unittest.TestCase):
    def test_proves_every_pair_without_reading_search_claims_or_mutating_model(self):
        model = model_with_distinct_outputs()
        model["stats"] = {"search": {"pairwise_irreducible": False}}
        before = copy.deepcopy(model)
        result = certify_model(model)
        self.assertEqual(result["status"], "proven")
        self.assertEqual(result["proven_pairs"], 3)
        self.assertEqual(result["same_boundary_pairs"], 3)
        self.assertEqual(result["attention_pairs_checked"], 1)
        self.assertTrue(result["pairwise_irreducible_proven"])
        self.assertFalse(result["global_minimum_proven"])
        self.assertEqual(model, before)

    def test_mergeable_counterexample_cannot_receive_certificate(self):
        model = model_with_distinct_outputs()
        model["snap"][1][1] = model["snap"][0][1]
        model["stats"] = {"search": {"pairwise_irreducible": True}}
        result = certify_model(model)
        self.assertEqual(result["status"], "inconclusive")
        self.assertEqual(result["unresolved_stage"], 2)
        self.assertFalse(result["pairwise_irreducible_proven"])
        self.assertEqual(result["proven_pairs"], 0)

    def test_unknown_snap_label_is_inconclusive(self):
        model = model_with_distinct_outputs()
        model["snap"].pop()
        result = certify_model(model)
        self.assertEqual(result["status"], "inconclusive")
        self.assertIn("fixed readout label", result["reason"])

    def test_wrong_vector_shape_and_cross_boundary_transition_are_errors(self):
        model = model_with_distinct_outputs()
        model["states"][0]["bits"] = [0, 1]
        with self.assertRaisesRegex(CertificateError, "state bits"):
            certify_model(model)
        model = model_with_distinct_outputs()
        model["attention"][0][0][1] = model["snap"][0][0]
        with self.assertRaisesRegex(CertificateError, "boundary 1"):
            certify_model(model)

    def test_collision_between_two_rewritten_full_histories_is_detected(self):
        model = model_with_distinct_outputs()
        # Neither normalized key [4,4] exists in the original table: the
        # checker must also compare normalized keys against each other.
        model["attention"] = [[[[3, 4], 5], [[4, 3], 6]]]
        self.assertEqual(certify_model(model)["status"], "proven")

    def test_different_prefix_lengths_do_not_create_a_false_collision(self):
        model = model_with_distinct_outputs()
        model["attention"] = [[[[3], 5], [[4, 3], 6]]]
        result = certify_model(model)
        self.assertEqual(result["status"], "inconclusive")
        self.assertEqual(result["unresolved_stage"], 0)
        self.assertEqual(result["unresolved_pair"], [3, 4])

    def test_missing_or_noninjective_mlp_is_inconclusive(self):
        model = model_with_distinct_outputs()
        model["mlp"][0].pop()
        self.assertEqual(certify_model(model)["status"], "inconclusive")
        model = model_with_distinct_outputs()
        model["mlp"][0][1][1] = model["mlp"][0][0][1]
        result = certify_model(model)
        self.assertEqual(result["unresolved_stage"], 1)
        self.assertFalse(result["pairwise_irreducible_proven"])

    def test_large_alphabet_uses_exact_tuple_fallback(self):
        result = certify_model(model_with_distinct_outputs(257))
        self.assertEqual(result["status"], "proven")
        self.assertEqual(result["attention_pairs_checked"], 257 * 256 // 2)
        self.assertEqual(result["proven_pairs"], 3 * 257 * 256 // 2)

    def test_cli_outputs_hash_and_distinct_result_exit_codes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            model = model_with_distinct_outputs()
            for expected_status, expected_code in (("proven", 0), ("inconclusive", 2)):
                path.write_text(json.dumps(model), encoding="utf-8")
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    code = main(["--model", str(path)])
                result = json.loads(output.getvalue())
                self.assertEqual(code, expected_code)
                self.assertEqual(result["status"], expected_status)
                self.assertEqual(len(result["model_sha256"]), 64)
                model["snap"].pop()
            path.write_text("not JSON", encoding="utf-8")
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                code = main(["--model", str(path)])
            self.assertEqual(code, 1)
            self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
