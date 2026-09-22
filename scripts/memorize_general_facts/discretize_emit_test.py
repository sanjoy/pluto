#!/usr/bin/env python3
"""CPU-only emitter invariants and separation of model versus evidence."""

import copy
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import discretize_emit


def fixture():
    return {
        "schema": 1, "width": 2, "layers": 1, "vocab_size": 4,
        "eos_token": 3, "prompt_tokens": 1,
        "vocabulary": [{"original_id": i+10, "hex": x.hex()}
                       for i, x in enumerate([b"A", b"B", b"C", b"<eos>"])],
        "samples": [{"tokens": [0, 1]}, {"tokens": [1]}],
        "states": [{"id": i, "stage": (i-4)//4, "bits": [i, 0]}
                   for i in range(4, 16)],
        "entry": [[0, 0, 4], [1, 0, 7], [1, 1, 5], [2, 1, 6]],
        "attention": [[[[4], 8], [[4, 5], 9], [[4, 6], 10], [[7], 11]]],
        "mlp": [[[8, 12], [9, 13], [10, 14], [11, 15]]],
        "snap": [[12, 1], [13, 3], [14, 0], [15, 3]],
        "stats": {},
    }


class EmitModelTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.model = fixture()

    def emit(self, name="generated", model=None):
        destination = self.root / name
        manifest = discretize_emit.emit_model(self.model if model is None else model, destination)
        return destination, manifest

    def test_plain_split_sources_reproducible_under_input_table_order(self):
        first, manifest = self.emit("first")
        reordered = copy.deepcopy(self.model)
        for field in ("states", "entry", "snap"):
            reordered[field].reverse()
        reordered["attention"][0].reverse()
        reordered["mlp"][0].reverse()
        second, repeated = self.emit("second", reordered)
        self.assertEqual(manifest, repeated)
        self.assertEqual({p.name: p.read_bytes() for p in first.iterdir()},
                         {p.name: p.read_bytes() for p in second.iterdir()})
        self.assertTrue((first / "attention_0.cc").exists())
        self.assertTrue((first / "mlp_0.cc").exists())
        build = (first / "BUILD.bazel").read_text()
        self.assertNotIn("genrule", build)
        self.assertNotIn("cuda", build.lower())
        self.assertIn(f'srcs = ["{discretize_emit.PACKAGE}:main.cc"]', build)
        self.assertNotIn(f'"{discretize_emit.PACKAGE}:main"', build)
        self.assertNotIn(str(self.root), "".join(p.read_text() for p in first.iterdir()))
        for name, digest in manifest["files"].items():
            self.assertEqual(hashlib.sha256((first / name).read_bytes()).hexdigest(), digest)

    def test_expected_suffixes_never_enter_production_model_sources(self):
        first, _ = self.emit("first")
        alternate = copy.deepcopy(self.model)
        alternate["samples"] = [{"tokens": [0, 2]}, {"tokens": [1]}]
        second, _ = self.emit("second", alternate)
        production = ["tables.h", "model.cc", "entry.cc", "attention_0.cc",
                      "mlp_0.cc", "snap.cc", "vocabulary.cc"]
        for name in production:
            self.assertEqual((first / name).read_bytes(), (second / name).read_bytes())
            text = (first / name).read_text()
            self.assertNotIn("kExpectedTokens", text)
            self.assertNotIn("sample", text.lower())
        self.assertNotEqual((first / "verification.cc").read_bytes(),
                            (second / "verification.cc").read_bytes())

    def test_exact_history_keys_are_flat_integer_sequences_not_hashes(self):
        destination, _ = self.emit()
        text = (destination / "attention_0.cc").read_text()
        self.assertIn("const StateId kKeys[]", text)
        self.assertIn("{1, 2, 9}", text)
        self.assertIn("{3, 2, 10}", text)
        self.assertNotIn("hash", text)
        self.assertNotIn("float", text)

    def test_arbitrary_bytes_are_escaped_without_utf8_roundtrip(self):
        self.model["vocabulary"][0]["hex"] = "0080ff225c0a"
        destination, _ = self.emit()
        self.assertIn(r'"\x00\x80\xff\x22\x5c\x0a", 6',
                      (destination / "vocabulary.cc").read_text())

    def test_existing_output_is_never_modified(self):
        destination, _ = self.emit()
        before = {p.name: p.read_bytes() for p in destination.iterdir()}
        with self.assertRaises(FileExistsError):
            self.emit()
        self.assertEqual(before, {p.name: p.read_bytes() for p in destination.iterdir()})

    def test_publication_race_refuses_even_empty_directory(self):
        publish = discretize_emit._publish
        def competitor(staging, destination):
            destination.mkdir()
            publish(staging, destination)
        with mock.patch.object(discretize_emit, "_publish", side_effect=competitor):
            with self.assertRaises(FileExistsError):
                self.emit()
        self.assertEqual(list((self.root / "generated").iterdir()), [])
        self.assertEqual(list(self.root.glob(".generated.emit-*")), [])

    def test_bad_models_fail_before_staging(self):
        invalid = []
        for value in (0, True, -1):
            model = fixture()
            model["width"] = value
            invalid.append(model)
        model = fixture()
        model["entry"].append(model["entry"][0])
        invalid.append(model)
        model = fixture()
        model["attention"][0].append([[4], 9])
        invalid.append(model)
        model = fixture()
        model["attention"][0][1][0][0] = 8  # Wrong boundary.
        invalid.append(model)
        model = fixture()
        model["snap"][0][1] = 4
        invalid.append(model)
        model = fixture()
        model["states"][0]["bits"][0] = 65536
        invalid.append(model)
        model = fixture()
        model["vocabulary"][0]["hex"] = "FF"
        invalid.append(model)
        for model in invalid:
            with self.subTest(model=model), mock.patch.object(discretize_emit.tempfile, "TemporaryDirectory") as staging:
                with self.assertRaises(ValueError):
                    self.emit(model=model)
                staging.assert_not_called()
        self.assertFalse((self.root / "generated").exists())

    def test_ambiguous_text_encoder_is_rejected_without_changing_token_semantics(self):
        self.model["vocabulary"][2]["hex"] = "4142"  # C decodes to AB.
        self.model["samples"].append({"tokens": [2]})
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            self.emit()
        self.assertFalse((self.root / "generated").exists())


if __name__ == "__main__":
    unittest.main()
