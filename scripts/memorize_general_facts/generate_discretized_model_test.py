"""Driver safety/provenance tests; no checkpoint or CUDA device is needed."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import generate_discretized_model as driver


def write_capture(path):
    header = {"schema": 1, "width": 16, "layers": 1, "vocab_size": 3,
              "eos_token": 2, "prompt_tokens": 1,
              "vocabulary": [{"original_id": i, "hex": word.encode().hex()}
                             for i, word in enumerate(["Hello", " world", "<EOS>"])]}
    sample = {"tokens": [0, 1], "predictions": [1, 2],
              "boundaries": [[[16256 + 2 * stage + row] * 16 for row in range(2)]
                             for stage in range(3)]}
    path.write_text(json.dumps(header) + "\n" + json.dumps(sample) + "\n")


class GenerateDriverTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.capture = self.root / "capture.jsonl"
        write_capture(self.capture)

    def arguments(self, *extra):
        return driver.parser().parse_args([
            "--capture", str(self.capture), "--output", str(self.root / "generated"),
            "--expected_samples=1", *extra])

    def test_generate_format_verify_and_record_hashes(self):
        args = self.arguments("--save_model", str(self.root / "model.json"))
        with contextlib.redirect_stdout(io.StringIO()):
            model = driver.generate(args)
        record = json.loads((args.output / "provenance.json").read_text())
        self.assertEqual(record["source_sha256"], driver.sha256(self.capture))
        self.assertEqual(record["verification"], {
            "samples": 1, "targets": 2, "errors": 0, "explicit_eos": 1})
        self.assertTrue(any(args.output.glob("*.cc")))
        self.assertTrue(record["generated_sources_sha256"])
        manifest = json.loads((args.output / "manifest.json").read_text())
        for name, digest in manifest["files"].items():
            self.assertEqual(driver.sha256(args.output / name), digest)
        self.assertEqual(len(model["states"]), 6)

    def test_existing_output_is_not_overwritten(self):
        args = self.arguments()
        args.output.mkdir()
        sentinel = args.output / "sentinel"
        sentinel.write_text("keep")
        with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
            driver.generate(args)
        self.assertEqual(sentinel.read_text(), "keep")

    def test_optional_state_index_is_inspection_only(self):
        args = self.arguments("--state_index")
        with contextlib.redirect_stdout(io.StringIO()):
            driver.generate(args)
        self.assertTrue((args.output / "state_index.tsv").is_file())
        self.assertNotIn("state_index.tsv", (args.output / "BUILD.bazel").read_text())

    def test_temporary_output_uses_repository_style(self):
        # A different style next to the generated files must not override the
        # repository's Google style (notably its left-aligned references).
        (self.root / ".clang-format").write_text(
            "BasedOnStyle: LLVM\nDerivePointerAlignment: false\n"
            "PointerAlignment: Right\n")
        args = self.arguments()
        with contextlib.redirect_stdout(io.StringIO()):
            driver.generate(args)
        source = (args.output / "model.cc").read_text()
        self.assertIn("const Model& GeneratedModel()", source)
        self.assertNotIn("const Model &GeneratedModel()", source)

    def test_completed_reduction_gets_independent_certificate(self):
        args = self.arguments("--reduce")
        with contextlib.redirect_stdout(io.StringIO()):
            model = driver.generate(args)
        record = json.loads((args.output / "provenance.json").read_text())
        certificate = record["irreducibility_certificate"]
        self.assertEqual(certificate["status"], "proven")
        self.assertEqual(certificate["states"], len(model["states"]))
        self.assertEqual(certificate["proven_pairs"],
                         certificate["same_boundary_pairs"])
        self.assertFalse(certificate["global_minimum_proven"])

    def test_wrong_sample_count_has_no_output(self):
        args = self.arguments("--expected_samples=2")
        with self.assertRaises(ValueError):
            driver.generate(args)
        self.assertFalse(args.output.exists())

    def test_malformed_capture_has_no_output(self):
        self.capture.write_text("not json\n")
        args = self.arguments()
        with self.assertRaises(ValueError):
            driver.generate(args)
        self.assertFalse(args.output.exists())


if __name__ == "__main__":
    unittest.main()
