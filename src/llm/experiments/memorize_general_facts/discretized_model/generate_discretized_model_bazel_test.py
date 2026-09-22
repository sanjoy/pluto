"""Exercise the real Bazel launcher with only its declared runfiles.

The binary path is supplied by BUILD.bazel as $(rootpath
:generate_discretized_model). No checkpoint, GPU, repository working directory,
or third-party Python package is needed.
"""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


_BINARY = None


def write_capture(path):
    """A two-token sentence with exact predictions and three residual stages."""
    header = {"schema": 1, "width": 16, "layers": 1, "vocab_size": 3,
              "eos_token": 2, "prompt_tokens": 1,
              "vocabulary": [{"original_id": i, "hex": word.encode().hex()}
                             for i, word in enumerate(["Hello", " world", "<EOS>"])]}
    sample = {"tokens": [0, 1], "predictions": [1, 2],
              "boundaries": [[[16256 + 2 * stage + row] * 16 for row in range(2)]
                             for stage in range(3)]}
    path.write_text(json.dumps(header) + "\n" + json.dumps(sample) + "\n")


class GenerateBazelBinaryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if _BINARY is None:
            raise unittest.SkipTest("This integration test needs the Bazel binary argument")
        if not _BINARY.is_file():
            raise AssertionError(f"Bazel binary is missing: {_BINARY}")

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def run_generator(self, binary, cwd, *arguments, environment=None):
        env = dict(os.environ)
        env.pop("BUILD_WORKING_DIRECTORY", None)
        if environment is not None:
            for key, value in environment.items():
                if value is None:
                    env.pop(key, None)
                else:
                    env[key] = value
        result = subprocess.run([str(binary), *arguments], cwd=cwd, env=env,
                                capture_output=True, text=True, timeout=90)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def assert_generated(self, output, compact):
        manifest = json.loads((output / "manifest.json").read_text())
        self.assertEqual(manifest["transition_representation"],
                         "control_flow" if compact else "tables")
        provenance = json.loads((output / "provenance.json").read_text())
        self.assertEqual(provenance["verification"], {
            "samples": 1, "targets": 2, "errors": 0, "explicit_eos": 1})
        source = (output / "model.cc").read_text()
        # Confirms that the declared Google-style config, not a caller's local
        # configuration or clang-format's fallback, was used.
        self.assertIn("const DiscreteModel& GeneratedModel()", source)
        self.assertTrue((output / "model.h").is_file())

    def test_launcher_generates_both_representations_from_unrelated_directory(self):
        write_capture(self.root / "capture.jsonl")
        (self.root / ".clang-format").write_text(
            "BasedOnStyle: LLVM\nDerivePointerAlignment: false\n"
            "PointerAlignment: Right\n")
        for compact in (False, True):
            with self.subTest(compact=compact):
                name = "compact" if compact else "tables"
                flags = ["--reduce", "--compact_transitions"] if compact else []
                self.run_generator(_BINARY, self.root,
                                   "--capture=capture.jsonl", f"--output={name}",
                                   "--expected_samples=1", *flags)
                self.assert_generated(self.root / name, compact)

    def test_bazel_run_uses_the_callers_working_directory_for_relative_paths(self):
        caller = self.root / "caller"
        launcher_cwd = self.root / "launcher"
        caller.mkdir()
        launcher_cwd.mkdir()
        write_capture(caller / "capture.jsonl")
        self.run_generator(
            _BINARY, launcher_cwd, "--capture=capture.jsonl", "--output=generated",
            "--save_model=symbolic.json", "--expected_samples=1",
            environment={"BUILD_WORKING_DIRECTORY": str(caller)})
        self.assert_generated(caller / "generated", compact=False)
        self.assertTrue((caller / "symbolic.json").is_file())
        self.assertFalse((launcher_cwd / "generated").exists())
        self.assertFalse((launcher_cwd / "symbolic.json").exists())

    def test_copied_runfiles_work_without_checkout_or_inherited_python_paths(self):
        runfiles_directory = os.environ.get("RUNFILES_DIR") or os.environ.get("TEST_SRCDIR")
        self.assertIsNotNone(runfiles_directory, "Bazel test must supply a runfiles directory")
        original_runfiles = Path(runfiles_directory)
        portable_binary = self.root / "portable_generator"
        shutil.copy2(_BINARY, portable_binary)
        portable_runfiles = Path(str(portable_binary) + ".runfiles")
        # Dereference every source symlink into a self-contained tree. Omit
        # manifests containing old absolute paths so no source-checkout lookup
        # can hide a missing dependency in the declared runfiles closure.
        shutil.copytree(original_runfiles, portable_runfiles, symlinks=False,
                        ignore=shutil.ignore_patterns("MANIFEST"))
        caller = self.root / "unrelated_caller"
        caller.mkdir()
        write_capture(caller / "capture.jsonl")
        clean_environment = {
            "PYTHONPATH": None, "PYTHONHOME": None,
            "RUNFILES_DIR": None, "RUNFILES_MANIFEST_FILE": None,
            "JAVA_RUNFILES": None, "TEST_SRCDIR": None, "TEST_WORKSPACE": None,
        }
        self.run_generator(
            portable_binary, caller, "--capture=capture.jsonl", "--output=portable",
            "--expected_samples=1", "--compact_transitions",
            environment=clean_environment)
        self.assert_generated(caller / "portable", compact=True)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: generate_discretized_model_bazel_test.py <bazel binary>")
    _BINARY = Path(sys.argv.pop(1)).absolute()
    unittest.main()
