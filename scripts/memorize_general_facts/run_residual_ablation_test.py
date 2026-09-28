#!/usr/bin/env python3
"""CPU-only checks for paired residual ablations and their command provenance."""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


SCRIPT = Path(__file__).with_name("run_residual_ablation.sh").resolve()
BASH = shutil.which("bash")

# Shell functions take precedence over CUDA tools added to PATH by the runner.
# No real build, Git mutation, or GPU command can run through this wrapper.
MOCK_TOOLS = r"""
bazel() { printf 'mock optimized build\n'; }
nvidia-smi() { printf 'mock GPU\n'; }
git() {
  case "$1" in
    rev-parse) printf 'mock-revision\n' ;;
    status|diff) ;;
    *) return 99 ;;
  esac
}
export -f bazel nvidia-smi git
exec bash "$@"
"""

MOCK_BINARY = r"""#!/usr/bin/env bash
printf '%s\n' "$@"
if [[ ${MOCK_FAIL_NO_RESIDUAL:-0} == 1 ]]; then
  for argument in "$@"; do
    if [[ $argument == --mlp_residual_connections=false ]]; then
      exit 7
    fi
  done
fi
"""


@unittest.skipUnless(BASH, "bash is required")
class ResidualAblationTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="residual ablation ")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.repo = self.root / "mock repo"
        self.script = self.repo / "scripts/memorize_general_facts" / SCRIPT.name
        self.script.parent.mkdir(parents=True)
        shutil.copyfile(SCRIPT, self.script)
        binary = self.repo / (
            "bazel-bin/src/llm/experiments/memorize_general_facts/"
            "memorize_general_facts"
        )
        binary.parent.mkdir(parents=True)
        binary.write_text(MOCK_BINARY)
        binary.chmod(0o755)
        self.corpus = self.repo / "testdata/general_facts_dataset.txt"
        self.corpus.parent.mkdir(parents=True)
        self.corpus.write_text("The capital of France is Paris.\n")
        self.checkpoint = self.root / "source checkpoint"
        self.checkpoint.mkdir()
        (self.checkpoint / "compact_vocabulary.tsv").write_text("fixture mapping\n")
        (self.checkpoint / "weight_0.bin").write_bytes(b"fixture weights")
        self.tokenizer = self.root / "source tokenizer"
        self.tokenizer.mkdir()
        (self.tokenizer / "tokenizer.json").write_text("{}\n")
        self.run_dir = self.root / "new run"

    def run_script(self, *arguments, raw=False, fail_no_residual=False):
        args = [] if raw else [
            f"--checkpoint={self.checkpoint}",
            f"--tokenizer={self.tokenizer}",
            f"--run_dir={self.run_dir}",
        ]
        environment = os.environ.copy()
        environment["MOCK_FAIL_NO_RESIDUAL"] = str(int(fail_no_residual))
        return subprocess.run(
            [BASH, "-c", MOCK_TOOLS, "mock-tools", str(self.script),
             *args, *arguments],
            text=True, capture_output=True, env=environment, timeout=30,
        )

    def condition_args(self, condition):
        return (self.run_dir / f"{condition}.log").read_text().splitlines()

    def test_help_is_non_mutating(self):
        result = self.run_script("--help", raw=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--iso_params", result.stdout)
        self.assertIn("1388 vs. 1380", result.stdout)
        self.assertFalse(self.run_dir.exists())

    def test_invalid_arguments_are_rejected_before_work(self):
        for args in (
            ("--unknown",), ("--steps=-1",), ("--batch_size=0",),
            ("--learning_rate=nan",), ("--iso_params",),
            ("--iso_params=true",), ("--stacked", "--mlp_width=20"),
        ):
            with self.subTest(args=args):
                result = self.run_script(*args)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertFalse(self.run_dir.exists())

    def test_default_pair_preserves_single_mlp_settings_and_snapshots(self):
        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stderr)
        for condition, residual in (("residual", "true"), ("no_residual", "false")):
            args = self.condition_args(condition)
            for expected in (
                "--train_mlp", "--mlp_width=150", "--steps=300000",
                "--batch_size=32", "--seed=3", "--learning_rate=0.01",
                f"--mlp_residual_connections={residual}",
            ):
                self.assertIn(expected, args)
            self.assertFalse(any(arg.startswith("--mlp_iso_parameters") for arg in args))
            self.assertEqual((self.run_dir / f"{condition}.exit_status").read_text(), "0\n")
        self.assertEqual(
            (self.run_dir / "inputs/run_residual_ablation.sh").read_bytes(),
            SCRIPT.read_bytes(),
        )
        self.assertEqual(
            (self.run_dir / "inputs/checkpoint/weight_0.bin").read_bytes(),
            b"fixture weights",
        )
        self.assertIn("./run_residual_ablation.sh", (self.run_dir / "input-sha256.txt").read_text())

    def test_stacked_pair_omits_single_mlp_width(self):
        result = self.run_script("--stacked", "--steps=0")
        self.assertEqual(result.returncode, 0, result.stderr)
        for condition in ("residual", "no_residual"):
            args = self.condition_args(condition)
            self.assertIn("--train_stacked_mlp", args)
            self.assertFalse(any(arg.startswith("--mlp_width") for arg in args))
            self.assertFalse(any(arg.startswith("--mlp_iso_parameters") for arg in args))

    def test_iso_pair_passes_budget_switch_to_both_conditions(self):
        result = self.run_script("--stacked", "--iso_params", "--steps=0")
        self.assertEqual(result.returncode, 0, result.stderr)
        for condition in ("residual", "no_residual"):
            args = self.condition_args(condition)
            self.assertIn("--train_stacked_mlp", args)
            self.assertIn("--mlp_iso_parameters=true", args)
            self.assertFalse(any(arg.startswith("--mlp_width") for arg in args))
        self.assertIn("iso_params=1\n", (self.run_dir / "provenance.txt").read_text())

    def test_failed_job_propagates_for_concurrent_and_serial_pairs(self):
        for serial in (False, True):
            with self.subTest(serial=serial):
                self.run_dir = self.root / f"failure run {serial}"
                args = ["--steps=0"] + (["--serial"] if serial else [])
                result = self.run_script(*args, fail_no_residual=True)
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertEqual((self.run_dir / "residual.exit_status").read_text(), "0\n")
                self.assertEqual((self.run_dir / "no_residual.exit_status").read_text(), "7\n")

    def test_existing_output_is_not_overwritten(self):
        self.run_dir.mkdir()
        evidence = self.run_dir / "evidence.txt"
        evidence.write_text("keep me\n")
        result = self.run_script("--steps=0")
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertEqual(evidence.read_text(), "keep me\n")
        self.assertFalse((self.run_dir / "inputs").exists())


if __name__ == "__main__":
    unittest.main()
