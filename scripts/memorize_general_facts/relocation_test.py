#!/usr/bin/env python3
"""CPU-only CLI/path regressions for the experiment utilities in scripts/."""

from contextlib import chdir, redirect_stderr, redirect_stdout
import io
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock

import run_depth_search
import run_width_depth_search
import summarize_width_depth


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY = SCRIPT_DIRECTORY.parents[1]
NATIVE_DIRECTORY = Path("src/llm/experiments/memorize_general_facts")
TOOLS = (
    "audit_prefixes",
    "compact_checkpoint",
    "run_depth_search",
    "run_width_depth_search",
    "summarize_width_depth",
    "verify_predictions",
)


class RelocationTest(unittest.TestCase):
    def test_all_cli_help_entrypoints_work_from_repository_without_gpu(self):
        # Argument parsing exits before optional tokenizer dependencies or any
        # native subprocess is needed. Disable bytecode even in child CLIs so
        # this smoke test leaves both the source and artifact trees untouched.
        environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        for tool in TOOLS:
            with self.subTest(tool=tool):
                script = Path("scripts/memorize_general_facts") / f"{tool}.py"
                self.assertTrue((REPOSITORY / script).is_file())
                self.assertFalse((REPOSITORY / NATIVE_DIRECTORY / f"{tool}.py").exists())
                result = subprocess.run(
                    [sys.executable, "-B", str(script), "--help"],
                    cwd=REPOSITORY,
                    env=environment,
                    capture_output=True,
                    text=True,
                    timeout=15,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("usage:", result.stdout)

    def test_driver_defaults_keep_native_binary_and_corpus_locations(self):
        # Move orchestration, not the Bazel target or the training data. These
        # defaults retain their documented repository-working-directory base.
        arguments = [
            "--tokenizer=tokenizer",
            "--checkpoint_dir=checkpoints",
            "--output_dir=artifacts",
        ]
        with chdir(REPOSITORY):
            for driver in (run_depth_search, run_width_depth_search):
                with self.subTest(driver=driver.__name__):
                    args = driver.parse_args(arguments)
                    self.assertEqual(
                        args.binary,
                        (REPOSITORY / "bazel-bin" / NATIVE_DIRECTORY /
                         "memorize_general_facts").resolve(),
                    )
                    self.assertEqual(
                        args.corpus,
                        REPOSITORY / "testdata/general_facts_dataset.txt",
                    )
                    self.assertEqual(driver.SCRIPT_DIRECTORY, SCRIPT_DIRECTORY)
                    self.assertTrue(
                        (driver.SCRIPT_DIRECTORY / "verify_predictions.py").is_file()
                    )

    def test_summary_keeps_explicit_historical_artifact_paths(self):
        # Historical evidence stays in src/.../runs. The reporting CLI must
        # pass the requested path through, not reinterpret it under scripts/.
        # Mock the evidence reader so the test needs no archived run snapshots.
        historical = NATIVE_DIRECTORY / "runs/old_run/width_depth_search_summary.json"
        runs = object()
        output = io.StringIO()
        with (
            mock.patch.object(
                summarize_width_depth, "load_runs", return_value=runs
            ) as load,
            mock.patch.object(
                summarize_width_depth, "render_markdown", return_value="report\n"
            ) as render,
            redirect_stdout(output),
        ):
            status = summarize_width_depth.main([str(historical)])
        self.assertEqual(status, 0)
        load.assert_called_once_with([historical], None)
        render.assert_called_once_with(runs)
        self.assertEqual(output.getvalue(), "report\n")

    def test_summary_does_not_silently_discover_or_move_existing_runs(self):
        # Explicit manifest selection was required before relocation. Keep
        # that behavior instead of changing the report's evidence selection.
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
            summarize_width_depth.main([])
        self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
