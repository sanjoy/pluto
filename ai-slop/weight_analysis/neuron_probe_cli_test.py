"""CPU-only CLI rejection tests for the optimized neuron validation binary.

Build first with `bazel build -c opt //ai-slop/weight_analysis:neuron_probe`.
All inputs here are temporary synthetic files. The empty checkpoint directory
is intentional: malformed groups must fail before checkpoint validation, and
even valid groups must stop there, before creating an executor or any model.
"""

import os
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
BINARY = ROOT / "bazel-bin/ai-slop/weight_analysis/neuron_probe"


class NeuronProbeCliTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not BINARY.is_file():
            raise unittest.SkipTest(
                "Build optimized binary first: bazel build -c opt "
                "//ai-slop/weight_analysis:neuron_probe")

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.checkpoint = self.directory / "empty_checkpoint"
        self.checkpoint.mkdir()
        self.batch = self.directory / "synthetic_batch.i32"
        self.batch.write_bytes(b"")
        self.groups = self.directory / "groups.i32"
        # Repeating the same permutation across the two blocks is legal;
        # uniqueness is required within each block, not across both blocks.
        self.ids = list(range(2048)) * 2
        self.write_groups()
        self.output = self.directory / "must_not_be_created"

    def write_groups(self, values=None, endian="<"):
        values = self.ids if values is None else values
        self.groups.write_bytes(struct.pack(endian + "4096i", *values))

    def command(self):
        return [str(BINARY), "--checkpoint=" + str(self.checkpoint),
                "--batch_tokens=" + str(self.batch),
                "--neuron_groups=" + str(self.groups),
                "--output_dir=" + str(self.output)]

    def reject(self, expected, command=None):
        environment = dict(os.environ)
        # These cases must be rejected by argument/file checks before CUDA.
        # No real checkpoint or real passage file is ever supplied.
        environment["CUDA_VISIBLE_DEVICES"] = ""
        result = subprocess.run(self.command() if command is None else command,
                                cwd=ROOT, env=environment, capture_output=True,
                                text=True, timeout=15, check=False)
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn(expected, result.stderr)
        self.assertNotIn("complete (", result.stdout)
        self.assertFalse(self.output.exists(), result.stderr)
        self.assertEqual(list(self.checkpoint.iterdir()), [])
        self.assertEqual(self.batch.read_bytes(), b"")
        return result

    def test_wrong_file_lengths(self):
        original = self.groups.read_bytes()
        for contents in (b"", original[:-1], original + b"\0\0\0\0"):
            with self.subTest(bytes=len(contents)):
                self.groups.write_bytes(contents)
                self.reject("neuron_groups must be exactly 16384 bytes")

    def test_duplicate_id_in_each_block(self):
        for block in range(2):
            with self.subTest(block=block + 6):
                values = self.ids.copy()
                values[block * 2048 + 1] = 0
                self.write_groups(values)
                self.reject(f"neuron_groups block {block + 6} must contain a separate permutation")

    def test_out_of_range_and_negative_little_endian_ids(self):
        for invalid in (-1, -(2**31), 2048, 2**31 - 1):
            with self.subTest(invalid=invalid):
                values = self.ids.copy()
                values[-1] = invalid
                self.write_groups(values)
                self.reject("neuron_groups block 7 must contain a separate permutation")

    def test_big_endian_groups_rejected(self):
        self.write_groups(endian=">")
        self.reject("neuron_groups block 6 must contain a separate permutation")

    def test_groups_directory_rejected(self):
        command = [argument for argument in self.command()
                   if not argument.startswith("--neuron_groups=")]
        command.append("--neuron_groups=" + str(self.checkpoint))
        self.reject("neuron_groups must be exactly 16384 bytes", command)

    def test_valid_permutations_reach_checkpoint_guard_without_cuda(self):
        # This tests the parser's acceptance boundary without constructing
        # weights of the correct sizes or running the model. The deliberately
        # invalid batch is never read because the checkpoint guard comes first.
        self.reject("full checkpoint must contain exactly 100 weight files")

    def test_unexpected_positional_argument(self):
        self.reject("Unexpected positional arguments", self.command() + ["extra"])

    def test_missing_required_flags(self):
        for prefix in ("--checkpoint=", "--batch_tokens=", "--neuron_groups=", "--output_dir="):
            with self.subTest(missing=prefix):
                command = [argument for argument in self.command() if not argument.startswith(prefix)]
                self.reject("required: --checkpoint --batch_tokens --neuron_groups --output_dir", command)

    def test_nonpositive_microbatch(self):
        for count in (0, -1):
            with self.subTest(count=count):
                self.reject("--batch_sequences must be positive",
                            self.command() + [f"--batch_sequences={count}"])

    def test_unknown_flag(self):
        self.reject("Unknown command line flag", self.command() + ["--not_a_probe_flag=1"])

    def test_invalid_microbatch_flag_type(self):
        self.reject("Illegal value", self.command() + ["--batch_sequences=not_an_integer"])


if __name__ == "__main__":
    unittest.main()
