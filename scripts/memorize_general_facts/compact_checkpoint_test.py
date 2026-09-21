#!/usr/bin/env python3
"""CPU-only compact-checkpoint validation and byte-preservation tests."""

from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest import mock

import compact_checkpoint
from compact_checkpoint import convert_checkpoint, main, parse_mapping


def mapping_bytes(original_ids):
    return (
        "compact_vocabulary_v1\n"
        "original_vocab_size\t50257\n"
        "original_eos_token\t50256\n"
        f"compact_vocab_size\t{len(original_ids)}\n"
        "compact_id\toriginal_id\n"
        + "".join(f"{compact}\t{original}\n"
                  for compact, original in enumerate(original_ids))
    ).encode("ascii")


class CompactCheckpointTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source, self.payloads = self.make_source("source", 1, 2, 3)
        self.destination = self.root / "compact"
        self.mapping = self.root / "mapping.tsv"
        self.ids = (0, 7, 256, 50256)
        self.mapping.write_bytes(mapping_bytes(self.ids))

    def make_source(self, name, layers, width, ff_width):
        source = self.root / name
        source.mkdir()
        # This independent fixture spells out the two norms and four affine
        # projections in each block, followed by the final norm.
        counts = [50272 * width, 1024 * width]
        for _ in range(layers):
            counts += [width, width, width * (3 * width), 3 * width,
                       width * width, width, width, width,
                       width * ff_width, ff_width, ff_width * width, width]
        counts += [width, width]
        embedding = b"".join(struct.pack("<I", original) * width
                             for original in range(50272))
        # Negative zero and a subnormal must survive row selection bit-for-bit.
        if width >= 2:
            embedding = struct.pack("<II", 0x80000000, 1) + embedding[8:]
        payloads = {f"weight_{index}.bin":
                    (embedding if index == 0 else bytes([index % 256]) * (4 * count))
                    for index, count in enumerate(counts)}
        for filename, contents in payloads.items():
            (source / filename).write_bytes(contents)
        return source, payloads

    def convert(self, **overrides):
        options = dict(source=self.source, destination=self.destination,
                       mapping=self.mapping, layers=1, model_width=2,
                       feed_forward_width=3)
        options.update(overrides)
        return convert_checkpoint(**options)

    def assert_no_output(self):
        self.assertFalse(self.destination.exists())
        self.assertFalse(self.destination.is_symlink())
        self.assertEqual(list(self.root.glob(".compact.compact-*")), [])

    def test_selected_rows_have_no_padding_and_other_weights_are_identical(self):
        (self.source / "optimizer_state.bin").write_bytes(b"not transferred")
        result = self.convert()
        expected = b"".join(self.payloads["weight_0.bin"][token * 8:(token + 1) * 8]
                            for token in self.ids)
        self.assertEqual((self.destination / "weight_0.bin").read_bytes(), expected)
        self.assertEqual(len(expected), 4 * 2 * 4)
        self.assertEqual(expected[:8], struct.pack("<II", 0x80000000, 1))
        for filename, contents in self.payloads.items():
            self.assertEqual((self.source / filename).read_bytes(), contents)
            if filename != "weight_0.bin":
                self.assertEqual((self.destination / filename).read_bytes(), contents)
        self.assertEqual({p.name for p in self.destination.iterdir()},
                         set(self.payloads) | {"compact_vocabulary.tsv"})
        self.assertEqual((self.destination / "compact_vocabulary.tsv").read_bytes(),
                         self.mapping.read_bytes())
        self.assertEqual(result["unique_weight_files"], 16)
        self.assertEqual(result["source_parameters"], 102645)
        self.assertEqual(result["compact_parameters"], 2109)
        self.assertEqual(result["compact_embedding_sha256"], hashlib.sha256(expected).hexdigest())

    def test_actual_4475_row_shape_has_114256_parameters_at_eight_by_sixteen(self):
        source, _ = self.make_source("eight_blocks", 8, 16, 64)
        self.mapping.write_bytes(mapping_bytes(tuple(range(4474)) + (50256,)))
        result = self.convert(source=source, layers=8, model_width=16,
                              feed_forward_width=64)
        self.assertEqual(result["source_parameters"], 847008)
        self.assertEqual(result["compact_parameters"], 114256)
        self.assertEqual(result["compact_vocab_size"], 4475)
        self.assertEqual((self.destination / "weight_0.bin").stat().st_size,
                         4475 * 16 * 4)
        self.assertEqual(result["unique_weight_files"], 100)

    def test_sixteen_blocks_copy_all_196_unique_arrays(self):
        source, payloads = self.make_source("sixteen_blocks", 16, 12, 48)
        result = self.convert(source=source, layers=16, model_width=12,
                              feed_forward_width=48)
        self.assertEqual(result["source_parameters"], 645720)
        self.assertEqual(result["unique_weight_files"], 196)
        self.assertEqual(len(list(self.destination.glob("weight_*.bin"))), 196)
        for index in range(1, 196):
            filename = f"weight_{index}.bin"
            self.assertEqual((self.destination / filename).read_bytes(), payloads[filename])

    def test_generic_eos_only_mapping_is_valid(self):
        self.mapping.write_bytes(mapping_bytes((50256,)))
        result = self.convert()
        self.assertEqual(result["compact_vocab_size"], 1)
        self.assertEqual((self.destination / "weight_0.bin").read_bytes(),
                         self.payloads["weight_0.bin"][50256 * 8:50257 * 8])

    def test_malformed_mappings_are_rejected_before_staging(self):
        valid = mapping_bytes(self.ids)
        invalid = [
            valid[:-1], valid + b"\n", valid.replace(b"\n", b"\r\n"),
            b"\xef\xbb\xbf" + valid, valid + b"\xff",
            valid.replace(b"compact_vocabulary_v1", b"compact_vocabulary_v2"),
            valid.replace(b"original_vocab_size\t50257", b"original_vocab_size\t50272"),
            valid.replace(b"original_eos_token\t50256", b"original_eos_token\t0"),
            valid.replace(b"compact_vocab_size\t4", b"compact_vocab_size\t04"),
            valid.replace(b"compact_vocab_size\t4", b"compact_vocab_size\t0"),
            valid.replace(b"compact_vocab_size\t4", b"compact_vocab_size\t50258"),
            valid.replace(b"compact_vocab_size\t4", b"compact_vocab_size\t3"),
            valid.replace(b"compact_id\toriginal_id", b"original_id\tcompact_id"),
            valid.replace(b"1\t7\n", b"2\t7\n"),
            valid.replace(b"1\t7\n", b"01\t7\n"),
            valid.replace(b"1\t7\n", b"1\t07\n"),
            valid.replace(b"1\t7\n", b"1\t-1\n"),
            valid.replace(b"1\t7\n", b"1\t7\textra\n"),
            mapping_bytes((0, 0, 256, 50256)), mapping_bytes((0, 256, 7, 50256)),
            mapping_bytes((0, 7, 256, 50257)), mapping_bytes((0, 7, 256, 50000)),
        ]
        for contents in invalid:
            with self.subTest(contents=contents), mock.patch(
                    "compact_checkpoint.tempfile.TemporaryDirectory") as staging:
                self.mapping.write_bytes(contents)
                with self.assertRaises(ValueError):
                    self.convert()
                staging.assert_not_called()
                self.assert_no_output()

    def test_missing_extra_and_noncanonical_weight_files_are_rejected(self):
        for extra in ("weight_16.bin", "weight_00.bin", "weight_bad.bin"):
            with self.subTest(extra=extra):
                path = self.source / extra
                path.write_bytes(b"extra")
                with self.assertRaises(ValueError):
                    self.convert()
                self.assert_no_output()
                path.unlink()
        (self.source / "weight_15.bin").unlink()
        with self.assertRaises(ValueError):
            self.convert()
        self.assert_no_output()

    def test_matching_count_with_a_hole_or_symlink_is_rejected(self):
        original = self.source / "weight_15.bin"
        replacement = self.source / "weight_16.bin"
        original.rename(replacement)
        with self.assertRaises(ValueError):
            self.convert()
        replacement.rename(original)
        original.unlink()
        original.symlink_to(self.source / "weight_14.bin")
        with self.assertRaises(ValueError):
            self.convert()
        self.assert_no_output()

    def test_every_source_weight_size_is_checked_before_staging(self):
        for filename, contents in self.payloads.items():
            with self.subTest(filename=filename), mock.patch(
                    "compact_checkpoint.tempfile.TemporaryDirectory") as staging:
                (self.source / filename).write_bytes(contents[:-4])
                with self.assertRaisesRegex(ValueError, "expected .* FP32 bytes"):
                    self.convert()
                staging.assert_not_called()
                self.assert_no_output()
                (self.source / filename).write_bytes(contents)

    def test_source_embedding_requires_all_50272_stored_rows(self):
        path = self.source / "weight_0.bin"
        path.write_bytes(self.payloads[path.name][:50257 * 2 * 4])
        with self.assertRaisesRegex(ValueError, "expected .* FP32 bytes"):
            self.convert()
        self.assert_no_output()

    def test_wrong_shape_and_invalid_integer_dimensions_are_rejected(self):
        invalid = [dict(layers=0), dict(layers=2), dict(layers=-1), dict(layers=True),
                   dict(layers=1.0), dict(layers=2**31), dict(model_width=0),
                   dict(model_width=3), dict(model_width=2**31 - 1),
                   dict(feed_forward_width=0), dict(feed_forward_width=4)]
        for options in invalid:
            with self.subTest(options=options):
                with self.assertRaises(ValueError):
                    self.convert(**options)
                self.assert_no_output()

    def test_existing_directory_file_and_dangling_symlink_are_never_overwritten(self):
        self.destination.mkdir()
        with self.assertRaises(FileExistsError):
            self.convert()
        self.assertEqual(list(self.destination.iterdir()), [])
        self.destination.rmdir()
        self.destination.write_bytes(b"keep this file")
        with self.assertRaises(FileExistsError):
            self.convert()
        self.assertEqual(self.destination.read_bytes(), b"keep this file")
        self.destination.unlink()
        self.destination.symlink_to(self.root / "missing")
        with self.assertRaises(FileExistsError):
            self.convert()
        self.assertTrue(self.destination.is_symlink())

    def test_nested_destination_is_rejected_without_source_changes(self):
        with self.assertRaisesRegex(ValueError, "nested"):
            self.convert(destination=self.source / "compact")
        self.assertEqual({p.name for p in self.source.iterdir()}, set(self.payloads))

    def test_failed_publication_cleans_staging_and_leaves_no_destination(self):
        with mock.patch("compact_checkpoint._publish_directory", side_effect=OSError("failed")):
            with self.assertRaisesRegex(OSError, "failed"):
                self.convert()
        self.assert_no_output()

    def test_concurrently_created_empty_destination_is_not_replaced(self):
        publish = compact_checkpoint._publish_directory

        def concurrent_creator(staging, destination):
            destination.mkdir()
            return publish(staging, destination)

        with mock.patch("compact_checkpoint._publish_directory", side_effect=concurrent_creator):
            with self.assertRaises(FileExistsError):
                self.convert()
        self.assertTrue(self.destination.is_dir())
        self.assertEqual(list(self.destination.iterdir()), [])
        self.assertEqual(list(self.root.glob(".compact.compact-*")), [])

    def test_cli_returns_conversion_metadata(self):
        output = io.StringIO()
        with redirect_stdout(output):
            status = main([f"--source={self.source}", f"--destination={self.destination}",
                           f"--mapping={self.mapping}", "--layers=1", "--model_width=2",
                           "--feed_forward_width=3"])
        self.assertEqual(status, 0)
        self.assertEqual(json.loads(output.getvalue())["compact_vocab_size"], 4)
        self.assertEqual(parse_mapping(self.mapping.read_bytes()), self.ids)


if __name__ == "__main__":
    unittest.main()
