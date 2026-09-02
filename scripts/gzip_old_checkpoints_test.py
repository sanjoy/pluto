#!/usr/bin/env python3

from pathlib import Path
import tarfile
import tempfile
import threading
import unittest
from unittest import mock

from gzip_old_checkpoints import archive_old_checkpoints


class ArchiveOldCheckpointsTest(unittest.TestCase):
    def test_archives_all_but_five_latest_numeric_steps(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            names = [
                "step_001",
                "step_2",
                "step_3",
                "step_4",
                "step_5",
                "step_9",
                "step_10",
                "step_11",
            ]
            for name in names:
                checkpoint = root / name
                checkpoint.mkdir()
                (checkpoint / "weight_0.bin").write_text(name)
            (root / "step_bad").mkdir()
            (root / "step_999").write_text("not a directory")

            actions = archive_old_checkpoints(root)

            self.assertEqual(
                [source.name for source, _ in actions],
                ["step_001", "step_2", "step_3"],
            )
            for name in names[:3]:
                self.assertFalse((root / name).exists())
                archive_path = root / f"{name}.tar.gz"
                self.assertTrue(archive_path.is_file())
                with tarfile.open(archive_path, mode="r:gz") as archive:
                    contents = archive.extractfile(f"{name}/weight_0.bin")
                    self.assertIsNotNone(contents)
                    self.assertEqual(contents.read().decode(), name)
            for name in names[3:]:
                self.assertTrue((root / name).is_dir())
            self.assertTrue((root / "step_bad").is_dir())
            self.assertTrue((root / "step_999").is_file())

    def test_existing_archive_aborts_before_removing_anything(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            for step in range(1, 7):
                (root / f"step_{step}").mkdir()
            (root / "step_1.tar.gz").write_bytes(b"existing archive")

            with self.assertRaises(FileExistsError):
                archive_old_checkpoints(root)

            for step in range(1, 7):
                self.assertTrue((root / f"step_{step}").is_dir())
            self.assertEqual(
                (root / "step_1.tar.gz").read_bytes(), b"existing archive"
            )

    def test_dry_run_does_not_modify_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            for step in range(1, 8):
                (root / f"step_{step}").mkdir()

            actions = archive_old_checkpoints(root, dry_run=True)

            self.assertEqual(
                [source.name for source, _ in actions], ["step_1", "step_2"]
            )
            for step in range(1, 8):
                self.assertTrue((root / f"step_{step}").is_dir())
                self.assertFalse((root / f"step_{step}.tar.gz").exists())

    def test_uses_all_available_cpus_concurrently(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            for step in range(1, 8):
                (root / f"step_{step}").mkdir()

            barrier = threading.Barrier(2, timeout=5)
            worker_threads: list[int] = []

            def observe_worker(source: Path, destination: Path) -> None:
                del source, destination
                worker_threads.append(threading.get_ident())
                barrier.wait()

            with mock.patch(
                "gzip_old_checkpoints.os.cpu_count", return_value=2
            ), mock.patch(
                "gzip_old_checkpoints._archive_directory",
                side_effect=observe_worker,
            ):
                archive_old_checkpoints(root)

            self.assertEqual(len(set(worker_threads)), 2)


if __name__ == "__main__":
    unittest.main()
