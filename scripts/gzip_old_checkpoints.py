#!/usr/bin/env python3
"""Archive all but the five newest step_N checkpoint directories."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import re
import shutil
import sys
import tarfile
import tempfile


_KEEP_LATEST = 5
_STEP_DIRECTORY_PATTERN = re.compile(r"^step_([0-9]+)$")


def _checkpoint_directories(checkpoint_directory: Path) -> list[Path]:
    """Returns real step_N directories ordered by their numeric step."""
    checkpoints: list[tuple[int, str, Path]] = []
    with os.scandir(checkpoint_directory) as entries:
        for entry in entries:
            match = _STEP_DIRECTORY_PATTERN.fullmatch(entry.name)
            if match is None or not entry.is_dir(follow_symlinks=False):
                continue
            path = checkpoint_directory / entry.name
            checkpoints.append((int(match.group(1)), entry.name, path))
    checkpoints.sort()
    return [path for _, _, path in checkpoints]


def _archive_directory(source: Path, destination: Path) -> None:
    """Atomically publishes destination, then removes source."""
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        # Store the step_N directory itself as the archive's top-level member,
        # so extracting in the checkpoint parent reconstructs the input tree.
        with tarfile.open(temporary_path, mode="w:gz") as archive:
            archive.add(source, arcname=source.name, recursive=True)
        os.replace(temporary_path, destination)
        shutil.rmtree(source)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def archive_old_checkpoints(
    checkpoint_directory: Path, *, dry_run: bool = False
) -> list[tuple[Path, Path]]:
    """Archives every numeric checkpoint except the newest five.

    Returns the source/destination pairs selected for archiving. Existing
    archives are rejected during preflight, before any source is removed.
    """
    if not checkpoint_directory.is_dir():
        raise NotADirectoryError(
            f"checkpoint directory does not exist: {checkpoint_directory}"
        )

    checkpoints = _checkpoint_directories(checkpoint_directory)
    old_checkpoints = checkpoints[:-_KEEP_LATEST]
    actions = [
        (source, source.with_name(f"{source.name}.tar.gz"))
        for source in old_checkpoints
    ]
    for _, destination in actions:
        if destination.exists():
            raise FileExistsError(f"archive already exists: {destination}")

    if not dry_run:
        # zlib compression releases the GIL, so independent archives can make
        # progress concurrently. Submit every archive before waiting and size
        # the pool to all logical CPUs; ThreadPoolExecutor creates only as many
        # threads as there are actions when fewer checkpoints need archiving.
        with ThreadPoolExecutor(max_workers=os.cpu_count() or 1) as executor:
            futures = [
                executor.submit(_archive_directory, source, destination)
                for source, destination in actions
            ]
            for future in futures:
                future.result()
    return actions


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "checkpoint_directory",
        type=Path,
        help="parent directory containing step_N checkpoint directories",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the archives that would be created without changing files",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        actions = archive_old_checkpoints(
            args.checkpoint_directory, dry_run=args.dry_run
        )
    except (OSError, tarfile.TarError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    verb = "would archive" if args.dry_run else "archived"
    for source, destination in actions:
        print(f"{verb} {source} -> {destination}")
    if not actions:
        print("no checkpoint directories need archiving")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
