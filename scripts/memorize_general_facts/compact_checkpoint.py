#!/usr/bin/env python3
"""Select active GPT-2 embedding rows into a fresh compact checkpoint.

The source must use the original 50,257-token vocabulary with 50,272 stored
embedding rows. Compact checkpoints store exactly K selected rows, with no
stored vocabulary padding; compute/logit padding is the native model's concern.
Every other unique weight file is copied byte-for-byte. This is a weights-only
transformation, not an optimizer-state resume. Removing output classes changes
softmax normalization and loss, so converted predictions require a fresh audit.

The mapping must use the canonical ASCII/LF compact_vocabulary_v1 format.
All mapping, shape, source-file and destination checks precede staging writes.
The destination parent must already exist. Publication uses Linux renameat2
with RENAME_NOREPLACE, atomically refusing even a concurrently created target.
No model, tokenizer, GPU, or third-party Python dependency is used.
"""

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile


ORIGINAL_VOCABULARY_SIZE = 50257
ORIGINAL_EOS_TOKEN = 50256
SOURCE_EMBEDDING_ROWS = 50272
CONTEXT_LENGTH = 1024
MAPPING_FILENAME = "compact_vocabulary.tsv"


def parse_mapping(contents):
    """Return original IDs, rejecting every noncanonical representation."""
    try:
        text = contents.decode("ascii")
    except UnicodeDecodeError as error:
        raise ValueError("mapping must be ASCII") from error
    if not text.endswith("\n") or "\r" in text:
        raise ValueError("mapping must use LF lines and end with exactly one LF")
    lines = text[:-1].split("\n")
    if len(lines) < 6 or lines[:3] != [
        "compact_vocabulary_v1",
        "original_vocab_size\t50257",
        "original_eos_token\t50256",
    ] or lines[4] != "compact_id\toriginal_id":
        raise ValueError("invalid compact vocabulary mapping header")
    count = re.fullmatch(r"compact_vocab_size\t([1-9][0-9]*)", lines[3])
    if count is None:
        raise ValueError("compact_vocab_size must be a canonical positive integer")
    vocabulary_size = int(count[1])
    if not 1 <= vocabulary_size <= ORIGINAL_VOCABULARY_SIZE:
        raise ValueError("compact vocabulary size is outside the original vocabulary")
    if len(lines) != 5 + vocabulary_size:
        raise ValueError("mapping row count disagrees with compact_vocab_size")
    original_ids = []
    for compact_id, line in enumerate(lines[5:]):
        match = re.fullmatch(r"(0|[1-9][0-9]*)\t(0|[1-9][0-9]*)", line)
        if match is None or int(match[1]) != compact_id:
            raise ValueError("mapping requires canonical contiguous compact IDs")
        original_id = int(match[2])
        if original_id >= ORIGINAL_VOCABULARY_SIZE:
            raise ValueError("mapping original ID is outside the GPT-2 vocabulary")
        if original_ids and original_id <= original_ids[-1]:
            raise ValueError("mapping original IDs must be strictly increasing")
        original_ids.append(original_id)
    if original_ids[-1] != ORIGINAL_EOS_TOKEN:
        raise ValueError("mapping must include the original GPT-2 EOS token")
    return tuple(original_ids)


def _validate_shape(layers, model_width, feed_forward_width):
    if type(layers) is not int or not 0 <= layers < 2**31:
        raise ValueError("layers must be a nonnegative int32 integer")
    if any(type(value) is not int or not 0 < value < 2**31
           for value in (model_width, feed_forward_width)):
        raise ValueError("model and feed-forward widths must be positive int32 integers")
    w, f = model_width, feed_forward_width
    if any(count >= 2**31 for count in (
        SOURCE_EMBEDDING_ROWS * w, 3 * w * w, w * f,
        CONTEXT_LENGTH * 3 * w, CONTEXT_LENGTH * f,
    )):
        raise ValueError("source recipe tensor exceeds the int32 element-count limit")


def _weight_element_counts(layers, width, feed_forward_width):
    yield SOURCE_EMBEDDING_ROWS * width
    yield CONTEXT_LENGTH * width
    block = (width, width, 3 * width * width, 3 * width,
             width * width, width, width, width,
             width * feed_forward_width, feed_forward_width,
             feed_forward_width * width, width)
    for _ in range(layers):
        yield from block
    yield width
    yield width


def _publish_directory(staging, destination):
    # os.rename alone can replace an existing empty directory. RENAME_NOREPLACE
    # also closes the race after the earlier destination-existence check.
    try:
        rename = ctypes.CDLL(None, use_errno=True).renameat2
    except AttributeError as error:
        raise OSError("atomic no-overwrite publication requires Linux renameat2") from error
    rename.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
                       ctypes.c_char_p, ctypes.c_uint)
    rename.restype = ctypes.c_int
    if rename(-100, os.fsencode(staging), -100, os.fsencode(destination), 1):
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(destination))


def _write_file(path, contents):
    with path.open("xb") as output:
        output.write(contents)
        output.flush()
        os.fsync(output.fileno())


def _sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def convert_checkpoint(source, destination, mapping, *, layers, model_width,
                       feed_forward_width):
    """Validate and snapshot all source bytes, then publish a fresh checkpoint."""
    _validate_shape(layers, model_width, feed_forward_width)
    source = Path(source).expanduser().resolve(strict=True)
    if not source.is_dir():
        raise ValueError("source checkpoint must be a directory")
    destination = Path(destination).expanduser()
    if not destination.name or destination.name in (".", ".."):
        raise ValueError("destination must name a fresh directory")
    destination = destination.parent.resolve(strict=True) / destination.name
    if not destination.parent.is_dir():
        raise ValueError("destination parent must be an existing directory")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing existing destination: {destination}")
    if source in destination.parents:
        raise ValueError("destination must not be nested inside the source checkpoint")
    mapping_contents = Path(mapping).expanduser().read_bytes()
    original_ids = parse_mapping(mapping_contents)

    files = {path.name: path for path in source.iterdir()
             if path.name.startswith("weight_") and path.name.endswith(".bin")}
    if len(files) != 4 + 12 * layers:
        raise ValueError(f"source must contain exactly {4 + 12 * layers} unique weight files")
    payloads = []
    for index, elements in enumerate(_weight_element_counts(layers, model_width,
                                                           feed_forward_width)):
        name = f"weight_{index}.bin"
        path = files.get(name)
        if path is None or path.is_symlink() or not path.is_file():
            raise ValueError(f"missing or noncanonical regular source weight: {name}")
        expected_bytes = elements * 4
        if path.stat().st_size != expected_bytes:
            raise ValueError(f"{name}: expected {expected_bytes} FP32 bytes")
        contents = path.read_bytes()
        if len(contents) != expected_bytes:
            raise ValueError(f"{name}: source size changed while reading")
        payloads.append(contents)

    source_parameters = sum(len(data) // 4 for data in payloads)
    row_bytes = model_width * 4
    source_embedding = payloads[0]
    payloads[0] = b"".join(source_embedding[token * row_bytes:(token + 1) * row_bytes]
                            for token in original_ids)
    with tempfile.TemporaryDirectory(prefix=f".{destination.name}.compact-",
                                     dir=destination.parent) as temporary:
        staging = Path(temporary)
        for index, contents in enumerate(payloads):
            _write_file(staging / f"weight_{index}.bin", contents)
        _write_file(staging / MAPPING_FILENAME, mapping_contents)
        _sync_directory(staging)
        _publish_directory(staging, destination)
        _sync_directory(destination.parent)

    return {
        "source": str(source), "destination": str(destination),
        "layers": layers, "model_width": model_width,
        "feed_forward_width": feed_forward_width,
        "original_vocab_size": ORIGINAL_VOCABULARY_SIZE,
        "compact_vocab_size": len(original_ids), "unique_weight_files": len(payloads),
        "source_parameters": source_parameters,
        "compact_parameters": sum(len(data) // 4 for data in payloads),
        "mapping_sha256": hashlib.sha256(mapping_contents).hexdigest(),
        "source_embedding_sha256": hashlib.sha256(source_embedding).hexdigest(),
        "compact_embedding_sha256": hashlib.sha256(payloads[0]).hexdigest(),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--layers", type=int, required=True)
    parser.add_argument("--model_width", type=int, required=True)
    parser.add_argument("--feed_forward_width", type=int, required=True)
    args = parser.parse_args(argv)
    try:
        result = convert_checkpoint(**vars(args))
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
