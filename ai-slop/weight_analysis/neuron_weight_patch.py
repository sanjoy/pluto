"""Make an independent checkpoint with selected MLP output rows attenuated.

This is a CPU-only preparation tool, not an inference result. An MLP output
weight has layout [neuron, residual_channel]; changing row i scales neuron i's
contribution at *every* position. It leaves the MLP output bias, feature input
weights, and every other parameter unchanged. Subsequent layers must still be
run to measure the causal effect; a direct-logit projection is not that effect.

Doses match MlpRowIntervention: zero clears bytes to positive zero, one restores
the original bits, and one half multiplies the original FP32 master row once.
BF16 conversion is deliberately left to the native forward implementation.
Existing output paths are never reused. A failed operation retains its partial
directory without a complete patch.json; choose a new output for retry. The
marker uses a distinct neuron-weight-patch format, not the donor-patch format.
"""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import stat

import numpy as np

from weight_analysis.checkpoint import (
    GPT2Checkpoint, GPT2Config, sha256_file, tensor_manifest)
from weight_analysis.paired_weight_patch import (
    _equal_bits, _hashes, _snapshot)


def _plain_path(path):
    """Reject symlinks in the supplied path, including before a '..' segment."""
    requested = Path(path).absolute()
    for component in (requested, *requested.parents):
        if component.is_symlink():
            raise ValueError(f'symlink paths are not allowed: {component}')
    return requested.resolve()


def _source_snapshot(directory, specs):
    if not directory.is_dir():
        raise ValueError(f'checkpoint is not a directory: {directory}')
    # Regular sidecars are harmless but are not copied. Reject all symlinks,
    # subdirectories, devices, and pipes rather than accidentally following one.
    for entry in directory.iterdir():
        if not stat.S_ISREG(entry.lstat().st_mode):
            raise ValueError(f'checkpoint entry must be a regular file: {entry}')
    return _snapshot(directory, specs)


def _scaled_row(source_row, scale):
    if scale == 0:
        # CUDA memset, unlike multiplication by zero, clears negative zero too.
        return np.zeros(source_row.shape, dtype='<f4')
    if scale == 1:
        return np.array(source_row, dtype='<f4', copy=True)
    return np.multiply(source_row, np.float32(0.5), dtype=np.float32).astype(
        '<f4', copy=False)


def _publish_marker(path, provenance):
    """Publish last and exclusively; remove our own incomplete marker on error."""
    payload = json.dumps(provenance, indent=2, sort_keys=True,
                         allow_nan=False) + '\n'
    with path.open('x') as stream:
        identity = os.fstat(stream.fileno())
        try:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            current = path.lstat()
            if (current.st_dev, current.st_ino) == (identity.st_dev, identity.st_ino):
                path.unlink()
            raise


def create_patch(checkpoint, output, *, block, neurons, scale,
                 config=GPT2Config()):
    """Copy all weights, scaling only selected rows of one block's MLP output.

    `block` and `neurons` are zero-based, unique integer IDs. Neurons are sorted
    in the artifact; an empty selection is permitted as an independent copy
    control. `scale` must be 0, 0.5, or 1 (not a boolean). Source and output must
    have the same step_N basename, with N fitting a signed 32-bit integer, as
    required by the native loss probe. The output parent must already exist.

    Rejects symlinks, overlapping paths, malformed/nonfinite source weights,
    and existing destinations. Sources must be completed immutable checkpoints;
    stat and SHA-256 checks detect modification during preparation. Success
    verifies all selected bits, all unselected bytes, and independent inodes.
    This helper neither runs a model nor demonstrates selective word causality.
    """
    if (isinstance(block, (bool, np.bool_)) or
            not isinstance(block, (int, np.integer)) or
            not 0 <= block < config.n_layers):
        raise ValueError('block must be a valid integer block index')
    block = int(block)
    if (isinstance(scale, (bool, np.bool_)) or
            not isinstance(scale, (int, float, np.integer, np.floating)) or
            not np.isfinite(scale) or scale not in (0, 0.5, 1)):
        raise ValueError('scale must be 0, 0.5, or 1')
    scale = float(scale)
    rows = []
    for neuron in neurons:
        if (isinstance(neuron, (bool, np.bool_)) or
                not isinstance(neuron, (int, np.integer)) or
                not 0 <= neuron < config.d_ff or int(neuron) in rows):
            raise ValueError(f'invalid or duplicate neuron index: {neuron}')
        rows.append(int(neuron))
    rows.sort()

    source = _plain_path(checkpoint)
    destination = _plain_path(output)
    if (destination == source or source in destination.parents or
            destination in source.parents):
        raise ValueError('source and output must not overlap or contain each other')
    if os.path.lexists(destination):
        raise FileExistsError(destination)
    step_match = re.fullmatch(r'step_([0-9]+)', source.name)
    if (not step_match or int(step_match[1]) > 2147483647 or
            destination.name != source.name):
        raise ValueError('source and output require matching step_N names, N <= INT_MAX')
    if not destination.parent.is_dir():
        raise ValueError('output parent must already be a directory')

    specs = tensor_manifest(config)
    name = f'blocks.{block}.mlp.output.weight'
    spec = next(item for item in specs if item.name == name)
    implementation_paths = [Path(__file__).resolve(),
                            Path(__file__).with_name('checkpoint.py').resolve(),
                            Path(__file__).with_name('paired_weight_patch.py').resolve()]
    implementation_hashes = {str(path): sha256_file(path) for path in implementation_paths}
    before = _source_snapshot(source, specs)
    original = GPT2Checkpoint(source, config, check_finite=True)
    source_hashes = _hashes(source, specs)
    if _source_snapshot(source, specs) != before:
        raise ValueError('source changed during validation')

    destination.mkdir(exist_ok=False)
    row_bytes = config.d_model * 4
    source_inodes = {tuple(record[:2]) for record in before['files'].values()}
    for item in specs:
        target = destination / item.filename
        with (source / item.filename).open('rb') as incoming, target.open('xb') as outgoing:
            shutil.copyfileobj(incoming, outgoing, length=1024 * 1024)
            if item.name == name:
                for row in rows:
                    outgoing.seek(row * row_bytes)
                    outgoing.write(_scaled_row(original[name][row], scale).tobytes())
            outgoing.flush()
            os.fsync(outgoing.fileno())
        info = target.lstat()
        if info.st_nlink != 1 or (info.st_dev, info.st_ino) in source_inodes:
            raise ValueError('output unexpectedly shares an inode')

    result = GPT2Checkpoint(destination, config, check_finite=True)
    output_hashes = _hashes(destination, specs)
    for item in specs:
        if item.name != name or not rows:
            if output_hashes[item.filename] != source_hashes[item.filename]:
                raise ValueError(f'unselected output bytes changed: {item.name}')
            continue
        first = 0
        for row in rows:
            if not _equal_bits(result[name][first:row], original[name][first:row]):
                raise ValueError('unselected MLP output rows changed')
            if not _equal_bits(result[name][row], _scaled_row(original[name][row], scale)):
                raise ValueError('selected MLP output row does not match the dose')
            first = row + 1
        if not _equal_bits(result[name][first:], original[name][first:]):
            raise ValueError('unselected MLP output rows changed')
    # Hash again, not merely stat: the artifact identifies the actual source
    # bytes on both sides of copying, including unselected tensors and padding.
    if (_hashes(source, specs) != source_hashes or
            _source_snapshot(source, specs) != before):
        raise ValueError('source changed during patch creation')
    if any(sha256_file(path) != digest for path, digest in implementation_hashes.items()):
        raise ValueError('implementation source changed during patch creation')

    provenance = {
        'format': 'pluto-neuron-weight-patch-v1', 'complete': True,
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'config': asdict(config),
        'source': {'path': str(source), 'step': int(step_match[1]),
                   'weights_sha256': source_hashes, 'stat_before_and_after': before},
        'selection': {'block': block, 'tensor': name, 'filename': spec.filename,
                      'neurons': rows, 'scale': scale,
                      'row_byte_ranges': [[row * row_bytes, (row + 1) * row_bytes]
                                          for row in rows],
                      'applies_at_all_sequence_positions': True},
        'operation': {'zero': 'positive-zero bytes, matching CUDA memset',
                      'half': 'original FP32 master row multiplied once by FP32 0.5',
                      'one': 'original FP32 bits unchanged, including signed zero',
                      'bf16_conversion': 'deferred to native forward execution',
                      'output_bias': 'unchanged'},
        'output': {'path': str(destination), 'weights_sha256': output_hashes},
        'validation': {'all_weights_finite': True, 'source_stat_and_hash_unchanged': True,
                       'selected_bytes_match_dose': True,
                       'unselected_bytes_equal_source': True, 'no_weight_hardlinks': True},
        'implementation': {'sources_sha256': implementation_hashes,
                           'numpy_version': np.__version__},
        'measurement': {'inference_performed': False, 'causal_effect_measured': False,
                        'goal_completion_claimed': False},
    }
    _publish_marker(destination / 'patch.json', provenance)
    return provenance


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--block', type=int, required=True)
    parser.add_argument('--neuron', type=int, action='append', default=[])
    parser.add_argument('--scale', type=float, choices=(0, 0.5, 1), required=True)
    args = parser.parse_args(argv)
    result = create_patch(args.checkpoint, args.output, block=args.block,
                          neurons=args.neuron, scale=args.scale)
    print(json.dumps({'complete': True, 'output': result['output']['path'],
                      'selection': result['selection'],
                      'measurement': result['measurement']}))


if __name__ == '__main__':
    main()
