"""Copy a new checkpoint, replacing selected weights with exact donor bytes.

Both sources are read-only. Output files are independent copies, never links.
On failure, partial output is retained without a complete patch.json marker;
the caller must inspect it and choose a new output directory for another try.
Empty selections produce an unmodified copy control. Token embedding patches
also change the tied language-model head, which has no separate weight file.
"""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import stat

import numpy as np

from weight_analysis.checkpoint import (
    GPT2Checkpoint, GPT2Config, sha256_file, tensor_manifest)


def _snapshot(directory, specs):
    expected = {spec.filename for spec in specs}
    actual = {path.name for path in directory.iterdir()
              if path.name.startswith('weight_') and path.name.endswith('.bin')}
    if actual != expected:
        raise ValueError('checkpoint weight file set mismatch')
    records = {}
    for spec in specs:
        info = (directory / spec.filename).lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_size != spec.nbytes:
            raise ValueError(f'expected regular {spec.nbytes}-byte {spec.filename}')
        records[spec.filename] = [info.st_dev, info.st_ino, info.st_size,
                                  info.st_mtime_ns, info.st_ctime_ns]
    info = directory.stat()
    return {'directory_identity': [info.st_dev, info.st_ino], 'files': records}


def _hashes(directory, specs):
    return {spec.filename: sha256_file(directory / spec.filename) for spec in specs}


def _equal_bits(left, right):
    """Compare finite FP32 data bitwise, including the sign of zero, in chunks."""
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    left = left.reshape(-1).view(np.uint8)
    right = right.reshape(-1).view(np.uint8)
    return all(np.array_equal(left[start:start + 1024 * 1024],
                              right[start:start + 1024 * 1024])
               for start in range(0, left.size, 1024 * 1024))


def create_patch(original, replacement, output, *, tensors=(), embedding_rows=(),
                 config=GPT2Config()):
    """Create a new original-plus-selected-donor checkpoint and return provenance.

    Tensor names follow GPT2Checkpoint; lm_head.weight aliases the shared token
    embedding. Row indices address logical token IDs, never vocabulary padding.
    The output parent must already exist. Existing outputs, including dangling
    symlinks, and outputs within either source checkpoint are refused.
    """
    original = Path(original).resolve(strict=True)
    replacement = Path(replacement).resolve(strict=True)
    requested_output = Path(output).absolute()
    if os.path.lexists(requested_output):
        raise FileExistsError(requested_output)
    output = requested_output.resolve()
    for source in (original, replacement):
        if output == source or source in output.parents:
            raise ValueError('output must not be inside either source checkpoint')
    specs = tensor_manifest(config)
    names = {spec.name for spec in specs}
    selected = []
    for name in tensors:
        if not isinstance(name, str):
            raise ValueError('tensor names must be strings')
        name = 'token_embedding.weight' if name == 'lm_head.weight' else name
        if name not in names or name in selected:
            raise ValueError(f'unknown or duplicate tensor selection: {name}')
        selected.append(name)
    rows = []
    for row in embedding_rows:
        if (isinstance(row, (bool, np.bool_)) or
                not isinstance(row, (int, np.integer)) or
                not 0 <= row < config.vocab_size or int(row) in rows):
            raise ValueError(f'invalid or duplicate logical embedding row: {row}')
        rows.append(int(row))
    selected.sort()
    rows.sort()
    if rows and 'token_embedding.weight' in selected:
        raise ValueError('whole embedding and embedding rows overlap')

    sources = {'original': original, 'replacement': replacement}
    before = {name: _snapshot(path, specs) for name, path in sources.items()}
    a = GPT2Checkpoint(original, config, check_finite=True)
    b = GPT2Checkpoint(replacement, config, check_finite=True)
    source_hashes = {name: _hashes(path, specs) for name, path in sources.items()}
    for name, path in sources.items():
        if _snapshot(path, specs) != before[name]:
            raise ValueError(f'{name} source changed during validation')

    output.mkdir(exist_ok=False)
    row_bytes = config.d_model * 4
    for spec in specs:
        source = replacement if spec.name in selected else original
        destination = output / spec.filename
        with (source / spec.filename).open('rb') as incoming, destination.open('xb') as outgoing:
            shutil.copyfileobj(incoming, outgoing, length=1024 * 1024)
            if spec.name == 'token_embedding.weight' and rows:
                for row in rows:
                    outgoing.seek(row * row_bytes)
                    outgoing.write(b[spec.name][row].tobytes(order='C'))
        identity = destination.stat()
        if any([identity.st_dev, identity.st_ino] == before[name]['files'][spec.filename][:2]
               for name in sources):
            raise ValueError('output unexpectedly shares a source inode')

    result = GPT2Checkpoint(output, config, check_finite=True)
    output_hashes = _hashes(output, specs)
    for spec in specs:
        if spec.name == 'token_embedding.weight' and rows:
            first = 0
            for row in rows:
                if (not _equal_bits(result[spec.name][first:row], a[spec.name][first:row]) or
                        not _equal_bits(result[spec.name][row], b[spec.name][row])):
                    raise ValueError('patched embedding bytes differ from selection')
                first = row + 1
            if not _equal_bits(result[spec.name][first:], a[spec.name][first:]):
                raise ValueError('unselected embedding bytes changed')
        else:
            source = 'replacement' if spec.name in selected else 'original'
            if output_hashes[spec.filename] != source_hashes[source][spec.filename]:
                raise ValueError(f'output bytes differ from selected source: {spec.name}')
    for name, path in sources.items():
        if _snapshot(path, specs) != before[name]:
            raise ValueError(f'{name} source changed during patch creation')

    provenance = {
        'format': 'pluto-paired-weight-patch-v1', 'complete': True,
        'created_utc': datetime.now(timezone.utc).isoformat(), 'config': asdict(config),
        'operation': 'exact donor byte replacement; no interpolation',
        'sources': {name: {'path': str(path), 'weights_sha256': source_hashes[name],
                           'stat_before_and_after': before[name]}
                    for name, path in sources.items()},
        'selection': {'tensors': selected, 'embedding_rows': rows,
                      'embedding_row_byte_ranges': [[row * row_bytes, (row + 1) * row_bytes]
                                                    for row in rows],
                      'embedding_is_tied_to_lm_head': True},
        'output': {'path': str(output), 'weights_sha256': output_hashes},
        'validation': {'all_weights_finite': True, 'sources_unchanged': True,
                       'selected_bytes_equal_replacement': True,
                       'unselected_bytes_equal_original': True, 'no_hardlinks': True},
        'implementation': {'path': str(Path(__file__).resolve()),
                           'sha256': sha256_file(__file__),
                           'checkpoint_loader_sha256': sha256_file(
                               Path(__file__).with_name('checkpoint.py'))},
    }
    # This completion marker is the only non-weight output and is written last.
    with (output / 'patch.json').open('x') as stream:
        json.dump(provenance, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    return provenance


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original', type=Path, required=True)
    parser.add_argument('--replacement', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tensor', action='append', default=[])
    parser.add_argument('--embedding-row', type=int, action='append', default=[])
    args = parser.parse_args(argv)
    result = create_patch(args.original, args.replacement, args.output,
                          tensors=args.tensor, embedding_rows=args.embedding_row)
    print(json.dumps({'output': result['output']['path'], 'complete': True,
                      'selection': result['selection']}))


if __name__ == '__main__':
    main()
