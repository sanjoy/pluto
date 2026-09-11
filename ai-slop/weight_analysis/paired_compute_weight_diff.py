"""Separate FP32 checkpoint changes from changes at native BF16 input casts.

This is a read-only numerical diagnostic, not a forward pass or an attribution
of the word to particular coordinates. The current GPT-2 recipe casts token
embeddings and dense matrices to BF16 before lookup/MMA. Position embeddings,
dense biases and LayerNorm parameters instead enter arithmetic as FP32. Rounding
those latter parameters individually would invent a precision boundary.

Equal rounded matrix entries cannot affect a fixed forward through that operand
alone. They can still matter to future optimizer updates; changed operands need
not change activations or probabilities. Full native forwards and interventions
remain necessary. This report does not claim a CPU implementation reproduces
the GPU's entire arithmetic, or that BF16 subnormal handling is device-verified.
"""

import argparse
from dataclasses import asdict
import hashlib
import math
from pathlib import Path

import numpy as np

from . import checkpoint, paired_weight_patch, paired_word_cases
from .phrase_reference import bf16


FORMAT = 'pluto-paired-compute-weight-diff-v1'
SOURCE_FILES = (
    'src/llm/recipes/gpt2.h', 'src/llm/recipes/gpt2.cc',
    'src/llm/layers/embedding.cc', 'src/llm/layers/fully_connected.cc',
    'src/llm/layers/norm.cc', 'ai-slop/weight_analysis/phrase_reference.py',
    'ai-slop/weight_analysis/checkpoint.py',
    'ai-slop/weight_analysis/paired_weight_patch.py',
    'ai-slop/weight_analysis/paired_word_cases.py',
    'ai-slop/weight_analysis/paired_compute_weight_diff.py',
)


def operand_precision(name):
    """Only matrices actually cast before lookup/MMA receive BF16 rounding."""
    if name == 'token_embedding.weight' or (
            name.startswith('blocks.') and name.endswith('.weight')):
        return 'bf16'
    return 'fp32'


def _empty():
    return dict(elements=0, stored_bits_changed=0, stored_values_changed=0,
                operand_bits_changed=0, operand_values_changed=0,
                stored_changes_hidden_by_cast=0, stored_subnormal_values=0,
                operand_subnormal_values=0, stored_delta_squared=0.,
                operand_delta_squared=0.)


def _measure(left, right, precision):
    """Bounded chunk counts include signed-zero bits but separate real values."""
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError('nonfinite checkpoint parameter')
    a, b = (bf16(left), bf16(right)) if precision == 'bf16' else (left, right)
    stored_changes = left.view(np.uint32) != right.view(np.uint32)
    operand_changes = a.view(np.uint32) != b.view(np.uint32)
    tiny = np.finfo(np.float32).tiny

    def subnormals(x, y):
        # BF16 has the same minimum normal exponent as FP32. Count both arms.
        return sum(int(np.count_nonzero((np.abs(v) < tiny) & (v != 0)))
                   for v in (x, y))

    delta = right.astype(np.float64) - left.astype(np.float64)
    effective = b.astype(np.float64) - a.astype(np.float64)
    return dict(elements=left.size,
                stored_bits_changed=int(np.count_nonzero(stored_changes)),
                stored_values_changed=int(np.count_nonzero(left != right)),
                operand_bits_changed=int(np.count_nonzero(operand_changes)),
                operand_values_changed=int(np.count_nonzero(a != b)),
                stored_changes_hidden_by_cast=int(np.count_nonzero(stored_changes & ~operand_changes)),
                stored_subnormal_values=subnormals(left, right),
                operand_subnormal_values=subnormals(a, b),
                stored_delta_squared=float(np.sum(delta * delta)),
                operand_delta_squared=float(np.sum(effective * effective)))


def _add(destination, source):
    for key, value in source.items():
        destination[key] += value


def _finish(values):
    result = dict(values)
    for prefix in ('stored', 'operand'):
        result[prefix + '_delta_l2'] = math.sqrt(result.pop(prefix + '_delta_squared'))
    result['hidden_fraction_of_stored_bit_changes'] = (
        values['stored_changes_hidden_by_cast'] / values['stored_bits_changed']
        if values['stored_bits_changed'] else None)
    return result


def compare(left, right, *, selected_rows=(), config=checkpoint.GPT2Config(),
            chunk_elements=262144):
    """Compare exact source bytes; do not identify causality from this alone.

    The primary totals exclude non-token vocabulary padding, which is reported
    separately and still finite-validated/hashed. The tied LM head is counted
    once. Selected logical embedding rows form an explicitly nonadditive subset.
    Hashes come from the bytes scanned; file identity/size/time checks around the
    scan detect ordinary concurrent writes. Inputs must be finished checkpoints.
    """
    if type(chunk_elements) is not int or chunk_elements < 1:
        raise ValueError('chunk_elements must be a positive integer')
    rows = list(selected_rows)
    if (any(type(row) is not int or not 0 <= row < config.vocab_size for row in rows)
            or rows != sorted(set(rows))):
        raise ValueError('selected rows must be unique increasing logical token IDs')
    paths = [Path(path).resolve(strict=True) for path in (left, right)]
    specs = checkpoint.tensor_manifest(config)
    before = [paired_weight_patch._snapshot(path, specs) for path in paths]
    arrays = [checkpoint.GPT2Checkpoint(path, config) for path in paths]
    hashes = [{}, {}]
    totals = {'bf16': _empty(), 'fp32': _empty()}
    padding, selected, tensors = _empty(), _empty(), []
    for spec in specs:
        precision = operand_precision(spec.name)
        flat = [array[spec.name].reshape(-1) for array in arrays]
        digests = [hashlib.sha256(), hashlib.sha256()]
        metrics = _empty()
        logical = (config.vocab_size * config.d_model
                   if spec.name == 'token_embedding.weight' else flat[0].size)
        for start in range(0, flat[0].size, chunk_elements):
            stop = min(start + chunk_elements, flat[0].size)
            raw = [array[start:stop] for array in flat]
            for digest, value in zip(digests, raw):
                digest.update(value.tobytes())
            split = min(max(logical - start, 0), stop - start)
            measured = _measure(raw[0][:split], raw[1][:split], precision)
            _add(metrics, measured)
            _add(totals[precision], measured)
            if split < stop - start:
                _add(padding, _measure(raw[0][split:], raw[1][split:], precision))
        for table, digest in zip(hashes, digests):
            table[spec.filename] = digest.hexdigest()
        tensors.append(dict(name=spec.name, filename=spec.filename,
                            operand_precision=precision, **_finish(metrics)))
        if spec.name == 'token_embedding.weight':
            for row in rows:
                _add(selected, _measure(arrays[0][spec.name][row],
                                        arrays[1][spec.name][row], precision))
    for path, snapshot in zip(paths, before):
        if paired_weight_patch._snapshot(path, specs) != snapshot:
            raise ValueError('checkpoint changed during comparison')
    combined = _empty()
    for metrics in totals.values():
        _add(combined, metrics)
    return dict(format=FORMAT, config=asdict(config),
                left=dict(path=str(paths[0]), weights_sha256=hashes[0]),
                right=dict(path=str(paths[1]), weights_sha256=hashes[1]),
                scope='Logical vocabulary only; tied dictionary counted once; no activation simulation.',
                rounding='IEEE BF16 round-to-nearest ties-to-even; device subnormal behavior not certified.',
                totals=_finish(combined), by_operand_precision={k: _finish(v) for k, v in totals.items()},
                vocabulary_padding=_finish(padding),
                selected_embedding_rows=dict(ids=rows, **_finish(selected)), tensors=tensors)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--left', type=Path, required=True)
    parser.add_argument('--right', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--row', action='append', type=int, default=[])
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    if any(output == path.resolve() or path.resolve() in output.parents
           for path in (args.left, args.right)):
        raise ValueError('output must be outside source checkpoints')
    repo = Path(__file__).resolve().parents[2]
    sources = [paired_word_cases._record(repo / path) for path in SOURCE_FILES]
    result = compare(args.left, args.right, selected_rows=args.row)
    if sources != [paired_word_cases._record(repo / path) for path in SOURCE_FILES]:
        raise ValueError('source changed during comparison')
    result['source_records'] = sources
    paired_word_cases._write_json(args.output, result)


if __name__ == '__main__':
    main()
