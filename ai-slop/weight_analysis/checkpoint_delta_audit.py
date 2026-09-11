"""Descriptive checkpoint differences, NOT gradients or a text extractor.

Read selected matrices from completed weight-only GPT-2 checkpoints. For every
adjacent pair compute raw delta=W_later-W_earlier in FP64, without assuming
learning rates, weight decay, optimizer continuity, moments, or training data.
Singular-value estimates come from the smaller Gram matrix. Squaring the
condition number loses precision in small singular values: an apparent small
tail is not an exact rank or input-span guarantee.

This tool accepts no corpus, tokenizer, prompts, or model execution. Embedding
padding is excluded. It neither ranks token associations nor reconstructs text.
All 100 weight hashes per checkpoint are checked before/after the calculation.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import sys
import time

import numpy as np

from .checkpoint import GPT2Checkpoint, sha256_file


def _matrix(value, name):
    result = np.asarray(value)
    if (result.ndim != 2 or not all(result.shape) or result.dtype.kind != 'f'
            or not np.isfinite(result).all()):
        raise ValueError(f'{name} must be a nonempty finite floating-point matrix')
    return result


def _f64_hash(array):
    canonical = np.ascontiguousarray(array, dtype='<f8')
    return hashlib.sha256(memoryview(canonical).cast('B')).hexdigest()


def gram_spectrum(delta):
    """Return all descending Gram eigenvalues and singular-value estimates.

    Use delta.T@delta for tall matrices and delta@delta.T for wide matrices.
    Small negative computed eigenvalues are retained in the raw spectrum and
    clipped only when taking square roots. A conservative numerical floor is
    reported explicitly. It is a diagnostic, NOT an interval bound, exact-rank
    result, or a theorem that a gradient/input subspace is present.
    """
    delta = _matrix(delta, 'delta').astype(np.float64, copy=False)
    rows, columns = delta.shape
    if rows >= columns:
        gram, side = delta.T @ delta, 'delta.T @ delta (column-coordinate Gram)'
    else:
        gram, side = delta @ delta.T, 'delta @ delta.T (row-coordinate Gram)'
    if not np.isfinite(gram).all():
        raise ValueError('Gram matrix exceeded finite FP64 range')
    eigenvalues = np.linalg.eigvalsh(gram)[::-1]
    if not np.isfinite(eigenvalues).all():
        raise ValueError('nonfinite Gram eigenvalues')
    largest = max(float(eigenvalues[0]), 0.0)
    # Include the contraction length and eigensolver dimension conservatively.
    # This cannot turn ordinary BLAS/LAPACK arithmetic into an interval proof.
    floor = 64 * np.finfo(np.float64).eps * max(rows, columns) * largest
    if float(eigenvalues[-1]) < -floor:
        raise ArithmeticError('Gram eigenvalue is negative beyond diagnostic roundoff floor')
    nonnegative = np.maximum(eigenvalues, 0)
    singular = np.sqrt(nonnegative)
    energy = float(nonnegative.sum())
    fractions = (nonnegative / energy if energy else np.zeros_like(nonnegative))
    cumulative = np.cumsum(fractions)
    top = {str(k): float(fractions[:min(k, len(fractions))].sum())
           for k in (1, 4, 8, 16, 32, 64, 128, 256, 512)}
    effective = (float(np.exp(-np.sum(fractions[fractions > 0] *
                                      np.log(fractions[fractions > 0])))) if energy else 0.0)
    return {
        'gram_definition': side, 'gram_shape': list(gram.shape),
        'gram_eigenvalues_descending': eigenvalues.tolist(),
        'singular_values_sqrt_clipped_gram_eigenvalues': singular.tolist(),
        'negative_gram_eigenvalue_count': int(np.count_nonzero(eigenvalues < 0)),
        'gram_eigenvalue_diagnostic_floor': floor,
        'singular_value_diagnostic_floor': float(np.sqrt(floor)),
        'singular_values_above_diagnostic_floor': int(np.count_nonzero(nonnegative > floor)),
        'diagnostic_floor_formula': '64 * eps_float64 * max(matrix_shape) * max(largest_gram_eigenvalue,0)',
        'rank_scope': 'count above an ordinary-FP64 diagnostic floor; NOT exact rank, gradient rank, or input-span certification',
        'top_k_squared_singular_value_energy_fraction': top,
        'components_for_energy_fraction': {
            str(fraction): (int(np.searchsorted(cumulative, fraction)) + 1 if energy else 0)
            for fraction in (0.5, 0.9, 0.95, 0.99)},
        'stable_rank': energy / largest if largest else 0.0,
        'entropy_effective_rank': effective,
        'gram_trace': float(np.trace(gram)),
        'sum_clipped_eigenvalues': energy,
    }


def audit_matrix(before, after, top_rows=64):
    """Raw change metrics with no optimizer or data assumptions.

    Row IDs mean token IDs only for the logical embedding. For other matrices
    they identify residual coordinates or hidden neurons, as recorded by the
    caller. No token labels or corpus frequencies enter this calculation.
    """
    before, after = _matrix(before, 'before'), _matrix(after, 'after')
    if before.shape != after.shape or type(top_rows) is not int or top_rows <= 0:
        raise ValueError('equal matrix shapes and positive top_rows required')
    # Nearby FP32 endpoints subtract exactly in FP64 (widely separated
    # exponents need not). This is still a difference of rounded WEIGHTS.
    delta = after.astype(np.float64, copy=True)
    delta -= before
    if not np.isfinite(delta).all():
        raise ValueError('raw difference exceeded finite FP64 range')
    row_squared = np.einsum('ij,ij->i', delta, delta, dtype=np.float64)
    row_norms = np.sqrt(row_squared)
    before_squared = float(np.einsum('ij,ij->', before, before, dtype=np.float64))
    after_squared = float(np.einsum('ij,ij->', after, after, dtype=np.float64))
    change_norm = float(np.sqrt(row_squared.sum()))
    before_norm, after_norm = float(np.sqrt(before_squared)), float(np.sqrt(after_squared))
    if not all(np.isfinite(value) for value in (change_norm, before_norm, after_norm)):
        raise ValueError('matrix norm exceeded finite FP64 range')
    rows = np.arange(len(row_norms))
    ranked = np.lexsort((rows, -row_norms))[:min(top_rows, len(rows))]
    same = int(np.count_nonzero(before == after))
    bitwise_same = None
    if before.dtype == after.dtype:
        byte_element = np.dtype(f'V{before.dtype.itemsize}')
        bitwise_same = int(np.count_nonzero(before.view(byte_element) == after.view(byte_element)))
    return {
        'shape': list(before.shape), 'before_storage_dtype': before.dtype.str,
        'after_storage_dtype': after.dtype.str, 'delta_dtype': '<f8',
        'delta_definition': 'float64(after_weight) - float64(before_weight); no decay correction or optimizer inversion',
        'element_count': int(before.size),
        'maximum_absolute_delta': max(float(delta.max()), -float(delta.min())),
        'delta_frobenius_norm': change_norm,
        'before_frobenius_norm': before_norm, 'after_frobenius_norm': after_norm,
        'delta_over_before_frobenius_norm': change_norm / before_norm if before_norm else None,
        'relative_norm_zero_denominator': before_norm == 0,
        'numerically_unchanged_elements': same,
        'numerically_unchanged_fraction': same / before.size,
        'bitwise_unchanged_elements': bitwise_same,
        'bitwise_unchanged_fraction': bitwise_same / before.size if bitwise_same is not None else None,
        'row_norm_definition': 'sqrt(sum_j delta[row,j]^2)',
        'row_norms_sha256_float64_le': _f64_hash(row_norms),
        'row_norm_count': len(row_norms),
        'row_norm_quantiles': {str(q): float(np.quantile(row_norms, q))
                               for q in (0, .01, .1, .25, .5, .75, .9, .95, .99, 1)},
        'exactly_unchanged_rows': int(np.count_nonzero(row_norms == 0)),
        'top_rows_by_norm': [{'row_id': int(i), 'delta_norm': float(row_norms[i])} for i in ranked],
        'spectrum': gram_spectrum(delta),
    }


def selected_matrices(checkpoint):
    """Yield the six fixed matrices, their row meaning and physical address."""
    last = checkpoint.config.n_layers - 1
    names = [
        ('token_embedding.weight', 'logical_token_id'),
        ('blocks.0.attn.qkv.weight', 'input_residual_coordinate'),
        ('blocks.0.mlp.input.weight', 'input_residual_coordinate'),
        ('blocks.0.mlp.output.weight', 'mlp_neuron'),
        (f'blocks.{last}.mlp.input.weight', 'input_residual_coordinate'),
        (f'blocks.{last}.mlp.output.weight', 'mlp_neuron'),
    ]
    if len({name for name, _ in names}) != 6:
        raise ValueError('audit requires at least two transformer blocks')
    for name, axis in names:
        value = checkpoint.token_embedding if name == 'token_embedding.weight' else checkpoint[name]
        spec = checkpoint.specs[name]
        yield name, value, {'filename': spec.filename, 'row_axis': axis,
                            'physical_shape': list(spec.shape), 'selected_shape': list(value.shape),
                            'selected_byte_range': [0, value.size * value.dtype.itemsize],
                            'storage_order': 'C', 'embedding_padding_excluded': name == 'token_embedding.weight'}


def _file_record(path):
    path = Path(path).resolve()
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha256_file(path)}


def _format_relative_norm(value):
    """Keep zero-baseline matrices valid in progress output as well as JSON."""
    return 'undefined (zero baseline)' if value is None else f'{value:.8g}'


def audit_checkpoints(checkpoint_root, steps, output, command=None):
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    if (not isinstance(steps, (list, tuple)) or len(steps) < 2
            or any(type(step) is not int or step < 0 for step in steps)
            or any(a >= b for a, b in zip(steps, steps[1:]))):
        raise ValueError('steps must be an increasing list of at least two integers')
    root = Path(checkpoint_root).resolve()
    if output.resolve().is_relative_to(root):
        raise ValueError('audit outputs must be outside the checkpoint collection')
    started = time.monotonic()
    sources = {'audit': _file_record(__file__), 'checkpoint_loader': _file_record(Path(__file__).with_name('checkpoint.py'))}
    checkpoints, before, inventories = {}, {}, []
    for step in steps:
        checkpoint = GPT2Checkpoint(root / f'step_{step}', check_finite=True)
        checkpoints[step] = checkpoint
        before[str(step)] = checkpoint.provenance(hash_weights=True)
        entries = sorted(checkpoint.directory.iterdir(), key=lambda p: p.name)
        weight_names = {spec.filename for spec in checkpoint}
        inventories.append({'step': step, 'directory': str(checkpoint.directory),
                            'weight_file_count': len(weight_names),
                            'weight_bytes': sum((checkpoint.directory / name).stat().st_size for name in weight_names),
                            'other_entries': [p.name for p in entries if p.name not in weight_names],
                            'symlinks': [p.name for p in entries if p.is_symlink()]})
    if any(checkpoint.config != checkpoints[steps[0]].config for checkpoint in checkpoints.values()):
        raise ValueError('checkpoint architectures differ')
    intervals = []
    for earlier, later in zip(steps, steps[1:]):
        matrices = []
        for (name, first, address), (second_name, second, second_address) in zip(
                selected_matrices(checkpoints[earlier]), selected_matrices(checkpoints[later])):
            if name != second_name or address != second_address:
                raise ValueError('checkpoint matrix layouts changed')
            timer = time.monotonic()
            measured = audit_matrix(first, second)
            matrices.append({'tensor': name, 'address': address, **measured})
            print(f'{earlier}->{later} {name}: ||delta||/||before||='
                  f'{_format_relative_norm(measured["delta_over_before_frobenius_norm"])}; '
                  f'{time.monotonic()-timer:.2f}s', flush=True)
        intervals.append({'earlier_step': earlier, 'later_step': later,
                          'step_label_difference': later - earlier, 'matrices': matrices})
    after = {str(step): checkpoint.provenance(hash_weights=True) for step, checkpoint in checkpoints.items()}
    if before != after:
        raise ValueError('checkpoint weights or loader provenance changed during audit')
    if any(_file_record(record['path']) != record for record in sources.values()):
        raise ValueError('audit implementation changed while running')
    result = {
        'schema_version': 1, 'stage': 'descriptive_weight_difference_feasibility_not_gradients_or_extraction',
        'created_utc': datetime.now(timezone.utc).isoformat(), 'command': command,
        'runtime': {'python': sys.version, 'numpy': np.__version__, 'platform': platform.platform(),
                    'environment': {key: os.environ.get(key) for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS')}},
        'sources': sources, 'inventory': inventories,
        'checkpoint_provenance_before': before, 'checkpoint_provenance_after': after,
        'all_checkpoint_weights_and_sources_unchanged': True,
        'intervals': intervals, 'elapsed_seconds': time.monotonic() - started,
        'limitations': [
            'Step labels and current source do not authenticate historical training settings or optimizer continuity.',
            'Raw differences of ten-step weight endpoints are not raw gradients or recoverable gradient averages.',
            'Gram spectra describe numerical weight changes, not identified training-token or activation subspaces.',
            'A small Gram eigenvalue can be numerical noise; counts above the diagnostic floor are not exact rank guarantees.',
            'Embedding updates include both tied output-head and input-embedding effects; row changes do not prove input-token exposure.',
            'No corpus, tokenizer labels, model forward/backward, target selection, or sequence decoding is used.',
        ],
    }
    serialized = json.dumps(result, indent=2, allow_nan=False) + '\n'
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        stream.write(serialized)
    print(json.dumps({'output': str(output.resolve()), 'sha256': sha256_file(output),
                      'elapsed_seconds': result['elapsed_seconds']}), flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint-root', type=Path, required=True)
    parser.add_argument('--steps', type=int, nargs='+', default=[12990, 13000, 13010, 13020, 13030])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    command = shlex.join([sys.executable, '-m', 'weight_analysis.checkpoint_delta_audit',
                          *(sys.argv[1:] if argv is None else argv)])
    audit_checkpoints(args.checkpoint_root, args.steps, args.output, command=command)


if __name__ == '__main__':
    main()
