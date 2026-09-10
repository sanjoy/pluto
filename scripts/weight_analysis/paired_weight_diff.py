"""Read-only, bounded-memory differences between paired Pluto GPT-2 runs.

Run with ``python -m scripts.weight_analysis.paired_weight_diff``. The function
API accepts an explicit GPT2Config for other architectures and small fixtures;
the CLI uses the current recipe's default layout, not a guessed architecture.

A weight difference is an intervention's training-trajectory effect only when
the experiment really holds everything else fixed. It is not evidence that a
word is stored in the largest-moving neuron. Different step counts, optimizer
states, initialization, token positions, or sample order can confound it. The
raw checkpoint format contains none of that provenance, so this report cannot
certify it. Layer/neuron patching and behavior controls are separate experiments.

The tied embedding/head allocation is counted once. Model norms include physical
padding, but embedding token rankings exclude it. All arithmetic is FP64 on
bounded FP32 mmap slices; the full checkpoint delta is never materialized.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np

from .checkpoint import GPT2Checkpoint, GPT2Config


DEFAULT_CHUNK_ELEMENTS = 256 * 1024


def _positive_integer(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f'{name} must be a positive integer')


def _snapshot(checkpoint):
    """Detect ordinary concurrent writes/replacement; not an adversarial proof."""
    result = {}
    for spec in checkpoint.manifest:
        stat = (checkpoint.directory / spec.filename).stat()
        result[spec.filename] = (stat.st_dev, stat.st_ino, stat.st_size,
                                 stat.st_mtime_ns, stat.st_ctime_ns)
    return result


def _cosine(dot, squared_a, squared_b):
    if not squared_a or not squared_b:
        return None
    # Independent floating-point reductions can land just outside [-1, 1].
    return max(-1., min(1., dot / (math.sqrt(squared_a) * math.sqrt(squared_b))))


def _metrics(sums, has_initial):
    a, b, delta = (math.sqrt(sums[key]) for key in ('a2', 'b2', 'delta2'))
    result = {
        'original_l2': a,
        'replacement_l2': b,
        'delta_l2': delta,
        'relative_delta_to_original_l2': delta / a if a else None,
        'original_replacement_cosine': _cosine(sums['ab'], sums['a2'], sums['b2']),
    }
    if has_initial:
        result['from_initial'] = {
            'original_delta_l2': math.sqrt(sums['ai2']),
            'replacement_delta_l2': math.sqrt(sums['bi2']),
            'delta_cosine': _cosine(sums['aibi'], sums['ai2'], sums['bi2']),
        }
    return result


def _ranking_axis(name):
    if name == 'token_embedding.weight' or name.endswith(
            ('.mlp.output.weight', '.attn.output.weight')):
        return 0
    if name.endswith(('.mlp.input.weight', '.attn.qkv.weight')):
        return 1
    return None


def _compare_tensor(arrays, spec, chunk_elements):
    """Single finite-validated scan, hashing exactly the bytes being compared.

    Group squared norms accumulate alongside tensor metrics. Integer indices
    are chunk-sized, not tensor-sized. Biases remain in tensor/model metrics,
    but neuron/head rankings explicitly describe projection weights only.
    """
    flat = [array.reshape(-1) for array in arrays]
    hashes = [hashlib.sha256() for _ in flat]
    axis = _ranking_axis(spec.name)
    energies = None if axis is None else np.zeros(spec.shape[axis], dtype=np.float64)
    keys = ['a2', 'b2', 'delta2', 'ab']
    if len(arrays) == 3:
        keys += ['ai2', 'bi2', 'aibi']
    # Neumaier accumulation keeps two scalars per metric, even if a caller
    # chooses tiny chunks. Keeping a list of every chunk subtotal would make
    # temporary memory grow with the checkpoint size instead.
    partials = {key: [0., 0.] for key in keys}

    def accumulate(key, value):
        total, correction = partials[key]
        updated = total + value
        if abs(total) >= abs(value):
            correction += (total - updated) + value
        else:
            correction += (value - updated) + total
        partials[key] = [updated, correction]

    for start in range(0, flat[0].size, chunk_elements):
        stop = min(start + chunk_elements, flat[0].size)
        chunks = []
        for index, array in enumerate(flat):
            raw = array[start:stop]
            if not np.isfinite(raw).all():
                label = ('original', 'replacement', 'initial')[index]
                raise ValueError(f'non-finite {label} parameter in {spec.name}')
            hashes[index].update(raw.tobytes(order='C'))
            chunks.append(np.asarray(raw, dtype=np.float64))
        a, b = chunks[:2]
        delta = b - a
        squared_delta = delta * delta
        accumulate('a2', float(np.sum(a * a)))
        accumulate('b2', float(np.sum(b * b)))
        accumulate('delta2', float(np.sum(squared_delta)))
        accumulate('ab', float(np.sum(a * b)))
        if len(chunks) == 3:
            ai, bi = a - chunks[2], b - chunks[2]
            accumulate('ai2', float(np.sum(ai * ai)))
            accumulate('bi2', float(np.sum(bi * bi)))
            accumulate('aibi', float(np.sum(ai * bi)))
        if energies is not None:
            positions = np.arange(start, stop, dtype=np.int64)
            groups = positions // spec.shape[1] if axis == 0 else positions % spec.shape[1]
            energies += np.bincount(groups, weights=squared_delta, minlength=len(energies))
    sums = {key: math.fsum(values) for key, values in partials.items()}
    if not all(math.isfinite(value) for value in sums.values()):
        raise ArithmeticError(f'non-finite comparison arithmetic in {spec.name}')
    return sums, energies, [digest.hexdigest() for digest in hashes]


def _rank(energies, top_k, index_name):
    """Stable ties choose the lower token/neuron/head index."""
    order = np.lexsort((np.arange(len(energies)), -energies))[:top_k]
    return [{index_name: int(index), 'delta_l2': math.sqrt(float(energies[index]))}
            for index in order]


def _group_rankings(energies, config, top_k):
    result = {
        'metric': 'L2 of replacement minus original projection weights; biases excluded',
        'embedding_rows': _rank(energies['token_embedding.weight'][:config.vocab_size],
                                top_k, 'token_id'),
        'blocks': [],
    }
    for block in range(config.n_layers):
        prefix = f'blocks.{block}'
        qkv = energies[f'{prefix}.attn.qkv.weight'].reshape(
            3, config.n_heads, config.head_dim).sum(axis=2)
        output = energies[f'{prefix}.attn.output.weight'].reshape(
            config.n_heads, config.head_dim).sum(axis=1)
        result['blocks'].append({
            'block': block,
            'mlp_input_columns': _rank(energies[f'{prefix}.mlp.input.weight'], top_k,
                                        'neuron'),
            'mlp_output_rows': _rank(energies[f'{prefix}.mlp.output.weight'], top_k,
                                      'neuron'),
            'attention_qkv_heads': {
                component: _rank(qkv[index], top_k, 'head')
                for index, component in enumerate(('q', 'k', 'v'))
            },
            'attention_output_heads': _rank(output, top_k, 'head'),
            # Equal head indices refer to corresponding Q/K/V channels and
            # output rows. This combined energy still is not head importance.
            'attention_combined_heads': _rank(qkv.sum(axis=0) + output, top_k, 'head'),
        })
    return result


def compare_checkpoints(original, replacement, *, initial=None,
                        config=GPT2Config(), top_k=20,
                        chunk_elements=DEFAULT_CHUNK_ELEMENTS):
    """Compare complete immutable checkpoints, returning only JSON-safe data.

    Missing/extra/missized weights or any NaN/Inf (even in padding, unranked
    tensors, or the initial checkpoint) fail the entire analysis. Null cosine
    and relative norm mean an undefined zero denominator, not zero similarity.
    Parameter identities are literal coordinates; no neuron alignment is fit.
    The file hashes and before/after stat checks identify inspected bytes, but
    do not authenticate same-init/same-step training or prove corpus causality.
    """
    _positive_integer(top_k, 'top_k')
    _positive_integer(chunk_elements, 'chunk_elements')
    paths = [original, replacement] + ([] if initial is None else [initial])
    checkpoints = [GPT2Checkpoint(path, config) for path in paths]
    before = [_snapshot(checkpoint) for checkpoint in checkpoints]
    metadata = []
    for checkpoint in checkpoints:
        match = re.fullmatch(r'step_([0-9]+)', checkpoint.directory.name)
        metadata.append({'directory': str(checkpoint.directory),
                         'step_from_directory_name': int(match[1]) if match else None,
                         'weight_sha256': {}})
    tensor_reports, totals, energies = [], {}, {}
    for spec in checkpoints[0].manifest:
        sums, group_energy, hashes = _compare_tensor(
            [checkpoint[spec.name] for checkpoint in checkpoints], spec, chunk_elements)
        for key, value in sums.items():
            totals.setdefault(key, []).append(value)
        for item, digest in zip(metadata, hashes):
            item['weight_sha256'][spec.filename] = digest
        tensor_reports.append(dict(spec.to_dict(), **_metrics(sums, initial is not None)))
        if group_energy is not None:
            energies[spec.name] = group_energy
    for checkpoint, snapshot in zip(checkpoints, before):
        # Reconstructing the reader revalidates the complete file set as well
        # as sizes; a new unexpected weight is not hidden by the original map.
        GPT2Checkpoint(checkpoint.directory, config)
        if _snapshot(checkpoint) != snapshot:
            raise ValueError(f'checkpoint changed during comparison: {checkpoint.directory}')
    warnings = [
        'Raw weights contain no training provenance: same initialization, optimizer state, '
        'sample order, steps, and edited corpus must be established separately.',
        'A large coordinate delta is not proof of word storage or causal importance. '
        'Paired retraining includes downstream trajectory changes.',
        'Equal wall-clock training budgets may produce unequal steps; compare matched '
        'steps as well as the requested endpoints.',
    ]
    steps = [item['step_from_directory_name'] for item in metadata[:2]]
    if None not in steps and steps[0] != steps[1]:
        warnings.append(f'UNMATCHED ENDPOINT STEPS: original={steps[0]}, replacement={steps[1]}.')
    return {
        'schema_version': 1,
        'config': asdict(config),
        'delta_definition': 'replacement minus original',
        'norm_scope': 'All unique physical FP32 parameters including vocabulary padding; '
                      'tied embedding and LM head counted once.',
        'zero_denominator_policy': 'Undefined relative norms and cosines are JSON null.',
        'arithmetic': 'Chunked FP64 differences/reductions of little-endian FP32 weights',
        'warnings': warnings,
        'original': metadata[0],
        'replacement': metadata[1],
        'initial': metadata[2] if initial is not None else None,
        'tensor_count': len(tensor_reports),
        'parameter_count': sum(spec.nbytes // 4 for spec in checkpoints[0].manifest),
        'model': _metrics({key: math.fsum(values) for key, values in totals.items()},
                          initial is not None),
        'tensors': tensor_reports,
        'largest_tensor_deltas': sorted(
            [{'name': item['name'], 'delta_l2': item['delta_l2']} for item in tensor_reports],
            key=lambda item: (-item['delta_l2'], item['name']))[:top_k],
        'group_rankings': _group_rankings(energies, config, top_k),
    }


def write_report(output, report):
    """Create a new report, never overwrite an existing file (or checkpoint)."""
    payload = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n'
    with Path(output).open('x', encoding='utf-8') as stream:
        stream.write(payload)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original', type=Path, required=True)
    parser.add_argument('--replacement', type=Path, required=True)
    parser.add_argument('--initial', type=Path)
    parser.add_argument('--output', type=Path, required=True,
                        help='New JSON file outside all input checkpoint directories')
    parser.add_argument('--top-k', type=int, default=20)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    for path in (args.original, args.replacement, args.initial):
        if path is not None and (output == path.resolve() or path.resolve() in output.parents):
            raise ValueError('report must be outside input checkpoint directories')
    report = compare_checkpoints(args.original, args.replacement, initial=args.initial,
                                 top_k=args.top_k)
    write_report(args.output, report)


if __name__ == '__main__':
    main()
