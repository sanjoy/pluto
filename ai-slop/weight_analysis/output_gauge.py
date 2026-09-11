"""Audit destination-ranking dependence on the embedding origin, not extract text.

The fixed shift is c = -mean(E) over the FULL LOGICAL vocabulary (no padding,
no selected-subset mean, and no shift optimization). In ideal arithmetic the
tied GPT-2 change E[t] += c, P[p] -= c leaves every input sum E[t]+P[p]
unchanged. All internal writes are therefore unchanged; every final output
logit gains the same h dot c, leaving softmax probabilities unchanged.
This is NOT a bitwise BF16-preserving transformation.

For a fixed analytical write w, raw destination scores obey
    w dot (E[t]+c) = w dot E[t] + w dot c.
Thus raw within-context destination ranks cannot change in exact arithmetic.
Destination cosine ranks can change because the target norms change. This
audit freezes the offset_paths metadata's S, input geometry and head pairing,
computes its normalized ClosedTrigramProbe writes ONCE, and measures this
output-coordinate sensitivity. It does not generate candidates, change the
frozen protocol, measure corpus matches, or claim the derivative is a full logit.

The raw-input ablation is rejected: it omits P and does not inherit the exact
input-sum compensation. No model forward/backward, tokenizer, or corpus is read.
Run with OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 for bounded CPU parallelism.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import sys
import time

import numpy as np

from .checkpoint import GPT2Checkpoint, GPT2Config, sha256_file
from .closed_trigram import ClosedTrigramProbe, validate_recipe_epsilon
from .trigram import _top, weight_provenance


def _hash_array(values, dtype='<f8'):
    return hashlib.sha256(np.asarray(values, dtype=dtype).tobytes()).hexdigest()


def destination_scores(writes, targets):
    """Pure FP64 raw/cosine scores; zero directions have undefined cosines.

    Zero targets are still valid raw-dot destinations (score zero). Cosine
    scores use -inf exclusively as an exclusion sentinel, never as an output
    number in the JSON report. Inputs are not modified.
    """
    writes, targets = (np.asarray(x, dtype=np.float64) for x in (writes, targets))
    if (writes.ndim != 2 or targets.ndim != 2 or not all(writes.shape) or
            not all(targets.shape) or writes.shape[1] != targets.shape[1]):
        raise ValueError('expected nonempty writes[M,D] and targets[V,D]')
    if not np.isfinite(writes).all() or not np.isfinite(targets).all():
        raise ValueError('writes and targets must be finite')
    raw = writes @ targets.T
    write_norms, target_norms = np.linalg.norm(writes, axis=1), np.linalg.norm(targets, axis=1)
    denominators = write_norms[:, None] * target_norms[None, :]
    if any(not np.isfinite(x).all() for x in (raw, write_norms, target_norms, denominators)):
        raise ValueError('score arithmetic exceeded finite FP64 range')
    cosine = np.full(raw.shape, -np.inf)
    np.divide(raw, denominators, out=cosine, where=denominators > 0)
    return {'raw': raw, 'cosine': cosine,
            'write_norms': write_norms, 'target_norms': target_norms}


def _rank_agreement(before, after, ids, active, top_k, tolerances=None):
    """Deterministic ID-tied rankings plus explicitly labeled numerical cases.

    Ranks use exact computed scores: tolerance never silently changes the
    ranking policy. For raw scores only, disagreements are additionally tested
    for compatibility with the conservative per-row roundoff bound. This does
    not claim that close values are mathematically equal.
    """
    comparable = top1_equal = sets_equal = both_empty = one_empty = 0
    top1_compatible = sets_compatible = 0
    jaccard_sum = 0.0
    for row in np.flatnonzero(active):
        a, b = _top(before[row], top_k, ids), _top(after[row], top_k, ids)
        if not len(a) or not len(b):
            both_empty += int(not len(a) and not len(b))
            one_empty += int(bool(len(a)) != bool(len(b)))
            continue
        comparable += 1
        equal_top = a[0] == b[0]
        sa, sb = set(a.tolist()), set(b.tolist())
        equal_set = sa == sb
        top1_equal += int(equal_top)
        sets_equal += int(equal_set)
        jaccard_sum += len(sa & sb) / len(sa | sb)
        if tolerances is not None:
            tol = 2 * tolerances[row]
            if not equal_top:
                top1_compatible += int(
                    abs(before[row, a[0]] - before[row, b[0]]) <= tol and
                    abs(after[row, a[0]] - after[row, b[0]]) <= tol)
            if not equal_set:
                removed, added = list(sa - sb), list(sb - sa)
                # Raw has identical valid destinations on both sides, so a
                # differing top-k set has nonempty added and removed parts.
                sets_compatible += int(bool(removed and added) and
                    max(before[row, removed]) - min(before[row, added]) <= tol and
                    max(after[row, added]) - min(after[row, removed]) <= tol)
    result = {
        'comparable_nonzero_write_rows': comparable,
        'both_rankings_empty_nonzero_write_rows': both_empty,
        'one_ranking_empty_nonzero_write_rows': one_empty,
        'top1_exact_id_agreement_count': top1_equal,
        'top1_exact_id_agreement_fraction': top1_equal / comparable if comparable else None,
        'top_k': top_k,
        'top_k_exact_set_agreement_count': sets_equal,
        'top_k_exact_set_agreement_fraction': sets_equal / comparable if comparable else None,
        'top_k_mean_jaccard': jaccard_sum / comparable if comparable else None,
    }
    if tolerances is not None:
        result.update(
            top1_disagreements_compatible_with_roundoff=top1_compatible,
            top1_disagreements_beyond_roundoff=comparable - top1_equal - top1_compatible,
            top_k_disagreements_compatible_with_roundoff=sets_compatible,
            top_k_disagreements_beyond_roundoff=comparable - sets_equal - sets_compatible)
    return result


def audit_translation(writes, targets, shift, token_ids=None, top_k=4):
    """Pure comparison of fixed writes against targets and targets + shift.

    The CLI chooses -mean(full logical E); this helper accepts arbitrary shifts
    for algebraic unit tests. Raw arithmetic residuals are checked against a
    conservative FP64 roundoff scale. Large shifts may erase meaningful small
    gaps in floating point: those cases are reported, not called exact ties.
    """
    writes, targets, shift = (np.asarray(x, dtype=np.float64)
                              for x in (writes, targets, shift))
    before = destination_scores(writes, targets)
    if shift.shape != (targets.shape[1],) or not np.isfinite(shift).all():
        raise ValueError('shift must be a finite vector of width D')
    if type(top_k) is not int or top_k <= 0:
        raise ValueError('top_k must be a positive integer')
    ids = np.arange(len(targets)) if token_ids is None else np.asarray(token_ids)
    if (ids.shape != (len(targets),) or not np.issubdtype(ids.dtype, np.integer) or
            len(np.unique(ids)) != len(ids) or np.any(ids < 0)):
        raise ValueError('target IDs must be distinct nonnegative integers')
    with np.errstate(over='ignore', invalid='ignore'):
        shifted = targets + shift
    after = destination_scores(writes, shifted)
    common = writes @ shift
    residual = after['raw'] - before['raw'] - common[:, None]
    # For standard dot-product error gamma_n, sums of absolute products are
    # bounded by ||w||_1 * max|target|. Include both target coordinate systems,
    # translation addition, the common shift dot product, and subtraction.
    # Factor 8 is deliberately conservative, not a score/tie optimization.
    unit_roundoff = np.finfo(np.float64).eps / 2
    n = targets.shape[1] + 2
    gamma_n = n * unit_roundoff / (1 - n * unit_roundoff)
    scale = (np.abs(writes).sum(axis=1) *
             (np.max(np.abs(targets)) + np.max(np.abs(shifted)) + np.max(np.abs(shift))))
    tolerance = 8 * gamma_n * scale + 32 * n * np.nextafter(0., 1.)
    if any(not np.isfinite(x).all() for x in (common, residual, tolerance)):
        raise ValueError('translation audit exceeded finite FP64 range')
    active = before['write_norms'] > 0
    return {
        'write_rows': len(writes), 'target_rows': len(targets), 'width': targets.shape[1],
        'zero_write_rows_excluded_from_rank_agreement': int(np.count_nonzero(~active)),
        'original_zero_target_rows': int(np.count_nonzero(before['target_norms'] == 0)),
        'shifted_zero_target_rows': int(np.count_nonzero(after['target_norms'] == 0)),
        'rank_policy': 'computed score descending, exact score ties by ascending token ID; zero-norm directions excluded for cosine',
        'raw_translation_identity': {
            'equation': 'shifted_raw[row,t] - original_raw[row,t] = write[row] dot shift',
            'max_absolute_residual': float(np.max(np.abs(residual))),
            'max_roundoff_bound': float(np.max(tolerance)),
            'rows_exceeding_roundoff_bound': int(np.count_nonzero(
                np.max(np.abs(residual), axis=1) > tolerance)),
            'roundoff_bound_rule': '8*gamma_(D+2)*||write||_1*(maxabs(targets)+maxabs(shifted_targets)+maxabs(shift)) plus subnormal allowance',
            'interpretation': 'roundoff-compatible is not a declaration of a true mathematical tie',
        },
        'raw_dot': _rank_agreement(before['raw'], after['raw'], ids, active, top_k, tolerance),
        'cosine': _rank_agreement(before['cosine'], after['cosine'], ids, active, top_k),
    }


def probe_from_metadata(checkpoint, metadata):
    """Validate the frozen numerical protocol, then instantiate exactly its S.

    Selection is not recomputed. Labels, candidate text and corpus fields are
    not consulted. Exact checkpoint/source/input hashes detect incompatible
    artifacts rather than quietly switching geometries or head pairing.
    """
    if (metadata.get('stage') != 'corpus_blind_extraction' or
            metadata.get('protocol') != 'fixed final-output-offset vocabulary with normalized first-order closed trigram paths'):
        raise ValueError('expected frozen offset_paths metadata')
    if metadata['checkpoint']['config'] != asdict(checkpoint.config):
        raise ValueError('checkpoint configuration differs from frozen metadata')
    expected_hashes = metadata['checkpoint']['weight_sha256']
    if set(expected_hashes) != {spec.filename for spec in checkpoint}:
        raise ValueError('metadata must hash every checkpoint weight')
    for filename, expected in expected_hashes.items():
        if sha256_file(checkpoint.directory / filename) != expected:
            raise ValueError(f'checkpoint hash mismatch: {filename}')
    for name in ('offset_paths.py', 'closed_trigram.py', 'trigram.py', 'checkpoint.py'):
        if metadata['source_sha256'].get(name) != sha256_file(Path(__file__).with_name(name)):
            raise ValueError(f'probe source hash mismatch: {name}')
    geometry = metadata['geometry_provenance']
    if (metadata['parameters']['input_geometry'] != 'normalized' or
            geometry['input_geometry'] != 'normalized' or
            geometry['previous_position'] != 0 or geometry['current_position'] != 1):
        raise ValueError('gauge proof requires normalized E+P input geometry at positions 0/1')
    epsilon = validate_recipe_epsilon()
    if geometry['epsilon'] != epsilon['fp64_promoted_value']:
        raise ValueError('frozen LayerNorm epsilon differs from current recipe')
    selection = metadata['vocabulary_selection']
    ids = np.asarray(selection['ids_ascending'])
    if (ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer) or
            _hash_array(ids, '<u4') != selection['ids_little_endian_uint32_sha256']):
        raise ValueError('frozen vocabulary hash or ID type is invalid')
    routing = metadata['routing_weight_provenance']
    pairing = [row['content_head'] for row in routing['head_slices']]
    expected_routing = weight_provenance(checkpoint, 0, pairing)
    if routing['block'] != 0 or routing['head_slices'] != expected_routing['head_slices']:
        raise ValueError('routing slices do not match the declared head pairing')
    expected_pairing = (np.arange(checkpoint.config.n_heads) +
                        int(metadata['parameters']['broken_routing_control'])) % checkpoint.config.n_heads
    if pairing != expected_pairing.tolist():
        raise ValueError('head pairing disagrees with frozen control flag')
    probe = ClosedTrigramProbe(checkpoint, ids, 'normalized', pairing)
    for name, array in [('X0', probe.inputs0), ('X1', probe.inputs1)]:
        if _hash_array(array) != metadata['numerical_geometry_hashes'][name + '_little_endian_float64_sha256']:
            raise ValueError(f'{name} coordinate hash differs from frozen probe')
    return probe, epsilon


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--candidates-metadata', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args_list = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(args_list)
    if os.path.lexists(args.output):
        raise FileExistsError(f'refusing to overwrite {args.output}')
    started = time.perf_counter()
    metadata = json.loads(args.candidates_metadata.read_text())
    checkpoint = GPT2Checkpoint(args.checkpoint, GPT2Config(**metadata['checkpoint']['config']),
                                 check_finite=True)
    probe, epsilon = probe_from_metadata(checkpoint, metadata)
    # Each interaction write is evaluated exactly once and shared by both
    # destination comparisons. S and normalized input coordinates stay frozen.
    writes = np.concatenate([probe.interaction_writes(current)
                             for current in range(len(probe.vocabulary_ids))])
    full_mean = np.mean(checkpoint.token_embedding, axis=0, dtype=np.float64)
    shift = -full_mean
    result = audit_translation(writes, probe.embedding, shift, probe.vocabulary_ids)
    result.update({
        'schema_version': 1, 'stage': 'weight_only_gauge_audit_not_extraction',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'command': shlex.join([sys.executable, '-m', 'weight_analysis.output_gauge', *args_list]),
        'elapsed_seconds': time.perf_counter() - started,
        'checkpoint': {'directory': str(checkpoint.directory), 'config': asdict(checkpoint.config),
                       'weight_sha256': metadata['checkpoint']['weight_sha256']},
        'frozen_metadata': {'path': str(args.candidates_metadata.resolve()),
                            'sha256': sha256_file(args.candidates_metadata),
                            'candidate_sha256_reference_only': metadata.get('candidate_sha256'),
                            'candidate_file_read': False},
        'vocabulary_ids_ascending': probe.vocabulary_ids.tolist(),
        'vocabulary_ids_sha256': _hash_array(probe.vocabulary_ids, '<u4'),
        'geometry': 'frozen normalized first LayerNorm dictionary at positions 0/1',
        'ov_pairing': [row['content_head'] for row in metadata['routing_weight_provenance']['head_slices']],
        'epsilon_validation': epsilon,
        'shift': {'rule': '-mean(E) over every logical vocabulary row; padding excluded',
                  'logical_rows': checkpoint.config.vocab_size, 'optimized': False,
                  'norm': float(np.linalg.norm(shift)), 'float64_sha256': _hash_array(shift)},
        'fixed_writes_float64_sha256': _hash_array(writes),
        'write_order': 'current S index outer, previous S index inner',
        'source_sha256': {name: sha256_file(Path(__file__).with_name(name))
                          for name in ('output_gauge.py', 'closed_trigram.py', 'trigram.py', 'checkpoint.py')},
        'limitations': [
            'Ideal arithmetic gauge identity; BF16/FP32 rounding equivalence is not claimed.',
            'Source selection and write vectors stay frozen; this is not a new extraction protocol.',
            'Raw destination ranking invariance does not make derivative scores full-model logits.',
            'Cosine changes reveal origin dependence, not which ranking recovers training text.',
            'Zero writes are excluded from agreement denominators; zero target directions are excluded only for cosine.',
            'No corpus access, token labels, candidate generation, model inference or training.',
        ],
    })
    encoded = json.dumps(result, indent=2, allow_nan=False) + '\n'
    with args.output.open('x', encoding='utf-8') as stream:
        stream.write(encoded)
    print(json.dumps({'output': str(args.output), 'raw_dot': result['raw_dot'],
                      'cosine': result['cosine']}, allow_nan=False))


if __name__ == '__main__':
    main()
