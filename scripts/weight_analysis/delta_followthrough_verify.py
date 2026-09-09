"""Verify frozen later-batch token candidates at prespecified replay phases.

The candidates precede corpus access. This tests temporal localization of an
unordered bag, not reconstruction of token order or historical sampler state.
"""

import argparse
import datetime
import math
from pathlib import Path
import subprocess

import numpy as np

from . import delta_followthrough as extraction
from . import restart_delta as delta
from . import restart_delta_verify as earlier
from .late_mlp_paths import file_record, write_exclusive


VOCABULARY = 50257
DRAW_COUNT = 1200
SEEDS = (17, *range(10000, 11000))


def read_finite_json(path):
    """Reject duplicate keys, nonfinite literals, and overflow such as 1e999."""
    result = earlier.cv.read_json(path)
    pending = [result]
    while pending:
        value = pending.pop()
        if isinstance(value, dict):
            pending.extend(value.values())
        elif isinstance(value, list):
            pending.extend(value)
        elif isinstance(value, float) and not math.isfinite(value):
            raise ValueError('nonfinite JSON number')
    return result


def phase_slices(batch):
    """Ten training steps contain 10*batch independent sequence draws."""
    if type(batch) is not int or batch not in (1, 10):
        raise ValueError('only the two prescribed batch-size conditions are allowed')
    return {'predicted': (100 * batch, 110 * batch),
            'previous': (90 * batch, 100 * batch),
            'next': (110 * batch, 120 * batch), 'reset': (0, 10 * batch)}


def authenticate(frozen_path):
    frozen_path = Path(frozen_path).resolve()
    identity = file_record(frozen_path)
    frozen = read_finite_json(frozen_path)
    if (frozen.get('complete') is not True or frozen.get('schema_version') != 1
            or frozen.get('stage') != 'delta_followthrough_candidate_freeze_not_sequence_recovery'):
        raise ValueError('incomplete follow-through candidate freeze')
    source_dir = Path(__file__).resolve().parent
    expected_sources = {'delta_followthrough.py', 'restart_delta.py',
                        'checkpoint_archive.py', 'checkpoint.py', 'late_mlp_paths.py',
                        'protocol', 'history_inventory'}
    if set(frozen['sources']) != expected_sources:
        raise ValueError('unexpected frozen source set')
    for name, record in frozen['sources'].items():
        if name.endswith('.py') and Path(record['path']) != source_dir / name:
            raise ValueError('frozen source path substituted')
        earlier.check_record(record)
    expected_files = {'plan': 'plan.json', 'candidates': 'candidates.json',
                      'complete_scores': 'scores.npz'}
    for boundary in delta.BOUNDARIES:
        for step in extraction.checkpoint_steps(boundary)[:-1]:
            key = f'interval_{step}_{step + 10}'
            expected_files[key] = key + '.npz'
    if set(frozen['files']) != set(expected_files):
        raise ValueError('missing or extra frozen output files')
    for name, record in frozen['files'].items():
        if Path(record['path']) != frozen_path.parent / expected_files[name]:
            raise ValueError('frozen output path substituted')
        earlier.check_record(record)
    plan = read_finite_json(frozen_path.parent / 'plan.json')
    if (plan.get('stage') != 'delta_followthrough_plan_no_corpus_or_model'
            or plan.get('checkpoint_offsets') != [90, 100, 110, 120]
            or plan.get('center_interval_offsets') != [100, 110]
            or plan.get('conditional_prediction_one_based_windows_at_sequence_batch_size_one') != [101, 110]
            or plan['sources'] != frozen['sources'] or plan['boundaries'] != list(delta.BOUNDARIES)
            or plan['logical_vocab_size'] != VOCABULARY or plan['top_k'] != 128
            or plan['seed'] != delta.SEED):
        raise ValueError('plan settings differ from prescribed experiment')
    candidates = read_finite_json(frozen_path.parent / 'candidates.json')
    if (candidates['stage'] != 'frozen_unordered_followthrough_token_candidates'
            or candidates['top_k'] != 128
            or candidates.get('no_corpus_or_tokenizer_access') is not True
            or candidates.get('no_model_execution') is not True
            or set(candidates['candidate_lists']) != extraction.expected_names()):
        raise ValueError('candidate list set differs')
    dates = [datetime.datetime.fromisoformat(value) for value in
             (plan['created_utc_before_weight_reads'], candidates['created_utc'],
              frozen['frozen_utc_before_corpus_verification'])]
    if any(value.tzinfo is None for value in dates) or not dates[0] <= dates[1] <= dates[2]:
        raise ValueError('freeze timestamps must be timezone-aware and ordered')
    with np.load(frozen_path.parent / 'scores.npz', allow_pickle=False) as arrays:
        if set(arrays.files) != extraction.expected_names() | {'identity_permutation'}:
            raise ValueError('complete score array keys differ')
        permutation = np.random.Generator(np.random.PCG64(delta.SEED)).permutation(VOCABULARY)
        if (arrays['identity_permutation'].dtype.kind not in 'iu'
                or not np.array_equal(arrays['identity_permutation'], permutation)):
            raise ValueError('identity permutation differs')
        for name, ranking in candidates['candidate_lists'].items():
            scores = arrays[name]
            if scores.shape != (VOCABULARY,) or scores.dtype.kind != 'f' or not np.isfinite(scores).all():
                raise ValueError('invalid full scores')
            ids = np.lexsort((np.arange(VOCABULARY), -scores))[:128]
            if (ids.tolist() != ranking['token_ids'] or scores[ids].tolist() != ranking['scores']
                    or permutation[ids].tolist() != ranking['identity_shuffled_token_ids']):
                raise ValueError('candidate ranking differs from stored scores')
            if name.startswith('shared_'):
                variant = name.rsplit('_', 1)[1]
                components = np.array([arrays[f'followthrough_{variant}_{n}'][ids] for n in delta.BOUNDARIES])
                if (components.tolist() != ranking['component_scores']
                        or (components > 0).sum(axis=0).tolist() != ranking['positive_component_count']):
                    raise ValueError('shared candidate components differ')
    for record in [identity, *frozen['sources'].values(), *frozen['files'].values()]:
        earlier.check_record(record)
    return frozen, candidates, identity


def sampler_chunk(sampler, total, first, count):
    """Cap each subprocess below the standalone sampler's one-million draws."""
    if type(count) is not int or not 0 < count <= 500:
        raise ValueError('sampler chunk must contain 1..500 seeds')
    command = [str(sampler), str(total), '1024', str(DRAW_COUNT), str(first), str(count)]
    result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=30)
    lines = result.stdout.splitlines()
    if result.stderr or len(lines) != count or len(result.stdout) > count * (DRAW_COUNT + 1) * 22:
        raise ValueError('unexpected sampler output')
    rows = []
    for index, line in enumerate(lines):
        fields = line.split()
        if (len(fields) != DRAW_COUNT + 1
                or any(not word.isascii() or not word.isdecimal() for word in fields)):
            raise ValueError('malformed sampler row')
        values = list(map(int, fields))
        if values[0] != first + index or any(not 0 <= v < total - 1024 for v in values[1:]):
            raise ValueError('sampler seed or window range differs')
        rows.append(values[1:])
    return np.asarray(rows, dtype=np.int64)


def phase_membership(tokens, starts, batch):
    """All seeds at predicted phase; seed17 alone at the three time controls."""
    result = {}
    for name, (begin, end) in phase_slices(batch).items():
        selected = starts[:, begin:end] if name == 'predicted' else starts[:1, begin:end]
        result[name] = earlier.window_presence(tokens, selected, 1024, end - begin, VOCABULARY)
    return result


def verify(frozen_path, corpus_path, tokens_path, offsets_path, tokenizer_dir, sampler, output):
    output = Path(output).absolute()
    start_path = output.with_name(output.stem + '_start.json')
    starts_path = output.with_name(output.stem + '_starts.npz')
    for path in (output, start_path, starts_path):
        if path.exists() or path.is_symlink():
            raise FileExistsError(path)
    frozen, candidates, frozen_identity = authenticate(frozen_path)
    source_dir = Path(__file__).resolve().parent
    sources = {name: file_record(source_dir / name) for name in
               ('delta_followthrough_verify.py', 'restart_delta_verify.py', 'verify.py',
                'vocabulary_verify.py', 'causal_validation.py', 'replay_sampler.cc')}
    sampler = Path(sampler).resolve()
    sources['sampler_binary'] = file_record(sampler)
    paths = {'corpus': str(Path(corpus_path).resolve()), 'native_tokens': str(Path(tokens_path).resolve()),
             'native_offsets': str(Path(offsets_path).resolve()),
             'tokenizer': str((Path(tokenizer_dir) / 'tokenizer.json').resolve())}
    write_exclusive(start_path, {
        'stage': 'followthrough_verification_start_before_corpus_access',
        'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'frozen': frozen_identity, 'verification_sources': sources,
        'input_paths_not_yet_opened': paths,
        'phases_by_batch': {str(b): phase_slices(b) for b in (1, 10)},
        'conditional_seed': 17, 'alternative_seeds': [10000, 10999]})
    start_identity = file_record(start_path)
    inputs = {key: file_record(path) for key, path in paths.items()}
    corpus, tokens, offsets, vocabulary, boundary, split = earlier.load_native(
        paths['corpus'], paths['native_tokens'], paths['native_offsets'], paths['tokenizer'])
    implementation = subprocess.run([str(sampler), '--implementation'], capture_output=True,
                                    text=True, check=True, timeout=30)
    if implementation.stderr:
        raise ValueError('sampler implementation query failed')
    starts = np.concatenate([sampler_chunk(sampler, split, first, count) for first, count in
                             ((17, 1), (10000, 500), (10500, 500))])
    starts_identity = delta.write_npz(starts_path, seeds=np.array(SEEDS, dtype=np.int64), starts=starts)
    frequency = earlier.frequencies(tokens, split, VOCABULARY)
    masks = {str(b): phase_membership(tokens[:split], starts, b) for b in (1, 10)}
    results = {}
    for name, ranking in candidates['candidate_lists'].items():
        results[name] = {}
        for label, field in (('real', 'token_ids'), ('identity_shuffled', 'identity_shuffled_token_ids')):
            ids = ranking[field]
            results[name][label] = {
                'token_ids': ids,
                'pieces': [{'token_id': t, 'raw_hex': vocabulary[t].hex(),
                            'escaped': repr(vocabulary[t].decode('utf-8', errors='backslashreplace'))} for t in ids],
                'corpus_frequency': {key: earlier.summarize_frequencies(values, ids) for key, values in frequency.items()},
                'batch_conditions': {batch: {
                    'predicted': earlier.summarize_overlap(phases['predicted'], ids),
                    'seed17_phase_overlap': {key: int(mask[0, ids].sum()) for key, mask in phases.items()},
                    'seed17_membership_in_rank_order': {key: mask[0, ids].tolist() for key, mask in phases.items()}}
                    for batch, phases in masks.items()}}
    after, _, identity_after = authenticate(frozen_path)
    if after != frozen or identity_after != frozen_identity:
        raise ValueError('frozen extraction changed during verification')
    for record in [*inputs.values(), *sources.values(), start_identity, starts_identity]:
        earlier.check_record(record)
    result = {'schema_version': 1, 'complete': True,
              'stage': 'prospective_later_batch_bag_verification_not_sequence_recovery',
              'completed_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'frozen': frozen_identity, 'verification_start': start_identity,
              'sources': sources, 'inputs': inputs, 'replay_starts': starts_identity,
              'native_export_every_byte_validated': True,
              'split': {'byte_boundary': boundary, 'prefix_tokens': split, 'total_tokens': len(tokens)},
              'sampler_implementation': implementation.stdout.strip(),
              'phases_by_batch': {str(b): phase_slices(b) for b in (1, 10)},
              'distinct_tokens_by_batch_and_phase': {batch: {phase: mask.sum(axis=1).tolist()
                                                             for phase, mask in phases.items()}
                                                    for batch, phases in masks.items()},
              'results': results, 'all_sources_inputs_and_frozen_artifacts_unchanged': True,
              'limitations': ['Conditional replay is not authenticated historical sampler state.',
                              'Unordered token pieces, no text sequence or multiplicity recovery.',
                              'Protocol chosen after earlier boundary outcome, before these new weight reads.',
                              'Alternative-seed comparisons are descriptive, not p-values.']}
    write_exclusive(output, result)
    print(f'Verified {len(results)} later-batch rankings and identity controls.', flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('frozen', 'corpus', 'native-tokens', 'native-offsets', 'tokenizer-dir', 'sampler', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args(argv)
    verify(args.frozen, args.corpus, args.native_tokens, args.native_offsets,
           args.tokenizer_dir, args.sampler, args.output)


if __name__ == '__main__':
    main()
