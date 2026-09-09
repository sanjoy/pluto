"""Scalar follow-through replay verification, never a text extractor.

Only runs after a completed report, pinned candidate freeze, and every reported
input/artifact identity have passed. Uses the independently implemented Python
MT19937-64 oracle, set unions, Counter and statistics, not verifier helpers,
NumPy membership masks, or a C++ sampler subprocess. NumPy only reads NPZ files.
"""

from collections import Counter
import datetime
import hashlib
import json
import math
from pathlib import Path
import statistics
import struct
import sys
import time

import numpy as np

sys.path.insert(0, '/home/ubuntu/code/pluto')
from scripts.weight_analysis.replay_sampler_test import libstdcxx13_starts


RUN = Path('/tmp/pluto-delta-followthrough.VjsYYn/run')
FROZEN_SHA256 = 'b42aba12eb4cb5072a58cca6daaf94aab4fd9539627d31875f02ce60b17797ed'
CANDIDATES_SHA256 = '999a9606c936b3ea7132a0321d420d176d9bd1ae45713b90a9fbc051fa039a83'
SEEDS = [17, *range(10000, 11000)]
VOCABULARY = 50257
CONTEXT = 1024
PHASES = {
    '1': {'predicted': [100, 110], 'previous': [90, 100], 'next': [110, 120], 'reset': [0, 10]},
    '10': {'predicted': [1000, 1100], 'previous': [900, 1000], 'next': [1100, 1200], 'reset': [0, 100]},
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_record(path):
    path = Path(path).resolve()
    info = path.stat()
    checksum = hashlib.sha256()
    with path.open('rb') as handle:
        while block := handle.read(8 * 1024 * 1024):
            checksum.update(block)
    after = path.stat()
    require((info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns) ==
            (after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns),
            'file changed while hashing: ' + str(path))
    return {'path': str(path), 'bytes': info.st_size, 'sha256': checksum.hexdigest()}


def read_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'duplicate JSON key')
            result[key] = value
        return result

    def finite(value):
        if isinstance(value, float):
            require(math.isfinite(value), 'nonfinite JSON number')
        elif isinstance(value, dict):
            for item in value.values():
                finite(item)
        elif isinstance(value, list):
            for item in value:
                finite(item)
    value = json.loads(Path(path).read_text(), object_pairs_hook=pairs)
    finite(value)
    return value


def check_record(record):
    require(set(record) == {'path', 'bytes', 'sha256'}, 'malformed input identity record')
    require(file_record(record['path']) == record, 'identity mismatch: ' + record['path'])


def token_bag(tokens, starts, begin, end, context=CONTEXT):
    require(type(context) is int and 0 < context < len(tokens), 'invalid context')
    require(type(begin) is int and type(end) is int and 0 <= begin < end <= len(starts),
            'invalid draw slice')
    result = set()
    for start in starts[begin:end]:
        require(type(start) is int and 0 <= start < len(tokens) - context, 'invalid window start')
        # End is exclusive in Python; the +1 includes the last shifted target.
        result.update(tokens[start:start + context + 1])
    return result


def overlap_summary(overlaps):
    actual, alternatives = overlaps[0], overlaps[1:]
    require(len(alternatives) == 1000, 'need every fixed alternative seed')
    greater = sum(value > actual for value in alternatives)
    equal = sum(value == actual for value in alternatives)
    return {
        'seed17_overlap': actual, 'seed17_overlap_fraction': actual / 128,
        'alternative_overlap_counts_in_seed_order': alternatives,
        'alternative_minimum': min(alternatives), 'alternative_maximum': max(alternatives),
        'alternative_mean': statistics.mean(alternatives),
        'alternative_median': statistics.median(alternatives),
        'alternative_histogram': {str(key): value for key, value in Counter(alternatives).items()},
        'alternatives_strictly_greater_than_seed17': greater,
        'alternatives_equal_to_seed17': equal,
        'alternatives_greater_or_equal_to_seed17': greater + equal,
        'seed17_descriptive_rank_among_all_seeds': 1 + greater,
    }


def frequency_summary(counts, ids, total):
    selected = [counts[token] for token in ids]
    present = sum(value > 0 for value in selected)
    return {
        'ranked_token_ids_present': present, 'ranked_token_ids_absent': 128 - present,
        'presence_fraction': present / 128, 'counts_in_rank_order': selected,
        'selected_token_occurrences': sum(selected),
        'selected_share_of_corpus_tokens': sum(selected) / total,
        'minimum_frequency': min(selected), 'maximum_frequency': max(selected),
        'mean_frequency': statistics.mean(selected), 'median_frequency': statistics.median(selected),
    }


def self_test():
    require(token_bag([1, 2, 3, 4, 5], [0, 0, 2], 0, 1, 2) == {1, 2, 3}, 'first endpoint')
    require(token_bag([1, 2, 3, 4, 5], [0, 0, 2], 2, 3, 2) == {3, 4, 5}, 'last target')
    require(token_bag([1, 2, 3, 4, 5], [0, 0, 2], 0, 3, 2) == {1, 2, 3, 4, 5}, 'duplicates')
    for begin, end in [(-1, 2), (1, 1), (0, 4)]:
        try:
            token_bag([1, 2, 3, 4, 5], [0, 0, 2], begin, end, 2)
        except ValueError:
            pass
        else:
            raise ValueError('invalid slice accepted')
    overlaps = overlap_summary([1, *([0, 1] * 500)])
    require(overlaps['alternative_mean'] == .5 and overlaps['alternatives_equal_to_seed17'] == 500
            and overlaps['seed17_descriptive_rank_among_all_seeds'] == 1, 'summary fixture')


def main():
    started = time.monotonic()
    source_records = {'auditor': file_record(__file__),
                      'scalar_sampler_oracle': file_record('/home/ubuntu/code/pluto/scripts/weight_analysis/replay_sampler_test.py')}
    output = RUN / 'independent_replay_check.json'
    require(not output.exists() and not output.is_symlink(), 'output must be new')
    report_path = RUN / 'verification.json'
    # A missing/incomplete report fails before native inputs can be opened.
    report_record = file_record(report_path)
    report = read_json(report_path)
    require(report.get('complete') is True and report.get('native_export_every_byte_validated') is True
            and report.get('all_sources_inputs_and_frozen_artifacts_unchanged') is True
            and report.get('stage') == 'prospective_later_batch_bag_verification_not_sequence_recovery',
            'need completed primary verification')
    require(report['frozen']['sha256'] == FROZEN_SHA256, 'wrong pinned freeze')
    check_record(report['frozen'])
    frozen = read_json(report['frozen']['path'])
    require(frozen.get('complete') is True, 'incomplete candidate freeze')
    require(frozen['files']['candidates']['sha256'] == CANDIDATES_SHA256, 'wrong pinned candidates')
    records = [*source_records.values(), report_record, report['frozen'], report['verification_start'], report['replay_starts'],
               *frozen['files'].values(), *frozen['sources'].values(), *report['sources'].values()]
    for record in records:
        check_record(record)
    candidates = read_json(frozen['files']['candidates']['path'])['candidate_lists']
    expected_names = {f'followthrough_{variant}_{boundary}' for variant in ['raw', 'adjusted']
                      for boundary in [580, 1780, 4280, 7230, 8160]} | {
                          'shared_followthrough_raw', 'shared_followthrough_adjusted'}
    require(set(candidates) == set(report['results']) == expected_names, 'missing/extra candidate lists')
    require(report['phases_by_batch'] == PHASES, 'phase boundaries differ')
    start = read_json(report['verification_start']['path'])
    times = [datetime.datetime.fromisoformat(value) for value in [
        frozen['frozen_utc_before_corpus_verification'], start['created_utc'], report['completed_utc']]]
    require(all(value.tzinfo is not None for value in times) and times[0] <= times[1] <= times[2],
            'verification chronology is not ordered and timezone-aware')
    require(start['frozen'] == report['frozen'] and start['phases_by_batch'] == PHASES,
            'verification start does not match frozen experiment')
    with np.load(report['replay_starts']['path'], allow_pickle=False) as source:
        require(set(source.files) == {'seeds', 'starts'}, 'unexpected start archive keys')
        seeds, saved_starts = source['seeds'], source['starts']
        require(seeds.dtype == np.dtype('<i8') and saved_starts.dtype == np.dtype('<i8'),
                'saved seed/start dtype differs')
        require(seeds.tolist() == SEEDS and saved_starts.shape == (1001, 1200), 'saved start shape/seed labels differ')
        saved_starts = saved_starts.tolist()

    # Only after completed verification, pinned freeze, and replay-file checks
    # are corpus-input hashes checked. Token bytes are parsed only after every
    # input hash has matched. No corpus text is decoded or retokenized.
    require(set(report['inputs']) == {'corpus', 'native_tokens', 'native_offsets', 'tokenizer'},
            'unexpected corpus input set')
    for record in report['inputs'].values():
        check_record(record)
    records.extend(report['inputs'].values())
    payload = Path(report['inputs']['native_tokens']['path']).read_bytes()
    require(len(payload) % 4 == 0, 'partial native uint32')
    tokens = [token for (token,) in struct.iter_unpack('<I', payload)]
    require(len(tokens) == report['split']['total_tokens'] and all(token < VOCABULARY for token in tokens),
            'native token export invalid')
    prefix_count = report['split']['prefix_tokens']
    require(type(prefix_count) is int and CONTEXT < prefix_count < len(tokens), 'invalid prefix length')
    prefix = tokens[:prefix_count]
    counts = {'full': Counter(tokens), 'current_prefix': Counter(prefix), 'current_suffix': Counter(tokens[prefix_count:])}
    totals = {'full': len(tokens), 'current_prefix': prefix_count, 'current_suffix': len(tokens) - prefix_count}
    print('Completed report, pinned freeze, starts and all input hashes validated; scalar replay begins.', flush=True)

    starts = []
    for row, seed in enumerate(SEEDS):
        actual = libstdcxx13_starts(prefix_count, CONTEXT, 1200, seed)
        require(actual == saved_starts[row], f'independent sampler row differs for seed {seed}')
        starts.append(actual)
    print('All 1,201,200 sampled starts match the independent Python oracle.', flush=True)

    frequency_checks = []
    for name, variants in report['results'].items():
        require(set(variants) == {'real', 'identity_shuffled'}, 'missing identity-control variant')
        for label, result in variants.items():
            ids = result['token_ids']
            field = 'token_ids' if label == 'real' else 'identity_shuffled_token_ids'
            require(ids == candidates[name][field] and len(ids) == len(set(ids)) == 128
                    and all(type(token) is int and 0 <= token < VOCABULARY for token in ids),
                    'candidate IDs differ from pinned freeze')
            for corpus_label, counter in counts.items():
                expected = frequency_summary(counter, ids, totals[corpus_label])
                claimed = result['corpus_frequency'][corpus_label]
                require(all(claimed[key] == value for key, value in expected.items()),
                        f'frequency mismatch: {name}/{label}/{corpus_label}')
            frequency_checks.append({'name': name, 'variant': label, 'all_three_frequency_summaries_exact': True})

    checks = []
    for batch, phases in PHASES.items():
        begin, end = phases['predicted']
        bags = [token_bag(prefix, row, begin, end) for row in starts]
        controls = {phase: token_bag(prefix, starts[0], first, last)
                    for phase, (first, last) in phases.items() if phase != 'predicted'}
        seed17_bags = {'predicted': bags[0], **controls}
        claimed_distinct = report['distinct_tokens_by_batch_and_phase'][batch]
        require(set(claimed_distinct) == set(PHASES[batch]), 'distinct phase labels differ')
        require(claimed_distinct['predicted'] == list(map(len, bags)), 'predicted bag sizes differ')
        for phase, token_set in controls.items():
            require(claimed_distinct[phase] == [len(token_set)], 'seed17 control bag size differs')
        for name, variants in report['results'].items():
            for label, result in variants.items():
                ids = result['token_ids']
                id_set = set(ids)
                expected = overlap_summary([len(id_set & token_set) for token_set in bags])
                claimed = result['batch_conditions'][batch]
                require(all(claimed['predicted'][key] == value for key, value in expected.items()),
                        f'predicted overlap mismatch: {name}/{label}/batch{batch}')
                phase_counts = {phase: len(id_set & token_set) for phase, token_set in seed17_bags.items()}
                membership = {phase: [token in token_set for token in ids] for phase, token_set in seed17_bags.items()}
                require(claimed['seed17_phase_overlap'] == phase_counts, 'phase overlap counts differ')
                require(claimed['seed17_membership_in_rank_order'] == membership, 'per-ID phase membership differs')
                checks.append({'name': name, 'variant': label, 'sequence_batch_size': int(batch),
                               'predicted_overlap': expected, 'seed17_phase_overlap': phase_counts,
                               'seed17_membership_in_rank_order': membership})
        print('Scalar bags and all overlaps/membership checked for batch', batch, flush=True)
    require(len(checks) == 48 and len(frequency_checks) == 24, 'audit coverage differs')
    for record in records:
        check_record(record)
    result = {
        'schema': 'followthrough_independent_scalar_replay_audit_v1', 'complete': True,
        'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'report': report_record, 'checked_input_records': records, 'sources': source_records,
        'independent_sampler_rows_exact': 1001, 'draws_per_row': 1200,
        'total_draws_exact': 1201200, 'predicted_overlap_vectors_exact': 48,
        'alternative_overlap_counts_exact': 48000,
        'seed17_phase_overlap_counts_exact': 192, 'per_id_phase_membership_bits_exact': 24576,
        'candidate_frequency_variants_checked': 24, 'corpus_frequency_summaries_exact': 72,
        'all_reported_frequency_summary_fields_checked': True,
        'all_inputs_and_frozen_artifacts_unchanged': True,
        'phase_slices': PHASES, 'checks': checks, 'frequency_checks': frequency_checks,
        'elapsed_seconds': time.monotonic() - started,
        'limitations': ['Audits conditional replay, not authenticated historical training state.',
                        'No word ordering or passage extraction.',
                        'Uses the independently tested Python sampler oracle, not a C++ subprocess.',
                        'Native byte-to-token/tokenizer validation remains the prior verifier scope; this audit checks their exact input hashes and native-ID counts.',
                        'No model forward/backward, optimizer, GPU work, checkpoint reads, or candidate changes.'],
    }
    with output.open('x') as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')
    print(json.dumps({'output': file_record(output), 'complete': True,
                      'draws_exact': 1201200, 'predicted_vectors_exact': 48,
                      'elapsed_seconds': result['elapsed_seconds']}, indent=2))


if __name__ == '__main__':
    self_test()
    if sys.argv[1:] == ['--self-test']:
        print('Scalar audit self-tests pass; no report/corpus opened.')
    else:
        require(not sys.argv[1:], 'unexpected arguments')
        main()
