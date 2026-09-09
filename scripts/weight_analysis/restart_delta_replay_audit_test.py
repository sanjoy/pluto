"""Independent synthetic checks of replay-audit coverage and tamper rejection.

No corpus, tokenizer, checkpoints, CUDA, or C++ subprocess is used. The report
fixture has all 42 labels and both identity variants, but only two possible
synthetic windows. Their bags are known analytically ({0} and {1000}), so the
expected counts are not produced by the verifier or by the auditor under test.
"""

import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest import mock

from . import restart_delta_replay_audit as auditor


BOUNDARIES = [580, 1780, 4280, 7230, 8160]


def expected_labels():
    names = [f'endpoint_norm_{step}' for step in BOUNDARIES]
    for variant in ['raw', 'adjusted']:
        for arm in ['boundary', 'ordinary_before', 'ordinary_after']:
            names.extend(f'{arm}_{variant}_{step}' for step in BOUNDARIES)
            names.append(f'shared_{arm}_{variant}')
    return names + ['shared_endpoint_norm']


def fixture_identity(path):
    data = path.read_bytes()
    return {'path': str(path.resolve()), 'bytes': len(data),
            'sha256': hashlib.sha256(data).hexdigest()}


class BagTest(unittest.TestCase):
    def test_first_input_and_last_shifted_target_included(self):
        tokens = [7, 7, 9, 11, 9, 13]
        self.assertEqual(auditor.bag(tokens, [0], 2), {7, 9})
        # Last valid start is len(tokens)-context-1, not one less.
        self.assertEqual(auditor.bag(tokens, [3], 2), {11, 9, 13})
        self.assertEqual(auditor.bag([1, 2, 3], [0], 2), {1, 2, 3})

    def test_duplicates_overlapping_windows_and_input_immutability(self):
        tokens = [7, 7, 9, 11, 9, 13]
        starts = [0, 0, 1, 3, 3]
        before_tokens, before_starts = tokens.copy(), starts.copy()
        self.assertEqual(auditor.bag(tokens, starts, 2), {7, 9, 11, 13})
        self.assertEqual(tokens, before_tokens)
        self.assertEqual(starts, before_starts)

    def test_negative_past_end_or_too_short_windows_rejected(self):
        for tokens, starts, context in [([0, 1, 2], [-1], 1),
                                        ([0, 1, 2], [2], 1),
                                        ([0, 1], [0], 2), ([], [0], 1)]:
            with self.subTest(tokens=tokens, starts=starts, context=context):
                with self.assertRaises(ValueError):
                    auditor.bag(tokens, starts, context)

    def test_context_and_start_types_fail_closed(self):
        for context in [0, -1, True, 1.0, '1', 3]:
            with self.subTest(context=context), self.assertRaises(ValueError):
                auditor.bag([0, 1, 2], [0], context)
        for start in [False, 0.0, '0', None]:
            with self.subTest(start=start), self.assertRaises(ValueError):
                auditor.bag([0, 1, 2], [start], 1)


class ReplayAuditFixtureTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.report_path = self.root / 'report.json'
        self.output = self.root / 'audit.json'
        self.tokens_path = self.root / 'native_tokens.bin'
        self.tokens = [0] * 1025 + [1000] * 1025 + [2] * 7
        self.tokens_path.write_bytes(struct.pack('<' + 'I' * len(self.tokens), *self.tokens))
        self.inputs = {'native_tokens': fixture_identity(self.tokens_path)}
        for name in ['corpus', 'native_offsets', 'tokenizer']:
            path = self.root / name
            path.write_bytes(b'not interpreted by the scalar replay auditor\n')
            self.inputs[name] = fixture_identity(path)
        self.rows = [[0] * 100] + [[0 if seed % 2 == 0 else 1025] * 100
                                  for seed in range(10000, 11000)]
        self.real = list(reversed(range(128)))
        self.shuffled = [1000, *range(2000, 2127)]
        self.report = self.fixture_report()

    def fixture_report(self):
        frequency = {
            'full': {0: 1025, 1000: 1025, 2: 7},
            'current_prefix': {0: 1025, 1000: 1025},
            'current_suffix': {2: 7},
        }
        variants = {}
        for label, ids in [('real', self.real), ('identity_shuffled', self.shuffled)]:
            # The synthetic windows consist entirely of one token each.
            actual = int((0 if label == 'real' else 1000) == 0)
            alternatives = [int((seed % 2 == 0) == (label == 'real'))
                            for seed in range(10000, 11000)]
            greater = sum(value > actual for value in alternatives)
            equal = alternatives.count(actual)
            claim = {
                'seed17_overlap': actual,
                'seed17_overlap_fraction': actual / 128,
                'alternative_overlap_counts_in_seed_order': alternatives,
                'alternative_minimum': 0, 'alternative_maximum': 1,
                'alternative_mean': .5, 'alternative_median': .5,
                'alternative_histogram': {'0': 500, '1': 500},
                'alternatives_strictly_greater_than_seed17': greater,
                'alternatives_equal_to_seed17': equal,
                'alternatives_greater_or_equal_to_seed17': greater + equal,
                'seed17_descriptive_rank_among_all_seeds': 1 + greater,
            }
            variants[label] = {
                'token_ids_in_frozen_rank_order': ids.copy(),
                'conditional_replay': {'10': copy.deepcopy(claim), '100': copy.deepcopy(claim)},
                'corpus_frequency': {
                    name: {
                        'counts_in_rank_order': [counts.get(token, 0) for token in ids],
                        'ranked_token_ids_present': sum(token in counts for token in ids),
                        'selected_token_occurrences': sum(counts.get(token, 0) for token in ids),
                    } for name, counts in frequency.items()
                },
            }
        return {
            'schema_version': 1,
            'stage': 'frozen_delta_token_bag_verification_not_text_reconstruction',
            'complete': True, 'native_export_every_byte_validated': True,
            'inputs': copy.deepcopy(self.inputs),
            'split': {'corpus_tokens': 2057, 'prefix_tokens': 2050,
                      'native_token_boundary': 2050, 'suffix_tokens': 7},
            'conditional_replay': {
                'context': 1024, 'window_counts': [10, 100],
                'tokens_per_window_including_input_and_target': 1025,
                'sequence_batch_sizes_for_ten_steps': [1, 10],
                'actual': {'seed_start': 17, 'seed_count': 1,
                           'starts_per_seed': copy.deepcopy(self.rows[:1])},
                'alternatives': {'seed_start': 10000, 'seed_count': 1000,
                                 'starts_per_seed': copy.deepcopy(self.rows[1:])},
                'distinct_token_ids_per_seed': {'10': [1] * 1001, '100': [1] * 1001},
            },
            'results': {name: copy.deepcopy(variants) for name in expected_labels()},
        }

    def sampler(self, total, context, count, seed):
        self.assertEqual((total, context, count), (2050, 1024, 100))
        self.assertTrue(seed == 17 or 10000 <= seed < 11000)
        return [0 if seed == 17 or seed % 2 == 0 else 1025] * 100

    def run_audit(self, *, sampler=None):
        self.report_path.write_text(json.dumps(self.report))
        # The oracle is mocked only to avoid irrelevant 100,100 PRNG draws.
        # bag() still executes on actual synthetic token arrays. This cache
        # reuses identical set-valued windows across repeated seed rows.
        original_bag = auditor.bag
        cache = {}

        def cached_bag(tokens, starts, context):
            key = tuple(starts), context
            if key not in cache:
                cache[key] = original_bag(tokens, starts, context)
            return cache[key]

        with mock.patch.object(auditor, 'libstdcxx13_starts', side_effect=sampler or self.sampler) as oracle:
            with mock.patch.object(auditor, 'bag', side_effect=cached_bag):
                with contextlib.redirect_stdout(io.StringIO()):
                    auditor.audit(self.report_path, self.output)
        return oracle

    def test_complete_fixture_audits_every_label_variant_and_condition(self):
        before = {record['path']: fixture_identity(Path(record['path'])) for record in self.inputs.values()}
        oracle = self.run_audit()
        result = json.loads(self.output.read_text())
        self.assertEqual(oracle.call_count, 1001)
        self.assertEqual([call.args[-1] for call in oracle.call_args_list], [17, *range(10000, 11000)])
        self.assertTrue(result['complete'])
        self.assertEqual(result['independent_sampler_rows_exact'], 1001)
        self.assertEqual(result['independent_overlap_lists_exact'], 168)
        self.assertEqual(len(result['checks']), 168)
        self.assertTrue(result['all_corpus_counts_in_rank_order_exact'])
        self.assertEqual({item['name'] for item in result['checks']}, set(expected_labels()))
        self.assertEqual({item['variant'] for item in result['checks']}, {'real', 'identity_shuffled'})
        self.assertEqual({item['windows'] for item in result['checks']}, {10, 100})
        self.assertEqual(before, {path: fixture_identity(Path(path)) for path in before})

    def test_tampered_actual_or_alternative_overlap_rejected(self):
        for key in ['seed17_overlap', 'alternative_overlap_counts_in_seed_order']:
            with self.subTest(key=key):
                self.report = self.fixture_report()
                claim = self.report['results']['boundary_raw_580']['real']['conditional_replay']['10']
                if key == 'seed17_overlap':
                    claim[key] += 1
                else:
                    claim[key][317] += 1
                with self.assertRaisesRegex(ValueError, 'bag intersections'):
                    self.run_audit()
                self.assertFalse(self.output.exists())

    def test_tampered_actual_or_alternative_seed_row_rejected(self):
        for group, row in [('actual', 0), ('alternatives', 17)]:
            with self.subTest(group=group):
                self.report = self.fixture_report()
                self.report['conditional_replay'][group]['starts_per_seed'][row][31] = 17
                with self.assertRaisesRegex(ValueError, 'independent sampler differs'):
                    self.run_audit()
                self.assertFalse(self.output.exists())

    def test_missing_extra_or_reordered_seed_rows_rejected(self):
        for mode in ['missing', 'extra', 'reordered']:
            with self.subTest(mode=mode):
                self.report = self.fixture_report()
                rows = self.report['conditional_replay']['alternatives']['starts_per_seed']
                if mode == 'missing':
                    rows.pop()
                elif mode == 'extra':
                    rows.append(rows[0])
                else:
                    rows[0], rows[1] = rows[1], rows[0]
                with self.assertRaisesRegex(ValueError, 'independent sampler differs|seed labels/counts'):
                    self.run_audit()

    def test_missing_extra_or_empty_comparison_arm_sets_rejected(self):
        for mode in ['missing', 'extra', 'empty']:
            with self.subTest(mode=mode):
                self.report = self.fixture_report()
                if mode == 'missing':
                    self.report['results'].pop('shared_endpoint_norm')
                elif mode == 'extra':
                    self.report['results']['unknown_arm'] = copy.deepcopy(self.report['results']['shared_endpoint_norm'])
                else:
                    self.report['results'] = {}
                with self.assertRaisesRegex(ValueError, 'all 42'):
                    self.run_audit()
                self.assertFalse(self.output.exists())

    def test_missing_or_extra_identity_control_variant_rejected(self):
        for mode in ['missing', 'extra']:
            with self.subTest(mode=mode):
                self.report = self.fixture_report()
                variants = self.report['results']['endpoint_norm_580']
                if mode == 'missing':
                    variants.pop('identity_shuffled')
                else:
                    variants['invented'] = copy.deepcopy(variants['real'])
                with self.assertRaisesRegex(ValueError, 'identity-control variants'):
                    self.run_audit()

    def test_sampler_seed_labels_and_counts_rejected(self):
        for label, key, value in [('actual', 'seed_start', 18),
                                  ('actual', 'seed_count', 2),
                                  ('alternatives', 'seed_start', 10001),
                                  ('alternatives', 'seed_count', 999)]:
            with self.subTest(label=label, key=key):
                self.report = self.fixture_report()
                self.report['conditional_replay'][label][key] = value
                with self.assertRaisesRegex(ValueError, 'seed labels/counts'):
                    self.run_audit()

    def test_inconsistent_or_empty_split_rejected(self):
        for key, value in [('prefix_tokens', 0), ('prefix_tokens', -1),
                           ('prefix_tokens', 2057), ('prefix_tokens', 2050.0),
                           ('prefix_tokens', True), ('suffix_tokens', 8),
                           ('native_token_boundary', 2049)]:
            with self.subTest(key=key, value=value):
                self.report = self.fixture_report()
                self.report['split'][key] = value
                with self.assertRaisesRegex(ValueError, 'prefix/suffix split'):
                    self.run_audit()

    def test_completion_flags_and_replay_settings_rejected(self):
        for key, value in [('complete', False), ('complete', 1),
                           ('native_export_every_byte_validated', False),
                           ('native_export_every_byte_validated', 'true')]:
            with self.subTest(key=key, value=value):
                self.report = self.fixture_report()
                self.report[key] = value
                with self.assertRaisesRegex(ValueError, 'incomplete verification'):
                    self.run_audit()
        for key, value in [('context', 1023), ('window_counts', [100, 10])]:
            with self.subTest(key=key):
                self.report = self.fixture_report()
                self.report['conditional_replay'][key] = value
                with self.assertRaisesRegex(ValueError, 'unexpected replay settings'):
                    self.run_audit()

    def test_consistent_split_too_short_for_one_window_rejected(self):
        for prefix in (1, 1023, 1024):
            with self.subTest(prefix=prefix):
                self.report = self.fixture_report()
                self.report['split'].update(prefix_tokens=prefix,
                                            native_token_boundary=prefix,
                                            suffix_tokens=2057 - prefix)
                with self.assertRaisesRegex(ValueError, 'prefix/suffix split'):
                    self.run_audit()

    def test_tampered_overlap_summary_and_vocabulary_size_rejected(self):
        for mutation in ['mean', 'histogram', 'rank', 'distinct']:
            with self.subTest(mutation=mutation):
                self.report = self.fixture_report()
                claim = self.report['results']['shared_boundary_adjusted']['identity_shuffled']['conditional_replay']['100']
                if mutation == 'mean':
                    claim['alternative_mean'] = .6
                elif mutation == 'histogram':
                    claim['alternative_histogram']['1'] = 501
                elif mutation == 'rank':
                    claim['seed17_descriptive_rank_among_all_seeds'] += 1
                else:
                    self.report['conditional_replay']['distinct_token_ids_per_seed']['100'][93] = 2
                with self.assertRaisesRegex(ValueError, 'summaries differ|vocabulary sizes differ'):
                    self.run_audit()

    def test_tampered_rank_order_counts_presence_or_occurrences_rejected(self):
        for key in ['counts_in_rank_order', 'ranked_token_ids_present', 'selected_token_occurrences']:
            with self.subTest(key=key):
                self.report = self.fixture_report()
                counts = self.report['results']['endpoint_norm_8160']['real']['corpus_frequency']['full']
                if key == 'counts_in_rank_order':
                    counts[key][0] += 1
                else:
                    counts[key] += 1
                with self.assertRaisesRegex(ValueError, 'corpus frequencies differ'):
                    self.run_audit()

    def test_duplicate_candidate_ids_rejected(self):
        ids = self.report['results']['boundary_raw_580']['real']['token_ids_in_frozen_rank_order']
        ids[1] = ids[0]
        with self.assertRaisesRegex(ValueError, '128 distinct'):
            self.run_audit()

    def test_noninteger_or_out_of_range_candidate_ids_rejected(self):
        for value in [-1, 50257, True, 127.0, '127']:
            with self.subTest(value=value):
                self.report = self.fixture_report()
                ids = self.report['results']['endpoint_norm_580']['real']['token_ids_in_frozen_rank_order']
                ids[0] = value
                with self.assertRaisesRegex(ValueError, '128 distinct'):
                    self.run_audit()

    def test_modified_input_before_audit_rejected(self):
        self.tokens_path.write_bytes(self.tokens_path.read_bytes() + b'changed')
        with self.assertRaisesRegex(ValueError, 'input identity differs'):
            self.run_audit()

    def test_modified_input_during_audit_rejected_without_output(self):
        altered = False

        def change_after_input_was_read(*args):
            nonlocal altered
            if not altered:
                self.tokens_path.write_bytes(self.tokens_path.read_bytes() + b'changed')
                altered = True
            return self.sampler(*args)

        with self.assertRaisesRegex(ValueError, 'input changed during audit'):
            self.run_audit(sampler=change_after_input_was_read)
        self.assertTrue(altered)
        self.assertFalse(self.output.exists())

    def test_out_of_vocabulary_native_token_rejected_after_rehash(self):
        payload = bytearray(self.tokens_path.read_bytes())
        payload[:4] = struct.pack('<I', 50257)
        self.tokens_path.write_bytes(payload)
        self.report['inputs']['native_tokens'] = fixture_identity(self.tokens_path)
        with self.assertRaisesRegex(ValueError, 'invalid native token'):
            self.run_audit()

    def test_existing_output_rejected_and_preserved(self):
        self.output.write_text('preserve existing result')
        with mock.patch.object(auditor, 'identity', side_effect=AssertionError('unexpected input access')):
            with self.assertRaises(FileExistsError):
                auditor.audit(self.report_path, self.output)
        self.assertEqual(self.output.read_text(), 'preserve existing result')


if __name__ == '__main__':
    unittest.main()
