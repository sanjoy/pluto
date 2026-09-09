"""Synthetic phase, parser, provenance-gate, and verifier-orchestration tests.

No real corpus, tokenizer, checkpoint, sampler process, or GPU is accessed.
"""

import contextlib
import copy
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

import numpy as np

from . import delta_followthrough_verify as verifier


class PhaseTest(unittest.TestCase):
    def test_prescribed_zero_based_draw_slices(self):
        self.assertEqual(verifier.phase_slices(1), {
            'predicted': (100, 110), 'previous': (90, 100),
            'next': (110, 120), 'reset': (0, 10)})
        self.assertEqual(verifier.phase_slices(10), {
            'predicted': (1000, 1100), 'previous': (900, 1000),
            'next': (1100, 1200), 'reset': (0, 100)})
        for invalid in [0, 2, 100, True, 1.0, '1', None]:
            with self.subTest(batch=invalid), self.assertRaises(ValueError):
                verifier.phase_slices(invalid)

    def test_predicted_uses_all_seeds_controls_only_first_and_inclusive_target(self):
        tokens = np.arange(2500, dtype=np.int32)
        starts = np.arange(1200, dtype=np.int64)[None, :] + np.array([0, 100, 200])[:, None]
        before_tokens, before_starts = tokens.copy(), starts.copy()
        with mock.patch.object(verifier, 'VOCABULARY', 2500):
            for batch in [1, 10]:
                with self.subTest(batch=batch):
                    phases = verifier.phase_membership(tokens, starts, batch)
                    for name, (begin, end) in verifier.phase_slices(batch).items():
                        mask = phases[name]
                        self.assertEqual(mask.shape, (3 if name == 'predicted' else 1, 2500))
                        for row in range(len(mask)):
                            first = begin + row * 100
                            # Last sampled start is end-1; shifted targets
                            # include its start+1024, not just start+1023.
                            exclusive_end = end + row * 100 + 1024
                            np.testing.assert_array_equal(np.flatnonzero(mask[row]),
                                                          np.arange(first, exclusive_end))
                            self.assertTrue(mask[row, first])
                            self.assertTrue(mask[row, exclusive_end - 1])
                            self.assertFalse(mask[row, exclusive_end])
                            if first:
                                self.assertFalse(mask[row, first - 1])
        np.testing.assert_array_equal(tokens, before_tokens)
        np.testing.assert_array_equal(starts, before_starts)

    def test_repeated_windows_count_ids_once(self):
        tokens = np.arange(1025, dtype=np.int32)
        starts = np.zeros((2, 1200), dtype=np.int64)
        with mock.patch.object(verifier, 'VOCABULARY', 1025):
            for batch in [1, 10]:
                masks = verifier.phase_membership(tokens, starts, batch)
                for mask in masks.values():
                    self.assertTrue(mask.all())
                    np.testing.assert_array_equal(mask.sum(axis=1), np.full(len(mask), 1025))

    def test_short_or_out_of_bounds_phase_draws_rejected(self):
        with mock.patch.object(verifier, 'VOCABULARY', 1026):
            for starts, batch in [(np.zeros((2, 109), dtype=np.int64), 1),
                                   (np.zeros((2, 1199), dtype=np.int64), 10),
                                   (np.full((2, 1200), 2, dtype=np.int64), 1),
                                   (np.full((2, 1200), -1, dtype=np.int64), 10)]:
                with self.subTest(shape=starts.shape, batch=batch), self.assertRaises(ValueError):
                    verifier.phase_membership(np.arange(1026), starts, batch)


class SamplerChunkTest(unittest.TestCase):
    @staticmethod
    def rows(first, count, total=2050):
        return ''.join(str(seed) + ' ' + ' '.join([str((seed - first) % (total - 1024))] * 1200) + '\n'
                       for seed in range(first, first + count))

    def test_two_prescribed_500_seed_chunks_parse_all_1200_draws(self):
        outputs = [subprocess.CompletedProcess([], 0, self.rows(first, 500), '')
                   for first in [10000, 10500]]
        with mock.patch.object(subprocess, 'run', side_effect=outputs) as run:
            first = verifier.sampler_chunk('/fake/sampler', 2050, 10000, 500)
            second = verifier.sampler_chunk('/fake/sampler', 2050, 10500, 500)
        self.assertEqual(first.shape, (500, 1200))
        self.assertEqual(second.shape, (500, 1200))
        for actual in (first, second):
            np.testing.assert_array_equal(actual, np.repeat(np.arange(500)[:, None], 1200, axis=1))
        self.assertEqual([call.args[0] for call in run.call_args_list], [
            ['/fake/sampler', '2050', '1024', '1200', '10000', '500'],
            ['/fake/sampler', '2050', '1024', '1200', '10500', '500']])
        for call in run.call_args_list:
            self.assertTrue(call.kwargs['check'])
            self.assertEqual(call.kwargs['timeout'], 30)

    def test_seed17_row_uses_inclusive_last_valid_window_start(self):
        output = '17 ' + ' '.join(['1025'] * 1200) + '\n'
        with mock.patch.object(subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, output, '')):
            starts = verifier.sampler_chunk('/fake/sampler', 2050, 17, 1)
        np.testing.assert_array_equal(starts, np.full((1, 1200), 1025))

    def test_invalid_chunk_counts_fail_before_subprocess(self):
        for count in [0, 501, -1, True, 1.0, '1']:
            with self.subTest(count=count):
                with mock.patch.object(subprocess, 'run', side_effect=AssertionError('subprocess launched')):
                    with self.assertRaises(ValueError):
                        verifier.sampler_chunk('/fake/sampler', 2050, 17, count)

    def test_malformed_seed_rows_syntax_bounds_count_and_stderr(self):
        valid = '17 ' + ' '.join(['0'] * 1200) + '\n'
        cases = [valid.replace('17 ', '18 ', 1), valid + valid,
                 '17 ' + ' '.join(['0'] * 1199) + '\n',
                 '17 ' + ' '.join(['0'] * 1201) + '\n']
        for value in ['-1', '+0', '0.0', '١', '1026']:
            cases.append('17 ' + value + ' ' + ' '.join(['0'] * 1199) + '\n')
        for output, error in [(text, '') for text in cases] + [(valid, 'warning')]:
            with self.subTest(prefix=output[:20], error=error):
                with mock.patch.object(subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, output, error)):
                    with self.assertRaises(ValueError):
                        verifier.sampler_chunk('/fake/sampler', 2050, 17, 1)


class FrozenAuthenticationTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.freeze = self.root / 'frozen.json'
        self.output = self.root / 'verified.json'

    def write_json(self, path, value):
        path.write_text(json.dumps(value))
        return verifier.file_record(path)

    def fixture(self):
        vocabulary = 256
        source_dir = Path(verifier.__file__).parent
        sources = {name: verifier.file_record(source_dir / name) for name in
                   ['delta_followthrough.py', 'restart_delta.py', 'checkpoint_archive.py',
                    'checkpoint.py', 'late_mlp_paths.py']}
        for name in ['protocol', 'history_inventory']:
            path = self.root / name
            path.write_text('synthetic fixture source\n')
            sources[name] = verifier.file_record(path)
        files = {}
        for boundary in [580, 1780, 4280, 7230, 8160]:
            for step in [boundary + 90, boundary + 100, boundary + 110]:
                name = f'interval_{step}_{step + 10}'
                path = self.root / (name + '.npz')
                # Authentication checks interval-file identities; separate
                # numerical tests validate the interval arrays themselves.
                path.write_bytes(b'synthetic activity file identity only')
                files[name] = verifier.file_record(path)
        scores = {name: np.arange(vocabulary, dtype=np.float64)
                  for name in verifier.extraction.expected_names()}
        permutation = np.random.Generator(np.random.PCG64(20260909)).permutation(vocabulary)
        candidates = {}
        for name, score in scores.items():
            ids = sorted(range(vocabulary), key=lambda token: (-float(score[token]), token))[:128]
            item = {'token_ids': ids, 'scores': score[ids].tolist(),
                    'identity_shuffled_token_ids': permutation[ids].tolist()}
            if name.startswith('shared_'):
                item['component_scores'] = [score[ids].tolist() for _ in range(5)]
                item['positive_component_count'] = [5] * 128
            candidates[name] = item
        with (self.root / 'scores.npz').open('wb') as handle:
            np.savez(handle, **scores, identity_permutation=permutation)
        files['complete_scores'] = verifier.file_record(self.root / 'scores.npz')
        self.plan = {'stage': 'delta_followthrough_plan_no_corpus_or_model',
                     'created_utc_before_weight_reads': '2026-09-09T00:00:00+00:00',
                     'sources': sources, 'boundaries': [580, 1780, 4280, 7230, 8160],
                     'logical_vocab_size': vocabulary, 'top_k': 128, 'seed': 20260909,
                     'checkpoint_offsets': [90, 100, 110, 120],
                     'center_interval_offsets': [100, 110],
                     'conditional_prediction_one_based_windows_at_sequence_batch_size_one': [101, 110]}
        files['plan'] = self.write_json(self.root / 'plan.json', self.plan)
        self.candidates = {'stage': 'frozen_unordered_followthrough_token_candidates',
                           'created_utc': '2026-09-09T00:00:01+00:00',
                           'top_k': 128, 'no_corpus_or_tokenizer_access': True,
                           'no_model_execution': True, 'candidate_lists': candidates}
        files['candidates'] = self.write_json(self.root / 'candidates.json', self.candidates)
        self.frozen = {'schema_version': 1, 'complete': True,
                       'stage': 'delta_followthrough_candidate_freeze_not_sequence_recovery',
                       'frozen_utc_before_corpus_verification': '2026-09-09T00:00:02+00:00',
                       'sources': sources, 'files': files}
        self.write_json(self.freeze, self.frozen)

    def test_valid_synthetic_freeze_authenticates_all_twelve_lists(self):
        self.fixture()
        with mock.patch.object(verifier, 'VOCABULARY', 256):
            frozen, candidates, identity = verifier.authenticate(self.freeze)
        self.assertEqual(frozen, self.frozen)
        self.assertEqual(candidates, self.candidates)
        self.assertEqual(identity, verifier.file_record(self.freeze))
        self.assertEqual(len(candidates['candidate_lists']), 12)

    def test_score_or_shared_annotation_tampering_rejected(self):
        self.fixture()
        for kind in ['ranking', 'component']:
            with self.subTest(kind=kind):
                changed = copy.deepcopy(self.candidates)
                item = changed['candidate_lists']['shared_followthrough_adjusted']
                if kind == 'ranking':
                    item['token_ids'][0], item['token_ids'][1] = item['token_ids'][1], item['token_ids'][0]
                else:
                    item['component_scores'][0][0] += 1
                self.frozen['files']['candidates'] = self.write_json(self.root / 'candidates.json', changed)
                self.write_json(self.freeze, self.frozen)
                with mock.patch.object(verifier, 'VOCABULARY', 256):
                    with self.assertRaisesRegex(ValueError, 'ranking differs|components differ'):
                        verifier.authenticate(self.freeze)

    def test_rehashed_wrong_plan_stage_or_phase_prediction_rejected(self):
        for key, value in [('stage', 'unrelated_plan'),
                           ('checkpoint_offsets', [0, 10, 20, 30]),
                           ('center_interval_offsets', [90, 100]),
                           ('conditional_prediction_one_based_windows_at_sequence_batch_size_one', [100, 109])]:
            with self.subTest(key=key):
                self.fixture()
                self.plan[key] = value
                self.frozen['files']['plan'] = self.write_json(self.root / 'plan.json', self.plan)
                self.write_json(self.freeze, self.frozen)
                with mock.patch.object(verifier, 'VOCABULARY', 256):
                    with self.assertRaisesRegex(ValueError, 'plan settings differ'):
                        verifier.authenticate(self.freeze)

    def test_rehashed_out_of_order_or_timezone_free_chronology_rejected(self):
        for item, value in [('plan', '2026-09-09T00:00:02+00:00'),
                            ('candidates', '2026-09-08T00:00:00+00:00'),
                            ('candidates', '2026-09-09T00:00:03+00:00'),
                            ('freeze', '2026-09-09T00:00:00+00:00'),
                            ('plan', '2026-09-09T00:00:00')]:
            with self.subTest(item=item, value=value):
                self.fixture()
                if item == 'plan':
                    self.plan['created_utc_before_weight_reads'] = value
                    self.frozen['files']['plan'] = self.write_json(self.root / 'plan.json', self.plan)
                elif item == 'candidates':
                    self.candidates['created_utc'] = value
                    self.frozen['files']['candidates'] = self.write_json(self.root / 'candidates.json', self.candidates)
                else:
                    self.frozen['frozen_utc_before_corpus_verification'] = value
                self.write_json(self.freeze, self.frozen)
                with mock.patch.object(verifier, 'VOCABULARY', 256):
                    with self.assertRaisesRegex(ValueError, 'timezone-aware and ordered'):
                        verifier.authenticate(self.freeze)

    def test_chronology_compares_instants_not_timezone_strings(self):
        self.fixture()
        self.plan['created_utc_before_weight_reads'] = '2026-09-09T01:00:00+01:00'
        self.frozen['files']['plan'] = self.write_json(self.root / 'plan.json', self.plan)
        self.write_json(self.freeze, self.frozen)
        with mock.patch.object(verifier, 'VOCABULARY', 256):
            verifier.authenticate(self.freeze)

    def test_duplicate_json_keys_rejected_in_each_frozen_document(self):
        for document in ['frozen', 'plan', 'candidates']:
            with self.subTest(document=document):
                self.fixture()
                value = {'frozen': self.frozen, 'plan': self.plan, 'candidates': self.candidates}[document]
                path = self.root / (document + '.json')
                path.write_text(json.dumps(value)[:-1] + ', "duplicate": 1, "duplicate": 2}')
                if document != 'frozen':
                    self.frozen['files'][document] = verifier.file_record(path)
                    self.write_json(self.freeze, self.frozen)
                with mock.patch.object(verifier, 'VOCABULARY', 256):
                    with self.assertRaisesRegex(ValueError, 'duplicate JSON'):
                        verifier.authenticate(self.freeze)

    def test_nonfinite_json_rejected_even_in_unreferenced_nested_metadata(self):
        for document in ['frozen', 'plan', 'candidates']:
            for value in ['NaN', 'Infinity', '-Infinity', '1e999', '-1e999']:
                with self.subTest(document=document, value=value):
                    self.fixture()
                    original = {'frozen': self.frozen, 'plan': self.plan, 'candidates': self.candidates}[document]
                    path = self.root / (document + '.json')
                    path.write_text(json.dumps(original)[:-1] + ', "unused": {"nested": [0, ' + value + ']}}')
                    if document != 'frozen':
                        self.frozen['files'][document] = verifier.file_record(path)
                        self.write_json(self.freeze, self.frozen)
                    with mock.patch.object(verifier, 'VOCABULARY', 256):
                        with self.assertRaisesRegex(ValueError, 'nonfinite|non-finite'):
                            verifier.authenticate(self.freeze)

    def test_integer_scores_rejected_even_when_all_numerical_values_match(self):
        self.fixture()
        path = self.root / 'scores.npz'
        with np.load(path, allow_pickle=False) as source:
            arrays = {name: source[name] for name in source.files}
        name = 'followthrough_raw_580'
        original = arrays[name].copy()
        arrays[name] = arrays[name].astype(np.int64)
        np.testing.assert_array_equal(arrays[name], original)
        with path.open('wb') as stream:
            np.savez(stream, **arrays)
        self.frozen['files']['complete_scores'] = verifier.file_record(path)
        self.write_json(self.freeze, self.frozen)
        with mock.patch.object(verifier, 'VOCABULARY', 256):
            with self.assertRaisesRegex(ValueError, 'invalid full scores'):
                verifier.authenticate(self.freeze)

    def test_floating_permutation_rejected_even_when_all_values_match(self):
        self.fixture()
        path = self.root / 'scores.npz'
        with np.load(path, allow_pickle=False) as source:
            arrays = {name: source[name] for name in source.files}
        original = arrays['identity_permutation'].copy()
        arrays['identity_permutation'] = original.astype(np.float64)
        np.testing.assert_array_equal(arrays['identity_permutation'], original)
        with path.open('wb') as stream:
            np.savez(stream, **arrays)
        self.frozen['files']['complete_scores'] = verifier.file_record(path)
        self.write_json(self.freeze, self.frozen)
        with mock.patch.object(verifier, 'VOCABULARY', 256):
            with self.assertRaisesRegex(ValueError, 'identity permutation differs'):
                verifier.authenticate(self.freeze)

    def test_changed_frozen_artifact_cannot_open_corpus_or_write_start(self):
        self.fixture()
        with (self.root / 'scores.npz').open('ab') as handle:
            handle.write(b'tampering')
        with mock.patch.object(verifier, 'VOCABULARY', 256):
            with mock.patch.object(verifier.earlier, 'load_native', side_effect=AssertionError('corpus opened')):
                with self.assertRaisesRegex(ValueError, 'frozen file or source changed'):
                    verifier.verify(self.freeze, self.root / 'corpus', self.root / 'tokens',
                                    self.root / 'offsets', self.root, self.root / 'sampler', self.output)
        self.assertFalse(self.output.exists())
        self.assertFalse((self.root / 'verified_start.json').exists())
        self.assertFalse((self.root / 'verified_starts.npz').exists())

    def test_missing_or_incomplete_freeze_cannot_open_corpus(self):
        for incomplete in [False, True]:
            if incomplete:
                self.write_json(self.freeze, {'schema_version': 1, 'complete': False})
            with self.subTest(incomplete=incomplete):
                with mock.patch.object(verifier.earlier, 'load_native', side_effect=AssertionError('corpus opened')):
                    with self.assertRaises((FileNotFoundError, ValueError)):
                        verifier.verify(self.freeze, self.root / 'corpus', self.root / 'tokens',
                                        self.root / 'offsets', self.root, self.root / 'sampler', self.output)
                self.assertFalse(self.output.exists())
                self.assertFalse((self.root / 'verified_start.json').exists())
                self.assertFalse((self.root / 'verified_starts.npz').exists())

    def test_existing_result_start_or_starts_fail_before_authentication(self):
        for name in ['verified.json', 'verified_start.json', 'verified_starts.npz']:
            path = self.root / name
            path.write_bytes(b'preserve existing evidence')
            with self.subTest(name=name):
                with mock.patch.object(verifier, 'authenticate', side_effect=AssertionError('unexpected input read')):
                    with self.assertRaises(FileExistsError):
                        verifier.verify(self.freeze, self.root / 'corpus', self.root / 'tokens',
                                        self.root / 'offsets', self.root, self.root / 'sampler', self.output)
                self.assertEqual(path.read_bytes(), b'preserve existing evidence')
            path.unlink()


class VerificationOrchestrationTest(unittest.TestCase):
    def test_start_record_precedes_inputs_and_alternatives_are_two_500_seed_chunks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / 'result.json'
            start_path = root / 'result_start.json'
            freeze = root / 'frozen.json'
            freeze.write_text('authentication is stubbed independently in this orchestration test')
            sampler = root / 'sampler'
            sampler.write_bytes(b'not an executable; subprocess is mocked')
            paths = {name: root / name for name in ['corpus', 'tokens', 'offsets', 'tokenizer.json']}
            for path in paths.values():
                path.write_bytes(b'synthetic placeholder')
            original_record = verifier.file_record
            freeze_identity = original_record(freeze)
            opened = []

            def tracked_record(path):
                if Path(path) in paths.values():
                    self.assertTrue(start_path.exists(), 'corpus hash preceded start record')
                    opened.append(str(path))
                return original_record(path)

            def load_native(*args):
                self.assertTrue(start_path.exists())
                self.assertEqual(set(opened), {str(path) for path in paths.values()})
                tokens = np.arange(2051, dtype=np.int32) % 256
                return b'fixture', tokens, np.arange(2052), {token: str(token).encode() for token in range(256)}, 2050, 2050

            def fake_sampler(binary, total, first, count):
                self.assertEqual((binary, total), (sampler.resolve(), 2050))
                return np.zeros((count, 1200), dtype=np.int64)

            def fake_membership(tokens, starts, batch):
                self.assertEqual(len(tokens), 2050)
                self.assertEqual(starts.shape, (1001, 1200))
                predicted = np.zeros((1001, 256), dtype=bool)
                predicted[0, :128] = True
                predicted[1:, 128:] = True
                previous = np.zeros((1, 256), dtype=bool)
                previous[0, :64] = True
                next_ = np.zeros((1, 256), dtype=bool)
                next_[0, 64:128] = True
                reset = np.zeros((1, 256), dtype=bool)
                reset[0, 128:] = True
                return {'predicted': predicted, 'previous': previous, 'next': next_, 'reset': reset}

            candidates = {'candidate_lists': {name: {'token_ids': list(range(128)),
                                                     'identity_shuffled_token_ids': list(range(128, 256))}
                                              for name in verifier.extraction.expected_names()}}
            with mock.patch.object(verifier, 'VOCABULARY', 256):
                with mock.patch.object(verifier, 'authenticate', return_value=({'fixture': True}, candidates, freeze_identity)) as authenticate:
                    with mock.patch.object(verifier, 'file_record', side_effect=tracked_record):
                        with mock.patch.object(verifier.earlier, 'load_native', side_effect=load_native):
                            with mock.patch.object(verifier, 'sampler_chunk', side_effect=fake_sampler) as chunks:
                                with mock.patch.object(verifier, 'phase_membership', side_effect=fake_membership) as phases:
                                    with mock.patch.object(subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, 'synthetic implementation', '')):
                                        with contextlib.redirect_stdout(io.StringIO()):
                                            result = verifier.verify(freeze, paths['corpus'], paths['tokens'], paths['offsets'], root, sampler, output)
            self.assertEqual(authenticate.call_count, 2)
            self.assertEqual([call.args[2:] for call in chunks.call_args_list], [(17, 1), (10000, 500), (10500, 500)])
            self.assertEqual([call.args[-1] for call in phases.call_args_list], [1, 10])
            with np.load(root / 'result_starts.npz', allow_pickle=False) as starts:
                np.testing.assert_array_equal(starts['seeds'], [17, *range(10000, 11000)])
                self.assertEqual(starts['starts'].shape, (1001, 1200))
            self.assertTrue(result['complete'])
            self.assertEqual(len(result['results']), 12)
            real = result['results']['followthrough_raw_580']['real']['batch_conditions']['1']
            self.assertEqual(real['seed17_phase_overlap'], {'predicted': 128, 'previous': 64, 'next': 64, 'reset': 0})
            self.assertEqual(real['predicted']['seed17_overlap'], 128)
            self.assertEqual(real['predicted']['alternative_overlap_counts_in_seed_order'], [0] * 1000)
            self.assertEqual(real['seed17_membership_in_rank_order']['previous'], [True] * 64 + [False] * 64)


if __name__ == '__main__':
    unittest.main()
