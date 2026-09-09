"""Synthetic checks of production sampling and generation-to-trace binding."""

import copy
import json
import math
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from .checkpoint import GPT2Checkpoint, GPT2Config, tensor_manifest
from .phrase_reference import forward
from .phrase_trace_analysis import residual_stage_names
from . import token_trace_analysis as analysis


def f32(value):
    return struct.unpack('<f', struct.pack('<f', float(value)))[0]


def scalar_distribution(values, temperature, uniform, *, premature_fp64=False):
    """Independent scalar oracle: round the subtraction, then sequential sums."""
    values = [f32(value) for value in values]
    maximum = max(values)
    shifts = [value-maximum if premature_fp64 else f32(value-maximum) for value in values]
    weights = [math.exp(shift/temperature) for shift in shifts]
    denominator = 0.
    for weight in weights:
        denominator = denominator+weight
    probabilities = [weight/denominator for weight in weights]
    cdf, total = [], 0.
    for p in probabilities:
        total = total+p
        cdf.append(total)
    cdf[-1] = 1.
    selected = next(i for i, cumulative in enumerate(cdf) if cumulative >= uniform)
    return probabilities, cdf, selected


class SamplingMathTest(unittest.TestCase):
    def test_production_distribution_matches_independent_scalar_oracle(self):
        for values in [[1, -2**-24, 0], [-14, 2.125, -3.5, .0625],
                       [1000, 999, -1000], [0., -0., 0., 0.]]:
            for temperature in [.2, .8, 1., 3.]:
                for uniform in [0., .125, .5, .9999999999999999]:
                    expected = scalar_distribution(values, temperature, uniform)
                    actual = analysis.production_distribution(values, temperature, uniform)
                    np.testing.assert_array_equal(actual[0], expected[0])
                    self.assertEqual(actual[1:], expected[1:])

    def test_fp64_before_subtraction_changes_cdf_and_can_change_selected_token(self):
        values = np.array([1., -2**-24, 0.], dtype=np.float32)
        correct = scalar_distribution(values, .8, .5)
        wrong = scalar_distribution(values, .8, .5, premature_fp64=True)
        self.assertNotEqual(correct[1][0], wrong[1][0])
        uniform = (correct[1][0]+wrong[1][0])/2
        actual = analysis.production_distribution(values, .8, uniform)
        self.assertEqual(actual[2], 1)
        self.assertEqual(scalar_distribution(values, .8, uniform, premature_fp64=True)[2], 0)

    def test_lower_bound_cdf_boundary_and_zero_probability_edge_semantics(self):
        probabilities, cdf, _ = analysis.production_distribution([0, 0, 0], 1, .5)
        for index in range(2):
            self.assertEqual(analysis.production_distribution([0, 0, 0], 1, cdf[index])[2], index)
            self.assertEqual(analysis.production_distribution([0, 0, 0], 1,
                math.nextafter(cdf[index], 1.))[2], index+1)
        self.assertEqual(cdf[-1], 1.)
        # std::lower_bound really chooses index0 at u=0 even if its interval
        # has zero probability after exp underflow. Do not silently use upper_bound.
        probabilities, cdf, selected = analysis.production_distribution([-1000, 0], 1, 0.)
        self.assertEqual(probabilities[0], 0)
        self.assertEqual(selected, 0)
        self.assertEqual(analysis.production_distribution([-1000, 0], 1,
            math.nextafter(0., 1.))[2], 1)

    def test_target_readout_keeps_target_distinct_from_greedy_and_sampled(self):
        pieces = {0: b'a', 1: b' b', 2: b'\xff'}
        info = analysis.readout(np.array([2, 2, 0], dtype=np.float32), 2, pieces, .8, 0.)
        self.assertEqual(info['winner']['id'], 0)
        self.assertEqual(info['target']['id'], 2)
        self.assertEqual(info['target']['rank'], 3)
        self.assertEqual(info['same_uniform_selected_id'], 0)
        expected, cdf, _ = scalar_distribution([2, 2, 0], .8, 0.)
        self.assertEqual(info['target_probability_at_generation_temperature'], expected[2])
        self.assertEqual(info['target_cdf_lower'], cdf[1])
        self.assertEqual(info['target_cdf_upper'], 1.)
        self.assertIn('\\xff', info['target']['piece_escaped'])

    def test_sampling_inputs_and_jsonable_preserve_exact_values(self):
        for values in [[], [1], [[1, 2]], [1, np.nan], [np.inf, 1]]:
            with self.assertRaises(ValueError):
                analysis.production_distribution(values, .8, .5)
        for temperature, uniform in [(0, .5), (-1, .5), (np.inf, .5), (np.nan, .5),
                                      (.8, -1), (.8, 1), (.8, np.nan), (.8, np.inf)]:
            with self.assertRaises(ValueError):
                analysis.production_distribution([0, 1], temperature, uniform)
        value = {'array': np.array([1, 2], dtype=np.int32),
                 'tuple': (np.float64(.125), np.bool_(True))}
        self.assertEqual(analysis.jsonable(value), {'array': [1, 2], 'tuple': [.125, True]})


class TokenTraceBindingTest(unittest.TestCase):
    """Tiny fabricated file schema; CPU-generated arrays are not native claims."""
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.native, self.generation, self.tokenizer = [self.base/name for name in
                                                       ('native', 'generation', 'tokenizer')]
        for directory in [self.native, self.generation, self.tokenizer]:
            directory.mkdir()
        self.checkpoint_directory = self.base/'checkpoint'
        self.checkpoint_directory.mkdir()
        self.config = GPT2Config(vocab_size=7, padded_vocab_size=8, context_length=6,
                                 n_layers=2, d_model=4, n_heads=2, d_ff=6)
        rng = np.random.default_rng(891)
        for spec in tensor_manifest(self.config):
            value = rng.normal(0, .15, spec.shape).astype('<f4')
            if spec.name.endswith('.scale'):
                value += 1
            value.tofile(self.checkpoint_directory/spec.filename)
        self.checkpoint = GPT2Checkpoint(self.checkpoint_directory, self.config)
        self.ids = [1, 5, 2]
        reference = forward(self.checkpoint, self.ids, 'bf16')
        self.row = reference['logits'][-1]
        order = np.argsort(-self.row, kind='stable')
        self.target = int(order[4])  # Deliberately non-greedy, like sampled 'ag'.
        self.pieces = {token: ('t'+str(token)).encode() for token in range(7)}
        self.pieces[6] = b'\xff'
        (self.tokenizer/'tokenizer.json').write_text('{}')
        self.meta = {'complete': True, 'probe_kind': 'token_trace',
                     'no_bos': True, 'autoregressive': False, 'retokenized': False,
                     'token_ids': self.ids, 'selected_row': 2, 'target_id': self.target,
                     'prompt_rows': 3, 'vocab_size': 7, 'padded_vocab_size': 8,
                     'context_length': 6, 'checkpoint_directory': str(self.checkpoint_directory),
                     'target_logit': float(self.row[self.target]), 'target_rank': 5,
                     'greedy_id': int(order[0]), 'files': {}, 'lens': {}, 'interventions': [],
                     'parity_files': {'alternate_padding': 'alternate_padding.logits.f32',
                                      'clean_replay': 'clean_replay.logits.f32'},
                     'checks': {name: True for name in (
                         'final_lens_logits_byte_equal', 'alternate_padding_logits_byte_equal',
                         'clean_replay_logits_byte_equal', 'attention_residual_replay_byte_equal',
                         'mlp_residual_replay_byte_equal')}}
        for name, values in reference['stages'].items():
            path = self.native/(name+'.bf16')
            (values.view(np.uint32) >> 16).astype('<u2').tofile(path)
            self.meta['files'][name] = {'file': path.name, 'dtype': 'bf16', 'shape': list(values.shape)}
        np.array(self.ids, dtype='<i4').tofile(self.native/'tokens.i32')
        self.meta['files']['tokens'] = {'file': 'tokens.i32', 'dtype': 'int32', 'shape': [3, 1]}
        physical = np.full((1, 8), -np.finfo(np.float32).max, dtype='<f4')
        physical[0, :7] = self.row
        self.physical = physical
        physical.tofile(self.native/'logits.f32')
        self.meta['files']['logits'] = {'file': 'logits.f32', 'dtype': 'float32', 'shape': [1, 8]}
        for filename in self.meta['parity_files'].values():
            physical.tofile(self.native/filename)
        for name in residual_stage_names(2):
            filename = 'lens.'+name+'.f32'
            physical.tofile(self.native/filename)
            self.meta['lens'][name] = {'file': filename, 'dtype': 'float32', 'shape': [1, 8]}
        self.gen = {'complete': True, 'autoregressive': True, 'steps': 2,
                    'initial_token_ids': self.ids[:2], 'generated_token_ids': [2, self.target],
                    'all_token_ids': self.ids+[self.target], 'temperature': .8,
                    'checkpoint_directory': str(self.checkpoint_directory),
                    'tokenizer_directory': str(self.tokenizer), 'context_length': 6,
                    'vocab_size': 7, 'events_file': 'events.jsonl', 'logits_file': 'logits.f32',
                    'logits_shape': [2, 7], 'files': {
                        'events': {'file': 'events.jsonl'},
                        'logits': {'file': 'logits.f32', 'dtype': 'float32', 'shape': [2, 7]}}}
        generation_logits = reference['logits'][1:].copy()
        generation_logits.tofile(self.generation/'logits.f32')
        self.events = []
        for step, target in enumerate([2, self.target]):
            probabilities, cdf, _ = scalar_distribution(generation_logits[step], .8, .5)
            uniform = ((cdf[target-1] if target else 0)+cdf[target])/2
            order = np.argsort(-generation_logits[step], kind='stable')
            self.events.append({'index': step, 'step': step,
                'absolute_token_index': 2+step, 'context_start': 0, 'context_length': 2+step,
                'output_row': 1+step, 'token_id': target, 'piece_hex': self.pieces[target].hex(),
                'logits_byte_offset': step*7*4, 'uniform': uniform,
                'sampled_probability': probabilities[target],
                'cdf_lower': cdf[target-1] if target else 0., 'cdf_upper': cdf[target],
                'raw_logit': float(generation_logits[step, target]),
                'raw_rank': int(np.flatnonzero(order == target)[0])+1,
                'raw_argmax_token_id': int(order[0]), 'rng_state_verified': True,
                'cdf_selected_verified': True})
        self.output_index = 0

    def write_metadata(self):
        (self.native/'metadata.json').write_text(json.dumps(self.meta))
        (self.generation/'metadata.json').write_text(json.dumps(self.gen))
        (self.generation/'events.jsonl').write_text('\n'.join(json.dumps(event) for event in self.events)+'\n')

    def run_analysis(self, step=1):
        self.write_metadata()
        self.output_index += 1
        fake = SimpleNamespace(Tokenizer=SimpleNamespace(from_file=lambda _: object()))
        with mock.patch.dict('sys.modules', {'tokenizers': fake}), \
                mock.patch.object(analysis, 'GPT2Checkpoint', return_value=self.checkpoint), \
                mock.patch.object(analysis, 'gpt2_token_bytes', return_value=self.pieces), \
                mock.patch('builtins.print'):
            return analysis.run(self.native, self.generation, step, self.checkpoint_directory,
                                self.tokenizer, self.base/f'analysis_{self.output_index}')

    def test_small_bound_trace_tracks_nonwinning_target_and_exact_original_logits(self):
        report = self.run_analysis()
        self.assertTrue(report['complete'])
        self.assertTrue(report['baseline_logits_equal_original_generation'])
        self.assertEqual(report['context_token_ids'], self.ids)
        self.assertEqual(report['target_id'], self.target)
        self.assertEqual(report['baseline']['target']['rank'], 5)
        self.assertEqual(report['baseline']['same_uniform_selected_id'], self.target)
        self.assertLess(report['ledger']['native_margin'], 0)
        expected_margin = float(self.row[self.target])-float(self.row[report['competitor_id']])
        self.assertEqual(report['ledger']['native_margin'], expected_margin)
        self.assertAlmostEqual(math.fsum(report['ledger']['terms'].values()), expected_margin, places=12)
        for label in ['events', 'generation_logits', 'parity:clean_replay', 'parity:alternate_padding']:
            self.assertIn(label, report['inputs'])

    def test_original_logit_bytes_and_native_parity_must_match(self):
        original = (self.generation/'logits.f32').read_bytes()
        values = np.frombuffer(original, dtype='<f4').copy().reshape(2, 7)
        values[1, 0] = np.nextafter(values[1, 0], np.float32(np.inf))
        values.tofile(self.generation/'logits.f32')
        with self.assertRaisesRegex(ValueError, 'original autoregressive logits'):
            self.run_analysis()
        (self.generation/'logits.f32').write_bytes(original)
        altered = self.physical.copy()
        altered[0, 0] = np.nextafter(altered[0, 0], np.float32(np.inf))
        altered.tofile(self.native/'clean_replay.logits.f32')
        with self.assertRaisesRegex(ValueError, 'replay bytes'):
            self.run_analysis()

    def test_exact_context_target_event_offsets_and_geometry_are_bound(self):
        original = copy.deepcopy(self.events[1])
        for key, value in [('index', 0), ('absolute_token_index', 2), ('context_start', 1),
                           ('context_length', 2), ('token_id', (self.target+1)%7),
                           ('logits_byte_offset', 0), ('piece_hex', '00')]:
            self.events[1] = dict(original, **{key: value})
            with self.subTest(field=key), self.assertRaises(ValueError):
                self.run_analysis()
        self.events[1] = original
        original_ids = self.meta['token_ids']
        self.meta['token_ids'] = [5, 1, 2]
        with self.assertRaises(ValueError):
            self.run_analysis()
        self.meta['token_ids'] = original_ids
        self.meta['padded_vocab_size'] = 7
        with self.assertRaises(ValueError):
            self.run_analysis()

    def test_sampling_event_probability_or_uniform_mismatch_is_rejected(self):
        self.events[1]['sampled_probability'] += .01
        with self.assertRaisesRegex(ValueError, 'sampling'):
            self.run_analysis()
        self.events[1]['sampled_probability'] -= .01
        self.events[1]['uniform'] = (self.events[1]['cdf_lower']/2 if self.target else
                                      (self.events[1]['cdf_upper']+1)/2)
        with self.assertRaisesRegex(ValueError, 'sampling'):
            self.run_analysis()

    def test_stage_dtype_shape_and_raw_context_ids_are_checked(self):
        record = self.meta['files']['blocks.0.ln2']
        record['dtype'] = 'float32'
        with self.assertRaisesRegex(ValueError, 'dtype/shape'):
            self.run_analysis()
        record['dtype'] = 'bf16'
        record['shape'] = [3, 1]
        with self.assertRaisesRegex(ValueError, 'dtype/shape'):
            self.run_analysis()
        record['shape'] = [3, 4]
        np.array([1, 5, 3], dtype='<i4').tofile(self.native/'tokens.i32')
        with self.assertRaisesRegex(ValueError, 'token dump'):
            self.run_analysis()

    def test_strict_ids_counts_and_exact_native_trace_contract(self):
        original_meta, original_gen = copy.deepcopy(self.meta), copy.deepcopy(self.gen)
        for key, value in [('token_ids', [True, 5, 2]), ('selected_row', 2.),
                           ('target_id', float(self.target)), ('no_bos', False),
                           ('retokenized', True), ('autoregressive', True)]:
            self.meta = dict(original_meta, **{key: value})
            with self.subTest(native=key), self.assertRaises(ValueError):
                self.run_analysis()
        self.meta = original_meta
        for key, value in [('all_token_ids', [float(i) for i in self.gen['all_token_ids']]),
                           ('steps', 2.), ('steps', 1), ('initial_token_ids', []),
                           ('tokenizer_directory', str(self.base/'different-tokenizer'))]:
            self.gen = dict(original_gen, **{key: value})
            with self.subTest(generation=key), self.assertRaises(ValueError):
                self.run_analysis()
        self.gen = original_gen
        for step in [-1, 2, True, 1.]:
            with self.subTest(step=step), self.assertRaises(ValueError):
                self.run_analysis(step)

    def test_recorded_raw_rank_logit_cdf_and_output_row_are_not_blindly_trusted(self):
        original = copy.deepcopy(self.events[1])
        for key, value in [('output_row', 1), ('output_row', 2.),
                           ('raw_rank', original['raw_rank']+1),
                           ('raw_rank', float(original['raw_rank'])),
                           ('raw_argmax_token_id', (original['raw_argmax_token_id']+1)%7),
                           ('raw_logit', original['raw_logit']+.1),
                           ('cdf_lower', original['cdf_lower']+1e-5),
                           ('cdf_upper', original['cdf_upper']-1e-5)]:
            self.events[1] = dict(original, **{key: value})
            with self.subTest(field=key), self.assertRaises(ValueError):
                self.run_analysis()

    def test_generation_artifacts_cannot_escape_or_use_symlinks(self):
        original = copy.deepcopy(self.gen)
        for key, value in [('events_file', '../generation/events.jsonl'),
                           ('events_file', str(self.generation/'events.jsonl')),
                           ('logits_file', '../generation/logits.f32'),
                           ('logits_file', str(self.generation/'logits.f32'))]:
            self.gen = dict(original, **{key: value})
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'local filename'):
                self.run_analysis()
        self.gen = original
        (self.generation/'events_link.jsonl').symlink_to(self.generation/'events.jsonl')
        self.gen['events_file'] = 'events_link.jsonl'
        with self.assertRaisesRegex(ValueError, 'nonsymlink'):
            self.run_analysis()

    def test_native_parity_and_lens_require_separately_named_files(self):
        self.meta['parity_files']['clean_replay'] = 'logits.f32'
        with self.assertRaisesRegex(ValueError, 'independently named'):
            self.run_analysis()
        self.meta['parity_files']['clean_replay'] = 'clean_replay.logits.f32'
        self.meta['lens']['positioned']['file'] = 'logits.f32'
        with self.assertRaisesRegex(ValueError, 'independently named'):
            self.run_analysis()


if __name__ == '__main__':
    unittest.main()
