"""Tiny complete checkpoints and independent dense oracles; no GPU required."""

import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from . import paired_weight_diff as diff
from .checkpoint import GPT2Config, tensor_manifest


CONFIG = GPT2Config(vocab_size=5, padded_vocab_size=8, context_length=3,
                    n_layers=2, d_model=4, n_heads=2, d_ff=6)


def write_checkpoint(path, values=None):
    path.mkdir()
    values = values or {}
    result = {}
    for spec in tensor_manifest(CONFIG):
        array = np.asarray(values.get(spec.name, np.zeros(spec.shape)), dtype='<f4')
        if array.shape != spec.shape:
            raise AssertionError((spec.name, array.shape, spec.shape))
        array.tofile(path / spec.filename)
        result[spec.name] = array
    return result


def flatten(values):
    return np.concatenate([values[spec.name].ravel().astype(np.float64)
                           for spec in tensor_manifest(CONFIG)])


class PairedWeightDiffTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.original = self.root / 'step_10'
        self.replacement = self.root / 'step_12'
        self.initial = self.root / 'initial'

    def compare(self, **kwargs):
        return diff.compare_checkpoints(self.original, self.replacement,
                                         config=CONFIG, **kwargs)

    def test_dense_oracle_all_tensors_and_model_with_initial(self):
        rng = np.random.default_rng(19)
        all_values = []
        for path in (self.original, self.replacement, self.initial):
            all_values.append(write_checkpoint(path, {
                spec.name: rng.normal(size=spec.shape) for spec in tensor_manifest(CONFIG)}))
        result = self.compare(initial=self.initial, chunk_elements=7, top_k=3)

        def verify(actual, a, b, initial):
            a, b, initial = (np.asarray(value, dtype=np.float64).ravel()
                              for value in (a, b, initial))
            delta, ai, bi = b - a, a - initial, b - initial
            expected = {
                'original_l2': np.linalg.norm(a),
                'replacement_l2': np.linalg.norm(b),
                'delta_l2': np.linalg.norm(delta),
                'relative_delta_to_original_l2': np.linalg.norm(delta) / np.linalg.norm(a),
                'original_replacement_cosine': np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b),
            }
            for name, value in expected.items():
                self.assertAlmostEqual(actual[name], value, delta=1e-12)
            self.assertAlmostEqual(actual['from_initial']['original_delta_l2'], np.linalg.norm(ai))
            self.assertAlmostEqual(actual['from_initial']['replacement_delta_l2'], np.linalg.norm(bi))
            self.assertAlmostEqual(actual['from_initial']['delta_cosine'],
                                   np.dot(ai, bi) / np.linalg.norm(ai) / np.linalg.norm(bi))

        verify(result['model'], *[flatten(values) for values in all_values])
        for item in result['tensors']:
            verify(item, *[values[item['name']] for values in all_values])
        self.assertEqual(result['parameter_count'], len(flatten(all_values[0])))
        self.assertEqual(result['tensor_count'], len(tensor_manifest(CONFIG)))
        self.assertEqual(len(result['largest_tensor_deltas']), 3)
        for label, path in (('original', self.original), ('replacement', self.replacement),
                            ('initial', self.initial)):
            for spec in tensor_manifest(CONFIG):
                self.assertEqual(result[label]['weight_sha256'][spec.filename],
                                 hashlib.sha256((path / spec.filename).read_bytes()).hexdigest())
        json.dumps(result, allow_nan=False)

    def test_logical_embedding_rows_exclude_padding_but_model_norm_includes_it(self):
        write_checkpoint(self.original)
        embedding = np.zeros((8, 4))
        embedding[2, 0], embedding[4, 1], embedding[7, 0] = 3, 4, 100
        write_checkpoint(self.replacement, {'token_embedding.weight': embedding})
        result = self.compare(top_k=20, chunk_elements=3)
        rows = result['group_rankings']['embedding_rows']
        self.assertEqual([row['token_id'] for row in rows], [4, 2, 0, 1, 3])
        self.assertEqual([row['delta_l2'] for row in rows], [4, 3, 0, 0, 0])
        self.assertAlmostEqual(result['model']['delta_l2'], math.sqrt(10025))

    def test_mlp_input_columns_and_output_rows_have_correct_orientation(self):
        write_checkpoint(self.original)
        input_weight, output_weight = np.zeros((4, 6)), np.zeros((6, 4))
        input_weight[0, 5], input_weight[3, 5], input_weight[2, 1] = 3, 4, 6
        output_weight[3, 0], output_weight[3, 2], output_weight[2, 1] = 5, 12, 11
        write_checkpoint(self.replacement, {'blocks.1.mlp.input.weight': input_weight,
                                             'blocks.1.mlp.output.weight': output_weight,
                                             'blocks.1.mlp.input.bias': np.full(6, 100)})
        result = self.compare(top_k=2, chunk_elements=5)['group_rankings']['blocks'][1]
        self.assertEqual(result['mlp_input_columns'], [
            {'neuron': 1, 'delta_l2': 6.}, {'neuron': 5, 'delta_l2': 5.}])
        self.assertEqual(result['mlp_output_rows'], [
            {'neuron': 3, 'delta_l2': 13.}, {'neuron': 2, 'delta_l2': 11.}])

    def test_attention_qkv_layout_and_output_row_groups(self):
        write_checkpoint(self.original)
        qkv, output = np.zeros((4, 12)), np.zeros((4, 4))
        # Packed Q then K then V, two channels per head. Output heads slice
        # rows instead of columns; differing placements catch transposition.
        qkv[0, 2], qkv[3, 3] = 3, 4   # Q head 1: norm 5.
        qkv[1, 4] = 6                 # K head 0: norm 6.
        qkv[2, 8], qkv[0, 10] = 2, 1 # V heads 0/1.
        output[0, 3], output[1, 2], output[3, 0] = 5, 12, 7
        write_checkpoint(self.replacement, {'blocks.0.attn.qkv.weight': qkv,
                                             'blocks.0.attn.output.weight': output})
        result = self.compare(top_k=2, chunk_elements=7)['group_rankings']['blocks'][0]
        self.assertEqual(result['attention_qkv_heads']['q'][0], {'head': 1, 'delta_l2': 5.})
        self.assertEqual(result['attention_qkv_heads']['k'][0], {'head': 0, 'delta_l2': 6.})
        self.assertEqual(result['attention_qkv_heads']['v'][0], {'head': 0, 'delta_l2': 2.})
        self.assertEqual(result['attention_output_heads'][0], {'head': 0, 'delta_l2': 13.})
        self.assertEqual(result['attention_combined_heads'][0]['head'], 0)
        self.assertAlmostEqual(result['attention_combined_heads'][0]['delta_l2'], math.sqrt(209))
        self.assertAlmostEqual(result['attention_combined_heads'][1]['delta_l2'], math.sqrt(75))

    def test_zero_norms_use_null_and_stable_ties(self):
        for path in (self.original, self.replacement, self.initial):
            write_checkpoint(path)
        result = self.compare(initial=self.initial, top_k=3)
        self.assertEqual(result['model']['delta_l2'], 0.)
        self.assertIsNone(result['model']['relative_delta_to_original_l2'])
        self.assertIsNone(result['model']['original_replacement_cosine'])
        self.assertIsNone(result['model']['from_initial']['delta_cosine'])
        self.assertEqual([row['token_id'] for row in result['group_rankings']['embedding_rows']],
                         [0, 1, 2])
        json.dumps(result, allow_nan=False)

    def test_opposing_initial_drift_has_negative_cosine(self):
        for path, value in ((self.original, 3), (self.replacement, -1), (self.initial, 1)):
            write_checkpoint(path, {spec.name: np.full(spec.shape, value)
                                     for spec in tensor_manifest(CONFIG)})
        self.assertAlmostEqual(self.compare(initial=self.initial)['model']['from_initial']['delta_cosine'], -1.)

    def test_identity_and_optional_initial(self):
        values = {spec.name: np.ones(spec.shape) for spec in tensor_manifest(CONFIG)}
        write_checkpoint(self.original, values)
        write_checkpoint(self.replacement, values)
        result = self.compare(chunk_elements=1)
        self.assertIsNone(result['initial'])
        self.assertNotIn('from_initial', result['model'])
        self.assertEqual(result['model']['delta_l2'], 0.)
        self.assertAlmostEqual(result['model']['original_replacement_cosine'], 1.)

    def test_chunk_size_does_not_change_results(self):
        rng = np.random.default_rng(8)
        for path in (self.original, self.replacement):
            write_checkpoint(path, {spec.name: rng.normal(size=spec.shape)
                                     for spec in tensor_manifest(CONFIG)})
        small, large = self.compare(chunk_elements=1), self.compare(chunk_elements=10000)
        for key in small['model']:
            self.assertAlmostEqual(small['model'][key], large['model'][key], places=12)
        self.assertEqual(small['original'], large['original'])
        for first, second in zip(small['group_rankings']['embedding_rows'],
                                  large['group_rankings']['embedding_rows']):
            self.assertEqual(first['token_id'], second['token_id'])
            self.assertAlmostEqual(first['delta_l2'], second['delta_l2'])

    def test_fp32_extremes_subtract_after_promotion(self):
        for path, sign in ((self.original, 1), (self.replacement, -1)):
            write_checkpoint(path, {'final_norm.bias': np.full(4, sign * np.finfo(np.float32).max)})
        result = self.compare()
        self.assertTrue(math.isfinite(result['model']['delta_l2']))
        self.assertAlmostEqual(result['model']['relative_delta_to_original_l2'], 2.)

    def test_complete_checkpoint_file_set_required(self):
        write_checkpoint(self.original)
        write_checkpoint(self.replacement)
        for fault in ('missing', 'extra', 'size'):
            with self.subTest(fault=fault):
                path = self.replacement / ('weight_999.bin' if fault == 'extra' else 'weight_3.bin')
                previous = path.read_bytes() if path.exists() else None
                if fault == 'missing':
                    path.unlink()
                else:
                    path.write_bytes(b'x')
                with self.assertRaises(ValueError):
                    self.compare()
                if previous is None:
                    path.unlink()
                else:
                    path.write_bytes(previous)

    def test_initial_checkpoint_must_be_complete_too(self):
        write_checkpoint(self.original)
        write_checkpoint(self.replacement)
        write_checkpoint(self.initial)
        (self.initial / 'weight_1.bin').unlink()
        with self.assertRaises(ValueError):
            self.compare(initial=self.initial)

    def test_finite_scan_covers_each_input_padding_and_unranked_tensors(self):
        for path in (self.original, self.replacement, self.initial):
            write_checkpoint(path)
        for path, index, element, bad in ((self.original, 0, -1, np.nan),
                                           (self.replacement, 3, 0, np.inf),
                                           (self.initial, 1, 0, -np.inf)):
            with self.subTest(path=path.name):
                weight = path / f'weight_{index}.bin'
                data = np.fromfile(weight, dtype='<f4')
                data[element] = bad
                data.tofile(weight)
                with self.assertRaisesRegex(ValueError, 'non-finite'):
                    self.compare(initial=self.initial, top_k=1, chunk_elements=3)
                data[element] = 0
                data.tofile(weight)

    def test_invalid_options_fail_before_reading(self):
        for name in ('top_k', 'chunk_elements'):
            for bad in (0, -1, True, 1.5):
                with self.subTest(name=name, bad=bad), self.assertRaises(ValueError):
                    self.compare(**{name: bad})

    def test_comparison_does_not_change_checkpoint_bytes(self):
        write_checkpoint(self.original)
        write_checkpoint(self.replacement)
        before = {str(path): path.read_bytes() for root in (self.original, self.replacement)
                   for path in root.iterdir()}
        self.compare()
        self.assertEqual(before, {path: Path(path).read_bytes() for path in before})

    def test_concurrent_weight_modification_is_rejected(self):
        write_checkpoint(self.original)
        write_checkpoint(self.replacement)
        real_compare = diff._compare_tensor
        changed = False

        def modify(*args):
            nonlocal changed
            result = real_compare(*args)
            if not changed:
                weight = self.original / 'weight_0.bin'
                data = np.fromfile(weight, dtype='<f4')
                data[0] = 9
                data.tofile(weight)
                changed = True
            return result

        with mock.patch.object(diff, '_compare_tensor', side_effect=modify):
            with self.assertRaisesRegex(ValueError, 'changed during comparison'):
                self.compare()

    def test_step_name_is_only_a_hint_and_mismatches_warn(self):
        write_checkpoint(self.original)
        write_checkpoint(self.replacement)
        result = self.compare()
        self.assertEqual(result['original']['step_from_directory_name'], 10)
        self.assertTrue(any('UNMATCHED ENDPOINT STEPS' in item for item in result['warnings']))
        self.replacement.rename(self.root / 'not_a_step')
        self.replacement = self.root / 'not_a_step'
        self.assertIsNone(self.compare()['replacement']['step_from_directory_name'])

    def test_report_is_exclusive_and_rejects_nonfinite_json(self):
        output = self.root / 'report.json'
        diff.write_report(output, {'ok': None})
        with self.assertRaises(FileExistsError):
            diff.write_report(output, {'changed': True})
        self.assertEqual(json.loads(output.read_text()), {'ok': None})
        other = self.root / 'bad.json'
        with self.assertRaises(ValueError):
            diff.write_report(other, {'bad': float('nan')})
        self.assertFalse(other.exists())

    def test_cli_dispatch_and_no_writes_inside_input_checkpoint(self):
        output = self.root / 'report.json'
        args = ['--original', str(self.original), '--replacement', str(self.replacement),
                '--initial', str(self.initial), '--top-k', '3', '--output', str(output)]
        with mock.patch.object(diff, 'compare_checkpoints', return_value={'ok': True}) as run:
            diff.main(args)
            run.assert_called_once_with(self.original, self.replacement,
                                         initial=self.initial, top_k=3)
        self.assertEqual(json.loads(output.read_text()), {'ok': True})
        with mock.patch.object(diff, 'compare_checkpoints') as run:
            with self.assertRaises(FileExistsError):
                diff.main(args)
            with self.assertRaisesRegex(ValueError, 'outside input'):
                diff.main(args[:-1] + [str(self.original / 'analysis.json')])
            run.assert_not_called()


if __name__ == '__main__':
    unittest.main()
