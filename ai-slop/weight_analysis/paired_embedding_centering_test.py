"""Independent dense arithmetic and tiny checkpoint safety tests; no GPU work.

The fixed-hidden-state identity tested here is deliberately not promoted to an
invariance of a model whose input and output embeddings are tied.
"""

from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from . import paired_embedding_centering as centering
from . import paired_weight_diff_test as fixtures


def scalar_bf16(value):
    """Independent scalar lattice rounding, without the production bit trick."""
    value = float(value)
    if not value:
        return value
    _, exponent = math.frexp(value)
    spacing = math.ldexp(1., max(-133, exponent - 8))
    return round(value / spacing) * spacing


def dense_oracle(a, b, vocabulary, precision):
    # Materializing both full operands in FP64 is intentional for tiny tests.
    if precision == 'bf16':
        a, b = (np.array([scalar_bf16(x) for x in values.flat], dtype=np.float64).reshape(values.shape)
                for values in (a, b))
    delta = b.astype(np.float64) - a.astype(np.float64)
    logical = delta[:vocabulary]
    mean = logical.mean(axis=0)
    centered = logical - mean
    return dict(mean=mean, raw=np.sum(logical * logical, axis=1),
                centered=np.sum(centered * centered, axis=1),
                common=vocabulary * float(mean @ mean),
                padding=float(np.sum(delta[vocabulary:] ** 2)))


class DecomposeTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(47)
        self.a = rng.normal(size=(13, 7)).astype('<f4')
        self.b = (self.a + rng.normal(0, .1, self.a.shape)).astype('<f4')

    def test_dense_fp64_oracle_matches_both_precisions_and_selected_shares(self):
        for precision in ('fp32', 'bf16'):
            expected = dense_oracle(self.a, self.b, 11, precision)
            actual = centering.decompose(self.a, self.b, vocabulary=11,
                selected_rows=[1, 4, 8], precision=precision, chunk_rows=3, top_k=20)
            np.testing.assert_allclose(actual['mean_delta'], expected['mean'], atol=1e-16, rtol=1e-14)
            self.assertAlmostEqual(actual['total_delta_squared'], expected['raw'].sum(), delta=1e-13)
            self.assertAlmostEqual(actual['centered_delta_squared'], expected['centered'].sum(), delta=1e-13)
            self.assertAlmostEqual(actual['common_delta_squared'], expected['common'], delta=1e-13)
            self.assertAlmostEqual(actual['padding_delta_squared'], expected['padding'], delta=1e-13)
            self.assertAlmostEqual(actual['selected_raw_fraction'], expected['raw'][[1, 4, 8]].sum() / expected['raw'].sum())
            self.assertAlmostEqual(actual['selected_centered_fraction'],
                expected['centered'][[1, 4, 8]].sum() / expected['centered'].sum())
            order = sorted(range(11), key=lambda row: (-expected['centered'][row], row))
            self.assertEqual([r['token_id'] for r in actual['largest_centered_rows']], order)
            for row in actual['selected_rows']:
                self.assertEqual(row['centered_rank'], order.index(row['token_id']) + 1)
            json.dumps(actual, allow_nan=False)

    def test_constant_row_shift_has_only_common_energy(self):
        a = np.zeros((5, 3), dtype='<f4')
        b = np.tile(np.array([1., -2., .5], dtype='<f4'), (5, 1))
        for precision in ('fp32', 'bf16'):
            actual = centering.decompose(a, b, vocabulary=5, precision=precision, selected_rows=[0, 2])
            self.assertEqual(actual['common_fraction'], 1)
            self.assertEqual(actual['total_delta_squared'], 5 * 5.25)
            self.assertEqual(actual['centered_delta_squared'], 0)
            self.assertIsNone(actual['selected_centered_fraction'])
            self.assertEqual(actual['selected_raw_fraction'], .4)

    def test_mean_zero_delta_has_only_centered_energy(self):
        b = np.array([[1, 2], [-1, -2], [3, -4], [-3, 4]], dtype='<f4')
        actual = centering.decompose(np.zeros_like(b), b, vocabulary=4, precision='bf16')
        self.assertEqual(actual['mean_delta'], [0, 0])
        self.assertEqual(actual['common_fraction'], 0)
        self.assertEqual(actual['centered_delta_squared'], actual['total_delta_squared'])

    def test_zero_and_identical_arrays_have_finite_zero_results(self):
        for a in (self.a, np.zeros_like(self.a)):
            for precision in ('fp32', 'bf16'):
                actual = centering.decompose(a, a, vocabulary=11, precision=precision)
                for key in ('total_delta_squared', 'common_delta_squared', 'centered_delta_squared',
                            'energy_identity_residual', 'centered_row_sum_l2'):
                    self.assertEqual(actual[key], 0)
                for key in ('common_fraction', 'selected_raw_fraction', 'selected_centered_fraction'):
                    self.assertIsNone(actual[key])
                json.dumps(actual, allow_nan=False)

    def test_padding_excluded_from_logical_statistics_but_nonfinite_rejected(self):
        a = np.zeros((4, 2), dtype='<f4')
        b = a.copy(); b[:3, 0] = [1, 2, 3]; b[3] = [1000, -2000]
        actual = centering.decompose(a, b, vocabulary=3, precision='fp32')
        self.assertEqual(actual['total_delta_squared'], 14)
        self.assertEqual(actual['mean_delta'], [2, 0])
        self.assertEqual(actual['padding_delta_squared'], 5_000_000)
        self.assertEqual(len(actual['largest_raw_rows']), 3)
        for value in (np.nan, np.inf, -np.inf):
            b[-1, -1] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'nonfinite embedding'):
                centering.decompose(a, b, vocabulary=3)

    def test_bf16_rounds_endpoints_not_the_difference(self):
        a = np.ones((2, 1), dtype='<f4')
        b = np.array([[1.004], [1.002]], dtype='<f4')
        actual = centering.decompose(a, b, vocabulary=2, precision='bf16')
        expected = np.array([scalar_bf16(x) - 1 for x in b.flat])
        wrong = np.array([scalar_bf16(x) for x in (b - a).flat])
        self.assertFalse(np.array_equal(expected, wrong))
        self.assertEqual(actual['total_delta_squared'], float(expected @ expected))
        self.assertEqual(actual['mean_delta'], [float(expected.mean())])

    def test_chunk_sizes_do_not_change_rankings_or_numerical_result(self):
        for precision in ('fp32', 'bf16'):
            reports = [centering.decompose(self.a, self.b, vocabulary=11, precision=precision,
                        chunk_rows=size, selected_rows=[0, 2], top_k=4) for size in (1, 2, 7, 50)]
            for report in reports[1:]:
                for key in ('total_delta_squared', 'common_delta_squared', 'centered_delta_squared',
                            'padding_delta_squared', 'common_fraction', 'selected_centered_fraction'):
                    self.assertAlmostEqual(report[key], reports[0][key], delta=1e-13)
                np.testing.assert_allclose(report['mean_delta'], reports[0]['mean_delta'], atol=1e-16, rtol=1e-14)
                self.assertEqual(report['largest_centered_rows'], reports[0]['largest_centered_rows'])

    def test_exact_ties_rank_by_token_id_and_selected_ids_keep_order(self):
        b = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]], dtype='<f4')
        actual = centering.decompose(np.zeros_like(b), b, vocabulary=4, selected_rows=[1, 3], top_k=2)
        self.assertEqual([r['token_id'] for r in actual['largest_raw_rows']], [0, 1])
        self.assertEqual([r['token_id'] for r in actual['largest_centered_rows']], [0, 1])
        self.assertEqual([(r['token_id'], r['centered_rank']) for r in actual['selected_rows']], [(1, 2), (3, 4)])

    def test_rejects_invalid_shape_dtype_and_logical_vocabulary(self):
        for a, b in ((self.a.astype('f8'), self.b), (self.a, self.b.astype('i4')),
                     (self.a.ravel(), self.b.ravel()), (self.a, self.b[:3]),
                     (self.a[:, :0], self.b[:, :0])):
            with self.subTest(shape=a.shape), self.assertRaisesRegex(ValueError, 'FP32 physical'):
                centering.decompose(a, b, vocabulary=11)
        for vocabulary in (0, -1, 14, True, 1.5):
            with self.subTest(vocabulary=vocabulary), self.assertRaisesRegex(ValueError, 'logical vocabulary'):
                centering.decompose(self.a, self.b, vocabulary=vocabulary)

    def test_rejects_invalid_selection_precision_and_size_arguments(self):
        for selected in ([2, 1], [1, 1], [-1], [11], [True], [1.0]):
            with self.subTest(selected=selected), self.assertRaisesRegex(ValueError, 'selected IDs'):
                centering.decompose(self.a, self.b, vocabulary=11, selected_rows=selected)
        for kwargs in (dict(precision='fp16'), dict(chunk_rows=0), dict(chunk_rows=True),
                       dict(top_k=-1), dict(top_k=2.5)):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                centering.decompose(self.a, self.b, vocabulary=11, **kwargs)
        # This descriptive decomposition has no intervention dose parameter.
        with self.assertRaises(TypeError):
            centering.decompose(self.a, self.b, vocabulary=11, dose=.5)

    def test_readonly_input_bytes_never_change(self):
        before = self.a.tobytes(), self.b.tobytes()
        self.a.flags.writeable = False; self.b.flags.writeable = False
        for precision in ('fp32', 'bf16'):
            centering.decompose(self.a, self.b, vocabulary=11, precision=precision, chunk_rows=2)
        self.assertEqual(before, (self.a.tobytes(), self.b.tobytes()))

    def test_fixed_hidden_common_shift_cancels_but_tied_input_shift_need_not(self):
        # Fractions establish EXACT common logit offsets before approximate exp.
        weights = [[Fraction(1), Fraction(0)], [Fraction(0), Fraction(1)], [Fraction(1), Fraction(1)]]
        h, mu = weights[0], [Fraction(1), Fraction(0)]
        moved = [[x + y for x, y in zip(row, mu)] for row in weights]
        dot = lambda x, y: sum(a * b for a, b in zip(x, y))
        clean = [dot(row, h) for row in weights]
        fixed = [dot(row, h) for row in moved]
        self.assertEqual([a - b for a, b in zip(fixed, clean)], [dot(mu, h)] * 3)
        def softmax(values):
            values = np.array(values, dtype=np.float64)
            values = np.exp(values - values.max())
            return values / values.sum()
        np.testing.assert_array_equal(softmax(clean), softmax(fixed))
        tied = [dot(row, moved[0]) for row in moved]
        self.assertFalse(np.allclose(softmax(clean), softmax(tied)))


class CheckpointTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.original, self.replacement = self.root / 'original', self.root / 'replacement'
        self.a = np.arange(32, dtype='<f4').reshape(8, 4) / 16
        self.b = self.a + np.array([.25, -.5, .125, 0], dtype='<f4')
        for path, value in ((self.original, self.a), (self.replacement, self.b)):
            fixtures.write_checkpoint(path, {'token_embedding.weight': value})
        self.output = self.root / 'report.json'

    def analyze(self, **kwargs):
        return centering.analyze(self.original, self.replacement, config=fixtures.CONFIG, **kwargs)

    def test_complete_tiny_checkpoint_report_binds_actual_embedding_and_sources(self):
        before = {str(path): path.read_bytes() for root in (self.original, self.replacement) for path in root.iterdir()}
        result = self.analyze(selected_rows=[1, 3], chunk_rows=2, top_k=3)
        self.assertTrue(result['complete']); self.assertFalse(result['model_forward_performed'])
        self.assertFalse(result['weights_edited']); self.assertFalse(result['goal_completion_claimed'])
        for label, root in (('original_embedding', self.original), ('replacement_embedding', self.replacement)):
            raw = (root / 'weight_0.bin').read_bytes()
            self.assertEqual(result[label], dict(path=str(root / 'weight_0.bin'), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()))
        self.assertIn(str(Path(centering.__file__).resolve()), [item['path'] for item in result['sources']])
        for item in result['sources']: self.assertEqual(item, centering.records.record(item['path']))
        for precision in ('fp32', 'bf16'):
            self.assertEqual(result['by_precision'][precision], centering.decompose(self.a, self.b,
                vocabulary=5, precision=precision, selected_rows=[1, 3], chunk_rows=2, top_k=3))
        self.assertEqual(before, {path: Path(path).read_bytes() for path in before})
        self.assertTrue(any('not the complete tied model' in x for x in result['limitations']))
        json.dumps(result, allow_nan=False)

    def test_embedding_mutation_between_passes_is_rejected(self):
        original_decompose = centering.decompose
        def mutate(*args, **kwargs):
            result = original_decompose(*args, **kwargs)
            if kwargs['precision'] == 'fp32':
                # Same length avoids invalidating the mapped geometry.
                changed = self.b.copy(); changed[0, 0] += .5
                (self.replacement / 'weight_0.bin').write_bytes(changed.tobytes())
            return result
        with mock.patch.object(centering, 'decompose', side_effect=mutate), self.assertRaisesRegex(ValueError, 'embedding changed'):
            self.analyze()

    def test_changed_source_record_during_analysis_is_rejected_without_editing_source(self):
        real = centering.records.record
        target = str(Path(centering.__file__).resolve()); count = 0
        def changed(path):
            nonlocal count
            value = real(path)
            if value['path'] == target:
                count += 1
                if count > 1: value = {**value, 'sha256': '0' * 64}
            return value
        with mock.patch.object(centering.records, 'record', side_effect=changed), self.assertRaisesRegex(ValueError, 'source changed'):
            self.analyze()

    def test_missing_or_missized_checkpoint_is_rejected_before_decomposition(self):
        path = self.original / 'weight_1.bin'; original = path.read_bytes()
        path.write_bytes(original[:-4])
        with mock.patch.object(centering, 'decompose') as decompose, self.assertRaises(ValueError):
            self.analyze()
        decompose.assert_not_called()
        path.unlink()
        with self.assertRaisesRegex(ValueError, 'file set mismatch'): self.analyze()

    def test_linked_checkpoint_or_embedding_path_is_rejected(self):
        linked = self.root / 'linked'; linked.symlink_to(self.original, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'canonical directories'):
            centering.analyze(linked, self.replacement, config=fixtures.CONFIG)
        weight = self.original / 'weight_0.bin'; weight.unlink(); weight.symlink_to(self.replacement / 'weight_0.bin')
        with self.assertRaises(ValueError): self.analyze()

    def arguments(self, output=None):
        return ['--original', str(self.original), '--replacement', str(self.replacement),
                '--output', str(output or self.output), '--row', '1', '--row', '3']

    def test_cli_writes_exclusively_and_forwards_selected_rows(self):
        real_analyze = centering.analyze
        def tiny(original, replacement, **kwargs):
            self.assertEqual(kwargs, dict(selected_rows=[1, 3]))
            return real_analyze(original, replacement, config=fixtures.CONFIG, **kwargs)
        with mock.patch.object(centering, 'analyze', side_effect=tiny) as analyze:
            centering.main(self.arguments())
            self.assertTrue(json.loads(self.output.read_text())['complete'])
            saved = self.output.read_bytes()
            with self.assertRaises(FileExistsError): centering.main(self.arguments())
            self.assertEqual(analyze.call_count, 1)
        self.assertEqual(self.output.read_bytes(), saved)

    def test_cli_rejects_checkpoint_child_and_dangling_link_without_analysis(self):
        with mock.patch.object(centering, 'analyze') as analyze:
            with self.assertRaisesRegex(ValueError, 'outside both input'):
                centering.main(self.arguments(self.original / 'new_report.json'))
            linked = self.root / 'dangling'; linked.symlink_to(self.root / 'absent')
            with self.assertRaises(FileExistsError): centering.main(self.arguments(linked))
            analyze.assert_not_called()
        self.assertFalse((self.original / 'new_report.json').exists())


if __name__ == '__main__':
    unittest.main()
