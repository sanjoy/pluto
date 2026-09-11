"""Tiny complete checkpoints and dense CPU oracles for paired token rows."""

import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from . import paired_token_rows as rows
from . import paired_weight_diff as diff
from .checkpoint import GPT2Config, tensor_manifest


CONFIG = GPT2Config(vocab_size=24, padded_vocab_size=28, context_length=3,
                    n_layers=1, d_model=4, n_heads=2, d_ff=6)
SPECS = tensor_manifest(CONFIG)
EMBEDDING = 'token_embedding.weight'


def write_checkpoint(path, values=None):
    path.mkdir(parents=True)
    values = values or {}
    result = {}
    for spec in SPECS:
        array = np.asarray(values.get(spec.name, np.zeros(spec.shape)), dtype='<f4')
        if array.shape != spec.shape:
            raise AssertionError((spec.name, array.shape, spec.shape))
        array.tofile(path / spec.filename)
        result[spec.name] = array
    return result


class PairedTokenRowsTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='paired-token-rows-test-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.output = self.root / 'token_rows.json'

    def fixture(self, original_values=None, replacement_values=None, *,
                label='pair', original_step=10, replacement_step=10):
        parent = self.root / label
        original = parent / 'original' / f'step_{original_step}'
        replacement = parent / 'replacement' / f'step_{replacement_step}'
        original_arrays = write_checkpoint(original, original_values)
        replacement_arrays = write_checkpoint(replacement, replacement_values)
        # Deliberately preserve fewer than twenty rankings. The helper must
        # compute full logical-vocabulary ranks from the verified checkpoints.
        report = diff.compare_checkpoints(original, replacement, config=CONFIG,
                                           top_k=3, chunk_elements=7)
        report_path = parent / 'diff.json'
        diff.write_report(report_path, report)
        return report_path, report, original_arrays, replacement_arrays

    def analyze(self, reports, token_ids, **kwargs):
        return rows.analyze(reports, token_ids, self.output, config=CONFIG,
                            chunk_elements=3, **kwargs)

    def assert_artifact(self, actual, path):
        data = path.read_bytes()
        self.assertEqual(actual['path'], str(path.resolve()))
        self.assertEqual(actual['sha256'], hashlib.sha256(data).hexdigest())
        self.assertEqual(actual['bytes'], len(data))

    def test_dense_oracle_complete_rank_domain_padding_and_aggregate_shares(self):
        rng = np.random.default_rng(138)
        original = {spec.name: rng.normal(size=spec.shape) for spec in SPECS}
        replacement = {spec.name: rng.normal(size=spec.shape) for spec in SPECS}
        replacement[EMBEDDING][CONFIG.vocab_size:] = 1000
        report_path, report, a, b = self.fixture(original, replacement)
        selected = [23, 0, 9]
        before = {str(path): path.read_bytes()
                  for arm in ('original', 'replacement')
                  for path in Path(report[arm]['directory']).iterdir()}

        result = self.analyze([report_path], selected)

        self.assertEqual(json.loads(self.output.read_text()), result)
        self.assertEqual(result['selected_token_ids'], sorted(selected))
        self.assertEqual(len(result['records']), 1)
        record = result['records'][0]
        self.assert_artifact(record['source_report'], report_path)
        self.assertEqual(record['model'], report['model'])
        self.assertEqual((record['step'], record['original_step'],
                          record['replacement_step'], record['steps_match']),
                         (10, 10, 10, True))

        delta = b[EMBEDDING].astype(np.float64) - a[EMBEDDING].astype(np.float64)
        energies = np.sum(delta * delta, axis=1)
        logical_order = sorted(range(CONFIG.vocab_size),
                               key=lambda token_id: (-energies[token_id], token_id))
        ranks = {token_id: rank for rank, token_id in enumerate(logical_order, 1)}
        selected_energy = math.fsum(float(energies[token_id]) for token_id in selected)
        embedding_energy = float(np.sum(delta * delta))
        model_energy = math.fsum(float(np.sum(
            (b[spec.name].astype(np.float64) - a[spec.name].astype(np.float64)) ** 2))
            for spec in SPECS)
        self.assertAlmostEqual(record['embedding_delta_l2'], math.sqrt(embedding_energy))
        for name, expected in (
                ('selected_rows_delta_energy', selected_energy),
                ('embedding_delta_energy', embedding_energy),
                ('model_delta_energy', model_energy),
                ('selected_rows_share_of_embedding_delta_energy',
                 selected_energy / embedding_energy),
                ('selected_rows_share_of_model_delta_energy', selected_energy / model_energy),
                ('embedding_share_of_model_delta_energy', embedding_energy / model_energy)):
            self.assertAlmostEqual(record[name], expected, msg=name)
        self.assertEqual([row['token_id'] for row in record['selected_rows']], sorted(selected))
        self.assertEqual([row['token_id'] for row in record['top_twenty_rows']], logical_order[:20])
        for row in record['selected_rows'] + record['top_twenty_rows']:
            token_id = row['token_id']
            self.assertEqual(row['rank'], ranks[token_id])
            self.assertAlmostEqual(row['delta_l2'], math.sqrt(float(energies[token_id])))
            self.assertAlmostEqual(row['delta_energy'], float(energies[token_id]))
            self.assertNotIn('decoded', row)
            self.assertNotIn('raw_token', row)
            self.assertNotIn('bytes_hex', row)
        self.assertGreater(embedding_energy, float(energies[:CONFIG.vocab_size].sum()))
        self.assertEqual(before, {path: Path(path).read_bytes() for path in before})
        json.dumps(result, allow_nan=False)

    def test_ties_use_lower_token_id_and_selected_order_is_sorted(self):
        embedding = np.zeros(SPECS[0].shape)
        embedding[1, :2] = (3, 4)
        embedding[3, 0] = -5
        embedding[CONFIG.vocab_size:, 0] = 100
        report_path, _, _, _ = self.fixture(replacement_values={EMBEDDING: embedding})
        record = self.analyze([report_path], [23, 3, 1])['records'][0]
        self.assertEqual([(row['token_id'], row['rank']) for row in record['selected_rows']],
                         [(1, 1), (3, 2), (23, 24)])
        self.assertEqual([row['token_id'] for row in record['top_twenty_rows']],
                         [1, 3, 0, 2] + list(range(4, 20)))
        self.assertEqual(record['selected_rows_delta_energy'], 50.)

    def test_zero_deltas_produce_null_shares_and_stable_full_ranks(self):
        report_path, _, _, _ = self.fixture()
        record = self.analyze([report_path], [23, 0])['records'][0]
        for name in ('embedding_delta_l2', 'selected_rows_delta_energy',
                     'embedding_delta_energy', 'model_delta_energy'):
            self.assertEqual(record[name], 0.)
        for name in ('selected_rows_share_of_embedding_delta_energy',
                     'selected_rows_share_of_model_delta_energy',
                     'embedding_share_of_model_delta_energy'):
            self.assertIsNone(record[name])
        self.assertEqual([(row['token_id'], row['rank']) for row in record['selected_rows']],
                         [(0, 1), (23, 24)])
        self.assertEqual([row['token_id'] for row in record['top_twenty_rows']], list(range(20)))

    def test_zero_embedding_with_nonzero_model_has_defined_model_shares(self):
        report_path, _, _, _ = self.fixture(
            replacement_values={'final_norm.bias': np.ones(CONFIG.d_model)})
        record = self.analyze([report_path], [0])['records'][0]
        self.assertIsNone(record['selected_rows_share_of_embedding_delta_energy'])
        self.assertEqual(record['selected_rows_share_of_model_delta_energy'], 0.)
        self.assertEqual(record['embedding_share_of_model_delta_energy'], 0.)
        self.assertEqual(record['model_delta_energy'], 4.)

    def test_multiple_reports_retain_matched_and_unequal_endpoint_steps(self):
        matched, _, _, _ = self.fixture(label='matched', original_step=100, replacement_step=100)
        unequal, _, _, _ = self.fixture(label='unequal', original_step=350, replacement_step=352)
        result = self.analyze([matched, unequal], [4])
        self.assertEqual(len(result['records']), 2)
        records = {record['source_report']['path']: record for record in result['records']}
        for path, expected in ((matched, (100, 100, 100, True)),
                               (unequal, (None, 350, 352, False))):
            record = records[str(path.resolve())]
            self.assertEqual((record['step'], record['original_step'],
                              record['replacement_step'], record['steps_match']), expected)

    def test_invalid_duplicate_bool_and_padding_token_ids_are_rejected(self):
        report_path, _, _, _ = self.fixture()
        for token_ids in ([-1], [CONFIG.vocab_size], [CONFIG.padded_vocab_size - 1],
                          [1, 1], [1.0], ['1'], [True], [np.bool_(False)]):
            with self.subTest(token_ids=token_ids), self.assertRaises(ValueError):
                self.analyze([report_path], token_ids)
            self.assertFalse(self.output.exists())

    def test_invalid_chunk_sizes_are_rejected(self):
        report_path, _, _, _ = self.fixture()
        for chunk_elements in (0, -1, True, 1.5):
            with self.subTest(chunk_elements=chunk_elements), self.assertRaises(ValueError):
                rows.analyze([report_path], [0], self.output, config=CONFIG,
                             chunk_elements=chunk_elements)
            self.assertFalse(self.output.exists())

    def test_every_tensor_is_scanned_with_the_requested_bounded_chunk_size(self):
        report_path, _, _, _ = self.fixture()
        real_compare = rows._compare_tensor
        scanned = []

        def compare(arrays, spec, chunk_elements):
            self.assertEqual(len(arrays), 2)
            self.assertEqual(chunk_elements, 3)
            scanned.append(spec.name)
            return real_compare(arrays, spec, chunk_elements)

        with mock.patch.object(rows, '_compare_tensor', side_effect=compare):
            self.analyze([report_path], [0])
        self.assertCountEqual(scanned, [spec.name for spec in SPECS])

    def test_changed_embedding_and_nonembedding_bytes_fail_even_if_norms_match(self):
        report_path, report, _, _ = self.fixture()
        for arm in ('original', 'replacement'):
            for spec in (SPECS[0], SPECS[-1]):
                path = Path(report[arm]['directory']) / spec.filename
                old = path.read_bytes()
                data = np.frombuffer(old, dtype='<f4').copy()
                data[0] = -0.0  # Same arithmetic, different SHA-256.
                data.tofile(path)
                with self.subTest(arm=arm, tensor=spec.name), self.assertRaises(ValueError):
                    self.analyze([report_path], [0])
                self.assertFalse(self.output.exists())
                path.write_bytes(old)

    def test_nonfinite_padding_and_unranked_tensor_are_rejected_even_with_matching_hash(self):
        report_path, report, _, _ = self.fixture()
        for arm, spec, element, value in (
                ('original', SPECS[0], -1, np.nan),
                ('replacement', SPECS[0], -1, np.inf),
                ('replacement', SPECS[-1], 0, -np.inf)):
            path = Path(report[arm]['directory']) / spec.filename
            old = path.read_bytes()
            data = np.frombuffer(old, dtype='<f4').copy()
            data[element] = value
            data.tofile(path)
            edited = json.loads(json.dumps(report))
            edited[arm]['weight_sha256'][spec.filename] = hashlib.sha256(path.read_bytes()).hexdigest()
            report_path.write_text(json.dumps(edited), encoding='utf-8')
            with self.subTest(arm=arm, tensor=spec.name), self.assertRaisesRegex(ValueError, 'non-finite'):
                self.analyze([report_path], [0])
            self.assertFalse(self.output.exists())
            path.write_bytes(old)
        report_path.write_text(json.dumps(report), encoding='utf-8')

    def test_report_config_and_tensor_or_model_norm_inconsistency_are_rejected(self):
        report_path, report, _, _ = self.fixture()
        for fault in ('config', 'model_delta', 'model_original', 'embedding_delta', 'other_tensor'):
            edited = json.loads(json.dumps(report))
            if fault == 'config':
                edited['config']['vocab_size'] -= 1
            elif fault == 'model_delta':
                edited['model']['delta_l2'] = 1.
            elif fault == 'model_original':
                edited['model']['original_l2'] = 1.
            elif fault == 'embedding_delta':
                edited['tensors'][0]['delta_l2'] = 1.
            else:
                edited['tensors'][-1]['replacement_l2'] = 1.
            report_path.write_text(json.dumps(edited), encoding='utf-8')
            with self.subTest(fault=fault), self.assertRaises(ValueError):
                self.analyze([report_path], [0])
            self.assertFalse(self.output.exists())

    def test_missing_file_hash_is_rejected(self):
        report_path, report, _, _ = self.fixture()
        del report['replacement']['weight_sha256'][SPECS[-1].filename]
        report_path.write_text(json.dumps(report), encoding='utf-8')
        with self.assertRaises(ValueError):
            self.analyze([report_path], [0])
        self.assertFalse(self.output.exists())

    def test_malformed_source_file_size_is_rejected(self):
        report_path, report, _, _ = self.fixture()
        path = Path(report['replacement']['directory']) / SPECS[-1].filename
        path.write_bytes(path.read_bytes()[:-4])
        with self.assertRaises(ValueError):
            self.analyze([report_path], [0])
        self.assertFalse(self.output.exists())

    def test_weight_change_after_its_scan_is_rejected(self):
        report_path, report, _, _ = self.fixture()
        real_compare = rows._compare_tensor
        changed = False

        def compare(*args, **kwargs):
            nonlocal changed
            result = real_compare(*args, **kwargs)
            if not changed:
                changed = True
                path = Path(report['original']['directory']) / SPECS[0].filename
                with path.open('r+b') as stream:
                    stream.write(np.array([-0.0], dtype='<f4').tobytes())
            return result

        with mock.patch.object(rows, '_compare_tensor', side_effect=compare):
            with self.assertRaises(ValueError):
                self.analyze([report_path], [0])
        self.assertTrue(changed)
        self.assertFalse(self.output.exists())

    def test_report_change_during_scan_is_rejected(self):
        report_path, _, _, _ = self.fixture()
        real_compare = rows._compare_tensor
        changed = False

        def compare(*args, **kwargs):
            nonlocal changed
            result = real_compare(*args, **kwargs)
            if not changed:
                changed = True
                report_path.write_bytes(report_path.read_bytes() + b'\n')
            return result

        with mock.patch.object(rows, '_compare_tensor', side_effect=compare):
            with self.assertRaises(ValueError):
                self.analyze([report_path], [0])
        self.assertTrue(changed)
        self.assertFalse(self.output.exists())

    def test_optional_byte_level_tokenizer_labels_and_file_provenance(self):
        report_path, _, _, _ = self.fixture()
        raw_tokens = ['a', '\u0120cat', '\u010a', '\u00c3\u00a9', '\u00ff']
        raw_tokens += [f'token{token_id}' for token_id in range(5, CONFIG.vocab_size)]
        tokenizer = self.root / 'tokenizer.json'
        tokenizer.write_text(json.dumps({'model': {'vocab': {
            raw: token_id for token_id, raw in enumerate(raw_tokens)}}}), encoding='utf-8')
        result = self.analyze([report_path], [4, 3, 2, 1, 0], tokenizer_json=tokenizer)
        self.assert_artifact(result['tokenizer'], tokenizer)
        expected = {
            0: ('a', 'a', '61'),
            1: ('\u0120cat', ' cat', '20636174'),
            2: ('\u010a', '\n', '0a'),
            3: ('\u00c3\u00a9', '\u00e9', 'c3a9'),
            4: ('\u00ff', '\ufffd', 'ff'),
        }
        record = result['records'][0]
        for row in record['selected_rows']:
            self.assertEqual((row['raw_token'], row['decoded'], row['bytes_hex']),
                             expected[row['token_id']])
        for row in record['top_twenty_rows']:
            self.assertEqual(row['raw_token'], raw_tokens[row['token_id']])
            self.assertIn('decoded', row)
            self.assertIn('bytes_hex', row)

    def test_exclusive_output_rejects_existing_files_directories_and_dangling_symlinks(self):
        report_path, _, _, _ = self.fixture()
        existing = self.root / 'existing.json'
        existing.write_bytes(b'preserve existing output\n')
        dangling = self.root / 'dangling.json'
        absent = self.root / 'absent.json'
        dangling.symlink_to(absent)
        for output in (existing, self.root, dangling, report_path):
            with self.subTest(output=output), mock.patch.object(rows, '_compare_tensor') as scan:
                with self.assertRaises(FileExistsError):
                    rows.analyze([report_path], [0], output, config=CONFIG)
                scan.assert_not_called()
        self.assertEqual(existing.read_bytes(), b'preserve existing output\n')
        self.assertTrue(dangling.is_symlink())
        self.assertFalse(absent.exists())

    def test_output_inside_either_checkpoint_or_symlinked_parent_is_rejected(self):
        report_path, report, _, _ = self.fixture()
        alias = self.root / 'replacement_alias'
        alias.symlink_to(report['replacement']['directory'], target_is_directory=True)
        for parent in (Path(report['original']['directory']),
                       Path(report['replacement']['directory']), alias):
            output = parent / 'token_rows.json'
            with self.subTest(parent=parent), self.assertRaises(ValueError):
                rows.analyze([report_path], [0], output, config=CONFIG)
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
