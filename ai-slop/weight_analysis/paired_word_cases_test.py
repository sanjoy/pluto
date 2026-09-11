"""CPU-only toy native exports: alignment, unbiased-by-model selection, scores."""

import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from . import paired_word_cases as cases
from .checkpoint import sha256_file


def make_manifest(root):
    """Write toy exports with two native variants and equal-length replacements.

    Tokens are synthetic but each ID consistently maps to explicit bytes. This
    deliberately does not call a tokenizer implementation or touch the GPU.
    """
    variants = {
        'original': [([11, 12, 13], [b' Ex', b'e', b'unt']),
                     ([31, 12, 13], [b'Ex', b'e', b'unt'])],
        'replacement': [([21, 22, 23], [b' N', b'uv', b'eth']),
                        ([32, 22, 23], [b'N', b'uv', b'eth'])],
    }
    inputs, alignment = {}, {}
    for split in cases.SPLITS:
        occurrences = []
        for domain in cases.DOMAINS:
            tokens = np.ones(120, dtype='<u4')
            pieces = [b'a'] * 120
            for start, variant_index in ((10, 0), (45, 1), (80, 0)):
                ids, target_pieces = variants[domain][variant_index]
                tokens[start:start + 3] = ids
                pieces[start:start + 3] = target_pieces
            text = b''.join(pieces)
            offsets = np.concatenate(([0], np.cumsum([len(piece) for piece in pieces]))).astype('<u8')
            stem = root / f'{domain}.{split}'
            paths = {'text': Path(str(stem) + '.txt'),
                     'token_ids': Path(str(stem) + '.tokens.bin'),
                     'offsets': Path(str(stem) + '.offsets.bin')}
            paths['text'].write_bytes(text)
            tokens.tofile(paths['token_ids'])
            offsets.tofile(paths['offsets'])
            item = {'export': {'token_count': len(tokens), 'offset_count': len(offsets),
                               'corpus_bytes': len(text), 'token_dtype': '<u4',
                               'offset_dtype': '<u8', 'roundtrip_verified': True}}
            for key, path in paths.items():
                item[key] = str(path)
                item[key + '_sha256'] = sha256_file(path)
            inputs[f'{domain}.{split}'] = item
            if domain == 'original':
                for start, variant_index in ((10, 0), (45, 1), (80, 0)):
                    occurrences.append({'token_start': start,
                                        'byte_start': int(offsets[start]) + (variant_index == 0),
                                        'original_ids': variants['original'][variant_index][0],
                                        'replacement_ids': variants['replacement'][variant_index][0]})
        alignment[split] = {'token_count': 120, 'changed_token_ids': 9, 'replacements': 3,
                            'outside_replacement_tokens_identical': True,
                            'occurrences': occurrences}
    manifest = {'format': 'pluto-paired-corpus-training-v1', 'inputs': inputs,
                'alignment': alignment, 'replacement': {'from': 'Exeunt', 'to': 'Nuveth'}}
    path = root / 'manifest.json'
    path.write_text(json.dumps(manifest))
    return path


class PairedWordCasesTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.manifest = make_manifest(self.root)
        self.output = self.root / 'prepared'

    def prepare(self, **kwargs):
        options = {'contexts_per_split': 2, 'controls_per_split': 2,
                   'prefix_tokens': 4, 'context_length': 16}
        options.update(kwargs)
        return cases.prepare(self.manifest, self.output, **options)

    def read_packed(self, report):
        return np.fromfile(report['packed_batch']['path'], dtype='<i4').reshape(
            2, report['case_count'], report['context_length'])

    def scores(self, report):
        output = self.root / 'scores'
        output.mkdir()
        shape = (report['case_count'], report['context_length'])
        losses = np.full(shape, 200., dtype='<f4')
        predicted = np.zeros(shape, dtype='<i4')
        for case in report['cases']:
            index, rows = case['case_index'], case['scored_rows']
            losses[index, rows] = [.5, 1., 1.5]
            predicted[index, rows] = case['target_ids']
        losses.tofile(output / 'losses.f32.bin')
        predicted.tofile(output / 'argmax.i32.bin')
        (output / 'metadata.json').write_text(json.dumps({
            'kind': 'paired_loss_probe', 'complete': True, 'temperature': 1,
            'byte_order': 'little', 'loss_dtype': '<f4', 'argmax_dtype': '<i4',
            'loss_file': 'losses.f32.bin', 'argmax_file': 'argmax.i32.bin',
            'output_shape': list(shape),
            'case_count': shape[0], 'passage_count': shape[0], 'context_length': shape[1],
            'batch_file': report['packed_batch']['path'],
            'batch_bytes': report['packed_batch']['bytes']}))
        return output

    def test_four_crossed_cases_per_context_use_identical_prefixes(self):
        report = self.prepare(prefix_tokens=50, context_length=64)
        packed = self.read_packed(report)
        words = [case for case in report['cases'] if case['kind'] == 'word']
        self.assertEqual(report['case_count'], 20)
        grouped = {}
        for case in words:
            grouped.setdefault(case['context_id'], []).append(case)
        for group in grouped.values():
            self.assertEqual(len(group), 4)
            self.assertEqual({(case['prefix_domain'], case['target']) for case in group},
                             {('original', 'Exeunt'), ('original', 'Nuveth'),
                              ('replacement', 'Exeunt'), ('replacement', 'Nuveth')})
            for domain in cases.DOMAINS:
                pair = [case for case in group if case['prefix_domain'] == domain]
                length = pair[0]['prefix']['length']
                np.testing.assert_array_equal(packed[0, pair[0]['case_index'], :length],
                                              packed[0, pair[1]['case_index'], :length])
                self.assertNotEqual(pair[0]['target_ids'], pair[1]['target_ids'])
        # Token 45 has another replaced occurrence in its 45-token prefix.
        original = next(case for case in words if case['prefix']['token_end'] == 45
                        and case['prefix_domain'] == 'original' and case['target'] == 'Exeunt')
        replacement = next(case for case in words if case['context_id'] == original['context_id']
                           and case['prefix_domain'] == 'replacement' and case['target'] == 'Exeunt')
        self.assertNotEqual(original['prefix']['token_ids_sha256'],
                            replacement['prefix']['token_ids_sha256'])

    def test_exact_teacher_forcing_rows_eos_padding_and_target_bytes(self):
        report = self.prepare()
        packed = self.read_packed(report)
        for case in report['cases']:
            index, length = case['case_index'], case['prefix']['length']
            self.assertEqual(case['scored_rows'], [length - 1, length, length + 1])
            self.assertEqual(packed[1, index, case['scored_rows']].tolist(), case['target_ids'])
            np.testing.assert_array_equal(packed[0, index, 1:], packed[1, index, :-1])
            self.assertTrue(np.all(packed[1, index, length + 2:] == cases.EOS))
            source = case['target_source']
            self.assertEqual(bytes.fromhex(source['bytes_hex']),
                             b''.join(bytes.fromhex(value) for value in source['native_piece_bytes_hex']))
            if case['kind'] == 'word':
                self.assertEqual(source['decoded_text'].lstrip(), case['target'])

    def test_both_variants_are_selected_and_reproducible_without_scores(self):
        first = self.prepare()
        self.output = self.root / 'prepared_again'
        second = self.prepare()
        self.assertEqual(first['cases'], second['cases'])
        self.assertEqual(first['selection'], second['selection'])
        self.assertEqual(first['packed_batch']['sha256'], second['packed_batch']['sha256'])
        self.assertFalse(first['selection']['model_outputs_used'])
        for split in cases.SPLITS:
            variants = {tuple(case['target_ids']) for case in first['cases']
                        if case['split'] == split and case['target'] == 'Exeunt'}
            self.assertEqual(variants, {(11, 12, 13), (31, 12, 13)})

    def test_all_control_prefix_and_target_tokens_are_disjoint_and_identical(self):
        report = self.prepare(controls_per_split=100)
        manifest = json.loads(self.manifest.read_text())
        for case in report['cases']:
            if case['kind'] != 'control':
                continue
            start, end = case['prefix']['token_start'], case['target_source']['token_end']
            for occurrence in manifest['alignment'][case['split']]['occurrences']:
                slot = occurrence['token_start']
                self.assertFalse(start < slot + 3 and slot < end)
            a, b = [np.fromfile(manifest['inputs'][f'{domain}.{case["split"]}']['token_ids'], dtype='<u4')
                     for domain in cases.DOMAINS]
            np.testing.assert_array_equal(a[start:end], b[start:end])

    def test_initial_short_prefix_is_allowed_but_empty_prefix_is_not(self):
        x, y, rows = cases._case_sequence([9], [1, 2, 3], 4)
        self.assertEqual(rows, [0, 1, 2])
        self.assertEqual(x.tolist(), [9, 1, 2, 3])
        self.assertEqual(y.tolist(), [1, 2, 3, cases.EOS])
        with self.assertRaises(ValueError):
            cases._case_sequence([], [1, 2, 3], 4)
        with self.assertRaises(ValueError):
            cases._case_sequence([1, 2, 3], [1, 2, 3], 4)

    def test_selection_count_validation_and_existing_directory(self):
        with self.assertRaisesRegex(ValueError, 'all available variants'):
            self.prepare(contexts_per_split=1)
        self.assertFalse(self.output.exists())
        for option, value in (('contexts_per_split', 0), ('controls_per_split', -1),
                               ('prefix_tokens', 15), ('seed', True), ('prefix_tokens', 0)):
            with self.subTest(option=option), self.assertRaises(ValueError):
                self.prepare(**{option: value})
        self.prepare(controls_per_split=0)
        with self.assertRaises(FileExistsError):
            self.prepare()

    def test_changed_native_input_rejected_before_creating_output(self):
        manifest = json.loads(self.manifest.read_text())
        path = Path(manifest['inputs']['original.training']['token_ids'])
        path.write_bytes(path.read_bytes() + b'x')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_wrong_native_size_and_offsets_rejected_even_with_new_hash(self):
        manifest = json.loads(self.manifest.read_text())
        item = manifest['inputs']['original.training']
        path = Path(item['offsets'])
        offsets = np.fromfile(path, dtype='<u8')
        offsets[2] = offsets[1]
        offsets.tofile(path)
        item['offsets_sha256'] = sha256_file(path)
        self.manifest.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'offset coverage'):
            self.prepare()

    def test_undeclared_changes_outside_replacement_slots_are_rejected(self):
        manifest = json.loads(self.manifest.read_text())
        item = manifest['inputs']['replacement.training']
        path = Path(item['token_ids'])
        tokens = np.fromfile(path, dtype='<u4')
        tokens[0] = 99
        tokens.tofile(path)
        item['token_ids_sha256'] = sha256_file(path)
        self.manifest.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'outside replacement slots'):
            self.prepare()

    def test_wrong_occurrence_candidate_or_coordinate_is_rejected(self):
        original = json.loads(self.manifest.read_text())
        for field, value in (('original_ids', [7, 8, 9]), ('byte_start', 1234)):
            manifest = json.loads(json.dumps(original))
            manifest['alignment']['training']['occurrences'][0][field] = value
            self.manifest.write_text(json.dumps(manifest))
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.prepare()

    def test_summary_scores_only_frozen_rows_and_labels_probability_means(self):
        report = self.prepare()
        scores = self.scores(report)
        summary = cases.summarize(self.output / 'cases.json', scores, self.root / 'summary.json')
        self.assertEqual(summary['case_count'], 20)
        self.assertEqual(len(summary['groups']), 10)
        for case in summary['per_case']:
            self.assertEqual(case['token_nll'], [.5, 1., 1.5])
            self.assertEqual(case['sequence_log_probability'], -3.)
            self.assertAlmostEqual(case['sequence_probability'], math.exp(-3))
            self.assertTrue(case['all_three_argmax_match'])
        for group in summary['groups']:
            self.assertEqual(group['case_count'], 2)
            self.assertEqual(group['mean_sequence_nll'], 3.)
            self.assertEqual(group['mean_token_nll'], 1.)
            self.assertAlmostEqual(group['geometric_mean_sequence_probability'], math.exp(-3))
            self.assertEqual(group['teacher_forced_argmax_token_accuracy'], 1.)
        json.dumps(summary, allow_nan=False)

    def test_probability_geometric_and_arithmetic_means_are_distinct(self):
        report = self.prepare()
        scores = self.scores(report)
        path = scores / 'losses.f32.bin'
        losses = np.fromfile(path, dtype='<f4').reshape(report['case_count'], 16)
        first = report['cases'][0]
        losses[first['case_index'], first['scored_rows']] = 0
        losses.tofile(path)
        summary = cases.summarize(self.output / 'cases.json', scores, self.root / 'summary.json')
        group = next(group for group in summary['groups'] if group['split'] == first['split']
                     and group['prefix_domain'] == first['prefix_domain'] and group['target'] == first['target'])
        self.assertAlmostEqual(group['geometric_mean_sequence_probability'], math.exp(-1.5))
        self.assertAlmostEqual(group['arithmetic_mean_sequence_probability'], (1 + math.exp(-3)) / 2)

    def test_nonfinite_negative_loss_and_invalid_argmax_rejected_anywhere(self):
        report = self.prepare()
        scores = self.scores(report)
        for filename, dtype, bad in (('losses.f32.bin', '<f4', np.nan),
                                      ('losses.f32.bin', '<f4', np.inf),
                                      ('losses.f32.bin', '<f4', -1),
                                      ('argmax.i32.bin', '<i4', -1),
                                      ('argmax.i32.bin', '<i4', cases.VOCAB_SIZE)):
            path = scores / filename
            old = path.read_bytes()
            data = np.fromfile(path, dtype=dtype)
            data[-1] = bad  # Even an unscored padding row must be valid.
            data.tofile(path)
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                cases.summarize(self.output / 'cases.json', scores, self.root / 'summary.json')
            path.write_bytes(old)

    def test_probe_wrong_size_shape_or_input_path_rejected(self):
        report = self.prepare()
        scores = self.scores(report)
        path = scores / 'losses.f32.bin'
        original = path.read_bytes()
        path.write_bytes(original[:-1])
        with self.assertRaisesRegex(ValueError, 'file size'):
            cases.summarize(self.output / 'cases.json', scores, self.root / 'summary.json')
        path.write_bytes(original)
        metadata_path = scores / 'metadata.json'
        metadata = json.loads(metadata_path.read_text())
        for key, value in (('case_count', 999), ('context_length', 1), ('batch_file', '/different.bin')):
            bad = dict(metadata, **{key: value})
            metadata_path.write_text(json.dumps(bad))
            with self.subTest(key=key), self.assertRaises(ValueError):
                cases.summarize(self.output / 'cases.json', scores, self.root / 'summary.json')

    def test_incomplete_or_incompatible_probe_metadata_is_rejected(self):
        report = self.prepare()
        scores = self.scores(report)
        metadata_path = scores / 'metadata.json'
        metadata = json.loads(metadata_path.read_text())
        for key, value in (('complete', False), ('temperature', .8), ('kind', 'another_probe'),
                            ('byte_order', 'big'), ('loss_dtype', '<f8'), ('output_shape', [1, 2])):
            metadata_path.write_text(json.dumps(dict(metadata, **{key: value})))
            with self.subTest(key=key), self.assertRaises(ValueError):
                cases.summarize(self.output / 'cases.json', scores, self.root / 'summary.json')

    def test_frozen_packed_hash_and_case_scoring_alignment_are_checked(self):
        report = self.prepare()
        scores = self.scores(report)
        packed_path = Path(report['packed_batch']['path'])
        original = packed_path.read_bytes()
        packed_path.write_bytes(b'X' + original[1:])
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            cases.summarize(self.output / 'cases.json', scores, self.root / 'summary.json')
        packed_path.write_bytes(original)
        report['cases'][0]['scored_rows'][0] = 0
        (self.output / 'cases.json').write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, 'alignment'):
            cases.summarize(self.output / 'cases.json', scores, self.root / 'summary.json')

    def test_summary_output_is_exclusive_and_outside_inputs(self):
        report = self.prepare()
        scores = self.scores(report)
        output = self.root / 'summary.json'
        cases.summarize(self.output / 'cases.json', scores, output)
        with self.assertRaises(FileExistsError):
            cases.summarize(self.output / 'cases.json', scores, output)
        for directory in (self.output, scores):
            with self.assertRaisesRegex(ValueError, 'outside input'):
                cases.summarize(self.output / 'cases.json', scores, directory / 'new.json')

    def test_underflowed_sequence_probability_is_zero_not_nonfinite(self):
        report = self.prepare()
        scores = self.scores(report)
        np.full((report['case_count'], 16), 1000., dtype='<f4').tofile(scores / 'losses.f32.bin')
        summary = cases.summarize(self.output / 'cases.json', scores, self.root / 'summary.json')
        self.assertEqual(summary['per_case'][0]['sequence_probability'], 0.)
        self.assertEqual(summary['per_case'][0]['sequence_log_probability'], -3000.)

    def test_cli_dispatch(self):
        with mock.patch.object(cases, 'prepare') as prepare:
            cases.main(['prepare', '--manifest', 'm.json', '--output', 'new', '--seed', '9'])
            prepare.assert_called_once_with(Path('m.json'), Path('new'), contexts_per_split=16,
                                             controls_per_split=16, prefix_tokens=128, seed=9)
        with mock.patch.object(cases, 'summarize') as summarize:
            cases.main(['summarize', '--cases', 'cases.json', '--scores', 'scores', '--output', 'new.json'])
            summarize.assert_called_once_with(Path('cases.json'), Path('scores'), Path('new.json'))


if __name__ == '__main__':
    unittest.main()
