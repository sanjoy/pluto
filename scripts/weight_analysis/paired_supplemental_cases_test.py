"""CPU-only tests with native-export fixtures; no model outputs select cases."""

import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from . import paired_supplemental_cases as supplemental
from . import paired_word_cases
from .checkpoint import sha256_file
from .paired_word_cases_test import make_manifest


def add_piece_controls(manifest_path, truncate=None):
    """Place real piece bytes away from replacements in both toy corpora."""
    manifest = json.loads(manifest_path.read_text())
    piece_bytes = {11: b' Ex', 12: b'e', 13: b'unt', 21: b' N',
                   22: b'uv', 23: b'eth', 31: b'Ex', 32: b'N'}
    for split in paired_word_cases.SPLITS:
        for domain in paired_word_cases.DOMAINS:
            item = manifest['inputs'][f'{domain}.{split}']
            tokens = np.fromfile(item['token_ids'], dtype='<u4')
            offsets = np.fromfile(item['offsets'], dtype='<u8')
            text = Path(item['text']).read_bytes()
            pieces = [text[int(offsets[i]):int(offsets[i + 1])]
                      for i in range(len(tokens))]
            for start, token in zip((20, 26, 33, 54, 61, 68, 88, 95), sorted(piece_bytes)):
                if token == 13 and split == 'test':
                    continue  # Explicitly test missing eligible coverage for this row.
                tokens[start], pieces[start] = token, piece_bytes[token]
            # The first three are ineligible: short prefix, replacement overlap,
            # and insufficient following targets, respectively.
            for start in (1, 14, 119, 102, 109):
                tokens[start], pieces[start] = 11, piece_bytes[11]
            if truncate:
                tokens, pieces = tokens[:truncate], pieces[:truncate]
            offsets = np.concatenate(([0], np.cumsum([len(piece) for piece in pieces]))).astype('<u8')
            text = b''.join(pieces)
            Path(item['text']).write_bytes(text)
            tokens.tofile(item['token_ids'])
            offsets.tofile(item['offsets'])
            item['export'].update(token_count=len(tokens), offset_count=len(offsets), corpus_bytes=len(text))
            for key in ('text', 'token_ids', 'offsets'):
                item[key + '_sha256'] = sha256_file(item[key])
            if domain == 'original':
                for occurrence in manifest['alignment'][split]['occurrences']:
                    start = occurrence['token_start']
                    occurrence['byte_start'] = int(offsets[start]) + pieces[start].startswith(b' ')
        manifest['alignment'][split]['token_count'] = len(tokens)
    manifest_path.write_text(json.dumps(manifest))


class SupplementalCasesTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.manifest = make_manifest(self.root)
        add_piece_controls(self.manifest)
        self.word_output = self.root / 'word_cases'
        self.output = self.root / 'supplemental'

    def freeze_words(self):
        return paired_word_cases.prepare(self.manifest, self.word_output,
                                         contexts_per_split=3, controls_per_split=1,
                                         prefix_tokens=4, context_length=16)

    def prepare(self, **kwargs):
        if not self.word_output.exists():
            self.freeze_words()
        return supplemental.prepare(self.manifest, self.word_output / 'cases.json', self.output,
                                    prefix_tokens=4, controls_per_piece=kwargs.get('controls_per_piece', 2))

    def scores(self, plan):
        directory = self.root / 'scores'
        directory.mkdir()
        shape = [plan['case_count'], plan['context_length']]
        losses, argmax = np.full(shape, 100, dtype='<f4'), np.zeros(shape, dtype='<i4')
        for case in plan['cases']:
            index, rows = case['case_index'], case['scored_rows']
            losses[index, rows] = [.5, 1., 1.5, 2.][:len(rows)]
            argmax[index, rows] = case['target_ids']
        losses.tofile(directory / 'losses.f32.bin')
        argmax.tofile(directory / 'argmax.i32.bin')
        (directory / 'metadata.json').write_text(json.dumps({
            'kind': 'paired_loss_probe', 'complete': True, 'temperature': 1,
            'byte_order': 'little', 'loss_dtype': '<f4', 'argmax_dtype': '<i4',
            'loss_file': 'losses.f32.bin', 'argmax_file': 'argmax.i32.bin',
            'case_count': shape[0], 'passage_count': shape[0], 'context_length': shape[1],
            'output_shape': shape, 'batch_file': plan['packed_batch']['path'],
            'batch_bytes': plan['packed_batch']['bytes']}))
        return directory

    def test_preserves_every_word_crossing_and_exact_next_native_token(self):
        frozen = self.freeze_words()
        report = self.prepare()
        packed, _ = supplemental._packed(report)
        words = [case for case in report['cases'] if case['kind'] == 'word_next_native']
        self.assertEqual(len(words), 24)
        self.assertFalse(report['selection']['excluded_word_cases'])
        for case in words:
            old = frozen['cases'][case['source_case_index']]
            self.assertEqual(case['prefix'], old['prefix'])
            self.assertEqual(case['target_ids'][:3], old['target_ids'])
            self.assertEqual(case['context_id'], old['context_id'])
            self.assertEqual(case['prefix_domain'], old['prefix_domain'])
            self.assertEqual(case['target'], old['target'])
            self.assertEqual(case['scored_rows'], old['scored_rows'] + [old['scored_rows'][-1] + 1])
            self.assertEqual(case['next_native_token']['id'], 1)
            # The fixture's actual next token is "a", deliberately NOT a delimiter.
            self.assertEqual(case['next_native_token']['decoded_text'], 'a')
            self.assertEqual(packed[1, case['case_index'], case['scored_rows']].tolist(),
                             case['target_ids'])
        self.assertIn('not any possible delimiter', report['limitations'][0])

    def test_shared_piece_coverage_is_complete_where_eligible_and_explicit_where_missing(self):
        report = self.prepare()
        manifest = json.loads(self.manifest.read_text())
        selection = report['selection']
        self.assertEqual(selection['union_piece_ids'], [11, 12, 13, 21, 22, 23, 31, 32])
        self.assertEqual(selection['missing_piece_counts'], {'training': 0, 'test': 1})
        missing = [row for row in selection['coverage']['test'] if row['missing_coverage']]
        self.assertEqual([row['piece_id'] for row in missing], [13])
        self.assertEqual(missing[0]['outside_replacement_occurrences'], 0)
        row = next(row for row in selection['coverage']['training'] if row['piece_id'] == 11)
        self.assertEqual(row['outside_replacement_occurrences'], 6)
        self.assertEqual(row['eligible_window_count'], 3)
        for case in report['cases']:
            if case['kind'] != 'shared_piece':
                continue
            self.assertEqual(case['target_ids'][0], case['piece_id'])
            self.assertEqual(case['prefix']['length'], 4)
            begin, end = case['prefix']['token_start'], case['target_source']['token_end']
            for occurrence in manifest['alignment'][case['split']]['occurrences']:
                start = occurrence['token_start']
                self.assertFalse(begin < start + 3 and start < end)
            streams = [np.fromfile(manifest['inputs'][f'{domain}.{case["split"]}']['token_ids'], dtype='<u4')
                       for domain in paired_word_cases.DOMAINS]
            np.testing.assert_array_equal(streams[0][begin:end], streams[1][begin:end])

    def test_fixed_seed_reproduces_cases_and_packed_bytes(self):
        first = self.prepare()
        self.output = self.root / 'again'
        second = self.prepare()
        self.assertEqual(first['cases'], second['cases'])
        self.assertEqual(first['selection'], second['selection'])
        self.assertEqual(first['packed_batch']['sha256'], second['packed_batch']['sha256'])
        self.assertEqual(first['selection']['seed'], 17)
        self.assertFalse(first['selection']['model_outputs_used'])

    def test_terminal_word_excluded_without_inventing_next_eos(self):
        add_piece_controls(self.manifest, truncate=83)
        report = self.prepare()
        excluded = report['selection']['excluded_word_cases']
        self.assertEqual(len(excluded), 8)  # Last occurrence x four crossings x two splits.
        self.assertTrue(all('EOS was not invented' in case['reason'] for case in excluded))
        self.assertEqual(sum(case['kind'] == 'word_next_native' for case in report['cases']), 16)

    def test_prepare_rejects_changed_manifest_native_export_or_packed_bytes(self):
        frozen = self.freeze_words()
        for path in (self.manifest, Path(frozen['source_inputs'][0]['path']),
                     Path(frozen['packed_batch']['path'])):
            original = path.read_bytes()
            path.write_bytes(original + b' ')
            with self.subTest(path=path), self.assertRaises(ValueError):
                self.prepare()
            self.assertFalse(self.output.exists())
            path.write_bytes(original)

    def test_prepare_rejects_tampered_word_context_and_existing_output(self):
        frozen = self.freeze_words()
        frozen['cases'][0]['prefix']['token_start'] += 1
        (self.word_output / 'cases.json').write_text(json.dumps(frozen))
        with self.assertRaisesRegex(ValueError, 'native source'):
            self.prepare()
        self.assertFalse(self.output.exists())
        self.output.mkdir()
        with self.assertRaises(FileExistsError):
            self.prepare()

    def test_missing_declared_crossing_is_rejected(self):
        frozen = self.freeze_words()
        frozen['cases'][0]['kind'] = 'control'
        (self.word_output / 'cases.json').write_text(json.dumps(frozen))
        with self.assertRaisesRegex(ValueError, 'every declared'):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_default_controls_require_full_128_token_prefix(self):
        paired_word_cases.prepare(self.manifest, self.word_output,
                                  contexts_per_split=3, controls_per_split=0,
                                  prefix_tokens=128, context_length=1024)
        report = supplemental.prepare(self.manifest, self.word_output / 'cases.json', self.output)
        self.assertEqual(report['selection']['shared_piece_prefix_tokens'], 128)
        self.assertEqual(report['selection']['missing_piece_counts'], {'training': 8, 'test': 8})
        self.assertFalse(any(case['kind'] == 'shared_piece' for case in report['cases']))
        # The 120-token corpus still has word cases with preserved short prefixes;
        # it cannot supply a full 128-token control prefix.
        self.assertEqual(report['case_count'], 24)

    def test_sequence_checks_all_target_rows_fit(self):
        x, y, rows = supplemental._sequence([7], [1, 2, 3, 4], 4)
        self.assertEqual(rows, [0, 1, 2, 3])
        self.assertEqual(x.tolist(), [7, 1, 2, 3])
        self.assertEqual(y.tolist(), [1, 2, 3, 4])
        for prefix, target in (([], [1, 2, 3]), ([7, 8], [1, 2, 3, 4]), ([7], [1, 2])):
            with self.assertRaises(ValueError):
                supplemental._sequence(prefix, target, 4)

    def test_summary_separates_word_boundary_joint_and_shared_piece_nll(self):
        report = self.prepare()
        scores = self.scores(report)
        summary = supplemental.summarize(self.output / 'cases.json', scores, self.root / 'summary.json')
        self.assertEqual(summary['selection']['missing_piece_counts']['test'], 1)
        for case in summary['per_case']:
            self.assertTrue(case['all_target_argmax_match'])
            if case['kind'] == 'word_next_native':
                self.assertEqual(case['word_three_nll'], 3)
                self.assertEqual(case['next_native_token_nll'], 2)
                self.assertEqual(case['word_and_exact_next_token_nll'], 5)
                self.assertAlmostEqual(case['word_three_probability'], math.exp(-3))
                self.assertAlmostEqual(case['next_native_token_conditional_probability'], math.exp(-2))
                self.assertAlmostEqual(case['sequence_probability'], math.exp(-5))
            else:
                self.assertEqual(case['shared_piece_nll'], .5)
                self.assertEqual(case['sequence_nll'], 3)
        for group in summary['groups']:
            self.assertEqual(group['all_target_argmax_match_fraction'], 1)
            if group['kind'] == 'word_next_native':
                self.assertEqual(group['mean_word_three_nll'], 3)
                self.assertEqual(group['mean_next_native_token_nll'], 2)
            else:
                self.assertEqual(group['mean_shared_piece_nll'], .5)

    def test_summary_rejects_bad_metadata_sizes_and_values(self):
        report = self.prepare()
        scores = self.scores(report)
        meta_path = scores / 'metadata.json'
        metadata = json.loads(meta_path.read_text())
        for key, value in (('temperature', .8), ('complete', False), ('output_shape', [1, 2]),
                           ('batch_bytes', 0), ('batch_file', '/other'), ('batch_sha256', 'bad')):
            meta_path.write_text(json.dumps({**metadata, key: value}))
            with self.subTest(key=key), self.assertRaises(ValueError):
                supplemental.summarize(self.output / 'cases.json', scores, self.root / 'summary.json')
        meta_path.write_text(json.dumps(metadata))
        for filename, dtype, value in (('losses.f32.bin', '<f4', np.nan),
                                       ('losses.f32.bin', '<f4', -1),
                                       ('argmax.i32.bin', '<i4', 50257)):
            path = scores / filename
            original = path.read_bytes()
            values = np.fromfile(path, dtype=dtype)
            values[-1] = value  # Validate even the unscored padding.
            values.tofile(path)
            with self.assertRaises(ValueError):
                supplemental.summarize(self.output / 'cases.json', scores, self.root / 'summary.json')
            path.write_bytes(original[:-1])
            with self.assertRaisesRegex(ValueError, 'size mismatch'):
                supplemental.summarize(self.output / 'cases.json', scores, self.root / 'summary.json')
            path.write_bytes(original)

    def test_summary_checks_packed_hash_and_scoring_alignment(self):
        report = self.prepare()
        scores = self.scores(report)
        packed = Path(report['packed_batch']['path'])
        original = packed.read_bytes()
        packed.write_bytes(b'X' + original[1:])
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            supplemental.summarize(self.output / 'cases.json', scores, self.root / 'summary.json')
        packed.write_bytes(original)
        report['cases'][0]['scored_rows'][-1] -= 1
        (self.output / 'cases.json').write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, 'alignment'):
            supplemental.summarize(self.output / 'cases.json', scores, self.root / 'summary.json')

    def test_summary_rejects_wrong_target_identity_and_prefix_hash(self):
        report = self.prepare()
        scores = self.scores(report)
        case_path = self.output / 'cases.json'
        original = case_path.read_text()
        changes = [lambda plan: plan['cases'][0]['next_native_token'].update(id=42),
                   lambda plan: plan['cases'][0]['prefix'].update(token_ids_sha256='bad'),
                   lambda plan: next(case for case in plan['cases']
                                     if case['kind'] == 'shared_piece').update(piece_id=42)]
        for change in changes:
            plan = json.loads(original)
            change(plan)
            case_path.write_text(json.dumps(plan))
            with self.assertRaises(ValueError):
                supplemental.summarize(case_path, scores, self.root / 'summary.json')

    def test_summary_keeps_log_probability_when_probability_underflows(self):
        report = self.prepare()
        scores = self.scores(report)
        np.full((report['case_count'], 16), 1000., dtype='<f4').tofile(scores / 'losses.f32.bin')
        summary = supplemental.summarize(self.output / 'cases.json', scores, self.root / 'summary.json')
        word = summary['per_case'][0]
        self.assertEqual(word['word_three_nll'], 3000.)
        self.assertEqual(word['next_native_token_nll'], 1000.)
        self.assertEqual(word['sequence_log_probability'], -4000.)
        self.assertEqual(word['sequence_probability'], 0.)

    def test_summary_exclusive_and_outside_input_directories(self):
        report = self.prepare()
        scores = self.scores(report)
        output = self.root / 'summary.json'
        supplemental.summarize(self.output / 'cases.json', scores, output)
        with self.assertRaises(FileExistsError):
            supplemental.summarize(self.output / 'cases.json', scores, output)
        for directory in (self.output, scores):
            with self.assertRaisesRegex(ValueError, 'outside input'):
                supplemental.summarize(self.output / 'cases.json', scores, directory / 'new.json')

    def test_cli_dispatch(self):
        with mock.patch.object(supplemental, 'prepare') as prepare:
            supplemental.main(['prepare', '--manifest', 'm.json', '--word-cases', 'w.json', '--output', 'new'])
            prepare.assert_called_once_with(Path('m.json'), Path('w.json'), Path('new'),
                                            controls_per_piece=8, prefix_tokens=128)
        with mock.patch.object(supplemental, 'summarize') as summarize:
            supplemental.main(['summarize', '--cases', 'c.json', '--scores', 'scores', '--output', 's.json'])
            summarize.assert_called_once_with(Path('c.json'), Path('scores'), Path('s.json'))


if __name__ == '__main__':
    unittest.main()
