"""CPU-only score/label corruption tests on actual-format amended fixtures."""

import copy
import json
import math
from pathlib import Path
import unittest

import numpy as np

from . import paired_lowercase_scores as scores
from . import paired_lowercase_cases_test as fixtures


class LowercaseScoresTest(unittest.TestCase):
    def setUp(self):
        fixture = fixtures.LowercaseCasesTest()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.root = fixture.root
        self.plan = fixture.prepare()
        self.path = fixture.output/'cases.json'
        self.directory = self.root/'scores'
        self.directory.mkdir()
        self.output = self.root/'summary.json'
        shape = self.plan['case_count'], self.plan['context_length']
        self.losses = np.full(shape, 30., dtype='<f4')
        self.argmax = np.zeros(shape, dtype='<i4')
        for case in self.plan['cases']:
            values = [1., 2., 3.] if case['target_source_domain'] == 'original' else [2., 4., 6.]
            self.losses[case['case_index'], case['scored_rows']] = values
            self.argmax[case['case_index'], case['scored_rows']] = case['target_ids']
        self.losses.tofile(self.directory/'losses.f32.bin')
        self.argmax.tofile(self.directory/'argmax.i32.bin')
        self.metadata = dict(kind='paired_loss_probe', complete=True, temperature=1,
            byte_order='little', loss_dtype='<f4', argmax_dtype='<i4',
            loss_file='losses.f32.bin', argmax_file='argmax.i32.bin',
            case_count=shape[0], passage_count=shape[0], context_length=shape[1],
            output_shape=list(shape), batch_file=self.plan['packed_batch']['path'],
            batch_bytes=self.plan['packed_batch']['bytes'], checkpoint_directory='/fixture/model')
        self.save()

    def save(self):
        self.path.write_text(json.dumps(self.plan))
        (self.directory/'metadata.json').write_text(json.dumps(self.metadata))

    def summarize(self):
        return scores.summarize(self.path, self.directory, self.output)

    def test_case_aware_probability_and_piece_decomposition(self):
        result = self.summarize()
        self.assertEqual(result['case_count'], 36)
        self.assertEqual(len(result['candidate_ratios']), 16)
        for row in result['candidate_ratios']:
            self.assertEqual(row['log_probability_ratio_original_over_replacement'], 6.)
            self.assertEqual(row['first_piece_log_ratio'], 1.)
            self.assertEqual(row['conditional_spelling_log_ratio'], 5.)
        lower = [g for g in result['groups'] if g['spelling_variant']=='lowercase']
        self.assertEqual(len(lower), 4)
        self.assertEqual({g['target'] for g in lower}, {'exeunt', 'nuveth'})
        self.assertTrue(all(g['split']=='training' for g in lower))
        self.assertAlmostEqual(result['per_case'][0]['sequence_probability'], math.exp(-6))
        self.assertEqual(result['per_case'][0]['conditional_spelling_nll'], 5.)
        self.assertTrue(all(g['all_three_argmax_match_fraction']==1 for g in result['groups']))

    def test_identical_models_have_zero_deltas(self):
        result = self.summarize()
        compared = scores.compare(result, result)
        self.assertTrue(all(x['nll_replacement_model_minus_original_model']==0 for x in compared['case_deltas']))
        self.assertTrue(all(x['original_minus_replacement_model_log_ratio']==0 for x in compared['candidate_ratio_deltas']))

    def test_model_comparison_rejects_truncated_and_different_cases(self):
        result = self.summarize()
        for mutate in (lambda r:r['per_case'].pop(),
                       lambda r:r['candidate_ratios'].pop(),
                       lambda r:r['per_case'][0].update(target='exeunt'),
                       lambda r:r['cases'].update(sha256='wrong')):
            other = copy.deepcopy(result); mutate(other)
            with self.assertRaises(ValueError):
                scores.compare(result, other)

    def test_lowercase_cannot_be_mislabeled_title(self):
        self.plan['cases'][-1]['spelling_variant']='title'; self.save()
        with self.assertRaisesRegex(ValueError, 'spelling'):
            self.summarize()

    def test_target_role_must_match_spelling(self):
        self.plan['cases'][-1]['target_source_domain']='original'; self.save()
        with self.assertRaisesRegex(ValueError, 'spelling'):
            self.summarize()

    def test_native_target_bytes_must_match_spelling(self):
        self.plan['cases'][-1]['target_source']['bytes_hex']=b' Exeunt'.hex(); self.save()
        with self.assertRaisesRegex(ValueError, 'native target bytes'):
            self.summarize()

    def test_duplicate_crossing_rejected(self):
        self.plan['cases'][-1]['prefix_domain']='original'; self.save()
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            self.summarize()

    def test_provenance_changes_rejected(self):
        Path(self.plan['provenance'][0]['path']).write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'frozen evidence'):
            self.summarize()

    def test_wrong_format_rejected(self):
        self.plan['format']='pluto-paired-word-cases-v1'; self.save()
        with self.assertRaisesRegex(ValueError, 'amended word cases'):
            self.summarize()

    def test_wrong_probe_metadata_rejected(self):
        for key, value in (('temperature', .7), ('complete', False), ('batch_file','/wrong'),
                           ('batch_sha256','wrong'), ('loss_dtype','<f8'), ('case_count',1)):
            with self.subTest(key=key):
                old = self.metadata.copy(); self.metadata[key]=value; self.save()
                with self.assertRaisesRegex(ValueError, 'metadata'):
                    self.summarize()
                self.metadata=old

    def test_nan_negative_and_truncated_losses_rejected(self):
        for value in (float('nan'), -1.):
            self.losses[0,0]=value; self.losses.tofile(self.directory/'losses.f32.bin')
            with self.assertRaisesRegex(ValueError, 'invalid loss'):
                self.summarize()
        (self.directory/'losses.f32.bin').write_bytes(b'1234')
        with self.assertRaisesRegex(ValueError, 'size mismatch'):
            self.summarize()

    def test_invalid_argmax_rejected(self):
        self.argmax[0,0]=50257; self.argmax.tofile(self.directory/'argmax.i32.bin')
        with self.assertRaisesRegex(ValueError, 'argmax'):
            self.summarize()

    def test_no_overwrite_or_output_inside_inputs(self):
        self.output.write_text('preserve')
        with self.assertRaises(FileExistsError):
            self.summarize()
        self.assertEqual(self.output.read_text(), 'preserve')
        for parent in (self.path.parent,self.directory):
            with self.assertRaisesRegex(ValueError,'outside input'):
                scores.summarize(self.path,self.directory,parent/'new.json')


if __name__ == '__main__':
    unittest.main()
