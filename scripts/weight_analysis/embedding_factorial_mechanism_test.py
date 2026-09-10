"""Postprocessor integration on synthetic CPU evidence, never native inference."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from . import embedding_factorial_mechanism as mechanism
from . import embedding_factorial_readout as base
from .embedding_factorial_readout_test import Fixture, write_json


class IntegrationTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.fixture = Fixture(self.root)
        self.output = self.root / 'mechanism.json'

    def prepare(self, *, execution=True):
        if execution:
            self.fixture.execution()
        return self.fixture.analyze(**(
            {'execution_record': self.fixture.execution_path} if execution else {}))

    def analyze(self, **kwargs):
        return mechanism.analyze(self.fixture.output, self.output,
                                 config=self.fixture.config, **kwargs)

    def test_full_revalidation_preserves_labels_targets_and_sequence_effects(self):
        saved = self.prepare()
        result = self.analyze()
        self.assertTrue(result['complete'])
        self.assertFalse(result['model_forward_performed'])
        self.assertFalse(result['weights_edited'])
        self.assertFalse(result['goal_completion_claimed'])
        self.assertEqual(len(result['per_case']), len(saved['per_case']))
        for item, original in zip(result['per_case'], saved['per_case']):
            for field in ('target_ids', 'source_case_index', 'prefix_domain', 'split'):
                self.assertEqual(item[field], original[field])
            for kind, value in item['selected_sequence']['log_probability'].items():
                self.assertAlmostEqual(value, original['log_probability_effects'][kind], places=12)
            for token, exposure in zip(item['tokens'], original['predictions']):
                self.assertEqual(token['causal_prefix_sha256'], exposure['causal_prefix_sha256'])
                self.assertNotEqual(token['rival_id'], token['target_id'])
                for kind in token['effects']['margin']:
                    self.assertAlmostEqual(token['effects']['margin'][kind]
                        - token['effects']['normalizer'][kind], token['effects']['log_probability'][kind], places=12)
        self.assertEqual(json.loads(self.output.read_text()), result)
        self.assertEqual(base._json(self.fixture.output), saved)
        for record in result['files']:
            self.assertEqual(base._record(record['path']), record)

    def test_word_suffix_and_following_native_token_are_separated(self):
        other = self.root / 'boundary'; other.mkdir()
        self.fixture = Fixture(other, boundary=True)
        self.prepare()
        result = self.analyze()
        for item in result['per_case']:
            self.assertEqual(item['kind'], 'word_next_native')
            self.assertEqual(len(item['tokens']), 4)
            for name in ('margin', 'normalizer', 'log_probability'):
                for kind in ('input', 'output', 'interaction', 'joint'):
                    self.assertAlmostEqual(item['first_piece'][name][kind] + item['suffix'][name][kind],
                                           item['word_three'][name][kind], places=12)
                    self.assertAlmostEqual(item['word_three'][name][kind] + item['exact_next_native_token'][name][kind],
                                           item['selected_sequence'][name][kind], places=12)

    def test_missing_execution_record_is_rejected_before_revalidation(self):
        self.prepare(execution=False)
        with mock.patch.object(base, 'analyze') as analyze, self.assertRaisesRegex(ValueError, 'execution provenance'):
            self.analyze()
        analyze.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_edited_saved_score_or_label_cannot_replace_recomputed_evidence(self):
        saved = self.prepare()
        saved['per_case'][0]['target'] = 'unearned mechanistic claim'
        write_json(self.fixture.output, saved)
        with self.assertRaisesRegex(ValueError, 'independent revalidation'):
            self.analyze()
        self.assertFalse(self.output.exists())

    def test_changed_logit_bytes_fail_execution_and_source_checks(self):
        self.prepare()
        path = self.fixture.scores / 'JJ.logits.f32.bin'
        raw = bytearray(path.read_bytes()); raw[0] ^= 1; path.write_bytes(raw)
        with self.assertRaises(ValueError):
            self.analyze()
        self.assertFalse(self.output.exists())

    def test_changed_checkpoint_is_not_accepted_from_stored_readout(self):
        self.prepare()
        path = self.fixture.a / 'weight_1.bin'
        raw = bytearray(path.read_bytes()); raw[0] ^= 1; path.write_bytes(raw)
        with self.assertRaises(ValueError):
            self.analyze()
        self.assertFalse(self.output.exists())

    def test_existing_output_does_not_trigger_revalidation_or_overwrite(self):
        self.prepare(); self.output.write_text('keep this')
        with mock.patch.object(base, 'analyze') as analyze, self.assertRaisesRegex(ValueError, 'already exists'):
            self.analyze()
        analyze.assert_not_called()
        self.assertEqual(self.output.read_text(), 'keep this')

    def test_cannot_write_inside_source_checkpoint_or_native_score_tree(self):
        self.prepare()
        for parent in (self.fixture.a, self.fixture.d, self.fixture.j, self.fixture.scores):
            self.output = parent / 'new_mechanism.json'
            with self.subTest(parent=parent), self.assertRaisesRegex(ValueError, 'outside source'):
                self.analyze()
            self.assertFalse(self.output.exists())

    def test_input_mutation_after_revalidation_prevents_publication(self):
        self.prepare()
        original = mechanism.decompose_logits
        changed = False
        def mutate(*args, **kwargs):
            nonlocal changed
            result = original(*args, **kwargs)
            if not changed:
                self.fixture.binary.write_bytes(b'changed after revalidation')
                changed = True
            return result
        with mock.patch.object(mechanism, 'decompose_logits', side_effect=mutate):
            with self.assertRaisesRegex(ValueError, 'source changed'):
                self.analyze()
        self.assertFalse(self.output.exists())

    def test_cli_passes_paths_without_replacing_geometry_or_other_options(self):
        with mock.patch.object(mechanism, 'analyze') as analyze:
            mechanism.main(['--readout', str(self.fixture.output), '--output', str(self.output)])
        analyze.assert_called_once_with(self.fixture.output, self.output)


if __name__ == '__main__':
    unittest.main()
