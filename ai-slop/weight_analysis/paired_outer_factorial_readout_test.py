"""Tiny synthetic CPU evidence tests, never GPU execution or forward proof."""

import copy
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from . import embedding_factorial_readout as native
from . import paired_outer_factorial_readout as outer
from .checkpoint import tensor_manifest
from .embedding_factorial_readout_test import Fixture, write_json
from .paired_weight_patch import create_patch


class OuterFixture:
    """Real tiny checkpoint/manifest validation with explicitly invented logits."""

    def __init__(self, root, *, boundary=False, omit_complement=False):
        self.root = root
        self.fixtures = {}
        for name in ('base', 'conditioned', 'donor'):
            directory = root / name
            directory.mkdir()
            self.fixtures[name] = Fixture(directory, boundary=boundary)
        a = self.fixtures['base']
        self.config, self.rows = a.config, a.rows
        self.c = root / 'complement'
        self.tensors = sorted(spec.name for spec in tensor_manifest(self.config) if spec.index)
        create_patch(a.a, a.d, self.c, tensors=self.tensors[1:] if omit_complement else self.tensors,
                     config=self.config)
        for name in ('conditioned', 'donor'):
            f = self.fixtures[name]
            f.a, f.d = (self.c, a.d) if name == 'conditioned' else (a.d, a.a)
            f.j = root / (name + '_patched')
            f.patch = create_patch(f.a, f.d, f.j, embedding_rows=self.rows, config=self.config)
            # Every readout must use the SAME authenticated case source, not
            # merely equal-looking labels copied from a separate document.
            f.cases_path, f.plan = a.cases_path, a.plan
            f.cases, f.packed, f.rows = copy.deepcopy(a.cases), a.packed.copy(), list(a.rows)
            f.configure_selection(a.kind)
            offset = np.asarray((np.arange(self.config.vocab_size) % 3) * .125, dtype='<f4')
            selected = np.asarray(np.isin(np.arange(self.config.vocab_size), self.rows) * .25,
                                  dtype='<f4')
            for cell in native.CELLS:
                f.arrays[cell] += offset * (1 if name == 'conditioned' else 2)
                if name == 'conditioned' and cell in ('AJ', 'JJ'):
                    f.arrays[cell] += selected
            f.write_scores()
        self.saved = {}
        for name, f in self.fixtures.items():
            f.execution()
            self.saved[name] = f.analyze(execution_record=f.execution_path)

    def analyze(self, output):
        return outer.analyze(*(self.fixtures[name].output for name in
                               ('base', 'conditioned', 'donor')), output, config=self.config)


class MathTest(unittest.TestCase):
    def rows(self):
        return {name: np.asarray(values, dtype='<f4') for name, values in {
            'A': [1, 2, 0, -1], 'E': [2, 2, 1, -1], 'C': [0, 1, 3, -1],
            'EC': [4, 0, 2, -1], 'D': [-2, 1, 6, 3]}.items()}

    def test_independent_scalar_oracle_effect_names_and_fixed_donor_rival(self):
        rows = self.rows()
        result = outer.decompose_logits(rows, 0)
        self.assertEqual(result['rival_id'], 1)  # D's winner is 2; do not switch.
        values = {}
        for name, row in rows.items():
            margin = float(row[0]) - float(row[1])
            normalizer = math.log(sum(math.exp(float(x) - float(row[1])) for x in row))
            values[name] = dict(margin=margin, normalizer=normalizer,
                                log_probability=margin-normalizer)
            self.assertAlmostEqual(result['cells'][name]['log_probability'], margin-normalizer, places=13)
            self.assertAlmostEqual(result['cells'][name]['probability'], math.exp(margin-normalizer), places=13)
        for name in outer.METRICS:
            v = {cell: row[name] for cell, row in values.items()}
            expected = dict(E=v['E']-v['A'], C=v['C']-v['A'],
                            interaction=v['EC']-v['E']-v['C']+v['A'], joint=v['EC']-v['A'])
            self.assertEqual(set(result['effects'][name]), set(outer.EFFECTS))
            for key, value in expected.items():
                self.assertAlmostEqual(result['effects'][name][key], value, places=13)
            self.assertAlmostEqual(result['remaining_embedding_residual'][name], v['D']-v['EC'], places=13)
        self.assertLess(result['max_effect_identity_error'], 1e-12)

    def test_common_shifts_are_invariant_and_input_bytes_unchanged(self):
        rows = self.rows(); before = {name: row.tobytes() for name, row in rows.items()}
        original = outer.decompose_logits(rows, 0)
        shifted = outer.decompose_logits({name: row + np.float32(i*8)
                                         for i, (name, row) in enumerate(rows.items())}, 0)
        for name in outer.METRICS:
            for effect in outer.EFFECTS:
                self.assertAlmostEqual(original['effects'][name][effect], shifted['effects'][name][effect], places=13)
            self.assertAlmostEqual(original['remaining_embedding_residual'][name],
                                   shifted['remaining_embedding_residual'][name], places=13)
        self.assertEqual(before, {name: row.tobytes() for name, row in rows.items()})

    def test_identical_rows_zero_effects_and_stable_large_logits(self):
        row = np.asarray([1000, 999, 999], dtype='<f4')
        result = outer.decompose_logits({name: row for name in outer.CELLS}, 0)
        self.assertEqual(result['rival_id'], 1)
        for values in result['effects'].values():
            self.assertTrue(all(value == 0 for value in values.values()))
        self.assertTrue(all(value == 0 for value in result['remaining_embedding_residual'].values()))
        self.assertAlmostEqual(result['cells']['A']['log_probability'], -math.log(1+2/math.e), places=13)

    def test_invalid_rows_targets_and_nonfinite_donor_are_rejected(self):
        for target in (-1, 4, True, 1.5, np.int32(0)):
            with self.subTest(target=target), self.assertRaises(ValueError):
                outer.decompose_logits(self.rows(), target)
        for value in (np.ones(4, dtype='<f8'), np.ones((1, 4), dtype='<f4'),
                      np.ones(3, dtype='<f4'), np.asarray([1, 2, np.nan, 4], dtype='<f4'),
                      np.asarray([1, 2, np.inf, 4], dtype='<f4')):
            rows = self.rows(); rows['D'] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                outer.decompose_logits(rows, 0)
        rows = self.rows(); del rows['C']
        with self.assertRaises(ValueError):
            outer.decompose_logits(rows, 0)


class IntegrationTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='pluto-outer-test-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.output = self.root / 'outer.json'

    def prepare(self, **kwargs):
        self.fixture = OuterFixture(self.root, **kwargs)
        return self.fixture

    def test_three_reports_and_exact_complement_revalidated_without_mutation(self):
        f = self.prepare()
        before = {name: fixture.output.read_bytes() for name, fixture in f.fixtures.items()}
        result = f.analyze(self.output)
        self.assertTrue(result['complete'])
        self.assertTrue(result['conditioned_checkpoint']['exact_bytes_verified'])
        self.assertEqual(result['conditioned_checkpoint']['selected_tensors'], f.tensors)
        for key in ('model_forward_performed', 'weights_edited', 'goal_completion_claimed'):
            self.assertFalse(result[key])
        self.assertEqual(json.loads(self.output.read_text()), result)
        self.assertEqual(before, {name: fixture.output.read_bytes() for name, fixture in f.fixtures.items()})
        for record in result['files']:
            self.assertEqual(native._record(record['path']), record)
        source_paths = {record['path'] for record in result['files']}
        self.assertIn(str(Path(outer.__file__).resolve()), source_paths)
        self.assertIn(str(Path(outer.mechanism.__file__).resolve()), source_paths)
        mapping = {'A': ('base', 'AA'), 'E': ('base', 'JJ'), 'C': ('conditioned', 'AA'),
                   'EC': ('conditioned', 'JJ'), 'D': ('donor', 'AA')}
        for i, case in enumerate(result['per_case']):
            for cell, (source, inner) in mapping.items():
                self.assertAlmostEqual(case['selected_sequence']['cells'][cell]['log_probability'],
                    f.saved[source]['per_case'][i]['cells'][inner]['sequence_log_probability'], places=12)
            for token in case['tokens']:
                self.assertNotIn('input_path_must_be_unchanged', token)
                self.assertIn('selected_embedding_rows_absent_from_prefix', token)
                for name in outer.METRICS:
                    self.assertEqual(set(token['effects'][name]), set(outer.EFFECTS))
                self.assertLess(token['max_effect_identity_error'], 1e-12)

    def test_first_suffix_word_and_exact_fourth_target_remain_separate(self):
        f = self.prepare(boundary=True)
        result = f.analyze(self.output)
        for case in result['per_case']:
            self.assertEqual(len(case['tokens']), 4)
            for cell in outer.CELLS:
                field = 'log_probability'
                self.assertAlmostEqual(case['first_piece']['cells'][cell][field] +
                    case['suffix']['cells'][cell][field], case['word_three']['cells'][cell][field], places=12)
                self.assertAlmostEqual(case['word_three']['cells'][cell][field] +
                    case['exact_next_native_token']['cells'][cell][field],
                    case['selected_sequence']['cells'][cell][field], places=12)
            for metric in outer.METRICS:
                self.assertAlmostEqual(case['word_three']['remaining_embedding_residual'][metric] +
                    case['exact_next_native_token']['remaining_embedding_residual'][metric],
                    case['selected_sequence']['remaining_embedding_residual'][metric], places=12)

    def test_wrong_conditioned_tensor_selection_is_rejected(self):
        f = self.prepare(omit_complement=True)
        with self.assertRaisesRegex(ValueError, 'exactly every non-token-embedding'):
            f.analyze(self.output)
        self.assertFalse(self.output.exists())

    def test_forged_complete_selection_and_flags_do_not_replace_byte_check(self):
        f = self.prepare(omit_complement=True)
        patch_path = f.c / 'patch.json'; patch = native._json(patch_path)
        patch['selection']['tensors'] = f.tensors
        write_json(patch_path, patch)
        with self.assertRaisesRegex(ValueError, 'C bytes do not match'):
            f.analyze(self.output)
        self.assertFalse(self.output.exists())

    def test_wrong_complement_source_binding_is_rejected(self):
        f = self.prepare()
        path = f.c / 'patch.json'; patch = native._json(path)
        patch['sources']['original']['path'] = str(f.fixtures['base'].d)
        write_json(path, patch)
        with self.assertRaisesRegex(ValueError, 'source/output paths'):
            f.analyze(self.output)

    def test_source_hardlink_cannot_pass_matching_hashes(self):
        f = self.prepare()
        target = f.c / 'weight_1.bin'; target.unlink()
        target.hardlink_to(f.fixtures['base'].d / target.name)
        with self.assertRaisesRegex(ValueError, 'shares a source inode'):
            f.analyze(self.output)

    def test_conditioned_configuration_rejects_boolean_integer_alias(self):
        f = self.prepare()
        path = f.c / 'patch.json'; patch = native._json(path)
        patch['config']['n_layers'] = True  # Equal to 1 in dict equality, not a dimension.
        write_json(path, patch)
        with self.assertRaisesRegex(ValueError, 'positive integer'):
            f.analyze(self.output)

    def test_wrong_donor_endpoint_report_is_rejected(self):
        f = self.prepare()
        with self.assertRaisesRegex(ValueError, 'reciprocal base endpoints'):
            outer.analyze(f.fixtures['base'].output, f.fixtures['conditioned'].output,
                          f.fixtures['base'].output, self.output, config=f.config)

    def test_changed_saved_case_label_is_not_accepted(self):
        f = self.prepare(); path = f.fixtures['conditioned'].output
        saved = native._json(path); saved['per_case'][0]['context_id'] = 'different'
        write_json(path, saved)
        with self.assertRaisesRegex(ValueError, 'independent revalidation'):
            f.analyze(self.output)

    def test_valid_but_different_case_source_is_rejected(self):
        f = self.prepare(); conditioned = f.fixtures['conditioned']
        conditioned.cases_path = conditioned.root / 'other_cases.json'
        plan = copy.deepcopy(conditioned.plan)
        plan['cases'][0]['context_id'] = 'different-authenticated-context'
        conditioned.cases = plan['cases']
        write_json(conditioned.cases_path, plan)
        conditioned.output = conditioned.root / 'other_readout.json'
        conditioned.write_scores(); conditioned.execution()
        conditioned.analyze(execution_record=conditioned.execution_path)
        with self.assertRaisesRegex(ValueError, 'case source/suite/order mismatch'):
            f.analyze(self.output)

    def test_missing_execution_provenance_fails_before_native_reader(self):
        f = self.prepare(); path = f.fixtures['donor'].output
        saved = native._json(path); saved['execution_provenance']['verified'] = False
        write_json(path, saved)
        with mock.patch.object(native, 'analyze') as rerun, self.assertRaisesRegex(ValueError, 'execution provenance'):
            f.analyze(self.output)
        rerun.assert_not_called()

    def test_changed_logit_or_checkpoint_bytes_cannot_use_stale_report(self):
        f = self.prepare()
        path = f.fixtures['base'].scores / 'JJ.logits.f32.bin'
        original = path.read_bytes(); data = bytearray(original); data[0] ^= 1
        path.write_bytes(data)
        with self.assertRaises(ValueError):
            f.analyze(self.output)
        path.write_bytes(original)
        path = f.c / 'weight_0.bin'; data = bytearray(path.read_bytes()); data[0] ^= 1
        path.write_bytes(data)
        with self.assertRaises(ValueError):
            f.analyze(self.output)
        self.assertFalse(self.output.exists())

    def test_existing_or_dangling_output_is_preserved_without_revalidation(self):
        f = self.prepare(); self.output.write_text('keep')
        with mock.patch.object(native, 'analyze') as rerun, self.assertRaisesRegex(ValueError, 'already exists'):
            f.analyze(self.output)
        rerun.assert_not_called(); self.assertEqual(self.output.read_text(), 'keep')
        self.output.unlink(); self.output.symlink_to(self.root / 'missing')
        with self.assertRaisesRegex(ValueError, 'already exists'):
            f.analyze(self.output)
        self.assertTrue(self.output.is_symlink())

    def test_output_inside_any_checkpoint_or_native_scores_is_rejected(self):
        f = self.prepare()
        paths = {f.c}
        for fixture in f.fixtures.values():
            paths.update((fixture.a, fixture.d, fixture.j, fixture.scores))
        for path in paths:
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, 'outside source checkpoints'):
                f.analyze(path / 'outer.json')
            self.assertFalse((path / 'outer.json').exists())

    def test_mutation_during_math_prevents_publication(self):
        f = self.prepare(); original = outer.decompose_logits; changed = False
        def mutate(*args, **kwargs):
            nonlocal changed
            result = original(*args, **kwargs)
            if not changed:
                f.fixtures['base'].binary.write_bytes(b'changed during CPU analysis')
                changed = True
            return result
        with mock.patch.object(outer, 'decompose_logits', side_effect=mutate):
            with self.assertRaisesRegex(ValueError, 'source changed'):
                f.analyze(self.output)
        self.assertFalse(self.output.exists())

    def test_cli_has_explicit_outer_report_names(self):
        with mock.patch.object(outer, 'analyze') as analyze:
            outer.main(['--base-readout', 'base.json', '--conditioned-readout', 'c.json',
                        '--donor-readout', 'd.json', '--output', 'outer.json'])
        analyze.assert_called_once_with(Path('base.json'), Path('c.json'), Path('d.json'), Path('outer.json'))


if __name__ == '__main__':
    unittest.main()
