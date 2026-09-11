"""CPU-only contracts for the fixed endpoint E-by-C factorial follow-up."""

import copy
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from . import paired_outer_factorial as outer
from . import checkpoint, paired_weight_patch

training = outer.training


class OuterEffectsTest(unittest.TestCase):
    def test_outer_contrasts_retain_both_conditional_effects_and_donor_gap(self):
        effects = outer.outer_effect(dict(A=-11.0, E=-8.0, C=-7.0, EC=-2.0, D=-1.0))
        self.assertEqual(effects, dict(E_minus_A=3.0, C_minus_A=4.0,
            EC_minus_A=9.0, EC_minus_E=6.0, EC_minus_C=5.0,
            interaction=2.0, D_minus_EC=1.0))

    def test_additive_transfers_have_zero_interaction(self):
        effects = outer.outer_effect(dict(A=-10.0, E=-8.0, C=-7.0, EC=-5.0, D=-6.0))
        self.assertEqual(effects['interaction'], 0.0)
        self.assertEqual(effects['D_minus_EC'], -1.0)

    def test_common_log_probability_offset_does_not_change_contrasts(self):
        values = dict(A=-12.0, E=-8.0, C=-9.0, EC=-4.0, D=-3.0)
        self.assertEqual(outer.outer_effect(values),
                         outer.outer_effect({key: value - 100 for key, value in values.items()}))

    def test_missing_extra_and_nonfinite_cells_are_rejected(self):
        values = dict(A=-11.0, E=-8.0, C=-7.0, EC=-2.0, D=-1.0)
        invalid = [dict(values, AJ=-3.0)]
        invalid.extend({key: value for key, value in values.items() if key != missing}
                       for missing in values)
        invalid.extend(dict(values, **{cell: value}) for cell in values
                       for value in (float('nan'), float('inf'), -float('inf')))
        for cells in invalid:
            with self.subTest(cells=cells), self.assertRaises(ValueError):
                outer.outer_effect(cells)


class EndpointSelectionTest(unittest.TestCase):
    def setUp(self):
        self.rows = [45, 68, 303, 400, 409, 1475, 2797, 3109, 14364, 21733, 45177]
        self.checkpoints = {arm: dict(step=331, path=f'/{arm}/checkpoints/step_331',
            sha256={'weight_0.bin': arm}) for arm in ('original', 'replacement')}
        self.items = []
        for recipient, donor in (('original', 'replacement'), ('replacement', 'original')):
            self.items.append(dict(name=f'step_331_{donor}_to_{recipient}_word_rows',
                kind='embedding_rows', step=331, recipient_arm=recipient, donor_arm=donor,
                recipient_checkpoint=copy.deepcopy(self.checkpoints[recipient]),
                donor_checkpoint=copy.deepcopy(self.checkpoints[donor]),
                tensors=[], embedding_rows=list(self.rows)))
        self.plan = dict(word_piece_ids=list(self.rows), interventions=copy.deepcopy(self.items))

    def test_selects_only_both_fixed_endpoint_directions_without_mutating_plan(self):
        earlier = copy.deepcopy(self.items[0])
        earlier.update(name='earlier', step=100)
        other = dict(name='unrelated_branch', kind='attention_whole_branch', step=331)
        self.plan['interventions'] = [other, earlier, *reversed(self.plan['interventions'])]
        before = copy.deepcopy(self.plan)
        actual = outer.select_endpoints(self.plan)
        self.assertEqual(len(actual), 2)
        self.assertEqual({item['recipient_arm'] for item in actual}, {'original', 'replacement'})
        self.assertEqual({item['step'] for item in actual}, {331})
        self.assertEqual(self.plan, before)

    def test_missing_duplicate_or_same_direction_is_rejected(self):
        invalid = [self.items[:1], self.items + self.items[:1], [self.items[0], self.items[0]]]
        for items in invalid:
            with self.subTest(items=items), self.assertRaises(ValueError):
                outer.select_endpoints(dict(self.plan, interventions=copy.deepcopy(items)))

    def test_unequal_checkpoint_steps_and_nonreciprocal_records_are_rejected(self):
        for role, field, value in (('donor_checkpoint', 'step', 333),
                                  ('recipient_checkpoint', 'step', 330),
                                  ('donor_checkpoint', 'path', '/other/step_331'),
                                  ('donor_checkpoint', 'sha256', {'weight_0.bin': 'changed'})):
            plan = copy.deepcopy(self.plan)
            plan['interventions'][0][role][field] = value
            with self.subTest(role=role, field=field), self.assertRaises(ValueError):
                outer.select_endpoints(plan)

    def test_selected_rows_must_equal_frozen_eleven_row_selection(self):
        for field, value in (('embedding_rows', self.rows[:-1]),
                             ('embedding_rows', list(reversed(self.rows))),
                             ('tensors', ['final_norm.bias'])):
            plan = copy.deepcopy(self.plan)
            plan['interventions'][0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                outer.select_endpoints(plan)
        plan = copy.deepcopy(self.plan)
        plan['word_piece_ids'] = self.rows[:-1]
        for item in plan['interventions']:
            item['embedding_rows'] = self.rows[:-1]
        with self.assertRaises(ValueError):
            outer.select_endpoints(plan)


class PathAndProcessGatesTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='outer-factorial-test-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.recovery = self.root / 'recovery'
        self.recovery.mkdir()
        self.output = self.root / 'outer'
        self.identity = dict(pid=100, start_ticks=200, argv=['prior-analysis'])

    def test_new_output_gate_only_validates_without_creating_files(self):
        actual = outer.require_new_output(self.root, self.recovery, self.output)
        self.assertEqual(actual, (self.root, self.recovery, self.output))
        self.assertFalse(self.output.exists())

    def test_existing_file_directory_or_dangling_symlink_cannot_be_overwritten(self):
        preserved = self.root / 'preserved'
        preserved.write_text('keep')
        dangling = self.root / 'dangling'
        dangling.symlink_to(self.root / 'absent')
        for output in (preserved, self.recovery, dangling):
            with self.subTest(output=output), self.assertRaises((ValueError, FileExistsError)):
                outer.require_new_output(self.root, self.recovery, output)
        self.assertEqual(preserved.read_text(), 'keep')
        self.assertTrue(dangling.is_symlink())
        self.assertFalse((self.root / 'absent').exists())

    def test_nested_outputs_and_symlink_recovery_are_rejected(self):
        alias = self.root / 'alias'
        alias.symlink_to(self.recovery, target_is_directory=True)
        for recovery, output in ((self.recovery, self.recovery / 'nested'),
                                 (self.recovery, self.root / 'new_parent' / 'nested'),
                                 (alias, self.output),
                                 (self.root, self.output)):
            with self.subTest(recovery=recovery, output=output), self.assertRaises(ValueError):
                outer.require_new_output(self.root, recovery, output)
        self.assertFalse(self.output.exists())

    def test_confirmed_exit_is_required(self):
        with mock.patch.object(training, 'process_live', return_value=False) as live:
            outer.require_exited(self.identity)
        live.assert_called_once_with(self.identity)
        with mock.patch.object(training, 'process_live', return_value=True):
            with self.assertRaises(RuntimeError):
                outer.require_exited(self.identity)

    def test_pid_reuse_or_transient_read_failure_never_counts_as_exit(self):
        for error in (RuntimeError('PID reused'), OSError('transient read'), ValueError('bad handle')):
            with self.subTest(error=error), mock.patch.object(training, 'process_live', side_effect=error):
                with self.assertRaises(type(error)):
                    outer.require_exited(self.identity)


class NativeFailureTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='outer-factorial-failure-test-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.output = self.root / 'output'
        self.source = self.root / 'frozen'
        self.source.write_text('original')
        self.frozen = [training.record(self.source)]
        self.enterContext(mock.patch.object(outer, '_suite_inputs',
            return_value=(self.root / 'cases.json', '/batch', '/rows', 3)))
        self.inputs = self.enterContext(mock.patch.object(
            outer.screen.execution, 'checkpoint_inputs', return_value=[]))
        self.native = self.enterContext(mock.patch.object(outer.screen, 'run_native'))
        self.reader = self.enterContext(mock.patch.object(outer.reader, 'analyze',
            return_value={'files': []}))

    def run_suite(self):
        return outer.run_suite(self.root, {}, 'main', '/C', '/EC', [1],
                               self.output, '/probe', {}, self.frozen)

    def test_frozen_source_mutation_blocks_before_native_launch(self):
        self.source.write_text('modified')
        with self.assertRaisesRegex(ValueError, 'frozen evidence'):
            self.run_suite()
        self.inputs.assert_not_called()
        self.native.assert_not_called()
        self.reader.assert_not_called()

    def test_native_failure_never_publishes_a_successful_readout(self):
        self.native.side_effect = RuntimeError('native child failed')
        with self.assertRaisesRegex(RuntimeError, 'native child failed'):
            self.run_suite()
        self.native.assert_called_once()
        self.reader.assert_not_called()
        self.assertFalse((self.output / 'readout.json').exists())

    def test_frozen_source_is_rechecked_after_native_execution(self):
        self.native.side_effect = lambda *args, **kwargs: self.source.write_text('modified')
        with self.assertRaisesRegex(ValueError, 'frozen evidence'):
            self.run_suite()
        self.native.assert_called_once()

    def test_validation_failure_is_recorded_without_automatic_restart(self):
        previous = self.root / 'recovery'
        previous.mkdir()
        with mock.patch.object(outer, 'validate_recovery', return_value=({}, {}, [])), \
             mock.patch.object(outer.screen.planner, 'prepare',
                               side_effect=ValueError('frozen plan differs')):
            with self.assertRaisesRegex(ValueError, 'frozen plan differs'):
                outer.run(self.root, previous, self.output)
        failure = training.read_json(self.output / 'failure.json')
        self.assertEqual(failure['type'], 'ValueError')
        self.assertEqual(failure['completed'], [])
        self.assertIs(failure['no_automatic_restart'], True)
        self.assertIs(failure['training_restarted'], False)
        self.assertIs(failure['goal_completion_claimed'], False)
        self.assertFalse((self.output / 'summary.json').exists())
        self.native.assert_not_called()


class ModelProvenanceTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='outer-factorial-model-test-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.directory = self.root / 'direction'
        self.directory.mkdir()
        # Keep the production 100-tensor layout, but use tiny CPU arrays.
        self.config = checkpoint.GPT2Config(vocab_size=16, padded_vocab_size=20,
            context_length=4, n_layers=8, d_model=2, n_heads=1, d_ff=4)
        self.specs = checkpoint.tensor_manifest(self.config)
        self.sources = {}
        for arm, shift in (('original', 0), ('replacement', 1000)):
            path = self.root / arm / 'step_331'
            path.mkdir(parents=True)
            for spec in self.specs:
                values = np.arange(np.prod(spec.shape), dtype='<f4').reshape(spec.shape)
                values = values + np.float32(shift + 10 * spec.index)
                values.tofile(path / spec.filename)
            self.sources[arm] = dict(step=331, path=str(path), sha256=self.hashes(path))
        self.item = dict(name='step_331_replacement_to_original_word_rows',
            kind='embedding_rows', step=331, recipient_arm='original', donor_arm='replacement',
            recipient_checkpoint=self.sources['original'], donor_checkpoint=self.sources['replacement'],
            tensors=[], embedding_rows=list(range(11)))
        create_patch = paired_weight_patch.create_patch

        def tiny_patch(original, replacement, output, **kwargs):
            kwargs['config'] = self.config
            return create_patch(original, replacement, output, **kwargs)

        self.patch = self.enterContext(mock.patch.object(
            outer.screen.paired_weight_patch, 'create_patch', side_effect=tiny_patch))
        self.native = self.enterContext(mock.patch.object(outer.screen, 'run_native',
            side_effect=AssertionError('CPU model preparation must not launch a native process')))

    def hashes(self, path):
        return {spec.filename: checkpoint.sha256_file(path / spec.filename) for spec in self.specs}

    def test_c_then_ec_are_exact_staged_patches_and_donor_copy_is_independent(self):
        result = outer.prepare_models(self.item, self.directory)
        self.assertEqual(set(result), {'C', 'EC', 'donor_copy'})
        self.assertEqual(self.patch.call_count, 3)
        a = Path(self.sources['original']['path'])
        d = Path(self.sources['replacement']['path'])
        c = self.directory / 'C' / 'step_331'
        ec = self.directory / 'EC' / 'step_331'
        donor_copy = self.directory / 'donor_copy' / 'step_331'
        calls = self.patch.call_args_list
        self.assertEqual(tuple(map(Path, calls[0].args[:3])), (a, d, c))
        self.assertEqual(set(calls[0].kwargs['tensors']), {spec.name for spec in self.specs[1:]})
        self.assertEqual(len(calls[0].kwargs['tensors']), 99)
        self.assertFalse(calls[0].kwargs.get('embedding_rows', []))
        self.assertEqual(tuple(map(Path, calls[1].args[:3])), (c, d, ec))
        self.assertEqual(calls[1].kwargs['embedding_rows'], self.item['embedding_rows'])
        self.assertFalse(calls[1].kwargs.get('tensors', []))
        self.assertEqual(tuple(map(Path, calls[2].args[:3])), (d, d, donor_copy))
        self.assertFalse(calls[2].kwargs.get('tensors', []))
        self.assertFalse(calls[2].kwargs.get('embedding_rows', []))
        for spec in self.specs:
            expected = d if spec.index else a
            self.assertEqual((c / spec.filename).read_bytes(), (expected / spec.filename).read_bytes())
            if spec.index:
                self.assertEqual((ec / spec.filename).read_bytes(), (d / spec.filename).read_bytes())
        actual_embedding = np.fromfile(ec / 'weight_0.bin', dtype='<f4').reshape(self.specs[0].shape)
        expected_embedding = np.fromfile(a / 'weight_0.bin', dtype='<f4').reshape(self.specs[0].shape)
        donor_embedding = np.fromfile(d / 'weight_0.bin', dtype='<f4').reshape(self.specs[0].shape)
        expected_embedding[self.item['embedding_rows']] = donor_embedding[self.item['embedding_rows']]
        self.assertEqual(actual_embedding.tobytes(), expected_embedding.tobytes())
        provenance = training.read_json(ec / 'patch.json')
        self.assertEqual(Path(provenance['sources']['original']['path']), c)
        self.assertEqual(provenance['selection']['tensors'], [])
        self.assertEqual(provenance['selection']['embedding_rows'], self.item['embedding_rows'])
        self.assertEqual(self.hashes(donor_copy), self.sources['replacement']['sha256'])
        for spec in self.specs:
            source_info, copy_info = (d / spec.filename).stat(), (donor_copy / spec.filename).stat()
            self.assertNotEqual((source_info.st_dev, source_info.st_ino),
                                (copy_info.st_dev, copy_info.st_ino))
        for source in self.sources.values():
            self.assertEqual(self.hashes(Path(source['path'])), source['sha256'])
        self.native.assert_not_called()

    def test_donor_copy_must_match_all_hundred_declared_donor_hashes(self):
        self.item['donor_checkpoint']['sha256']['weight_99.bin'] = 'wrong'
        with self.assertRaises(ValueError):
            outer.prepare_models(self.item, self.directory)
        self.native.assert_not_called()


if __name__ == '__main__':
    unittest.main()
