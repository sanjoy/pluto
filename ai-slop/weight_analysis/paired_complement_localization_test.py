"""Small CPU-only checks for exact complement patches and factorial arithmetic.

The fixtures deliberately differ in every weight tensor. A validation test
therefore cannot pass by trusting a patch's descriptive selection alone.
Nothing here starts a training process, a native executable, or a GPU kernel.
"""

from dataclasses import replace
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from . import paired_complement_localization as localization
from . import paired_branch_readout as branch
from .paired_branch_readout_test import Fixture as NativeLedgerFixture
from .checkpoint import GPT2Checkpoint, GPT2Config, sha256_file, tensor_manifest


CONFIG = GPT2Config(vocab_size=16, padded_vocab_size=16, context_length=4,
                    n_layers=2, d_model=4, n_heads=2, d_ff=8)


class ComplementArithmeticTest(unittest.TestCase):
    def test_production_groups_partition_every_nonembedding_tensor(self):
        groups = localization.groups()
        self.assertEqual(set(groups), {'Q', 'H', 'L'})
        self.assertEqual({name: len(values) for name, values in groups.items()},
                         {'Q': 3, 'H': 48, 'L': 48})
        self.assertEqual(set(groups['Q']), {'position_embedding.weight',
                                          'final_norm.scale', 'final_norm.bias'})
        flattened = [name for group in groups.values() for name in group]
        self.assertEqual(len(flattened), len(set(flattened)))
        self.assertEqual(set(flattened), {spec.name for spec in tensor_manifest()
                                        if spec.name != 'token_embedding.weight'})
        for group, blocks in (('H', range(4)), ('L', range(4, 8))):
            self.assertEqual(set(groups[group]),
                             {spec.name for spec in tensor_manifest()
                              if any(spec.name.startswith(f'blocks.{block}.')
                                     for block in blocks)})

    def test_toy_group_boundaries_and_invalid_odd_layer_counts(self):
        groups = localization.groups(CONFIG)
        self.assertEqual(len(groups['Q']), 3)
        self.assertEqual(len(groups['H']), 12)
        self.assertEqual(len(groups['L']), 12)
        self.assertTrue(all(name.startswith('blocks.0.') for name in groups['H']))
        self.assertTrue(all(name.startswith('blocks.1.') for name in groups['L']))
        for layers in (1, 3, 5):
            with self.subTest(layers=layers), self.assertRaises(ValueError):
                localization.groups(replace(CONFIG, n_layers=layers))

    def test_cube_has_exactly_the_eight_predeclared_subsets(self):
        self.assertEqual(localization.cube_subsets(), {
            'E': (), 'Q': ('Q',), 'H': ('H',), 'L': ('L',),
            'QH': ('Q', 'H'), 'QL': ('Q', 'L'), 'HL': ('H', 'L'),
            'EC': ('Q', 'H', 'L')})

    def test_factorial_effects_include_the_conditioned_three_way_term(self):
        # Construct the full cube from a polynomial rather than reproducing
        # the implementation's finite-difference expressions in this fixture.
        values = {}
        for name, subset in localization.cube_subsets().items():
            q, h, l = (float(group in subset) for group in ('Q', 'H', 'L'))
            values[name] = (-100.0 + 2*q + 3*h + 5*l + 7*q*h + 11*q*l
                            + 13*h*l + 17*q*h*l)
        scored = localization.score_cube(values)
        self.assertEqual(scored['additions'], {'Q': 2.0, 'H': 3.0, 'L': 5.0})
        self.assertEqual(scored['removals'], {'Q': 37.0, 'H': 40.0, 'L': 46.0})
        self.assertEqual(scored['pair_interactions'], {
            'QH': {'absent': 7.0, 'present': 24.0},
            'QL': {'absent': 11.0, 'present': 28.0},
            'HL': {'absent': 13.0, 'present': 30.0}})
        self.assertEqual(scored['three_way'], 17.0)

    def test_constant_cube_has_zero_effects_and_signed_effects_are_preserved(self):
        values = {name: -3.5 for name in localization.cube_subsets()}
        scored = localization.score_cube(values)
        self.assertEqual(scored['additions'], dict.fromkeys(('Q', 'H', 'L'), 0.0))
        self.assertEqual(scored['removals'], dict.fromkeys(('Q', 'H', 'L'), 0.0))
        self.assertEqual(scored['three_way'], 0.0)
        for pair in scored['pair_interactions'].values():
            self.assertEqual(pair, {'absent': 0.0, 'present': 0.0})
        for name, subset in localization.cube_subsets().items():
            values[name] = -3.5 - 2.0 * ('H' in subset)
        scored = localization.score_cube(values)
        self.assertEqual(scored['additions']['H'], -2.0)
        self.assertEqual(scored['removals']['H'], -2.0)

    def test_missing_extra_nonfinite_boolean_and_nonnumeric_scores_rejected(self):
        values = {name: -1.0 for name in localization.cube_subsets()}
        for invalid in ({k: v for k, v in values.items() if k != 'E'},
                        dict(values, unexpected=-1.0)):
            with self.assertRaises(ValueError):
                localization.score_cube(invalid)
        for value in (float('nan'), float('inf'), -float('inf'), True,
                      np.bool_(False), '-1.0', None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                localization.score_cube(dict(values, EC=value))


class ComplementPatchTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='complement-localization-test-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.a = self.root / 'recipient'
        self.d = self.root / 'donor'
        self.output = self.root / 'patch'
        self.rows = [1, 2]
        self.specs = tensor_manifest(CONFIG)
        for path, shift in ((self.a, 0), (self.d, 10000)):
            path.mkdir()
            for spec in self.specs:
                values = (np.arange(np.prod(spec.shape), dtype='<f4')
                          + np.float32(100 * spec.index + shift))
                values.reshape(spec.shape).tofile(path / spec.filename)
        self.original_hashes = {str(path): self.hashes(path) for path in (self.a, self.d)}

    def hashes(self, path):
        return {spec.filename: sha256_file(path / spec.filename) for spec in self.specs}

    def build(self, subset=(), rows=None, output=None):
        return localization.build_model(self.a, self.d, output or self.output,
                                        subset, self.rows if rows is None else rows,
                                        config=CONFIG)

    def validate(self, subset=(), rows=None, output=None):
        return localization.validate_model((output or self.output) / 'patch.json', self.a, self.d,
                                           subset, self.rows if rows is None else rows,
                                           config=CONFIG)

    def read_metadata(self):
        return json.loads((self.output / 'patch.json').read_text())

    def write_metadata(self, value):
        (self.output / 'patch.json').write_text(json.dumps(value))

    def test_all_eight_models_are_exact_independent_copies_and_sources_unchanged(self):
        recipient = GPT2Checkpoint(self.a, CONFIG)
        donor = GPT2Checkpoint(self.d, CONFIG)
        groups = localization.groups(CONFIG)
        for name, subset in localization.cube_subsets().items():
            output = self.root / name
            self.build(subset, output=output)
            records = self.validate(subset, output=output)
            self.assertTrue(records)
            model = GPT2Checkpoint(output, CONFIG, check_finite=True)
            selected = {tensor for group in subset for tensor in groups[group]}
            for spec in self.specs:
                expected = np.array(donor[spec.name] if spec.name in selected
                                    else recipient[spec.name])
                if spec.name == 'token_embedding.weight':
                    expected[self.rows] = donor[spec.name][self.rows]
                self.assertEqual(model[spec.name].tobytes(), expected.tobytes(),
                                 f'{name}: {spec.name}')
                dest = (output / spec.filename).stat()
                for source in (self.a, self.d):
                    origin = (source / spec.filename).stat()
                    self.assertNotEqual((dest.st_dev, dest.st_ino),
                                        (origin.st_dev, origin.st_ino))
        for source in (self.a, self.d):
            self.assertEqual(self.hashes(source), self.original_hashes[str(source)])

    def test_wrong_selection_or_forged_selection_cannot_pass_from_metadata_alone(self):
        self.build(())
        with self.assertRaises(ValueError):
            self.validate(('H',))
        metadata = self.read_metadata()
        metadata['selection']['tensors'] = sorted(localization.groups(CONFIG)['H'])
        self.write_metadata(metadata)
        with self.assertRaises(ValueError):
            self.validate(('H',))

    def test_corrupted_weight_rejected_even_with_a_forged_matching_output_hash(self):
        self.build(('Q',))
        path = self.output / 'weight_1.bin'
        values = np.fromfile(path, dtype='<f4')
        values[0] += np.float32(1)
        values.tofile(path)
        with self.assertRaises(ValueError):
            self.validate(('Q',))
        metadata = self.read_metadata()
        metadata['output']['weights_sha256'][path.name] = sha256_file(path)
        self.write_metadata(metadata)
        with self.assertRaises(ValueError):
            self.validate(('Q',))

    def test_nonfinite_weight_is_rejected_even_when_its_hash_is_updated(self):
        self.build(())
        path = self.output / 'weight_0.bin'
        values = np.fromfile(path, dtype='<f4')
        values[-1] = np.nan
        values.tofile(path)
        metadata = self.read_metadata()
        metadata['output']['weights_sha256'][path.name] = sha256_file(path)
        self.write_metadata(metadata)
        with self.assertRaises(ValueError):
            self.validate(())

    def test_missing_weight_and_unexpected_entries_are_rejected(self):
        self.build(())
        path = self.output / 'weight_1.bin'
        saved = path.read_bytes()
        path.unlink()
        with self.assertRaises(ValueError):
            self.validate(())
        path.write_bytes(saved)
        (self.output / 'unexpected.txt').write_text('not checkpoint metadata')
        with self.assertRaises(ValueError):
            self.validate(())

    def test_source_hardlink_is_rejected_even_when_the_expected_bytes_match(self):
        self.build(())
        target = self.output / 'weight_1.bin'
        target.unlink()
        os.link(self.a / target.name, target)
        with self.assertRaises(ValueError):
            self.validate(())

    def test_symlinked_weight_is_not_accepted_as_an_independent_copy(self):
        self.build(())
        target = self.output / 'weight_1.bin'
        target.unlink()
        target.symlink_to(self.a / target.name)
        with self.assertRaises(ValueError):
            self.validate(())

    def test_source_bindings_and_source_hashes_are_verified(self):
        self.build(())
        baseline = self.read_metadata()
        changed = self.read_metadata()
        changed['sources']['original']['path'] = str(self.d)
        self.write_metadata(changed)
        with self.assertRaises(ValueError):
            self.validate(())
        self.write_metadata(baseline)
        path = self.d / 'weight_1.bin'
        values = np.fromfile(path, dtype='<f4')
        values[0] += np.float32(1)
        values.tofile(path)
        with self.assertRaises(ValueError):
            self.validate(())

    def test_wrong_requested_embedding_rows_are_rejected(self):
        self.build(())
        with self.assertRaises(ValueError):
            self.validate((), rows=[1, 3])

    def test_invalid_group_names_and_rows_fail_before_output_creation(self):
        for subset in (('E',), ('unknown',), ('Q', 'Q'), ('EC',)):
            with self.subTest(subset=subset), self.assertRaises(ValueError):
                self.build(subset)
            self.assertFalse(self.output.exists())
        for rows in ([-1], [16], [1, 1], [True], [1.0]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                self.build((), rows=rows)
            self.assertFalse(self.output.exists())

    def test_existing_output_is_preserved(self):
        self.build(('Q',))
        before = {path.name: path.read_bytes() for path in self.output.iterdir()}
        with self.assertRaises(FileExistsError):
            self.build(('H',))
        self.assertEqual(before, {path.name: path.read_bytes()
                                 for path in self.output.iterdir()})


class ComplementNativeValidationTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='complement-native-test-')
        self.addCleanup(temporary.cleanup)
        self.fixture = NativeLedgerFixture(Path(temporary.name))
        # This established fixture has one block and no localization groups.
        # Its independently validated source map is sufficient for load_native,
        # which checks output/ledger integrity without classifying the patch.
        records = {}
        validated = branch._patch(self.fixture.native_paths['patched'] / 'patch.json',
                                  records, self.fixture.config)
        self.model = dict(paths={k: str(v) for k, v in validated['paths'].items()},
                          hashes=validated['hashes'],
                          weight_records=validated['weight_records'],
                          records=list(records.values()))

    def load(self, suite='main'):
        f = self.fixture
        return localization.load_native(self.model, f.cases[suite], suite,
                                         f.scores['patched'][suite],
                                         f.executions['patched'][suite], f.config)

    def reference(self, measured):
        """Synthetic FP64 scoring report with explicit original-row identities."""
        cases = measured['cases']
        result = dict(format='pluto-embedding-factorial-readout-v1',
                      cases=copy.deepcopy(cases['record']),
                      case_kind_selection=None, per_case=[])
        for case in cases['plan']['cases']:
            index = case['case_index']
            rows = case['scored_rows']
            item = {key: copy.deepcopy(case[key]) for key in
                    ('kind', 'split', 'context_id', 'prefix_domain', 'target', 'target_ids')}
            item['source_case_index'] = index
            item['predictions'] = [dict(
                prediction_row=row, causal_prefix_length=row+1,
                causal_prefix_sha256=hashlib.sha256(
                    cases['packed'][0, index, :row+1].tobytes()).hexdigest())
                for row in rows]
            item['cells'] = {'JJ': dict(
                token_nll=[float(measured['losses'][index, row]) for row in rows],
                argmax_ids=[int(measured['argmax'][index, row]) for row in rows])}
            result['per_case'].append(item)
        return result

    def test_native_main_and_supplemental_require_complete_successful_ledgers(self):
        for suite in ('main', 'supplemental'):
            with self.subTest(suite=suite):
                measured = self.load(suite)
                self.assertEqual(measured['losses'].shape,
                                 (self.fixture.plans[suite]['case_count'], 16))
                self.assertEqual(measured['argmax'].shape, measured['losses'].shape)
                self.assertTrue(measured['records'])

    def test_missing_or_failed_execution_is_not_valid_native_evidence(self):
        path = self.fixture.executions['patched']['main']
        saved = path.read_text()
        path.unlink()
        with self.assertRaises(ValueError):
            self.load()
        value = json.loads(saved)
        value['returncode'] = 127
        path.write_text(json.dumps(value))
        with self.assertRaises(ValueError):
            self.load()

    def test_tampered_output_and_nonfinite_output_with_updated_ledger_fail(self):
        path = self.fixture.scores['patched']['main'] / 'losses.f32.bin'
        original = path.read_bytes()
        data = np.frombuffer(original, dtype='<f4').copy()
        data[0] += np.float32(1)
        data.tofile(path)
        with self.assertRaises(ValueError):
            self.load()
        data[0] = np.nan
        data.tofile(path)
        self.fixture.refresh_execution('patched', 'main')
        with self.assertRaises(ValueError):
            self.load()

    def test_argmax_must_not_depend_on_target_for_identical_causal_prefix(self):
        path = self.fixture.scores['patched']['main'] / 'argmax.i32.bin'
        winners = np.fromfile(path, dtype='<i4').reshape(-1, 16)
        for case in self.fixture.plans['main']['cases']:
            if case['target'] == 'Nuveth':
                winners[case['case_index'], case['scored_rows'][0]] = 2
        winners.tofile(path)
        self.fixture.refresh_execution('patched', 'main')
        with self.assertRaises(ValueError):
            self.load()

    def test_fp64_reference_agreement_and_small_rounding_error_are_explicit(self):
        measured = self.load()
        reference = self.reference(measured)
        result = localization.compare_fp64_reference(measured, reference, 'JJ')
        count = sum(len(case['target_ids']) for case in measured['cases']['plan']['cases'])
        self.assertEqual(result['checked_predictions'], count)
        self.assertEqual(result['maximum_absolute_nll_error'], 0.0)
        self.assertFalse(result['exact_byte_equality_claimed'])
        reference['per_case'][0]['cells']['JJ']['token_nll'][0] += 5e-6
        result = localization.compare_fp64_reference(measured, reference, 'JJ')
        self.assertGreater(result['maximum_absolute_nll_error'], 0)
        self.assertLess(result['maximum_absolute_nll_error'], result['absolute_tolerance'])

    def test_fp64_reference_excess_error_or_different_argmax_is_rejected(self):
        measured = self.load()
        reference = self.reference(measured)
        baseline = reference['per_case'][0]['cells']['JJ']['token_nll'][0]
        reference['per_case'][0]['cells']['JJ']['token_nll'][0] += 2 * (
            localization.FP32_ABSOLUTE_TOLERANCE
            + localization.FP32_RELATIVE_TOLERANCE * abs(baseline))
        with self.assertRaises(ValueError):
            localization.compare_fp64_reference(measured, reference, 'JJ')
        reference = self.reference(measured)
        reference['per_case'][0]['cells']['JJ']['argmax_ids'][0] = 2
        with self.assertRaises(ValueError):
            localization.compare_fp64_reference(measured, reference, 'JJ')

    def test_fp64_reference_prefix_and_target_identity_are_checked(self):
        measured = self.load()
        for field in ('prefix', 'target'):
            reference = self.reference(measured)
            if field == 'prefix':
                reference['per_case'][0]['predictions'][0]['causal_prefix_sha256'] = '0' * 64
            else:
                reference['per_case'][0]['target_ids'][0] += 1
            with self.subTest(field=field), self.assertRaises(ValueError):
                localization.compare_fp64_reference(measured, reference, 'JJ')

    def test_fp64_reference_cannot_omit_or_duplicate_selected_cases(self):
        measured = self.load()
        for change in ('empty', 'missing', 'duplicate'):
            reference = self.reference(measured)
            if change == 'empty':
                reference['per_case'] = []
            elif change == 'missing':
                reference['per_case'].pop()
            else:
                reference['per_case'].append(copy.deepcopy(reference['per_case'][0]))
            with self.subTest(change=change), self.assertRaises(ValueError):
                localization.compare_fp64_reference(measured, reference, 'JJ')

    def test_fp64_reference_explicit_kind_filter_has_exact_selected_coverage(self):
        measured = self.load()
        reference = self.reference(measured)
        reference['case_kind_selection'] = 'word'
        reference['per_case'] = [item for item in reference['per_case']
                                 if item['kind'] == 'word']
        result = localization.compare_fp64_reference(measured, reference, 'JJ')
        self.assertEqual(result['checked_predictions'], 3 * len(reference['per_case']))
        reference['per_case'].pop()
        with self.assertRaises(ValueError):
            localization.compare_fp64_reference(measured, reference, 'JJ')

    def test_fp64_reference_must_bind_to_the_same_case_file(self):
        measured = self.load()
        reference = self.reference(measured)
        reference['cases']['sha256'] = '0' * 64
        with self.assertRaises(ValueError):
            localization.compare_fp64_reference(measured, reference, 'JJ')

    def test_native_copy_parity_requires_every_loss_bit_and_same_case_source(self):
        measured = self.load()
        copied = copy.deepcopy(measured)
        result = localization.assert_native_copy_parity(measured, copied)
        self.assertTrue(result['all_context_loss_and_argmax_byte_equal'])
        self.assertEqual(result['predictions'], measured['losses'].size)
        # Mutate an unscored context position by one FP32 bit: checking only
        # the target rows or using a numerical tolerance would miss this.
        copied['losses'].view('<u4')[0, 0] ^= np.uint32(1)
        with self.assertRaises(ValueError):
            localization.assert_native_copy_parity(measured, copied)
        copied = copy.deepcopy(measured)
        copied['cases']['record']['sha256'] = '0' * 64
        with self.assertRaises(ValueError):
            localization.assert_native_copy_parity(measured, copied)


class ComplementHandoffTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='complement-handoff-test-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.outer = self.root / 'outer'
        self.outer.mkdir()
        (self.outer / 'plan').mkdir()
        self.gpu = {'uuid': 'CPU-fixture-only'}
        self.write(self.root / 'request.json', {'gpu': self.gpu})
        frozen = self.root / 'frozen_input'
        frozen.write_bytes(b'unchanged CPU evidence')
        binary = self.root / 'native_binary'
        binary.write_bytes(b'not an executable')
        record = localization.native._record
        self.inputs = [record(frozen), record(binary), record(self.root / 'request.json')]
        endpoints = {name: dict(step=331, path=str(self.root / name / 'step_331'),
                               sha256={'weight_0.bin': 'synthetic'})
                     for name in ('original', 'replacement')}
        self.items = [dict(name=donor + '_to_' + recipient,
                           recipient_arm=recipient, donor_arm=donor,
                           recipient_checkpoint=endpoints[recipient],
                           donor_checkpoint=endpoints[donor], embedding_rows=[1, 2])
                      for recipient, donor in (('original', 'replacement'),
                                               ('replacement', 'original'))]
        self.identity = dict(pid=12345, start_ticks=54321, argv=[
            'python', '-m', 'weight_analysis.paired_outer_factorial',
            '--root', str(self.root), '--output', str(self.outer)])
        self.request = dict(format=localization.outer.FORMAT, amendment_root=str(self.root),
                            fixed_step=331, training_restarted=False,
                            runner_identity=self.identity, frozen_inputs=self.inputs,
                            directions=self.items, binaries={'probe': record(binary)},
                            runtime=dict(complete=True, runtime_dlopen_covered=False,
                                         environment={'LD_LIBRARY_PATH': str(self.root / 'runtime')},
                                         frozen_records=[record(frozen)]))
        self.write(self.outer / 'request.json', self.request)
        self.write(self.outer / 'plan/plan.json', {'fixture_plan': True})
        (self.outer / 'result.bin').write_bytes(b'completed synthetic output')
        completed = [direction + '/' + suite
                     for direction in ('replacement_to_original', 'original_to_replacement')
                     for suite in localization.outer.SUITES]
        self.write(self.outer / 'state.json', dict(format=localization.outer.FORMAT,
                   phase='complete', runner_pid=self.identity['pid'], completed=completed))
        self.summary = dict(format=localization.outer.FORMAT, complete=True,
                            amendment_root=str(self.root), fixed_step=331,
                            training_restarted=False, goal_completion_claimed=False,
                            completed=completed,
                            frozen_inputs=[*self.inputs, record(self.outer / 'request.json')],
                            artifacts=[record(path) for path in sorted(self.outer.rglob('*'))
                                       if path.is_file()])
        self.write(self.outer / 'summary.json', self.summary)
        self.live = self.enterContext(mock.patch.object(localization.training, 'process_live',
                                                        return_value=False))
        self.idle = self.enterContext(mock.patch.object(localization.training, 'require_idle_gpu'))
        self.endpoints = self.enterContext(mock.patch.object(localization.outer, 'select_endpoints',
                                                             return_value=self.items))
        # The handoff's fixed production layout is separately tested; these
        # tests authenticate actual tiny ledgers without making 100 huge files.
        self.weights = self.enterContext(mock.patch.object(branch, '_weights', return_value=[]))

    @staticmethod
    def write(path, value):
        path.write_text(json.dumps(value))

    def handoff(self):
        return localization.validate_outer_handoff(self.root, self.outer)

    def test_complete_handoff_checks_real_artifacts_and_observes_exit_twice(self):
        result = self.handoff()
        self.assertEqual(result['items'], self.items)
        self.assertEqual(self.live.call_count, 2)
        self.idle.assert_called_once_with(self.gpu)
        self.assertEqual(self.weights.call_count, 4)
        localization.training.verify_records(result['records'])
        self.assertTrue(any(item['path'].endswith('/result.bin')
                            for item in result['records']))

    def test_invalid_request_or_runner_identity_fails_before_process_or_gpu_checks(self):
        for field in ('format', 'root', 'pid', 'argv'):
            request = copy.deepcopy(self.request)
            if field == 'format':
                request['format'] = 'unrelated-controller'
            elif field == 'root':
                request['amendment_root'] = str(self.root / 'elsewhere')
            elif field == 'pid':
                request['runner_identity']['pid'] = True
            else:
                request['runner_identity']['argv'][2] = 'unrelated.module'
            self.write(self.outer / 'request.json', request)
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.handoff()
        self.live.assert_not_called()
        self.idle.assert_not_called()

    def test_live_upstream_forbids_summary_reads_and_gpu_checks(self):
        self.live.return_value = True
        (self.outer / 'summary.json').unlink()
        with mock.patch.object(localization.native, '_json', wraps=localization.native._json) as read:
            with self.assertRaisesRegex(ValueError, 'still live'):
                self.handoff()
        self.assertNotIn(mock.call(self.outer / 'summary.json'), read.call_args_list)
        self.idle.assert_not_called()
        self.weights.assert_not_called()

    def test_missing_or_incomplete_summary_does_not_authorize_handoff(self):
        path = self.outer / 'summary.json'
        path.unlink()
        with self.assertRaises((FileNotFoundError, ValueError)):
            self.handoff()
        self.write(path, dict(self.summary, complete=False))
        with self.assertRaises(ValueError):
            self.handoff()
        self.idle.assert_not_called()

    def test_unreadable_or_reused_process_identity_fails_closed(self):
        for error in (OSError('transient proc read'), RuntimeError('PID reused')):
            self.live.side_effect = error
            with self.subTest(error=error), self.assertRaises(type(error)):
                self.handoff()
        self.idle.assert_not_called()
        self.weights.assert_not_called()

    def test_second_exit_observation_must_also_confirm_termination(self):
        self.live.side_effect = [False, True]
        with self.assertRaisesRegex(ValueError, 'became live'):
            self.handoff()
        self.idle.assert_not_called()

    def test_changed_artifact_or_unlisted_file_rejected_before_gpu_check(self):
        path = self.outer / 'result.bin'
        before = path.read_bytes()
        path.write_bytes(b'changed after completion')
        with self.assertRaises(ValueError):
            self.handoff()
        path.write_bytes(before)
        (self.outer / 'unlisted.txt').write_text('not frozen evidence')
        with self.assertRaisesRegex(ValueError, 'inventory'):
            self.handoff()
        self.idle.assert_not_called()

    def test_busy_gpu_rejects_completed_cpu_handoff(self):
        self.idle.side_effect = RuntimeError('GPU is busy')
        with self.assertRaisesRegex(RuntimeError, 'busy'):
            self.handoff()
        self.assertEqual(self.live.call_count, 2)
        self.idle.assert_called_once_with(self.gpu)

    def test_existing_plan_output_is_preserved_before_any_handoff_inspection(self):
        output = self.root / 'existing_plan'
        output.mkdir()
        marker = output / 'preserve'
        marker.write_bytes(b'prior evidence')
        with mock.patch.object(localization, 'validate_outer_handoff') as handoff:
            with self.assertRaises(ValueError):
                localization.prepare_plan(self.root, self.outer, output)
        handoff.assert_not_called()
        self.assertEqual(marker.read_bytes(), b'prior evidence')


class ComplementAggregationTest(unittest.TestCase):
    def test_domain_aliases_and_fourth_token_variants_do_not_reweight_words(self):
        items = []
        for domain in ('original', 'replacement'):
            for fourth, fourth_logp in ((9, -2.0), (10, -4.0)):
                items.append(dict(suite='supplemental', kind='word_next_native',
                    split='training', spelling_variant='title', prefix_domain=domain,
                    word_has_leading_space=False, target='Exeunt',
                    prefix_sha256='same-causal-prefix', prefix_length=4,
                    target_ids=[11, 12, 13, fourth],
                    cells={cell: dict(token_log_probability=[-1.0, -0.5, -0.25, fourth_logp])
                           for cell in localization.cube_subsets()}))
        groups = localization._aggregates(items)
        self.assertEqual({item['prefix_domain'] for item in groups},
                         {'original', 'replacement', 'deduplicated_all'})
        aggregate = next(item for item in groups if item['prefix_domain'] == 'deduplicated_all')
        word = aggregate['metrics']['word_three']
        self.assertEqual(word['unique_event_count'], 1)
        self.assertEqual(word['alias_case_count'], 4)
        self.assertEqual(word['cells']['E']['mean_log_probability'], -1.75)
        self.assertEqual(aggregate['metrics']['selected_sequence']['unique_event_count'], 2)
        self.assertEqual(aggregate['metrics']['exact_next_native_token']['unique_event_count'], 2)
        self.assertEqual(aggregate['metrics']['exact_next_native_token']['cells']['E']
                         ['mean_log_probability'], -3.0)
        self.assertEqual(word['effects']['three_way'], 0.0)

    def test_aliases_with_different_scores_fail_instead_of_silent_deduplication(self):
        item = dict(suite='main', kind='word', split='training', prefix_domain='original',
                    target='Exeunt', prefix_sha256='same', prefix_length=4,
                    target_ids=[11, 12, 13],
                    cells={cell: dict(token_log_probability=[-1., -2., -3.])
                           for cell in localization.cube_subsets()})
        alias = copy.deepcopy(item)
        alias['prefix_domain'] = 'replacement'
        alias['cells']['Q']['token_log_probability'][0] -= 0.25
        with self.assertRaises(ValueError):
            localization._aggregates([item, alias])


if __name__ == '__main__':
    unittest.main()
