"""Tiny CPU-only schema fixtures: no executable or native GPU result is made.

The fabricated arrays test byte/geometry/ledger gates, not the model's numerical
correctness. Genuine case authentication and model patch validation use existing
readers; simulated trace files are explicitly local temporary test artifacts.
"""

import copy
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from . import paired_case_trace as reader
from .paired_branch_readout_test import Fixture


def write_json(path, value):
    Path(path).write_text(json.dumps(value, allow_nan=False))


class CasesFixture(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='paired-case-trace-test-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.fixture = Fixture(self.root)
        self.config = self.fixture.config
        self.cases = reader.core.branch._cases(self.fixture.cases['main'], 'main', {}, self.config)


class BindingTest(CasesFixture):
    def test_exact_teacher_forced_prefix_all_target_positions(self):
        for case_index in (0, 1, 8):
            case = self.cases['plan']['cases'][case_index]
            for position in range(3):
                result = reader.bind_case(self.cases, case_index, position, self.config)
                expected = [*self.cases['packed'][0, case_index, :4].tolist(), *case['target_ids'][:position]]
                self.assertEqual(result['token_ids'], expected)
                self.assertEqual(result['target_id'], case['target_ids'][position])
                self.assertEqual(result['selected_row'], 3+position)
                self.assertEqual(result['identity']['causal_prefix_sha256'], hashlib.sha256(
                    np.asarray(expected, dtype='<i4').tobytes()).hexdigest())
                self.assertFalse(result['generated_sampling_event'])

    def test_supplemental_exact_fourth_target_and_shared_piece(self):
        cases = reader.core.branch._cases(self.fixture.cases['supplemental'], 'supplemental', {}, self.config)
        result = reader.bind_case(cases, 0, 3, self.config)
        self.assertEqual(result['identity']['suite'], 'supplemental')
        self.assertEqual(result['target_id'], 9)
        self.assertEqual(result['token_ids'], [1, 2, 3, 4, 11, 12, 13])
        self.assertEqual(reader.bind_case(cases, 8, 0, self.config)['identity']['kind'], 'shared_piece')

    def test_amended_lowercase_retains_original_case_identity(self):
        self.fixture.amend_capitalization()
        cases = reader.core.branch._cases(self.fixture.cases['main'], 'main', {}, self.config)
        result = reader.bind_case(cases, 0, 1, self.config)
        self.assertEqual(result['identity']['target'], 'exeunt')
        self.assertEqual(result['token_ids'], [1, 2, 3, 4, 31])
        self.assertEqual(result['target_id'], 12)

    def test_bad_indices_and_boolean_coordinates_are_rejected(self):
        for index, position in ((-1, 0), (1000, 0), (True, 0), (0, True), (0, -1), (0, 3), (0, 1.0)):
            with self.subTest(index=index, position=position), self.assertRaises(ValueError):
                reader.bind_case(self.cases, index, position, self.config)

    def test_supplied_array_dtype_or_bytes_cannot_replace_pinned_batch(self):
        for dtype in ('<i8', '<f4', '<i4'):
            packed = np.asarray(self.cases['packed'], dtype=dtype).copy()
            packed[0, 0, 0] += 1
            with self.assertRaises(ValueError):
                reader.bind_case(dict(self.cases, packed=packed), 0, 0, self.config)

    def test_rehashed_bad_row_or_prefix_hash_still_fails(self):
        for name, value in (('scored_rows', [4, 5, 6]), ('scored_rows', [True, 4, 5])):
            plan = copy.deepcopy(self.cases['plan'])
            plan['cases'][0][name] = value
            write_json(self.fixture.cases['main'], plan)
            cases = dict(self.cases, plan=plan, record=reader.history.file_record(self.fixture.cases['main']))
            with self.assertRaises(ValueError):
                reader.bind_case(cases, 0, 0, self.config)

    def test_export_is_exclusive_reloadable_and_exact_int32_bytes(self):
        output = self.root/'prefix_export'
        result = reader.write_case_prefix(self.cases, 0, 2, output, self.config)
        self.assertEqual(set(result), {'record', 'plan', 'prefix'})
        self.assertEqual(reader.read_binding(result['record'], self.config), result)
        self.assertEqual((output/'prefix.i32').read_bytes(), np.asarray([1, 2, 3, 4, 11, 12], dtype='<i4').tobytes())
        with self.assertRaises(ValueError):
            reader.write_case_prefix(self.cases, 0, 2, output, self.config)
        self.assertEqual({p.name for p in output.iterdir()}, {'prefix.i32', 'binding.json'})

    def test_export_rejects_case_checkpoint_forbidden_and_symlink_ancestors(self):
        protected = self.root/'protected'
        protected.mkdir()
        symlink = self.root/'linked'
        symlink.symlink_to(protected, target_is_directory=True)
        for output in (self.fixture.cases['main'].parent/'new', self.fixture.paths['patched']/'new',
                       protected/'new', symlink/'new'):
            with self.subTest(output=output), self.assertRaises(ValueError):
                reader.write_case_prefix(self.cases, 0, 0, output, self.config,
                                         forbidden_directories=[protected])
            self.assertFalse(output.exists())

    def test_export_tampering_even_with_new_binding_hash_is_rejected(self):
        output = self.root/'prefix_export'
        result = reader.write_case_prefix(self.cases, 0, 1, output, self.config)
        plan = copy.deepcopy(result['plan'])
        plan['target_id'] += 1
        write_json(output/'binding.json', plan)
        with self.assertRaises(ValueError):
            reader.read_binding(reader.history.file_record(output/'binding.json'), self.config)

    def test_extra_export_file_and_prefix_symlink_are_rejected(self):
        output = self.root/'prefix_export'
        result = reader.write_case_prefix(self.cases, 0, 0, output, self.config)
        (output/'extra').write_text('unlisted')
        with self.assertRaises(ValueError):
            reader.read_binding(result['record'], self.config)
        (output/'extra').unlink()
        contents = (output/'prefix.i32').read_bytes()
        (output/'prefix.i32').unlink()
        (self.root/'outside.i32').write_bytes(contents)
        (output/'prefix.i32').symlink_to(self.root/'outside.i32')
        with self.assertRaises(ValueError):
            reader.read_binding(result['record'], self.config)


class NativeTraceTest(CasesFixture):
    def setUp(self):
        super().setUp()
        self.binding = reader.write_case_prefix(self.cases, 0, 1, self.root/'prefix_export', self.config)
        records = {}
        patch = reader.core.branch._patch(self.fixture.paths['patched']/'patch.json', records, self.config)
        self.model = dict(paths={k: str(v) for k, v in patch['paths'].items()}, hashes=patch['hashes'],
            weight_records=patch['weight_records'], patch=patch['record'], records=list(records.values()))
        # A uniform logical distribution deliberately makes the investigated
        # target non-greedy. This also exercises lowest-ID tie handling.
        directory = self.fixture.scores['patched']['main']
        shape = (self.cases['plan']['case_count'], self.config.context_length)
        np.full(shape, math.log(self.config.vocab_size), dtype='<f4').tofile(directory/'losses.f32.bin')
        np.zeros(shape, dtype='<i4').tofile(directory/'argmax.i32.bin')
        self.fixture.refresh_execution('patched', 'main')
        self.reference = reader.core.load_native(self.model, self.fixture.cases['main'], 'main', directory,
            self.fixture.executions['patched']['main'], self.config)
        self.native = self.root/'native_trace'
        self.native.mkdir()
        self.execution = self.root/'native_trace_execution.json'
        self.log = self.root/'native_trace.log'
        self.log.write_text('Synthetic CPU fixture: not an executed native binary.\n')
        ids = self.binding['plan']['token_ids']
        target = self.binding['plan']['target_id']
        weights = len(self.fixture.specs)
        self.metadata = dict(schema_version=1, probe_kind='token_trace', complete=True, token_ids=ids,
            prompt_rows=len(ids), selected_row=len(ids)-1, target_id=target, target_logit=0.,
            target_rank=target+1, greedy_id=0, context_length=self.config.context_length,
            vocab_size=self.config.vocab_size, padded_vocab_size=self.config.padded_vocab_size,
            no_bos=True, autoregressive=False, retokenized=False, pad_token_id=ids[-1],
            alternate_pad_token_id=1 if ids[-1] == 0 else 0,
            checkpoint_directory=self.model['paths']['patched'], tokens_file=self.binding['prefix']['path'],
            binary_file=str(self.fixture.binary), checkpoint_unique_weight_count=weights,
            checkpoint_raw_weight_count=weights+1, byte_order='little', optimizer_steps=0,
            backward_calls=0, checkpoint_writes=0, original_forward='unmodified CreateGpt2',
            logit_parity_scope='selected_row', residual_parity_scope='full_prefix',
            files={}, lens={}, interventions=[], parity_files={
                'alternate_padding': 'alternate_padding.logits.f32', 'clean_replay': 'clean_replay.logits.f32'},
            checks={name: True for name in ('final_lens_logits_byte_equal', 'alternate_padding_logits_byte_equal',
                'clean_replay_logits_byte_equal', 'attention_residual_replay_byte_equal',
                'mlp_residual_replay_byte_equal', 'checkpoint_file_stats_unchanged',
                'token_file_stats_unchanged', 'all_exported_activations_finite')})
        for name, width in reader.trace.stage_widths(self.config).items():
            np.zeros((len(ids), width), dtype='<u2').tofile(self.native/(name+'.bf16'))
            replay = name == 'embedding' or name.endswith(('.attention_projected', '.mlp_projected'))
            self.metadata['files'][name] = dict(file=name+'.bf16', dtype='bf16', shape=[len(ids), width],
                role='full_prefix', source='native_replay_saved_input' if replay else 'native_original_forward')
        np.asarray(ids, dtype='<i4').tofile(self.native/'tokens.i32')
        self.metadata['files']['tokens'] = dict(file='tokens.i32', dtype='int32', shape=[len(ids), 1],
                                               source='exact_input_token_ids', role='full_prefix')
        self.logits = np.full((1, self.config.padded_vocab_size), -np.finfo(np.float32).max, dtype='<f4')
        self.logits[:, :self.config.vocab_size] = 0
        self.logits.tofile(self.native/'logits.f32')
        self.metadata['files']['logits'] = dict(file='logits.f32', dtype='float32', shape=list(self.logits.shape),
                                               source='native_original_forward', role='selected_row')
        for name in reader.trace.residual_stage_names(self.config.n_layers):
            filename = 'lens.'+name+'.f32'
            self.logits.tofile(self.native/filename)
            self.metadata['lens'][name] = dict(file=filename, dtype='float32', shape=list(self.logits.shape),
                source='native_diagnostic_final_norm_and_tied_head', role='selected_row')
        for filename in self.metadata['parity_files'].values():
            self.logits.tofile(self.native/filename)
        self.publish()

    def publish(self, *, inputs=None, command=None, update_metadata=True):
        if update_metadata:
            write_json(self.native/'metadata.json', self.metadata)
        inputs = inputs if inputs is not None else [self.binding['record'], self.binding['prefix'],
            self.binding['plan']['cases'], self.binding['plan']['packed_batch'],
            reader.history.file_record(self.fixture.binary), *self.model['weight_records']['patched'], self.model['patch']]
        command = command or [str(self.fixture.binary), '--checkpoint='+self.model['paths']['patched'],
            '--tokens_file='+self.binding['prefix']['path'], '--target_id='+str(self.binding['plan']['target_id']),
            '--output_dir='+str(self.native), '--interventions=false']
        execution = dict(format='pluto-paired-probe-execution-v1', pid=12345, returncode=0,
            started_utc='2026-09-10T12:00:00+00:00', finished_utc='2026-09-10T12:01:00+00:00',
            command=command, log=reader.history.file_record(self.log), inputs_before=inputs,
            inputs_after=inputs, outputs=[reader.history.file_record(p) for p in sorted(self.native.iterdir())
                                         if p.is_file() and not p.is_symlink()])
        write_json(self.execution, execution)
        return execution

    def validate(self):
        return reader.validate_trace(self.native, self.binding, self.model, self.execution,
                                     self.reference, self.config)

    def test_full_saved_evidence_validates_without_a_generation_event(self):
        result = self.validate()
        self.assertEqual(set(result['stages']), set(reader.trace.stage_widths(self.config)))
        self.assertEqual(result['reference']['native_argmax'], 0)
        self.assertLess(result['reference']['absolute_nll_error'], 5e-5)
        self.assertFalse(result['reference']['full_logit_byte_parity_claimed'])
        self.assertFalse(result['generated_sampling_event'])
        self.assertFalse(result['native_execution_performed'])
        np.testing.assert_array_equal(result['logits'], self.logits)
        # Newly introduced validation sources are NOT required as past native
        # inputs. Both simulated ledgers omit this reader and still validate.
        ledger = json.loads(self.execution.read_text())
        self.assertNotIn(str(Path(reader.__file__).resolve()), [x['path'] for x in ledger['inputs_before']])

    def test_wrong_row_target_padding_or_checkpoint_metadata_rejected(self):
        for key, value in (('selected_row', 2), ('target_id', 7), ('pad_token_id', 0),
                           ('checkpoint_directory', str(self.fixture.paths['donor'])),
                           ('autoregressive', True), ('retokenized', True), ('prompt_rows', True)):
            with self.subTest(key=key):
                original = self.metadata[key]
                self.metadata[key] = value
                self.publish()
                with self.assertRaises(ValueError):
                    self.validate()
                self.metadata[key] = original

    def test_false_checks_fail_even_when_every_file_and_ledger_is_rehashed(self):
        self.metadata['checks']['clean_replay_logits_byte_equal'] = False
        self.publish()
        with self.assertRaisesRegex(ValueError, 'replay checks'):
            self.validate()

    def test_actual_replay_bytes_are_checked_not_only_flags(self):
        values = self.logits.copy()
        values[0, 0] = 1
        values.tofile(self.native/'clean_replay.logits.f32')
        self.publish()
        with self.assertRaisesRegex(ValueError, 'replay bytes'):
            self.validate()

    def test_residual_replay_is_independently_checked_in_bf16(self):
        name = 'blocks.0.mlp_projected'
        values = np.zeros((len(self.metadata['token_ids']), self.config.d_model), dtype='<u2')
        values[0, 0] = 0x3f80  # Exact BF16 1.0, inconsistent with zero residual.
        values.tofile(self.native/(name+'.bf16'))
        self.publish()
        with self.assertRaisesRegex(ValueError, 'residual replay'):
            self.validate()

    def test_final_lens_and_tokens_bytes_are_checked(self):
        path = self.native/('lens.'+reader.trace.residual_stage_names(self.config.n_layers)[-1]+'.f32')
        changed = self.logits.copy()
        changed[0, 0] = 1
        changed.tofile(path)
        self.publish()
        with self.assertRaisesRegex(ValueError, 'final lens'):
            self.validate()
        self.logits.tofile(path)
        np.zeros(len(self.metadata['token_ids']), dtype='<i4').tofile(self.native/'tokens.i32')
        self.publish()
        with self.assertRaisesRegex(ValueError, 'token dump'):
            self.validate()

    def test_missing_extra_aliased_or_escaping_native_tensor_rejected(self):
        original = copy.deepcopy(self.metadata['files']['final_norm'])
        for descriptor in (dict(original, file='logits.f32'), dict(original, file='../outside'),
                           dict(original, dtype='float32'), dict(original, shape=[1, 4])):
            self.metadata['files']['final_norm'] = descriptor
            self.publish()
            with self.assertRaises(ValueError):
                self.validate()
        self.metadata['files']['final_norm'] = original
        (self.native/'extra').write_text('unlisted')
        self.publish()
        with self.assertRaisesRegex(ValueError, 'inventory'):
            self.validate()

    def test_nonfinite_native_tensor_rejected_even_when_rehashed(self):
        values = np.zeros((len(self.metadata['token_ids']), self.config.d_model), dtype='<u2')
        values[0, 0] = 0x7f80
        values.tofile(self.native/'final_norm.bf16')
        self.publish()
        with self.assertRaises(ValueError):
            self.validate()

    def test_missing_binding_input_and_duplicate_execution_records_rejected(self):
        ledger = self.publish()
        inputs = ledger['inputs_before']
        for invalid in (inputs[1:], inputs+[inputs[0]]):
            self.publish(inputs=invalid)
            with self.assertRaises(ValueError):
                self.validate()

    def test_wrong_native_flags_and_unrecorded_interventions_rejected(self):
        command = self.publish()['command']
        for invalid in (command+['--target_id=7'], command+['--ablate_neuron=0:2'],
                        [*command[:-1], '--interventions=true'], command+['--unexpected=true']):
            self.publish(command=invalid)
            with self.assertRaises(ValueError):
                self.validate()

    def test_wrong_returncode_output_inventory_timestamps_or_log_rejected(self):
        for mutate in (lambda e: e.update(returncode=1), lambda e: e.update(outputs=e['outputs'][:-1]),
                       lambda e: e.update(started_utc='2026-09-11T12:00:00+00:00'),
                       lambda e: e['log'].update(sha256='0'*64)):
            execution = self.publish()
            mutate(execution)
            write_json(self.execution, execution)
            with self.assertRaises(ValueError):
                self.validate()

    def test_actual_reference_arrays_cannot_be_replaced_by_a_scalar_or_memory_edit(self):
        self.reference['losses'][0, self.binding['plan']['selected_row']] += 1
        with self.assertRaisesRegex(ValueError, 'reference arrays'):
            self.validate()

    def test_nll_mismatch_fails_even_if_trace_replays_are_self_consistent(self):
        # Change all selected-row logit files together, preserving replay gates.
        changed = self.logits.copy()
        changed[0, self.binding['plan']['target_id']] = -1
        for path in self.native.glob('*.f32'):
            changed.tofile(path)
        self.metadata['target_logit'] = -1
        self.metadata['target_rank'] = self.config.vocab_size
        self.publish()
        with self.assertRaisesRegex(ValueError, 'native reference'):
            self.validate()

    def test_argmax_mismatch_fails_even_with_negligible_loss_difference(self):
        changed = self.logits.copy()
        changed[0, 1] = 1e-7
        for path in self.native.glob('*.f32'):
            changed.tofile(path)
        self.metadata['greedy_id'] = 1
        self.metadata['target_rank'] += 1
        self.publish()
        with self.assertRaisesRegex(ValueError, 'native reference'):
            self.validate()

    def test_large_shared_logit_offset_does_not_erase_normalizer(self):
        changed = self.logits.copy()
        changed[:, :self.config.vocab_size] = np.float32(1e20)
        for path in self.native.glob('*.f32'):
            changed.tofile(path)
        self.metadata['target_logit'] = float(changed[0, 0])
        self.publish()
        result = self.validate()
        self.assertEqual(result['reference']['fp64_nll'], math.log(self.config.vocab_size))

    def test_boolean_token_ids_and_changed_descriptor_roles_are_rejected(self):
        original = copy.deepcopy(self.metadata)
        changes = (lambda: self.metadata['token_ids'].__setitem__(0, True),
                   lambda: self.metadata['files']['tokens'].update(role='selected_row'),
                   lambda: self.metadata['files']['logits'].update(source='native_replay_saved_input'),
                   lambda: self.metadata['lens']['positioned'].update(role='full_prefix'))
        for change in changes:
            self.metadata = copy.deepcopy(original)
            change()
            self.publish()
            with self.assertRaises(ValueError):
                self.validate()

    def test_old_reference_case_or_checkpoint_cannot_be_substituted(self):
        self.reference['cases']['record'] = reader.history.file_record(self.fixture.cases['supplemental'])
        with self.assertRaisesRegex(ValueError, 'reference case'):
            self.validate()

    def test_native_symlink_directory_and_changed_input_hash_fail(self):
        alias = self.root/'native_alias'
        alias.symlink_to(self.native, target_is_directory=True)
        with self.assertRaises(ValueError):
            reader.validate_trace(alias, self.binding, self.model, self.execution, self.reference, self.config)
        self.fixture.binary.write_bytes(b'changed executable bytes')
        with self.assertRaises(ValueError):
            self.validate()


if __name__ == '__main__':
    unittest.main()
