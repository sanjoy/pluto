"""Small explicitly synthetic archives/producers; never CUDA or model results."""

import copy
import json
from pathlib import Path
import unittest
from unittest import mock

import numpy as np

from . import head_position_cases_test as case_fixtures
from . import head_position_readout as reader


def write_json(path, value):
    Path(path).write_text(json.dumps(value, allow_nan=False))


class ReadoutTest(unittest.TestCase):
    def setUp(self):
        # Reuse a genuinely authenticated tiny archive, not a mocked descriptor
        # validator. Padding is adapted to the new CLI's -FLT_MAX contract.
        self.maker = case_fixtures.HeadPositionCasesTest()
        self.maker.setUp()
        self.addCleanup(self.maker.doCleanups)
        fixture = self.maker.fixture
        self.config = fixture.config
        for directory in fixture.root.glob('trace_step_*'):
            for path in directory.glob('*.f32'):
                values = np.fromfile(path, dtype='<f4')
                values[self.config.vocab_size:] = -np.finfo(np.float32).max
                path.write_bytes(values.tobytes())
        fixture.refresh()
        self.document = self.maker.prepare()
        self.descriptor = self.maker.output / 'cases.json'
        self.root = self.maker.output.parent / 'native_fixture'
        self.root.mkdir()
        self.binary = self.root / 'mock_binary'
        self.binary.write_bytes(b'never executed; explicitly synthetic CPU fixture')
        self.native = {}
        self.executions = {}
        self.producer = self.enterContext(mock.patch.object(reader.historical.subprocess, 'Popen',
            side_effect=AssertionError('readout cannot execute anything')))
        self.shell = self.enterContext(mock.patch.object(reader.historical.subprocess, 'run',
            side_effect=AssertionError('readout cannot execute anything')))

    def record(self, path):
        return reader.historical.record(path)

    def arrays(self, case):
        c = self.config
        context = np.full((c.context_length, c.d_model), 0x3f80, dtype='<u2')
        before = context.copy()
        for name, array in (('context', context), ('before', before)):
            array[:case['visible_rows']] = np.fromfile(case['captures'][name]['record']['path'], dtype='<u2').reshape(
                case['visible_rows'], c.d_model)
        clean = np.tile(np.fromfile(case['calibration']['clean_logits']['path'], dtype='<f4'), (c.context_length, 1))
        zero_row = np.fromfile(case['calibration']['all_query_weight_zero_logits']['path'], dtype='<f4')
        return context, before, clean, zero_row

    def arm(self, context, clean, zero_row, case, scope, dose):
        c = self.config
        rows = reader.oracle.selected_rows(c.context_length, c.context_length, 0, case['query'], scope)
        changed = context.copy()
        cols = np.arange(case['head'] * c.head_dim, (case['head'] + 1) * c.head_dim)
        changed[np.ix_(rows, cols)] = reader.oracle.scale_bf16(context[np.ix_(rows, cols)], dose)
        logits = clean.copy()
        if dose != 1 and len(rows):
            first = int(rows[0])
            if scope == 'all_queries':
                if dose == 0:
                    logits[first:] = zero_row
                else:
                    logits[first:, :c.vocab_size] += (1 - dose) * (zero_row[:c.vocab_size] - clean[0, :c.vocab_size])
            else:
                logits[first:, case['target_id']] += (1 - dose) * (0.25 if scope == 'query_only' else -0.75)
        return changed, logits

    def build_native(self, index=0):
        case = self.document['cases'][index]; c = self.config
        directory = self.root / f'case_{index}'
        directory.mkdir(); self.native[index] = directory
        tokens = np.fromfile(case['prefix']['path'], dtype='<i4')
        (directory / 'tokens.i32').write_bytes(tokens.tobytes())
        padded = np.full(c.context_length, tokens[-1], dtype='<i4'); padded[:len(tokens)] = tokens
        (directory / 'padded_tokens.i32').write_bytes(padded.tobytes())
        context, before, clean, zero_row = self.arrays(case)
        for name, array in (('original_before.bf16', before), ('original_context.bf16', context),
                            ('original_qkv.bf16', np.full((c.context_length, c.d_model * 3), 0x3f80, dtype='<u2')),
                            ('clean_logits.f32', clean)):
            (directory / name).write_bytes(array.tobytes())
        calibration = directory / 'calibration_all_queries_zero'; calibration.mkdir()
        changed, logits = self.arm(context, clean, zero_row, case, 'all_queries', 0)
        (calibration / 'context.bf16').write_bytes(changed.tobytes())
        (calibration / 'logits.f32').write_bytes(logits.tobytes())
        arms = []
        for scope in reader.oracle.SCOPES:
            for label, dose in reader.DOSES:
                name = f'{scope}_{label}'; arm = directory / name; arm.mkdir()
                changed, logits = self.arm(context, clean, zero_row, case, scope, dose)
                (arm / 'context.bf16').write_bytes(changed.tobytes())
                (arm / 'logits.f32').write_bytes(logits.tobytes())
                arms.append(dict(directory=name, scope=scope, dose_label=label, scale=dose,
                                 context_file='context.bf16', logits_file='logits.f32'))
        flags = dict(checkpoint=self.document['checkpoint_directory'], tokens_file=case['prefix']['path'], output_dir=str(directory),
            block=case['block'], head=case['head'], query=case['query'], target_id=case['target_id'],
            expected_clean_logits=case['calibration']['clean_logits']['path'],
            expected_all_query_zero_logits=case['calibration']['all_query_weight_zero_logits']['path'])
        command = [str(self.binary), *[f'--{key}={value}' for key, value in flags.items()]]
        metadata = dict(format='pluto-head-context-probe-v1', complete=True, binary=str(self.binary), **flags,
            sequence=0, query_token_id=int(tokens[case['query']]), prefix_rows=len(tokens), execution_rows=c.context_length,
            context_length=c.context_length, width=c.d_model, heads=c.n_heads, head_dimension=c.head_dim, blocks=c.n_layers,
            vocabulary=c.vocab_size, padded_vocabulary=c.padded_vocab_size, byte_order='little', padding='repeat_last_token',
            context_dtype='BF16/<u2', logits_dtype='<f4', context_shape=[c.context_length, c.d_model],
            qkv_shape=[c.context_length, c.d_model * 3], logits_shape=[c.context_length, c.padded_vocab_size],
            scopes=list(reader.oracle.SCOPES), doses=[1, .5, 0, 1], calibration_directory='calibration_all_queries_zero',
            arm_count=12, arms=arms, **{key: True for key in reader.CHECKS},
            new_paired_model_result=False, goal_completion_claimed=False, command=command)
        write_json(directory / 'metadata.json', metadata)
        log = self.root / f'case_{index}.log'; log.write_text('Synthetic producer fixture: not a measured GPU result')
        required = [self.record(self.descriptor), self.record(self.binary), *self.document['provenance'],
                    *self.document['implementation'], *self.document['weights']]
        records = list({r['path']: r for r in required}.values())
        execution = dict(format=reader.EXECUTION_FORMAT, command=command, pid=123, returncode=0,
            started_utc='2026-09-10T00:00:00+00:00', finished_utc='2026-09-10T00:00:01+00:00',
            inputs_before=records, inputs_after=copy.deepcopy(records), log=self.record(log), outputs=[])
        path = self.root / f'case_{index}_execution.json'; self.executions[index] = path
        write_json(path, execution); self.refresh_outputs(index)

    def refresh_outputs(self, index=0):
        path = self.executions[index]; execution = json.loads(path.read_text())
        execution['outputs'] = [self.record(p) for p in sorted(self.native[index].rglob('*')) if p.is_file()]
        write_json(path, execution)

    def read(self, index=0, **kwargs):
        return reader.analyze_case(self.descriptor, index, self.native[index], self.executions[index],
                                  config=self.config, allow_synthetic=True, chunk_rows=1, **kwargs)

    def mutate_array(self, name, mutate, dtype='<f4', index=0):
        path = self.native[index] / name
        array = np.fromfile(path, dtype=dtype)
        mutate(array)
        path.write_bytes(array.tobytes())
        self.refresh_outputs(index)

    def test_complete_synthetic_case_has_calibrated_fixed_rival_effects(self):
        self.build_native()
        result = self.read()
        self.assertTrue(result['complete']); self.assertTrue(result['synthetic_fixture'])
        self.assertEqual(set(result['effects_by_dose']), {'0', '0.5', '1'})
        self.assertEqual(len(result['arms']), 12)
        self.assertTrue(result['historical_only']); self.assertFalse(result['new_paired_model_result'])
        self.assertFalse(result['goal_completion_claimed']); self.assertFalse(result['full_word_probability_claimed'])
        self.assertIn(str(Path(reader.source_value_readout.__file__).resolve()),
                      [r['path'] for r in result['implementation']])
        rivals = [p['fixed_rival_id'] for p in result['effects_by_dose'].values()]
        self.assertEqual(rivals, [result['baseline']['fixed_rival_id']] * 3)
        for effect in result['effects_by_dose']['1']['effects'].values():
            self.assertTrue(all(value == 0 for value in effect.values()))
        self.producer.assert_not_called(); self.shell.assert_not_called()

    def test_toy_geometry_requires_explicit_permission(self):
        self.build_native()
        with self.assertRaisesRegex(ValueError, 'synthetic permission'):
            reader.analyze_case(self.descriptor, 0, self.native[0], self.executions[0], config=self.config)

    def test_missing_completion_gate_or_wrong_geometry_rejected(self):
        self.build_native()
        metadata = json.loads((self.native[0] / 'metadata.json').read_text())
        metadata['native_clean_attention_identity'] = False
        write_json(self.native[0] / 'metadata.json', metadata); self.refresh_outputs()
        with self.assertRaisesRegex(ValueError, 'metadata/geometry/checks'): self.read()

    def test_metadata_command_cannot_substitute_for_invocation_record(self):
        self.build_native()
        value = json.loads(self.executions[0].read_text()); value['format'] = 'untrusted metadata'
        write_json(self.executions[0], value)
        with self.assertRaisesRegex(ValueError, 'authenticated successful'): self.read()

    def test_missing_binary_hash_or_source_record_rejected(self):
        self.build_native()
        value = json.loads(self.executions[0].read_text())
        value['inputs_before'] = [r for r in value['inputs_before'] if r['path'] != str(self.binary)]
        value['inputs_after'] = copy.deepcopy(value['inputs_before'])
        write_json(self.executions[0], value)
        with self.assertRaisesRegex(ValueError, 'binary is not hash-bound'): self.read()

    def test_wrong_query_flag_or_duplicate_flag_rejected(self):
        self.build_native()
        value = json.loads(self.executions[0].read_text())
        value['command'][-1] = '--query=0'
        write_json(self.executions[0], value)
        with self.assertRaisesRegex(ValueError, 'duplicate native command'): self.read()

    def test_changed_input_before_after_and_changed_binary_rejected(self):
        self.build_native()
        self.binary.write_bytes(b'changed executable after invocation')
        with self.assertRaisesRegex(ValueError, 'recorded bytes changed'): self.read()

    def test_input_record_mutation_during_invocation_rejected(self):
        self.build_native()
        value = json.loads(self.executions[0].read_text())
        value['inputs_after'][0]['sha256'] = '0' * 64
        write_json(self.executions[0], value)
        with self.assertRaisesRegex(ValueError, 'inputs changed during invocation'): self.read()

    def test_extra_and_missing_output_files_rejected(self):
        self.build_native()
        (self.native[0] / 'extra').write_text('unrecorded'); self.refresh_outputs()
        with self.assertRaisesRegex(ValueError, 'output inventory'): self.read()

    def test_truncated_tensor_rejected_after_reenrolled_hash(self):
        self.build_native()
        path = self.native[0] / 'query_only_half/logits.f32'
        path.write_bytes(path.read_bytes()[:-4]); self.refresh_outputs()
        with self.assertRaisesRegex(ValueError, 'tensor size differs'): self.read()

    def test_prefix_padding_and_visible_archived_context_are_checked(self):
        self.build_native()
        self.mutate_array('padded_tokens.i32', lambda a: a.__setitem__(-1, int(a[-1]) + 1), '<i4')
        with self.assertRaisesRegex(ValueError, 'repeat-last padding'): self.read()

    def test_visible_original_context_cannot_be_changed(self):
        self.build_native()
        self.mutate_array('original_context.bf16', lambda a: a.__setitem__(0, int(a[0]) ^ 1), '<u2')
        with self.assertRaisesRegex(ValueError, 'visible capture differs'): self.read()

    def test_offscope_context_edit_rejected(self):
        self.build_native()
        # Query is 1, so row0 must not change under query_only.
        self.mutate_array('query_only_zero/context.bf16', lambda a: a.__setitem__(0, 0), '<u2')
        with self.assertRaisesRegex(ValueError, 'exact position/head dose'): self.read()

    def test_causally_earlier_logits_cannot_change(self):
        self.build_native()
        self.mutate_array('query_only_zero/logits.f32', lambda a: a.__setitem__(0, float(a[0]) + 1))
        with self.assertRaisesRegex(ValueError, 'causal descendants'): self.read()

    def test_padding_cannot_change_even_at_causally_affected_rows(self):
        self.build_native()
        self.mutate_array('all_queries_half/logits.f32', lambda a: a.__setitem__(-1, 0))
        with self.assertRaisesRegex(ValueError, 'padding logits changed'): self.read()

    def test_nonfinite_any_full_logit_row_is_rejected(self):
        self.build_native()
        self.mutate_array('all_queries_half/logits.f32', lambda a: a.__setitem__(0, np.nan))
        with self.assertRaisesRegex(ValueError, 'nonfinite native'): self.read()

    def test_clean_calibration_is_bit_exact(self):
        self.build_native()
        query = self.document['cases'][0]['query']
        self.mutate_array('clean_logits.f32', lambda a: a.__setitem__(query * self.config.padded_vocab_size, 123))
        with self.assertRaisesRegex(ValueError, 'clean selected logits'): self.read()

    def test_zero_calibration_and_repeated_full_zero_are_independent_gates(self):
        self.build_native()
        self.mutate_array('all_queries_zero/logits.f32', lambda a: a.__setitem__(0, float(a[0]) + 1))
        with self.assertRaisesRegex(ValueError, 'repeated calibration'): self.read()

    def test_identity_repeat_cannot_change_logits(self):
        self.build_native()
        self.mutate_array('other_queries_one_after/logits.f32', lambda a: a.__setitem__(0, float(a[0]) + 1))
        with self.assertRaisesRegex(ValueError, 'causal descendants'): self.read()

    def test_identity_audit_distinguishes_signed_zero_bytes(self):
        clean = np.zeros((self.config.context_length, self.config.padded_vocab_size), dtype='<f4')
        changed = clean.copy(); changed[0, 0] = -0.0
        self.assertTrue(np.array_equal(clean, changed))
        with self.assertRaisesRegex(ValueError, 'causal descendants'):
            reader.audit_logits(clean, changed, self.config, 1, 'all_queries', 1, 1)

    def test_source_capture_bytes_are_rechecked(self):
        self.build_native()
        path = Path(self.document['cases'][0]['captures']['before']['record']['path'])
        path.write_bytes(b'changed old archived capture')
        with self.assertRaisesRegex(ValueError, 'recorded bytes changed'): self.read()

    def test_aggregate_requires_all_cases_and_full_clean_baselines_agree(self):
        for index in range(self.document['case_count']): self.build_native(index)
        runs = [dict(case_index=i, directory=str(self.native[i]), execution=str(self.executions[i])) for i in self.native]
        with self.assertRaisesRegex(ValueError, 'every case exactly once'):
            reader.analyze_all(self.descriptor, runs[:-1], config=self.config, allow_synthetic=True)
        result = reader.analyze_all(self.descriptor, runs, config=self.config, allow_synthetic=True, chunk_rows=1)
        self.assertTrue(result['clean_full_logits_equal_across_heads'])
        self.assertEqual(len(result['cases']), 4)
        # Modify an unarchived clean row consistently in every output of head1.
        # Individual case checks pass, but repeated full baselines must disagree.
        for path in self.native[1].rglob('*logits.f32'):
            values = np.fromfile(path, dtype='<f4'); values[0] += .125
            path.write_bytes(values.tobytes())
        self.refresh_outputs(1)
        self.read(1)
        with self.assertRaisesRegex(ValueError, 'baselines differ across heads'):
            reader.analyze_all(self.descriptor, runs, config=self.config, allow_synthetic=True, chunk_rows=1)


if __name__ == '__main__':
    unittest.main()
