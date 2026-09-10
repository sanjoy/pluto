"""Synthetic CPU archives test preparation, never native head effects."""

import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

import numpy as np

from . import head_position_cases as cases
from . import historical_word_ablations as historical
from .historical_word_ablations_test import Fixture, write_json


class HeadPositionCasesTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.fixture = Fixture(temporary.name)
        self.output = Path(temporary.name) / 'prepared'
        self.events = (('xy', 0, 2, '2078'), ('pqr', 2, 4, '70'))
        self.heads = ((0, 0), (0, 1))
        # Enrich the existing generation/ablation fixture with the exact extra
        # inputs needed for a context replay. Capture values are made up; only
        # shape, finite-value, hash and selection contracts are being tested.
        for run in self.fixture.plan['runs']:
            step = run['step']
            ids = run['context_token_ids']
            prefix = self.fixture.root / f'prefix_step_{step}.i32'
            prefix.write_bytes(np.asarray(ids, dtype='<i4').tobytes())
            run['prefix'] = historical.record(prefix)
            metadata = self.fixture.metadata(step)
            metadata.update(tokens_file=str(prefix), pad_token_id=ids[-1])
            native = self.fixture.root / f'trace_step_{step}'
            for name in ('positioned', 'blocks.0.attention',
                         'blocks.0.attention_projected', 'blocks.0.after_attention'):
                array = (np.arange(len(ids) * 2, dtype=np.float32) + 1).reshape(len(ids), 2)
                raw = (array.view(np.uint32) >> 16).astype('<u2').tobytes()
                filename = name + '.bf16'
                (native / filename).write_bytes(raw)
                metadata['files'][name] = dict(file=filename, dtype='bf16',
                    shape=[len(ids), 2], role='full_prefix', source='synthetic_fixture')
            write_json(native / 'metadata.json', metadata)
        self.fixture.refresh()

    def prepare(self, **kwargs):
        arguments = dict(producer_source=self.fixture.source,
                         config=self.fixture.config, events=self.events, heads=self.heads)
        arguments.update(kwargs)
        return cases.prepare(self.fixture.manifest_path, self.fixture.root,
                             self.fixture.repository, self.output, **arguments)

    def change_metadata(self, mutate, step=0):
        value = self.fixture.metadata(step)
        mutate(value)
        self.fixture.replace_metadata(value, step)

    def test_descriptor_is_prepared_not_measured_and_all_inputs_are_bound(self):
        result = self.prepare()
        self.assertEqual(json.loads((self.output / 'cases.json').read_text()), result)
        self.assertEqual(result['format'], cases.FORMAT)
        self.assertEqual(result['case_count'], 4)
        self.assertTrue(result['complete'])
        self.assertTrue(result['historical_only'])
        for key in ('production_fixed_assay', 'gpu_work_performed', 'goal_completion_claimed',
                    'calibration_executed', 'new_paired_model_result'):
            self.assertFalse(result[key])
        self.assertEqual(len(result['weights']), 16)
        for record in result['provenance'] + result['implementation']:
            self.assertEqual(historical.record(record['path']), record)
        for item in result['cases']:
            self.assertEqual(item['projection']['weight'], result['weights'][6])
            self.assertEqual(item['projection']['bias'], result['weights'][7])
            self.assertTrue(item['projection']['bias_unchanged'])
            self.assertTrue(item['calibration']['require_all_query_context_zero_matches_weight_zero_bytes'])
            self.assertFalse(item['calibration']['calibration_executed'])
        self.assertEqual({p.name for p in self.output.iterdir()}, {'cases.json'})

    def test_short_prefix_uses_actual_query_and_repeat_last_padding(self):
        result = self.prepare()
        short, full = result['cases'][0], result['cases'][2]
        self.assertEqual((short['visible_rows'], short['query']), (2, 1))
        self.assertEqual(short['padding'], dict(strategy='repeat_last_prefix_token',
                                               token_id=1, row_interval=[2, 4]))
        self.assertEqual(short['captures']['context']['shape'], [2, 2])
        self.assertEqual(short['interventions']['query_only'], [[1, 2]])
        self.assertEqual(short['interventions']['other_queries'], [[0, 1], [2, 4]])
        self.assertEqual(short['interventions']['all_queries'], [[0, 4]])
        self.assertEqual(short['interventions']['captured_query_interval'], [0, 2])
        self.assertEqual(short['interventions']['future_padding_interval'], [2, 4])
        self.assertTrue(short['interventions']['future_padding_causal_null_at_selected_query'])
        self.assertEqual((full['visible_rows'], full['query']), (4, 3))
        self.assertEqual(full['padding']['row_interval'], [4, 4])
        self.assertEqual(full['interventions']['query_only'], [[3, 4]])
        self.assertEqual(full['interventions']['other_queries'], [[0, 3]])
        self.assertEqual(full['interventions']['all_queries'], [[0, 4]])
        self.assertEqual(result['cases'][0]['head_channel_interval'], [0, 1])
        self.assertEqual(result['cases'][1]['head_channel_interval'], [1, 2])

    def test_explicit_source_never_executes_commands_from_any_manifest(self):
        self.fixture.plan['command'] = ['must-not-run']
        self.fixture.plan['runs'][0]['command'] = ['must-not-run-either']
        self.fixture.refresh()
        with mock.patch.object(historical.subprocess, 'run') as run, \
             mock.patch.object(historical.subprocess, 'Popen') as popen:
            self.prepare()
        run.assert_not_called()
        popen.assert_not_called()

    def test_git_recovery_uses_only_known_source_and_full_recorded_commit(self):
        result = subprocess.CompletedProcess([], 0, stdout=self.fixture.source_bytes, stderr=b'')
        with mock.patch.object(historical.subprocess, 'run', return_value=result) as run:
            output = self.prepare(producer_source=None)
        self.assertTrue(output['producer_source']['matched_historical_plan'])
        self.assertEqual(run.call_args.args[0], ['git', '-C', str(self.fixture.repository),
            'show', 'a' * 40 + ':' + historical.PRODUCER_SOURCE])

    def test_explicit_source_must_match_archived_bytes(self):
        self.fixture.source.write_bytes(b'changed source')
        with self.assertRaisesRegex(ValueError, 'explicit producer source'):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_refuses_existing_output_and_dangling_symlink(self):
        self.output.mkdir()
        with self.assertRaises(FileExistsError):
            self.prepare()
        self.output.rmdir()
        self.output.symlink_to(self.output.parent / 'missing')
        with self.assertRaises(FileExistsError):
            self.prepare()

    def test_refuses_output_under_preserved_inputs(self):
        self.output = self.fixture.root / 'new-output'
        with self.assertRaisesRegex(ValueError, 'overlaps'):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_changed_prefix_hash_is_rejected(self):
        path = self.fixture.root / 'prefix_step_0.i32'
        path.write_bytes(np.asarray([1, 1], dtype='<i4').tobytes())
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            self.prepare()

    def test_reenrolled_wrong_prefix_or_padding_is_rejected_semantically(self):
        path = self.fixture.root / 'prefix_step_0.i32'
        path.write_bytes(np.asarray([1, 1], dtype='<i4').tobytes())
        self.fixture.plan['runs'][0]['prefix'] = historical.record(path)
        self.fixture.refresh()
        with self.assertRaisesRegex(ValueError, 'prefix bytes'):
            self.prepare()

    def test_wrong_padding_is_rejected(self):
        self.change_metadata(lambda m: m.update(pad_token_id=0))
        with self.assertRaisesRegex(ValueError, 'padding'):
            self.prepare()

    def test_target_or_first_event_cannot_be_reselected(self):
        for change in ((('xy', 0, 3, '2078'), self.events[1]),
                       (('xy', 1, 3, '79'), self.events[1])):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.prepare(events=change)

    def test_first_piece_label_cannot_hide_a_different_whole_word(self):
        self.fixture.plan['selections'][0]['end'] = 3
        self.fixture.refresh()
        with self.assertRaisesRegex(ValueError, 'pieces do not spell'):
            self.prepare()

    def test_missing_full_prefix_capture_role_is_rejected(self):
        self.change_metadata(lambda m: m['files']['blocks.0.attention'].pop('role'))
        with self.assertRaisesRegex(ValueError, 'capture scope'):
            self.prepare()

    def test_duplicate_or_out_of_range_heads_are_rejected(self):
        for heads in (((0, 0), (0, 0)), ((1, 0),), ((0, 2),)):
            with self.subTest(heads=heads), self.assertRaises(ValueError):
                self.prepare(heads=heads)

    def test_nonfinite_and_wrong_size_weights_are_rejected(self):
        path = self.fixture.checkpoint / 'weight_0.bin'
        saved = path.read_bytes()
        for raw in (np.float32(np.nan).tobytes() + saved[4:], saved[:-4]):
            with self.subTest(size=len(raw)):
                path.write_bytes(raw)
                self.fixture.refresh()
                with self.assertRaisesRegex(ValueError, 'checkpoint weight'):
                    self.prepare()
        self.assertFalse(self.output.exists())

    def test_extra_checkpoint_file_is_rejected(self):
        (self.fixture.checkpoint / 'other').write_bytes(b'extra')
        self.fixture.refresh()
        with self.assertRaisesRegex(ValueError, 'canonical weight'):
            self.prepare()

    def test_capture_geometry_and_nonfinite_bf16_are_rejected(self):
        self.change_metadata(lambda m: m['files']['blocks.0.attention'].update(shape=[4, 2]))
        with self.assertRaisesRegex(ValueError, 'shape/dtype'):
            self.prepare()
        self.change_metadata(lambda m: m['files']['blocks.0.attention'].update(shape=[2, 2]))
        path = self.fixture.root / 'trace_step_0/blocks.0.attention.bf16'
        path.write_bytes(np.asarray([0x7f80, 0, 0, 0], dtype='<u2').tobytes())
        self.fixture.refresh()
        with self.assertRaisesRegex(ValueError, 'nonfinite native capture'):
            self.prepare()

    def test_selected_head_weight_zero_scope_requires_restore_and_zero_scale(self):
        self.change_metadata(lambda m: m['interventions'][2].update(scale=0.5))
        with self.assertRaisesRegex(ValueError, 'ablation semantics'):
            self.prepare()

    def test_reenrolled_calibration_nan_or_wrong_size_is_rejected(self):
        path = self.fixture.root / 'trace_step_0/ablation.block0.head0.f32'
        saved = path.read_bytes()
        for raw in (saved[:-4], np.float32(np.nan).tobytes() + saved[4:]):
            with self.subTest(size=len(raw)):
                path.write_bytes(raw)
                self.fixture.refresh()
                with self.assertRaisesRegex(ValueError, 'weight-zero calibration'):
                    self.prepare()

    def test_reenrolled_clean_replay_mismatch_is_rejected(self):
        path = self.fixture.root / 'trace_step_0/replay.f32'
        path.write_bytes(np.zeros(8, dtype='<f4').tobytes())
        self.fixture.refresh()
        with self.assertRaisesRegex(ValueError, 'replay/padding bytes'):
            self.prepare()

    def test_case_paths_cannot_escape_native_directory(self):
        self.change_metadata(lambda m: m['files']['blocks.0.attention'].update(file='../outside'))
        with self.assertRaisesRegex(ValueError, 'one local filename'):
            self.prepare()


if __name__ == '__main__':
    unittest.main()
