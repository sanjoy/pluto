"""Small synthetic tests of selection, windowing, byte parity and safe reuse."""

import json
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from . import multiword_trace as trace


def fixture(padding=0):
    pieces = {0: b'hi', 1: b' grand', 2: b'am', 3: b'.', 4: b' '}
    generated = [4]*padding+[1, 2, 3]
    meta = dict(complete=True, autoregressive=True, prompt='hi',
                initial_token_ids=[0], generated_token_ids=generated,
                all_token_ids=[0]+generated, steps=len(generated),
                context_length=4, vocab_size=len(pieces))
    events, cursor = [], 0
    for index, token in enumerate(generated):
        absolute = index+1
        start = max(0, absolute-4)
        events.append(dict(index=index, absolute_token_index=absolute,
            token_id=token, generated_byte_start=cursor,
            generated_byte_end=cursor+len(pieces[token]), piece_hex=pieces[token].hex(),
            context_start=start, context_length=absolute-start,
            output_row=absolute-start-1, logits_byte_offset=index*len(pieces)*4))
        cursor += len(pieces[token])
    return meta, events, pieces, b''.join(pieces[t] for t in generated)


class SelectionTest(unittest.TestCase):
    def select(self, padding=0):
        return trace.select_steps(*fixture(padding),
                                  [trace.parse_word(f'grandam:{padding}:{padding+2}')])

    def test_half_open_selection_and_boundary_baseline(self):
        words, runs = self.select()
        self.assertEqual(words[0]['word_evidence']['word'], 'grandam')
        self.assertEqual(words[0]['boundary_step'], 2)
        self.assertEqual([run['step'] for run in runs], [0, 1, 2])
        self.assertEqual([run['interventions'] for run in runs], [True, True, False])
        self.assertEqual([run['context_token_ids'] for run in runs], [[0], [0, 1], [0, 1, 2]])

    def test_sliding_window_uses_context_relative_positions(self):
        _, runs = self.select(padding=7)
        self.assertEqual([run['event']['context_start'] for run in runs], [4, 5, 6])
        self.assertEqual([run['context_token_ids'] for run in runs],
                         [[4, 4, 4, 4], [4, 4, 4, 1], [4, 4, 1, 2]])
        self.assertEqual([run['event']['output_row'] for run in runs], [3, 3, 3])

    def test_invalid_word_specs_rejected(self):
        for value in ('grandam', '../grandam:0:2', 'a:0:1', 'a:3:2', 'a:-1:2',
                      'two words:0:2', 'é:0:2', 'a:0:2:3'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                trace.parse_word(value)

    def test_wrong_word_or_span_rejected(self):
        for value in ('grandem:0:2', 'grandam:1:3', 'Grandam:0:2', 'grandam:0:3'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                trace.select_steps(*fixture(), [trace.parse_word(value)])

    def test_whole_word_vocabulary_variant_rejected(self):
        meta, events, pieces, raw = fixture()
        pieces[4] = b' GRANDAM'
        with self.assertRaises(ValueError):
            trace.select_steps(meta, events, pieces, raw, [trace.parse_word('grandam:0:2')])

    def test_duplicate_selection_rejected(self):
        with self.assertRaises(ValueError):
            trace.select_steps(*fixture(), [trace.parse_word('grandam:0:2')]*2)

    def test_invalid_native_coordinates_rejected(self):
        for field in ('context_start', 'context_length', 'output_row', 'logits_byte_offset'):
            values = list(fixture())
            values[1][0][field] += 1
            with self.subTest(field=field), self.assertRaises(ValueError):
                trace.select_steps(*values, [trace.parse_word('grandam:0:2')])

    def test_incomplete_or_changed_generation_rejected(self):
        values = list(fixture())
        values[0]['complete'] = False
        with self.assertRaises(ValueError):
            trace.select_steps(*values, [trace.parse_word('grandam:0:2')])
        values = list(fixture())
        values[3] += b'x'
        with self.assertRaises(ValueError):
            trace.select_steps(*values, [trace.parse_word('grandam:0:2')])

    def test_contraction_fragment_rejected(self):
        meta, events, pieces, _ = fixture()
        pieces[3] = b"'"
        events[2]['piece_hex'] = pieces[3].hex()
        with self.assertRaises(ValueError):
            trace.select_steps(meta, events, pieces, b" grandam'",
                               [trace.parse_word('grandam:0:2')])


class EvidenceTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_exclusive_json_and_changed_input(self):
        path = self.root/'input.json'
        trace.write_json(path, dict(value=1))
        identity = trace.record(path)
        trace.verify_records([identity])
        with self.assertRaises(FileExistsError):
            trace.write_json(path, dict(value=2))
        path.write_text('changed')
        with self.assertRaises(ValueError):
            trace.verify_records([identity])

    def test_no_symlink_or_parent_escape(self):
        path = self.root/'source'
        path.write_bytes(b'ok')
        (self.root/'link').symlink_to(path)
        for filename in ('link', '../source', '/tmp/source', '.'):
            with self.subTest(filename=filename), self.assertRaises(ValueError):
                trace.local_file(self.root, filename)
        with self.assertRaises(ValueError):
            trace.record(self.root/'link')
        with self.assertRaises(ValueError):
            trace.artifact_records(self.root)

    def native_fixture(self):
        native = self.root/'trace'
        native.mkdir()
        logits = struct.pack('<fff', 1., 2., 0.)
        (native/'logits.f32').write_bytes(logits)
        original = self.root/'generation.f32'
        original.write_bytes(logits[:8])
        checks = dict.fromkeys(('final_lens_logits_byte_equal',
            'alternate_padding_logits_byte_equal', 'clean_replay_logits_byte_equal',
            'attention_residual_replay_byte_equal', 'mlp_residual_replay_byte_equal'), True)
        metadata = dict(complete=True, probe_kind='token_trace', token_ids=[0],
            target_id=1, checks=checks, interventions=[],
            files={'logits':dict(file='logits.f32', dtype='float32', shape=[1, 3])})
        trace.write_json(native/'metadata.json', metadata)
        plan = dict(vocab_size=2, padded_vocab_size=3, n_layers=8, n_heads=8,
                    generation_logits=str(original))
        run = dict(context_token_ids=[0], interventions=False,
                   event=dict(token_id=1, logits_byte_offset=0))
        return plan, run, native, metadata

    def test_native_byte_parity_ignores_only_physical_padding(self):
        plan, run, native, metadata = self.native_fixture()
        self.assertEqual(trace.check_native(plan, run, native), metadata)
        (native/'logits.f32').write_bytes(struct.pack('<fff', 1., 2., 999.))
        trace.check_native(plan, run, native)
        (native/'logits.f32').write_bytes(struct.pack('<fff', 1., 3., 999.))
        with self.assertRaises(ValueError):
            trace.check_native(plan, run, native)

    def test_native_missing_interventions_rejected(self):
        plan, run, native, _ = self.native_fixture()
        run['interventions'] = True
        with self.assertRaises(ValueError):
            trace.check_native(plan, run, native)

    def test_partial_run_never_relaunched(self):
        plan = dict(output_directory=str(self.root), inputs=[], runs=[dict(step=7)])
        trace.write_json(self.root/'plan.json', plan)
        (self.root/'trace_step_7').mkdir()
        with mock.patch('subprocess.Popen') as launch:
            with self.assertRaises(FileExistsError):
                trace.execute(plan)
            launch.assert_not_called()

    def test_failed_run_never_relaunched(self):
        plan = dict(output_directory=str(self.root), inputs=[], runs=[dict(step=7)])
        trace.write_json(self.root/'plan.json', plan)
        trace.write_json(self.root/'run_step_7.result.json', dict(complete=False))
        with mock.patch('subprocess.Popen') as launch:
            with self.assertRaises(ValueError):
                trace.execute(plan)
            launch.assert_not_called()

    def test_resume_refuses_new_selections(self):
        with mock.patch('sys.stderr'), self.assertRaises(SystemExit):
            trace.main(['--output-directory', str(self.root), '--resume', '--word', 'x:0:2'])

    def execution_fixture(self):
        plan, run, original_native, _ = self.native_fixture()
        run.update(step=0, prefix=dict(path=str(self.root/'prefix.i32')))
        plan.update(output_directory=str(self.root), inputs=[], runs=[run],
                    binary='/fake/probe', checkpoint_directory='/fake/checkpoint',
                    tokenizer_directory='/fake/tokenizer', generation_directory='/fake/generation')
        trace.write_json(self.root/'plan.json', plan)

        def launch(command, **kwargs):
            original_native.rename(self.root/'trace_step_0')
            return SimpleNamespace(pid=12345, wait=lambda: 0)

        def analyze(native, generation, step, checkpoint, tokenizer, output):
            output.mkdir()
            trace.write_json(output/'analysis.json', dict(complete=True))

        return plan, launch, analyze

    def test_completed_step_and_final_result_reuse_without_launch(self):
        plan, launch, analyze = self.execution_fixture()
        with mock.patch('subprocess.Popen', side_effect=launch) as process:
            with mock.patch('scripts.weight_analysis.token_trace_analysis.run', side_effect=analyze):
                final = trace.execute(plan)
        self.assertTrue(final['complete'])
        self.assertTrue(final['all_inputs_unchanged'])
        self.assertEqual(process.call_count, 1)
        process_record = json.loads((self.root/'run_step_0.process.json').read_text())
        self.assertEqual(process_record['pid'], 12345)
        with mock.patch('subprocess.Popen') as process:
            self.assertEqual(trace.execute(plan), final)
            process.assert_not_called()

    def test_changed_completed_output_refuses_resume(self):
        plan, launch, analyze = self.execution_fixture()
        with mock.patch('subprocess.Popen', side_effect=launch):
            with mock.patch('scripts.weight_analysis.token_trace_analysis.run', side_effect=analyze):
                trace.execute(plan)
        (self.root/'analysis_step_0/analysis.json').write_text('changed')
        with mock.patch('subprocess.Popen') as process:
            with self.assertRaises(ValueError):
                trace.execute(plan)
            process.assert_not_called()

    def test_new_completed_output_refuses_resume(self):
        plan, launch, analyze = self.execution_fixture()
        with mock.patch('subprocess.Popen', side_effect=launch):
            with mock.patch('scripts.weight_analysis.token_trace_analysis.run', side_effect=analyze):
                trace.execute(plan)
        (self.root/'analysis_step_0/new.json').write_text('{}')
        with mock.patch('subprocess.Popen') as process:
            with self.assertRaises(ValueError):
                trace.execute(plan)
            process.assert_not_called()

    def test_failed_analysis_saves_terminal_result_and_refuses_resume(self):
        plan, launch, _ = self.execution_fixture()
        with mock.patch('subprocess.Popen', side_effect=launch):
            with mock.patch('scripts.weight_analysis.token_trace_analysis.run',
                            side_effect=ValueError('injected analysis failure')):
                with self.assertRaises(ValueError):
                    trace.execute(plan)
        result = json.loads((self.root/'run_step_0.result.json').read_text())
        self.assertFalse(result['complete'])
        self.assertEqual(result['returncode'], 0)
        self.assertIn('injected analysis failure', result['error'])
        self.assertIn('ended_utc', result)
        with mock.patch('subprocess.Popen') as process:
            with self.assertRaises(ValueError):
                trace.execute(plan)
            process.assert_not_called()


if __name__ == '__main__':
    unittest.main()
