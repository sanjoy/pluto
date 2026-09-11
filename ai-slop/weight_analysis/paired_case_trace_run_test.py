"""CPU-only fixed-case trace-runner tests with no native/GPU execution.

Small real files exercise hashes, source/binary copies and output inventories.
Only the predecessor audit, binding/capture adapter, runtime discovery, GPU
availability gate, and native launch are replaced with explicit fixtures.
"""

import copy
from contextlib import redirect_stdout
from dataclasses import asdict
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from . import paired_case_trace_run as runner


DIRECTIONS = ('original_to_replacement', 'replacement_to_original')
TITLE_CONTEXT = 'training:token_start:79444:title'
TITLE_HASH = '3652da5542c4dad6f987776c330d8318fff9fa3ce6c7d1f9756bdc3ab0766599'
LOWER_CONTEXT = 'training:token_start:238155:lowercase'
LOWER_HASH = 'cc7a302e808136d4af89ef597d53eaedd471909fc27785608f55cdcc0c314e67'
PIECE_CONTEXT = 'training:piece:303:start:1023605'
PIECE_HASH = '06ce1ad462295458976de0d214d79e0c317e6eed0e89b826659fd215beb9a50b'


class Files(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='paired-case-trace-run-test-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.previous, self.output = self.root/'previous', self.root/'new'
        self.previous.mkdir()
        self.source = self.file('executing_source.py', {'fixture': 'frozen source'})
        self.protocol = self.file('protocol.md', {'fixture': 'fixed trace protocol'})
        self.binary = self.file('bazel-out/fixture-opt/bin/ai-slop/weight_analysis/token_trace_probe',
                                {'fixture': 'never executed'})
        Path(self.binary['path']).chmod(0o755)
        self.owner = self.file('previous/request.json', {'runner_identity': {'pid': 123, 'start_ticks': 456}})
        self.summary = self.file('previous/summary.json', {'complete': True})
        self.archive_record = self.file('archive.json', {})
        self.cases = {suite: self.make_cases(suite) for suite in ('main', 'supplemental')}
        self.references = {}
        model_records = []
        for direction in DIRECTIONS:
            self.references[direction] = {}
            for cell in ('E', 'M'):
                checkpoint = self.previous/direction/cell/'step_331'
                patch = self.file(str((checkpoint/'patch.json').relative_to(self.root)), {'cell': cell})
                weight = self.file(str((checkpoint/'weight_0.bin').relative_to(self.root)), {'cell': cell})
                model = dict(paths={'patched': str(checkpoint)}, patch=patch,
                    weight_records={'patched': [weight]}, records=[patch, weight],
                    fixture_direction=direction, fixture_cell=cell)
                measured = {suite: dict(cases=cases, records=[cases['record'], cases['batch']],
                                       fixture_direction=direction, fixture_cell=cell)
                            for suite, cases in self.cases.items()}
                self.references[direction][cell] = dict(model=model, measured=measured)
                model_records.extend([patch, weight])
        self.request = dict(gpu={'uuid': 'fixture-no-GPU'}, source_archive=self.archive_record,
            runner_identity={'pid': 123, 'start_ticks': 456},
            directions={direction: {} for direction in DIRECTIONS},
            cases={suite: cases['record'] for suite, cases in self.cases.items()})
        self.handoff = dict(request=self.request, summary_record=self.summary, references=self.references,
            records=[self.owner, self.summary, self.archive_record, *model_records,
                     *[record for cases in self.cases.values() for record in (cases['record'], cases['batch'])]])

    def file(self, relative, value):
        path = self.root/relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return runner.history.file_record(path)

    def make_cases(self, suite):
        count = 188 if suite == 'main' else 265
        cases = [None]*count
        selected = ((0, 'Exeunt', [1475, 68, 2797], TITLE_CONTEXT, TITLE_HASH, 'title'),
                    (1, 'Nuveth', [21733, 303, 400], TITLE_CONTEXT, TITLE_HASH, 'title'),
                    (160, 'exeunt', [409, 68, 2797], LOWER_CONTEXT, LOWER_HASH, 'lowercase'))
        if suite == 'supplemental':
            selected = ((176, 'control_next_3', [303, 428, 11], PIECE_CONTEXT, PIECE_HASH, None),)
        for index, target, ids, context, prefix_hash, spelling in selected:
            item = dict(case_index=index, kind='word' if suite == 'main' else 'shared_piece',
                split='training', context_id=context, target=target, target_ids=ids,
                prefix_domain='original' if suite == 'main' else 'shared',
                scored_rows=[127, 128, 129], prefix={'length': 128, 'token_ids_sha256': prefix_hash},
                spelling_variant=spelling)
            if suite == 'supplemental':
                item['piece_id'] = 303
            cases[index] = item
        batch = self.file(suite+'/batch.i32', {'fixture': suite})
        plan = dict(case_count=count, cases=cases, packed_batch=batch, context_length=1024, vocab_size=50257)
        record = self.file(suite+'/cases.json', plan)
        return dict(plan=plan, record=record, batch=batch)

    def bind(self, cases, case_index, target_position, *args, **kwargs):
        item = cases['plan']['cases'][case_index]
        ids = [220]*128 + item['target_ids'][:target_position]
        return dict(format=runner.adapter.FORMAT, config=asdict(runner.adapter.GPT2Config()),
            token_dtype='<i4', token_ids=ids, selected_row=127+target_position,
            target_id=item['target_ids'][target_position], cases=cases['record'], packed_batch=cases['batch'],
            identity=dict(suite='supplemental' if item['kind'] == 'shared_piece' else 'main',
                case_index=case_index, target_position=target_position, kind=item['kind'], split=item['split'],
                context_id=item['context_id'], target=item['target'], prefix_domain=item['prefix_domain'],
                original_prefix_length=128, original_prefix_sha256=item['prefix']['token_ids_sha256'],
                causal_prefix_length=len(ids), causal_prefix_sha256='a'*64),
            implementation=[], caller_authenticated_case_provenance=True,
            generated_sampling_event=False, native_execution_performed=False)


def expected_coordinates():
    rows = ((DIRECTIONS[0], 'main', 0, (0, 1, 2), (1475, 68, 2797)),
            (DIRECTIONS[1], 'main', 1, (0, 1, 2), (21733, 303, 400)),
            (DIRECTIONS[0], 'main', 160, (1, 2), (409, 68, 2797)),
            (DIRECTIONS[0], 'supplemental', 176, (0,), (303, 428, 11)))
    return [(direction, suite, index, position, cell, ids[position])
            for direction, suite, index, positions, ids in rows
            for position in positions for cell in ('E', 'M')]


class FixedSpecsTest(Files):
    def setUp(self):
        super().setUp()
        self.bound = self.enterContext(mock.patch.object(runner.adapter, 'bind_case', side_effect=self.bind))

    def test_all_eighteen_coordinates_are_position_major_and_bound(self):
        specs = runner.fixed_specs(self.handoff)
        actual = [tuple(spec[key] for key in ('direction', 'suite', 'case_index', 'target_position', 'cell', 'target_id'))
                  for spec in specs]
        self.assertEqual(actual, expected_coordinates())
        self.assertEqual(len({runner._name(spec) for spec in specs}), 18)
        self.assertEqual(self.bound.call_count, 18)
        for spec in specs:
            self.assertEqual(spec['binding_plan']['selected_row'], 127+spec['target_position'])
            self.assertEqual(spec['binding_plan']['identity']['split'], 'training')
            self.assertEqual(spec['binding_plan']['identity']['original_prefix_sha256'], spec['prefix_sha256'])
            self.assertEqual(len(spec['binding_plan']['token_ids']), 128+spec['target_position'])

    def test_wrong_full_target_triple_fails_even_if_selected_piece_is_unchanged(self):
        self.cases['supplemental']['plan']['cases'][176]['target_ids'][2] = 12
        with self.assertRaisesRegex(ValueError, 'targets'):
            runner.fixed_specs(self.handoff)

    def test_wrong_context_target_split_domain_hash_or_row_is_rejected(self):
        mutations = [lambda plan: plan['identity'].update(context_id='wrong context'),
                     lambda plan: plan['identity'].update(target='Nuveth'),
                     lambda plan: plan['identity'].update(split='test'),
                     lambda plan: plan['identity'].update(prefix_domain='replacement'),
                     lambda plan: plan['identity'].update(kind='control'),
                     lambda plan: plan['identity'].update(original_prefix_sha256='0'*64),
                     lambda plan: plan['identity'].update(original_prefix_length=127),
                     lambda plan: plan.update(selected_row=128),
                     lambda plan: plan.update(target_id=999)]
        for change in mutations:
            def altered(*args, **kwargs):
                result = self.bind(*args, **kwargs); change(result); return result
            self.bound.side_effect = altered
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, 'identity'):
                runner.fixed_specs(self.handoff)

    def test_rebound_suite_record_is_rejected(self):
        self.handoff['request']['cases']['main'] = self.summary
        with self.assertRaisesRegex(ValueError, 'suite'):
            runner.fixed_specs(self.handoff)

    def test_paired_cells_must_share_complete_causal_binding(self):
        calls = 0
        def altered(*args, **kwargs):
            nonlocal calls
            calls += 1
            result = self.bind(*args, **kwargs)
            if calls == 2:
                result['token_ids'][0] = 219
            return result
        self.bound.side_effect = altered
        with self.assertRaisesRegex(ValueError, 'different causal cases'):
            runner.fixed_specs(self.handoff)


class RunnerTest(Files):
    def setUp(self):
        super().setUp()
        self.commands, self.bindings = [], {}
        self.export_count = 0
        self.before_export = None
        self.execution_mutation = None
        self.after_execute = None
        self.enterContext(mock.patch.dict(os.environ))
        os.environ.pop('LD_PRELOAD', None)
        os.environ.pop('LD_AUDIT', None)
        self.enterContext(mock.patch.object(runner, 'IMPORTED_SOURCES', [self.source]))
        self.enterContext(mock.patch.object(runner, 'DIRECT_NATIVE_SOURCES', []))
        self.enterContext(mock.patch.object(runner, 'BUILD_EVIDENCE', []))
        self.enterContext(mock.patch.object(runner, 'TEST_NAMES', ()))
        self.enterContext(mock.patch.object(runner, 'PROTOCOL', Path(self.protocol['path'])))
        self.handoff_mock = self.enterContext(mock.patch.object(runner.predecessor, 'validate_handoff', return_value=self.handoff))
        self.idle = self.enterContext(mock.patch.object(runner.training, 'require_idle_gpu'))
        self.bound = self.enterContext(mock.patch.object(runner.adapter, 'bind_case', side_effect=self.bind))
        self.export = self.enterContext(mock.patch.object(runner.adapter, 'write_case_prefix', side_effect=self.write_binding))
        self.validator = self.enterContext(mock.patch.object(runner.adapter, 'validate_trace', side_effect=self.validate_capture))
        self.runtime = self.enterContext(mock.patch.object(runner.runtime, 'freeze_runtime', side_effect=self.freeze_runtime))
        self.native = self.enterContext(mock.patch.object(runner.screen, 'run_native', side_effect=self.execute))

    def freeze_runtime(self, reference, relocated, destination):
        destination.mkdir()
        dependency = self.file(str((destination/'fixture.so').relative_to(self.root)), {'fixture': 'runtime'})
        return dict(complete=True, runtime_dlopen_covered=False,
            reference_executable=runner.history.file_record(reference),
            relocated_executable=runner.history.file_record(relocated),
            environment={'LD_LIBRARY_PATH': str(destination)}, frozen_records=[dependency])

    def write_binding(self, cases, index, position, output, **kwargs):
        self.export_count += 1
        if self.before_export:
            self.before_export(self.export_count)
        self.assertIn(self.previous, kwargs['forbidden_directories'])
        plan = self.bind(cases, index, position)
        output.mkdir()
        prefix = output/'prefix.i32'
        prefix.write_bytes(b'fixture exact prefix')
        prefix_record = runner.history.file_record(prefix)
        plan['prefix'] = prefix_record
        marker = output/'binding.json'; marker.write_text(json.dumps(plan))
        result = dict(record=runner.history.file_record(marker), plan=plan, prefix=prefix_record)
        self.bindings[str(prefix)] = result
        return result

    def execute(self, command, inputs, output, log, ledger, gpu):
        self.assertEqual(gpu, self.request['gpu'])
        self.assertEqual(os.environ['LD_LIBRARY_PATH'], str(self.output/'runtime'))
        flags = dict(argument[2:].split('=', 1) for argument in command[1:])
        binding = self.bindings[flags['tokens_file']]
        self.assertEqual(command[0], str(self.output/'bin/token_trace_probe'))
        self.assertEqual(set(flags), {'checkpoint', 'tokens_file', 'target_id', 'output_dir', 'interventions'})
        self.assertEqual(flags['interventions'], 'false')
        self.assertEqual(flags['target_id'], str(binding['plan']['target_id']))
        self.assertEqual(flags['output_dir'], str(output))
        required = [self.source['path'], self.owner['path'], self.summary['path'], self.binary['path'],
                    str(self.output/'request.json'), str(self.output/'bin/token_trace_probe'),
                    str(self.output/'runtime/fixture.so'), binding['record']['path'], binding['prefix']['path'],
                    binding['plan']['cases']['path'], binding['plan']['packed_batch']['path'],
                    str(Path(flags['checkpoint'])/'patch.json'), str(Path(flags['checkpoint'])/'weight_0.bin')]
        self.assertTrue(set(required) <= set(inputs))
        self.assertEqual(len(inputs), len(set(inputs)))
        self.commands.append(command)
        output.mkdir()
        (output/'metadata.json').write_text(json.dumps({'target_rank': 2, 'greedy_id': 7}))
        (output/'capture.bf16').write_bytes(b'fixture capture')
        log.write_text(f"Native token trace: prefix={len(binding['plan']['token_ids'])}, "
            f"target={binding['plan']['target_id']}, target rank=2, greedy=7\n"
            'Complete: '+json.dumps(str(output))+'\n')
        before = [runner.history.file_record(path) for path in inputs]
        value = dict(format='pluto-paired-probe-execution-v1', pid=123, returncode=0,
            command=command, inputs_before=before, inputs_after=copy.deepcopy(before),
            log=runner.history.file_record(log), outputs=[runner.history.file_record(path) for path in output.iterdir()])
        if self.execution_mutation:
            self.execution_mutation(value)
        ledger.write_text(json.dumps(value))
        if self.after_execute:
            self.after_execute(output, log, ledger)

    def validate_capture(self, directory, binding, model, execution, measured):
        self.assertEqual(binding['plan']['cases'], measured['cases']['record'])
        self.assertEqual(Path(execution).parent, directory.parent)
        return dict(interventions={}, generated_sampling_event=False, native_execution_performed=False,
            reference=dict(native_nll=.5, native_argmax=7, fp64_nll=.5, absolute_nll_error=0.),
            records=[runner.history.file_record(path) for path in directory.iterdir()])

    def run_capture(self, *, output=None, probe=None):
        with redirect_stdout(io.StringIO()):
            return runner.run(self.previous, self.summary['sha256'], self.output if output is None else output,
                              self.binary['path'] if probe is None else probe)

    def test_all_eighteen_captures_and_exact_summary_artifact_inventory(self):
        os.environ['LD_LIBRARY_PATH'] = '/original/runtime'
        self.run_capture()
        summary = runner.native._json(self.output/'summary.json')
        request = runner.native._json(self.output/'request.json')
        coordinates = [tuple(spec[key] for key in ('direction', 'suite', 'case_index', 'target_position', 'cell', 'target_id'))
                       for spec in request['specifications']]
        self.assertEqual(coordinates, expected_coordinates())
        self.assertEqual(len(self.commands), 18)
        self.assertEqual(summary['native_trace_count'], 18)
        self.assertEqual(summary['completed'], [runner._name(spec) for spec in request['specifications']])
        self.assertEqual(len(summary['reports']), 18)
        self.assertTrue(summary['case_reference_parity_certified'])
        for flag in ('training_restarted', 'checkpoint_mutations', 'interventions',
                     'generated_sampling_event', 'goal_completion_claimed'):
            self.assertFalse(summary[flag])
        actual = {str(path) for path in self.output.rglob('*') if path.is_file() and path != self.output/'summary.json'}
        self.assertEqual(actual, {record['path'] for record in summary['artifacts']})
        self.assertEqual(len(actual), len(summary['artifacts']))
        runner.training.verify_records(summary['artifacts'])
        runner.training.verify_records(summary['frozen_inputs'])
        self.assertEqual(summary['request'], runner.history.file_record(self.output/'request.json'))
        self.assertEqual(os.environ['LD_LIBRARY_PATH'], '/original/runtime')
        self.assertFalse((self.output/'failure.json').exists())

    def test_source_binary_and_runtime_are_real_independent_frozen_copies(self):
        self.run_capture()
        request = runner.native._json(self.output/'request.json')
        for pair in request['source_snapshots']:
            self.assertEqual(Path(pair['original']['path']).read_bytes(), Path(pair['copy']['path']).read_bytes())
            self.assertNotEqual(Path(pair['original']['path']).stat().st_ino, Path(pair['copy']['path']).stat().st_ino)
        original, copied = Path(self.binary['path']), Path(request['binary']['path'])
        self.assertEqual(original.read_bytes(), copied.read_bytes())
        self.assertNotEqual(original.stat().st_ino, copied.stat().st_ino)
        self.assertEqual(copied.stat().st_nlink, 1)
        self.assertTrue(os.access(copied, os.X_OK))
        self.assertEqual(request['runtime']['environment'], {'LD_LIBRARY_PATH': str(self.output/'runtime')})
        self.runtime.assert_called_once_with(str(original), str(copied), self.output/'runtime')

    def test_existing_nested_and_symlink_outputs_are_rejected_before_handoff(self):
        existing = self.root/'existing'; existing.mkdir(); (existing/'keep').write_text('untouched')
        alias = self.root/'alias'; alias.symlink_to(existing, target_is_directory=True)
        dangling = self.root/'dangling'; dangling.symlink_to(self.root/'absent', target_is_directory=True)
        for output in (existing, self.previous/'nested', alias, dangling):
            with self.subTest(output=str(output)), self.assertRaises(ValueError):
                self.run_capture(output=output)
        self.handoff_mock.assert_not_called()
        self.native.assert_not_called()
        self.assertEqual((existing/'keep').read_text(), 'untouched')

    def test_nonoptimized_misnamed_or_nonexecutable_probe_is_rejected(self):
        for name in ('other_probe', 'bazel-out/fixture-dbg/bin/ai-slop/weight_analysis/token_trace_probe',
                     'bazel-out/fixture-opt/bin/ai-slop/weight_analysis/other_probe'):
            probe = Path(self.file(name, {})['path']); probe.chmod(0o755)
            with self.subTest(probe=str(probe)), self.assertRaises(ValueError):
                self.run_capture(probe=probe)
        Path(self.binary['path']).chmod(0o644)
        with self.assertRaises(ValueError):
            self.run_capture()
        self.handoff_mock.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_stale_source_during_preflight_creates_no_output(self):
        def mutate(*args):
            Path(self.source['path']).write_text('changed during handoff')
            return self.handoff
        self.handoff_mock.side_effect = mutate
        with self.assertRaisesRegex(ValueError, 'frozen capture input changed'):
            self.run_capture()
        self.assertFalse(self.output.exists())
        self.native.assert_not_called()

    def test_source_change_before_second_capture_stops_after_one(self):
        def mutate(count):
            if count == 2:
                Path(self.source['path']).write_text('changed before second trace')
        self.before_export = mutate
        with self.assertRaisesRegex(ValueError, 'frozen capture input changed'):
            self.run_capture()
        self.assertEqual(len(self.commands), 1)
        self.assertFalse((self.output/'summary.json').exists())
        self.assertEqual(len(runner.native._json(self.output/'failure.json')['completed']), 1)

    def test_current_owner_change_before_second_capture_is_rejected(self):
        def mutate(count):
            if count == 2:
                (self.output/'request.json').write_text('changed current owner')
        self.before_export = mutate
        with self.assertRaisesRegex(ValueError, 'frozen capture input changed'):
            self.run_capture()
        self.assertEqual(len(self.commands), 1)
        self.assertFalse((self.output/'summary.json').exists())

    def test_busy_gpu_preflight_never_launches_or_creates_output(self):
        self.idle.side_effect = RuntimeError('GPU is busy')
        with self.assertRaisesRegex(RuntimeError, 'GPU is busy'):
            self.run_capture()
        self.native.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_failed_native_stops_without_retry_or_summary_and_restores_environment(self):
        os.environ['LD_LIBRARY_PATH'] = '/before'
        self.native.side_effect = RuntimeError('native failure')
        with self.assertRaisesRegex(RuntimeError, 'native failure'):
            self.run_capture()
        self.assertEqual(self.native.call_count, 1)
        self.validator.assert_not_called()
        self.assertFalse((self.output/'summary.json').exists())
        self.assertTrue(runner.native._json(self.output/'failure.json')['no_automatic_restart'])
        self.assertEqual(os.environ['LD_LIBRARY_PATH'], '/before')

    def test_validator_failure_stops_after_first_native_capture(self):
        self.validator.side_effect = ValueError('trace/reference mismatch')
        with self.assertRaisesRegex(ValueError, 'trace/reference mismatch'):
            self.run_capture()
        self.assertEqual(self.native.call_count, 1)
        self.assertFalse((self.output/'summary.json').exists())
        self.assertTrue((self.output/'failure.json').exists())

    def test_missing_owner_native_input_is_rejected(self):
        def mutate(value):
            value['inputs_before'] = [record for record in value['inputs_before']
                                      if record['path'] != str(self.output/'request.json')]
        self.execution_mutation = mutate
        with self.assertRaisesRegex(ValueError, 'owning input membership'):
            self.run_capture()
        self.assertEqual(self.native.call_count, 1)
        self.assertFalse((self.output/'summary.json').exists())

    def test_duplicate_native_input_is_rejected_even_when_mapping_matches(self):
        def mutate(value):
            value['inputs_after'].append(copy.deepcopy(value['inputs_after'][0]))
        self.execution_mutation = mutate
        with self.assertRaisesRegex(ValueError, 'owning input membership'):
            self.run_capture()
        self.assertEqual(self.native.call_count, 1)
        self.assertFalse((self.output/'summary.json').exists())

    def test_untracked_trace_file_is_rejected(self):
        def mutate(directory, log, ledger):
            (directory.parent/'untracked.txt').write_text('not native or owned execution evidence')
        self.after_execute = mutate
        with self.assertRaisesRegex(ValueError, 'unexpected file or directory in trace capture'):
            self.run_capture()
        self.assertEqual(self.native.call_count, 1)
        self.assertFalse((self.output/'summary.json').exists())

    def test_untracked_root_file_is_rejected_before_publishing_summary(self):
        def mutate(directory, log, ledger):
            (self.output/'untracked.txt').write_text('not a frozen or validated capture artifact')
        self.after_execute = mutate
        with self.assertRaisesRegex(ValueError, 'unexpected file in final capture inventory'):
            self.run_capture()
        self.assertEqual(self.native.call_count, 18)
        self.assertFalse((self.output/'summary.json').exists())

    def test_wrong_completion_stdout_is_not_accepted_as_execution_evidence(self):
        def mutate(output, log, ledger):
            log.write_text('Native token trace for a different case\nComplete: "wrong"\n')
            value = runner.native._json(ledger); value['log'] = runner.history.file_record(log)
            ledger.write_text(json.dumps(value))
        self.after_execute = mutate
        with self.assertRaisesRegex(ValueError, 'stdout'):
            self.run_capture()
        self.assertEqual(self.native.call_count, 1)
        self.assertFalse((self.output/'summary.json').exists())


if __name__ == '__main__':
    unittest.main()
