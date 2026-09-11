"""CPU-only fixtures for partial early-branch provenance and available effects."""

import copy
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import unittest
from unittest import mock

import numpy as np

from . import historical_source_archive_test as archive_tests
from . import paired_early_branches_partial as partial

runner, core, history, training = partial.runner, partial.core, partial.history, partial.training


def row(prefix='one', cells=('E', 'AM', 'R', 'EC', 'A', 'M'), **changes):
    value = dict(suite='main', kind='word', split='test', spelling_variant='title',
        prefix_domain='original', word_has_leading_space=True, target='Exeunt', target_ids=[1, 2, 3],
        prefix_sha256=hashlib.sha256(prefix.encode()).hexdigest(), prefix_length=128,
        cells={cell: dict(token_log_probability=[-float(i+1)]*3, argmax_ids=[1, 2, 3])
               for i, cell in enumerate(cells)})
    value.update(changes)
    return value


class AggregateTest(unittest.TestCase):
    def test_available_contrasts_and_full_factorial_are_not_imputed(self):
        item = row(cells=('E', 'EC', 'A'))
        with mock.patch.object(core, 'score_cube', side_effect=AssertionError('incomplete cube')):
            result = partial.aggregate([item], tuple(item['cells']))[0]['metrics']['suffix']
        self.assertEqual(result['available_contrasts'], {'A-E': -4.})
        self.assertNotIn('effects', result)
        self.assertEqual(result['gap_to_EC_log_probability']['A'], 2.)

    def test_full_cube_matches_explicit_factorial_math(self):
        item = row(cells=tuple(core.cube_subsets(layout=runner.LAYOUT)))
        metric = partial.aggregate([item], tuple(item['cells']))[0]['metrics']['token_0']
        values = {cell: value['token_log_probability'][0] for cell, value in item['cells'].items()}
        self.assertEqual(metric['effects'], core.score_cube(values, layout=runner.LAYOUT))
        self.assertEqual(metric['available_contrasts']['three_way'], metric['effects']['three_way'])
        self.assertEqual(metric['available_contrasts']['AM-A-M+E'],
                         values['AM']-values['A']-values['M']+values['E'])

    def test_domains_spelling_spaces_targets_and_controls_remain_separate(self):
        items = [row(str(i), **change) for i, change in enumerate([
            {}, {'prefix_domain': 'replacement'}, {'split': 'training'},
            {'spelling_variant': 'lowercase', 'target': 'exeunt'},
            {'word_has_leading_space': False}, {'target': 'Nuveth'},
            {'kind': 'control', 'prefix_domain': 'shared', 'target': 'control_next_3'},
            {'suite': 'supplemental', 'kind': 'shared_piece', 'prefix_domain': 'shared', 'piece_id': 1},
            {'suite': 'supplemental', 'kind': 'shared_piece', 'prefix_domain': 'shared', 'piece_id': 2}])]
        self.assertEqual(len(partial.aggregate(items, tuple(items[0]['cells']))), len(items))

    def test_invalid_coverage_scores_ids_and_duplicate_winners_rejected(self):
        for cells in (('E', 'A'), ('EC',), ('E', 'EC', 'H'), ('E', 'EC', 'E')):
            with self.subTest(cells=cells), self.assertRaises(ValueError): partial.aggregate([], cells)
        for invalid in (math.nan, math.inf, True, .1):
            item = row(); item['cells']['A']['token_log_probability'][0] = invalid
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                partial.aggregate([item], tuple(item['cells']))
        for field, value in (('argmax_ids', [1, True, 3]), ('token_log_probability', [-1.])):
            item = row(); item['cells']['A'][field] = value
            with self.assertRaises(ValueError): partial.aggregate([item], tuple(item['cells']))
        item, duplicate = row(), row()
        duplicate['cells']['A']['argmax_ids'][1] = 77
        with self.assertRaisesRegex(ValueError, 'conflicting'):
            partial.aggregate([item, duplicate], tuple(item['cells']))


class RequestTest(unittest.TestCase):
    """Real exact-byte source archive; synthetic metadata, no real checkpoints."""
    def setUp(self):
        self.archive_fixture = archive_tests.SourceArchiveTest(methodName='runTest')
        self.archive_fixture.setUp(); self.addCleanup(self.archive_fixture.doCleanups)
        f = self.archive_fixture
        self.root, self.previous = f.root, f.request.parent
        self.directory = self.root/'new'; self.directory.mkdir()
        self.binary = self.file('binary', {'synthetic': True})
        self.cases = {suite: self.file(suite+'.json', {'synthetic': True}) for suite in runner.SUITES}
        self.reference = self.file('reference.json', {'synthetic': True})
        references = {cell: {name: self.reference for name in ('main', 'word_next_native', 'shared_piece')}
                      for cell in ('E', 'EC')}
        self.plan = dict(format=core.FORMAT, phase='planned', config=asdict(core.GPT2Config()), step=331,
            amendment_root=str(self.root), groups=core.groups(),
            cells={key: list(value) for key, value in core.cube_subsets().items()},
            loss_tolerance=dict(absolute=core.FP32_ABSOLUTE_TOLERANCE,
                relative=core.FP32_RELATIVE_TOLERANCE, applies_to='Fixed descriptive metadata is allowed'),
            cases=self.cases, binaries={'loss_probe': self.binary}, gpu={'synthetic': True},
            runtime=dict(frozen_records=[], complete=True, runtime_dlopen_covered=False,
                         environment={'LD_LIBRARY_PATH': '/synthetic'}),
            directions={name: dict(recipient={'path': str(self.root/'recipient')},
                donor={'path': str(self.root/'donor')}, rows=[1, 2], references=references)
                for name in runner.DIRECTIONS})
        self.plan_record = self.file('plan.json', self.plan)
        old_request = dict(format=runner.OLD_FORMAT, plan=self.plan_record,
                           frozen_inputs=[f.original, f.external_record])
        f.request.write_text(json.dumps(old_request)); f.owner = history.file_record(f.request)
        f.document['owning_run_request'] = f.owner; f.publish_manifest()
        self.old_models = {cell: {'old_cell': cell} for cell in core.cube_subsets()}
        model_records = [self.file(str(self.previous.relative_to(self.root)/name/'models.json'),
                                  dict(cube=self.old_models)) for name in runner.DIRECTIONS]
        self.summary = dict(format=runner.OLD_FORMAT, complete=True, baseline_controls_certified=True,
            native_measurement_count=40, plan=self.plan_record,
            frozen_inputs=[f.owner, self.plan_record, f.original, f.external_record], artifacts=model_records)
        self.summary_record = self.file(str(self.previous.relative_to(self.root)/'summary.json'), self.summary)
        self.archive = f.make()
        originals = core.outer._unique([*self.summary['artifacts'], *self.summary['frozen_inputs'],
            self.summary_record, f.manifest_record, self.binary, self.reference, *self.cases.values()])
        self.mapping = dict(format=runner.FORMAT, source_archive=f.manifest_record,
            previous_summary=self.summary_record, original_identities_rewritten=False,
            bindings=self.archive.verify(originals))
        mapping_record = self.file('new/historical_sources.json', self.mapping)
        modules = (runner, core, history, core.native, core.branch, core.patcher, core.outer,
                   runner.screen, runner.execution, training, core.branch.case_contract)
        implementation = core.outer._unique([history.file_record(Path(module.__file__).resolve()) for module in modules])
        tests = [history.file_record(Path(runner.__file__).with_name(name)) for name in runner.TEST_NAMES]
        protocol = history.file_record(runner.PROTOCOL_PATH)
        sources = core.outer._unique([*implementation, *tests, protocol])
        copies = []
        (self.directory/'source').mkdir()
        for record in sources:
            copy_path = self.directory/'source'/Path(record['path']).name
            copy_path.write_bytes(Path(record['path']).read_bytes())
            copies.append(history.file_record(copy_path))
        frozen = core.outer._unique([*[value['physical'] for value in self.mapping['bindings']],
                                    *sources, *copies, mapping_record])
        self.request = dict(format=runner.FORMAT, layout=runner.LAYOUT, config=self.plan['config'],
            groups=core.groups(layout=runner.LAYOUT),
            cells={key: list(value) for key, value in core.cube_subsets(layout=runner.LAYOUT).items()},
            inherited_cells=runner.ANCHORS, cell_order=list(runner.CELL_ORDER), suites=list(runner.SUITES),
            native_measurement_count=32, training_restarted=False, goal_completion_claimed=False,
            previous_summary=self.summary_record, source_archive=f.manifest_record,
            implementation=implementation, test_sources=tests, protocol=protocol, frozen_inputs=frozen,
            **{key: self.plan[key] for key in ('directions', 'cases', 'binaries', 'runtime', 'gpu')},
            runner_identity=dict(pid=123, start_ticks=456, argv=['python', '-B', '-m',
                'weight_analysis.paired_early_branches', '--previous', str(self.previous),
                '--source-archive', f.manifest_record['path'], '--summary-sha256', self.summary_record['sha256'],
                '--archive-sha256', f.manifest_record['sha256'], '--output', str(self.directory)]))
        self.publish()
        self.enterContext(mock.patch.object(runner, 'validate_handoff', side_effect=AssertionError('no handoff replay')))
        self.enterContext(mock.patch.object(core.outer, 'require_exited', side_effect=AssertionError('no exit query')))
        self.enterContext(mock.patch.object(training, 'require_idle_gpu', side_effect=AssertionError('no GPU query')))

    def file(self, relative, value):
        path = self.root/relative; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value)); return history.file_record(path)

    def publish(self):
        return self.file('new/request.json', self.request)

    def publish_mapping(self):
        record = self.file('new/historical_sources.json', self.mapping)
        self.request['frozen_inputs'] = [record if r['path'] == record['path'] else r for r in self.request['frozen_inputs']]
        self.publish()

    def validate(self):
        return partial.validate_request(self.directory, runner.DIRECTIONS[0])

    def test_live_request_with_descriptive_tolerance_and_real_archive_is_valid(self):
        result = self.validate()
        self.assertEqual(result['request'], self.request)
        self.assertEqual(result['archive'].owning_request_record, self.archive_fixture.owner)
        paths = {record['path'] for record in result['records']}
        self.assertNotIn(self.archive_fixture.original['path'], paths)
        self.assertIn(str(self.archive_fixture.a), paths)

    def test_wrong_scope_invocation_and_omitted_current_source_rejected(self):
        original = copy.deepcopy(self.request)
        changes = [lambda r: r.update(native_measurement_count=31),
                   lambda r: r['runner_identity']['argv'].__setitem__(-1, '/another/run'),
                   lambda r: r['implementation'].pop(),
                   lambda r: r['frozen_inputs'].append(r['frozen_inputs'][0])]
        for change in changes:
            self.request = copy.deepcopy(original); change(self.request); self.publish()
            with self.assertRaises(ValueError): self.validate()

    def test_changed_frozen_source_copy_is_rejected(self):
        (self.directory/'source'/Path(runner.__file__).name).write_text('different source')
        with self.assertRaisesRegex(ValueError, 'frozen evidence changed'): self.validate()

    def test_incomplete_or_rewritten_historical_provenance_rejected(self):
        original = copy.deepcopy(self.mapping)
        mutations = [lambda m: m.update(original_identities_rewritten=True),
                     lambda m: m['bindings'].pop(next(i for i, binding in enumerate(m['bindings'])
                         if binding['original'] == self.summary['artifacts'][0])),
                     lambda m: m['bindings'][0]['physical'].update(sha256='0'*64),
                     lambda m: m['bindings'].append(m['bindings'][0])]
        for change in mutations:
            self.mapping = copy.deepcopy(original); change(self.mapping); self.publish_mapping()
            with self.assertRaises(ValueError): self.validate()

    def test_wrong_archive_pin_is_rejected(self):
        old = self.request['source_archive']
        self.request['source_archive'] = dict(old, sha256='0'*64)
        self.request['frozen_inputs'] = [self.request['source_archive'] if r['path'] == old['path'] else r
                                         for r in self.request['frozen_inputs']]
        self.publish()
        with self.assertRaises(ValueError): self.validate()

    def test_snapshot_refuses_missing_anchors_existing_file_and_live_output(self):
        output = self.root/'partial.json'
        with self.assertRaisesRegex(ValueError, 'completed E and EC'):
            partial.analyze(self.directory, runner.DIRECTIONS[0], output)
        self.assertFalse(output.exists())
        output.write_text('keep')
        with self.assertRaises(ValueError): partial.analyze(self.directory, runner.DIRECTIONS[0], output)
        self.assertEqual(output.read_text(), 'keep')
        with self.assertRaises(ValueError):
            partial.analyze(self.directory, runner.DIRECTIONS[0], self.directory/'new.json')

    def test_reader_source_edit_during_request_validation_is_not_rebound(self):
        path = self.root/'loaded_reader.py'; path.write_text('loaded bytes')
        record = history.file_record(path); validate = partial.validate_request
        def change(*args):
            result = validate(*args); path.write_text('new bytes'); return result
        with mock.patch.object(partial, 'READER_SOURCES', [record]), \
                mock.patch.object(partial, 'validate_request', side_effect=change):
            with self.assertRaisesRegex(ValueError, 'frozen evidence changed'):
                partial.analyze(self.directory, runner.DIRECTIONS[0], self.root/'partial.json')


class ExecutionBindingTest(unittest.TestCase):
    def setUp(self):
        self.fixture = RequestTest(methodName='runTest'); self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.directory = self.fixture.directory/'native'; self.directory.mkdir()
        self.log = self.directory/'process.log'; self.log.write_text('native output')
        self.request_record = history.file_record(self.fixture.directory/'request.json')
        self.models_record = self.fixture.file('models.json', {})
        self.value = dict(format='pluto-paired-probe-execution-v1', returncode=0,
            command=[self.fixture.binary['path']], log=history.file_record(self.log),
            inputs_before=[self.request_record, self.models_record, *self.fixture.request['frozen_inputs']])
        self.value['inputs_after'] = copy.deepcopy(self.value['inputs_before'])

    def check(self):
        path = self.directory/'execution.json'; path.write_text(json.dumps(self.value))
        return partial._execution_binding({'records': [history.file_record(path)]}, self.directory,
            self.request_record, self.models_record, self.fixture.request)

    def test_owner_and_exact_log_are_bound(self):
        self.assertEqual(self.check()[1], history.file_record(self.log))

    def test_missing_duplicate_failed_or_rebound_execution_rejected(self):
        original = copy.deepcopy(self.value)
        mutations = [lambda r: r['inputs_before'].pop(0),
                     lambda r: r['inputs_before'].append(r['inputs_before'][0]),
                     lambda r: r.update(returncode=1),
                     lambda r: r['command'].__setitem__(0, '/other/binary'),
                     lambda r: r['log'].update(path='/other/process.log')]
        for change in mutations:
            self.value = copy.deepcopy(original); change(self.value)
            self.value['inputs_after'] = copy.deepcopy(self.value['inputs_before'])
            with self.assertRaises(ValueError): self.check()


class PipelineTest(unittest.TestCase):
    """Real archives/inventories/ledgers; mock only checkpoint/native loading."""
    def setUp(self):
        self.fixture = RequestTest(methodName='runTest'); self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        f = self.fixture
        self.stage = f.directory/runner.DIRECTIONS[0]; self.stage.mkdir()
        self.output = f.root/'partial.json'
        self.models, self.loaded = {}, {}
        for cell in runner.CELL_ORDER:
            path = self.stage/'cells'/cell/'step_331'; path.mkdir(parents=True)
            (path/'patch.json').write_text('{}')
            self.models[cell] = dict(paths={'patched': str(path)}, cell=cell)
        self.models_record = f.file(str((self.stage/'models.json').relative_to(f.root)), self.models)
        self.fp64 = {'checked_predictions': 3}
        for cell in ('E', 'AM', 'R', 'EC', 'A'):
            self.loaded[cell] = {}
            for suite in runner.SUITES:
                directory = self.stage/'cells'/cell/suite; directory.mkdir()
                scores = directory/'scores'; scores.mkdir()
                (scores/'metadata.json').write_text('{}')
                losses = np.array([[1., 2., 3., 4.]], dtype='<f4')
                argmax = np.array([[1, 2, 3, 4]], dtype='<i4')
                losses.tofile(scores/'losses.f32.bin'); argmax.tofile(scores/'argmax.i32.bin')
                log = directory/'process.log'; log.write_text('synthetic native process')
                before = [history.file_record(f.directory/'request.json'), self.models_record, *f.request['frozen_inputs']]
                ledger = directory/'execution.json'
                ledger.write_text(json.dumps(dict(format='pluto-paired-probe-execution-v1', returncode=0,
                    command=[f.binary['path']], inputs_before=before, inputs_after=before, log=history.file_record(log))))
                case = dict(kind='word' if suite == 'main' else 'word_next_native', split='test',
                    context_id='one', prefix_domain='original', target='Exeunt', case_index=0,
                    target_ids=[1, 2, 3] if suite == 'main' else [1, 2, 3, 4],
                    scored_rows=[0, 1, 2] if suite == 'main' else [0, 1, 2, 3],
                    prefix={'token_ids_sha256': 'a'*64, 'length': 1},
                    target_source={'native_piece_bytes_hex': ['204578', '65', '756e74']})
                plan = dict(format='pluto-paired-word-cases-v1' if suite == 'main' else
                    'pluto-paired-supplemental-cases-v1', cases=[case],
                    source_word_cases=f.cases['main'], source_word_packed_batch={'mock': 'batch'})
                self.loaded[cell][suite] = dict(cases={'record': f.cases[suite], 'plan': plan, 'batch': {'mock': 'batch'}},
                    losses=losses, argmax=argmax,
                    records=[history.file_record(path) for path in (ledger, log, *scores.iterdir())])
            controls = {}
            if cell in runner.ANCHORS:
                for suite in runner.SUITES:
                    controls[suite] = dict(previous_cell=runner.ANCHORS[cell], exact_native_parity=
                        core.assert_native_copy_parity(self.loaded[cell][suite], self.loaded[cell][suite]))
                    if cell in ('E', 'EC'):
                        names = ('main',) if suite == 'main' else ('word_next_native', 'shared_piece')
                        controls[suite]['fp64_reference'] = {name: self.fp64 for name in names}
            cell_dir = self.stage/'cells'/cell
            runner.screen.stage_result(cell_dir, dict(cell=cell, groups=f.request['cells'][cell],
                scores={suite: str(cell_dir/suite/'scores') for suite in runner.SUITES},
                executions={suite: str(cell_dir/suite/'execution.json') for suite in runner.SUITES},
                first_three_cross_suite_byte_equal=True, anchor_controls=controls))
        self.validate_model = self.enterContext(mock.patch.object(core, 'validate_model',
            side_effect=lambda path, *args, **kwargs: self.models[Path(path).parent.parent.name]))
        self.historical_model = self.enterContext(mock.patch.object(runner, 'historical_model',
            side_effect=lambda fresh, old, archive: dict(fresh, historical=True)))
        def load(model, cases_path, suite, scores, execution, **kwargs):
            self.assertIsInstance(kwargs['case_archive'], history.SourceArchive)
            self.assertEqual('execution_archive' in kwargs, bool(model.get('historical')))
            return copy.deepcopy(self.loaded[model['cell']][suite])
        self.load = self.enterContext(mock.patch.object(core, 'load_native', side_effect=load))
        self.compare = self.enterContext(mock.patch.object(core, 'compare_fp64_reference', return_value=self.fp64))

    def analyze(self):
        return partial.analyze(self.fixture.directory, runner.DIRECTIONS[0], self.output)

    def test_partial_snapshot_keeps_missing_cells_and_certifies_all_completed_anchors(self):
        result = self.analyze()
        self.assertEqual(result['available_cells'], ['E', 'AM', 'R', 'EC', 'A'])
        self.assertEqual(result['missing_cells'], ['M', 'AR', 'MR'])
        self.assertEqual(set(result['anchor_controls']), set(runner.ANCHORS))
        self.assertFalse(result['controller_completion_claimed'])
        self.assertFalse(result['native_execution_performed'])
        self.assertEqual(self.validate_model.call_count, 5)
        self.assertEqual(self.historical_model.call_count, 4)
        self.assertEqual(self.load.call_count, 18)
        self.assertEqual(self.compare.call_count, 6)
        self.assertTrue(all(call.kwargs['layout'] == runner.LAYOUT for call in self.validate_model.call_args_list))
        training.verify_records(result['files'])

    def test_false_control_extra_inventory_and_wrong_selection_fail_closed(self):
        marker = self.stage/'cells/AM/complete.json'
        original = json.loads(marker.read_text())
        for change in (lambda m: m['anchor_controls']['main'].update(previous_cell='Q'),
                       lambda m: m.update(groups=['A'])):
            value = copy.deepcopy(original); change(value); marker.write_text(json.dumps(value))
            with self.assertRaises(ValueError): self.analyze()
            self.assertFalse(self.output.exists())
        marker.write_text(json.dumps(original))
        (self.stage/'cells/AM/extra').write_text('unlisted')
        with self.assertRaisesRegex(ValueError, 'inventory'): self.analyze()

    def test_native_parity_is_recomputed_not_trusted_from_marker(self):
        original = self.load.side_effect
        def changed(model, *args, **kwargs):
            measured = original(model, *args, **kwargs)
            if model.get('historical'): measured['losses'][0, 0] += 1
            return measured
        self.load.side_effect = changed
        with self.assertRaisesRegex(ValueError, 'copy control'): self.analyze()
        self.assertFalse(self.output.exists())

    def test_post_read_mutation_prevents_publication(self):
        aggregate = partial.aggregate
        def changed(*args):
            result = aggregate(*args)
            self.fixture.archive_fixture.external.write_text('changed evidence')
            return result
        with mock.patch.object(partial, 'aggregate', side_effect=changed):
            with self.assertRaisesRegex(ValueError, 'frozen evidence changed'): self.analyze()
        self.assertFalse(self.output.exists())


if __name__ == '__main__':
    unittest.main()
