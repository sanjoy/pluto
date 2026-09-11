"""Separate early LN2 affine changes from MLP projections, with native controls.

This owns a NEW experiment after the full early-branch run has exited. It
never resumes that run, trains a model, or overwrites a checkpoint. E and M
are independently copied and rescored before the new N/F cells. Their full
native arrays must match the preceding measurements, not just rounded means.
"""

import argparse
from dataclasses import asdict
import os
from pathlib import Path
import signal
import sys

from . import paired_early_branches as previous_runner
from . import paired_early_branches_partial as previous_reader
from . import paired_mlp_affine as models
from . import paired_mlp_affine_readout as readout

core = previous_runner.core
history = previous_runner.history
training = core.training
native = core.native
outer = core.outer
screen = previous_runner.screen
FORMAT = 'pluto-paired-mlp-affine-run-v1'
DIRECTIONS = previous_runner.DIRECTIONS
SUITES = previous_runner.SUITES
CELL_ORDER = ('E', 'M', 'N', 'F')
ANCHORS = ('E', 'M')
_require = native._require
IMPORTED_SOURCES = outer._unique([history.file_record(Path(__file__).resolve()), *[
    history.file_record(Path(module.__file__).resolve())
    for name, module in tuple(sys.modules.items())
    if name.startswith('weight_analysis.')
    and getattr(module, '__file__', '').endswith('.py')]])
TEST_NAMES = ('paired_mlp_affine_test.py', 'paired_mlp_affine_readout_test.py',
              'paired_mlp_affine_run_test.py')
PROTOCOL = Path(__file__).resolve().parents[1]/'research/weight_memorization/EXEUNT_MLP_AFFINE_PROTOCOL.md'


def validate_completed_run(previous, summary_sha256):
    """Require both recorded full completion and absence of the exact owner.

    A PID disappearing alone is not success; a successful-looking state file
    alone is not process exit. The summary must bind the exact full output
    inventory, including state, owner request, both readouts and every cell.
    """
    previous = Path(previous).absolute()
    _require(previous.resolve(strict=True) == previous and previous.is_dir(),
             'invalid predecessor directory')
    summary_record = previous_runner._pinned(previous/'summary.json', summary_sha256)
    request_record = history.file_record(previous/'request.json')
    request = history._json_record(request_record)
    outer.require_exited(request['runner_identity'])
    summary = history._json_record(summary_record)
    state = native._json(previous/'state.json')
    expected = {f'{direction}/{cell}/{suite}' for direction in DIRECTIONS
                for cell in previous_runner.CELL_ORDER for suite in SUITES}
    _require(not (previous/'failure.json').exists()
             and request.get('format') == summary.get('format') == state.get('format') == previous_runner.FORMAT
             and summary.get('complete') is True and summary.get('baseline_controls_certified') is True
             and request.get('native_measurement_count') == summary.get('native_measurement_count') == 32
             and len(summary['completed']) == len(set(summary['completed'])) == 32
             and set(summary['completed']) == expected
             and state.get('phase') == 'complete' and state.get('completed') == summary['completed']
             and state.get('runner_pid') == request['runner_identity']['pid']
             and summary.get('training_restarted') is False and summary.get('goal_completion_claimed') is False
             and summary.get('source_archive') == request['source_archive']
             and summary.get('previous_summary') == request['previous_summary'],
             'predecessor completion/owner/scope differs')
    previous_runner._inventory(previous, previous/'summary.json', summary['artifacts'])
    frozen = {record['path']: record for record in previous_reader._records(summary['frozen_inputs'])}
    for record in [request_record, *request['frozen_inputs']]:
        previous_reader._member(frozen, record, 'completion omitted predecessor input')
    records = outer._unique([summary_record, request_record, *summary['artifacts'], *summary['frozen_inputs']])
    training.verify_records(records)
    return dict(request=request, request_record=request_record, summary=summary,
                summary_record=summary_record, records=records)


def validate_handoff(previous, summary_sha256):
    """Reauthenticate E/M bytes and native ledgers; never relabel old sources."""
    previous = Path(previous).absolute()
    completed = validate_completed_run(previous, summary_sha256)
    inherited = previous_reader.validate_request(previous, DIRECTIONS[0])
    request, archive = inherited['request'], inherited['archive']
    _require(request == completed['request']
             and inherited['request_record'] == completed['request_record'], 'predecessor request changed')
    records = [*completed['records'], *inherited['records']]
    references = {}
    for direction in DIRECTIONS:
        stage = previous/direction
        marker, evidence = previous_reader.previous_partial.completed_leaf(stage)
        _require(marker.get('direction') == direction and marker.get('cube_complete') is True
                 and marker.get('baseline_controls_certified') is True
                 and marker['readout'] == history.file_record(stage/'cube_readout.json')
                 and marker['baseline_controls'] == history.file_record(stage/'baseline_controls.json'),
                 'predecessor direction lacks bound completion and controls')
        report = history._json_record(marker['readout'])
        control = history._json_record(marker['baseline_controls'])
        _require(report.get('format') == core.EARLY_BRANCHES_FORMAT
                 and report.get('layout') == previous_runner.LAYOUT and report.get('complete') is True
                 and report.get('cells') == request['cells']
                 and report.get('rows') == request['directions'][direction]['rows']
                 and control.get('format') == previous_runner.FORMAT
                 and control.get('inherited_all_context_loss_and_argmax_byte_equal') is True
                 and set(control['controls']) == set(previous_runner.ANCHORS), 'predecessor readout/control scope differs')
        records.extend([*evidence, *report['files']])
        declared_record = history.file_record(stage/'models.json')
        declared = history._json_record(declared_record)
        _require(set(declared) == set(request['cells']), 'predecessor model coverage differs')
        records.append(declared_record)
        plan = request['directions'][direction]
        references[direction] = {}
        for cell in ANCHORS:
            directory = stage/'cells'/cell
            model = models.validate_model(directory/'step_331/patch.json',
                plan['recipient']['path'], plan['donor']['path'], cell, plan['rows'])
            # The older contract spells M as one group; compare selections and
            # concrete source/copy identities, not that experiment's labels.
            for key in ('paths', 'hashes', 'selection', 'patch', 'weight_records'):
                _require(model[key] == declared[cell][key], 'predecessor anchor model differs: '+key)
            leaf, leaf_records = previous_reader.previous_partial.completed_leaf(directory)
            _require(leaf.get('cell') == cell and leaf.get('groups') == request['cells'][cell]
                     and leaf.get('first_three_cross_suite_byte_equal') is True,
                     'predecessor anchor identity differs')
            records.extend([*model['records'], *leaf_records])
            measured = {}
            for suite in SUITES:
                path = directory/suite
                measured[suite] = core.load_native(model, request['cases'][suite]['path'], suite,
                    path/'scores', path/'execution.json', case_archive=archive)
                records.extend(measured[suite]['records'])
                records.extend(previous_reader._execution_binding(measured[suite], path,
                    completed['request_record'], declared_record, request))
            core._cross_suite(measured['main'], measured['supplemental'])
            references[direction][cell] = dict(model=model, measured=measured)
    records = outer._unique(records)
    training.verify_records(records)
    outer.require_exited(request['runner_identity'])
    training.require_idle_gpu(request['gpu'])
    return dict(request=request, archive=archive, summary_record=completed['summary_record'],
                references=references, records=records)


def _state(output, completed, phase, **details):
    training.publish(output/'state.json', dict(format=FORMAT, phase=phase, runner_pid=os.getpid(),
        updated_utc=training.now(), completed=list(completed), **details), exclusive=False)


def validate_current_copy(fresh, prior, config=core.GPT2Config()):
    """Compare two validated CURRENT-path models without a historical resolver.

    Their patch files retain the original recorded producers. Additional
    current reader source records are evidence of this validation, not inputs
    that supposedly existed during the old native execution. The pre-sharing
    archive must never be asked to invent bindings for these new modules.
    """
    for key in ('selection', 'hashes'):
        _require(fresh[key] == prior[key], 'copied anchor differs: '+key)
    _require(all(fresh['paths'][role] == prior['paths'][role] for role in ('recipient', 'donor'))
             and fresh['paths']['patched'] != prior['paths']['patched'], 'anchor source/copy identity differs')
    training.verify_records([*fresh['records'], *prior['records']])
    for spec in core.tensor_manifest(config):
        a = (Path(fresh['paths']['patched'])/spec.filename).stat()
        b = (Path(prior['paths']['patched'])/spec.filename).stat()
        _require((a.st_dev, a.st_ino) != (b.st_dev, b.st_ino), 'anchor shares an inode with predecessor')


def run(previous, summary_sha256, output):
    """Run sixteen serial measurements, stopping on the first failed control."""
    previous, output = Path(previous).absolute(), Path(output).absolute()
    _require(output.parent == previous.parent and output.parent.resolve(strict=True) == output.parent
             and not output.exists() and not output.is_symlink(), 'output must be a NEW sibling directory')
    for variable in ('LD_PRELOAD', 'LD_AUDIT'):
        _require(variable not in os.environ, 'unset inherited '+variable)
    tests = [history.file_record(Path(__file__).with_name(name)) for name in TEST_NAMES]
    protocol = history.file_record(PROTOCOL)
    sources = outer._unique([*IMPORTED_SOURCES, *tests, protocol])
    handoff = validate_handoff(previous, summary_sha256)
    training.verify_records(sources)
    _require(len({Path(r['path']).name for r in sources}) == len(sources), 'source snapshot basenames collide')
    plan, archive = handoff['request'], handoff['archive']
    runtime = plan['runtime']
    _require(runtime.get('complete') is True and runtime.get('runtime_dlopen_covered') is False
             and set(runtime['environment']) == {'LD_LIBRARY_PATH'}, 'unexpected frozen runtime')
    os.environ.update(runtime['environment'])
    output.mkdir()
    completed = []
    try:
        (output/'source').mkdir()
        copies = [previous_runner.execution.paired_training.freeze_file(r['path'],
            output/'source'/Path(r['path']).name) for r in sources]
        _require(all(a['sha256'] == b['sha256'] and a['bytes'] == b['bytes']
                     for a, b in zip(sources, copies)), 'source snapshot differs')
        frozen = outer._unique([*handoff['records'], *sources, *copies])
        training.verify_records(frozen)
        request = dict(format=FORMAT, created_utc=training.now(),
            runner_identity=training.process_identity(os.getpid()), config=asdict(core.GPT2Config()),
            implementation=IMPORTED_SOURCES, test_sources=tests, protocol=protocol,
            cells=models.selections(), cell_order=list(CELL_ORDER), inherited_cells=list(ANCHORS),
            directions=plan['directions'], cases=plan['cases'], suites=list(SUITES), native_measurement_count=16,
            previous_summary=handoff['summary_record'], source_archive=plan['source_archive'],
            binaries=plan['binaries'], runtime=runtime, gpu=plan['gpu'], frozen_inputs=frozen,
            training_restarted=False, goal_completion_claimed=False)
        training.publish(output/'request.json', request)
        frozen = outer._unique([*frozen, history.file_record(output/'request.json')])
        final_records = []
        for direction in DIRECTIONS:
            _state(output, completed, 'preparing_models', direction=direction)
            plan_direction = plan['directions'][direction]
            stage = output/direction
            (stage/'cells').mkdir(parents=True)
            built = {}
            for cell in CELL_ORDER:
                parent = stage/'cells'/cell
                parent.mkdir()
                built[cell] = models.build_model(plan_direction['recipient']['path'],
                    plan_direction['donor']['path'], parent/'step_331', cell, plan_direction['rows'])
            training.publish(stage/'models.json', built)
            active = outer._unique([*frozen, history.file_record(stage/'models.json'),
                *(record for model in built.values() for record in model['records'])])
            loaded, controls = {}, {}
            for cell in CELL_ORDER:
                if cell in ANCHORS:
                    validate_current_copy(built[cell], handoff['references'][direction][cell]['model'])
                loaded[cell], cell_controls = {}, {}
                for suite in SUITES:
                    _state(output, completed, 'native_measurement', direction=direction, cell=cell, suite=suite)
                    print(f'{training.now()} {direction}/{cell}/{suite}', flush=True)
                    directory = stage/'cells'/cell/suite
                    directory.mkdir()
                    cases = plan['cases'][suite]['path']
                    batch = native._json(cases)['packed_batch']['path']
                    binary = plan['binaries']['loss_probe']['path']
                    command = [binary, '--checkpoint='+built[cell]['paths']['patched'], '--batch='+batch,
                               '--output_dir='+str(directory/'scores'), '--batch_sequences=1']
                    training.verify_records(active)
                    screen.run_native(command, [r['path'] for r in active]+[cases, batch, binary],
                        directory/'scores', directory/'process.log', directory/'execution.json', plan['gpu'])
                    measured = core.load_native(built[cell], cases, suite, directory/'scores',
                        directory/'execution.json', case_archive=archive)
                    if cell in ANCHORS:
                        cell_controls[suite] = dict(exact_native_parity=core.assert_native_copy_parity(
                            handoff['references'][direction][cell]['measured'][suite], measured))
                        if cell == 'E':
                            names = ('main',) if suite == 'main' else ('word_next_native', 'shared_piece')
                            cell_controls[suite]['fp64_reference'] = {name: core.compare_fp64_reference(measured,
                                history._json_record(plan_direction['references']['E'][name]), 'JJ') for name in names}
                    loaded[cell][suite] = measured
                    active = outer._unique([*active, *measured['records'], history.file_record(directory/'process.log')])
                    training.verify_records(active)
                    completed.append(f'{direction}/{cell}/{suite}')
                core._cross_suite(loaded[cell]['main'], loaded[cell]['supplemental'])
                if cell in ANCHORS:
                    controls[cell] = cell_controls
                active = outer._unique([*active, *screen.stage_result(stage/'cells'/cell,
                    dict(cell=cell, selection=request['cells'][cell], controls=cell_controls,
                         first_three_cross_suite_byte_equal=True))])
                print(f'{training.now()} completed {direction}/{cell}; {len(completed)}/16 measurements', flush=True)
            _require(set(controls) == set(ANCHORS)
                     and all(set(value) == set(SUITES) for value in controls.values()), 'missing native copy controls')
            _state(output, completed, 'readout', direction=direction)
            per_case = previous_reader.previous_partial.selected_cases(loaded)
            report = dict(format=FORMAT, direction=direction, complete=True,
                baseline_controls_certified=True, cells=request['cells'], controls=controls,
                per_case=per_case, groups_readout=readout.aggregate(per_case), files=active,
                interpretation='E fixed in both tied roles; N is LN2 affine, F is MLP projections and biases. No computation is removed.',
                goal_completion_claimed=False)
            training.publish(stage/'readout.json', report)
            training.verify_records(active)
            _require(native._json(stage/'readout.json') == report, 'published readout differs')
            final_records.extend(screen.stage_result(stage, dict(direction=direction, complete=True,
                baseline_controls_certified=True, readout=history.file_record(stage/'readout.json'))))
        _require(len(completed) == len(set(completed)) == 16, 'incomplete measurement coverage')
        training.verify_records([*frozen, *final_records])
        _state(output, completed, 'complete')
        training.publish(output/'summary.json', dict(format=FORMAT, complete=True, completed_utc=training.now(),
            completed=completed, native_measurement_count=16, baseline_controls_certified=True,
            previous_summary=handoff['summary_record'], source_archive=plan['source_archive'],
            artifacts=[history.file_record(p) for p in sorted(output.rglob('*')) if p.is_file()],
            frozen_inputs=frozen, training_restarted=False, goal_completion_claimed=False))
    except BaseException as error:
        training.publish(output/'failure.json', dict(format=FORMAT, failed_utc=training.now(),
            error_type=type(error).__name__, error=str(error), completed=completed, no_automatic_restart=True,
            training_restarted=False, goal_completion_claimed=False))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous', required=True, type=Path)
    parser.add_argument('--summary-sha256', required=True)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'LN2/MLP runner received signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    run(args.previous, args.summary_sha256, args.output)


if __name__ == '__main__':
    main()
