"""Separate early attention/LN1 and MLP/LN2 in two residual backgrounds.

This is a NEW experiment, not a continuation or rewrite of the Q/H/L cube.
All eight checkpoints fix the eleven donor embedding rows in both tied roles.
A transfers the early attention branches, M their MLP branches, and R the
remaining nonembedding weights. The four inherited cells are independently
copied and rescored before the four new cells. Every inherited loss/argmax
byte must match its authenticated historical counterpart, including padding.

Historical source identities are verified against a caller-pinned archive;
old commands and ledgers are never relabeled as executions of relocated code.
Only current physical paths enter this experiment's fresh native ledgers.
No training, overwrite, automatic retry, or mechanistic-goal completion occurs.
"""

import argparse
from dataclasses import asdict
import os
from pathlib import Path
import signal
import sys

from . import paired_complement_localization as core
from . import historical_source_archive as history

training = core.training
native = core.native
outer = core.outer
screen = outer.screen
execution = screen.execution
FORMAT = 'pluto-paired-early-branches-run-v1'
OLD_FORMAT = 'pluto-paired-complement-localization-run-v1'
LAYOUT = 'early_branches'
DIRECTIONS = ('original_to_replacement', 'replacement_to_original')
SUITES = ('main', 'supplemental')
ANCHORS = {'E': 'E', 'AM': 'H', 'R': 'QL', 'EC': 'EC'}
CELL_ORDER = ('E', 'AM', 'R', 'EC', 'A', 'M', 'AR', 'MR')
_require = native._require
# Capture imported code before any potentially long hash audit.
IMPORTED_SOURCES = outer._unique([history.file_record(Path(__file__).resolve()), *[
    history.file_record(Path(module.__file__).resolve())
    for name, module in tuple(sys.modules.items())
    if name.startswith('weight_analysis.') and
    getattr(module, '__file__', '').endswith('.py')]])
TEST_NAMES = ('paired_early_branches_test.py', 'paired_early_branches_localization_test.py',
              'historical_source_archive_test.py', 'paired_branch_archive_test.py')
PROTOCOL_PATH = Path(__file__).resolve().parents[1]/'research/weight_memorization/EXEUNT_EARLY_BRANCH_PROTOCOL.md'


def _pinned(path, sha256):
    record = history.file_record(Path(path).absolute())
    _require(record['sha256'] == sha256, 'caller-pinned artifact hash differs')
    return record


def _inventory(directory, marker, declared):
    """Exact output inventory; a summary flag does not replace actual files."""
    paths = list(directory.rglob('*'))
    _require(not any(path.is_symlink() for path in paths), 'symlink in old output')
    unique = outer._unique(declared)
    _require(len(unique) == len(declared) and
             {str(path) for path in paths if path.is_file() and path != marker} ==
             {record['path'] for record in declared}, 'old output inventory differs')


def validate_handoff(previous, archive_path, summary_sha256, archive_sha256):
    """Authenticate the completed cube without executing archived Python."""
    previous = Path(previous).absolute()
    summary_record = _pinned(previous/'summary.json', summary_sha256)
    request_record = history.file_record(previous/'request.json')
    archive_record = _pinned(archive_path, archive_sha256)
    archive = history.SourceArchive(archive_path, expected_manifest_record=archive_record,
                                   owning_request_record=request_record)
    request = native._json(request_record['path'])
    summary = native._json(summary_record['path'])
    state = native._json(previous/'state.json')
    _require(not (previous/'failure.json').exists(), 'old experiment recorded failure')
    identity = request['runner_identity']
    argv = identity['argv']
    _require(type(identity['pid']) is int and identity['pid'] > 0 and
             type(identity['start_ticks']) is int and identity['start_ticks'] > 0 and
             argv.count('-m') == argv.count('--plan') == argv.count('--output') == 1 and
             argv[argv.index('-m')+1] == 'scripts.weight_analysis.paired_complement_localization_run' and
             argv[argv.index('--output')+1] == str(previous) and
             argv[argv.index('--plan')+1] == request['plan']['path'],
             'historical runner identity differs')
    outer.require_exited(identity)
    expected = {direction+'/'+cell+'/'+suite for direction in DIRECTIONS
                for cell in ('reference_E', 'reference_EC', 'E', 'EC', 'Q', 'H', 'L', 'QH', 'QL', 'HL')
                for suite in SUITES}
    _require(request.get('format') == summary.get('format') == state.get('format') == OLD_FORMAT and
             summary.get('complete') is True and summary.get('baseline_controls_certified') is True and
             summary.get('native_measurement_count') == request.get('native_measurement_count') == 40 and
             len(summary['completed']) == len(set(summary['completed'])) == 40 and
             set(summary['completed']) == expected and state.get('completed') == summary['completed'] and
             state.get('phase') == 'complete' and state.get('runner_pid') == identity['pid'] and
             summary.get('training_restarted') is False and summary.get('goal_completion_claimed') is False,
             'historical cube completion is incomplete or inconsistent')
    _require(request.get('directions') == list(DIRECTIONS) and request.get('suites') == list(SUITES) and
             request.get('cell_order') == ['E', 'EC', 'Q', 'H', 'L', 'QH', 'QL', 'HL'] and
             request.get('training_restarted') is False and request.get('goal_completion_claimed') is False and
             summary['plan'] == request['plan'], 'historical request scope differs')
    plan = native._json(request['plan']['path'])
    _require(plan.get('format') == core.FORMAT and plan.get('phase') == 'planned' and
             plan.get('config') == asdict(core.GPT2Config()) and plan.get('step') == 331 and
             plan.get('amendment_root') == str(previous.parent) and
             plan.get('groups') == core.groups() and
             plan.get('cells') == {name: list(value) for name, value in core.cube_subsets().items()} and
             plan.get('native_execution_performed') is False and
             set(plan['directions']) == set(DIRECTIONS) and set(plan['cases']) == set(SUITES),
             'historical plan differs from fixed endpoint cube')
    for key in ('binaries', 'runtime', 'gpu'):
        _require(request[key] == plan[key], 'historical request/plan binding differs: '+key)
    _require(plan['loss_tolerance']['absolute'] == core.FP32_ABSOLUTE_TOLERANCE and
             plan['loss_tolerance']['relative'] == core.FP32_RELATIVE_TOLERANCE,
             'historical rounding tolerance changed')
    frozen = {record['path']: record for record in summary['frozen_inputs']}
    _require(all(frozen.get(record['path']) == record for record in
                 [*request['frozen_inputs'], *plan['frozen_inputs'], request_record, request['plan']]),
             'historical completion omitted frozen inputs')
    _inventory(previous, previous/'summary.json', summary['artifacts'])
    records = [*summary['artifacts'], *summary['frozen_inputs'], summary_record, archive_record]
    old_models = {}
    for name in DIRECTIONS:
        directory = previous/name
        marker = native._json(directory/'complete.json')
        _require(marker.get('direction') == name and marker.get('cube_complete') is True and
                 marker.get('baseline_controls_certified') is True and
                 marker['readout'] == history.file_record(directory/'cube_readout.json') and
                 marker['baseline_controls'] == history.file_record(directory/'baseline_controls.json'),
                 'historical direction lacks bound readout and controls')
        _inventory(directory, directory/'complete.json', marker['artifacts'])
        report = native._json(marker['readout']['path'])
        _require(report.get('format') == core.FORMAT and report.get('complete') is True and
                 report.get('groups') == core.groups() and report.get('rows') == plan['directions'][name]['rows'],
                 'historical numerical readout differs')
        models = native._json(directory/'models.json')['cube']
        _require(set(models) == set(core.cube_subsets()), 'historical model cube is incomplete')
        old_models[name] = models
        records.extend(report['files'])
    bindings = archive.verify(outer._unique(records))
    physical = outer._unique([binding['physical'] for binding in bindings] +
                            [archive_record, request_record, summary_record])
    # Case provenance still names historical producers. The explicit resolver
    # verifies those exact bytes while leaving the case JSON and batch intact.
    case_records = {}
    for suite, expected_count in (('main', 188), ('supplemental', 265)):
        cases = core.branch._cases(plan['cases'][suite]['path'], suite, case_records,
                                   core.GPT2Config(), source_archive=archive)
        _require(cases['record'] == plan['cases'][suite] and
                 cases['plan']['case_count'] == expected_count, 'historical case coverage changed')
    physical = outer._unique([*physical, *case_records.values()])
    outer.require_exited(identity)
    training.require_idle_gpu(plan['gpu'])
    return dict(previous=previous, archive=archive, request=request, summary=summary, plan=plan,
                old_models=old_models, bindings=bindings, physical=physical,
                summary_record=summary_record, archive_record=archive_record)


def historical_model(fresh, old, archive, *, config=core.GPT2Config()):
    """An explicit read-only adapter, not a rewritten historical patch.

    Fresh construction independently checks selected and unselected bytes.
    Matching every old source/output hash plus rereading the old allocations
    proves that the two coordinate-defined interventions are identical.
    """
    _require(fresh['selection'] == old['selection'] and fresh['hashes'] == old['hashes'] and
             all(fresh['paths'][role] == old['paths'][role] for role in ('recipient', 'donor')),
             'inherited cell is not the same intervention')
    _require(fresh['paths']['patched'] != old['paths']['patched'], 'anchor must be an independent copy')
    bindings = archive.verify(old['records'])
    records = {binding['physical']['path']: binding['physical'] for binding in bindings}
    weights = {role: core.branch._weights(path, old['hashes'][role], records, config)
               for role, path in old['paths'].items()}
    for spec in core.tensor_manifest(config):
        a = (Path(fresh['paths']['patched'])/spec.filename).stat()
        b = (Path(old['paths']['patched'])/spec.filename).stat()
        _require((a.st_dev, a.st_ino) != (b.st_dev, b.st_ino), 'anchor copy shares an inode')
    return dict(paths=old['paths'], hashes=old['hashes'], weight_records=weights,
                records=outer._unique(list(records.values())), patch=old['patch'],
                historical_only=True, original_model_contract=old)


def _state(output, completed, phase, **details):
    training.publish(output/'state.json', dict(format=FORMAT, phase=phase, runner_pid=os.getpid(),
        updated_utc=training.now(), completed=list(completed), **details), exclusive=False)


def run(previous, archive_path, summary_sha256, archive_sha256, output):
    """Own one serial 32-measurement run; preserve partial output on failure."""
    output, previous = Path(output).absolute(), Path(previous).absolute()
    _require(output.parent == previous.parent and not output.exists() and not output.is_symlink(),
             'new experiment must be a NEW sibling of the previous run')
    for variable in ('LD_PRELOAD', 'LD_AUDIT'):
        _require(variable not in os.environ, 'unset inherited '+variable)
    test_sources = [history.file_record(Path(__file__).with_name(name).resolve()) for name in TEST_NAMES]
    protocol = history.file_record(PROTOCOL_PATH)
    handoff = validate_handoff(previous, archive_path, summary_sha256, archive_sha256)
    sources = outer._unique([*IMPORTED_SOURCES, *test_sources, protocol])
    training.verify_records(sources)
    _require(len({Path(record['path']).name for record in sources}) == len(sources),
             'source snapshot basenames collide')
    output.mkdir()
    completed, final_records = [], []
    try:
        (output/'source').mkdir()
        copies = [execution.paired_training.freeze_file(record['path'], output/'source'/Path(record['path']).name)
                  for record in sources]
        _require(all(source['bytes'] == copy['bytes'] and source['sha256'] == copy['sha256']
                     for source, copy in zip(sources, copies)), 'executing source snapshot differs')
        training.publish(output/'historical_sources.json', dict(format=FORMAT,
            source_archive=handoff['archive_record'], previous_summary=handoff['summary_record'],
            bindings=handoff['bindings'], original_identities_rewritten=False))
        plan = handoff['plan']
        runtime = plan['runtime']
        _require(runtime.get('complete') is True and runtime.get('runtime_dlopen_covered') is False and
                 set(runtime['environment']) == {'LD_LIBRARY_PATH'}, 'unexpected frozen runtime')
        os.environ.update(runtime['environment'])
        frozen = outer._unique([*handoff['physical'], *sources, *copies,
                               training.record(output/'historical_sources.json')])
        training.verify_records(frozen)
        request = dict(format=FORMAT, created_utc=training.now(),
            runner_identity=training.process_identity(os.getpid()), config=asdict(core.GPT2Config()),
            implementation=IMPORTED_SOURCES, test_sources=test_sources,
            protocol=protocol,
            layout=LAYOUT, groups=core.groups(layout=LAYOUT),
            cells={key:list(value) for key,value in core.cube_subsets(layout=LAYOUT).items()},
            directions=plan['directions'], cases=plan['cases'], inherited_cells=ANCHORS,
            cell_order=list(CELL_ORDER), suites=list(SUITES), native_measurement_count=32,
            previous_summary=handoff['summary_record'], source_archive=handoff['archive_record'],
            binaries=plan['binaries'], runtime=runtime, gpu=plan['gpu'], frozen_inputs=frozen,
            training_restarted=False, goal_completion_claimed=False,
            interpretation='A includes LN1 and attention; M includes LN2 and MLP. E is tied in both roles. R contains upstream positions too; these are weight-bundle interventions, not unique storage.')
        training.publish(output/'request.json', request)
        frozen = outer._unique([*frozen, training.record(output/'request.json')])
        archive = handoff['archive']
        for name in DIRECTIONS:
            print(f'{training.now()} preparing {name}: eight early-branch cells', flush=True)
            _state(output, completed, 'preparing_models', direction=name)
            direction = plan['directions'][name]
            stage = output/name
            (stage/'cells').mkdir(parents=True)
            models = {}
            for cell in CELL_ORDER:
                parent = stage/'cells'/cell
                parent.mkdir()
                models[cell] = core.build_model(direction['recipient']['path'], direction['donor']['path'],
                    parent/'step_331', request['cells'][cell], direction['rows'], layout=LAYOUT)
            training.publish(stage/'models.json', models)
            active = outer._unique([*frozen, training.record(stage/'models.json'),
                                    *(record for model in models.values() for record in model['records'])])
            scores, ledgers, controls = {}, {}, {}
            for cell in CELL_ORDER:
                scores[cell], ledgers[cell], loaded = {}, {}, {}
                reference = (historical_model(models[cell], handoff['old_models'][name][ANCHORS[cell]], archive)
                             if cell in ANCHORS else None)
                cell_controls = {}
                for suite in SUITES:
                    _state(output, completed, 'native_measurement', direction=name, cell=cell, suite=suite)
                    print(f'{training.now()} {name}/{cell}/{suite}', flush=True)
                    directory = stage/'cells'/cell/suite
                    directory.mkdir()
                    score_dir, ledger = directory/'scores', directory/'execution.json'
                    cases_path = plan['cases'][suite]['path']
                    batch = native._json(cases_path)['packed_batch']['path']
                    binary = plan['binaries']['loss_probe']['path']
                    command = [binary, '--checkpoint='+models[cell]['paths']['patched'],
                               '--batch='+batch, '--output_dir='+str(score_dir), '--batch_sequences=1']
                    training.verify_records(active)
                    inputs = [record['path'] for record in active] + [cases_path, batch, binary]
                    screen.run_native(command, inputs, score_dir, directory/'process.log', ledger, plan['gpu'])
                    measured = core.load_native(models[cell], cases_path, suite, score_dir, ledger,
                                                case_archive=archive)
                    if reference is not None:
                        previous_dir = previous/name/'cells'/ANCHORS[cell]/suite
                        old = core.load_native(reference, cases_path, suite, previous_dir/'scores',
                            previous_dir/'execution.json', case_archive=archive, execution_archive=archive)
                        cell_controls[suite] = dict(previous_cell=ANCHORS[cell],
                            exact_native_parity=core.assert_native_copy_parity(old, measured))
                        if cell in ('E', 'EC'):
                            reference_names = ('main',) if suite == 'main' else ('word_next_native', 'shared_piece')
                            cell_controls[suite]['fp64_reference'] = {
                                label: core.compare_fp64_reference(measured,
                                    native._json(direction['references'][cell][label]['path']), 'JJ')
                                for label in reference_names}
                    loaded[suite] = measured
                    scores[cell][suite], ledgers[cell][suite] = str(score_dir), str(ledger)
                    active = outer._unique([*active, *measured['records'], training.record(directory/'process.log')])
                    training.verify_records(active)
                    completed.append(name+'/'+cell+'/'+suite)
                core._cross_suite(loaded['main'], loaded['supplemental'])
                if cell in ANCHORS:
                    controls[cell] = cell_controls
                active = outer._unique([*active, *screen.stage_result(stage/'cells'/cell,
                    dict(cell=cell, groups=request['cells'][cell], anchor_controls=cell_controls,
                         first_three_cross_suite_byte_equal=True, scores=scores[cell], executions=ledgers[cell]))])
                print(f'{training.now()} completed {name}/{cell}; {len(completed)}/32 native measurements', flush=True)
            _require(set(controls) == set(ANCHORS) and all(set(value) == set(SUITES) for value in controls.values()),
                     'missing inherited-cell controls')
            training.publish(stage/'baseline_controls.json', dict(format=FORMAT, controls=controls,
                inherited_all_context_loss_and_argmax_byte_equal=True))
            baseline_record = training.record(stage/'baseline_controls.json')
            active = outer._unique([*active, baseline_record])
            _state(output, completed, 'cube_readout', direction=name)
            report = core.analyze_cube(models, {suite:plan['cases'][suite]['path'] for suite in SUITES},
                scores, ledgers, stage/'cube_readout.json', rows=direction['rows'], layout=LAYOUT, case_archive=archive)
            _require(report.get('format') == core.EARLY_BRANCHES_FORMAT and report.get('layout') == LAYOUT and
                     report.get('complete') is True and report.get('baseline_controls_certified') is False and
                     report.get('cells') == request['cells'], 'incomplete or misrouted numerical readout')
            report_record = training.record(stage/'cube_readout.json')
            _require(native._json(report_record['path']) == report, 'published numerical readout differs')
            training.verify_records([*active, *report['files'], report_record])
            final_records.extend(screen.stage_result(stage, dict(direction=name, cube_complete=True,
                baseline_controls_certified=True, readout=report_record,
                baseline_controls=baseline_record)))
        _require(len(completed) == len(set(completed)) == 32, 'incomplete measurement coverage')
        training.verify_records([*frozen, *final_records])
        _state(output, completed, 'complete')
        training.publish(output/'summary.json', dict(format=FORMAT, complete=True, completed_utc=training.now(),
            completed=completed, native_measurement_count=32, baseline_controls_certified=True,
            previous_summary=handoff['summary_record'], source_archive=handoff['archive_record'],
            artifacts=[training.record(path) for path in sorted(output.rglob('*')) if path.is_file()],
            frozen_inputs=frozen, training_restarted=False, goal_completion_claimed=False))
    except BaseException as error:
        training.publish(output/'failure.json', dict(format=FORMAT, failed_utc=training.now(),
            error_type=type(error).__name__, error=str(error), completed=completed,
            no_automatic_restart=True, training_restarted=False, goal_completion_claimed=False))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous', required=True, type=Path)
    parser.add_argument('--source-archive', required=True, type=Path)
    parser.add_argument('--summary-sha256', required=True)
    parser.add_argument('--archive-sha256', required=True)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'early-branch runner received signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    run(args.previous, args.source_archive, args.summary_sha256, args.archive_sha256, args.output)


if __name__ == '__main__':
    main()
