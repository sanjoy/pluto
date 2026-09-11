"""Own the fixed E-conditioned Q/early/late experiment in a NEW directory.

The prepare-only plan and all prior evidence remain immutable. This runner
never starts training, rebuilds native code, or retries an interrupted run.
Preserved E/EC checkpoints and fresh independent copies are scored with the
same native loss kernel; every loss and argmax bit must agree. Comparing that
kernel with saved-logit FP64 likelihoods is a separate bounded rounding check.

Each completed cell is recorded before moving on. Only after all eight cells,
both suites, and the two baseline controls pass is a direction certified.
The final summary means this fixed experiment finished, not that the word's
mechanism has been localized or that the broader research goal is complete.
"""

import argparse
from dataclasses import asdict
import os
from pathlib import Path
import signal

from . import paired_complement_localization as core

outer = core.outer
training = core.training
screen = outer.screen
execution = screen.execution
FORMAT = 'pluto-paired-complement-localization-run-v1'
DIRECTIONS = ('original_to_replacement', 'replacement_to_original')
CELL_ORDER = ('E', 'EC', 'Q', 'H', 'L', 'QH', 'QL', 'HL')
SUITES = ('main', 'supplemental')
_require = core._require
# Capture the new runner at import, before a potentially long handoff audit.
# Later edits must not be recorded as if they were the code already executing.
RUNNER_SOURCE = training.record(__file__)


def validate_plan(plan_path, output):
    """Require the frozen, fixed scope, actual upstream exit, and idle GPU."""
    plan_path, output = Path(plan_path).absolute(), Path(output).absolute()
    _require(plan_path.is_file() and not plan_path.is_symlink()
             and plan_path.name == 'plan.json', 'expected a regular prepared plan.json')
    plan_record = training.record(plan_path)
    plan = training.read_json(plan_path)
    root = Path(plan['amendment_root']).resolve(strict=True)
    _require(plan_path.parent.parent == root and not plan_path.parent.is_symlink()
             and output.parent == root and output != plan_path.parent
             and not output.exists() and not output.is_symlink(),
             'runner output must be a NEW sibling of the prepared plan')
    config = core.GPT2Config()
    _require(plan.get('format') == core.FORMAT and plan.get('phase') == 'planned'
             and plan.get('config') == asdict(config) and plan.get('step') == 331
             and plan.get('native_execution_performed') is False
             and plan.get('runner_implemented') is False
             and plan.get('groups') == core.groups(config)
             and plan.get('cells') == {k: list(v) for k, v in core.cube_subsets().items()}
             and set(plan['directions']) == set(DIRECTIONS)
             and set(plan['cases']) == set(SUITES), 'plan differs from fixed endpoint cube')
    _require(plan['loss_tolerance']['absolute'] == core.FP32_ABSOLUTE_TOLERANCE
             and plan['loss_tolerance']['relative'] == core.FP32_RELATIVE_TOLERANCE,
             'prepared FP32 comparison tolerance changed')
    for key in ('LD_PRELOAD', 'LD_AUDIT'):
        _require(key not in os.environ, 'unset inherited ' + key)
    training.verify_records(plan['frozen_inputs'])
    frozen = {r['path']: r for r in outer._unique(plan['frozen_inputs'])}
    _require(training.record(core.__file__) == frozen.get(str(Path(core.__file__).resolve())),
             'prepared localization implementation changed')
    handoff = core.validate_outer_handoff(root, plan['outer_directory'])
    _require(plan['outer_runner'] == handoff['request']['runner_identity']
             and plan['outer_summary'] == training.record(Path(plan['outer_directory'])/'summary.json')
             and plan['gpu'] == handoff['gpu']
             and plan['binaries'] == handoff['request']['binaries']
             and plan['runtime'] == handoff['request']['runtime'], 'upstream/runtime binding differs')
    for suite, folder in (('main', 'word_cases'), ('supplemental', 'supplemental_cases')):
        expected = training.record(root/folder/'cases.json')
        _require(plan['cases'][suite] == expected and frozen.get(expected['path']) == expected,
                 'case source differs from prepared corpus')
    for item in handoff['items']:
        name = item['donor_arm']+'_to_'+item['recipient_arm']
        direction = plan['directions'][name]
        _require(direction['recipient'] == item['recipient_checkpoint']
                 and direction['donor'] == item['donor_checkpoint']
                 and direction['rows'] == item['embedding_rows'], 'direction endpoint/rows differ')
        _require(set(direction['references']) == set(direction['reference_checkpoints']) == {'E', 'EC'},
                 'missing preserved E/EC references')
        for cell in ('E', 'EC'):
            _require(set(direction['references'][cell]) == set(outer.SUITES), 'reference suites differ')
            for suite, record in direction['references'][cell].items():
                _require(frozen.get(record['path']) == record, 'reference readout was not frozen')
                report = training.read_json(record['path'])
                _require(report['execution_provenance'].get('verified') is True
                         and report['patch']['paths']['J'] == direction['reference_checkpoints'][cell]
                         and report['patch']['selected_rows'] == direction['rows']
                         and report['cases'] == plan['cases']['main' if suite == 'main' else 'supplemental'],
                         'preserved reference readout binding differs')
    training.verify_records(plan['frozen_inputs'])
    outer.require_exited(plan['outer_runner'])
    training.require_idle_gpu(plan['gpu'])
    training.verify_records([plan_record, RUNNER_SOURCE])
    return plan


def reference_model(copied_model, preserved_path, config=core.GPT2Config()):
    """Bind a preserved checkpoint truthfully, without rewriting patch history.

    In particular the old EC patch is C -> EC, not the direct A -> EC cube
    construction. Its real recipient/donor sources are retained. Equivalence
    to the independent cube copy is established by ALL actual output bytes.
    This adapter is for load_native, never for direct-cube analyze_cube.
    """
    path = Path(preserved_path).absolute()
    copied = Path(copied_model['paths']['patched'])
    _require(path != copied and path.is_dir() and not path.is_symlink(),
             'preserved reference must be a distinct real checkpoint')
    records = {r['path']: r for r in outer._unique(copied_model['records'])}
    training.verify_records(list(records.values()))
    patch_record = core.native._register(records, path/'patch.json')
    patch = training.read_json(path/'patch.json')
    _require(patch.get('format') == 'pluto-paired-weight-patch-v1'
             and patch.get('complete') is True and patch.get('config') == asdict(config)
             and patch['output']['path'] == str(path), 'invalid preserved patch identity')
    paths = dict(recipient=patch['sources']['original']['path'],
                 donor=patch['sources']['replacement']['path'], patched=str(path))
    hashes = dict(recipient=patch['sources']['original']['weights_sha256'],
                  donor=patch['sources']['replacement']['weights_sha256'],
                  patched=patch['output']['weights_sha256'])
    _require(hashes['patched'] == copied_model['hashes']['patched'],
             'preserved reference differs from independently built cube model')
    weights = {role: core.branch._weights(p, hashes[role], records, config)
               for role, p in paths.items()}
    old = core.GPT2Checkpoint(path, config, check_finite=True)
    fresh = core.GPT2Checkpoint(copied, config, check_finite=True)
    for spec in core.tensor_manifest(config):
        _require(core.native._bits_equal(old[spec.name], fresh[spec.name]),
                 'preserved/copy bytes differ')
        left, right = (p.stat() for p in (path/spec.filename, copied/spec.filename))
        _require((left.st_dev, left.st_ino) != (right.st_dev, right.st_ino),
                 'preserved and copied weights share an inode')
    training.verify_records(list(records.values()))
    return dict(paths=paths, hashes=hashes, weight_records=weights, patch=patch_record,
                records=outer._unique(list(records.values())), reference_only=True,
                expected_cube_copy_patch=copied_model['patch'],
                preserved_patch_schema_not_reinterpreted=True)


def run_native(plan, model, suite, directory, frozen):
    """One serial, recorded native measurement; no retry or background child."""
    directory = Path(directory)
    directory.mkdir()
    source = plan['cases'][suite]['path']
    batch = training.read_json(source)['packed_batch']['path']
    scores, ledger = directory/'scores', directory/'execution.json'
    binary = plan['binaries']['loss_probe']['path']
    command = [binary, '--checkpoint='+model['paths']['patched'], '--batch='+batch,
               '--output_dir='+str(scores), '--batch_sequences=1']
    active = outer._unique([*frozen, *model['records']])
    training.verify_records(active)
    inputs = [binary, source, batch, *execution.checkpoint_inputs(model['paths']['patched']),
              *(r['path'] for r in active)]
    screen.run_native(command, inputs, scores, directory/'process.log', ledger, plan['gpu'])
    measured = core.load_native(model, source, suite, scores, ledger)
    training.verify_records([*active, *measured['records']])
    return measured, str(scores), str(ledger)


def compare_references(measured, direction, cell, suite):
    """Compare every prescribed target, retaining the fixed rounding bound."""
    names = ('main',) if suite == 'main' else ('word_next_native', 'shared_piece')
    result = {}
    for name in names:
        record = direction['references'][cell][name]
        training.verify_records([record])
        report = training.read_json(record['path'])
        result[name] = core.compare_fp64_reference(measured, report, 'JJ')
    return result


def _state(output, completed, phase, **details):
    training.publish(output/'state.json', dict(format=FORMAT, phase=phase,
        runner_pid=os.getpid(), updated_utc=training.now(), completed=list(completed), **details), exclusive=False)


def _measurement_records(measured, directory):
    return outer._unique([*measured['records'], training.record(Path(directory)/'process.log')])


def run(plan_path, output):
    """Execute the preserved plan into a NEW sibling, retaining partial work."""
    plan_record = training.record(plan_path)
    plan = validate_plan(plan_path, output)
    training.verify_records([plan_record, RUNNER_SOURCE])
    output = Path(output).absolute()
    output.mkdir()
    completed, final_records = [], []
    try:
        sources = outer._unique([RUNNER_SOURCE, *[training.record(p) for p in
            (core.__file__, screen.__file__, execution.__file__, training.__file__)]])
        (output/'source').mkdir()
        copies = [execution.paired_training.freeze_file(r['path'], output/'source'/Path(r['path']).name)
                  for r in sources]
        _require(all(all(source[key] == copied[key] for key in ('bytes', 'sha256'))
                     for source, copied in zip(sources, copies)), 'source copy differs from executing source')
        frozen = outer._unique([*plan['frozen_inputs'], plan_record, *sources, *copies])
        training.verify_records(frozen)
        os.environ.update(plan['runtime']['environment'])
        training.publish(output/'request.json', dict(format=FORMAT, created_utc=training.now(),
            plan=plan_record, runner_identity=training.process_identity(os.getpid()),
            directions=list(DIRECTIONS), cell_order=list(CELL_ORDER), suites=list(SUITES),
            native_measurement_count=40, binaries=plan['binaries'], runtime=plan['runtime'],
            gpu=plan['gpu'], frozen_inputs=frozen, training_restarted=False,
            goal_completion_claimed=False))
        frozen = outer._unique([*frozen, training.record(output/'request.json')])
        for name in DIRECTIONS:
            print(f'{training.now()} preparing {name}: eight E-fixed cube copies', flush=True)
            _state(output, completed, 'preparing_models', direction=name)
            direction = plan['directions'][name]
            stage = output/name
            stage.mkdir(); (stage/'cells').mkdir(); (stage/'references').mkdir()
            models, references = {}, {}
            for cell in CELL_ORDER:
                parent = stage/'cells'/cell
                parent.mkdir()
                models[cell] = core.build_model(direction['recipient']['path'], direction['donor']['path'],
                    parent/'step_331', plan['cells'][cell], direction['rows'])
            for cell in ('E', 'EC'):
                references[cell] = reference_model(models[cell], direction['reference_checkpoints'][cell])
                (stage/'references'/cell).mkdir()
            training.publish(stage/'models.json', dict(cube=models, preserved_references=references))
            active = outer._unique([*frozen, training.record(stage/'models.json'),
                *(r for model in [*models.values(), *references.values()] for r in model['records'])])
            training.verify_records(active)
            reference_scores, reference_checks = {}, {}
            for cell in ('E', 'EC'):
                reference_scores[cell], reference_checks[cell] = {}, {}
                for suite in SUITES:
                    _state(output, completed, 'reference_measurement', direction=name, cell=cell, suite=suite)
                    print(f'{training.now()} {name}/reference_{cell}/{suite}', flush=True)
                    directory = stage/'references'/cell/suite
                    measured, _, _ = run_native(plan, references[cell], suite, directory, active)
                    reference_scores[cell][suite] = measured
                    reference_checks[cell][suite] = compare_references(measured, direction, cell, suite)
                    active = outer._unique([*active, *_measurement_records(measured, directory)])
                    completed.append(name+'/reference_'+cell+'/'+suite)
                core._cross_suite(reference_scores[cell]['main'], reference_scores[cell]['supplemental'])
                active = outer._unique([*active, *screen.stage_result(stage/'references'/cell,
                    dict(cell=cell, fp64_reference_checks=reference_checks[cell],
                         all_targets_checked=True, first_three_cross_suite_byte_equal=True))])
            scores, ledgers, controls = {}, {}, {}
            for cell in CELL_ORDER:
                scores[cell], ledgers[cell], loaded = {}, {}, {}
                controls_for_cell = {}
                for suite in SUITES:
                    _state(output, completed, 'cube_measurement', direction=name, cell=cell, suite=suite)
                    print(f'{training.now()} {name}/{cell}/{suite}', flush=True)
                    directory = stage/'cells'/cell/suite
                    measured, scores[cell][suite], ledgers[cell][suite] = run_native(plan, models[cell], suite, directory, active)
                    loaded[suite] = measured
                    if cell in ('E', 'EC'):
                        controls_for_cell[suite] = dict(
                            same_native_copy=core.assert_native_copy_parity(reference_scores[cell][suite], measured),
                            fp64_reference=compare_references(measured, direction, cell, suite))
                    active = outer._unique([*active, *_measurement_records(measured, directory)])
                    completed.append(name+'/'+cell+'/'+suite)
                core._cross_suite(loaded['main'], loaded['supplemental'])
                if cell in ('E', 'EC'):
                    controls[cell] = controls_for_cell
                training.verify_records(active)
                active = outer._unique([*active, *screen.stage_result(stage/'cells'/cell,
                    dict(cell=cell, groups=plan['cells'][cell], copy_controls=controls_for_cell,
                         first_three_cross_suite_byte_equal=True, scores=scores[cell], executions=ledgers[cell]))])
                print(f'{training.now()} completed {name}/{cell}; {len(completed)}/40 native measurements', flush=True)
            _require(set(controls) == {'E', 'EC'} and all(set(v) == set(SUITES) for v in controls.values()),
                     'direction cannot complete without all E/EC controls')
            training.publish(stage/'baseline_controls.json', dict(format=FORMAT,
                baseline_controls_certified=True, reference_checks=reference_checks, copied_model_checks=controls))
            active = outer._unique([*active, training.record(stage/'baseline_controls.json')])
            _state(output, completed, 'cube_readout', direction=name)
            report_path = stage/'cube_readout.json'
            report = core.analyze_cube(models, {suite:plan['cases'][suite]['path'] for suite in SUITES},
                scores, ledgers, report_path, rows=direction['rows'])
            _require(report.get('complete') is True and report.get('baseline_controls_certified') is False,
                     'unexpected core readout certification')
            training.verify_records([*active, *report['files']])
            final_records.extend(screen.stage_result(stage, dict(direction=name,
                readout=training.record(report_path), baseline_controls=training.record(stage/'baseline_controls.json'),
                baseline_controls_certified=True, cube_complete=True)))
            print(f'{training.now()} certified complete direction {name}', flush=True)
        _require(len(completed) == 40 and len(set(completed)) == 40, 'native measurement coverage differs')
        training.verify_records([*frozen, *final_records])
        _state(output, completed, 'complete')
        artifacts = [training.record(p) for p in sorted(output.rglob('*')) if p.is_file()]
        training.publish(output/'summary.json', dict(format=FORMAT, complete=True, completed_utc=training.now(),
            plan=plan_record, completed=completed, native_measurement_count=40,
            baseline_controls_certified=True, artifacts=artifacts, frozen_inputs=frozen,
            training_restarted=False, goal_completion_claimed=False,
            completion_meaning='Both fixed endpoint cubes and their baseline controls measured; not mechanistic-goal completion.'))
        return training.read_json(output/'summary.json')
    except BaseException as error:
        training.publish(output/'failure.json', dict(format=FORMAT, failed_utc=training.now(),
            error_type=type(error).__name__, error=str(error), completed=completed,
            no_automatic_restart=True, training_restarted=False, goal_completion_claimed=False))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'localization runner received signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    run(args.plan, args.output)


if __name__ == '__main__':
    main()
