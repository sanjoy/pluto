"""Queue the fixed HISTORICAL suffix assay after amended paired interventions.

Only the queue adapter is new. The historical event, five source controls,
four doses, native probe and independent readout are unchanged. In particular,
these measurements are NOT from either new paired model and do not test the
initial Ex/ex prediction. No training process is started, restarted or signaled.
"""

import argparse
import os
from pathlib import Path
import signal
import sys
import time

import numpy as np

from . import paired_lowercase_causal_followup as combined
from . import paired_lowercase_intervention_plan as planner
from . import paired_lowercase_training as training
from . import source_value_followup as historical
from . import source_value_readout as readout

FORMAT = 'pluto-paired-lowercase-source-value-followup-v1'
MODULE = 'weight_analysis.paired_lowercase_causal_followup'


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _flag(argv, flag):
    _require(argv.count(flag) == 1, 'missing or repeated observer flag: ' + flag)
    index = argv.index(flag)
    _require(index + 1 < len(argv), 'missing observer flag value: ' + flag)
    return argv[index + 1]


def observer_identity(root, upstream, pid, ticks, request):
    """Bind a live publisher to the actual immutable combined-stage request."""
    actual = training.process_identity(pid)
    argv = actual['argv']
    saved = request['runner_identity']
    _require(request.get('format') == combined.FORMAT
             and request.get('stage') == 'embedding_and_branches'
             and request.get('amendment_root') == str(root)
             and request.get('goal_completion_claimed') is False,
             'not the amended combined-stage request')
    _require(type(ticks) is int and actual['start_ticks'] == ticks
             and actual['state'] not in ('Z', 'X', 'x')
             and all(actual[key] == saved[key] for key in ('pid', 'start_ticks', 'argv'))
             and argv.count('-m') == 1 and argv.index('-m') + 1 < len(argv)
             and argv[argv.index('-m') + 1] == MODULE,
             'upstream is not the exact live combined observer')
    _require(Path(_flag(argv, '--root')).resolve() == root
             and Path(_flag(argv, '--output')).resolve() == upstream,
             'combined observer root/output differs')
    return actual


def wait_for_causal(path, expected, *, poll_seconds=30):
    """A completion marker never substitutes for two confirmed publisher exits."""
    if not 0 < poll_seconds <= 60:
        raise ValueError('poll interval must be in (0, 60]')
    path = Path(path)
    missing = errors = 0
    while True:
        if (path.parent / 'failure.json').exists():
            raise RuntimeError('upstream causal stage failed; no GPU work started')
        try:
            live = training.process_live(expected)
            errors = 0
        except (OSError, ValueError) as error:
            missing = 0
            errors += 1
            if errors >= 4:
                raise RuntimeError('cannot verify combined observer; no GPU work started') from error
            time.sleep(poll_seconds)
            continue
        missing = 0 if live else missing + 1
        if missing >= 2 and path.is_file():
            summary = training.read_json(path)
            _require(summary.get('format') == combined.FORMAT
                     and summary.get('complete') is True
                     and summary.get('goal_completion_claimed') is False,
                     'not a completed amended causal summary')
            return summary
        if missing >= 4:
            raise RuntimeError('combined observer exited without completed evidence')
        time.sleep(poll_seconds)


def validate_upstream(root, upstream, summary, output):
    """Freshly revalidate training and every prescribed causal intervention.

    Rebuilding the immutable plan in OUR output reruns the existing amended
    training, deterministic-control, checkpoint, native-case and actual-root
    validators. Comparing the complete plan avoids a second, weaker imitation
    of those gates. Nothing is written into training or upstream directories.
    """
    request_path = upstream / 'request.json'
    request_record = training.record(request_path)
    summary_path = upstream / 'summary.json'
    summary_record = training.record(summary_path)
    _require(training.read_json(summary_path) == summary,
             'upstream summary changed since confirmed handoff')
    request = training.read_json(request_path)
    _require(request.get('format') == combined.FORMAT
             and request.get('stage') == 'embedding_and_branches'
             and request.get('amendment_root') == str(root)
             and request.get('goal_completion_claimed') is False,
             'wrong amended upstream request')
    _require(summary.get('format') == combined.FORMAT
             and summary.get('complete') is True
             and summary.get('goal_completion_claimed') is False,
             'incomplete amended upstream summary')
    plan_path = upstream / 'plan' / 'plan.json'
    plan_record = training.record(plan_path)
    _require(summary.get('plan') == plan_record
             and summary.get('frozen_inputs') == [*request['frozen_inputs'], plan_record],
             'upstream summary differs from its frozen request and plan')
    training.verify_records(summary['frozen_inputs'])
    plan = training.read_json(plan_path)
    regenerated = planner.prepare(root, root / 'analysis_trajectory' / 'summary.json',
                                  root / 'causal_cases' / 'exports.json', output)
    _require(regenerated == plan, 'fresh completed-training plan differs from causal plan')
    training_request = training.read_json(root / 'request.json')
    _require(plan.get('format') == planner.FORMAT
             and plan.get('complete') is True
             and plan.get('patches_materialized') is False
             and plan.get('goal_completion_claimed') is False
             and plan.get('arm_roots') == {
                 'original': training_request['legacy_root'], 'replacement': str(root)},
             'plan relabels actual source roots or has wrong completion meaning')
    expected = [i for i in plan['interventions'] if i['kind'] == 'copy_control']
    expected += [i for i in plan['interventions'] if i['kind'] == 'embedding_rows']
    expected += [i for i in plan['interventions'] if i['tensors']]
    _require(len(expected) == len(plan['interventions']) and expected
             and len({i['name'] for i in expected}) == len(expected),
             'invalid prescribed intervention inventory')
    completed = summary.get('completed')
    _require(isinstance(completed, list)
             and [item.get('intervention') for item in completed] == expected,
             'upstream did not complete exactly the prescribed ordered screen')
    artifacts = []
    for item in completed:
        name = item['intervention']['name']
        _require(Path(name).name == name and name not in ('', '.', '..'),
                 'invalid completed intervention name')
        directory = upstream / name
        marker = directory / 'complete.json'
        _require(training.read_json(marker) == item,
                 'per-intervention completion differs from summary')
        declared = [*item['artifacts'], training.record(marker)]
        actual = historical.recursive_outputs(directory)
        _require(declared and sorted(declared, key=lambda r: r['path']) == actual,
                 'completed intervention artifact inventory differs from disk')
        training.verify_records(declared)
        artifacts.extend(declared)
    _require(summary.get('completed_artifact_records') == artifacts,
             'combined completed-artifact ledger differs')
    test = summary['gpu_test']
    test_record = training.record(upstream / 'gpu_test_execution.json')
    _require(test.get('returncode') == 0
             and training.read_json(upstream / 'gpu_test_execution.json') == test,
             'upstream native-test execution differs')
    training.verify_records([*test['inputs_before'], *test['outputs'], test['log']])
    historical.causal.require_gpu_test_success(upstream / 'gpu_validation' / 'gtest.json')
    records = [request_record, summary_record,
               plan_record, training.record(output / 'plan.json'),
               test_record,
               *summary['frozen_inputs'], *plan['frozen_inputs'], *plan['implementation'],
               *artifacts, *test['inputs_before'], *test['outputs'], test['log']]
    # Reject accidental competing hashes rather than silently overwriting one.
    unique = {}
    for item in records:
        _require(item['path'] not in unique or unique[item['path']] == item,
                 'conflicting upstream provenance records')
        unique[item['path']] = item
    records = list(unique.values())
    training.verify_records(records)
    return records


def source_paths():
    # Reuse the old collector: it includes all currently imported analysis
    # modules, the readout, and the native diagnostic source set. Under -m this
    # adapter is __main__, so it MUST be added explicitly instead of relying
    # on a canonical weight_analysis.* entry in sys.modules.
    return sorted({Path(__file__).resolve(), *historical.source_paths()})


def run(args):
    root = Path(args.root).resolve(strict=True)
    upstream = Path(args.causal_output).resolve(strict=True)
    output = Path(args.output).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    _require(upstream.parent == root and output.parent == root,
             'upstream/output must be immediate amended-root children')
    request_path = upstream / 'request.json'
    upstream_request = training.read_json(request_path)
    expected = observer_identity(root, upstream, args.observer_pid,
                                 args.observer_start_ticks, upstream_request)
    training_request = training.read_json(root / 'request.json')
    _require(training_request.get('format') == training.FORMAT
             and training_request.get('amendment_root') == str(root),
             'wrong amended training request')
    training.verify_records([*training_request['frozen_inputs'],
                             *upstream_request['frozen_inputs']])
    checkpoint = Path(args.historical_checkpoint)
    _require(not checkpoint.is_symlink(), 'historical checkpoint must not be a link')
    assay, historical_inputs = historical.authenticate_assay(
        Path(args.historical_manifest).resolve(), Path(args.historical_root).resolve(),
        checkpoint.resolve())
    # Metadata only: an active trainer is expected here and is not disturbed.
    _require(training.gpu_snapshot()['device'] == training_request['gpu'],
             'GPU identity differs from amended training')
    originals = [training.record(path) for path in source_paths()]
    binary_sources = {name: training.record(getattr(args, name))
                      for name in ('probe', 'gpu_test')}
    output.mkdir()
    (output / 'bin').mkdir()
    binaries = {name: historical.paired_training.freeze_file(
        Path(item['path']), output / 'bin' / name, executable=True)
        for name, item in binary_sources.items()}
    _require(all(binaries[name]['sha256'] == item['sha256']
                 and binaries[name]['bytes'] == item['bytes']
                 for name, item in binary_sources.items()), 'native binary changed during freeze')
    repo = Path(__file__).resolve().parents[2]
    copies = []
    for item in originals:
        source = Path(item['path'])
        destination = output / 'source' / source.relative_to(repo)
        destination.parent.mkdir(parents=True, exist_ok=True)
        copies.append(historical.paired_training.freeze_file(source, destination))
    frozen = [training.record(root / 'request.json'), training.record(request_path),
              *historical_inputs, *originals, *copies,
              *binary_sources.values(), *binaries.values()]
    training.verify_records(frozen)
    request = dict(format=FORMAT, created_utc=training.now(),
        runner_identity=training.process_identity(os.getpid()), upstream_identity=expected,
        amendment_root=str(root), causal_output=str(upstream), gpu=training_request['gpu'],
        stage='historical_source_value', new_paired_model_result=False,
        initial_piece_assay=False, goal_completion_claimed=False, assay=assay,
        binaries=binaries, frozen_inputs=frozen,
        python_version=sys.version, numpy_version=np.__version__)
    training.publish(output / 'request.json', request)
    try:
        print(f'{training.now()} waiting for amended causal observer; historical assay only, no GPU work', flush=True)
        training.publish(output / 'state.json', dict(format=FORMAT, phase='waiting_causal'))
        summary = wait_for_causal(upstream / 'summary.json', expected)
        training.verify_records(frozen)
        upstream_inputs = validate_upstream(root, upstream, summary, output / 'upstream_revalidation_plan')
        training.publish(output / 'validated_upstream.json', dict(inputs=upstream_inputs,
            observer_identity=expected, observer_exit_code=None,
            note='Not our child; confirmed exit and complete artifacts, no invented exit code.'))
        training.verify_records([*frozen, *upstream_inputs])
        training.require_idle_gpu(request['gpu'])
        tests = output / 'gpu_validation'
        tests.mkdir()
        training.publish(output / 'state.json', dict(format=FORMAT, phase='gpu_validation'), exclusive=False)
        test_file = tests / 'gtest.json'
        test_run = historical.causal.execute_recorded(
            [binaries['gpu_test']['path'], f'--gtest_output=json:{test_file}'],
            [item['path'] for item in frozen], tests, output / 'gpu_test.log',
            output / 'gpu_test_execution.json')
        historical.causal.require_gpu_test_success(test_file)
        training.verify_records([*frozen, *upstream_inputs])
        training.require_idle_gpu(request['gpu'])
        training.publish(output / 'state.json', dict(format=FORMAT, phase='historical_source_value'), exclusive=False)
        native = output / 'native'
        command = [binaries['probe']['path'],
            f'--checkpoint={assay["historical_checkpoint"]}', f'--tokens_file={assay["prefix"]["path"]}',
            f'--output_dir={native}', f'--block={assay["block"]}', f'--head={assay["head"]}',
            f'--query={assay["query"]}', f'--sources={",".join(map(str, assay["sources"]))}',
            f'--target_id={assay["target_id"]}', f'--expected_logits={assay["expected_logits"]["path"]}']
        execution_path = output / 'probe_execution.json'
        execution = historical.execute_tree(command, [item['path'] for item in frozen],
            native, output / 'probe.log', execution_path)
        training.verify_records([*frozen, *upstream_inputs])
        report = readout.summarize(native, execution_path, assay)
        training.publish(output / 'readout.json', report)
        training.verify_records([*frozen, *upstream_inputs, *execution['outputs'],
                                 *test_run['inputs_before'], *test_run['outputs'], test_run['log']])
        result = dict(format=FORMAT, complete=True, completed_utc=training.now(),
            stage='historical_source_value', new_paired_model_result=False,
            initial_piece_assay=False, goal_completion_claimed=False, assay=assay,
            frozen_inputs=frozen, gpu_test=test_run, execution=training.record(execution_path),
            readout=training.record(output / 'readout.json'),
            upstream=training.record(output / 'validated_upstream.json'),
            limitations=['One historical unt prediction, not complete-word or new paired-model evidence.',
                         'Source-V query-local interventions are not whole-head or initial-piece ablations.',
                         'Integrate measured effects with paired transfers and wider context controls.'])
        training.publish(output / 'summary.json', result)
        training.publish(output / 'state.json', dict(format=FORMAT, phase='complete'), exclusive=False)
        print(f'{training.now()} historical suffix source-value assay complete: {output / "readout.json"}', flush=True)
        return result
    except BaseException as error:
        training.publish(output / 'failure.json', dict(type=type(error).__name__, error=str(error),
            failed_utc=training.now(), training_restarted=False, no_automatic_restart=True,
            goal_completion_claimed=False))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'causal-output', 'output', 'historical-manifest',
                 'historical-root', 'historical-checkpoint', 'probe', 'gpu-test'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('observer-pid', 'observer-start-ticks'):
        parser.add_argument('--' + name, type=int, required=True)
    args = parser.parse_args(argv)
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'historical source observer received signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    run(args)


if __name__ == '__main__':
    main()
