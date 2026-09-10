"""Recover the paired causal screen after its confirmed loader-only failure.

This is an explicit new analysis run, not an automatic retry or a training
restart. The failed directory is immutable evidence. We accept only failure
before the first intervention, authenticate the completed training/scoring,
and require the freshly prepared intervention plan to equal the original plan.
The native executables remain byte-identical; only their missing runtime
libraries are packaged. No measurement or failed execution is relabeled.
"""

import argparse
import os
from pathlib import Path
import signal

from . import paired_lowercase_causal_followup as screen
from . import native_runtime_bundle

training = screen.training
execution = screen.execution
FORMAT = 'pluto-paired-causal-loader-recovery-v1'


def require_exited(identity):
    """An unreadable/reused process is an error, not permission to relaunch."""
    if training.process_live(identity):
        raise RuntimeError('previous analysis process is still live')


def validate_failure(root, previous):
    """Only recover the known pre-intervention missing-library failure."""
    if previous.parent != root or previous.is_symlink():
        raise ValueError('previous analysis must be a real immediate child')
    request = training.read_json(previous / 'request.json')
    state = training.read_json(previous / 'state.json')
    failure = training.read_json(previous / 'failure.json')
    if (request.get('format') != screen.FORMAT
            or request.get('amendment_root') != str(root)
            or request.get('stage') != 'embedding_and_branches'
            or state.get('phase') != 'gpu_validation'
            or state.get('completed') != []
            or state.get('runner_pid') != request['runner_identity']['pid']
            or failure.get('type') != 'RuntimeError'
            or failure.get('training_restarted') is not False
            or failure.get('error') !=
            f'analysis child exited 127; inspect {previous / "gpu_test.log"}'):
        raise ValueError('not a loader-only failure before interventions')
    if 'error while loading shared libraries:' not in (previous / 'gpu_test.log').read_text():
        raise ValueError('missing native loader-error evidence')
    for name in ('summary.json', 'gpu_test_execution.json'):
        if (previous / name).exists():
            raise ValueError('previous stage contains completed execution')
    plan = training.read_json(previous / 'plan/plan.json')
    if any((previous / item['name']).exists() for item in plan['interventions']):
        raise ValueError('an intervention was already materialized')
    require_exited(request['runner_identity'])
    require_exited(request['upstream_identity'])
    training.verify_records(request['frozen_inputs'])
    records = [training.record(previous / name) for name in
               ('request.json', 'state.json', 'failure.json', 'gpu_test.log',
                'plan/plan.json')]
    return request, plan, records


def check_downstream(root):
    """Do not race the retired source/head observers or silently restart them."""
    records = []
    expected_errors = {
        'historical_source_value_stage': 'upstream causal stage failed; no GPU work started',
        'historical_head_position_stage': 'historical source stage failed; no GPU work authorized',
    }
    for name, expected_error in expected_errors.items():
        directory = root / name
        if not directory.exists():
            continue
        request = training.read_json(directory / 'request.json')
        require_exited(request['runner_identity'])
        failure = training.read_json(directory / 'failure.json')
        if (failure.get('error') != expected_error
                or failure.get('training_restarted') is not False):
            raise ValueError('downstream did not stop on upstream failure')
        records.extend(training.record(directory / filename)
                       for filename in ('request.json', 'failure.json'))
    return records


def require_same_plan(actual, original):
    if actual != original:
        raise ValueError('fresh intervention plan differs from predeclared plan')


def run(args):
    root = Path(args.root).resolve(strict=True)
    previous = Path(args.previous).absolute()
    output = Path(args.output).absolute()
    if output.parent != root or output.exists() or output.is_symlink():
        raise ValueError('recovery requires a NEW immediate-child directory')
    prior, old_plan, records = validate_failure(root, previous)
    records.extend(check_downstream(root))
    summary_path = root / 'analysis_trajectory/summary.json'
    summary = training.read_json(summary_path)
    if (summary.get('format') != screen.trajectory.FORMAT
            or summary.get('complete') is not True
            or summary.get('goal_completion_claimed') is not False):
        raise ValueError('trajectory is not complete')
    training.verify_records([*summary['frozen_inputs'], *summary['terminal_records']])
    records.append(training.record(summary_path))
    # This process can own only its new files and native children.
    output.mkdir()
    (output / 'bin').mkdir()
    (output / 'source').mkdir()
    try:
        plan = screen.planner.prepare(root, summary_path, root / 'causal_cases/exports.json',
                                      output / 'plan')
        require_same_plan(plan, old_plan)
        binaries = {
            name: execution.paired_training.freeze_file(
                item['path'], output / 'bin' / name, executable=True)
            for name, item in prior['binaries'].items()
        }
        for name, item in binaries.items():
            if any(item[key] != prior['binaries'][name][key] for key in ('bytes', 'sha256')):
                raise ValueError('native executable changed during recovery')
        runtime = native_runtime_bundle.freeze_runtime(
            args.reference_gpu_test, binaries['gpu_test']['path'], output / 'runtime')
        # Only the explicit library-search override is recorded. Never dump the
        # inherited environment, which can contain unrelated credentials.
        os.environ.update(runtime['environment'])
        sources = [training.record(path) for path in
                   (__file__, native_runtime_bundle.__file__)]
        copies = [execution.paired_training.freeze_file(
            item['path'], output / 'source' / Path(item['path']).name) for item in sources]
        frozen = [*prior['frozen_inputs'], *records, *sources, *copies,
                  *binaries.values(), *runtime['frozen_records'],
                  training.record(output / 'plan/plan.json')]
        frozen = list({item['path']: item for item in frozen}.values())
        training.verify_records(frozen)
        identity = training.process_identity(os.getpid())
        training.publish(output / 'request.json', dict(
            format=FORMAT, stage='embedding_and_branches', created_utc=training.now(),
            amendment_root=str(root), previous_analysis=str(previous),
            runner_identity=identity, binaries=binaries, runtime=runtime,
            frozen_inputs=frozen, training_restarted=False,
            same_predeclared_plan=True, goal_completion_claimed=False))
        frozen.append(training.record(output / 'request.json'))
        # Recheck handles after potentially slow hashing/packaging, before GPU.
        validate_failure(root, previous)
        check_downstream(root)
        gpu = training.read_json(root / 'request.json')['gpu']
        tests = output / 'gpu_validation'
        tests.mkdir()
        test_file = tests / 'gtest.json'
        training.publish(output / 'state.json', dict(
            format=FORMAT, phase='gpu_validation', runner_pid=os.getpid(), completed=[]))
        print(f'{training.now()} validating unchanged GPU test with packaged runtime', flush=True)
        test_run = screen.run_native(
            [binaries['gpu_test']['path'], f'--gtest_output=json:{test_file}'],
            [binaries['gpu_test']['path'], *(item['path'] for item in runtime['frozen_records'])],
            tests, output / 'gpu_test.log', output / 'gpu_test_execution.json', gpu)
        execution.require_gpu_test_success(test_file)
        training.verify_records(frozen)
        print(f'{training.now()} GPU tests passed; starting unchanged intervention plan', flush=True)
        completed, artifacts = screen.run_interventions(
            root, output, summary, plan, binaries, gpu, frozen)
        training.verify_records([*frozen, *test_run['outputs'], test_run['log']])
        result = dict(format=screen.FORMAT, complete=True, goal_completion_claimed=False,
                      completed_utc=training.now(), recovery_format=FORMAT,
                      plan=training.record(output / 'plan/plan.json'), gpu_test=test_run,
                      completed=completed, completed_artifact_records=artifacts,
                      frozen_inputs=frozen, runtime=runtime, training_restarted=False,
                      remaining_research=[
                          'Interpret transfer causality and collateral controls.',
                          'Localize the implicated computations; test interactions and rescue.',
                          'Recover separately queued historical follow-ups, without relabeling them.'])
        training.publish(output / 'summary.json', result)
        training.publish(output / 'state.json', dict(
            format=FORMAT, phase='complete', runner_pid=os.getpid(),
            completed=[item['intervention']['name'] for item in completed]), exclusive=False)
        return result
    except BaseException as error:
        training.publish(output / 'failure.json', dict(
            type=type(error).__name__, error=str(error), failed_utc=training.now(),
            no_automatic_restart=True, training_restarted=False))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'previous', 'output', 'reference-gpu-test'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args(argv)

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'causal recovery received signal {signum}')

    signal.signal(signal.SIGTERM, interrupted)
    run(args)


if __name__ == '__main__':
    main()
