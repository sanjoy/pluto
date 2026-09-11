"""Run the calibrated historical head-position screen after the existing queue.

The earlier source-value observer must actually exit successfully with verified
evidence before this controller can use the GPU. Timed training is never
restarted, signaled, or competed with. Each new native invocation, complete
output tree, and independent readout is preserved. Historical first-piece
effects are not paired-model or complete-word results, and completing this
screen does not complete the mechanistic research goal.
"""

import argparse
import os
from pathlib import Path
import signal
import sys

import numpy as np

from . import checkpoint, head_position_handoff as handoff
from . import head_position_readout as readout
from . import paired_lowercase_training as training
from . import source_value_followup as common

FORMAT = 'pluto-head-position-followup-v1'
NATIVE_TEST_NAMES = {
    'NativeScopesDosesAndIndependentProjectionZeroAgree',
    'OtherSequenceAndFutureOnlyQueriesCannotChangeLogits',
    'RejectsWrongIdentityMalformedInputsAndOriginalMutation',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(path, *, directory=False):
    path = Path(path).absolute()
    require(not path.is_symlink() and path.resolve() == path
            and (path.is_dir() if directory else path.is_file()),
            'expected canonical existing nonsymlink input: ' + str(path))
    return path


def source_paths():
    """Freeze imported Python and native source, including this -m entry point.

    __main__ is not indexed by its normal package name. Including __file__
    explicitly prevents a long wait from silently outliving controller edits.
    Existing observers keep their own immutable lists; no such file is edited.
    """
    repo = Path(__file__).resolve().parents[2]
    paths = {Path(__file__).resolve()}
    for name, module in tuple(sys.modules.items()):
        path = getattr(module, '__file__', None)
        if name.startswith('weight_analysis.') and path and path.endswith('.py'):
            paths.add(Path(path).resolve())
    paths.update(repo / name for name in checkpoint.SOURCE_FILES)
    # These sources supply the native replay, actual GPU test, and public
    # CUDA/layer interfaces. Globbing is only for immutable preparation, never
    # a deletion or a repeated comparison whose target could silently expand.
    for directory in (repo / 'ai-slop/weight_analysis/head_context',
                      repo / 'src/cuda', repo / 'src/llm',
                      repo / 'src/llm/layers', repo / 'src/util'):
        for pattern in ('*.h', '*.cc', 'BUILD.bazel'):
            paths.update(directory.glob(pattern))
    return sorted(paths)


def command_for_case(binary, document, case, directory):
    """Construct the exact native command; never execute saved JSON commands."""
    return [str(binary), f'--checkpoint={document["checkpoint_directory"]}',
        f'--tokens_file={case["prefix"]["path"]}', f'--output_dir={directory}',
        f'--block={case["block"]}', f'--head={case["head"]}',
        f'--query={case["query"]}', f'--target_id={case["target_id"]}',
        f'--expected_clean_logits={case["calibration"]["clean_logits"]["path"]}',
        '--expected_all_query_zero_logits=' + case['calibration']['all_query_weight_zero_logits']['path']]


def require_native_tests(path):
    """A filtered, skipped, or unrelated successful test is insufficient."""
    common.causal.require_gpu_test_success(path)
    result = training.read_json(path)
    suites = result.get('testsuites', [])
    require(result['tests'] == len(NATIVE_TEST_NAMES) and len(suites) == 1
            and suites[0].get('name') == 'HeadContextGpuTest'
            and {case.get('name') for case in suites[0].get('testsuite', [])} == NATIVE_TEST_NAMES,
            'native validation must run the exact complete head-context suite')


def execute_tree(command, inputs, directory, log, execution_path):
    """Record an owned child and all 33 native files around immutable inputs.

    The native executable exclusively creates its directory. Failure leaves
    its partial files and log intact, without a successful execution record.
    The shared executor owns and reaps only this new child on interruption.
    """
    directory = Path(directory)
    for destination in (directory, Path(log), Path(execution_path)):
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(destination)
    paths = sorted({str(canonical(path)) for path in inputs})
    require(paths and str(canonical(command[0])) in paths,
            'actual native executable must be an input')
    before = [training.record(path) for path in paths]
    result = common.causal.execute(command, log)
    after = [training.record(path) for path in paths]
    require(before == after, 'native inputs changed during execution')
    readout.native_inventory(directory)
    result.update(format=readout.EXECUTION_FORMAT, inputs_before=before,
                  inputs_after=after, outputs=common.recursive_outputs(directory))
    training.publish(execution_path, result)
    return result


def run(args):
    root = canonical(args.root, directory=True)
    upstream = canonical(args.source_output, directory=True)
    descriptor = canonical(args.cases)
    output = Path(args.output).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    require(upstream.parent == root and output.parent == root
            and output.resolve() == output,
            'upstream/output must be canonical direct amendment-root children')
    upstream_request_path = upstream / 'request.json'
    upstream_request = training.read_json(upstream_request_path)
    expected = handoff.observer_identity(root, upstream, args.observer_pid,
                                         args.observer_start_ticks, upstream_request)
    training_request = training.read_json(root / 'request.json')
    require(training_request.get('format') == training.FORMAT
            and training_request.get('amendment_root') == str(root),
            'wrong amended training request')
    training.verify_records([*training_request['frozen_inputs'],
                             *upstream_request['frozen_inputs']])
    ledger = readout.Ledger()
    document, descriptor_record = readout.validate_descriptor(
        descriptor, ledger, checkpoint.GPT2Config(), False)
    ledger.recheck()
    # A live trainer is EXPECTED during preparation. This only observes GPU
    # identity; the exclusive-idle requirement belongs after queue completion.
    require(training.gpu_snapshot()['device'] == training_request['gpu'],
            'GPU identity differs from amended training')
    original_sources = [training.record(path) for path in source_paths()]
    binary_sources = {name: training.record(canonical(getattr(args, name)))
                      for name in ('probe', 'gpu_test')}
    output.mkdir()
    (output / 'bin').mkdir()
    binaries = {name: common.paired_training.freeze_file(
        Path(item['path']), output / 'bin' / name, executable=True)
        for name, item in binary_sources.items()}
    require(all(binaries[name]['sha256'] == item['sha256']
                and binaries[name]['bytes'] == item['bytes']
                for name, item in binary_sources.items()), 'native binary changed while freezing')
    repo = Path(__file__).resolve().parents[2]
    copies = []
    for item in original_sources:
        source = Path(item['path'])
        target = output / 'source' / source.relative_to(repo)
        target.parent.mkdir(parents=True, exist_ok=True)
        copies.append(common.paired_training.freeze_file(source, target))
    candidates = [training.record(root / 'request.json'),
        training.record(upstream_request_path), descriptor_record,
        *ledger.records.values(), *original_sources, *copies,
        *binary_sources.values(), *binaries.values()]
    frozen_map = {}
    for item in candidates:
        require(item['path'] not in frozen_map or frozen_map[item['path']] == item,
                'conflicting frozen input records')
        frozen_map[item['path']] = item
    frozen = [frozen_map[path] for path in sorted(frozen_map)]
    training.verify_records(frozen)
    request = dict(format=FORMAT, stage='historical_head_position',
        created_utc=training.now(), runner_identity=training.process_identity(os.getpid()),
        upstream_identity=expected, amendment_root=str(root), source_output=str(upstream),
        cases=descriptor_record, case_count=document['case_count'], gpu=training_request['gpu'],
        binaries=binaries, frozen_inputs=frozen, historical_only=True,
        new_paired_model_result=False, full_word_probability_claimed=False,
        goal_completion_claimed=False, python_version=sys.version, numpy_version=np.__version__)
    training.publish(output / 'request.json', request)
    completed = []
    try:
        training.publish(output / 'state.json', dict(format=FORMAT, phase='waiting_source_value', completed=[]))
        print(f'{training.now()} waiting for exact historical source-value observer exit; no GPU work', flush=True)
        summary = handoff.wait_for_source(upstream / 'summary.json', expected)
        training.verify_records(frozen)
        upstream_records = handoff.validate_upstream(root, upstream, summary, output / 'upstream_revalidation')
        upstream_record = output / 'validated_upstream.json'
        training.publish(upstream_record, dict(inputs=upstream_records, observer_identity=expected,
            observer_exit_code=None, note='Confirmed another parent\'s child exit; no exit code invented.'))
        training.verify_records([*frozen, *upstream_records])
        training.require_idle_gpu(request['gpu'])
        tests = output / 'gpu_validation'
        tests.mkdir()
        training.publish(output / 'state.json', dict(format=FORMAT, phase='gpu_validation', completed=[]), exclusive=False)
        test_file = tests / 'gtest.json'
        test_run = common.causal.execute_recorded(
            [binaries['gpu_test']['path'], f'--gtest_output=json:{test_file}'],
            [item['path'] for item in frozen], tests,
            output / 'gpu_test.log', output / 'gpu_test_execution.json')
        require_native_tests(test_file)
        for case in document['cases']:
            training.verify_records([*frozen, *upstream_records])
            training.require_idle_gpu(request['gpu'])
            index = case['case_index']
            directory = output / f'case_{index:02d}'
            directory.mkdir()
            native = directory / 'native'
            execution_path = directory / 'execution.json'
            training.publish(output / 'state.json', dict(format=FORMAT,
                phase='historical_head_position', case_index=index, completed=completed), exclusive=False)
            command = command_for_case(binaries['probe']['path'], document, case, native)
            execution = execute_tree(command, [item['path'] for item in frozen], native,
                                     directory / 'probe.log', execution_path)
            report = readout.analyze_case(descriptor, index, native, execution_path)
            training.publish(directory / 'readout.json', report)
            training.verify_records([*frozen, *upstream_records, *execution['outputs'], execution['log']])
            item = dict(case_index=index, directory=str(native), execution=str(execution_path),
                        readout=training.record(directory / 'readout.json'))
            training.publish(directory / 'complete.json', item)
            completed.append(item)
            print(f'{training.now()} completed historical head-position case {index + 1}/{len(document["cases"])}: '
                  f'{case["word"]} block={case["block"]} head={case["head"]}', flush=True)
        aggregate = readout.analyze_all(descriptor, completed)
        training.publish(output / 'readout.json', aggregate)
        artifacts = []
        for item in completed:
            artifacts.extend(common.recursive_outputs(Path(item['directory']).parent))
        training.verify_records([*frozen, *upstream_records, *artifacts,
                                 *test_run['inputs_before'], *test_run['outputs'], test_run['log']])
        result = dict(format=FORMAT, stage='historical_head_position', complete=True,
            completed_utc=training.now(), historical_only=True, new_paired_model_result=False,
            full_word_probability_claimed=False, goal_completion_claimed=False,
            cases=descriptor_record, frozen_inputs=frozen, gpu_test=test_run,
            upstream=training.record(upstream_record), completed=completed,
            completed_artifact_records=artifacts, readout=training.record(output / 'readout.json'),
            limitations=['Three historical first-piece events, not complete-word or new paired-model results.',
                         'Positions interact nonlinearly; effect sizes are not fractions of unique word storage.',
                         'Combine these results with paired weight transfers and unrelated context controls.'])
        training.publish(output / 'summary.json', result)
        training.publish(output / 'state.json', dict(format=FORMAT, phase='complete', completed=completed), exclusive=False)
        print(f'{training.now()} historical head-position screen complete: {output / "readout.json"}', flush=True)
        return result
    except BaseException as error:
        training.publish(output / 'failure.json', dict(type=type(error).__name__, error=str(error),
            failed_utc=training.now(), completed=completed, training_restarted=False,
            no_automatic_restart=True, goal_completion_claimed=False))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'source-output', 'output', 'cases', 'probe', 'gpu-test'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('observer-pid', 'observer-start-ticks'):
        parser.add_argument('--' + name, type=int, required=True)
    args = parser.parse_args(argv)

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'head-position observer received signal {signum}')

    signal.signal(signal.SIGTERM, interrupted)
    run(args)


if __name__ == '__main__':
    main()
