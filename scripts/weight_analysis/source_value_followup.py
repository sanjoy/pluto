"""Run the historical source-value assay only after the existing GPU queue.

This observer never supervises or restarts training. It follows one exact
branch-analysis process through confirmed exit, validates its completed
artifacts and the paired-training completion gate, then runs the new native
GPU test and the predeclared historical source assay. Inputs, implementations,
commands, outputs, and independent readouts are preserved in a new directory.
Completing this assay is NOT completion of the Exeunt-encoding research goal.
"""

import argparse
import importlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np

from . import checkpoint, paired_analysis, paired_training
from . import paired_causal_followup as causal
from . import paired_branch_followup as branch

FORMAT = 'pluto-source-value-followup-v1'
record, read_json = causal.record, causal.read_json
write_json, verify_records = causal.write_json, causal.verify_records


def wait_for_branches(summary_path, expected, *, poll_seconds=30):
    """Wait for actual exit, not merely publication of a completion marker."""
    if not 0 < poll_seconds <= 60:
        raise ValueError('poll interval must be in (0, 60]')
    summary_path = Path(summary_path)
    missing = errors = 0
    while True:
        if (summary_path.parent / 'failure.json').exists():
            raise RuntimeError('upstream branch analysis failed; no GPU work started')
        live = None
        if expected is not None:
            try:
                try:
                    live = paired_analysis._process_identity(expected['pid'])
                except (FileNotFoundError, ProcessLookupError):
                    if causal.process_still_live(expected):
                        missing = 0
                        time.sleep(poll_seconds)
                        continue
                errors = 0
            except (OSError, ValueError) as error:
                missing = 0
                errors += 1
                if errors >= 4:
                    raise RuntimeError('cannot verify branch observer; no GPU work started') from error
                time.sleep(poll_seconds)
                continue
            if live is not None and not causal.process_matches(live, expected):
                raise RuntimeError('branch observer identity changed; refusing to follow a restart')
        if live is None:
            missing += 1
            if summary_path.is_file() and (expected is None or missing >= 2):
                result = read_json(summary_path)
                if (result.get('format') != 'pluto-paired-branch-followup-v1'
                        or result.get('complete') is not True
                        or result.get('stage') != 'branches'
                        or result.get('goal_completion_claimed') is not False):
                    raise ValueError('upstream summary is not a completed branch stage')
                return result
            if expected is None or missing >= 4:
                raise RuntimeError('branch observer missing without completed summary')
        else:
            missing = 0
        time.sleep(poll_seconds)


def observer_identity(pid, ticks, root, upstream):
    actual = paired_analysis._process_identity(pid)
    argv = actual['argv']
    if (actual['start_ticks'] != ticks or argv.count('-m') != 1
            or argv.index('-m') + 1 >= len(argv)
            or argv[argv.index('-m') + 1] != 'scripts.weight_analysis.paired_branch_followup'
            or Path(branch._flag(argv, '--root')).resolve() != root
            or Path(branch._flag(argv, '--output')).resolve() != upstream):
        raise ValueError('observer is not the requested branch analysis')
    return actual


def require_observer_absent(pid):
    """Completed-artifact mode cannot skip exit checks on a live publisher."""
    if type(pid) is not int or pid <= 0:
        raise ValueError('upstream request has no valid observer PID')
    try:
        paired_analysis._process_identity(pid)
    except (FileNotFoundError, ProcessLookupError):
        # A failed /proc read alone is not an exit observation. Signal zero
        # checks existence without delivering a signal or changing the job.
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
    raise ValueError('upstream observer still exists; provide its live identity')


def authenticate_assay(manifest_path, historical_root, historical_checkpoint):
    """Bind the fixed source choices to actual archived inputs and weights.

    The old producer executable may since have changed; it is deliberately
    NOT required to equal today's binary. Its stored arrays and checkpoint
    bytes are authenticated through the old manifest and hashed plan instead.
    """
    manifest = read_json(manifest_path)
    if manifest.get('complete') is not True:
        raise ValueError('historical manifest is incomplete')
    known = {entry['original_path']: entry for entry in manifest['files']}
    if len(known) != len(manifest['files']):
        raise ValueError('duplicate historical manifest input')
    used = [record(manifest_path)]

    def checked(path, expected=None):
        path = Path(path)
        if path.is_symlink() or not path.is_file():
            raise ValueError('historical input must be a regular nonsymlink file')
        value = record(path)
        expected = known[str(path)] if expected is None else expected
        if (value['sha256'], value['bytes']) != (expected['sha256'], expected['bytes']):
            raise ValueError('historical input differs from recorded evidence: ' + str(path))
        used.append(value)
        return value

    plan_record = checked(historical_root / 'plan.json')
    plan = read_json(plan_record['path'])
    if (Path(plan['checkpoint_directory']).resolve() != historical_checkpoint
            or (plan['vocab_size'], plan['padded_vocab_size'], plan['n_layers'], plan['n_heads'])
            != (50257, 50272, 8, 8)):
        raise ValueError('historical model/checkpoint disagrees with assay')
    model = checkpoint.GPT2Checkpoint(historical_checkpoint, check_finite=True)
    if historical_checkpoint.is_symlink() or {
            p.name for p in historical_checkpoint.iterdir()} != {
                spec.filename for spec in model.manifest}:
        raise ValueError('historical checkpoint must contain exactly 100 canonical weight files')
    plan_inputs = {item['path']: item for item in plan['inputs']}
    for spec in model.manifest:
        path = historical_checkpoint / spec.filename
        checked(path, plan_inputs[str(path)])
    native = historical_root / 'trace_step_1342'
    meta_record = checked(native / 'metadata.json')
    meta = read_json(meta_record['path'])
    assay = {
        'block': 1, 'head': 2, 'query': 1023, 'target_id': 2797,
        'sources': [1022, 889, 1023, 1021, 512],
        'context_length': 1024, 'width': 512, 'head_dim': 64,
        'vocabulary': 50257, 'padded_vocabulary': 50272,
        'historical_checkpoint': str(historical_checkpoint),
        'prefix': checked(historical_root / 'prefix_step_1342.i32'),
        'expected_logits': checked(native / 'logits.f32'),
        'expected_qkv': checked(native / 'blocks.1.qkv.bf16'),
        'expected_context': checked(native / 'blocks.1.attention.bf16'),
    }
    ids = np.fromfile(assay['prefix']['path'], dtype='<i4')
    if (len(ids) != 1024 or ids.tolist() != meta['token_ids']
            or [int(ids[i]) for i in assay['sources']] != [1475, 508, 68, 220, 40802]
            or meta.get('complete') is not True or meta.get('selected_row') != 1023
            or meta.get('target_id') != 2797
            or meta.get('checkpoint_directory') != str(historical_checkpoint)):
        raise ValueError('historical token event does not match predeclared source controls')
    if (assay['expected_logits']['bytes'] != 50272 * 4
            or assay['expected_qkv']['bytes'] != 1024 * 1536 * 2
            or assay['expected_context']['bytes'] != 1024 * 512 * 2):
        raise ValueError('historical native tensor geometry changed')
    verify_records(used)
    return assay, used


def validate_upstream(root, upstream, summary):
    """Prove queue completion; this is not a new replication of branch effects."""
    manifest = read_json(root / 'manifest.json')
    paired_training.verify_frozen_inputs(manifest)
    state = read_json(root / 'state.json')
    completion = paired_analysis.validate_completion(root, state, manifest, all_matched_steps=True)
    paired_analysis.validate_determinism_gate(root, state, manifest, completion['initial'])
    request = read_json(upstream / 'request.json')
    if (summary.get('format') != 'pluto-paired-branch-followup-v1'
            or summary.get('complete') is not True or summary.get('stage') != 'branches'
            or summary.get('goal_completion_claimed') is not False
            or summary.get('frozen_inputs') != request['frozen_inputs']):
        raise ValueError('branch completion disagrees with its frozen request')
    inputs = [*summary['frozen_inputs'], *summary['completed_artifact_records'],
              summary['embedding_summary'], summary['plan'], record(upstream / 'summary.json')]
    if not summary['completed_artifact_records']:
        raise ValueError('branch summary has no completed evidence')
    verify_records(inputs)
    plan = read_json(summary['plan']['path'])
    if (plan.get('format') != branch.paired_intervention_plan.FORMAT
            or plan.get('complete') is not True or plan.get('patches_materialized') is not False):
        raise ValueError('branch plan is not the completed immutable screen')
    embedding_path = Path(request['embedding_output']) / 'summary.json'
    embedding = read_json(embedding_path)
    if (summary['embedding_summary'] != record(embedding_path)
            or embedding.get('format') != 'pluto-paired-causal-followup-v1'
            or embedding.get('complete') is not True
            or embedding.get('stage') != 'embedding'
            or embedding.get('goal_completion_claimed') is not False
            or embedding.get('plan') != summary['plan']):
        raise ValueError('branch completion refers to a different embedding stage')
    verify_records([plan['analysis_summary'], *plan['frozen_inputs']])
    if Path(plan['analysis_summary']['path']) != root / 'analysis_trajectory' / 'summary.json':
        raise ValueError('branch plan points to a different trajectory')
    trajectory = read_json(plan['analysis_summary']['path'])
    if (trajectory.get('format') != 'pluto-paired-analysis-v1'
            or trajectory.get('complete') is not True or trajectory['plan'] != completion):
        raise ValueError('branch trajectory does not match completed training')
    pairs, _ = branch.paired_intervention_plan._selected_pairs(trajectory)
    declared = [item for pair in pairs
                for item in branch.paired_intervention_plan._interventions(
                    pair, checkpoint.tensor_manifest(), plan['word_piece_ids'])]
    if plan['interventions'] != json.loads(json.dumps(declared)):
        raise ValueError('branch plan differs from predeclared matched-step interventions')
    expected = {item['name']: item for item in plan['interventions'] if item['tensors']}
    actual = [item['intervention'] for item in summary['completed']]
    if (not expected or len(actual) != len(expected)
            or {item['name'] for item in actual} != set(expected)
            or any(item != expected[item['name']] for item in actual)):
        raise ValueError('branch stage did not finish the entire predeclared screen')
    for item in summary['completed']:
        path = upstream / item['intervention']['name'] / 'complete.json'
        if record(path) not in summary['completed_artifact_records'] or read_json(path) != item:
            raise ValueError('branch per-intervention evidence disagrees with summary')
        verify_records(item['artifacts'])
    return [record(upstream / 'summary.json'), summary['plan'],
            plan['analysis_summary'], record(root / 'state.json')]


def recursive_outputs(directory):
    directory = Path(directory)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError('missing native output directory')
    entries = sorted(directory.rglob('*'))
    if any(path.is_symlink() or not (path.is_file() or path.is_dir()) for path in entries):
        raise ValueError('native evidence contains a symlink or special file')
    files = [record(path) for path in entries if path.is_file()]
    if not files:
        raise ValueError('native probe produced no evidence')
    return files


def execute_tree(command, inputs, directory, log, execution_path):
    paths = sorted({str(Path(path).resolve()) for path in inputs})
    before = [record(path) for path in paths]
    result = causal.execute(command, log)
    after = [record(path) for path in paths]
    if before != after:
        raise ValueError('source probe inputs changed during execution')
    result.update(format='pluto-source-value-execution-v1', inputs_before=before,
                  inputs_after=after, outputs=recursive_outputs(directory))
    write_json(execution_path, result)
    return result


def gpu_identity():
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'],
        check=True, text=True, capture_output=True, timeout=15)
    values = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if len(values) != 1 or not values[0].startswith('GPU-'):
        raise ValueError('source assay requires the verified single-GPU machine')
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    if visible not in (None, '', '0', values[0]):
        raise ValueError('CUDA visibility does not match the recorded GPU')
    if visible == '':
        raise ValueError('CUDA is disabled in this observer environment')
    return values[0]


def require_idle_gpu(expected_uuid):
    if gpu_identity() != expected_uuid:
        raise RuntimeError('GPU identity changed while queued')
    result = subprocess.run(
        ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
        check=True, text=True, capture_output=True, timeout=15)
    if result.stdout.strip():
        raise RuntimeError('GPU has another compute process; no new CUDA work started')


def source_paths():
    repo = Path(__file__).resolve().parents[2]
    paths = {Path(__file__).resolve()}
    for name, module in tuple(sys.modules.items()):
        path = getattr(module, '__file__', None)
        if name.startswith('scripts.weight_analysis.') and path and path.endswith('.py'):
            paths.add(Path(path).resolve())
    paths.update(repo / name for name in checkpoint.SOURCE_FILES)
    paths.update(repo / 'scripts' / 'weight_analysis' / name for name in (
        'source_value_probe.h', 'source_value_probe.cc', 'source_value_probe_main.cc',
        'source_value_probe_test.cc', 'source_value_probe_gpu_test.cc', 'source_value_probe.md',
        'phrase_probe.h', 'phrase_probe_lib.cc', 'causal_probe.h', 'causal_probe.cc',
        'embedding_factorial_probe.h', 'embedding_factorial_probe.cc',
        'token_trace_probe.h', 'token_trace_probe_lib.cc', 'BUILD.bazel'))
    return sorted(paths)


def run(args):
    readout = importlib.import_module('scripts.weight_analysis.source_value_readout')
    root, upstream = Path(args.root).resolve(), Path(args.branch_output).resolve()
    supplied = Path(args.output)
    output = supplied.resolve()
    if supplied.is_symlink() or output.exists():
        raise FileExistsError(output)
    # Require a new immediate child of this experiment. This excludes all
    # input subtrees without guessing which future subdirectories are safe.
    if output.parent != root or output == upstream:
        raise ValueError('source output must be a new direct experiment child')
    if (args.observer_pid is None) != (args.observer_start_ticks is None):
        raise ValueError('observer PID/start ticks must be supplied together')
    expected = (observer_identity(args.observer_pid, args.observer_start_ticks, root, upstream)
                if args.observer_pid is not None else None)
    summary_path = upstream / 'summary.json'
    if expected is None and not summary_path.is_file():
        raise ValueError('need a specific live branch observer or completed summary')
    upstream_record = record(upstream / 'request.json')
    upstream_request = read_json(upstream_record['path'])
    if (upstream_request.get('stage') != 'branches'
            or Path(upstream_request.get('root', '')).resolve() != root
            or expected is not None and upstream_request.get('runner_pid') != expected['pid']):
        raise ValueError('branch request disagrees with observer identity/root')
    if expected is None:
        require_observer_absent(upstream_request.get('runner_pid'))
    verify_records(upstream_request['frozen_inputs'])
    historical = Path(args.historical_checkpoint)
    if historical.is_symlink():
        raise ValueError('historical checkpoint must not be a symlink')
    assay, historical_inputs = authenticate_assay(
        Path(args.historical_manifest).resolve(), Path(args.historical_root).resolve(),
        historical.resolve())
    paired_training.verify_frozen_inputs(read_json(root / 'manifest.json'))
    gpu = gpu_identity()  # Read-only metadata query, never a CUDA context.
    originals = [record(path) for path in source_paths()]
    output.mkdir()
    (output / 'bin').mkdir()
    binaries = {name: paired_training.freeze_file(Path(getattr(args, name)).resolve(),
                    output / 'bin' / name, executable=True) for name in ('probe', 'gpu_test')}
    copies = []
    repo = Path(__file__).resolve().parents[2]
    for item in originals:
        path = Path(item['path'])
        destination = output / 'source' / path.relative_to(repo)
        destination.parent.mkdir(parents=True, exist_ok=True)
        copies.append(paired_training.freeze_file(path, destination))
    frozen = [record(root / 'manifest.json'), upstream_record, *binaries.values(),
              *historical_inputs, *originals, *copies]
    verify_records(frozen)
    request = {'format': 'pluto-source-value-followup-request-v1', 'created_utc': causal.now(),
        'runner_pid': os.getpid(), 'root': str(root), 'branch_output': str(upstream),
        'observer_identity': expected, 'gpu_uuid': gpu, 'assay': assay,
        'frozen_inputs': frozen, 'python_version': sys.version, 'numpy_version': np.__version__,
        'stage': 'historical_source_value', 'new_paired_model_result': False,
        'goal_completion_claimed': False}
    write_json(output / 'request.json', request)
    try:
        print(f'{causal.now()} waiting for verified branch observer; no GPU work yet', flush=True)
        summary = wait_for_branches(summary_path, expected)
        verify_records(frozen)
        upstream_inputs = validate_upstream(root, upstream, summary)
        write_json(output / 'validated_upstream.json', {
            'inputs': upstream_inputs, 'observer_identity': expected,
            'observer_exit_code': None,
            'note': 'Not our child; actual exit and completed artifacts checked, no invented exit code.'})
        require_idle_gpu(gpu)
        tests = output / 'gpu_validation'
        tests.mkdir()
        test_file = tests / 'gtest.json'
        test_run = causal.execute_recorded(
            [binaries['gpu_test']['path'], f'--gtest_output=json:{test_file}'],
            [item['path'] for item in frozen], tests, output / 'gpu_test.log',
            output / 'gpu_test_execution.json')
        causal.require_gpu_test_success(test_file)
        verify_records([*frozen, *upstream_inputs])
        require_idle_gpu(gpu)
        native = output / 'native'
        command = [binaries['probe']['path'],
            f'--checkpoint={assay["historical_checkpoint"]}', f'--tokens_file={assay["prefix"]["path"]}',
            f'--output_dir={native}', f'--block={assay["block"]}', f'--head={assay["head"]}',
            f'--query={assay["query"]}', f'--sources={",".join(map(str, assay["sources"]))}',
            f'--target_id={assay["target_id"]}', f'--expected_logits={assay["expected_logits"]["path"]}']
        execution_path = output / 'probe_execution.json'
        execution = execute_tree(command, [item['path'] for item in frozen],
            native, output / 'probe.log', execution_path)
        verify_records([*frozen, *upstream_inputs])
        report = readout.summarize(native, execution_path, assay)
        write_json(output / 'readout.json', report)
        verify_records([*execution['outputs'], *frozen, *upstream_inputs])
        result = {'format': FORMAT, 'complete': True, 'completed_utc': causal.now(),
            'stage': 'historical_source_value', 'new_paired_model_result': False,
            'goal_completion_claimed': False, 'frozen_inputs': frozen, 'assay': assay,
            'gpu_test': test_run, 'execution': record(execution_path),
            'readout': record(output / 'readout.json'),
            'upstream': record(output / 'validated_upstream.json'),
            'remaining_research': [
                'Integrate actual source effects with whole-head ablations and paired weight transfers.',
                'Test complete multi-token spelling across contexts; this assay targets one unt prediction.',
                'Distinguish functional contributions from lexical exclusivity or a unique storage claim.']}
        write_json(output / 'summary.json', result)
        print(f'{causal.now()} source-value assay complete; readout={output / "readout.json"}', flush=True)
        return result
    except BaseException as error:
        write_json(output / 'failure.json', {'failed_utc': causal.now(),
            'type': type(error).__name__, 'error': str(error),
            'training_restarted': False, 'goal_completion_claimed': False})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'branch-output', 'output', 'historical-manifest', 'historical-root',
                 'historical-checkpoint', 'probe', 'gpu-test'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--observer-pid', type=int)
    parser.add_argument('--observer-start-ticks', type=int)
    args = parser.parse_args(argv)
    if (args.observer_pid is None) != (args.observer_start_ticks is None):
        parser.error('observer PID and start ticks must be supplied together')
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'source-value observer received signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    run(args)


if __name__ == '__main__':
    main()
