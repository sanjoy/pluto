"""Queue the predeclared attention/MLP screen after embedding interventions.

This is an analysis observer, not a training supervisor. It never restarts or
changes training. The observed embedding process must actually exit before any
GPU child starts, and completed training/embedding artifacts are independently
validated. Every planned whole-branch and output-write-only donor transfer is
then run sequentially at matched steps in both directions, with independent
checkpoint copies and both frozen case suites. Completing this stage is NOT a
claim that the mechanistic research goal has been achieved.
"""

import argparse
import importlib
import json
import os
from pathlib import Path
import signal
import sys
import time

import numpy as np

from . import checkpoint, paired_analysis, paired_causal_followup as causal
from . import paired_intervention_plan, paired_training, paired_weight_patch
from . import paired_word_cases, paired_supplemental_cases


FORMAT = 'pluto-paired-branch-followup-v1'
record = causal.record
read_json = causal.read_json
write_json = causal.write_json
verify_records = causal.verify_records
execute_recorded = causal.execute_recorded


def wait_for_embedding(summary_path, expected, *, poll_seconds=30):
    """Wait on one exact process identity, never infer exit from a timeout.

    Completed-artifact mode (expected=None) is only for an already-published
    summary. Both modes still require the independent downstream validation.
    Exit observations are not claimed to expose another parent's exit code.
    """
    if not 0 < poll_seconds <= 60:
        raise ValueError('poll interval must be in (0, 60]')
    summary_path = Path(summary_path)
    missing, errors = 0, 0
    while True:
        if (summary_path.parent / 'failure.json').exists():
            raise RuntimeError('upstream embedding analysis failed; no GPU work started')
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
                errors += 1
                if errors >= 4:
                    raise RuntimeError('cannot verify embedding observer; no GPU work started') from error
                time.sleep(poll_seconds)
                continue
            if live is not None and not causal.process_matches(live, expected):
                raise RuntimeError('embedding observer identity changed; refusing to follow a restart')
        if live is None:
            missing += 1
            if summary_path.is_file() and (expected is None or missing >= 2):
                summary = read_json(summary_path)
                if (summary.get('format') != 'pluto-paired-causal-followup-v1'
                        or summary.get('complete') is not True or summary.get('stage') != 'embedding'
                        or summary.get('goal_completion_claimed') is not False):
                    raise ValueError('upstream summary is not a completed embedding stage')
                return summary
            if expected is None or missing >= 4:
                raise RuntimeError('embedding observer is missing without a completed summary')
        else:
            missing = 0
        time.sleep(poll_seconds)


def _flag(argv, flag):
    values = [arg[len(flag) + 1:] for arg in argv if arg.startswith(flag + '=')]
    values += [argv[i + 1] for i, arg in enumerate(argv[:-1]) if arg == flag]
    if len(values) != 1:
        raise ValueError(f'observed command needs exactly one {flag}')
    return values[0]


def observer_identity(pid, start_ticks, root, embedding_output):
    actual = paired_analysis._process_identity(pid)
    argv = actual['argv']
    if (actual['start_ticks'] != start_ticks or argv.count('-m') != 1
            or argv.index('-m') + 1 >= len(argv)
            or argv[argv.index('-m') + 1] != 'weight_analysis.paired_causal_followup'
            or Path(_flag(argv, '--root')).resolve() != root
            or Path(_flag(argv, '--output')).resolve() != embedding_output):
        raise ValueError('observer identity is not the requested embedding analysis')
    return actual


def _readout_module():
    # Eagerly called before source snapshots and before waiting, not after a
    # multi-hour upstream job where the working tree could have changed.
    return importlib.import_module('weight_analysis.paired_branch_readout')


def _source_paths():
    # Capture every project analysis module loaded transitively, not only the
    # imports written in this file. Standard-library/NumPy code is identified by
    # runtime version rather than copying the entire Python installation.
    paths = {Path(__file__).resolve()}
    for name, module in tuple(sys.modules.items()):
        path = getattr(module, '__file__', None)
        if name.startswith('weight_analysis.') and path and str(path).endswith('.py'):
            paths.add(Path(path).resolve())
    return sorted(paths)


def _validate_execution(path):
    execution = read_json(path)
    if (execution.get('format') != 'pluto-paired-probe-execution-v1'
            or execution.get('returncode') != 0
            or not execution.get('inputs_before') or not execution.get('outputs')
            or execution['inputs_before'] != execution.get('inputs_after')):
        raise ValueError('baseline execution lacks successful unchanged-input provenance')
    verify_records([*execution['inputs_before'], *execution['outputs'], execution['log']])
    return execution


def _validate_upstream(root, embedding_output, summary):
    """Validate all upstream completion artifacts before constructing GPU work."""
    manifest = read_json(root / 'manifest.json')
    paired_training.verify_frozen_inputs(manifest)
    state = read_json(root / 'state.json')
    completion = paired_analysis.validate_completion(root, state, manifest, all_matched_steps=True)
    paired_analysis.validate_determinism_gate(root, state, manifest, completion['initial'])
    if (summary.get('format') != 'pluto-paired-causal-followup-v1'
            or summary.get('complete') is not True or summary.get('stage') != 'embedding'
            or summary.get('goal_completion_claimed') is not False):
        raise ValueError('embedding summary is incomplete')
    verify_records([*summary['frozen_inputs'], *summary['completed_artifact_records'], summary['plan']])
    plan = read_json(summary['plan']['path'])
    if (plan.get('format') != paired_intervention_plan.FORMAT or plan.get('complete') is not True
            or plan.get('patches_materialized') is not False):
        raise ValueError('embedding stage did not use a completed immutable intervention plan')
    verify_records([plan['analysis_summary'], *plan['frozen_inputs']])
    if Path(plan['analysis_summary']['path']) != root / 'analysis_trajectory' / 'summary.json':
        raise ValueError('intervention plan points outside the current trajectory')
    trajectory = read_json(plan['analysis_summary']['path'])
    if (trajectory.get('format') != 'pluto-paired-analysis-v1' or trajectory.get('complete') is not True
            or trajectory['plan'] != completion):
        raise ValueError('trajectory no longer matches completed training')
    pairs, _ = paired_intervention_plan._selected_pairs(trajectory)
    specs = checkpoint.tensor_manifest()
    declared = [item for pair in pairs for item in paired_intervention_plan._interventions(
        pair, specs, plan['word_piece_ids'])]
    # TensorSpec uses tuples in Python; immutable plan JSON canonically uses
    # lists. Compare serialized schema values, not Python container types.
    declared = json.loads(json.dumps(declared))
    if plan['interventions'] != declared:
        raise ValueError('intervention plan differs from the predeclared matched-step screen')
    selected = [item for item in declared if item['tensors']]
    if summary['deferred_interventions'] != selected:
        raise ValueError('embedding stage deferred a different branch screen')
    expected_embedding = {item['name']: item for item in declared if not item['tensors']}
    stages = summary['completed']
    if (len(stages) != len(expected_embedding)
            or {item['intervention']['name'] for item in stages} != set(expected_embedding)):
        raise ValueError('embedding stage did not complete every predeclared control/row transfer')
    artifact_records = summary['completed_artifact_records']
    gpu_test = summary['gpu_test']
    if (gpu_test.get('format') != 'pluto-paired-probe-execution-v1'
            or gpu_test.get('returncode') != 0 or not gpu_test.get('outputs')):
        raise ValueError('embedding stage lacks native GPU validation')
    verify_records([*gpu_test['outputs'], *gpu_test['inputs_before'], gpu_test['log']])
    if gpu_test['inputs_before'] != gpu_test.get('inputs_after'):
        raise ValueError('GPU validation executable changed during execution')
    test_files = [item['path'] for item in gpu_test['outputs'] if Path(item['path']).name == 'gtest.json']
    if len(test_files) != 1:
        raise ValueError('missing native test completion output')
    causal.require_gpu_test_success(test_files[0])
    baselines = {}
    for evidence in stages:
        item = evidence['intervention']
        if item != expected_embedding[item['name']]:
            raise ValueError('completed embedding intervention disagrees with plan')
        stage = embedding_output / item['name']
        complete_record = record(stage / 'complete.json')
        if complete_record not in artifact_records or read_json(complete_record['path']) != evidence:
            raise ValueError('embedding per-stage completion marker disagrees with summary')
        verify_records(evidence['artifacts'])
        if item['embedding_rows']:
            continue
        if evidence.get('copy_control_all_scores_byte_equal') is not True:
            raise ValueError('copy control was not verified against trajectory scores')
        recipient = item['recipient_checkpoint']
        reference = trajectory['behavior'][recipient['path']]['report']
        verify_records([reference, evidence['supplemental'], evidence['patch']])
        reference_scores = read_json(reference['path'])['scores']
        for key, filename in (('losses', 'losses.f32.bin'), ('argmax', 'argmax.i32.bin')):
            verify_records([reference_scores[key]])
            if checkpoint.sha256_file(stage / 'copy_scores' / filename) != reference_scores[key]['sha256']:
                raise ValueError('copy baseline no longer reproduces full trajectory scores')
        executions = {'main': stage / 'copy_execution.json',
                      'supplemental': stage / 'supplemental_execution.json'}
        for path in executions.values():
            _validate_execution(path)
            if record(path) not in artifact_records:
                raise ValueError('baseline execution is not a completed embedding artifact')
        if recipient['path'] in baselines:
            raise ValueError('duplicate copy baseline for a matched checkpoint')
        baselines[recipient['path']] = {
            'scores': {'main': stage / 'copy_scores', 'supplemental': stage / 'supplemental_scores'},
            'executions': executions, 'completion': complete_record}
    for item in selected:
        for role in ('recipient_checkpoint', 'donor_checkpoint'):
            if item[role]['path'] not in baselines:
                raise ValueError('missing copy baseline for selected branch intervention')
    return plan, selected, baselines


def run(args):
    readout = _readout_module()
    root, embedding_output = Path(args.root).resolve(), Path(args.embedding_output).resolve()
    output = Path(args.output).resolve()
    if output.exists() or Path(args.output).is_symlink():
        raise FileExistsError(output)
    if output == root or output in root.parents or output == embedding_output or output in embedding_output.parents:
        raise ValueError('output must not replace an experiment or upstream directory')
    protected = [embedding_output, *(root / name for name in (
        'initial', 'original', 'replacement', 'word_cases', 'supplemental_cases', 'analysis_trajectory',
        'bin', 'inputs', 'source', 'tokenizer', 'control_a', 'control_b', 'control_replacement'))]
    if any(path == output or path in output.parents for path in protected):
        raise ValueError('output overlaps preserved experiment inputs')
    summary_path = embedding_output / 'summary.json'
    if (args.observer_pid is None) != (args.observer_start_ticks is None):
        raise ValueError('observer PID and start ticks must be supplied together')
    expected = (observer_identity(args.observer_pid, args.observer_start_ticks, root, embedding_output)
                if args.observer_pid is not None else None)
    if expected is None and not summary_path.is_file():
        raise ValueError('need a live specific embedding observer or completed summary')
    upstream_request = record(embedding_output / 'request.json')
    request = read_json(upstream_request['path'])
    if (request.get('stage') != 'embedding' or Path(request.get('root', '')).resolve() != root
            or expected is not None and request.get('runner_pid') != expected['pid']):
        raise ValueError('embedding request identity/root disagrees with requested observer')
    main_cases, supplemental_cases = (root / name / 'cases.json'
                                       for name in ('word_cases', 'supplemental_cases'))
    cases = {'main': read_json(main_cases), 'supplemental': read_json(supplemental_cases)}
    case_paths = {'main': main_cases, 'supplemental': supplemental_cases}
    paired_training.verify_frozen_inputs(read_json(root / 'manifest.json'))
    original_sources = [record(path) for path in _source_paths()]
    output.mkdir()
    (output / 'bin').mkdir()
    (output / 'source').mkdir()
    binary = paired_training.freeze_file(args.loss_probe, output / 'bin' / 'loss_probe', executable=True)
    sources = [paired_training.freeze_file(item['path'], output / 'source' / Path(item['path']).name)
               for item in original_sources]
    frozen = [record(root / 'manifest.json'), upstream_request, binary, *sources, *original_sources,
              *(record(path) for path in case_paths.values()),
              *(record(plan['packed_batch']['path']) for plan in cases.values())]
    verify_records(frozen)
    write_json(output / 'request.json', {'format': 'pluto-paired-branch-followup-request-v1',
        'created_utc': causal.now(), 'runner_pid': os.getpid(), 'root': str(root),
        'embedding_output': str(embedding_output), 'observer_identity': expected,
        'frozen_inputs': frozen, 'python_version': sys.version, 'numpy_version': np.__version__,
        'stage': 'branches', 'goal_completion_claimed': False})
    completed, completed_records = [], []
    try:
        print(f'{causal.now()} waiting for verified embedding observer; no GPU work yet', flush=True)
        summary = wait_for_embedding(summary_path, expected)
        verify_records(frozen)
        plan, selected, baselines = _validate_upstream(root, embedding_output, summary)
        upstream_records = [record(summary_path), summary['plan'], plan['analysis_summary'],
                            *summary['completed_artifact_records']]
        write_json(output / 'validated_upstream.json', {'embedding_summary': upstream_records[0],
            'intervention_plan': summary['plan'], 'trajectory_summary': plan['analysis_summary'],
            'observer_identity': expected, 'observer_exit_code': None,
            'observer_exit_code_note': 'Not our child; completion is verified by exit identity and artifacts.',
            'selected_interventions': selected,
            'baseline_completion_records': [value['completion'] for value in baselines.values()],
            'goal_completion_claimed': False})
        for index, item in enumerate(selected):
            verify_records([*frozen, *upstream_records[:3]])
            name = item['name']
            if Path(name).name != name or name in ('', '.', '..'):
                raise ValueError('unsafe intervention directory name')
            recipient, donor = (item[role]['path'] for role in ('recipient_checkpoint', 'donor_checkpoint'))
            paired_analysis._verify_checkpoint(item['recipient_checkpoint'])
            paired_analysis._verify_checkpoint(item['donor_checkpoint'])
            stage = output / name
            stage.mkdir()
            # paired_loss_probe validates the native checkpoint step name.
            # Keep each intervention's independent copy under its own stage.
            patch_dir = stage / f'step_{item["step"]}'
            paired_weight_patch.create_patch(recipient, donor, patch_dir,
                tensors=item['tensors'], embedding_rows=item['embedding_rows'])
            verify_records(original_sources)
            scores = {role: baselines[path]['scores'] for role, path in
                      (('recipient', recipient), ('donor', donor))}
            executions = {role: baselines[path]['executions'] for role, path in
                          (('recipient', recipient), ('donor', donor))}
            scores['patched'], executions['patched'] = {}, {}
            for suite in ('main', 'supplemental'):
                verify_records([*frozen, *upstream_records[:3]])
                scores['patched'][suite] = stage / f'{suite}_scores'
                executions['patched'][suite] = stage / f'{suite}_execution.json'
                batch = cases[suite]['packed_batch']['path']
                command = [binary['path'], f'--checkpoint={patch_dir}', f'--batch={batch}',
                           f'--output_dir={scores["patched"][suite]}', '--batch_sequences=1']
                execute_recorded(command, [binary['path'], case_paths[suite], batch,
                    *causal.checkpoint_inputs(patch_dir)], scores['patched'][suite],
                    stage / f'{suite}.log', executions['patched'][suite])
            verify_records(original_sources)
            report_path = stage / 'readout.json'
            readout.analyze(item, patch_dir / 'patch.json', main_cases, supplemental_cases,
                            scores, report_path, executions=executions)
            verify_records(original_sources)
            if any(path.is_symlink() for path in stage.rglob('*')):
                raise ValueError('unexpected symlink in branch evidence')
            artifacts = [record(path) for path in sorted(stage.rglob('*')) if path.is_file()]
            evidence = {'intervention': item, 'patch': record(patch_dir / 'patch.json'),
                        'readout': record(report_path), 'artifacts': artifacts,
                        'baseline_completion_records': [baselines[path]['completion'] for path in (recipient, donor)],
                        'goal_completion_claimed': False}
            write_json(stage / 'complete.json', evidence)
            completed.append(evidence)
            completed_records.extend([*artifacts, record(stage / 'complete.json')])
            print(f'{causal.now()} completed {index + 1}/{len(selected)} branch interventions: {name}', flush=True)
        verify_records([*frozen, *upstream_records, *completed_records])
        result = {'format': FORMAT, 'complete': True, 'stage': 'branches',
                  'goal_completion_claimed': False, 'completed_utc': causal.now(),
                  'embedding_summary': upstream_records[0], 'plan': summary['plan'],
                  'frozen_inputs': frozen, 'completed': completed,
                  'completed_artifact_records': completed_records,
                  'remaining_research': ['Integrate branch effects with checkpoint trajectories and embedding factorials.',
                      'Trace and causally confirm computations producing each word piece; rule out broad damage.',
                      'Report controls, limitations, and unresolved questions; branch ranking is not a storage explanation.']}
        write_json(output / 'summary.json', result)
        return result
    except BaseException as error:
        write_json(output / 'failure.json', {'type': type(error).__name__, 'error': str(error),
            'failed_utc': causal.now(), 'completed': completed, 'training_restarted': False,
            'goal_completion_claimed': False})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'embedding-output', 'output', 'loss-probe'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--observer-pid', type=int)
    parser.add_argument('--observer-start-ticks', type=int)
    args = parser.parse_args(argv)
    if (args.observer_pid is None) != (args.observer_start_ticks is None):
        parser.error('observer PID and start ticks must be supplied together')

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'branch follow-up received signal {signum}')

    signal.signal(signal.SIGTERM, interrupted)
    run(args)


if __name__ == '__main__':
    main()
