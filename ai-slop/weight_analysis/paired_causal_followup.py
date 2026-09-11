"""Queue bounded embedding interventions after the paired trajectory scorer.

This is a follow-up analysis job, NEVER a training supervisor. It waits for a
specific already-running scorer, verifies its process identity, and refuses a
failed or missing upstream job. No CUDA test or probe starts until that scorer
has exited and a complete, validated trajectory is present. Its exit code is
not observable by this non-parent process; completion is established from the
independently verified artifacts, not an invented return code.
Training is not restarted, interrupted, extended, or modified here.

The first causal stage tests exact embedding-row transfers in both directions
at the first and last matched positive checkpoints. Copy controls, native GPU
unit tests, full-vocabulary factorial scores, following-token/shared-piece
scores, and before/after byte hashes are retained. Branch interventions remain
explicitly deferred work; completing this stage does not complete the research
goal. All outputs are new, exclusive files/directories.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from . import checkpoint, paired_analysis, paired_training, paired_weight_patch
from . import paired_word_cases, paired_supplemental_cases


def now():
    return datetime.now(timezone.utc).isoformat()


def record(path):
    return paired_word_cases._record(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    paired_training.write_json(path, value, exclusive=True)


def verify_records(records):
    for item in records:
        if record(item['path']) != item:
            raise ValueError(f'frozen evidence changed: {item["path"]}')


def process_matches(actual, expected):
    """PID equality is insufficient: argv and kernel start ticks must match."""
    return all(actual[key] == expected[key] for key in ('pid', 'start_ticks', 'argv'))


def process_still_live(expected):
    """Cross-check a failed identity read with signal-0 and a fresh /proc stat."""
    try:
        os.kill(expected['pid'], 0)  # Existence check, not a delivered signal.
    except ProcessLookupError:
        return False
    try:
        raw = (Path('/proc') / str(expected['pid']) / 'stat').read_text()
    except FileNotFoundError:
        return False
    fields = raw.rsplit(')', 1)[1].split()
    if int(fields[19]) != expected['start_ticks']:
        raise RuntimeError('upstream PID was reused during exit verification')
    return fields[0] not in ('Z', 'X', 'x')


def wait_for_scorer(summary_path, expected, *, poll_seconds=30):
    """Require two independently confirmed exit observations for a known scorer.

    Without a supplied live identity, this accepts an already-published artifact;
    the caller must still independently validate all completed-run evidence.
    Neither mode claims to observe another parent's child exit code.
    """
    if not 0 < poll_seconds <= 60:
        raise ValueError('poll interval must be in (0, 60]')
    summary_path = Path(summary_path)
    missing, observation_errors = 0, 0
    while True:
        failure = summary_path.parent / 'failure.json'
        if failure.exists():
            raise RuntimeError(f'upstream analysis failed: {failure}')
        live = None
        if expected is not None:
            try:
                try:
                    live = paired_analysis._process_identity(expected['pid'])
                except (FileNotFoundError, ProcessLookupError):
                    if process_still_live(expected):
                        missing = 0
                        time.sleep(poll_seconds)
                        continue
                observation_errors = 0
            except (OSError, ValueError) as error:
                observation_errors += 1
                if observation_errors >= 4:
                    raise RuntimeError('cannot verify upstream process; no GPU work started') from error
                time.sleep(poll_seconds)
                continue
            if live is not None and not process_matches(live, expected):
                raise RuntimeError('upstream process identity changed; refusing to follow a restart')
        if live is None:
            missing += 1
            if summary_path.is_file() and (expected is None or missing >= 2):
                summary = read_json(summary_path)
                if (summary.get('format') != 'pluto-paired-analysis-v1'
                        or summary.get('complete') is not True):
                    raise ValueError('upstream summary is not complete paired analysis')
                return summary
            if expected is None or missing >= 4:
                raise RuntimeError('upstream scorer is missing without a completed summary')
        else:
            missing = 0
        # A temporarily unreadable handle is never grounds to restart a job.
        time.sleep(poll_seconds)


def execute(command, log_path):
    """Own, terminate on interruption, and reap exactly one analysis child."""
    started = now()
    with Path(log_path).open('xb') as log:
        child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        print(f'{now()} started analysis child pid={child.pid}: {json.dumps(list(command))}', flush=True)
        try:
            returncode = child.wait()
        except BaseException:
            try:
                child.terminate()
            except ProcessLookupError:
                pass
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
            raise
    if returncode:
        raise RuntimeError(f'analysis child exited {returncode}; inspect {log_path}')
    return {'command': list(command), 'pid': child.pid, 'returncode': returncode,
            'started_utc': started, 'finished_utc': now(), 'log': record(log_path)}


def execute_recorded(command, inputs, outputs_directory, log_path, record_path):
    """Bind execution to input/output bytes, not merely filenames or exit status."""
    paths = sorted({str(Path(path).resolve()) for path in inputs})
    before = [record(path) for path in paths]
    result = execute(command, log_path)
    after = [record(path) for path in paths]
    if before != after:
        raise ValueError('probe inputs changed during execution')
    output = Path(outputs_directory)
    files = sorted(output.iterdir())
    if not files or any(path.is_symlink() or not path.is_file() for path in files):
        raise ValueError('probe output must contain regular evidence files')
    result.update(format='pluto-paired-probe-execution-v1',
                  inputs_before=before, inputs_after=after,
                  outputs=[record(path) for path in files])
    write_json(record_path, result)
    return result


def checkpoint_inputs(directory):
    directory = Path(directory)
    result = [directory / f'weight_{i}.bin' for i in range(100)]
    if (directory / 'patch.json').exists():
        result.append(directory / 'patch.json')
    return result


def require_gpu_test_success(path):
    """An all-skipped/no-tests CUDA run is not validation."""
    result = read_json(path)
    if (type(result.get('tests')) is not int or result['tests'] < 1
            or result.get('failures') != 0 or result.get('errors') != 0
            or result.get('disabled') != 0):
        raise ValueError('native GPU test did not pass every enabled test')
    cases = [case for suite in result.get('testsuites', [])
             for case in suite.get('testsuite', [])]
    if (len(cases) != result['tests'] or
            any(case.get('status') != 'RUN' or case.get('result') != 'COMPLETED'
                or 'skipped' in case or 'failures' in case for case in cases)):
        raise ValueError('GPU tests were missing, skipped, or incomplete')


def run(args):
    # Import all analysis code before waiting. Later working-tree edits must not
    # silently alter a long-running job's implementation.
    from . import paired_intervention_plan, embedding_factorial_readout, paired_input_exposure

    root, output = Path(args.root).resolve(), Path(args.output).resolve()
    summary_path = root / 'analysis_trajectory' / 'summary.json'
    main_cases = root / 'word_cases' / 'cases.json'
    supplemental_cases = root / 'supplemental_cases' / 'cases.json'
    if output.exists() or Path(args.output).is_symlink():
        raise FileExistsError(output)
    if output == root or output in root.parents:
        raise ValueError('output must not be the experiment or its ancestor')
    for name in ('initial', 'original', 'replacement', 'word_cases', 'supplemental_cases',
                 'analysis_trajectory', 'bin', 'inputs', 'source', 'tokenizer',
                 'control_a', 'control_b', 'control_replacement'):
        forbidden = root / name
        if forbidden == output or forbidden in output.parents:
            raise ValueError('output overlaps preserved experiment inputs')
    expected = None
    if args.observer_pid is not None:
        expected = paired_analysis._process_identity(args.observer_pid)
        if (expected['start_ticks'] != args.observer_start_ticks
                or '-m' not in expected['argv']
                or 'weight_analysis.paired_analysis' not in expected['argv']
                or str(root) not in expected['argv']
                or str(summary_path.parent) not in expected['argv']):
            raise ValueError('observer identity is not the requested paired scorer')
    elif not summary_path.is_file():
        raise ValueError('need a live specific observer or completed summary')
    manifest = read_json(root / 'manifest.json')
    paired_training.verify_frozen_inputs(manifest)
    output.mkdir()
    (output / 'bin').mkdir()
    (output / 'source').mkdir()
    binaries = {name: paired_training.freeze_file(getattr(args, name), output / 'bin' / name,
                                                 executable=True)
                for name in ('loss_probe', 'factorial_probe', 'gpu_test')}
    modules = (checkpoint, paired_analysis, paired_training, paired_weight_patch,
               paired_word_cases, paired_supplemental_cases, paired_intervention_plan,
               embedding_factorial_readout, paired_input_exposure)
    original_sources = [record(module.__file__) for module in modules] + [record(__file__)]
    sources = [paired_training.freeze_file(module.__file__,
                output / 'source' / Path(module.__file__).name) for module in modules]
    sources.append(paired_training.freeze_file(__file__, output / 'source' / Path(__file__).name))
    frozen = [record(root / 'manifest.json'), record(main_cases), record(supplemental_cases),
              record(read_json(main_cases)['packed_batch']['path']),
              record(read_json(supplemental_cases)['packed_batch']['path']),
              *binaries.values(), *sources, *original_sources]
    verify_records(original_sources)
    write_json(output / 'request.json', {'format': 'pluto-paired-causal-followup-request-v1',
               'created_utc': now(), 'runner_pid': os.getpid(), 'stage': 'embedding', 'root': str(root),
               'observer_identity': expected, 'frozen_inputs': frozen,
               'goal_completion_claimed': False})
    completed, completed_records = [], []
    try:
        print(f'{now()} waiting for verified trajectory scorer; no GPU work yet', flush=True)
        summary = wait_for_scorer(summary_path, expected)
        verify_records(frozen)
        state = read_json(root / 'state.json')
        verified_plan = paired_analysis.validate_completion(root, state, manifest,
                                                             all_matched_steps=True)
        paired_analysis.validate_determinism_gate(root, state, manifest, verified_plan['initial'])
        plan = paired_intervention_plan.prepare(summary_path, main_cases, supplemental_cases,
                                                output / 'plan')
        plan_record = record(output / 'plan' / 'plan.json')
        upstream_record = plan['analysis_summary']
        export_records = [export[key] for export in plan['exports'].values()
                          for key in ('cases_json', 'packed_batch', 'selected_rows')]
        verify_records([upstream_record, *export_records])
        print(f'{now()} upstream complete; validating native factorial GPU test', flush=True)
        tests = output / 'gpu_validation'
        tests.mkdir()
        test_result = tests / 'gtest.json'
        test_run = execute_recorded(
            [binaries['gpu_test']['path'], f'--gtest_output=json:{test_result}'],
            [binaries['gpu_test']['path']], tests, output / 'gpu_test.log',
            output / 'gpu_test_execution.json')
        require_gpu_test_success(test_result)
        original_main = read_json(main_cases)
        original_supplemental = read_json(supplemental_cases)

        # Copy controls first. A checksum-equivalent checkpoint must reproduce
        # the upstream scorer's COMPLETE FP32 loss and argmax dumps, not just
        # the selected word rows or rounded aggregate losses.
        selected = [item for item in plan['interventions']
                    if not item['tensors'] and not item['embedding_rows']]
        selected += [item for item in plan['interventions'] if item['embedding_rows']]
        for item in selected:
            verify_records([*frozen, plan_record, upstream_record, *export_records])
            name = item['name']
            if Path(name).name != name or name in ('', '.', '..'):
                raise ValueError('invalid intervention directory name')
            stage = output / name
            stage.mkdir()
            recipient = item['recipient_checkpoint']['path']
            donor = item['donor_checkpoint']['path']
            paired_analysis._verify_checkpoint(item['recipient_checkpoint'])
            paired_analysis._verify_checkpoint(item['donor_checkpoint'])
            # The loss probe uses the production checkpoint inspector, which
            # parses a step_N basename before doing any CUDA work. Preserve the
            # recipient's optimizer-step label on both copy and row patches.
            # Their donor/selection identity remains in patch.json, not the name.
            patch_dir = stage / f'step_{item["recipient_checkpoint"]["step"]}'
            paired_weight_patch.create_patch(recipient, donor, patch_dir,
                tensors=item['tensors'], embedding_rows=item['embedding_rows'])
            verify_records(original_sources)
            evidence = {'intervention': item, 'patch': record(patch_dir / 'patch.json')}
            if not item['embedding_rows']:
                scores = stage / 'copy_scores'
                inputs = [binaries['loss_probe']['path'], main_cases,
                          original_main['packed_batch']['path'], *checkpoint_inputs(patch_dir)]
                command = [binaries['loss_probe']['path'], f'--checkpoint={patch_dir}',
                           f'--batch={original_main["packed_batch"]["path"]}',
                           f'--output_dir={scores}', '--batch_sequences=1']
                execute_recorded(command, inputs, scores, stage / 'copy_probe.log',
                                 stage / 'copy_execution.json')
                reference = summary['behavior'][recipient]
                report_record = reference['report']
                verify_records([report_record])
                reference_scores = read_json(report_record['path'])['scores']
                for key, filename in (('losses', 'losses.f32.bin'), ('argmax', 'argmax.i32.bin')):
                    verify_records([reference_scores[key]])
                    if checkpoint.sha256_file(scores / filename) != reference_scores[key]['sha256']:
                        raise ValueError(f'copy control changed native {key}')
                evidence['copy_control_all_scores_byte_equal'] = True
            else:
                evidence['factorial'] = {}
                for export_name, source_cases, case_kind in (
                        ('main', main_cases, None),
                        ('word_next_native', supplemental_cases, 'word_next_native')):
                    export = plan['exports'][export_name]
                    verify_records([export[key] for key in ('cases_json', 'packed_batch', 'selected_rows')])
                    batch = (original_main['packed_batch']['path'] if export_name == 'main'
                             else export['packed_batch']['path'])
                    rows = export['selected_rows']['path']
                    scores = stage / f'factorial_{export_name}'
                    execution_path = stage / f'factorial_{export_name}_execution.json'
                    command = [binaries['factorial_probe']['path'], f'--recipient={recipient}',
                               f'--patched={patch_dir}', f'--batch={batch}', f'--rows={rows}',
                               f'--rows_per_case={export["rows_per_case"]}', '--batch_sequences=1',
                               f'--output_dir={scores}']
                    execute_recorded(command, [binaries['factorial_probe']['path'], source_cases,
                        batch, rows, *checkpoint_inputs(recipient), *checkpoint_inputs(patch_dir)],
                        scores, stage / f'factorial_{export_name}.log', execution_path)
                    report = stage / f'factorial_{export_name}_readout.json'
                    verify_records(original_sources)
                    embedding_factorial_readout.analyze(source_cases, scores, patch_dir / 'patch.json',
                        report, expected_rows=item['embedding_rows'], case_kind=case_kind,
                        execution_record=execution_path)
                    verify_records(original_sources)
                    evidence['factorial'][export_name] = record(report)
            # Score copy controls as well as donor-row patches, so shared-piece
            # collateral changes have a measured unmodified baseline.
            scores = stage / 'supplemental_scores'
            batch = original_supplemental['packed_batch']['path']
            command = [binaries['loss_probe']['path'], f'--checkpoint={patch_dir}',
                       f'--batch={batch}', f'--output_dir={scores}', '--batch_sequences=1']
            execute_recorded(command, [binaries['loss_probe']['path'], supplemental_cases,
                batch, *checkpoint_inputs(patch_dir)], scores, stage / 'supplemental.log',
                stage / 'supplemental_execution.json')
            report = stage / 'supplemental_readout.json'
            verify_records(original_sources)
            paired_supplemental_cases.summarize(supplemental_cases, scores, report)
            verify_records(original_sources)
            evidence['supplemental'] = record(report)
            artifacts = sorted(path for path in stage.rglob('*') if path.is_file())
            if any(path.is_symlink() for path in stage.rglob('*')):
                raise ValueError('unexpected symlink in completed evidence')
            evidence['artifacts'] = [record(path) for path in artifacts]
            completed.append(evidence)
            write_json(stage / 'complete.json', evidence)
            completed_records.extend([*evidence['artifacts'], record(stage / 'complete.json')])
            print(f'{now()} completed {name}; {len(completed)}/{len(selected)} embedding-stage arms',
                  flush=True)
        verify_records([*frozen, plan_record, upstream_record, *export_records, *completed_records,
                        *test_run.get('outputs', [])])
        result = {'format': 'pluto-paired-causal-followup-v1', 'complete': True,
                  'stage': 'embedding', 'goal_completion_claimed': False,
                  'completed_utc': now(), 'plan': plan_record, 'completed': completed,
                  'gpu_test': test_run, 'frozen_inputs': frozen,
                  'completed_artifact_records': completed_records,
                  'deferred_interventions': [item for item in plan['interventions'] if item['tensors']],
                  'remaining_research': ['Upstream attention/MLP causal localization and confirmation.',
                    'Mechanistic account of the computations producing every word piece.',
                    'Controls, limitations, and integration with matched checkpoint trajectories.']}
        write_json(output / 'summary.json', result)
        return result
    except BaseException as error:
        write_json(output / 'failure.json', {'type': type(error).__name__, 'error': str(error),
                   'completed': completed, 'failed_utc': now(), 'training_restarted': False})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'output', 'loss-probe', 'factorial-probe', 'gpu-test'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--observer-pid', type=int)
    parser.add_argument('--observer-start-ticks', type=int)
    args = parser.parse_args(argv)
    if (args.observer_pid is None) != (args.observer_start_ticks is None):
        parser.error('observer PID and start ticks must be supplied together')

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'causal follow-up received signal {signum}')

    signal.signal(signal.SIGTERM, interrupted)
    run(args)


if __name__ == '__main__':
    main()
