"""Postprocess completed paired training; optionally wait without using the GPU.

This is not a training supervisor and never starts/restarts a training arm.
Waiting validates live /proc identities, not just stale JSON or lock files.
Only a verified training_complete state, terminal logs, and immutable complete
checkpoints permit the first GPU scoring process. An interrupted scorer is
terminated and reaped; all analysis products are preserved in a new directory.

Dependencies are imported eagerly, before waiting. Later source edits cannot
silently replace the in-memory comparison/preparation code halfway through the
job. Source hashes below identify the files at this process's import time.
"""

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time

from . import checkpoint, paired_training, paired_weight_diff, paired_word_cases


def _record(path):
    path = Path(path).resolve()
    return {'path': str(path), 'bytes': path.stat().st_size,
            'sha256': checkpoint.sha256_file(path)}


LOADED_SOURCES = [_record(module.__file__) for module in
                  (checkpoint, paired_training, paired_weight_diff, paired_word_cases)]
LOADED_SOURCES.append(_record(__file__))


def _json(path):
    return json.loads(Path(path).read_text())


def _write(path, value):
    paired_training.write_json(path, value, exclusive=True)


def _process_identity(pid):
    """Read identity twice to detect exit/PID reuse during /proc inspection."""
    if type(pid) is not int or pid <= 0:
        raise ValueError('invalid process ID')
    proc = Path('/proc') / str(pid)

    def stat():
        # The parenthesized process name can itself contain spaces/parentheses.
        fields = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
        if fields[0] in ('Z', 'X', 'x'):
            raise ProcessLookupError(f'process {pid} is terminal')
        return int(fields[1]), int(fields[19])  # ppid and field22=starttime.

    parent, started = stat()
    argv = [piece.decode() for piece in (proc / 'cmdline').read_bytes().split(b'\0') if piece]
    if stat() != (parent, started) or not argv:
        raise ProcessLookupError(f'process {pid} changed while inspected')
    boot = next(int(line.split()[1]) for line in Path('/proc/stat').read_text().splitlines()
                if line.startswith('btime '))
    return {'pid': pid, 'parent_pid': parent, 'start_ticks': started, 'argv': argv,
            'started_unix': boot + started / os.sysconf('SC_CLK_TCK')}


def _validate_live(root, state, seen):
    supervisor = _process_identity(state['runner_pid'])
    expected_suffix = ['-m', 'weight_analysis.paired_training', 'run', '--output', str(root)]
    if supervisor['argv'][-5:] != expected_suffix:
        raise ValueError('live supervisor command does not match the paired experiment')
    identities = [(supervisor, state['started_utc'])]
    current = state.get('runs', {}).get(state['phase'])
    if current is not None and 'returncode' not in current and 'pid' in current:
        try:
            child = _process_identity(current['pid'])
        except (FileNotFoundError, ProcessLookupError):
            # The supervisor validates/hashes all checkpoints AFTER reaping its
            # child, before publishing the terminal record. Its own verified
            # live handle is sufficient during that potentially long phase.
            child = None
        if child is not None:
            if child['argv'] != current['command'] or child['parent_pid'] != supervisor['pid']:
                raise ValueError('live training child command/parent does not match supervisor state')
            identities.append((child, current['started_utc']))
    for identity, timestamp in identities:
        started = datetime.fromisoformat(timestamp).timestamp()
        if abs(identity['started_unix'] - started) > 10:
            raise ValueError('live process start time disagrees with recorded run start')
        old = seen.get(str(identity['pid']))
        if old is not None and (old['start_ticks'], old['argv']) != (
                identity['start_ticks'], identity['argv']):
            raise ValueError('process identity changed: possible PID reuse or restart')
        seen[str(identity['pid'])] = identity


def wait_for_training(root, *, wait=False, poll_seconds=30):
    """Return a terminal state, or fail safely after robust missing-handle checks.

    A child can exit while its verified still-live supervisor hashes the
    inventory; that remains a live job, without an arbitrary completion timeout.
    Four failed supervisor/identity observations allow transient races before
    failing, but never accept a stale file as liveness. Each pause is <=60
    seconds. No subprocess or CUDA operation occurs here.
    """
    if not math.isfinite(poll_seconds) or not 0 < poll_seconds <= 60:
        raise ValueError('poll_seconds must be in (0, 60]')
    root = Path(root).resolve()
    seen, failures, last_phase = {}, 0, None
    supervisor_pid = None
    while True:
        try:
            state = _json(root / 'state.json')
            phase = state['phase']
            if phase == 'training_complete':
                return state, seen
            if phase == 'failed':
                raise RuntimeError(f'training failed: {state.get("error", "unknown error")}')
            if not wait:
                raise RuntimeError(f'training is not complete (phase={phase}); no GPU analysis started')
            if supervisor_pid is not None and state['runner_pid'] != supervisor_pid:
                raise ValueError('training supervisor PID changed; refusing to follow a restart')
            _validate_live(root, state, seen)
            supervisor_pid = state['runner_pid']
            failures = 0
            if phase != last_phase:
                print(f'Waiting for training: phase={phase}, supervisor={supervisor_pid}', flush=True)
                last_phase = phase
        except (OSError, ValueError, KeyError) as error:
            failures += 1
            if not wait or failures >= 4:
                raise RuntimeError('cannot verify live paired-training handles after rechecking') from error
        time.sleep(poll_seconds)


def _inventory(root, arm):
    records = _json(root / arm / 'checkpoints.json')
    result = {}
    for item in records:
        step = item['step']
        if type(step) is not int or step < 0 or step in result:
            raise ValueError('invalid/duplicate checkpoint inventory step')
        expected = root / arm / 'checkpoints' / f'step_{step}'
        if Path(item['path']).resolve() != expected.resolve():
            raise ValueError('checkpoint inventory path does not match its arm/step')
        result[step] = item
    if 0 not in result:
        raise ValueError('checkpoint inventory lacks shared initialization')
    return result


def plan_comparisons(original, replacement, original_final, replacement_final, *,
                     all_matched_steps=False):
    """Plan endpoints plus the latest common step, or every common saved step.

    A coincident endpoint pair also serves its matched-step comparison. Identical
    step_0 copies share one probe result; aliases make that reuse explicit.
    """
    if original_final not in original or replacement_final not in replacement:
        raise ValueError('final checkpoint absent from inventory')
    pairs = [{'name': 'final', 'original': original[original_final],
              'replacement': replacement[replacement_final]}]
    common = sorted((set(original) & set(replacement)) - {0})
    matched = common[-1] if common else None
    selected = sorted(set(original) & set(replacement)) if all_matched_steps else (
        [] if matched is None else [matched])
    for step in selected:
        if step == original_final == replacement_final:
            continue
        name = ('initial' if step == 0 else f'matched_step_{step}') if all_matched_steps else 'matched_step'
        pairs.append({'name': name, 'original': original[step], 'replacement': replacement[step]})
    unique = {item['path']: item for pair in pairs for item in (pair['original'], pair['replacement'])}
    aliases = {}
    if all_matched_steps and 0 in original and 0 in replacement:
        a, b = original[0], replacement[0]
        if a.get('sha256') and a['sha256'] == b.get('sha256') and a['path'] != b['path']:
            aliases[b['path']] = a['path']
            unique.pop(b['path'], None)
    return {'pairs': pairs, 'checkpoints': list(unique.values()), 'matched_step': matched,
            'matched_steps': sorted(set(original) & set(replacement)) if all_matched_steps else selected,
            'all_matched_steps': all_matched_steps, 'behavior_aliases': aliases,
            'matched_is_final_pair': matched is not None and matched == original_final == replacement_final}


def validate_completion(root, state, manifest, *, all_matched_steps=False):
    """Independent log/command/inventory gates, not the phase string alone."""
    if state.get('phase') != 'training_complete' or manifest['seconds_per_arm'] != 14400:
        raise ValueError('expected completed four-hour-per-arm experiment')
    inventories = {arm: _inventory(root, arm) for arm in ('initial', 'original', 'replacement')}
    initial_hashes = inventories['initial'][0]['sha256']
    for arm in ('original', 'replacement'):
        record = state['runs'][arm]
        if record.get('returncode') != 0 or record.get('initial_weights_match') is not True:
            raise ValueError(f'{arm} lacks successful completion/shared-initialization status')
        if record['command'] != paired_training.training_command(
                manifest, arm, arm, seconds=manifest['seconds_per_arm']):
            raise ValueError(f'{arm} command does not match frozen experiment settings')
        logged = paired_training.parse_training_result((root / arm / 'train.log').read_text(), seconds=14400)
        if any(record.get(key) != value for key, value in logged.items()):
            raise ValueError(f'{arm} terminal state disagrees with actual training log')
        final = inventories[arm].get(logged['final_step'])
        if final is None or Path(record['final_checkpoint']).resolve() != Path(final['path']).resolve():
            raise ValueError(f'{arm} terminal checkpoint disagrees with inventory')
        if max(inventories[arm]) != logged['final_step']:
            raise ValueError(f'{arm} final step is not its latest checkpoint')
        if inventories[arm][0]['sha256'] != initial_hashes:
            raise ValueError(f'{arm} step_0 differs from shared initialization')
    plan = plan_comparisons(inventories['original'], inventories['replacement'],
                            state['runs']['original']['final_step'], state['runs']['replacement']['final_step'],
                            all_matched_steps=all_matched_steps)
    plan['initial'] = inventories['initial'][0]
    plan['initial_copies'] = [inventories[arm][0] for arm in ('original', 'replacement')]
    return plan


def _verify_checkpoint(item):
    model = checkpoint.GPT2Checkpoint(item['path'], check_finite=True)
    actual = {spec.filename: checkpoint.sha256_file(model.directory / spec.filename)
              for spec in model.manifest}
    if actual != item['sha256']:
        raise ValueError(f'checkpoint differs from completed inventory: {item["path"]}')
    return {'path': str(model.directory), 'weight_sha256': actual}


def validate_determinism_gate(root, state, manifest, initial):
    """Independently rehash the controls and verify the saved preflight evidence.

    The trainer's verified flag is not sufficient: every control weight must
    still match its inventory, its paired repeat, and the published gate.
    """
    if not manifest.get('determinism', {}).get('required', False):
        return None
    inventories = {arm: list(_inventory(root, arm).values())
                   for arm in ('control_a', 'control_b')}
    items = [item for inventory in inventories.values() for item in inventory]
    verified = [_verify_checkpoint(item) for item in items]
    current = paired_training.compare_determinism_controls(
        inventories['control_a'], inventories['control_b'])
    for item in items:
        if item['step'] == 0 and item['sha256'] != initial['sha256']:
            raise ValueError('determinism control differs from shared initialization')
    evidence = _json(root / 'determinism_gate.json')
    if state.get('determinism_gate') != evidence:
        raise ValueError('determinism gate evidence disagrees with terminal state')
    if (not isinstance(evidence.get('verified_utc'), str) or
            {key: value for key, value in evidence.items() if key != 'verified_utc'} !=
            {key: value for key, value in current.items() if key != 'verified_utc'}):
        raise ValueError('determinism gate evidence disagrees with current control checkpoints')
    return {'evidence': evidence, 'independently_verified_utc': current['verified_utc'],
            'items': items, 'checkpoints': verified,
            'records': [_record(root / 'determinism_gate.json'),
                        *(_record(root / arm / 'checkpoints.json') for arm in inventories)]}


def _run_probe(probe, checkpoint_path, batch_path, output, log_path):
    command = [str(probe), f'--checkpoint={checkpoint_path}', f'--batch={batch_path}',
               f'--output_dir={output}', '--batch_sequences=1']
    with log_path.open('xb') as stream:
        child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stream,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        try:
            returncode = child.wait()
        except BaseException:
            try:
                child.terminate()
            except ProcessLookupError:
                pass  # The child may have exited concurrently with interruption.
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
            raise
    if returncode:
        raise RuntimeError(f'native paired scorer exited {returncode}; inspect {log_path}')
    return {'command': command, 'pid': child.pid, 'returncode': returncode, 'log': _record(log_path)}


def analyze(root, cases_path, probe, output, *, wait=False, poll_seconds=30,
            all_matched_steps=False):
    if Path(output).is_symlink():
        raise FileExistsError(output)
    root, cases_path, probe, output = (Path(path).resolve() for path in (root, cases_path, probe, output))
    if output.exists():
        raise FileExistsError(output)
    if any(directory in output.parents or output == directory for directory in
           (cases_path.parent, *(root / arm / 'checkpoints' for arm in ('initial', 'original', 'replacement')))):
        raise ValueError('analysis output must be outside case/checkpoint inputs')
    manifest = _json(root / 'manifest.json')
    if Path(manifest['root']).resolve() != root:
        raise ValueError('manifest root disagrees with requested experiment')
    paired_training.verify_frozen_inputs(manifest)
    cases = _json(cases_path)
    frozen = [_record(root / 'manifest.json'), _record(cases_path), _record(probe)]
    if cases['manifest'] != frozen[0] or _record(cases['packed_batch']['path']) != cases['packed_batch']:
        raise ValueError('behavioral cases do not match frozen experiment/batch')
    frozen.append(cases['packed_batch'])
    output.mkdir()
    _write(output / 'request.json', {'root': str(root), 'wait_for_training': wait,
                                    'all_matched_steps': all_matched_steps,
                                    'frozen_inputs': frozen, 'loaded_sources': LOADED_SOURCES})
    try:
        state, handles = wait_for_training(root, wait=wait, poll_seconds=poll_seconds)
        plan = validate_completion(root, state, manifest, all_matched_steps=all_matched_steps)
        determinism = validate_determinism_gate(root, state, manifest, plan['initial'])
        paired_training.verify_frozen_inputs(manifest)
        for record in frozen:
            if _record(record['path']) != record:
                raise ValueError(f'frozen analysis input changed while waiting: {record["path"]}')
        checked = [plan['initial'], *plan['initial_copies'], *plan['checkpoints']]
        checkpoints = [_verify_checkpoint(item) for item in checked]
        terminal_records = [_record(root / 'state.json')]
        if determinism is not None:
            checked.extend(determinism['items'])
            checkpoints.extend(determinism['checkpoints'])
            terminal_records.extend(determinism['records'])
        for arm in ('initial', 'original', 'replacement'):
            terminal_records += [_record(root / arm / 'checkpoints.json'), _record(root / arm / 'train.log')]
        _write(output / 'analysis_plan.json', {'plan': plan, 'terminal_state': state,
                                              'verified_process_identities': handles,
                                              'determinism_verification': determinism,
                                              'checkpoints': checkpoints, 'terminal_records': terminal_records})
        weights = {}
        for pair in plan['pairs']:
            path = output / f'{pair["name"]}_weights.json'
            report = paired_weight_diff.compare_checkpoints(pair['original']['path'],
                     pair['replacement']['path'], initial=plan['initial']['path'])
            paired_weight_diff.write_report(path, report)
            weights[pair['name']] = {'report': _record(path), 'model': report['model']}
        behaviors = {}
        for index, item in enumerate(plan['checkpoints']):
            # No GPU subprocess is reachable before every completion/hash/finite
            # gate above. Recheck the binary and frozen batch before each launch.
            for record in frozen[2:]:
                if _record(record['path']) != record:
                    raise ValueError('frozen probe/cases changed before GPU scoring')
            score_dir = output / f'checkpoint_{index}_scores'
            process = _run_probe(probe, item['path'], cases['packed_batch']['path'], score_dir,
                                 output / f'checkpoint_{index}_process.log')
            summary_path = output / f'checkpoint_{index}_behavior.json'
            summary = paired_word_cases.summarize(cases_path, score_dir, summary_path)
            if Path(summary['probe_metadata']['checkpoint_directory']).resolve() != Path(item['path']).resolve():
                raise ValueError('probe output names an unexpected checkpoint')
            behaviors[item['path']] = {'process': process, 'report': _record(summary_path),
                                       'groups': summary['groups']}
        for alias, source in plan['behavior_aliases'].items():
            behaviors[alias] = dict(behaviors[source], shared_score_from=source)
        for item in checked:
            _verify_checkpoint(item)
        for record in [*frozen, *terminal_records]:
            if _record(record['path']) != record:
                raise ValueError(f'input changed during analysis: {record["path"]}')
        result = {'format': 'pluto-paired-analysis-v1', 'complete': True,
                  'completed_utc': datetime.now(timezone.utc).isoformat(), 'plan': plan,
                  'weights': weights, 'behavior': behaviors, 'frozen_inputs': frozen,
                  'loaded_sources': LOADED_SOURCES, 'terminal_records': terminal_records,
                  'checkpoints': checkpoints,
                  'determinism_verification': determinism,
                  'limitations': ['Different final step counts confound time-matched weight differences.',
                                  'Matched-step deltas still include nonlinear training-trajectory effects.',
                                  'Behavioral tests use frozen sampled contexts, not all possible prompts.']}
        _write(output / 'summary.json', result)
        return result
    except BaseException as error:
        _write(output / 'failure.json', {'type': type(error).__name__, 'error': str(error)})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'cases', 'probe', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--wait-for-training', action='store_true')
    parser.add_argument('--all-matched-steps', action='store_true',
                        help='score shared initialization and every common saved step plus endpoints')
    args = parser.parse_args(argv)

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'analysis received signal {signum}')

    signal.signal(signal.SIGTERM, interrupted)
    analyze(args.root, args.cases, args.probe, args.output, wait=args.wait_for_training,
            all_matched_steps=args.all_matched_steps)


if __name__ == '__main__':
    main()
