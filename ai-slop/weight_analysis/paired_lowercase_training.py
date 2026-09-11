"""Safely hand off a completed original arm to an amended replacement arm.

The legacy supervisor is allowed to finish its original four-hour run. Its
pre-created replacement directory then causes one specific, intentional
FileExistsError before it can launch the superseded corpus. This runner owns
only NEW children under the amendment directory. It never signals, restarts,
or changes the existing trainer, and never interprets a timeout as its exit.

The original arm is reused by reference, not relabeled as another execution.
All saved optimizer steps, including the final time-limited step, are retained.
Interrupted new runs are intentionally not restartable: weight-only checkpoints
do not restore AdamW or random-window sampler state.
"""

import argparse
import csv
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time

import numpy as np

from . import checkpoint, paired_training

FORMAT = 'pluto-paired-lowercase-training-v1'
SOURCE_PATHS = [Path(m.__file__).resolve() for m in
                (checkpoint, paired_training)] + [Path(__file__).resolve()]
IMPORTED_HASHES = {str(p): checkpoint.sha256_file(p) for p in SOURCE_PATHS}
now = paired_training.utc_now


def read_json(path):
    return json.loads(Path(path).read_text())


def record(path):
    """Bind an actual regular file; reject links rather than resolving them."""
    path = Path(path).absolute()
    if path.is_symlink() or not path.is_file():
        raise ValueError(f'not a regular nonsymlink file: {path}')
    return dict(path=str(path), bytes=path.stat().st_size,
                sha256=checkpoint.sha256_file(path))


def verify_records(records):
    for item in records:
        if record(item['path']) != item:
            raise ValueError('frozen evidence changed: ' + item['path'])


def publish(path, value, *, exclusive=True):
    paired_training.write_json(path, value, exclusive=exclusive)


def process_identity(pid):
    """Require stable kernel identity across argv reads, including zombies."""
    if type(pid) is not int or pid <= 0:
        raise ValueError('invalid PID')
    proc = Path('/proc') / str(pid)

    def stat():
        fields = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
        return fields[0], int(fields[1]), int(fields[19])

    before = stat()
    argv = [s.decode() for s in (proc / 'cmdline').read_bytes().split(b'\0') if s]
    after = stat()
    if after[1:] != before[1:]:
        raise OSError('process changed during identity observation')
    return dict(pid=pid, state=after[0], parent_pid=before[1],
                start_ticks=before[2], argv=argv)


def process_live(expected):
    """A failed /proc read is not by itself permission to launch GPU work."""
    try:
        actual = process_identity(expected['pid'])
    except FileNotFoundError:
        try:
            os.kill(expected['pid'], 0)  # Existence only; no signal delivered.
        except ProcessLookupError:
            return False
        raise OSError('process still exists after failed identity read')
    if actual['start_ticks'] != expected['start_ticks']:
        raise RuntimeError('upstream PID was reused; refusing handoff')
    if actual['state'] in ('Z', 'X', 'x'):
        return False
    if actual['argv'] != expected['argv']:
        raise RuntimeError('upstream command changed; refusing handoff')
    return True


def wait_for_exit(identities, *, poll_seconds=30):
    """Require two consecutive confirmed exits of BOTH original processes."""
    if not 0 < poll_seconds <= 60:
        raise ValueError('poll interval must be in (0, 60]')
    missing = errors = 0
    while True:
        try:
            live = [process_live(identity) for identity in identities]
            errors = 0
        except OSError as error:
            missing = 0
            errors += 1
            if errors >= 4:
                raise RuntimeError('cannot establish upstream exit; no new child started') from error
            time.sleep(poll_seconds)
            continue
        missing = missing + 1 if not any(live) else 0
        if missing >= 2:
            return
        time.sleep(poll_seconds)


def gpu_snapshot():
    """Read hardware/process identity; this does not initialize CUDA."""
    def query(option):
        result = subprocess.run(['nvidia-smi', option, '--format=csv,noheader'],
                                check=True, capture_output=True, text=True, timeout=15)
        return [[v.strip() for v in row] for row in csv.reader(result.stdout.splitlines()) if row]
    devices = query('--query-gpu=index,uuid,name,driver_version')
    if len(devices) != 1 or len(devices[0]) != 4 or not devices[0][1].startswith('GPU-'):
        raise ValueError('handoff requires the recorded single-GPU machine')
    index, uuid, name, driver = devices[0]
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in (None, index, uuid):
        raise ValueError('CUDA visibility differs from the recorded GPU')
    processes = query('--query-compute-apps=pid,gpu_uuid')
    if any(len(row) != 2 for row in processes):
        raise ValueError('malformed GPU process list')
    return dict(device=dict(index=index, uuid=uuid, name=name, driver_version=driver),
                processes=[dict(pid=int(pid), uuid=u) for pid, u in processes])


def require_idle_gpu(device):
    snapshot = gpu_snapshot()
    if snapshot['device'] != device or snapshot['processes']:
        raise RuntimeError('GPU identity changed or another compute job is active; refusing launch')


def inventory(directory, *, config=checkpoint.GPT2Config()):
    """Validate exact canonical files, finite FP32 values, and stable hashes."""
    directory = Path(directory)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError('missing or linked checkpoint parent')
    output = []
    for path in sorted(directory.iterdir()):
        match = re.fullmatch(r'step_(0|[1-9][0-9]*)', path.name)
        if not match or path.is_symlink() or not path.is_dir():
            raise ValueError('noncanonical checkpoint directory: ' + str(path))
        model = checkpoint.GPT2Checkpoint(path, config=config, check_finite=True)
        if {p.name for p in path.iterdir()} != {s.filename for s in model.manifest}:
            raise ValueError('checkpoint has missing or extra files')
        records = [record(path / spec.filename) for spec in model.manifest]
        verify_records(records)
        output.append(dict(step=int(match[1]), path=str(path),
                           sha256={Path(r['path']).name: r['sha256'] for r in records}))
    if not output:
        raise ValueError('no checkpoints')
    return sorted(output, key=lambda item: item['step'])


def validate_inventory(items, terminal, initial_hashes, *, control=False):
    final = terminal['final_step']
    expected = list(range(3)) if control else sorted({0, final, *range(100, final + 1, 100)})
    if (type(final) is not int or final < 1 or [r['step'] for r in items] != expected
            or items[0]['sha256'] != initial_hashes):
        raise ValueError('missing periodic/final checkpoint or wrong shared initialization')


def command(manifest, amendment, output, *, control=False):
    """Construct the known trainer invocation; never execute manifest argv."""
    old = Path(manifest['root'])
    flags = dict(manifest['flags'])
    flags.update(corpus=amendment['inputs']['replacement.full']['text'],
                 tokenizer_dir=str(old / 'tokenizer'),
                 checkpoint_dir=str(output / 'checkpoints'), log_file=str(output / 'train.log'),
                 resume_from=str(old / 'initial' / 'checkpoints'),
                 training_seconds=manifest['seconds_per_arm'])
    if control:
        flags.update(steps=2, checkpoint_every=1)
    return [manifest['binaries']['trainer']['path']] + [
        f'--{key}={str(value).lower() if isinstance(value, bool) else value}'
        for key, value in sorted(flags.items())]


def input_records(item):
    result = []
    for name in ('text', 'token_ids', 'offsets'):
        rec = record(item[name])
        if rec['sha256'] != item[name + '_sha256']:
            raise ValueError('native input differs from manifest')
        result.append(rec)
    return result


def validate_amendment(amendment, manifest, old_manifest_record):
    """Independently establish exact text, split, and unchanged native slots."""
    if (amendment.get('format') != 'pluto-paired-lowercase-amendment-v1'
            or amendment.get('complete') is not True
            or amendment['old_manifest'] != old_manifest_record
            or amendment['split_byte'] != manifest['split_byte']):
        raise ValueError('amendment does not belong to this frozen experiment')
    records, full_text = [], {}
    for split in ('full', 'training', 'test'):
        old = amendment['inputs']['original.' + split]
        new = amendment['inputs']['replacement.' + split]
        if old != manifest['inputs']['original.' + split]:
            raise ValueError('amendment changed the original corpus input')
        arrays = []
        for item in (old, new):
            records.extend(input_records(item))
            tokens = np.fromfile(item['token_ids'], dtype='<u4')
            offsets = np.fromfile(item['offsets'], dtype='<u8')
            text = Path(item['text']).read_bytes()
            export = item['export']
            if (export.get('token_dtype') != '<u4' or export.get('offset_dtype') != '<u8'
                    or export.get('roundtrip_verified') is not True
                    or export.get('token_count') != len(tokens)
                    or export.get('offset_count') != len(offsets)
                    or export.get('corpus_bytes') != len(text)
                    or Path(item['token_ids']).stat().st_size != len(tokens)*4
                    or Path(item['offsets']).stat().st_size != len(offsets)*8
                    or len(offsets) != len(tokens)+1 or offsets[0] != 0
                    or offsets[-1] != len(text) or np.any(offsets[1:] <= offsets[:-1])
                    or np.any(tokens >= 50257)):
                raise ValueError('invalid native corpus export')
            arrays.append((text, tokens, offsets))
        (a, ai, ao), (b, bi, bo) = arrays
        if b != a.replace(b'Exeunt', b'Nuveth').replace(b'exeunt', b'nuveth') or len(ai) != len(bi):
            raise ValueError('amended corpus differs beyond the two requested spellings')
        allowed_ids = np.zeros(len(ai), dtype=bool)
        allowed_offsets = np.zeros(len(ao), dtype=bool)
        count = 0
        for match in re.finditer(b'Exeunt|exeunt', a):
            spans = []
            for offsets in (ao, bo):
                start = int(np.searchsorted(offsets, match.start(), side='right') - 1)
                end = int(np.searchsorted(offsets, match.end(), side='left'))
                if end-start != 3 or offsets[end] != match.end():
                    raise ValueError('replacement does not occupy exactly three native slots')
                spans.append((start, end))
            if spans[0] != spans[1] or ao[spans[0][0]] != bo[spans[0][0]]:
                raise ValueError('replacement moved native token coordinates')
            start, end = spans[0]
            allowed_ids[start:end] = True
            allowed_offsets[start+1:end] = True
            count += 1
        if (np.any(ai[~allowed_ids] != bi[~allowed_ids])
                or np.any(ao[~allowed_offsets] != bo[~allowed_offsets])
                or amendment['alignment'][split]['token_count'] != len(ai)
                or amendment['alignment'][split]['replacements'] != count):
            raise ValueError('native alignment outside requested replacements changed')
        if split == 'full':
            full_text = dict(original=a, replacement=b)
        else:
            boundary = manifest['split_byte']
            for domain, text in (('original', a), ('replacement', b)):
                expected = full_text[domain][:boundary] if split == 'training' else full_text[domain][boundary:]
                if text != expected:
                    raise ValueError('native split does not match full corpus')
    for text in full_text.values():
        if paired_training.split_boundary(text, manifest['flags']['test_fraction']) != manifest['split_byte']:
            raise ValueError('amendment moved the training/test split')
    verify_records(records)
    return records


def validate_guard(root, output, manifest_record):
    directory = root / 'replacement'
    path = directory / 'SUPERSEDED_BEFORE_LAUNCH.json'
    guard = read_json(path)
    if (directory.is_symlink() or set(directory.iterdir()) != {path}
            or guard.get('format') != 'pluto-replacement-handoff-guard-v1'
            or guard.get('original_training_must_continue') is not True
            or guard.get('old_replacement_was_never_started') is not True
            or guard.get('old_manifest_sha256') != manifest_record['sha256']
            or guard.get('amendment_root') != str(output)
            or guard.get('replacement_mappings') != {'Exeunt': 'Nuveth', 'exeunt': 'nuveth'}):
        raise ValueError('handoff reservation is not the authorized untouched guard')
    return record(path)


def validate_handoff(root, request, manifest):
    """Accept only the planned mkdir failure AFTER original clean completion."""
    if any(process_live(i) for i in request['upstream_processes']):
        raise ValueError('upstream still live; no handoff')
    verify_records(request['frozen_inputs'])
    validate_guard(root, Path(request['amendment_root']), request['legacy_manifest'])
    state_path = root / 'state.json'
    state = read_json(state_path)
    expected_error = f"FileExistsError: [Errno 17] File exists: '{root / 'replacement'}'"
    if (state.get('phase') != 'failed' or state.get('error') != expected_error
            or state.get('runner_pid') != request['upstream_processes'][0]['pid']
            or 'replacement' in state.get('runs', {})
            or set(state.get('runs', {})) != {'initial', 'control_a', 'control_b', 'control_replacement', 'original'}):
        raise ValueError('legacy supervisor did not reach the exact authorized handoff')
    run = state['runs']['original']
    expected_command = paired_training.training_command(manifest, 'original', 'original',
                                                        seconds=manifest['seconds_per_arm'])
    if (run.get('returncode') != 0 or run.get('pid') != request['upstream_processes'][1]['pid']
            or run.get('command') != expected_command
            or read_json(root / 'original' / 'command.json') != expected_command):
        raise ValueError('original execution identity or exit status differs')
    terminal = paired_training.parse_training_result((root / 'original' / 'train.log').read_text(),
                                                     seconds=manifest['seconds_per_arm'])
    if any(run.get(key) != value for key, value in terminal.items()):
        raise ValueError('original terminal state disagrees with actual log')
    items = inventory(root / 'original' / 'checkpoints')
    validate_inventory(items, terminal, request['initial_weights'])
    if (items != read_json(root / 'original' / 'checkpoints.json')
            or run.get('final_checkpoint') != items[-1]['path']
            or run.get('initial_weights_match') is not True
            or run['elapsed_process_seconds'] < terminal['training_elapsed_seconds']):
        raise ValueError('original completion inventory or timing differs')
    # Both original controls were already executed. Recheck their current bytes,
    # not merely their old equality assertion or two similar loss values.
    gate = read_json(root / 'determinism_gate.json')
    if (gate != request['determinism_gate'] or state.get('determinism_gate') != gate):
        raise ValueError('determinism evidence changed')
    for name in ('control_a', 'control_b'):
        if inventory(root / name / 'checkpoints') != gate['checkpoints'][name]:
            raise ValueError('determinism control weights changed')
    return dict(format='pluto-reused-original-training-v1', complete=True,
                reused_not_rerun=True, original_root=str(root), run=run,
                checkpoints=items, terminal=terminal,
                supervisor_exit_code_observed=False,
                intentional_handoff_error=expected_error,
                evidence=[record(state_path)] + [record(root / 'original' / name)
                    for name in ('command.json', 'train.log', 'process.log', 'checkpoints.json')])


def run_child(output, name, argv, state, request, *, control=False):
    """Own and reap only the child launched here; never signal legacy PIDs."""
    verify_records(request['frozen_inputs'])
    require_idle_gpu(request['gpu'])
    destination = output / name
    destination.mkdir(exist_ok=False)
    publish(destination / 'command.json', argv)
    run = dict(command=argv, started_utc=now())
    state['phase'] = name
    state['runs'][name] = run
    publish(output / 'state.json', state, exclusive=False)
    started = time.monotonic()
    with (destination / 'process.log').open('xb') as log:
        child = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        run['pid'] = child.pid
        try:
            run['process_identity'] = process_identity(child.pid)
            publish(output / 'state.json', state, exclusive=False)
            print(f'{now()} started amended {name}: pid={child.pid}', flush=True)
            run['returncode'] = child.wait()
        except BaseException:
            try:
                child.terminate()
            except ProcessLookupError:
                pass
            try:
                run['returncode'] = child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                child.kill()
                run['returncode'] = child.wait()
            raise
    run.update(finished_utc=now(), elapsed_process_seconds=time.monotonic()-started)
    if run['returncode']:
        raise RuntimeError(f'amended {name} exited {run["returncode"]}')
    terminal = paired_training.parse_training_result((destination / 'train.log').read_text(),
        seconds=None if control else request['seconds_per_arm'], expected_steps=2 if control else None)
    items = inventory(destination / 'checkpoints')
    validate_inventory(items, terminal, request['initial_weights'], control=control)
    verify_records(request['frozen_inputs'])
    run.update(terminal, initial_weights_match=True, final_checkpoint=items[-1]['path'])
    publish(destination / 'checkpoints.json', items)
    run['evidence'] = [record(destination / n) for n in ('command.json', 'train.log', 'process.log', 'checkpoints.json')]
    state['updated_utc'] = now()
    publish(output / 'state.json', state, exclusive=False)
    print(f'{now()} completed amended {name}: step={terminal["final_step"]}', flush=True)
    return run


def run(args):
    root, output = args.legacy_root.resolve(strict=True), args.amendment.resolve(strict=True)
    if output != root / 'lowercase_amendment' or args.amendment.is_symlink():
        raise ValueError('amendment must be the new immediate child lowercase_amendment')
    with (output / 'runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if any((output / name).exists() for name in
               ('request.json', 'state.json', 'failure.json', 'control_replacement', 'replacement')):
            raise ValueError('amended runner already started; inspect, never resume automatically')
        for path, digest in IMPORTED_HASHES.items():
            if checkpoint.sha256_file(path) != digest:
                raise ValueError('runner source changed after import')
        legacy_record = record(root / 'manifest.json')
        manifest = read_json(legacy_record['path'])
        if (manifest.get('format') != 'pluto-paired-corpus-training-v1'
                or manifest.get('root') != str(root) or manifest['seconds_per_arm'] != 14400
                or manifest['flags']['seed'] != 17 or manifest['flags']['batch_size'] != 10
                or manifest.get('determinism', {}).get('required') is not True):
            raise ValueError('legacy experiment differs from the requested controlled design')
        paired_training.verify_frozen_inputs(manifest)
        amendment_path = output / 'amendments.json'
        amendment_record = record(amendment_path)
        amendment = read_json(amendment_path)
        frozen = [legacy_record, amendment_record,
                  validate_guard(root, output, legacy_record)]
        frozen.extend(validate_amendment(amendment, manifest, legacy_record))
        identities = [process_identity(args.supervisor_pid), process_identity(args.original_pid)]
        supervisor, original = identities
        if (supervisor['start_ticks'] != args.supervisor_start_ticks
                or supervisor['argv'][-5:] != ['-m', 'weight_analysis.paired_training', 'run', '--output', str(root)]
                or original['start_ticks'] != args.original_start_ticks
                or original['parent_pid'] != supervisor['pid']
                or original['argv'] != paired_training.training_command(manifest, 'original', 'original', seconds=manifest['seconds_per_arm'])
                or any(p['state'] in ('Z', 'X', 'x') for p in identities)):
            raise ValueError('expected live original trainer/supervisor identity differs')
        snapshot = gpu_snapshot()
        if dict(pid=original['pid'], uuid=snapshot['device']['uuid']) not in snapshot['processes']:
            raise ValueError('original trainer is not on the recorded GPU')
        initial = inventory(root / 'initial' / 'checkpoints')
        if [r['step'] for r in initial] != [0]:
            raise ValueError('shared initialization is not exactly step zero')
        gate_path = root / 'determinism_gate.json'
        gate = read_json(gate_path)
        repeated = paired_training.compare_determinism_controls(
            inventory(root / 'control_a' / 'checkpoints'), inventory(root / 'control_b' / 'checkpoints'))
        if (gate.get('status') != 'verified' or gate['checkpoints'] != repeated['checkpoints']
                or gate['steps'] != [0, 1, 2]):
            raise ValueError('legacy determinism gate is not authenticated')
        frozen.append(record(gate_path))
        for item in manifest['binaries'].values():
            frozen.append(record(item['path']))
        for item in manifest['source_files'].values():
            frozen.append(record(item['path']))
        for item in manifest['tokenizer_files'].values():
            frozen.append(record(item['path']))
        frozen.append(record(manifest['sampling']['prefix_path']))
        for group in (initial, *gate['checkpoints'].values()):
            for item in group:
                frozen.extend(record(Path(item['path']) / filename) for filename in item['sha256'])
        copied = output / 'runner_source'
        copied.mkdir()
        for path in SOURCE_PATHS:
            if checkpoint.sha256_file(path) != IMPORTED_HASHES[str(path)]:
                raise ValueError('runner source changed while preparing provenance')
            frozen.append(record(path))
            frozen.append(paired_training.freeze_file(path, copied / path.name))
        # Bind provenance sources and exports of the CPU-only amendment as well.
        for key in ('provenance', 'sources'):
            values = amendment.get(key, [])
            if isinstance(values, dict):
                values = values.values()
            for item in values:
                if isinstance(item, dict) and all(k in item for k in ('path', 'bytes', 'sha256')):
                    frozen.append({k: item[k] for k in ('path', 'bytes', 'sha256')})
        frozen = list({item['path']: item for item in frozen}.values())
        verify_records(frozen)
        request = dict(format=FORMAT, created_utc=now(), amendment_root=str(output),
                       legacy_root=str(root), legacy_manifest=legacy_record,
                       amendment=amendment_record, upstream_processes=identities,
                       gpu=snapshot['device'], determinism_gate=gate,
                       initial_weights=initial[0]['sha256'], frozen_inputs=frozen,
                       seconds_per_arm=manifest['seconds_per_arm'],
                       commands={name: command(manifest, amendment, output / name, control=name=='control_replacement')
                                 for name in ('control_replacement', 'replacement')},
                       original_reused_not_rerun=True, goal_completion_claimed=False)
        publish(output / 'request.json', request)
        state = dict(format=FORMAT, phase='waiting_original', runner_pid=os.getpid(),
                     started_utc=now(), updated_utc=now(), runs={})
        publish(output / 'state.json', state)
        print(f'{now()} waiting for exact legacy original/supervisor exit; no new CUDA work', flush=True)
        try:
            wait_for_exit(identities)
            original_reference = validate_handoff(root, request, manifest)
            publish(output / 'original_reference.json', original_reference)
            state['original_reference'] = record(output / 'original_reference.json')
            for name in ('control_replacement', 'replacement'):
                run_child(output, name, request['commands'][name], state, request,
                          control=name=='control_replacement')
            state.update(phase='training_complete', complete=True,
                         goal_completion_claimed=False, updated_utc=now())
            publish(output / 'state.json', state, exclusive=False)
        except BaseException as error:
            state.update(phase='failed', error=f'{type(error).__name__}: {error}', updated_utc=now())
            publish(output / 'state.json', state, exclusive=False)
            publish(output / 'failure.json', dict(state, no_automatic_restart=True))
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    execute = sub.add_parser('run')
    execute.add_argument('--legacy-root', type=Path, required=True)
    execute.add_argument('--amendment', type=Path, required=True)
    for name in ('supervisor-pid', 'supervisor-start-ticks', 'original-pid', 'original-start-ticks'):
        execute.add_argument('--' + name, type=int, required=True)
    args = parser.parse_args(argv)
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'amended runner received signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    run(args)


if __name__ == '__main__':
    main()
