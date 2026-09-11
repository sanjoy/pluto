"""Prepare and run a preserved, same-initialization corpus-replacement study.

The two arms differ only in their corpus and output paths. Both start with
fresh AdamW state and the same saved step_0, use the native tokenizer, and draw
the same seeded random windows. Four-hour endpoints need not have the same
step count: periodic common-step checkpoints are the causal comparison.

Nothing here removes checkpoints or resumes a partially completed arm. An
interrupted experiment requires inspection, since Pluto's checkpoints do not
contain optimizer or sampler state. With --require-determinism, two short
unchanged-corpus repeats must have identical weights at every saved step before
either long arm can start. This checks the fixed-step contract on the same
hardware/build/runtime; it is not a proof for all possible training workloads.
Without that opt-in, legacy experiments only measure short-run numerical drift.
"""

import argparse
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import time

import numpy as np

from weight_analysis import checkpoint as checkpoint_module
from weight_analysis.checkpoint import GPT2Checkpoint, sha256_file


# Capture provenance at import, before a long preparation or run can outlive an
# edit to its working-tree source. A Git commit alone omits uncommitted changes.
SOURCE_PATHS = {'paired_training.py': Path(__file__).resolve(),
                'checkpoint.py': Path(checkpoint_module.__file__).resolve()}
IMPORTED_SOURCE_HASHES = {name: sha256_file(path) for name, path in SOURCE_PATHS.items()}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value, *, exclusive=False):
    """Publish state atomically, or create immutable provenance exclusively."""
    path = Path(path)
    data = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n'
    if exclusive:
        with path.open('x') as stream:
            stream.write(data)
        return
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temporary.open('x') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def split_boundary(corpus, test_fraction):
    """Mirror SplitCorpus's byte-based, newline-aligned split exactly."""
    if not math.isfinite(test_fraction) or not 0 < test_fraction < 1:
        raise ValueError('test_fraction must be finite and in (0, 1)')
    approximate = int(len(corpus) * (1.0 - test_fraction))
    newline = corpus.find(b'\n', approximate)
    boundary = newline + 1 if newline >= 0 else approximate
    if newline < 0:
        while boundary < len(corpus) and corpus[boundary] & 0xc0 == 0x80:
            boundary += 1
    if not 0 < boundary < len(corpus):
        raise ValueError('split must contain nonempty training and test text')
    return boundary


def verify_alignment(original, replacement, original_ids, replacement_ids,
                     original_offsets, replacement_offsets):
    """Prove substitution changes only three token slots per occurrence.

    Internal byte boundaries may differ (Ex/e/unt vs N/uve/th); outer word
    boundaries and every other token's identity and byte boundaries must not.
    Equal total counts alone would not establish this stronger invariant.
    """
    if replacement != original.replace(b'Exeunt', b'Nuveth'):
        raise ValueError('replacement contains unrelated text changes')
    if len(original_ids) != len(replacement_ids):
        raise ValueError('native token counts differ')
    n = len(original_ids)
    for offsets in (original_offsets, replacement_offsets):
        if (len(offsets) != n + 1 or offsets[0] != 0 or
                offsets[-1] != len(original) or np.any(offsets[1:] <= offsets[:-1])):
            raise ValueError('invalid native byte offsets')
    allowed_tokens = np.zeros(n, dtype=bool)
    allowed_boundaries = np.zeros(n + 1, dtype=bool)
    occurrences = []
    for match in re.finditer(b'Exeunt', original):
        spans = []
        for offsets in (original_offsets, replacement_offsets):
            first = int(np.searchsorted(offsets, match.start(), side='right') - 1)
            last = int(np.searchsorted(offsets, match.end(), side='left'))
            if last - first != 3 or offsets[last] != match.end():
                raise ValueError('a replacement does not span exactly three tokens')
            spans.append((first, last))
        if spans[0] != spans[1]:
            raise ValueError('replacement token positions shifted')
        first, last = spans[0]
        if original_offsets[first] != replacement_offsets[first]:
            raise ValueError('replacement outer byte boundary shifted')
        allowed_tokens[first:last] = True
        allowed_boundaries[first + 1:last] = True
        occurrences.append({'byte_start': match.start(), 'token_start': first,
                            'original_ids': original_ids[first:last].tolist(),
                            'replacement_ids': replacement_ids[first:last].tolist()})
    if np.any(original_ids[~allowed_tokens] != replacement_ids[~allowed_tokens]):
        raise ValueError('a token outside the replacement changed')
    if np.any(original_offsets[~allowed_boundaries] !=
              replacement_offsets[~allowed_boundaries]):
        raise ValueError('a byte boundary outside the replacement changed')
    return {'token_count': n, 'replacements': len(occurrences),
            'changed_token_ids': int(np.count_nonzero(original_ids != replacement_ids)),
            'outside_replacement_tokens_identical': True, 'occurrences': occurrences}


def freeze_file(source, destination, *, executable=False):
    if destination.exists():
        raise FileExistsError(destination)
    shutil.copyfile(source, destination)
    destination.chmod(0o555 if executable else 0o444)
    return {'path': str(destination), 'sha256': sha256_file(destination),
            'bytes': destination.stat().st_size}


def freeze_sources(root):
    """Preserve the launcher/helper sources, rejecting edits since import."""
    for name, path in SOURCE_PATHS.items():
        if sha256_file(path) != IMPORTED_SOURCE_HASHES[name]:
            raise ValueError(f'experiment source changed since import: {path}')
    destination = root / 'source'
    destination.mkdir()
    records = {}
    for name, path in SOURCE_PATHS.items():
        record = freeze_file(path, destination / name)
        if record['sha256'] != IMPORTED_SOURCE_HASHES[name]:
            raise ValueError(f'experiment source changed while copying: {path}')
        records[name] = dict(record, original_path=str(path))
    return records


def prepare(args):
    root = args.output.resolve()
    if not math.isfinite(args.seconds) or args.seconds <= 0 or args.batch_size <= 0:
        raise ValueError('seconds and batch_size must be positive')
    source = args.corpus.read_bytes()
    if b'nuveth' in source.lower() or not source.count(b'Exeunt'):
        raise ValueError('need Exeunt occurrences and an absent Nuveth control')
    replaced = source.replace(b'Exeunt', b'Nuveth')
    boundary = split_boundary(source, args.test_fraction)
    if split_boundary(replaced, args.test_fraction) != boundary:
        raise ValueError('replacement changed corpus split')
    root.mkdir(parents=True, exist_ok=False)
    (root / 'bin').mkdir()
    (root / 'inputs').mkdir()
    (root / 'tokenizer').mkdir()
    source_files = freeze_sources(root)
    binaries = {}
    for name, path in [('trainer', args.trainer), ('tokenize_corpus', args.tokenizer_binary),
                       ('replay_sampler', args.sampler_binary)]:
        binaries[name] = freeze_file(path, root / 'bin' / name, executable=True)
    tokenizer_files = {}
    for path in sorted(args.tokenizer_dir.iterdir()):
        if path.is_file():
            tokenizer_files[path.name] = freeze_file(path, root / 'tokenizer' / path.name)
    alignments = {}
    inputs = {}
    for split, begin, end in [('full', 0, len(source)), ('training', 0, boundary),
                              ('test', boundary, len(source))]:
        arrays = {}
        for arm, content in [('original', source[begin:end]),
                             ('replacement', replaced[begin:end])]:
            text_path = root / 'inputs' / f'{arm}.{split}.txt'
            with text_path.open('xb') as stream:
                stream.write(content)
            text_path.chmod(0o444)
            token_path = text_path.with_suffix('.tokens.bin')
            command = [binaries['tokenize_corpus']['path'], str(root / 'tokenizer'),
                       str(text_path), str(token_path)]
            exported = subprocess.run(command, capture_output=True, text=True, check=True)
            offsets_path = Path(str(token_path) + '.offsets.bin')
            arrays[arm] = (np.fromfile(token_path, dtype='<u4'),
                           np.fromfile(offsets_path, dtype='<u8'))
            inputs[f'{arm}.{split}'] = {
                'text': str(text_path), 'text_sha256': sha256_file(text_path),
                'token_ids': str(token_path), 'token_ids_sha256': sha256_file(token_path),
                'offsets': str(offsets_path), 'offsets_sha256': sha256_file(offsets_path),
                'export': json.loads(exported.stdout), 'command': command}
        alignments[split] = verify_alignment(
            source[begin:end], replaced[begin:end], arrays['original'][0],
            arrays['replacement'][0], arrays['original'][1], arrays['replacement'][1])
    sample_command = [binaries['replay_sampler']['path'],
                      str(alignments['training']['token_count']), '1024', '10000',
                      str(args.seed), '1']
    sample = subprocess.run(sample_command, capture_output=True, check=True)
    sample_path = root / 'inputs' / 'first_10000_sequence_starts.txt'
    with sample_path.open('xb') as stream:
        stream.write(sample.stdout)
    implementation = subprocess.run([binaries['replay_sampler']['path'], '--implementation'],
                                    capture_output=True, text=True, check=True).stdout.strip()
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True,
                            text=True, check=True).stdout.strip()
    require_determinism = getattr(args, 'require_determinism', False)
    limitations = ['time-matched endpoints can have different step counts',
                   'checkpoints contain weights, not AdamW or sampler state']
    if require_determinism:
        limitations.extend([
            'bitwise reproducibility requires the same hardware/build/runtime, '
            'inputs, seed, starting state, and completed step count',
            'short repeatability controls check the tested workload, not every '
            'possible workload or the complete four-hour trajectory'])
    else:
        limitations.extend([
            'CUDA atomic sums are not guaranteed bitwise deterministic',
            'short repeatability controls do not bound long-run drift'])
    manifest = {
        'format': 'pluto-paired-corpus-training-v1', 'created_utc': utc_now(),
        'root': str(root), 'code_commit': commit, 'binaries': binaries,
        'source_files': source_files,
        'tokenizer_files': tokenizer_files, 'inputs': inputs, 'alignment': alignments,
        'replacement': {'from': 'Exeunt', 'to': 'Nuveth', 'case_sensitive': True},
        'split_byte': boundary, 'seconds_per_arm': args.seconds,
        'flags': {'mode': 'train_model', 'batch_size': args.batch_size, 'seed': args.seed,
                  'learning_rate': 0.0003, 'adam_beta1': 0.9, 'adam_beta2': 0.95,
                  'adam_epsilon': 1e-8, 'weight_decay': 0.1,
                  'eval_batches': 4, 'test_fraction': args.test_fraction,
                  'training_eval_interval': 100, 'train_until_loss': -1.0,
                  'checkpoint_every': 100, 'checkpoint_initial': True},
        'sampling': {'kind': 'independent uniform random windows, not shuffled epochs',
                     'engine': 'std::mt19937_64', 'implementation': implementation,
                     'prefix_command': sample_command, 'prefix_path': str(sample_path),
                     'prefix_sha256': sha256_file(sample_path)},
        'determinism': {
            'required': require_determinism,
            'contract': ('same hardware/build/runtime, inputs, seed, starting state, '
                         'and completed step count'),
            'preflight': ('exact per-weight SHA-256 equality for control_a and '
                          'control_b at steps 0, 1, and 2 before long arms'
                          if require_determinism else 'measure short-run drift only')},
        'limitations': limitations}
    write_json(root / 'manifest.json', manifest, exclusive=True)
    write_json(root / 'state.json', {'phase': 'prepared', 'updated_utc': utc_now()}, exclusive=True)
    print(json.dumps({'root': str(root), 'replacements': source.count(b'Exeunt'),
                      'training_tokens': alignments['training']['token_count'],
                      'test_tokens': alignments['test']['token_count']}), flush=True)


def training_command(manifest, name, arm, *, steps=None, seconds=None, resume=True):
    root = Path(manifest['root'])
    output = root / name
    flags = dict(manifest['flags'])
    flags.update(corpus=manifest['inputs'][f'{arm}.full']['text'],
                 tokenizer_dir=str(root / 'tokenizer'), checkpoint_dir=str(output / 'checkpoints'),
                 log_file=str(output / 'train.log'))
    if resume:
        flags['resume_from'] = str(root / 'initial' / 'checkpoints')
    if steps is not None:
        flags['steps'] = steps
        if steps > 0:
            # Short repeatability controls use the same completed-step
            # synchronization as the timed arms, but hit their step cap first.
            flags['training_seconds'] = manifest['seconds_per_arm']
            if manifest.get('determinism', {}).get('required', False):
                # Save every control step: matching initialization or endpoints
                # alone would miss a divergence at an intermediate update.
                flags['checkpoint_every'] = 1
    if seconds is not None:
        flags['training_seconds'] = seconds
    return [manifest['binaries']['trainer']['path']] + [
        f'--{key}={str(value).lower() if isinstance(value, bool) else value}'
        for key, value in sorted(flags.items())]


def checkpoint_inventory(directory):
    """Validate completed output before declaring a child run successful."""
    checkpoints = []
    for path in sorted(directory.iterdir()):
        match = re.fullmatch(r'step_(\d+)', path.name)
        if match and path.is_dir():
            checkpoint = GPT2Checkpoint(path, check_finite=True)
            checkpoints.append({'step': int(match[1]), 'path': str(path),
                                'sha256': {s.filename: sha256_file(path / s.filename)
                                           for s in checkpoint.manifest}})
    if not checkpoints:
        raise ValueError(f'no checkpoints produced in {directory}')
    return sorted(checkpoints, key=lambda value: value['step'])


def compare_determinism_controls(control_a, control_b, *, expected_steps=2):
    """Require all saved weights to match at each fixed control step.

    Inventories are obtained by validating and hashing the actual checkpoint
    files, not inferred from a successful process exit or identical losses.
    Missing/duplicate steps or empty inventories must not vacuously pass.
    The returned evidence retains both checkpoint paths and every file hash.
    """
    expected = set(range(expected_steps + 1))
    by_step = {}
    for name, inventory in [('control_a', control_a), ('control_b', control_b)]:
        steps = [item['step'] for item in inventory]
        if len(set(steps)) != len(steps) or set(steps) != expected:
            raise ValueError(f'{name} must contain exactly control steps '
                             f'{sorted(expected)}; found {steps}')
        by_step[name] = {item['step']: item for item in inventory}
        for item in inventory:
            if not item['sha256']:
                raise ValueError(f'{name} step {item["step"]} has no weight hashes')
    for step in sorted(expected):
        a = by_step['control_a'][step]['sha256']
        b = by_step['control_b'][step]['sha256']
        if a != b:
            changed = [name for name in sorted(set(a) | set(b)) if a.get(name) != b.get(name)]
            raise ValueError(f'determinism control mismatch at step {step}: '
                             f'{len(changed)} weight files differ: {changed}')
    return {'status': 'verified', 'verified_utc': utc_now(),
            'method': 'exact SHA-256 equality for every weight at every control step',
            'steps': sorted(expected),
            'checkpoints': {'control_a': control_a, 'control_b': control_b}}


def verify_frozen_inputs(manifest):
    """Recheck provenance before every child, including the second long arm."""
    records = list(manifest['binaries'].values()) + list(manifest['tokenizer_files'].values())
    records.extend(manifest.get('source_files', {}).values())
    for record in manifest['inputs'].values():
        for key in ('text', 'token_ids', 'offsets'):
            records.append({'path': record[key], 'sha256': record[key + '_sha256']})
    records.append({'path': manifest['sampling']['prefix_path'],
                    'sha256': manifest['sampling']['prefix_sha256']})
    for record in records:
        if sha256_file(record['path']) != record['sha256']:
            raise ValueError(f'frozen experiment input changed: {record["path"]}')
    for name, record in manifest.get('source_files', {}).items():
        if IMPORTED_SOURCE_HASHES.get(name) != record['sha256']:
            raise ValueError(f'loaded experiment source differs from prepared source: {name}')


def parse_training_result(log_text, *, seconds=None, expected_steps=None):
    """Require a clean terminal result, not just an elapsed child process."""
    patterns = {'final_step': r'training stopped at step: (\d+)',
                'stop_reason': r'^training stop reason: (\w+)',
                'training_elapsed_seconds': r'^training elapsed seconds: (\S+)',
                'training_loss': r'^final training loss: (\S+)',
                'test_loss': r'^final test loss: (\S+)'}
    result = {}
    for key, pattern in patterns.items():
        values = re.findall(pattern, log_text, re.MULTILINE)
        if len(values) != 1:
            raise ValueError(f'missing or repeated terminal training field: {key}')
        result[key] = values[0]
    result['final_step'] = int(result['final_step'])
    for key in ('training_elapsed_seconds', 'training_loss', 'test_loss'):
        result[key] = float(result[key])
        if not math.isfinite(result[key]):
            raise ValueError(f'non-finite training result: {key}')
    if seconds is not None and (result['stop_reason'] != 'time_limit' or
                                result['training_elapsed_seconds'] < seconds):
        raise ValueError('training did not complete its full wall-clock budget')
    if expected_steps is not None and result['final_step'] != expected_steps:
        raise ValueError('training did not complete its requested step count')
    return result


def run(root):
    root = root.resolve()
    manifest = json.loads((root / 'manifest.json').read_text())
    # Keep this descriptor open for the whole workflow. State alone is not a
    # liveness check, and a second invocation must never overwrite an arm.
    with (root / 'runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = json.loads((root / 'state.json').read_text())
        if state['phase'] != 'prepared':
            raise ValueError('experiment already started; inspect it, do not restart automatically')
        verify_frozen_inputs(manifest)
        state = {'phase': 'starting', 'runner_pid': os.getpid(), 'started_utc': utc_now(),
                 'runs': {}}
        write_json(root / 'state.json', state)
        plan = [('initial', 'original', 0, None, False),
                ('control_a', 'original', 2, None, True),
                ('control_b', 'original', 2, None, True),
                ('control_replacement', 'replacement', 2, None, True),
                ('original', 'original', None, manifest['seconds_per_arm'], True),
                ('replacement', 'replacement', None, manifest['seconds_per_arm'], True)]
        try:
            initial_hashes = None
            for name, arm, steps, seconds, resume in plan:
                verify_frozen_inputs(manifest)
                output = root / name
                output.mkdir(exist_ok=False)
                command = training_command(manifest, name, arm, steps=steps,
                                           seconds=seconds, resume=resume)
                record = {'command': command, 'started_utc': utc_now()}
                state['phase'] = name
                state['runs'][name] = record
                write_json(output / 'command.json', command, exclusive=True)
                started = time.monotonic()
                with (output / 'process.log').open('xb') as log:
                    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                               stdin=subprocess.DEVNULL)
                    record['pid'] = process.pid
                    state['updated_utc'] = utc_now()
                    write_json(root / 'state.json', state)
                    print(f'{utc_now()} started {name}: pid={process.pid}', flush=True)
                    try:
                        record['returncode'] = process.wait()
                    except BaseException:
                        # A deliberate runner interruption must not leave an
                        # unmonitored child training after releasing the lock.
                        process.terminate()
                        try:
                            record['returncode'] = process.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            record['returncode'] = process.wait()
                        raise
                record['elapsed_process_seconds'] = time.monotonic() - started
                record['finished_utc'] = utc_now()
                if record['returncode']:
                    raise RuntimeError(f'{name} exited {record["returncode"]}; inspect {output}')
                inventory = checkpoint_inventory(output / 'checkpoints')
                if inventory[0]['step'] != 0:
                    raise ValueError(f'{name} did not preserve initial weights')
                if initial_hashes is None:
                    initial_hashes = inventory[0]['sha256']
                elif inventory[0]['sha256'] != initial_hashes:
                    raise ValueError(f'{name} initial weights do not match shared step_0')
                log_text = (output / 'train.log').read_text()
                terminal = parse_training_result(log_text, seconds=seconds, expected_steps=steps)
                if inventory[-1]['step'] != terminal['final_step']:
                    raise ValueError(f'{name} final checkpoint does not match logged final step')
                record.update(terminal)
                record['final_checkpoint'] = inventory[-1]['path']
                record['initial_weights_match'] = True
                write_json(output / 'checkpoints.json', inventory, exclusive=True)
                state['updated_utc'] = utc_now()
                write_json(root / 'state.json', state)
                print(f'{utc_now()} completed {name}: step={record["final_step"]}', flush=True)
                if (name == 'control_b' and
                        manifest.get('determinism', {}).get('required', False)):
                    # Re-read both directories at the gate, rather than trusting
                    # stale inventories from when the first control finished.
                    evidence = compare_determinism_controls(
                        checkpoint_inventory(root / 'control_a' / 'checkpoints'),
                        checkpoint_inventory(root / 'control_b' / 'checkpoints'))
                    write_json(root / 'determinism_gate.json', evidence, exclusive=True)
                    state['determinism_gate'] = evidence
                    state['updated_utc'] = utc_now()
                    write_json(root / 'state.json', state)
                    print(f'{utc_now()} verified exact deterministic controls at '
                          'steps 0, 1, and 2', flush=True)
            state['phase'] = 'training_complete'
        except BaseException as error:
            state['phase'] = 'failed'
            state['error'] = f'{type(error).__name__}: {error}'
            raise
        finally:
            state['updated_utc'] = utc_now()
            write_json(root / 'state.json', state)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    for name in ('output', 'corpus', 'tokenizer-dir', 'trainer', 'tokenizer-binary', 'sampler-binary'):
        prep.add_argument('--' + name, type=Path, required=True)
    prep.add_argument('--seconds', type=float, default=4 * 60 * 60)
    prep.add_argument('--batch-size', type=int, default=10)
    prep.add_argument('--seed', type=int, default=17)
    prep.add_argument('--test-fraction', type=float, default=0.1)
    prep.add_argument('--require-determinism', action='store_true',
                      help='require identical fixed-step repeat controls before long arms')
    execute = sub.add_parser('run')
    execute.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == 'prepare':
        prepare(args)
    else:
        def interrupted(signum, frame):
            raise KeyboardInterrupt(f'runner received signal {signum}')
        signal.signal(signal.SIGTERM, interrupted)
        run(args.output)


if __name__ == '__main__':
    main()
