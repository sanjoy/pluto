"""Trace previously observed multi-token words without changing generation.

Selections use generated-token half-open intervals, not tokenized target text.
The plan is frozen before any model execution. Each word token gets all branch
and head removals; its next token only gets a baseline trace. Lexical rarity and
training-corpus membership require separately documented review: the automatic
filter below establishes only a complete word absent as a vocabulary entry.

Use --freeze-only to inspect the immutable plan, then --resume to run it. Resume
accepts only a completed, hashed step result; partial steps are never overwritten
or silently restarted. A failed/abandoned step needs a fresh output directory.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import subprocess
import time

from .generated_words import analyze_native_events


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def record(path):
    """Hash an existing regular file; symlink leaves cannot be evidence."""
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError('expected regular nonsymlink file: '+str(path))
    path = path.resolve(strict=True)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return dict(path=str(path), bytes=path.stat().st_size,
                sha256=digest.hexdigest())


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def verify_records(records):
    for identity in records:
        if record(identity['path']) != identity:
            raise ValueError('evidence changed: '+identity['path'])


def local_file(directory, filename):
    relative = Path(filename)
    if (relative.is_absolute() or len(relative.parts) != 1
            or relative.name in ('', '.', '..')):
        raise ValueError('evidence filename must be local')
    path = Path(directory)/relative
    if path.is_symlink() or not path.is_file():
        raise ValueError('evidence must be a regular nonsymlink file')
    return path


def parse_word(value):
    """Parse NAME:FIRST:END, where END is excluded and IDs are generation-relative."""
    match = re.fullmatch(r'([A-Za-z]+):([0-9]+):([0-9]+)', value)
    if not match:
        raise ValueError('word must have the form ASCIIWORD:FIRST:END')
    word, first, end = match.group(1), int(match.group(2)), int(match.group(3))
    if end-first < 2:
        raise ValueError('a selected word needs at least two token occurrences')
    return dict(word=word, first=first, end=end)


def select_steps(metadata, events, token_bytes, generated_bytes, selections):
    """Validate all event windows and selected whole-word boundaries.

    The trace prefix is the recorder's last context_length IDs. In particular,
    a late word is not replayed with all preceding history or an arbitrary
    substring: absolute positions and learned positions follow native sliding.
    """
    report = analyze_native_events(metadata, events, token_bytes,
                                   generated_bytes=generated_bytes)
    context = metadata.get('context_length')
    vocab = metadata.get('vocab_size')
    if (type(context) is not int or context <= 0 or type(vocab) is not int
            or vocab != len(token_bytes) or metadata.get('autoregressive') is not True
            or not metadata['initial_token_ids']):
        raise ValueError('invalid native generation geometry')
    prompt_length = len(metadata['initial_token_ids'])
    for index, event in enumerate(events):
        absolute = prompt_length+index
        start = max(0, absolute-context)
        expected = dict(context_start=start, context_length=absolute-start,
                        output_row=absolute-start-1, logits_byte_offset=index*vocab*4)
        if any(type(event.get(key)) is not int or event[key] != value
               for key, value in expected.items()):
            raise ValueError('native event has an invalid context or logit offset')
    if not selections:
        raise ValueError('at least one word selection is required')
    seen, targets, selected = set(), set(), []
    for selection in selections:
        word, first, end = selection['word'], selection['first'], selection['end']
        if word.lower() in seen or not 0 <= first < end < len(events):
            raise ValueError('duplicate word or selection without a next boundary event')
        seen.add(word.lower())
        matches = [item for item in report['words']
                   if item['word'] == word and item['structural_candidate']
                   and [part['generated_index'] for part in item['token_overlaps']]
                   == list(range(first, end))]
        if len(matches) != 1:
            raise ValueError('selection is not a complete generated multi-token OOV word')
        if targets.intersection(range(first, end)):
            raise ValueError('word token selections overlap')
        targets.update(range(first, end))
        selected.append(dict(**selection, boundary_step=end, word_evidence=matches[0]))
    steps = sorted(targets | {item['end'] for item in selections})
    runs = []
    for step in steps:
        event = events[step]
        absolute = prompt_length+step
        ids = metadata['all_token_ids'][event['context_start']:absolute]
        runs.append(dict(step=step, interventions=step in targets,
                         event=event, context_token_ids=ids))
    return selected, runs


def freeze(generation_directory, checkpoint_directory, tokenizer_directory,
           binary, output_directory, selections):
    """Create a new frozen plan after validation; never reuse an existing path."""
    from tokenizers import Tokenizer
    from .checkpoint import GPT2Checkpoint
    from .verify import gpt2_token_bytes

    generation, checkpoint, tokenizer, binary = [Path(path).resolve(strict=True)
        for path in (generation_directory, checkpoint_directory, tokenizer_directory, binary)]
    output = Path(output_directory).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    metadata_path = local_file(generation, 'metadata.json')
    metadata = json.loads(metadata_path.read_text())
    if (Path(metadata['checkpoint_directory']).resolve() != checkpoint
            or Path(metadata['tokenizer_directory']).resolve() != tokenizer):
        raise ValueError('generation checkpoint/tokenizer differs from supplied inputs')
    paths = [metadata_path, tokenizer/'tokenizer.json', binary]
    files = metadata['files']
    artifacts = {name: local_file(generation, spec['file']) for name, spec in files.items()}
    paths.extend(artifacts.values())
    if (artifacts['events'].name != metadata['events_file']
            or artifacts['logits'].name != metadata['logits_file']):
        raise ValueError('generation file aliases disagree')
    events_raw = artifacts['events'].read_bytes()
    if not events_raw.endswith(b'\n'):
        raise ValueError('generation events must end in a complete line')
    events = [json.loads(line) for line in events_raw.splitlines()]
    token_bytes = gpt2_token_bytes(Tokenizer.from_file(str(tokenizer/'tokenizer.json')))
    selected, runs = select_steps(metadata, events, token_bytes,
                                 artifacts['generated_bytes'].read_bytes(), selections)
    ids = metadata['all_token_ids']
    if (files['tokens'].get('dtype') != 'int32'
            or files['tokens'].get('shape') != [len(ids)]
            or artifacts['tokens'].read_bytes() != struct.pack('<'+'i'*len(ids), *ids)):
        raise ValueError('generation token dump differs from metadata')
    expected_shape = [metadata['steps'], metadata['vocab_size']]
    if (metadata['logits_shape'] != expected_shape
            or files['logits'].get('shape') != expected_shape
            or files['logits'].get('dtype') != 'float32'
            or artifacts['logits'].stat().st_size != 4*expected_shape[0]*expected_shape[1]):
        raise ValueError('generation logit file shape or size is inconsistent')
    model = GPT2Checkpoint(checkpoint, check_finite=True)
    if (model.config.context_length != metadata['context_length']
            or model.config.vocab_size != metadata['vocab_size']):
        raise ValueError('checkpoint geometry differs from generation')
    paths.extend(checkpoint/spec.filename for spec in model.manifest)
    # Freeze the runner and analytical implementation, not unrelated work being
    # developed concurrently (or tests which do not participate in this run).
    # Binary hashing also fixes the linked implementation actually executed.
    scripts = Path(__file__).resolve().parent
    paths.extend(scripts/name for name in (
        'multiword_trace.py', 'generated_words.py', 'token_trace_analysis.py',
        'token_path_math.py', 'phrase_reference.py', 'phrase_trace_analysis.py',
        'checkpoint.py', 'verify.py', 'late_mlp_paths.py',
        'late_mlp_polynomial.py', 'path_diagnostics.py'))
    paths.extend(scripts.glob('token_trace_probe*'))
    paths.extend(scripts.glob('phrase_probe*'))
    identities = [record(path) for path in sorted(set(paths)) if path.is_file()]
    output.mkdir()
    with (output/'events_snapshot.jsonl').open('xb') as stream:
        stream.write(events_raw)
    for run in runs:
        prefix = output/f'prefix_step_{run["step"]}.i32'
        with prefix.open('xb') as stream:
            stream.write(struct.pack('<'+'i'*len(run['context_token_ids']),
                                     *run['context_token_ids']))
        run['prefix'] = record(prefix)
    identities.append(record(output/'events_snapshot.jsonl'))
    identities.extend(run['prefix'] for run in runs)
    verify_records(identities)
    plan = dict(schema_version=1, created_utc=timestamp(),
                generation_directory=str(generation), checkpoint_directory=str(checkpoint),
                tokenizer_directory=str(tokenizer), binary=str(binary),
                output_directory=str(output), selections=selected, runs=runs,
                vocab_size=metadata['vocab_size'], padded_vocab_size=model.config.padded_vocab_size,
                n_layers=model.config.n_layers, n_heads=model.config.n_heads,
                inputs=identities, generation_logits=str(artifacts['logits']),
                lexical_commonness_and_training_membership='Require separate documented review.',
                intervention_rule='All branches and heads independently removed for each word token; next boundary token baseline only.')
    write_json(output/'plan.json', plan)
    return plan


def check_native(plan, run, directory):
    """Require native completion, exact IDs, full removals, and original logit bytes."""
    metadata = json.loads(local_file(directory, 'metadata.json').read_text())
    required = ('final_lens_logits_byte_equal', 'alternate_padding_logits_byte_equal',
                'clean_replay_logits_byte_equal', 'attention_residual_replay_byte_equal',
                'mlp_residual_replay_byte_equal')
    if (metadata.get('complete') is not True or metadata.get('probe_kind') != 'token_trace'
            or metadata.get('token_ids') != run['context_token_ids']
            or metadata.get('target_id') != run['event']['token_id']
            or any(metadata.get('checks', {}).get(name) is not True for name in required)):
        raise ValueError('native trace is incomplete or differs from selected generation event')
    expected_arms = plan['n_layers']*(2+plan['n_heads']) if run['interventions'] else 0
    if len(metadata['interventions']) != expected_arms:
        raise ValueError('native trace omitted requested intervention arms')
    spec = metadata['files']['logits']
    if (spec['dtype'] != 'float32' or spec['shape'] != [1, plan['padded_vocab_size']]):
        raise ValueError('native logits have unexpected shape/type')
    raw = local_file(directory, spec['file']).read_bytes()
    if len(raw) != 4*plan['padded_vocab_size']:
        raise ValueError('native logit byte count is wrong')
    with Path(plan['generation_logits']).open('rb') as stream:
        stream.seek(run['event']['logits_byte_offset'])
        original = stream.read(4*plan['vocab_size'])
    if len(original) != 4*plan['vocab_size'] or raw[:len(original)] != original:
        raise ValueError('native logits are not byte-identical to original generation')
    return metadata


def artifact_records(directory):
    paths = sorted(Path(directory).rglob('*'))
    if any(path.is_symlink() for path in paths):
        raise ValueError('trace outputs must not contain symlinks')
    return [record(path) for path in paths if path.is_file()]


def verify_completed_run(plan, run, result):
    """A stale PID, metadata alone, or an analysis directory is not completion."""
    if (result.get('complete') is not True or result.get('returncode') != 0
            or result.get('step') != run['step']
            or result.get('baseline_logits_equal_original_generation') is not True
            or result.get('plan') != record(Path(plan['output_directory'])/'plan.json')
            or not result.get('outputs')):
        raise ValueError('refusing resume of an incomplete or different run')
    verify_records(result['outputs'])
    root = Path(plan['output_directory'])
    native = root/f'trace_step_{run["step"]}'
    analysis = root/f'analysis_step_{run["step"]}'
    expected = artifact_records(native)+artifact_records(analysis)
    if expected != result['outputs']:
        raise ValueError('completed run artifact set changed')
    check_native(plan, run, native)


def execute(plan):
    """Run frozen steps serially, recording PID before waiting and terminal status.

    No automatic retry is allowed: a killed runner might leave a live GPU child.
    Reusing only positively verified completed results makes that ambiguity
    visible rather than launching duplicate expensive work.
    """
    from .token_trace_analysis import run as analyze

    output = Path(plan['output_directory'])
    verify_records(plan['inputs'])
    plan_identity = record(output/'plan.json')
    if (output/'result.json').exists():
        completed = json.loads((output/'result.json').read_text())
        if completed.get('complete') is not True or completed.get('plan') != plan_identity:
            raise ValueError('invalid final result')
        for run in plan['runs']:
            verify_completed_run(plan, run, json.loads(
                (output/f'run_step_{run["step"]}.result.json').read_text()))
        return completed
    results = []
    for run in plan['runs']:
        step = run['step']
        result_path = output/f'run_step_{step}.result.json'
        native, analysis = output/f'trace_step_{step}', output/f'analysis_step_{step}'
        log, process_path = output/f'trace_step_{step}.stdout.log', output/f'run_step_{step}.process.json'
        if result_path.exists():
            result = json.loads(result_path.read_text())
            verify_completed_run(plan, run, result)
            results.append(record(result_path))
            continue
        for path in (native, analysis, log, process_path):
            if path.exists() or path.is_symlink():
                raise FileExistsError('partial run cannot be resumed or overwritten: '+str(path))
        command = [plan['binary'], '--checkpoint='+plan['checkpoint_directory'],
                   '--tokens_file='+run['prefix']['path'],
                   '--target_id='+str(run['event']['token_id']),
                   '--output_dir='+str(native),
                   '--interventions='+str(run['interventions']).lower()]
        started = time.monotonic()
        result = dict(step=step, command=command, plan=plan_identity,
                      started_utc=timestamp(), complete=False, returncode=None)
        print('Starting native trace', step, 'target', run['event']['token_id'], flush=True)
        try:
            with log.open('x') as stream:
                child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
                write_json(process_path, dict(pid=child.pid, runner_pid=os.getpid(),
                    started_utc=result['started_utc'], command=command, plan=plan_identity))
                result['returncode'] = child.wait()
            if result['returncode']:
                raise RuntimeError('native trace failed; inspect '+str(log))
            check_native(plan, run, native)
            analyze(native, plan['generation_directory'], step,
                    plan['checkpoint_directory'], plan['tokenizer_directory'], analysis)
            verify_records(plan['inputs'])
            result.update(complete=True, baseline_logits_equal_original_generation=True,
                          outputs=artifact_records(native)+artifact_records(analysis))
        except BaseException as error:
            result['error'] = type(error).__name__+': '+str(error)
            raise
        finally:
            result['ended_utc'] = timestamp()
            result['elapsed_seconds'] = time.monotonic()-started
            write_json(result_path, result)
        results.append(record(result_path))
        print('Completed native trace and analysis', step, flush=True)
    verify_records(plan['inputs'])
    final = dict(complete=True, ended_utc=timestamp(), plan=plan_identity,
                 all_inputs_unchanged=True, runs=results)
    write_json(output/'result.json', final)
    return final


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('generation-directory', 'checkpoint-directory', 'tokenizer-directory', 'binary'):
        parser.add_argument('--'+name, type=Path)
    parser.add_argument('--output-directory', type=Path, required=True)
    parser.add_argument('--word', action='append', default=[], help='WORD:FIRST:END (END excluded)')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--freeze-only', action='store_true')
    mode.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    inputs = [args.generation_directory, args.checkpoint_directory, args.tokenizer_directory, args.binary]
    if args.resume:
        if any(value is not None for value in inputs) or args.word:
            parser.error('--resume reads only the frozen plan; do not supply new selections or inputs')
        plan_path = local_file(args.output_directory, 'plan.json')
        plan = json.loads(plan_path.read_text())
        if Path(plan['output_directory']).resolve() != args.output_directory.resolve():
            raise ValueError('plan was moved to a different output directory')
    else:
        if any(value is None for value in inputs) or not args.word:
            parser.error('new runs need generation, checkpoint, tokenizer, binary, and --word')
        plan = freeze(*inputs, args.output_directory, [parse_word(word) for word in args.word])
    print('Frozen plan:', Path(plan['output_directory'])/'plan.json', flush=True)
    if not args.freeze_only:
        execute(plan)


if __name__ == '__main__':
    main()
