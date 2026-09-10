"""Count surviving lowercase exeunt presentations in the frozen sampler prefix.

The paired experiment intentionally replaces exact-case Exeunt only. Lowercase
exeunt survives and can still teach the e -> unt transition. This is a CPU-only
corpus/sampler audit: it reads saved starts and NEVER executes a manifest command
or a model. The original sampler replay was independently certified elsewhere;
these conditional counts alone do not prove that any training step ran.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from . import checkpoint, paired_training, paired_training_exposure, paired_word_cases


def surviving_positions(original, replacement, offsets, text):
    """Identify native [space-ex, e, unt] triples and verify the exact bytes.

    No substring or decoded-text approximation is used to infer token IDs.
    Adjacent prefixes and full-word boundaries matter: these counts establish
    the displayed lowercase spelling, not a claim about its syntactic role.
    """
    a, b, offsets = map(np.asarray, (original, replacement, offsets))
    if (a.ndim != 1 or b.shape != a.shape or offsets.shape != (len(a) + 1,)
            or a.dtype.kind not in 'ui' or b.dtype.kind not in 'ui'
            or offsets.dtype.kind not in 'ui' or not isinstance(text, bytes)
            or offsets[0] != 0 or offsets[-1] != len(text)
            or np.any(offsets[1:] <= offsets[:-1])):
        raise ValueError('invalid native streams or byte coverage')
    positions = np.flatnonzero((a[:-2] == 409) & (a[1:-1] == 68) & (a[2:] == 2797))
    for p in positions:
        pieces = [text[int(offsets[i]):int(offsets[i + 1])] for i in range(p, p + 3)]
        if pieces != [b' ex', b'e', b'unt'] or not np.array_equal(a[p:p+3], b[p:p+3]):
            raise ValueError('lowercase spelling bytes or surviving IDs disagree')
    return [int(p) for p in positions]


def read_starts(payload, *, seed, count, token_count, context_length):
    """Read the frozen native sampler output without running a saved command."""
    if (any(type(v) is not int for v in (seed, count, token_count, context_length))
            or count < 0 or not 3 <= context_length < token_count):
        raise ValueError('invalid sampler geometry')
    values = [int(v) for v in payload.split()]
    if (len(values) != count + 1 or values[0] != seed
            or any(not 0 <= s <= token_count-context_length-1 for s in values[1:])):
        raise ValueError('invalid sampler output')
    return values[1:]


def analyze(root, steps):
    root = Path(root).resolve(strict=True)
    steps = list(steps)
    if not steps or any(type(s) is not int or s < 0 for s in steps) or len(set(steps)) != len(steps):
        raise ValueError('steps must be distinct nonnegative integers')
    record = paired_word_cases._record
    manifest_path = root / 'manifest.json'
    used = {str(manifest_path): record(manifest_path)}
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('format') != 'pluto-paired-corpus-training-v1':
        raise ValueError('unexpected manifest')
    data = {domain: paired_word_cases._load_input(manifest['inputs'][domain+'.training'], used)
            for domain in ('original', 'replacement')}
    a, b = data['original'], data['replacement']
    alignment = paired_training.verify_alignment(a['text'], b['text'], a['tokens'], b['tokens'],
                                                 a['offsets'], b['offsets'])
    if alignment != manifest['alignment']['training']:
        raise ValueError('alignment differs from frozen manifest')
    positions = surviving_positions(a['tokens'], b['tokens'], a['offsets'], a['text'])
    # Verify byte spans in both arms, not merely an equality of numeric IDs.
    if surviving_positions(b['tokens'], a['tokens'], b['offsets'], b['text']) != positions:
        raise ValueError('lowercase survivors differ between corpora')
    flags = manifest['flags']
    batch, seed, context = flags['batch_size'], flags['seed'], 1024
    if type(batch) is not int or batch < 1:
        raise ValueError('invalid batch size')
    sampling = manifest['sampling']
    path = Path(sampling['prefix_path'])
    used[str(path)] = record(path)
    if used[str(path)]['sha256'] != sampling['prefix_sha256']:
        raise ValueError('frozen sampler prefix hash mismatch')
    command = sampling['prefix_command']
    # Authenticate the geometry in the recorded invocation; do not execute it.
    expected = [manifest['binaries']['replay_sampler']['path'], str(len(a['tokens'])),
                str(context), '10000', str(seed), '1']
    if command != expected:
        raise ValueError('saved sampler invocation disagrees with geometry')
    starts = read_starts(path.read_bytes(), seed=seed, count=10000,
                         token_count=len(a['tokens']), context_length=context)
    if max(steps) * batch > len(starts):
        raise ValueError('requested steps exceed frozen starts')
    reports = []
    for step in sorted(steps):
        counts = paired_training_exposure.exposure(starts[:step*batch], positions,
            token_count=len(a['tokens']), context_length=context)
        reports.append(dict(step=step, sequence_windows=step*batch,
            complete_presentations=sum(counts['complete_target_counts']),
            partial_presentations=sum(counts['partial_target_window_counts']),
            distinct_completely_presented=sum(n > 0 for n in counts['complete_target_counts']),
            **counts))
    sources = [record(module.__file__) for module in
               (checkpoint, paired_training, paired_training_exposure, paired_word_cases)] + [record(__file__)]
    for item in [*used.values(), *sources]:
        if record(item['path']) != item:
            raise ValueError('input or source changed during exposure analysis')
    return dict(format='pluto-lowercase-spelling-exposure-v1', complete=True,
        model_inference_performed=False, goal_completion_claimed=False,
        training_execution_verified=False, native_sampler_executed=False,
        source_manifest=used[str(manifest_path)], provenance=list(used.values()), sources=sources,
        native_sampler_recorded_command=command, context_length=context, batch_sequences=batch,
        occurrences=[dict(token_start=p, byte_start=int(a['offsets'][p]),
                          e_byte_start=int(a['offsets'][p+1]), native_ids=[409, 68, 2797])
                     for p in positions], reports=reports,
        limitations=[
            'Reads the saved deterministic sampler prefix; not a newly captured runtime batch trace.',
            'Step completion must be established separately for each arm; replacement counts are conditional.',
            'Supervised targets occupy [start+1,start+context_length+1), not the input interval.',
            'Counts apply to unchanged lowercase exeunt, not exact-case Exeunt or all uses of unt.',
            'Exposure is not evidence that the model memorized this text or causally uses these examples.'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--step', required=True, action='append', type=int)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    paired_word_cases._write_json(args.output, analyze(args.root, args.step))


if __name__ == '__main__':
    main()
