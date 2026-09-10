"""Count word exposure in a frozen deterministic training-window replay.

This is corpus/sampler arithmetic, not a model forward or a training-data
reconstruction from weights. Each input window is [s,s+C); its supervised
targets are [s+1,s+C+1). Count complete and partial word presentations separately.
Containing a case's prefix segment does not imply identical preceding context
or local learned position IDs. Replacement-arm counts are conditional on that
arm reaching the specified optimizer step, not proof it has already run.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess

import numpy as np

from . import paired_training, paired_word_cases
from .checkpoint import sha256_file


def exposure(starts, positions, *, token_count, context_length):
    """Return exact per-occurrence counts, including words clipped at edges."""
    if (type(token_count) is not int or type(context_length) is not int
            or not 3 <= context_length < token_count
            or token_count > np.iinfo(np.int64).max - 3):
        raise ValueError('invalid corpus/context size')
    for values in (starts, positions):
        if any(type(value) is not int for value in values):
            raise ValueError('token positions must be integers')
    if (any(not 0 <= value <= token_count-context_length-1 for value in starts)
            or any(not 0 <= value <= token_count-3 for value in positions)
            or any(b < a+3 for a, b in zip(positions, positions[1:]))):
        raise ValueError('invalid windows or overlapping/unsorted word positions')
    starts = np.asarray(starts, dtype=np.int64)
    positions = np.asarray(positions, dtype=np.int64)
    piece_counts = np.zeros((len(positions), 3), dtype=np.int64)
    full_counts = np.zeros(len(positions), dtype=np.int64)
    partial_counts = np.zeros(len(positions), dtype=np.int64)
    # Bound temporary storage even if the replay contains many windows. The
    # three coordinates remain separate; seeing a suffix is not seeing a word.
    for offset in range(0, len(starts), 128):
        begin = starts[offset:offset+128, None, None]+1
        tokens = positions[None, :, None]+np.arange(3)[None, None, :]
        hits = (tokens >= begin) & (tokens < begin+context_length)
        full = hits.all(axis=2)
        piece_counts += hits.sum(axis=0)
        full_counts += full.sum(axis=0)
        partial_counts += (hits.any(axis=2) & ~full).sum(axis=0)
    return dict(piece_target_counts=piece_counts.tolist(),
                complete_target_counts=full_counts.tolist(),
                partial_target_window_counts=partial_counts.tolist())


def case_coverage(starts, *, prefix_start, word_start, context_length):
    """Distinguish contained segments from identical initial causal context."""
    if (any(type(v) is not int for v in (*starts, prefix_start, word_start, context_length))
            or not 0 <= prefix_start < word_start or context_length < 3
            or any(s < 0 for s in starts)):
        raise ValueError('invalid case/window coordinates')
    complete = [s for s in starts if s+1 <= word_start and word_start+3 <= s+context_length+1]
    return dict(complete_target_presentations=len(complete),
                prefix_segment_and_targets_presentations=sum(s <= prefix_start for s in complete),
                identical_case_prefix_and_local_positions=sum(s == prefix_start for s in complete))


def analyze(root, steps):
    root = Path(root).resolve(strict=True)
    steps = list(steps)
    if not steps or any(type(s) is not int or s < 0 for s in steps) or len(set(steps)) != len(steps):
        raise ValueError('steps must be distinct nonnegative integers')
    source_paths = [Path(__file__), Path(paired_training.__file__), Path(paired_word_cases.__file__)]
    sources = [paired_word_cases._record(p) for p in source_paths]
    manifest_path = root/'manifest.json'
    manifest_record = paired_word_cases._record(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('format') != 'pluto-paired-corpus-training-v1':
        raise ValueError('unsupported experiment manifest')
    cases_path = root/'word_cases/cases.json'
    cases = json.loads(cases_path.read_text())
    if cases['manifest'] != manifest_record:
        raise ValueError('cases belong to a different manifest')
    used = {str(manifest_path): manifest_record,
            str(cases_path): paired_word_cases._record(cases_path)}
    inputs = {domain: paired_word_cases._load_input(
        manifest['inputs'][f'{domain}.training'], used) for domain in ('original','replacement')}
    original, replacement = inputs['original'], inputs['replacement']
    verified = paired_training.verify_alignment(original['text'], replacement['text'],
        original['tokens'], replacement['tokens'], original['offsets'], replacement['offsets'])
    if verified != manifest['alignment']['training']:
        raise ValueError('native corpus alignment differs from manifest')
    count = verified['token_count']
    context = cases['context_length']
    batch = manifest['flags']['batch_size']
    seed = manifest['flags']['seed']
    if any(type(v) is not int for v in (context,batch,seed)) or batch < 1:
        raise ValueError('invalid training geometry')
    sampling = manifest['sampling']
    replay = Path(sampling['prefix_path'])
    replay_record = paired_word_cases._record(replay)
    if replay_record['sha256'] != sampling['prefix_sha256']:
        raise ValueError('frozen sampler prefix changed')
    used[str(replay)] = replay_record
    binary = manifest['binaries']['replay_sampler']
    if paired_word_cases._record(binary['path']) != binary:
        raise ValueError('frozen sampler executable changed')
    used[binary['path']] = binary
    command = sampling['prefix_command']
    if command != [binary['path'],str(count),str(context),'10000',str(seed),'1']:
        raise ValueError('sampler command disagrees with training geometry')
    # This native executable is CPU-only. Rerun the exact frozen invocation,
    # rather than substituting NumPy's different RNG/distribution algorithm.
    repeated = subprocess.run(command, capture_output=True, check=True, timeout=30)
    if repeated.stdout != replay.read_bytes():
        raise ValueError('frozen native sampler no longer reproduces prefix')
    values = [int(v) for v in repeated.stdout.split()]
    if len(values) != 10001 or values[0] != seed or max(steps)*batch > 10000:
        raise ValueError('insufficient or malformed replay prefix')
    positions = [o['token_start'] for o in verified['occurrences']]
    selected = {}
    for case in cases['cases']:
        if case['kind'] != 'word' or case['split'] != 'training':
            continue
        index = case['occurrence_index']
        if (type(index) is not int or not 0 <= index < len(positions)
                or case['prefix']['token_end'] != positions[index]
                or case['target_source']['token_start'] != positions[index]):
            raise ValueError('case occurrence coordinate mismatch')
        geometry = (case['prefix']['token_start'], positions[index])
        if index in selected and selected[index] != geometry:
            raise ValueError('aliased case geometries disagree')
        selected[index] = geometry
    reports = []
    for step in sorted(steps):
        starts = values[1:1+step*batch]
        counts = exposure(starts,positions,token_count=count,context_length=context)
        complete = counts['complete_target_counts']
        partial = counts['partial_target_window_counts']
        reports.append(dict(step=step,sequence_windows=len(starts),target_token_presentations=len(starts)*context,
            complete_word_presentations=sum(complete),partial_word_window_presentations=sum(partial),
            distinct_fully_presented_occurrences=sum(n>0 for n in complete),
            distinct_partial_only_occurrences=sum(a==0 and b>0 for a,b in zip(complete,partial)),
            distinct_never_targeted_occurrences=sum(a==0 and b==0 for a,b in zip(complete,partial)),
            per_piece_presentations=np.asarray(counts['piece_target_counts'],dtype=np.int64).reshape(-1,3).sum(axis=0).tolist(),
            complete_presentation_histogram={str(k):v for k,v in sorted(Counter(complete).items())},
            occurrence_counts=counts,
            selected_training_cases=[dict(occurrence_index=i,word_start=word,prefix_start=prefix,
                **case_coverage(starts,prefix_start=prefix,word_start=word,context_length=context))
                for i,(prefix,word) in sorted(selected.items())]))
    for record in [*used.values(),*sources]:
        if paired_word_cases._record(record['path']) != record:
            raise ValueError('input/source changed during exposure analysis')
    return dict(format='pluto-paired-training-exposure-v1',complete=True,
        inference_performed=False,goal_completion_claimed=False,
        provenance=list(used.values()),sources=sources,native_replay_command=command,
        native_replay_returncode=repeated.returncode,native_replay_bytes_identical=True,
        batch_sequences=batch,context_length=context,training_word_occurrences=len(positions),reports=reports,
        limitations=['Conditional deterministic sampler replay, not a separately logged list of executed GPU batches.',
            'Step completion must be verified from the corresponding arm; replacement counts do not imply that arm has already run.',
            'Counts apply to corresponding corpus coordinates in each arm, not cross-domain prefixes seen in both arms.',
            'Containing a prefix segment permits earlier context and different position IDs; exact prefix equality is counted separately.',
            'Exposure does not establish memorization, causal storage, or complete-word probability.'])


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--step',type=int,action='append',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():raise FileExistsError(args.output)
    result=analyze(args.root,args.step)
    paired_word_cases._write_json(args.output,result)


if __name__=='__main__':main()
