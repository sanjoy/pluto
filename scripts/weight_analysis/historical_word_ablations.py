"""Join saved native ablations into same-intervention word probabilities.

This is a CPU-only readout of the historical three-word experiment, NOT a new
GPU experiment or a result from the deterministic Exeunt/Nuveth training arms.
Each word's successive teacher-forced contexts must form the original sliding
autoregressive chain. Only then may we multiply the conditional probabilities
for its pieces under one *unchanged* weight intervention.

The recorded sampler used temperature 0.8. We instead evaluate temperature-one
FP64 softmax on all logical-vocabulary FP32 logits, excluding padding. Summed
negative log probabilities preserve tiny probabilities even when exp underflows.
There is no intervened boundary prediction, so a word's complete token sequence
is measured, not its termination or freely sampled counterfactual continuation.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess

import numpy as np

from . import checkpoint as checkpoint_module
from .checkpoint import GPT2Config, sha256_file, tensor_manifest


PRODUCER_SOURCE = 'scripts/weight_analysis/token_trace_probe.cc'


def record(path):
    """Record a regular file without accepting a symlink as its identity."""
    path = Path(path).absolute()
    if path.is_symlink() or not path.is_file() or path.resolve() != path:
        raise ValueError('expected a regular nonsymlink file: ' + str(path))
    return dict(path=str(path), bytes=path.stat().st_size, sha256=sha256_file(path))


def _json(path):
    return json.loads(Path(path).read_text())


def _index(records, key):
    result = {}
    for item in records:
        name = item[key]
        if (not isinstance(name, str) or not Path(name).is_absolute()
                or name in result or type(item['bytes']) is not int
                or item['bytes'] < 0
                or not re.fullmatch(r'[0-9a-f]{64}', item['sha256'])):
            raise ValueError('invalid or duplicate historical file identity')
        result[name] = item
    return result


def _local(directory, filename):
    if (not isinstance(filename, str) or not filename
            or Path(filename).name != filename or filename in ('.', '..')):
        raise ValueError('native artifact must use one local filename')
    return Path(directory) / filename


class Evidence:
    """Authenticate exactly the files used, then recheck them before returning.

    The supplied manifest is the provenance anchor, not a cryptographic assertion
    about an outside authority. Some old producer binaries have been rebuilt;
    their current bytes are deliberately not substituted for historical bytes.
    """

    def __init__(self, manifest_path):
        first = record(manifest_path)
        self.manifest = _json(first['path'])
        if self.manifest.get('complete') is not True:
            raise ValueError('historical evidence manifest is incomplete')
        self.known = _index(self.manifest['files'], 'original_path')
        self.used = {first['path']: first}

    def checked(self, path, expected=None):
        actual = record(path)
        expected = self.known[actual['path']] if expected is None else expected
        identity = expected.get('path', expected.get('original_path'))
        if (identity != actual['path']
                or (actual['bytes'], actual['sha256']) !=
                   (expected['bytes'], expected['sha256'])):
            raise ValueError('historical artifact hash mismatch: ' + actual['path'])
        old = self.used.get(actual['path'])
        if old is not None and old != actual:
            raise ValueError('historical input changed during analysis')
        self.used[actual['path']] = actual
        return Path(actual['path'])

    def recheck(self):
        for old in self.used.values():
            if record(old['path']) != old:
                raise ValueError('historical input changed during analysis')


def recover_producer_source(repository, revision, expected):
    """Recover the exact recorded producer from Git, not the dirty working tree.

    Requiring a full commit ID also keeps user-controlled revision expressions
    out of the Git argument. The source hash must match the historical plan.
    This does not rerun or authenticate a present-day executable.
    """
    if not isinstance(revision, str) or not re.fullmatch(r'[0-9a-f]{40}', revision):
        raise ValueError('historical producer needs a full Git commit ID')
    repository = Path(repository).resolve(strict=True)
    path = str(repository / PRODUCER_SOURCE)
    if expected['path'] != path:
        raise ValueError('producer source identity differs from repository')
    result = subprocess.run(
        ['git', '-C', str(repository), 'show', revision + ':' + PRODUCER_SOURCE],
        check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
    if result.returncode:
        raise ValueError('could not recover historical producer source from Git')
    digest = hashlib.sha256(result.stdout).hexdigest()
    if (len(result.stdout), digest) != (expected['bytes'], expected['sha256']):
        raise ValueError('Git producer source differs from historical plan')
    return dict(repository=str(repository), revision=revision,
                repository_relative_path=PRODUCER_SOURCE, bytes=len(result.stdout),
                sha256=digest, matched_historical_plan=True,
                current_worktree_source_equality_claimed=False)


def token_readout(logits, target, vocab_size):
    """A scalar independent oracle: FP64 log-sum-exp, no sampled RNG outcome."""
    values = np.asarray(logits)
    if (values.ndim != 1 or type(vocab_size) is not int or vocab_size < 2
            or len(values) < vocab_size or type(target) is not int
            or not 0 <= target < vocab_size or not np.isfinite(values).all()):
        raise ValueError('need finite logits and a valid logical-vocabulary target')
    row = values[:vocab_size].astype(np.float64)
    shifted = row - row.max()
    normalizer = math.fsum(map(float, np.exp(shifted)))
    nll = math.log(normalizer) - float(shifted[target])
    return dict(target_id=target, target_logit=float(row[target]),
                target_rank=1 + int(np.count_nonzero(row > row[target])) +
                            int(np.count_nonzero(row[:target] == row[target])),
                probability=math.exp(-nll), nll=nll, log_probability=-nll)


def sequence_readout(nlls, clean_nlls=None):
    """Separate initiating a word from completing its teacher-forced suffix.

    A single arm's conditional NLLs are added; we never multiply probabilities
    from different best/worst heads selected independently for each piece.
    Probabilities may round to 0 or 1; finite NLL is the authoritative quantity.
    """
    if (not nlls or any(not math.isfinite(x) or x < 0 for x in nlls)
            or (clean_nlls is not None and
                (len(clean_nlls) != len(nlls)
                 or any(not math.isfinite(x) or x < 0 for x in clean_nlls)))):
        raise ValueError('need compatible finite nonnegative token NLLs')
    result = {}
    for name, indices in (('first_piece', slice(0, 1)),
                          ('conditional_suffix', slice(1, None)),
                          ('joint', slice(None))):
        nll = math.fsum(nlls[indices])
        item = dict(token_count=len(nlls[indices]), nll=nll,
                    log_probability=-nll, probability=math.exp(-nll))
        if clean_nlls is not None:
            delta = nll - math.fsum(clean_nlls[indices])
            # Ratios can overflow although each NLL remains perfectly usable.
            # JSON null plus the log ratio is preferable to nonstandard Infinity.
            try:
                ratio = math.exp(-delta)
            except OverflowError:
                ratio = None
            item.update(delta_nll=delta, log_probability_ratio=-delta,
                        probability_ratio=ratio)
        result[name] = item
    return result


def expected_arms(config):
    result = {}
    for block in range(config.n_layers):
        for kind in ('attention_branch', 'mlp_branch'):
            name = f'ablation.block{block}.{kind}'
            result[name] = dict(kind=kind, block=block)
        for head in range(config.n_heads):
            name = f'ablation.block{block}.head{head}'
            result[name] = dict(kind='attention_head', block=block,
                                head_or_neuron=head)
    return result


def validate_arms(arms, config):
    """Require the complete canonical same-intervention set at every piece."""
    expected = expected_arms(config)
    if not isinstance(arms, list) or len(arms) != len(expected):
        raise ValueError('incomplete native ablation set')
    actual = {}
    for arm in arms:
        name = arm['name']
        if name not in expected or name in actual:
            raise ValueError('duplicate or unexpected native ablation')
        if (any(arm.get(k) != v for k, v in expected[name].items())
                or type(arm.get('block')) is not int
                or (arm['kind'] == 'attention_head'
                    and type(arm.get('head_or_neuron')) is not int)
                or (arm['kind'] != 'attention_head' and 'head_or_neuron' in arm)
                or arm.get('scale') != 0 or type(arm.get('scale')) not in (int, float)
                or arm.get('shape') != [1, config.padded_vocab_size]
                or arm.get('role') != 'selected_row'
                or arm.get('restoration_verified_bytes') is not True):
            raise ValueError('native ablation semantics or restoration mismatch')
        _local(Path('/unused'), arm['logits_file'])
        actual[name] = arm
    if len({a['logits_file'] for a in arms}) != len(arms):
        raise ValueError('different native ablations share a logits file')
    return actual


def validate_context(metadata, run, all_ids, prompt_tokens, config, previous=None):
    """Check original generation coordinates and the sliding teacher-forced chain."""
    event = run['event']
    step = run['step']
    if type(step) is not int or not 0 <= step < len(all_ids) - prompt_tokens:
        raise ValueError('invalid original generation step')
    absolute = prompt_tokens + step
    start = max(0, absolute - config.context_length)
    ids = all_ids[start:absolute]
    if (run.get('interventions') is not True
            or event.get('index') != step or event.get('step') != step
            or event.get('absolute_token_index') != absolute
            or event.get('context_start') != start
            or event.get('context_length') != len(ids)
            or event.get('output_row') != len(ids) - 1
            or event.get('token_id') != all_ids[absolute]
            or event.get('logits_byte_offset') != step * config.vocab_size * 4
            or run.get('context_token_ids') != ids
            or metadata.get('token_ids') != ids
            or metadata.get('target_id') != all_ids[absolute]
            or metadata.get('prompt_rows') != len(ids)
            or metadata.get('selected_row') != len(ids) - 1):
        raise ValueError('native trace differs from the original generation context')
    if previous is not None:
        old_ids, old_target, old_step = previous
        if step != old_step + 1 or ids != (old_ids + [old_target])[-config.context_length:]:
            raise ValueError('word pieces do not form a sliding teacher-forced chain')
    return ids, all_ids[absolute], step


def analyze(evidence_manifest, run_directory, repository, *, config=GPT2Config()):
    """Authenticate raw measurements, then join all arms without running a model.

    The default layout is the exact eight-block historical recipe. The explicit
    config override exists only for small CPU fixtures; it is recorded in output.
    """
    evidence = Evidence(evidence_manifest)
    source_records = [record(__file__), record(checkpoint_module.__file__)]
    root = Path(run_directory).resolve(strict=True)
    plan = _json(evidence.checked(root / 'plan.json'))
    if (plan['output_directory'] != str(root)
            or any(plan[k] != getattr(config, k) for k in
                   ('vocab_size', 'padded_vocab_size', 'n_layers', 'n_heads'))):
        raise ValueError('historical plan has a different model or output directory')
    inputs = _index(plan['inputs'], 'path')
    producer_path = str(Path(repository).resolve(strict=True) / PRODUCER_SOURCE)
    producer = recover_producer_source(repository, evidence.manifest['source_commit_sha'],
                                       inputs[producer_path])
    checkpoint = Path(plan['checkpoint_directory'])
    specs = tensor_manifest(config)
    weight_names = {p.name for p in checkpoint.iterdir() if p.name.startswith('weight_')}
    if weight_names != {s.filename for s in specs}:
        raise ValueError('historical checkpoint has a different weight inventory')
    for spec in specs:
        path = checkpoint / spec.filename
        evidence.checked(path, inputs[str(path)])
        if path.stat().st_size != spec.nbytes or not np.isfinite(
                np.memmap(path, dtype='<f4', mode='r')).all():
            raise ValueError('historical checkpoint has wrong-sized or nonfinite weights')

    generation_dir = Path(plan['generation_directory'])
    path = generation_dir / 'metadata.json'
    generation = _json(evidence.checked(path, inputs[str(path)]))
    initial, generated = generation['initial_token_ids'], generation['generated_token_ids']
    all_ids = generation['all_token_ids']
    if (generation.get('complete') is not True or not initial
            or generation.get('checkpoint_directory') != str(checkpoint)
            or generation.get('context_length') != config.context_length
            or generation.get('vocab_size') != config.vocab_size
            or generation.get('padded_vocab_size') != config.padded_vocab_size
            or generation.get('steps') != len(generated)
            or all_ids != initial + generated
            or any(type(x) is not int or not 0 <= x < config.vocab_size for x in all_ids)):
        raise ValueError('invalid original generation identity or token stream')
    path = _local(generation_dir, generation['events_file'])
    events = [json.loads(line) for line in evidence.checked(path, inputs[str(path)]).read_text().splitlines()]
    if len(events) != len(generated):
        raise ValueError('original generation event count mismatch')
    generation_path = _local(generation_dir, generation['logits_file'])
    if str(generation_path) != plan['generation_logits']:
        raise ValueError('different original generation logits paths')
    evidence.checked(generation_path, inputs[str(generation_path)])
    if generation_path.stat().st_size != len(generated) * config.vocab_size * 4:
        raise ValueError('wrong original generation logits size')
    original_logits = np.memmap(generation_path, dtype='<f4', mode='r').reshape(-1, config.vocab_size)
    runs = {run['step']: run for run in plan['runs']}
    if len(runs) != len(plan['runs']):
        raise ValueError('duplicate original generation run')
    words, seen_words, seen_steps = [], set(), set()

    def raw_logits(native, filename):
        path = evidence.checked(_local(native, filename))
        raw = path.read_bytes()
        if len(raw) != 4 * config.padded_vocab_size:
            raise ValueError('wrong native logit size')
        values = np.frombuffer(raw, dtype='<f4')
        if not np.isfinite(values).all():
            raise ValueError('native logit row is nonfinite')
        return values

    for selection in plan['selections']:
        word, first, end = selection['word'], selection['first'], selection['end']
        if (not isinstance(word, str) or not word or word in seen_words
                or type(first) is not int or type(end) is not int
                or not 0 <= first < end <= len(generated) or end - first < 2
                or any(step in seen_steps for step in range(first, end))):
            raise ValueError('invalid, duplicate, or overlapping multi-token word selection')
        seen_words.add(word)
        seen_steps.update(range(first, end))
        pieces, previous = [], None
        by_arm = {name: [] for name in expected_arms(config)}
        for step in range(first, end):
            run, native = runs[step], root / f'trace_step_{step}'
            # The archived plan adds audit fields to the event. Every original
            # event field must nevertheless equal the authenticated JSONL event.
            if any(run['event'].get(k) != v for k, v in events[step].items()):
                raise ValueError('plan event differs from original generation event')
            metadata = _json(evidence.checked(native / 'metadata.json'))
            required_checks = ('clean_replay_logits_byte_equal', 'alternate_padding_logits_byte_equal',
                               'final_lens_logits_byte_equal', 'attention_residual_replay_byte_equal',
                               'mlp_residual_replay_byte_equal')
            if (metadata.get('complete') is not True or metadata.get('probe_kind') != 'token_trace'
                    or metadata.get('checkpoint_directory') != str(checkpoint)
                    or metadata.get('vocab_size') != config.vocab_size
                    or metadata.get('padded_vocab_size') != config.padded_vocab_size
                    or metadata.get('context_length') != config.context_length
                    or metadata.get('checkpoint_unique_weight_count') != len(specs)
                    or metadata.get('byte_order') != 'little'
                    or metadata.get('retokenized') is not False
                    or any(metadata.get(k) != 0 for k in
                           ('optimizer_steps', 'backward_calls', 'checkpoint_writes'))
                    or any(metadata.get('checks', {}).get(k) is not True for k in required_checks)):
                raise ValueError('incomplete native trace or incompatible checkpoint/parity metadata')
            previous = validate_context(metadata, run, all_ids, len(initial), config, previous)
            target = previous[1]
            spec = metadata['files']['logits']
            if spec['dtype'] != 'float32' or spec['shape'] != [1, config.padded_vocab_size]:
                raise ValueError('native baseline logits schema mismatch')
            clean = raw_logits(native, spec['file'])
            for key in ('clean_replay', 'alternate_padding'):
                other = raw_logits(native, metadata['parity_files'][key])
                if clean.tobytes() != other.tobytes():
                    raise ValueError('native full-padded clean/replay/padding logits differ')
            if clean[:config.vocab_size].tobytes() != original_logits[step].tobytes():
                raise ValueError('native clean logits differ from original generation')
            pieces.append(dict(generation_step=step, target_id=target,
                               piece_bytes_hex=run['event']['piece_hex'],
                               absolute_target_index=run['event']['absolute_token_index'],
                               context_start=run['event']['context_start'],
                               context_length=len(previous[0]), selected_row=len(previous[0])-1,
                               clean=token_readout(clean, target, config.vocab_size)))
            for name, arm in validate_arms(metadata['interventions'], config).items():
                value = raw_logits(native, arm['logits_file'])
                by_arm[name].append(dict(generation_step=step,
                    logits_file=str(native / arm['logits_file']),
                    **token_readout(value, target, config.vocab_size)))
        # Verify the label against the exact recorded token bytes. Preserve the
        # leading whitespace in the output token sequence rather than silently
        # suggesting the probability concerns a different bare-token spelling.
        sequence_bytes = b''.join(bytes.fromhex(p['piece_bytes_hex']) for p in pieces)
        if sequence_bytes.lstrip(b' ') != word.encode('utf-8'):
            raise ValueError('recorded token pieces do not spell the selected word')
        clean_nlls = [p['clean']['nll'] for p in pieces]
        joined = []
        for name, identity in expected_arms(config).items():
            values = by_arm[name]
            if [p['generation_step'] for p in values] != list(range(first, end)):
                raise ValueError('cannot join an intervention missing a word piece')
            for i, p in enumerate(values):
                p['delta_nll'] = p['nll'] - clean_nlls[i]
            joined.append(dict(name=name, **identity, scale=0, per_piece=values,
                               **sequence_readout([p['nll'] for p in values], clean_nlls)))
        heads = [arm for arm in joined if arm['kind'] == 'attention_head']
        worst_joint = max(heads, key=lambda arm: arm['joint']['delta_nll'])
        worst_suffix = max(heads, key=lambda arm: arm['conditional_suffix']['delta_nll'])
        words.append(dict(word=word, generation_steps=list(range(first, end)), pieces=pieces,
            token_sequence_bytes_hex=sequence_bytes.hex(),
            clean=sequence_readout(clean_nlls), interventions=joined,
            screen=dict(largest_joint_head_damage=worst_joint['name'],
                        largest_conditional_suffix_head_damage=worst_suffix['name'],
                        minimum_conditional_suffix_probability_over_all_heads=
                            min(arm['conditional_suffix']['probability'] for arm in heads))))
    if not words:
        raise ValueError('no multi-token word selections')
    comparisons = []
    for index, (name, identity) in enumerate(expected_arms(config).items()):
        comparisons.append(dict(name=name, **identity, words={word['word']: {
            key: word['interventions'][index][key] for key in
            ('first_piece', 'conditional_suffix', 'joint')} for word in words}))
    evidence.recheck()
    for old in source_records:
        if record(old['path']) != old:
            raise ValueError('analysis source changed during analysis')
    return dict(format='pluto-historical-word-ablations-v1', complete=True,
        goal_completion_claimed=False, new_deterministic_experiment_result=False,
        checkpoint_directory=str(checkpoint), config=asdict(config), temperature=1,
        probability_arithmetic='FP64 log-sum-exp on saved native FP32 logits; all logical vocabulary entries; no sampling',
        words=words, cross_word_comparisons=comparisons,
        counts=dict(words=len(words), token_predictions=len(seen_steps),
                    interventions_per_word=len(expected_arms(config)),
                    native_piece_ablations=len(seen_steps) * len(expected_arms(config))),
        checks=dict(sliding_teacher_forced_chains_verified=True,
                    same_intervention_joined_across_all_pieces=True,
                    full_padded_clean_replay_and_alternate_padding_byte_parity=True,
                    clean_logical_logits_byte_equal_original_generation=True,
                    all_checkpoint_weights_hashed_and_finite=True,
                    all_used_files_rehashed_at_end=True),
        intervention_scope=dict(
            attention_head='Zero 64 selected head-output projection rows (head_dim in explicit toy fixtures) at all query positions; output bias unchanged; rerun full model.',
            attention_branch='Zero output projection and bias at all positions; rerun full model.',
            mlp_branch='Zero output projection and bias at all positions; rerun full model.'),
        historical_producer_source=producer, provenance=list(evidence.used.values()),
        analysis_sources=source_records,
        limitations=[
            'Historical selected contexts from one checkpoint, not the new paired-training arms or a corpus-wide result.',
            'Token-sequence probabilities condition later pieces on the specified earlier pieces; counterfactual free generation was not sampled.',
            'Following word-boundary predictions were not intervened on; terminated-word probabilities are not measured.',
            'Head interventions remove all-query output paths, not a single source-token value or attention edge.',
            'Largest effects and cross-word contrasts are post hoc screens over unmatched contexts, not evidence of lexical specificity.',
            'Small singleton effects do not prove joint dispensability; interactions and redundant paths were not tested here.',
            'Probabilities can round to zero or one; finite NLL and log ratios are retained. An overflowing probability ratio is null.',
            'The manifest is a supplied provenance anchor. Git recovers exact producer source, but no current executable is claimed identical and no GPU experiment is rerun.',
            'These data show functional effects of interventions, not a unique location that stores a word.'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('evidence-manifest', 'run-directory', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--repository', type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    root = args.run_directory.resolve(strict=True)
    if root == args.output.resolve() or root in args.output.resolve().parents:
        raise ValueError('output must be outside historical run directory')
    result = analyze(args.evidence_manifest, args.run_directory, args.repository)
    for input_record in result['provenance'] + result['analysis_sources']:
        if args.output.resolve() == Path(input_record['path']):
            raise ValueError('output overlaps an input')
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print('Joined', result['counts']['native_piece_ablations'],
          'historical native ablations into', len(result['words']), 'word readouts at T=1.')


if __name__ == '__main__':
    main()
