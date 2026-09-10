"""Read-only whole-branch damage diagnostic for archived native word traces.

This reuses recorded GPU interventions, never executes a model. In particular,
it does not present an older model as an arm of the new deterministic paired
experiment. Probabilities here are newly computed at temperature ONE from the
complete logical-vocabulary FP32 logits. They are not a reconstruction of the
historical temperature-0.8 sampler (whose arithmetic subsequently changed).

The residual norms describe the unmodified forward at the selected position.
The ablation zeros the entire block-0 MLP output projection AND bias at every
position, allowing all later computations to change. Neither a large write norm
nor increased output entropy proves a gain-only mechanism or lexical specificity.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np

from .checkpoint import sha256_file, tensor_manifest


def record(path):
    path = Path(path).resolve(strict=True)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=sha256_file(path))


def distribution(logits):
    """Stable temperature-1 softmax in FP64; no sampling or GPU equivalence claim."""
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 1 or len(logits) < 2 or not np.isfinite(logits).all():
        raise ValueError('need a finite full-vocabulary logit row')
    shifted = logits - logits.max()
    logp = shifted - math.log(float(np.exp(shifted).sum()))
    p = np.exp(logp)
    nonzero = p > 0
    return p, logp, -float(np.sum(p[nonzero] * logp[nonzero]))


def compare_logits(clean, ablated, target):
    clean, ablated = np.asarray(clean), np.asarray(ablated)
    if clean.shape != ablated.shape or type(target) is not int or not 0 <= target < clean.size:
        raise ValueError('incompatible logit rows or target')
    results = {}
    for name, values in (('clean', clean), ('ablated', ablated)):
        p, logp, entropy = distribution(values)
        top = np.argsort(-values, kind='stable')[:5]
        results[name] = dict(target_probability=float(p[target]), target_log_probability=float(logp[target]),
            entropy_nats=entropy, effective_vocabulary_size=math.exp(entropy),
            target_rank=1 + int(np.count_nonzero(values > values[target])) +
                        int(np.count_nonzero(values[:target] == values[target])),
            top5=[dict(id=int(i), probability=float(p[i])) for i in top])
    results['delta_log_probability'] = results['ablated']['target_log_probability'] - results['clean']['target_log_probability']
    return results


def residual_metrics(before, write, after):
    values = [np.asarray(x, dtype=np.float64) for x in (before, write, after)]
    if (any(v.ndim != 1 or not v.size or not np.isfinite(v).all() for v in values)
            or len({v.shape for v in values}) != 1):
        raise ValueError('need equal finite residual vectors')
    norms = {name: float(np.linalg.norm(value)) for name, value in zip(('before', 'write', 'after'), values)}
    centered = {name: float(np.linalg.norm(value-value.mean())) for name, value in zip(('before', 'write', 'after'), values)}
    return dict(l2=norms, centered_l2=centered,
                write_to_before_l2_ratio=norms['write']/norms['before'] if norms['before'] else None,
                centered_write_to_before_ratio=centered['write']/centered['before'] if centered['before'] else None,
                residual_add_rounding_l2=float(np.linalg.norm(values[2]-values[0]-values[1])))


def analyze(evidence_manifest, run_directory):
    manifest_record = record(evidence_manifest)
    manifest = json.loads(Path(evidence_manifest).read_text())
    if manifest.get('complete') is not True:
        raise ValueError('historical evidence manifest is incomplete')
    known = {item['original_path']: item for item in manifest['files']}
    if len(known) != len(manifest['files']):
        raise ValueError('duplicate historical evidence identity')
    used = {manifest_record['path']: manifest_record}

    def checked(path, expected=None):
        path = Path(path)
        if path.is_symlink() or not path.is_file():
            raise ValueError('historical artifact must be a regular nonsymlink file')
        actual = record(path)
        expected = known[str(path)] if expected is None else expected
        if (actual['bytes'], actual['sha256']) != (expected['bytes'], expected['sha256']):
            raise ValueError('historical artifact hash mismatch: '+str(path))
        used[actual['path']] = actual
        return path

    root = Path(run_directory).resolve(strict=True)
    plan = json.loads(checked(root/'plan.json').read_text())
    if (plan['vocab_size'], plan['padded_vocab_size'], plan['n_layers'], plan['n_heads']) != (50257, 50272, 8, 8):
        raise ValueError('unexpected historical model layout')
    # Historical source/binary paths may since have been rebuilt. Validate the
    # immutable measured arrays and the CURRENT checkpoint bytes, not an invented
    # equality to today's executable. The old plan retains producer provenance.
    checkpoint = Path(plan['checkpoint_directory'])
    inputs = {item['path']: item for item in plan['inputs']}
    for spec in tensor_manifest():
        path = checkpoint/spec.filename
        checked(path, inputs[str(path)])
    generation_path = Path(plan['generation_logits'])
    checked(generation_path, inputs[str(generation_path)])
    generation = np.memmap(generation_path, dtype='<f4', mode='r').reshape(-1, 50257)
    runs = {run['step']: run for run in plan['runs']}
    events, writes = [], []
    for selection in plan['selections']:
        for step in range(selection['first'], selection['end']):
            run = runs[step]
            native = root/f'trace_step_{step}'
            metadata = json.loads(checked(native/'metadata.json').read_text())
            if (metadata.get('complete') is not True or metadata.get('probe_kind') != 'token_trace'
                    or metadata['checkpoint_directory'] != str(checkpoint)
                    or metadata['token_ids'] != run['context_token_ids']
                    or metadata['target_id'] != run['event']['token_id']
                    or metadata['selected_row'] != len(run['context_token_ids'])-1):
                raise ValueError('native trace disagrees with historical event')
            arm = next(a for a in metadata['interventions'] if a['name']=='ablation.block0.mlp_branch')
            if arm['scale'] != 0 or arm['restoration_verified_bytes'] is not True or arm['shape'] != [1, 50272]:
                raise ValueError('missing restored whole-MLP ablation')

            def logits(filename):
                path = checked(native/filename)
                if path.stat().st_size != 4*50272:
                    raise ValueError('wrong native logit shape')
                return np.fromfile(path, dtype='<f4')[:50257]

            clean = logits(metadata['files']['logits']['file'])
            ablated = logits(arm['logits_file'])
            replay = logits(metadata['parity_files']['clean_replay'])
            if clean.tobytes() != replay.tobytes() or clean.tobytes() != generation[step].tobytes():
                raise ValueError('native clean logits differ from replay/original generation')
            vectors = []
            for name in ('blocks.0.after_attention', 'blocks.0.mlp_projected', 'blocks.0.after_mlp'):
                item = metadata['files'][name]
                if item['dtype'] != 'bf16' or item['shape'] != [metadata['prompt_rows'],512]:
                    raise ValueError('wrong native activation shape/type')
                path = checked(native/item['file'])
                bits = np.fromfile(path, dtype='<u2').reshape(item['shape'])[metadata['selected_row']]
                vectors.append((bits.astype(np.uint32)<<16).view(np.float32))
            writes.append(vectors[1].astype(np.float64))
            events.append(dict(word=selection['word'], generation_step=step,
                target_id=metadata['target_id'], piece_bytes_hex=run['event']['piece_hex'],
                selected_row=metadata['selected_row'], prefix_length=metadata['prompt_rows'],
                probabilities=compare_logits(clean, ablated, metadata['target_id']),
                residual=residual_metrics(*vectors)))
    if not events:
        raise ValueError('no selected historical word events')
    centered = np.asarray(writes)
    centered -= centered.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered,axis=1)
    cosine = np.divide(centered@centered.T, norms[:,None]*norms[None,:],
                       out=np.full((len(events),len(events)),np.nan),
                       where=norms[:,None]*norms[None,:] != 0)
    for item in list(used.values()):
        if record(item['path']) != item:
            raise ValueError('historical input changed during diagnostic')
    return dict(format='pluto-historical-b0-diagnostic-v1', complete=True,
        goal_completion_claimed=False, new_deterministic_experiment_result=False,
        checkpoint_directory=str(checkpoint), temperature=1,
        probability_arithmetic='FP64 softmax of saved native FP32 logical-vocabulary logits; no sampling',
        events=events, centered_write_cosines=[[float(x) if math.isfinite(x) else None for x in row] for row in cosine],
        provenance=list(used.values()), source=record(__file__),
        limitations=['Seven selected events from one older model, not a corpus-wide or paired-training result.',
            'Whole branch including bias is removed at ALL positions, not just the selected word position.',
            'Large write norms do not establish a gain-only mechanism; direction, normalization, and downstream responses matter.',
            'Different output distributions and increased entropy do not locate a unique lexical memory.',
            'Current checkpoint bytes, historical raw artifacts and replay rows are rechecked; the old executable was not rerun.'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('evidence-manifest','run-directory','output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args=parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    if args.run_directory.resolve() in args.output.resolve().parents:
        raise ValueError('output must be outside historical run')
    result=analyze(args.evidence_manifest,args.run_directory)
    with args.output.open('x') as stream:
        json.dump(result,stream,indent=2,sort_keys=True,allow_nan=False)
        stream.write('\n')


if __name__=='__main__':
    main()
