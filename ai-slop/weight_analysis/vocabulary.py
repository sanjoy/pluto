"""Recover token-importance rankings from weights, without corpus or inference.

The output offset beta dot E[t] is the input-independent term of the tied
prediction head: logit[t] = (gamma * normalize(hidden)) dot E[t] + beta dot E[t].
Only the last term is computed here, not any hidden state or probability.
Geometric embedding norms and distances are comparison heuristics. None of
these scores is a token count or evidence of an ordered passage by itself.

All logical token IDs receive a score; padded physical rows never participate
in norms, the centroid, or sorting. Ranking ties are broken by token ID. The
full ranking artifact must be frozen before a separate verifier sees text.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from .checkpoint import GPT2Checkpoint, sha256_file


PROTOCOL_PATH = (Path(__file__).resolve().parents[2] /
                 'ai-slop/research/weight_memorization/VOCABULARY_PROTOCOL.md')


def weight_scores(embedding, beta):
    """Return three [vocabulary] score arrays, using FP64 mathematical weights.

    Norms are not invariant to arbitrary changes of residual coordinates.
    The centroid distance removes only a common row translation. The offset
    has an exact algebraic interpretation in this checkpoint's coordinates;
    the discarded contextual term can still dominate the actual prediction.
    Finite inputs whose arithmetic overflows are rejected, not ranked.
    """
    embedding = np.asarray(embedding, dtype=np.float64)
    beta = np.asarray(beta, dtype=np.float64)
    if (embedding.ndim != 2 or not all(embedding.shape) or
            beta.shape != (embedding.shape[1],)):
        raise ValueError('expected nonempty E[vocabulary,width] and beta[width]')
    if not np.isfinite(embedding).all() or not np.isfinite(beta).all():
        raise ValueError('weights must be finite')
    with np.errstate(over='ignore', invalid='ignore'):
        centered = embedding - embedding.mean(axis=0, keepdims=True)
        scores = {
            'embedding_norm': np.linalg.norm(embedding, axis=1),
            'centroid_distance': np.linalg.norm(centered, axis=1),
            'output_offset': embedding @ beta,
        }
    if not all(np.isfinite(values).all() for values in scores.values()):
        raise ValueError('nonfinite score arithmetic')
    return scores


def ranking(method, scores):
    """Retain exact scores and a deterministic complete token-ID permutation."""
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError('scores must be a nonempty finite vector')
    ids = np.lexsort((np.arange(len(scores)), -scores))
    return {'method': method, 'token_ids': ids.tolist(),
            'scores': scores[ids].tolist()}


def extract(checkpoint, baseline):
    """Two independent sets of scores; the baseline is NOT initialization.

    Row byte ranges are reconstructed as [id*row_bytes,(id+1)*row_bytes).
    output_offset additionally uses every byte of the shared final norm bias;
    centroid_distance additionally uses every logical embedding row. Merely
    addressing the selected row would omit real dependencies of those scores.
    """
    if checkpoint.config != baseline.config:
        raise ValueError('checkpoint and baseline architectures must match')
    results, provenance = [], {}
    for name, source in [('final', checkpoint), ('early', baseline)]:
        scores = weight_scores(source.token_embedding, source['final_norm.bias'])
        results.extend(ranking(f'{name}:{method}', values)
                       for method, values in scores.items())
        provenance[name] = source.provenance(hash_weights=True)
    embedding = checkpoint.specs['token_embedding.weight']
    beta = checkpoint.specs['final_norm.bias']
    return {
        'schema_version': 1,
        'stage': 'weight_only_vocabulary_rankings_not_model_inference',
        'vocab_size': checkpoint.config.vocab_size,
        'rankings': results,
        'checkpoint': provenance,
        'score_definitions': {
            'embedding_norm': 'Euclidean norm of logical embedding row',
            'centroid_distance': 'Euclidean distance from mean of all logical embedding rows',
            'output_offset': 'embedding row dot final LayerNorm beta; no contextual term',
        },
        'sorting': 'descending score, ascending token ID on ties',
        'arithmetic': 'FP64; BF16 rounding in model execution is not emulated',
        'weight_dependencies': {
            'embedding_file': embedding.filename,
            'row_bytes': 4 * checkpoint.config.d_model,
            'row_range_formula': '[token_id*row_bytes,(token_id+1)*row_bytes)',
            'centroid_logical_byte_range': [0, 4 * checkpoint.config.vocab_size * checkpoint.config.d_model],
            'output_offset_shared_beta_file': beta.filename,
            'output_offset_shared_beta_byte_range': list(beta.byte_range),
        },
        'protocol': {'file': str(PROTOCOL_PATH), 'sha256': sha256_file(PROTOCOL_PATH)},
        'source_sha256': {name: sha256_file(Path(__file__).with_name(name))
                          for name in ('vocabulary.py', 'checkpoint.py')},
        'limitations': [
            'No corpus or frequency information enters extraction.',
            'Rankings are not token counts, membership proofs, or recovered passages.',
            'The early checkpoint has already trained; it is not random initialization.',
            'Bare checkpoint files do not authenticate historical corpus or model source.',
            'A tied embedding receives gradients as both input and output dictionary.',
        ],
    }


def write_report(path, report):
    """Write once. Existing regular files and symlinks are never replaced."""
    encoded = json.dumps(report, indent=2, allow_nan=False) + '\n'
    with Path(path).open('x') as stream:
        stream.write(encoded)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--baseline', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError('refusing to overwrite vocabulary rankings')
    final = GPT2Checkpoint(args.checkpoint, check_finite=True)
    early = GPT2Checkpoint(args.baseline, check_finite=True)
    report = extract(final, early)
    write_report(args.output, report)
    print(json.dumps({'output': str(args.output), 'sha256': sha256_file(args.output),
                      'vocab_size': report['vocab_size'], 'rankings': len(report['rankings'])}))


if __name__ == '__main__':
    main()
