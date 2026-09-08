"""Corpus-blind signed aggregate MLP operators, NOT contextual inference.

For row-vector weights form A = W1 @ diag(g) @ W2, then score source/target
tokens with E[source] @ A @ E[target].T. Summing all signed neuron terms before
ranking preserves cancellations that a strongest-neuron or positive-only
probe discards. Default g=1 is a purely linear surrogate. --gate=bias_gelu
instead uses the analytic derivative of the recipe's tanh-GELU at the fixed
input-bias point b1, i.e. at zero residual input. Neither gate is evaluated on
text, and neither is a contextual transformer gate. Positions, attention,
LayerNorm, output biases, residual additions and other blocks are omitted.

The ranking score is cosine(E[source] @ A, E[target]); raw dot products and
both norms are also saved. Sources are the largest write norms over ALL real
vocabulary tokens, never corpus tokens or frequencies. This selection is
biased and is not a coverage estimate. Targets are searched over the full
logical vocabulary in chunks. The residual-space A is only d-by-d: no V-by-V
matrix is materialized. Computation is FP64 analysis, not GPU-bitwise replay.

Pairs retain each block separately. Paths use a bounded graph formed from
positive, non-self edges, taking the strongest cosine edge across blocks for
each token pair. Composing static edges is NOT a transformer forward pass,
nor proof of passage memory. Every edge names ALL contributing MLP weights.
The optional control pairs key j with value (j-1) mod m, preserving key/value
marginals but destroying their alignment. It independently reselects sources.

Examples:
  OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 python -m scripts.weight_analysis.aggregate \
    --checkpoint /path/to/step_N --tokenizer-dir /path/to/gpt2 \
    --output /tmp/aggregate.jsonl --include-control

The public neuron_contributions() function returns every signed additive term
for exact explanations/replay; its sum is the raw operator score. Its callers
must not mistake a top-neuron subset for the full sum without a residual.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

from .attention import _parse_blocks, _top_indices
from .checkpoint import GPT2Checkpoint, sha256_file
from .mlp import decode_paths


def gate_vector(bias, mode='identity'):
    """Fixed analytic gates; never reads data or computes contextual states.

    bias_gelu differentiates 0.5*x*(1+tanh(c*(x+a*x**3))) using c=0.7978845608
    and a=0.044715 from src/llm/layers/gelu.cc. The derivative can be negative;
    clipping it to [0,1] would change the stated aggregate and is not done.
    """
    bias = np.asarray(bias, dtype=np.float64)
    if bias.ndim != 1 or not bias.size or not np.isfinite(bias).all():
        raise ValueError('expected a finite nonempty bias vector')
    if mode == 'identity':
        return np.ones_like(bias)
    if mode != 'bias_gelu':
        raise ValueError('gate must be identity or bias_gelu')
    c, a = 0.7978845608, 0.044715
    tanh_inner = np.tanh(c * (bias + a * bias**3))
    result = (0.5 * (1 + tanh_inner) + 0.5 * bias * (1 - tanh_inner**2) *
              c * (1 + 3 * a * bias**2))
    if not np.isfinite(result).all():
        raise ValueError('nonfinite analytic GELU derivative')
    return result


def _components(w1, w2, gates, pairing):
    w1, w2, gates = (np.asarray(x, dtype=np.float64) for x in (w1, w2, gates))
    if (w1.ndim != 2 or not all(w1.shape) or w2.ndim != 2 or
            w2.shape != (w1.shape[1], w1.shape[0]) or gates.shape != (w1.shape[1],)):
        raise ValueError('expected W1[D,M], W2[M,D], gates[M]')
    if not all(np.isfinite(x).all() for x in (w1, w2, gates)):
        raise ValueError('aggregate weights/gates must be finite')
    if pairing is not None:
        pairing = np.asarray(pairing)
        if (pairing.shape != gates.shape or not np.issubdtype(pairing.dtype, np.integer) or
                not np.array_equal(np.sort(pairing), np.arange(len(gates)))):
            raise ValueError('pairing must be a permutation of all neuron indices')
        w2 = w2[pairing]
    return w1, w2, gates


def aggregate_operator(w1, w2, gates, pairing=None):
    """Return FP64 W1 diag(gates) W2 (optionally with permuted W2 rows)."""
    w1, w2, gates = _components(w1, w2, gates, pairing)
    result = (w1 * gates[None, :]) @ w2
    if not np.isfinite(result).all():
        raise ValueError('nonfinite aggregate operator')
    return result


def neuron_contributions(source, target, w1, w2, gates, pairing=None):
    """Every signed term of source W1 diag(g) W2 target.T, as an M-vector.

    The vector index is the KEY neuron index. Under a broken pairing, term j
    names value neuron pairing[j]. This is an algebraic certificate for the
    static probe, not an attribution of the actual language-model prediction.
    """
    w1, w2, gates = _components(w1, w2, gates, pairing)
    source, target = np.asarray(source, dtype=np.float64), np.asarray(target, dtype=np.float64)
    if (source.shape != (w1.shape[0],) or target.shape != source.shape or
            not np.isfinite(source).all() or not np.isfinite(target).all()):
        raise ValueError('source/target must be finite D-vectors')
    result = (source @ w1) * gates * (w2 @ target)
    if not np.isfinite(result).all():
        raise ValueError('nonfinite neuron contributions')
    return result


def extract_operator_edges(embedding, operator, *, source_count=256, top_k=4,
                           chunk_size=4096):
    """Yield cosine-ranked edges with bounded target-score working storage.

    Source write directions are materialized as V-by-D (never V-by-V). Source
    and target zero vectors are excluded, not assigned artificial directions.
    Negative top scores remain visible in pair records; only positive scores
    are used by the separate graph-building stage. Ties use ascending IDs.
    """
    for name, value in [('source_count', source_count), ('top_k', top_k),
                        ('chunk_size', chunk_size)]:
        if type(value) is not int or value <= 0:
            raise ValueError(f'{name} must be a positive integer')
    embedding, operator = (np.asarray(x, dtype=np.float64) for x in (embedding, operator))
    if (embedding.ndim != 2 or not all(embedding.shape) or
            operator.shape != (embedding.shape[1], embedding.shape[1]) or
            not np.isfinite(embedding).all() or not np.isfinite(operator).all()):
        raise ValueError('expected finite E[V,D], A[D,D]')
    writes = embedding @ operator
    norms = np.linalg.norm(writes, axis=1)
    target_norms = np.linalg.norm(embedding, axis=1)
    if not np.isfinite(norms).all() or not np.isfinite(target_norms).all():
        raise ValueError('nonfinite write/embedding norms')
    source_ids = _top_indices(np.where(norms > 0, norms, -np.inf), source_count)
    selected_writes = writes[source_ids]
    best_ids = [np.empty(0, dtype=np.int64) for _ in source_ids]
    best_scores = [np.empty(0) for _ in source_ids]
    best_raw = [np.empty(0) for _ in source_ids]
    for start in range(0, len(embedding), chunk_size):
        targets = embedding[start:start + chunk_size]
        ids = np.arange(start, start + len(targets))
        raw = selected_writes @ targets.T
        denominator = norms[source_ids, None] * target_norms[None, start:start + len(targets)]
        scores = np.full(raw.shape, -np.inf)
        np.divide(raw, denominator, out=scores, where=denominator > 0)
        if not np.isfinite(raw).all() or not np.isfinite(scores[denominator > 0]).all():
            raise ValueError('nonfinite vocabulary scores')
        for row in range(len(source_ids)):
            local = _top_indices(scores[row], top_k, ids)
            candidate_ids = np.concatenate((best_ids[row], ids[local]))
            candidate_scores = np.concatenate((best_scores[row], scores[row, local]))
            candidate_raw = np.concatenate((best_raw[row], raw[row, local]))
            selected = _top_indices(candidate_scores, top_k, candidate_ids)
            best_ids[row], best_scores[row], best_raw[row] = (
                candidate_ids[selected], candidate_scores[selected], candidate_raw[selected])
    for row, source in enumerate(source_ids):
        for rank, (target, score, raw) in enumerate(zip(
                best_ids[row], best_scores[row], best_raw[row]), start=1):
            yield {'source': int(source), 'target': int(target), 'score': float(score),
                   'raw_dot_product': float(raw), 'write_norm': float(norms[source]),
                   'target_norm': float(target_norms[target]),
                   'source_norm_rank': row + 1, 'target_rank': rank}


def _provenance(checkpoint, block, gate, broken):
    names = [f'blocks.{block}.mlp.input.weight', f'blocks.{block}.mlp.output.weight']
    if gate == 'bias_gelu':
        names.append(f'blocks.{block}.mlp.input.bias')
    tensors = []
    for name in names:
        spec = checkpoint.specs[name]
        tensors.append({'name': name, 'filename': spec.filename, 'shape': list(spec.shape),
                        'byte_range': list(spec.byte_range), 'all_elements_contribute': True})
    return {
        'block': block, 'tensors': tensors, 'gate': gate,
        'gate_definition': ('all ones' if gate == 'identity' else
                            'analytic tanh-GELU derivative at b1 (zero residual input)'),
        'pairing': ('value_index=(key_index-1)%d_ff' if broken else 'value_index=key_index'),
        'neuron_count': checkpoint.config.d_ff, 'broken_pairing_control': broken,
        'signed_sum': 'sum_j (E[source] W1[:,j]) g[j] (W2[pairing[j],:] E[target].T)',
        'embedding': {'name': 'token_embedding.weight', 'filename': 'weight_0.bin',
                      'row_bytes': checkpoint.config.d_model * 4,
                      'rows_supplied_by_edge_token_ids': True},
        'source_selection': 'top aggregate-write norms over all logical vocabulary tokens',
        'score_definition': 'raw_dot_product/(write_norm*target_norm)',
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--tokenizer-dir', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--gate', choices=['identity', 'bias_gelu'], default='identity')
    parser.add_argument('--source-count', type=int, default=256)
    parser.add_argument('--top-k', type=int, default=4)
    parser.add_argument('--chunk-size', type=int, default=4096)
    parser.add_argument('--path-length', type=int, default=8)
    parser.add_argument('--path-starts', type=int, default=128)
    parser.add_argument('--blocks', default='all')
    parser.add_argument('--include-control', action='store_true')
    args = parser.parse_args(argv)
    metadata_path = args.output.with_suffix(args.output.suffix + '.metadata.json')
    if args.output.exists() or metadata_path.exists():
        raise FileExistsError('refusing to overwrite extraction artifacts')
    if (min(args.source_count, args.top_k, args.chunk_size) <= 0 or
            args.path_length < 2 or args.path_starts < 0):
        parser.error('invalid extraction/path limits')
    checkpoint = GPT2Checkpoint(args.checkpoint, check_finite=True)
    blocks = _parse_blocks(args.blocks, checkpoint.config.n_layers)
    tokenizer_path = args.tokenizer_dir / 'tokenizer.json'
    vocabulary = json.loads(tokenizer_path.read_text())['model']['vocab']
    if (len(vocabulary) != checkpoint.config.vocab_size or
            set(vocabulary.values()) != set(range(checkpoint.config.vocab_size))):
        raise ValueError('tokenizer vocabulary does not match checkpoint')
    if args.include_control and checkpoint.config.d_ff < 2:
        raise ValueError('broken pairing requires at least two neurons')
    labels = {index: label for label, index in vocabulary.items()}
    counts, graphs = Counter(), {}
    methods = [(f'aggregate_{args.gate}', False)]
    if args.include_control:
        methods.append((f'broken_aggregate_{args.gate}', True))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Freeze hashes before writing candidates. No corpus has been read by this
    # program, and every option determining selection is recorded verbatim.
    metadata = {
        'schema_version': 1, 'stage': 'corpus_blind_extraction',
        'checkpoint': checkpoint.provenance(hash_weights=True),
        'parameters': {key: str(value) if isinstance(value, Path) else value
                       for key, value in vars(args).items()},
        'tokenizer_sha256': sha256_file(tokenizer_path),
        'extractor_sources': {name: sha256_file(Path(__file__).with_name(name))
                              for name in ['aggregate.py', 'attention.py', 'mlp.py', 'checkpoint.py']},
        'gelu_source_sha256': sha256_file(Path(__file__).resolve().parents[2] /
                                          'src/llm/layers/gelu.cc'),
        'numerics': 'float64 static analysis, not GPU-bitwise replay',
        'limitations': [
            'No contextual gates, attention, LayerNorm, positions, or model inference.',
            'Norm-selected sources are biased; pairs/paths are not corpus-coverage estimates.',
            'All signed neurons contribute; no individual neuron is asserted to own a passage.',
            'Paths compose strongest positive edges across blocks, not transformer execution.',
            'Control reselects sources; it is not a fixed-source paired comparison.'],
    }
    with args.output.open('x') as stream:
        def emit(record):
            record['vocabulary_labels'] = [labels[token] for token in record['token_ids']]
            stream.write(json.dumps(record, allow_nan=False) + '\n')
            counts[record['method']] += 1

        for block in blocks:
            w1 = checkpoint[f'blocks.{block}.mlp.input.weight']
            w2 = checkpoint[f'blocks.{block}.mlp.output.weight']
            gates = gate_vector(checkpoint[f'blocks.{block}.mlp.input.bias'], args.gate)
            for method, broken in methods:
                pairing = np.roll(np.arange(checkpoint.config.d_ff), 1) if broken else None
                operator = aggregate_operator(w1, w2, gates, pairing)
                base = _provenance(checkpoint, block, args.gate, broken)
                graph = graphs.setdefault(method, {})
                for edge in extract_operator_edges(checkpoint.token_embedding, operator,
                        source_count=args.source_count, top_k=args.top_k,
                        chunk_size=args.chunk_size):
                    pair = (edge['source'], edge['target'])
                    provenance = dict(base, edge_token_ids=list(pair), scores=edge)
                    emit({'candidate_id': f'{method}:b{block}:s{pair[0]}:t{pair[1]}',
                          'method': method + '_pair', 'token_ids': list(pair),
                          'score': edge['score'], 'provenance': provenance})
                    if edge['score'] > 0 and (pair not in graph or
                                               edge['score'] > graph[pair]['score']):
                        graph[pair] = {'score': edge['score'], 'provenance': provenance}
                stream.flush()
                print(f'aggregate block {block}, {method}: {sum(counts.values())} pairs',
                      file=sys.stderr, flush=True)
        for method, graph in graphs.items():
            for index, path in enumerate(decode_paths(graph, length=args.path_length,
                                                      starts=args.path_starts, beam_width=4)):
                emit(dict(path, candidate_id=f'{method}:path{index}', method=method + '_path'))
    metadata['method_counts'] = dict(counts)
    metadata['candidate_sha256'] = sha256_file(args.output)
    with metadata_path.open('x') as stream:
        json.dump(metadata, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(counts)), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
