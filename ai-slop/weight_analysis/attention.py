"""Extract *static token associations*, not attention or text, from OV weights.

For row-vector weights the score is E[source] @ WV @ WO @ E[target].T.
WV and WO are the input-value and output slices of one attention head. We
never evaluate the model, compute attention probabilities, read a corpus, or
use token frequency to choose candidates. This deliberately omits Q/K,
positions, biases, LayerNorm, the residual stream, and all contextual states.
It is therefore neither the head's actual contribution on text nor evidence
that a two-token association is a memorized training passage.

The default score is cosine(write, target_embedding), where write = E @ WV
@ WO. Raw dot products are retained too. Sources are chosen by the largest
write norms across the entire *logical* vocabulary. That selection is biased
towards strongly written directions; results do not estimate corpus coverage.
All vocabulary targets, including source==target, remain eligible by default.

Factorization avoids constructing a V-by-V matrix. Sources are selected from
E @ WV using WO @ WO.T; each target chunk is projected by WO.T. Peak working
memory is O(V * head_dim + source_count * chunk_size), in addition to mapped
weights. FP32 rounding can affect near-tied rankings; ties in computed scores
are resolved by token ID. No GPU library or device execution is involved.

Example (set BLAS limits before NumPy is imported):
  OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 python -m \
    weight_analysis.attention --checkpoint /path/to/step_N \
    --tokenizer-dir /path/to/gpt2 --output /tmp/ov_candidates.jsonl

The output has one JSON record per candidate and a separate .metadata.json
sidecar with checkpoint/source/tokenizer hashes. Neither file is overwritten.
--broken-head-control pairs WV[h] with WO[(h+1) % n_heads] within each block,
preserving each set of component matrices while breaking their learned pairing.
The control repeats the same norm-based source selection independently; it is
not a per-source paired comparison with the intact head.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

from weight_analysis.checkpoint import GPT2Checkpoint, sha256_file


def _top_indices(scores, count, ids=None):
    """Top finite values, descending score then ascending supplied ID.

    Partitioning finds the cutoff cheaply; explicitly choosing tied IDs avoids
    unspecified argpartition tie order. Invalid/zero-norm candidates are
    represented by -inf and do not appear in the result.
    """
    scores = np.asarray(scores)
    if ids is None:
        ids = np.arange(scores.size)
    finite = np.flatnonzero(np.isfinite(scores))
    count = min(count, finite.size)
    if count <= 0:
        return np.empty(0, dtype=np.int64)
    cutoff = np.partition(scores[finite], finite.size - count)[finite.size - count]
    above = finite[scores[finite] > cutoff]
    tied = finite[scores[finite] == cutoff]
    tied = tied[np.argsort(np.asarray(ids)[tied], kind='stable')[:count - above.size]]
    chosen = np.concatenate((above, tied))
    return chosen[np.lexsort((np.asarray(ids)[chosen], -scores[chosen]))]


def extract_head_edges(embedding, value_projection, output_projection, *,
                       source_count=32, top_k=4, chunk_size=4096,
                       score_mode='cosine', exclude_self=False):
    """Yield bounded, corpus-blind associations for one (possibly broken) head.

    Inputs are [vocab, d_model], [d_model, head_dim], [head_dim, d_model].
    Zero write vectors are never selected. In cosine mode zero target vectors
    are also excluded because their angle is undefined. No epsilon is added
    to disguise such undefined values. All input/output arithmetic is finite-
    checked. Source ranks reflect norm selection, not model probability.
    """
    for name, number in [('source_count', source_count), ('top_k', top_k),
                         ('chunk_size', chunk_size)]:
        if not isinstance(number, int) or isinstance(number, bool) or number <= 0:
            raise ValueError(f'{name} must be a positive integer')
    if score_mode not in ('raw', 'cosine'):
        raise ValueError('score_mode must be raw or cosine')
    embedding = np.asarray(embedding)
    value_projection = np.asarray(value_projection)
    output_projection = np.asarray(output_projection)
    if (embedding.ndim != 2 or value_projection.ndim != 2 or
            output_projection.ndim != 2 or not embedding.shape[0] or
            not embedding.shape[1] or not value_projection.shape[1] or
            value_projection.shape[0] != embedding.shape[1] or
            output_projection.shape !=
            (value_projection.shape[1], embedding.shape[1])):
        raise ValueError('expected E[V,D], WV[D,H], and WO[H,D]')
    if not all(np.isfinite(x).all() for x in
               (embedding, value_projection, output_projection)):
        raise ValueError('head inputs contain non-finite values')

    projected = embedding @ value_projection
    # ||a WO||^2 = a (WO WO.T) a.T. This avoids materializing V residual writes.
    gram = output_projection @ output_projection.T
    norm_squared = np.sum((projected @ gram) * projected, axis=1, dtype=np.float64)
    if not np.isfinite(norm_squared).all():
        raise ValueError('non-finite source write norms')
    # A tiny negative can arise from cancellation; it is not a real norm.
    selection_scores = np.where(norm_squared > 0, norm_squared, -np.inf)
    source_ids = _top_indices(selection_scores, source_count)
    if source_ids.size == 0:
        return
    source_values = projected[source_ids]
    writes = source_values @ output_projection
    write_norms = np.sqrt(np.sum(writes * writes, axis=1, dtype=np.float64))
    if not np.isfinite(write_norms).all():
        raise ValueError('non-finite selected source writes')
    valid_sources = write_norms > 0
    source_ids = source_ids[valid_sources]
    source_values = source_values[valid_sources]
    write_norms = write_norms[valid_sources]

    best_ids = [np.empty(0, dtype=np.int64) for _ in source_ids]
    best_scores = [np.empty(0, dtype=np.float64) for _ in source_ids]
    best_raw = [np.empty(0, dtype=np.float64) for _ in source_ids]
    for start in range(0, embedding.shape[0], chunk_size):
        targets = embedding[start:start + chunk_size]
        target_ids = np.arange(start, start + len(targets))
        # Associativity saves a factor d_model/head_dim in candidate scoring.
        raw = source_values @ (targets @ output_projection.T).T
        if not np.isfinite(raw).all():
            raise ValueError('non-finite vocabulary scores')
        if score_mode == 'cosine':
            target_norms = np.sqrt(np.sum(targets * targets, axis=1,
                                         dtype=np.float64))
            denominator = write_norms[:, None] * target_norms[None, :]
            scores = np.full(raw.shape, -np.inf, dtype=np.float64)
            np.divide(raw, denominator, out=scores, where=denominator > 0)
        else:
            scores = raw.astype(np.float64)
        for row, source in enumerate(source_ids):
            if exclude_self and start <= source < start + len(targets):
                scores[row, source - start] = -np.inf
            local = _top_indices(scores[row], top_k, target_ids)
            ids = np.concatenate((best_ids[row], target_ids[local]))
            values = np.concatenate((best_scores[row], scores[row, local]))
            raw_values = np.concatenate((best_raw[row], raw[row, local]))
            selected = _top_indices(values, top_k, ids)
            best_ids[row], best_scores[row], best_raw[row] = (
                ids[selected], values[selected], raw_values[selected])

    for row, source in enumerate(source_ids):
        for rank, (target, score, raw) in enumerate(zip(
                best_ids[row], best_scores[row], best_raw[row]), start=1):
            yield {
                'source': int(source), 'target': int(target),
                'source_norm_rank': row + 1, 'target_rank': rank,
                'source_write_norm': float(write_norms[row]),
                'score': float(score), 'raw_dot_product': float(raw),
            }


def _parse_blocks(value, n_layers):
    if value == 'all':
        return list(range(n_layers))
    try:
        blocks = [int(part) for part in value.split(',')]
    except ValueError as error:
        raise ValueError('--blocks must be all or comma-separated indices') from error
    if len(set(blocks)) != len(blocks) or any(b < 0 or b >= n_layers for b in blocks):
        raise ValueError('--blocks must be unique zero-based block indices')
    return blocks


def _head_provenance(checkpoint, block, value_head, output_head):
    """Give rectangular slices, not misleading contiguous byte intervals.

    QKV head columns are strided in the packed file: every residual row has
    its own slice. Shapes and half-open axis intervals are exact; byte offsets
    for an individual element follow the TensorSpec C-order description.
    """
    d, h = checkpoint.config.d_model, checkpoint.config.head_dim
    qkv = checkpoint.specs[f'blocks.{block}.attn.qkv.weight']
    output = checkpoint.specs[f'blocks.{block}.attn.output.weight']
    embedding = checkpoint.specs['token_embedding.weight']
    return {
        'block': block, 'value_head': value_head, 'output_head': output_head,
        'tensors': {
            'embedding': {'name': embedding.name, 'filename': embedding.filename,
                          'shape': list(embedding.shape),
                          'rows': [0, checkpoint.config.vocab_size]},
            'value_projection': {'name': qkv.name, 'filename': qkv.filename,
                                 'shape': list(qkv.shape), 'rows': [0, d],
                                 'columns': [2 * d + value_head * h,
                                             2 * d + (value_head + 1) * h]},
            'output_projection': {'name': output.name, 'filename': output.filename,
                                  'shape': list(output.shape),
                                  'rows': [output_head * h, (output_head + 1) * h],
                                  'columns': [0, d]},
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--tokenizer-dir', required=True)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--source-count', default=32, type=int)
    parser.add_argument('--top-k', default=4, type=int)
    parser.add_argument('--chunk-size', default=4096, type=int)
    parser.add_argument('--score-mode', choices=['raw', 'cosine'], default='cosine')
    parser.add_argument('--blocks', default='all', help='all or zero-based indices, e.g. 0,7')
    parser.add_argument('--exclude-self', action='store_true')
    parser.add_argument('--broken-head-control', action='store_true')
    args = parser.parse_args(argv)
    if min(args.source_count, args.top_k, args.chunk_size) <= 0:
        parser.error('source-count, top-k, and chunk-size must be positive')

    # Optional dependency is needed only for human-readable labels. Decoding a
    # candidate uses the supplied tokenizer dictionary, never an LLM or corpus.
    from tokenizers import Tokenizer
    tokenizer_path = Path(args.tokenizer_dir) / 'tokenizer.json'
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    checkpoint = GPT2Checkpoint(args.checkpoint, check_finite=True)
    if tokenizer.get_vocab_size() != checkpoint.config.vocab_size:
        parser.error('tokenizer vocabulary size differs from checkpoint configuration')
    try:
        blocks = _parse_blocks(args.blocks, checkpoint.config.n_layers)
    except ValueError as error:
        parser.error(str(error))
    if args.broken_head_control and checkpoint.config.n_heads < 2:
        parser.error('a head derangement requires at least two heads')
    sidecar = args.output.with_suffix(args.output.suffix + '.metadata.json')
    if args.output.exists() or sidecar.exists():
        parser.error('output or metadata sidecar already exists; choose a new path')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        'schema_version': 1, 'method': 'static_attention_ov',
        'checkpoint': checkpoint.provenance(hash_weights=True),
        'tokenizer_sha256': sha256_file(tokenizer_path),
        'extractor_sha256': sha256_file(__file__),
        'options': {key: str(value) if isinstance(value, Path) else value
                    for key, value in vars(args).items()},
        'input_contract': 'checkpoint weights plus tokenizer only; no corpus access',
        'source_selection': 'largest ||E[token] WV WO|| over logical vocabulary',
        'warning': ('Norm-selected token associations, not model predictions or '
                    'verified memorized text. Ignores attention routing, positions, '
                    'biases, LayerNorm, contextual states and other layers.'),
    }
    with sidecar.open('x') as stream:
        json.dump(metadata, stream, indent=2, allow_nan=False)
        stream.write('\n')
    count = 0
    with args.output.open('x') as stream:
        for block in blocks:
            for head in range(checkpoint.config.n_heads):
                output_head = ((head + 1) % checkpoint.config.n_heads
                               if args.broken_head_control else head)
                provenance = _head_provenance(checkpoint, block, head, output_head)
                provenance.update({
                    'metadata_file': sidecar.name, 'score_mode': args.score_mode,
                    'source_selection': metadata['source_selection'],
                    'broken_head_control': args.broken_head_control,
                })
                edges = extract_head_edges(
                    checkpoint.token_embedding, checkpoint.qkv(block, 'v', head),
                    checkpoint.attention_output_head(block, output_head),
                    source_count=args.source_count, top_k=args.top_k,
                    chunk_size=args.chunk_size, score_mode=args.score_mode,
                    exclude_self=args.exclude_self)
                for edge in edges:
                    token_ids = [edge['source'], edge['target']]
                    record = {
                        'candidate_id': (f'ov-b{block}-v{head}-o{output_head}-'
                                         f'{args.score_mode}-s{edge["source"]}-'
                                         f't{edge["target"]}'),
                        'method': 'static_attention_ov', 'token_ids': token_ids,
                        'token_strings': [tokenizer.id_to_token(i) for i in token_ids],
                        'decoded_display': tokenizer.decode(token_ids,
                                                             skip_special_tokens=False),
                        'provenance': dict(provenance, scores=edge),
                    }
                    stream.write(json.dumps(record, allow_nan=False) + '\n')
                    count += 1
                stream.flush()
                print(f'block {block}, WV head {head}, WO head {output_head}: '
                      f'{count} candidates written', file=sys.stderr, flush=True)
    print(f'Wrote {count} static associations to {args.output}', file=sys.stderr)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
