"""Corpus-blind, static QK-by-OV three-token interaction hypotheses.

For previous token b, current token a, and proposed next token c, define

  r_h(a,b) = (E[a] WQ[h] + bQ[h]) . ((E[b] - E[a]) WK[h]) / sqrt(head_dim)
  write(a,b) = 1/4 sum_h r_h(a,b) (E[b] - E[a]) WV[h] WO[h].

The destination score is cosine(write(a,b), E[c]). This is a STATIC weight
contraction. It is exactly the derivative at t=0 of two-position attention's
output when both attention scores are multiplied by t. It is NOT the complete
attention output: the uniform-routing zeroth-order term is absent. Query bias
is retained; key and value biases cancel in the differences. Output bias has
zero derivative. No softmax, LayerNorm, GELU, model calls, prompts, or training
corpus enter this tool. At the trained scale t=1 the linearization need not be
accurate, particularly for sharply selective heads. Positions, upstream states,
later layers, and residual/MLP contributions are deliberately absent. Block0
is the default because its pre-normalization inputs connect directly to E;
using later blocks makes the token-input substitution even more speculative.

Unlike independent pair chaining, a graph edge is (b,a)->(a,c): it jointly
depends on both preceding tokens and preserves signed cancellation across all
heads. A path can use ONLY recorded overlapping triples. It is still an
unverified candidate, not a recovered passage or a model prediction.

Default selection budget, fixed before inspecting any results:
  128 current tokens with largest joint query norm over the full vocabulary;
  8 previous tokens/current with largest norm of the SUMMED interaction write,
    searching the full vocabulary (not the sum of individual head norms);
  4 destinations/pair by cosine over the full vocabulary;
  at most 256 paths, up to 12 tokens, beam width 4, token repetition limit 2.
These are maxima: undefined zero-norm directions are never emitted. This is
an exact search within this declared norm-selected subset, not a globally
top-scoring search over all V^3 triples or an estimate of corpus coverage.

--broken-routing-control cyclically pairs QK[h] with intact OV[(h+1)%H].
It preserves every QK and OV operator, all marginal parameter sets, and the
current-token selection; previous-token selection can change. The same limits
apply in final, early, and broken runs. One cyclic shift is a diagnostic, not
a calibrated statistical null. Self transitions are retained in triple
records when defined but excluded from paths to avoid trivial copying chains.

Memory is O(V * heads * head_dim + chunk_size * d_model) for previous search,
plus O(current_count * previous_count * chunk_size) for destination search.
The norm of the signed sum requires the complete output projection; with the
default GPT-2 dimensions previous selection performs about 3.4 TFLOPs on CPU.
Set OPENBLAS_NUM_THREADS=8 before launching this program to bound CPU usage.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

import numpy as np

from .checkpoint import GPT2Checkpoint, sha256_file


def _positive_int(name, value):
    if type(value) is not int or value <= 0:
        raise ValueError(f'{name} must be a positive integer')


def _top(scores, count, ids=None):
    """Finite scores descending, breaking all ties by ascending token ID."""
    scores = np.asarray(scores)
    ids = np.arange(len(scores)) if ids is None else np.asarray(ids)
    eligible = np.flatnonzero(np.isfinite(scores))
    count = min(count, len(eligible))
    if not count:
        return np.empty(0, dtype=np.int64)
    threshold = np.partition(scores[eligible], len(eligible) - count)[len(eligible) - count]
    above = eligible[scores[eligible] > threshold]
    tied = eligible[scores[eligible] == threshold]
    tied = tied[np.argsort(ids[tied], kind='stable')[:count - len(above)]]
    selected = np.concatenate((above, tied))
    return selected[np.lexsort((ids[selected], -scores[selected]))]


def _squared_norms(rows):
    # Promote before multiplication, not just during the reduction. This also
    # avoids FP32 squaring overflow for otherwise finite parameter fixtures.
    result = np.sum(np.square(rows, dtype=np.float64), axis=1)
    if not np.isfinite(result).all():
        raise ValueError('non-finite squared norms')
    return result


class TrigramProbe:
    """Cached weight products; no contextual hidden state or model execution.

    Shapes: E[V,D], Q/K/V[H,D,C], O[H,C,D], query_bias[H,C]. The floating
    dtype is preserved (production uses FP32; finite-difference tests FP64).
    OV pairing maps routing head index to a complete value/output head pair.
    """

    def __init__(self, embedding, query, key, value, output, query_bias=None,
                 ov_pairing=None):
        embedding, query, key, value, output = (
            np.asarray(x) for x in (embedding, query, key, value, output))
        if (embedding.ndim != 2 or not all(embedding.shape) or query.ndim != 3
                or not all(query.shape) or key.shape != query.shape
                or value.shape != query.shape or query.shape[1] != embedding.shape[1]
                or output.shape != (query.shape[0], query.shape[2], embedding.shape[1])):
            raise ValueError('expected E[V,D], Q/K/V[H,D,C], O[H,C,D]')
        self.heads, self.width, self.head_dim = query.shape
        if query_bias is None:
            query_bias = np.zeros((self.heads, self.head_dim), dtype=query.dtype)
        query_bias = np.asarray(query_bias)
        if query_bias.shape != (self.heads, self.head_dim):
            raise ValueError('expected query_bias[H,C]')
        arrays = (embedding, query, key, value, output, query_bias)
        if any(not np.issubdtype(x.dtype, np.floating) or not np.isfinite(x).all()
               for x in arrays):
            raise ValueError('weights must be finite floating arrays')
        if ov_pairing is None:
            ov_pairing = np.arange(self.heads)
        ov_pairing = np.asarray(ov_pairing)
        if (ov_pairing.shape != (self.heads,) or
                not np.issubdtype(ov_pairing.dtype, np.integer) or
                sorted(ov_pairing.tolist()) != list(range(self.heads))):
            raise ValueError('ov_pairing must be a permutation of head indices')
        self.ov_pairing = ov_pairing.astype(np.int64)
        self.embedding = embedding
        self.query_bias = query_bias
        self.queries = (embedding @ np.concatenate(list(query), axis=1)).reshape(
            len(embedding), self.heads, self.head_dim) + query_bias
        self.keys = (embedding @ np.concatenate(list(key), axis=1)).reshape(
            len(embedding), self.heads, self.head_dim)
        self.values = (embedding @ np.concatenate(list(value[self.ov_pairing]), axis=1)).reshape(
            len(embedding), self.heads, self.head_dim)
        self.output = np.concatenate(list(output[self.ov_pairing]), axis=0)
        if not all(np.isfinite(x).all() for x in
                   (self.queries, self.keys, self.values, self.output)):
            raise ValueError('non-finite projected weights')
        self.query_norm_squared = _squared_norms(self.queries.reshape(len(embedding), -1))
        self.token_norms = np.sqrt(_squared_norms(embedding))

    def interaction_writes(self, current, previous_ids):
        """Signed [previous_count,D] derivatives, summing heads BEFORE ranking."""
        if type(current) not in (int, np.int64, np.int32) or not 0 <= current < len(self.embedding):
            raise ValueError('current token outside vocabulary')
        previous_ids = np.asarray(previous_ids)
        if (previous_ids.ndim != 1 or not np.issubdtype(previous_ids.dtype, np.integer)
                or np.any(previous_ids < 0) or np.any(previous_ids >= len(self.embedding))):
            raise ValueError('previous token IDs outside vocabulary')
        if not len(previous_ids):
            return np.empty((0, self.width), dtype=self.embedding.dtype)
        differences = self.keys[previous_ids] - self.keys[current]
        routing = np.sum(differences * self.queries[current], axis=2)
        routing /= 4 * np.sqrt(self.head_dim)
        weighted = (self.values[previous_ids] - self.values[current]) * routing[:, :, None]
        writes = weighted.reshape(len(previous_ids), -1) @ self.output
        if not np.isfinite(writes).all():
            raise ValueError('non-finite interaction writes')
        return writes

    def select_previous(self, current_count=128, previous_count=8, chunk_size=2048,
                        progress=None):
        """Select current/previous pairs independently of destination tokens.

        Query selection is invariant to OV pairing. A previous token equal to
        current has exactly zero interaction by the equation, not a heuristic
        mask. All other zero writes, including exact head cancellation, are
        excluded as well. Each returned tuple is (previous,current,write,norm,
        query_rank,previous_rank), with ranks one-based.
        """
        for name, value in [('current_count', current_count),
                            ('previous_count', previous_count), ('chunk_size', chunk_size)]:
            _positive_int(name, value)
        query_scores = np.where(self.query_norm_squared > 0,
                                self.query_norm_squared, -np.inf)
        currents = _top(query_scores, current_count)
        selected = []
        for query_rank, current in enumerate(currents, start=1):
            best_ids = np.empty(0, dtype=np.int64)
            best_norms = np.empty(0, dtype=np.float64)
            best_writes = np.empty((0, self.width), dtype=self.embedding.dtype)
            for start in range(0, len(self.embedding), chunk_size):
                ids = np.arange(start, min(start + chunk_size, len(self.embedding)))
                writes = self.interaction_writes(current, ids)
                norms = _squared_norms(writes)
                scores = np.where(norms > 0, norms, -np.inf)
                local = _top(scores, previous_count, ids)
                candidate_ids = np.concatenate((best_ids, ids[local]))
                candidate_norms = np.concatenate((best_norms, norms[local]))
                candidate_writes = np.concatenate((best_writes, writes[local]), axis=0)
                chosen = _top(candidate_norms, previous_count, candidate_ids)
                best_ids = candidate_ids[chosen]
                best_norms = candidate_norms[chosen]
                best_writes = candidate_writes[chosen]
            for previous_rank, (previous, norm, write) in enumerate(
                    zip(best_ids, best_norms, best_writes), start=1):
                selected.append((int(previous), int(current), write, float(np.sqrt(norm)),
                                 query_rank, previous_rank))
            if progress is not None:
                progress(query_rank, len(currents))
        return selected

    def destinations(self, selected_pairs, top_k=4, chunk_size=2048):
        """Full-vocabulary cosine search for the selected joint interaction writes."""
        _positive_int('top_k', top_k)
        _positive_int('chunk_size', chunk_size)
        if not selected_pairs:
            return
        writes = np.stack([pair[2] for pair in selected_pairs])
        write_norms = np.array([pair[3] for pair in selected_pairs])
        best_ids = [np.empty(0, dtype=np.int64) for _ in selected_pairs]
        best_scores = [np.empty(0) for _ in selected_pairs]
        best_raw = [np.empty(0) for _ in selected_pairs]
        for start in range(0, len(self.embedding), chunk_size):
            ids = np.arange(start, min(start + chunk_size, len(self.embedding)))
            raw = writes @ self.embedding[ids].T
            if not np.isfinite(raw).all():
                raise ValueError('non-finite destination projections')
            denominators = write_norms[:, None] * self.token_norms[ids][None, :]
            scores = np.full(raw.shape, -np.inf)
            np.divide(raw, denominators, out=scores, where=denominators > 0)
            for index in range(len(selected_pairs)):
                local = _top(scores[index], top_k, ids)
                candidate_ids = np.concatenate((best_ids[index], ids[local]))
                candidate_scores = np.concatenate((best_scores[index], scores[index, local]))
                candidate_raw = np.concatenate((best_raw[index], raw[index, local]))
                chosen = _top(candidate_scores, top_k, candidate_ids)
                best_ids[index] = candidate_ids[chosen]
                best_scores[index] = candidate_scores[chosen]
                best_raw[index] = candidate_raw[chosen]
        for index, pair in enumerate(selected_pairs):
            previous, current, _, write_norm, query_rank, previous_rank = pair
            for destination_rank, (destination, score, raw) in enumerate(
                    zip(best_ids[index], best_scores[index], best_raw[index]), start=1):
                yield {
                    'token_ids': [previous, current, int(destination)],
                    'score': float(score), 'raw_dot_product': float(raw),
                    'interaction_write_norm': write_norm,
                    'current_query_norm_rank': query_rank,
                    'previous_joint_write_norm_rank': previous_rank,
                    'destination_cosine_rank': destination_rank,
                }


def decode_trigram_paths(records, *, length=12, starts=256, beam_width=4):
    """Overlap triples exactly, rather than flattening them into pairwise edges.

    Starts are pairs ordered by their best outgoing score. At each extension,
    keep at most beam_width branches per start. Choose longest finished path,
    then highest mean cosine, then lexicographic token IDs. Emit only lengths
    >=4 since triples already have their own records. Consecutive equal tokens
    and a third occurrence of any token are forbidden only in this path stage.
    A dead end ends its path; it can never jump to a merely similar context.
    """
    for name, value in [('length', length), ('beam_width', beam_width)]:
        _positive_int(name, value)
    if length < 4 or type(starts) is not int or starts < 0:
        raise ValueError('path length must be >=4 and starts must be nonnegative')
    unique = {}
    for record in records:
        tokens = tuple(record['token_ids'])
        if len(tokens) != 3 or not np.isfinite(record['score']):
            raise ValueError('path edges must be finite-scored triples')
        if tokens[0] == tokens[1] or tokens[1] == tokens[2]:
            continue
        if tokens not in unique or record['score'] > unique[tokens]['score']:
            unique[tokens] = record
    adjacency = defaultdict(list)
    for tokens, record in unique.items():
        adjacency[tokens[:2]].append(record)
    for records_for_pair in adjacency.values():
        records_for_pair.sort(key=lambda r: (-r['score'], r['token_ids'][2], r['candidate_id']))
    initial = sorted(adjacency, key=lambda pair: (-adjacency[pair][0]['score'], pair))[:starts]
    for pair in initial:
        beam = [(pair, 0.0, [])]
        terminal = []
        for _ in range(length - 2):
            extended = []
            for tokens, score, provenance in beam:
                outgoing = [record for record in adjacency.get(tokens[-2:], [])
                            if tokens.count(record['token_ids'][2]) < 2][:beam_width]
                if not outgoing:
                    terminal.append((tokens, score, provenance))
                for record in outgoing:
                    extended.append((tokens + (record['token_ids'][2],), score + record['score'],
                                     provenance + [record['candidate_id']]))
            if not extended:
                beam = []
                break
            beam = sorted(extended, key=lambda entry: (-entry[1], entry[0]))[:beam_width]
        choices = [entry for entry in terminal + beam if len(entry[0]) >= 4]
        if choices:
            tokens, score, provenance = min(
                choices, key=lambda entry: (-len(entry[0]), -entry[1] / (len(entry[0]) - 2), entry[0]))
            yield {'token_ids': list(tokens), 'score': score / (len(tokens) - 2),
                   'provenance': {'trigram_candidate_ids': provenance,
                                  'search': 'exact overlapping recorded triples; no model calls'}}


def weight_provenance(checkpoint, block, ov_pairing):
    """Explicit strided matrix rectangles and physical shared embedding bytes.

    Each triple's score needs three embedding rows and all listed QK/OV head
    slices. Its selection rank additionally depends on the entire vocabulary.
    QKV matrices are [D,3D], so head columns are not contiguous file regions.
    """
    d, c = checkpoint.config.d_model, checkpoint.config.head_dim
    prefix = f'blocks.{block}'
    qkv = checkpoint.specs[prefix + '.attn.qkv.weight']
    bias = checkpoint.specs[prefix + '.attn.qkv.bias']
    output = checkpoint.specs[prefix + '.attn.output.weight']
    embedding = checkpoint.specs['token_embedding.weight']

    def rectangle(spec, row_start, row_end, col_start, col_end):
        return {'tensor': spec.name, 'filename': spec.filename,
                'rows': [row_start, row_end], 'columns': [col_start, col_end],
                'offset_bytes': spec.element_byte_range(row_start, col_start)[0],
                'row_stride_bytes': spec.shape[1] * 4, 'column_stride_bytes': 4}

    heads = []
    for route, content in enumerate(ov_pairing):
        content = int(content)
        heads.append({
            'routing_head': route, 'content_head': content,
            'query': rectangle(qkv, 0, d, route * c, (route + 1) * c),
            'key': rectangle(qkv, 0, d, d + route * c, d + (route + 1) * c),
            'query_bias': {'tensor': bias.name, 'filename': bias.filename,
                           'byte_range': [route * c * 4, (route + 1) * c * 4]},
            'value': rectangle(qkv, 0, d, 2 * d + content * c, 2 * d + (content + 1) * c),
            'output': rectangle(output, content * c, (content + 1) * c, 0, d),
        })
    return {'block': block, 'head_slices': heads,
            'embedding': {'tensor': embedding.name, 'filename': embedding.filename,
                          'logical_shape': [checkpoint.config.vocab_size, d],
                          'logical_byte_range': [0, checkpoint.config.vocab_size * d * 4],
                          'token_row_stride_bytes': d * 4},
            'score_reads': 'three embedding rows plus all head_slices',
            'selection_reads': 'full logical embedding vocabulary plus all head_slices'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--tokenizer-dir', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--block', type=int, default=0)
    parser.add_argument('--current-count', type=int, default=128)
    parser.add_argument('--previous-count', type=int, default=8)
    parser.add_argument('--top-k', type=int, default=4)
    parser.add_argument('--chunk-size', type=int, default=2048)
    parser.add_argument('--path-length', type=int, default=12)
    parser.add_argument('--path-starts', type=int, default=256)
    parser.add_argument('--beam-width', type=int, default=4)
    parser.add_argument('--broken-routing-control', action='store_true')
    args = parser.parse_args(argv)
    if min(args.current_count, args.previous_count, args.top_k,
           args.chunk_size, args.beam_width) <= 0:
        parser.error('selection and beam limits must be positive')
    if args.path_length < 4 or args.path_starts < 0:
        parser.error('path-length must be >=4 and path-starts must be nonnegative')
    sidecar = args.output.with_suffix(args.output.suffix + '.metadata.json')
    if args.output.exists() or sidecar.exists():
        raise FileExistsError('refusing to overwrite extraction output or metadata')
    checkpoint = GPT2Checkpoint(args.checkpoint, check_finite=True)
    config = checkpoint.config
    if not 0 <= args.block < config.n_layers:
        parser.error('block index outside checkpoint')
    if args.broken_routing_control and config.n_heads < 2:
        parser.error('broken routing control requires at least two heads')
    tokenizer_path = args.tokenizer_dir / 'tokenizer.json'
    vocabulary = json.loads(tokenizer_path.read_text())['model']['vocab']
    if (len(vocabulary) != config.vocab_size or
            set(vocabulary.values()) != set(range(config.vocab_size))):
        raise ValueError('tokenizer vocabulary must match logical checkpoint vocabulary')
    labels = {index: text for text, index in vocabulary.items()}
    pairing = np.arange(config.n_heads)
    if args.broken_routing_control:
        pairing = (pairing + 1) % config.n_heads
    projections = {kind: np.stack([checkpoint.qkv(args.block, kind, head)
                                  for head in range(config.n_heads)])
                   for kind in ('q', 'k', 'v')}
    output = np.stack([checkpoint.attention_output_head(args.block, head)
                       for head in range(config.n_heads)])
    query_bias = np.stack([checkpoint.qkv_bias(args.block, 'q', head)
                          for head in range(config.n_heads)])
    probe = TrigramProbe(checkpoint.token_embedding, projections['q'], projections['k'],
                         projections['v'], output, query_bias, pairing)

    def progress(count, total):
        if count % 8 == 0 or count == total:
            print(f'selected previous tokens for query {count}/{total}',
                  file=sys.stderr, flush=True)

    selected = probe.select_previous(args.current_count, args.previous_count,
                                      args.chunk_size, progress)
    method = 'broken_qkov_trigram' if args.broken_routing_control else 'static_qkov_trigram'
    records = []
    for index, record in enumerate(probe.destinations(selected, args.top_k, args.chunk_size)):
        record.update(candidate_id=f'{method}:block{args.block}:triple{index}', method=method,
                      vocabulary_labels=[labels[token] for token in record['token_ids']],
                      provenance={'metadata_file': sidecar.name, 'block': args.block,
                                  'token_embedding_row_offsets_bytes':
                                      [token * config.d_model * 4 for token in record['token_ids']],
                                  'ov_pairing': pairing.tolist()})
        records.append(record)
    paths = list(decode_trigram_paths(records, length=args.path_length,
                                      starts=args.path_starts, beam_width=args.beam_width))
    for index, path in enumerate(paths):
        path.update(candidate_id=f'{method}:block{args.block}:path{index}',
                    method=method + '_path',
                    vocabulary_labels=[labels[token] for token in path['token_ids']])
        path['provenance']['metadata_file'] = sidecar.name
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        for record in records + paths:
            stream.write(json.dumps(record, allow_nan=False) + '\n')
    counts = Counter(record['method'] for record in records + paths)
    metadata = {
        'schema_version': 1, 'stage': 'corpus_blind_extraction',
        'parameters': {name: str(value) if isinstance(value, Path) else value
                       for name, value in vars(args).items()},
        'method_counts': dict(counts),
        'selected_pair_count': len(selected),
        'selected_current_token_ids': list(dict.fromkeys(pair[1] for pair in selected)),
        'checkpoint': checkpoint.provenance(hash_weights=True),
        'tokenizer_sha256': sha256_file(tokenizer_path),
        'candidate_sha256': sha256_file(args.output),
        'extractor_sha256': sha256_file(__file__),
        'checkpoint_loader_sha256': sha256_file(Path(__file__).with_name('checkpoint.py')),
        'weight_provenance': weight_provenance(checkpoint, args.block, pairing),
        'equation': 'sum_h (((E[a]WQ[h]+bQ[h]) dot ((E[b]-E[a])WK[h])) '
                    '* ((E[b]-E[a])WV[p(h)]WO[p(h)])) / (4 sqrt(head_dim))',
        'selection': 'current joint Q norm; previous norm of signed sum; destination cosine',
        'limitations': [
            'A first derivative at uniform routing t=0, not full attention or predictions at t=1.',
            'No corpus, softmax, LayerNorm, GELU, positions, upstream states, or later layers.',
            'Norm-selected bounded subset; not global V^3 search or corpus coverage.',
            'Cyclic QK-to-intact-OV mismatch is one diagnostic, not a statistical null.',
            'Previous selection and graph topology can change under broken pairing.',
            'Overlapping triples do not prove passage memorization without independent verification.',
        ],
    }
    with sidecar.open('x') as stream:
        json.dump(metadata, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'method_counts': dict(counts), 'selected_pairs': len(selected)}), flush=True)


if __name__ == '__main__':
    main()
