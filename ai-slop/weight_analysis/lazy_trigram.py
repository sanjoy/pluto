"""Lazy, repetition-permitting paths from a frozen two-position polynomial.

This NEW exploratory experiment reads only checkpoints and the already frozen
weight-ranking artifact. It never reads a corpus, prompt, tokenizer, or model
hidden state, and never evaluates attention softmax or a transformer sequence.
The first block's ideal FP64 LayerNorm dictionary coordinates are X0[t] and
X1[t], at positions 0 and 1. Every graph edge reuses those two positions.

For previous b, current a and routing head h, define
  gap[h] = (X1[a] WQ[h] + bQ[h]) dot ((X0[b]-X1[a]) WK[h]) / sqrt(head_dim).
The complete content head paired with h is p[h]: WV, bV, and WO move together.
  uniform = .5 sum_h ((X0[b]+X1[a]) WV[p[h]] + 2 bV[p[h]]) WO[p[h]] + bO
  derivative = .25 sum_h gap[h] ((X0[b]-X1[a]) WV[p[h]]) WO[p[h]].
``first_order`` uses uniform+derivative; ``uniform`` and ``derivative`` are
declared ablations. The uniform component is unchanged by a full permutation
of content heads. Query bias matters; key bias cancels; value bias cancels
ONLY in the derivative; the global output bias is added exactly once.

The destination score is w dot (E[t]-mean_FULL_LOGICAL_VOCAB(E)) / ||w||.
There is no target-norm divisor. This preserves raw-dot destination ordering
within a pair and is invariant to common E shifts with compensated positions.
Scores are neither probabilities nor log likelihoods. Normalizing w discards
magnitude and can amplify tiny cancellation residuals: record their norms and
cancellation ratios, but never invent a data-dependent near-zero cutoff.

Vocabulary S is final:centroid_distance top8192. The first16 ranked IDs seed
all256 ordered pairs, shared by every checkpoint/component/control. Only
visited pairs are contracted and cached. Beam4 per seed keeps top4 outgoing
destinations, permits EVERY repetition, and extends to16 tokens. Zero writes
terminate branches. Final selection prefers length, then mean edge score,
then lexicographic IDs; at most4 paths per seed are emitted. An unextended
two-token seed is not generated text and appears only in diagnostics.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from .attention_certificates import error_bound, top1_certificate
from .checkpoint import GPT2Checkpoint, sha256_file
from .closed_trigram import geometry_provenance, normalized_inputs, validate_recipe_epsilon
from .offset_paths import validate_rankings, write_artifacts
from .routing_audit import summarize_bounds
from .trigram import _top, weight_provenance


PROTOCOL_PATH = (Path(__file__).resolve().parents[2] /
                 'ai-slop/research/weight_memorization/LAZY_POLYNOMIAL_PROTOCOL.md')
RANKING_PROTOCOL_PATH = PROTOCOL_PATH.with_name('VOCABULARY_PROTOCOL.md')
SELECTOR_METHOD = 'final:centroid_distance'
COMPONENTS = ('first_order', 'derivative', 'uniform')
TOP_K = 4
BEAM_WIDTH = 4
PAIR_CHUNK_SIZE = 128


def _hash_array(array, dtype='<f8'):
    return hashlib.sha256(np.asarray(array, dtype=dtype).tobytes()).hexdigest()


def _positive_int(name, value):
    if type(value) is not int or value <= 0:
        raise ValueError(f'{name} must be a positive integer')


def stable_norm(rows):
    """Last-axis Euclidean norms without avoidable squaring overflow/underflow.

    A finite nonzero direction is never classified as zero merely because
    squaring its coordinates underflows. Unrepresentable true norms fail.
    """
    rows = np.asarray(rows, dtype=np.float64)
    if not rows.ndim or not rows.shape[-1] or not np.isfinite(rows).all():
        raise ValueError('norm requires finite nonempty last-axis vectors')
    scale = np.max(np.abs(rows), axis=-1)
    normalized = np.zeros_like(rows)
    np.divide(rows, scale[..., None], out=normalized, where=scale[..., None] > 0)
    with np.errstate(over='ignore'):
        result = scale * np.sqrt(np.sum(normalized * normalized, axis=-1))
    if not np.isfinite(result).all():
        raise ValueError('vector norm exceeds finite FP64 range')
    return result


@dataclass(frozen=True)
class CentroidSelection:
    ids: np.ndarray
    ranked_ids: np.ndarray
    ranked_scores: np.ndarray
    seed_ids: np.ndarray
    checkpoint: GPT2Checkpoint
    checkpoint_provenance: dict
    rankings_sha256: str
    ranking_source_sha256: dict


def load_centroid_selection(rankings_path, checkpoint, vocabulary_size=8192, seed_count=16):
    """Validate all frozen rankings; independently authenticate this selector.

    Full final-checkpoint hashes and recomputed full-vocabulary centroid scores
    must agree with the ranking artifact. Score tolerance is 1e-12 relative and
    absolute for FP64 BLAS differences; the entire recomputed ordering must
    match exactly. Evaluation weights never choose S or seed IDs.
    """
    for name, value in [('vocabulary_size', vocabulary_size), ('seed_count', seed_count)]:
        _positive_int(name, value)
    if not seed_count <= vocabulary_size <= checkpoint.config.vocab_size:
        raise ValueError('require seed_count <= vocabulary_size <= logical vocabulary size')
    encoded = Path(rankings_path).read_bytes()
    artifact = json.loads(encoded)
    validate_rankings(artifact, checkpoint.config.vocab_size)
    entries = [entry for entry in artifact['rankings'] if entry['method'] == SELECTOR_METHOD]
    if len(entries) != 1:
        raise ValueError('expected exactly one final:centroid_distance ranking')
    if artifact.get('protocol', {}).get('sha256') != sha256_file(RANKING_PROTOCOL_PATH):
        raise ValueError('ranking artifact protocol hash mismatch')
    provenance = artifact.get('checkpoint', {}).get('final')
    if (not isinstance(provenance, dict) or provenance.get('config') != asdict(checkpoint.config) or
            not isinstance(provenance.get('checkpoint_directory'), str)):
        raise ValueError('final selector checkpoint provenance/config mismatch')
    directory = Path(provenance['checkpoint_directory']).resolve()
    selector = (checkpoint if directory == checkpoint.directory else
                GPT2Checkpoint(directory, config=checkpoint.config, check_finite=True))
    actual_provenance = selector.provenance(hash_weights=True)
    if actual_provenance['weight_sha256'] != provenance.get('weight_sha256'):
        raise ValueError('final selector weight hashes differ from frozen rankings')
    embedding = np.asarray(selector.token_embedding, dtype=np.float64)
    scores = np.linalg.norm(embedding - embedding.mean(axis=0, keepdims=True), axis=1)
    if not np.isfinite(scores).all():
        raise ValueError('non-finite recomputed centroid-distance scores')
    ids = np.asarray(entries[0]['token_ids'], dtype=np.int64)
    expected = np.lexsort((np.arange(len(scores)), -scores))
    if not np.array_equal(ids, expected):
        raise ValueError('centroid ranking order does not match selector checkpoint')
    values = np.asarray(entries[0]['scores'], dtype=np.float64)
    if not np.allclose(values, scores[ids], rtol=1e-12, atol=1e-12):
        raise ValueError('centroid ranking scores do not match selector checkpoint')
    ranked, selected_scores = ids[:vocabulary_size].copy(), values[:vocabulary_size].copy()
    canonical, seeds = np.sort(ranked), ranked[:seed_count].copy()
    for array in (ranked, selected_scores, canonical, seeds):
        array.setflags(write=False)
    return CentroidSelection(canonical, ranked, selected_scores, seeds, selector, actual_provenance,
                             hashlib.sha256(encoded).hexdigest(), artifact.get('source_sha256', {}))


class LazyTrigramProbe:
    """Cache only visited pairs; dictionary projections are not model states.

    Each pending chunk contracts at most128 pairs and scores all S destinations
    with a matrix multiplication. Batching is only an arithmetic optimization:
    each beam's choices and budget are independent, and every pair is scored
    once. No S^2 or S^3 graph is precomputed.
    """

    def __init__(self, checkpoint, vocabulary_ids, component='first_order', pairing=None):
        ids = np.asarray(vocabulary_ids)
        if (ids.ndim != 1 or not len(ids) or not np.issubdtype(ids.dtype, np.integer) or
                np.any(ids < 0) or np.any(ids >= checkpoint.config.vocab_size) or
                np.any(ids[1:] <= ids[:-1])):
            raise ValueError('S must be nonempty, sorted unique logical token IDs')
        if component not in COMPONENTS:
            raise ValueError('unknown polynomial component')
        if checkpoint.config.context_length < 2:
            raise ValueError('two-position probe requires context length >=2')
        self.vocabulary_ids = ids.astype(np.int64, copy=True)
        self.index = {int(token): index for index, token in enumerate(ids)}
        self.component = component
        self.heads, self.head_dim = checkpoint.config.n_heads, checkpoint.config.head_dim
        pairing = np.arange(self.heads) if pairing is None else np.asarray(pairing)
        if (pairing.shape != (self.heads,) or not np.issubdtype(pairing.dtype, np.integer) or
                sorted(pairing.tolist()) != list(range(self.heads))):
            raise ValueError('pairing must be a permutation of complete content heads')
        self.pairing = pairing.astype(np.int64, copy=True)
        self.inputs0 = normalized_inputs(checkpoint, 0, self.vocabulary_ids)
        self.inputs1 = normalized_inputs(checkpoint, 1, self.vocabulary_ids)
        self.output_mean = np.mean(checkpoint.token_embedding, axis=0, dtype=np.float64)
        self.centered_targets = np.asarray(checkpoint.token_embedding[ids], dtype=np.float64) - self.output_mean

        def projection(kind, order):
            return np.concatenate([checkpoint.qkv(0, kind, int(head)) for head in order], axis=1).astype(np.float64)
        shape = (len(ids), self.heads, self.head_dim)
        query_bias = np.stack([checkpoint.qkv_bias(0, 'q', head) for head in range(self.heads)]).astype(np.float64)
        self.queries1 = (self.inputs1 @ projection('q', range(self.heads))).reshape(shape) + query_bias
        keys = projection('k', range(self.heads))
        self.keys0, self.keys1 = [(inputs @ keys).reshape(shape) for inputs in (self.inputs0, self.inputs1)]
        values = projection('v', self.pairing)
        self.values0, self.values1 = [(inputs @ values).reshape(shape) for inputs in (self.inputs0, self.inputs1)]
        self.value_bias = np.stack([checkpoint.qkv_bias(0, 'v', int(head)) for head in self.pairing]).astype(np.float64)
        self.output = np.stack([checkpoint.attention_output_head(0, int(head)) for head in self.pairing]).astype(np.float64)
        self.output_bias = np.asarray(checkpoint['blocks.0.attn.output.bias'], dtype=np.float64)
        if not all(np.isfinite(array).all() for array in
                   (self.centered_targets, self.queries1, self.keys0, self.keys1,
                    self.values0, self.values1, self.value_bias, self.output, self.output_bias)):
            raise ValueError('non-finite dictionary projections')
        self.cache = {}
        self.pair_diagnostics = {}
        self.cache_requests = 0
        self.write_evaluations = 0

    def _pair(self, previous, current):
        if (type(previous) not in (int, np.int32, np.int64) or
                type(current) not in (int, np.int32, np.int64) or
                previous not in self.index or current not in self.index):
            raise ValueError('pair token IDs must belong to S')
        return int(previous), int(current)

    def components_for_pairs(self, pairs):
        """Return base/derivative/chosen vectors and signed head diagnostics.

        Key/value differences are formed before multiplication, preserving
        cancellation of their shared biases. Uniform uses both value biases;
        the global output bias is never duplicated per head.
        """
        pairs = [self._pair(*pair) for pair in pairs]
        if not pairs:
            raise ValueError('at least one pair is required')
        previous = np.array([self.index[p] for p, _ in pairs])
        current = np.array([self.index[c] for _, c in pairs])
        gaps = np.sum((self.keys0[previous] - self.keys1[current]) * self.queries1[current], axis=2) / np.sqrt(self.head_dim)
        differences = self.values0[previous] - self.values1[current]
        averages = .5 * (self.values0[previous] + self.values1[current]) + self.value_bias
        projected_differences = np.stack([differences[:, head] @ self.output[head]
                                          for head in range(self.heads)], axis=1)
        base_heads = np.stack([averages[:, head] @ self.output[head]
                               for head in range(self.heads)], axis=1)
        derivative_heads = projected_differences * (gaps[:, :, None] / 4)
        uniform = base_heads.sum(axis=1) + self.output_bias
        derivative = derivative_heads.sum(axis=1)
        if self.component == 'first_order':
            chosen, chosen_heads = uniform + derivative, base_heads + derivative_heads
        elif self.component == 'uniform':
            chosen, chosen_heads = uniform, base_heads
        else:
            chosen, chosen_heads = derivative, derivative_heads
        denominator = stable_norm(chosen_heads).sum(axis=1)
        if self.component != 'derivative':
            denominator += stable_norm(self.output_bias)
        chosen_norm = stable_norm(chosen)
        if not all(np.isfinite(array).all() for array in
                   (gaps, projected_differences, uniform, derivative, chosen, denominator)):
            raise ValueError('non-finite polynomial contraction')
        return {'uniform': uniform, 'derivative': derivative, 'chosen': chosen,
                'gaps': gaps, 'projected_value_difference_norms': stable_norm(projected_differences),
                'uniform_norms': stable_norm(uniform), 'derivative_norms': stable_norm(derivative),
                'chosen_norms': chosen_norm, 'head_norm_sum_plus_global_bias': denominator}

    def components_for_pair(self, previous, current):
        return {key: value[0] for key, value in self.components_for_pairs([(previous, current)]).items()}

    def ensure_pairs(self, pairs):
        """Evaluate each previously unseen pair once, retaining signed top4.

        Exact zero chosen writes have no direction: store an empty edge list,
        never arbitrary token-ID tie choices. Zero CENTERED TARGETS are valid
        dot-product destinations, since this score has no target denominator.
        """
        pairs = [self._pair(*pair) for pair in pairs]
        self.cache_requests += len(pairs)
        pending = sorted(set(pairs) - self.cache.keys())
        target_norms = stable_norm(self.centered_targets) if self.component == 'first_order' and pending else None
        for start in range(0, len(pending), PAIR_CHUNK_SIZE):
            batch = pending[start:start + PAIR_CHUNK_SIZE]
            components = self.components_for_pairs(batch)
            writes, norms = components['chosen'], components['chosen_norms']
            raw = writes @ self.centered_targets.T
            # Normalize BEFORE the dot: tiny finite directions must not lose
            # their ranking because an unnormalized product underflows. The
            # separate raw diagnostic can underflow; nonfinite raw values fail.
            directions = np.zeros_like(writes)
            np.divide(writes, norms[:, None], out=directions, where=norms[:, None] > 0)
            scores = directions @ self.centered_targets.T
            if not np.isfinite(raw).all() or not np.isfinite(scores).all():
                raise ValueError('non-finite centered destination scores')
            for row, pair in enumerate(batch):
                denominator = components['head_norm_sum_plus_global_bias'][row]
                ratio = float(norms[row] / denominator) if denominator else None
                self.pair_diagnostics[pair] = {
                    'gaps': components['gaps'][row].copy(),
                    'projected_value_difference_norms': components['projected_value_difference_norms'][row].copy(),
                    'uniform_norm': float(components['uniform_norms'][row]),
                    'derivative_norm': float(components['derivative_norms'][row]),
                    'chosen_norm': float(norms[row]), 'cancellation_ratio': ratio,
                }
                self.write_evaluations += 1
                if norms[row] == 0:
                    self.cache[pair] = ()
                    continue
                top = _top(scores[row], TOP_K, self.vocabulary_ids)
                if self.component == 'first_order' and len(top) >= 2:
                    epsilon = error_bound(components['gaps'][row], components['projected_value_difference_norms'][row])
                    margin = (scores[row, top[0]] - scores[row, top[1]]) * norms[row]
                    self.pair_diagnostics[pair]['top1_certificate'] = top1_certificate(
                        margin, epsilon, target_norms[top[0]], np.max(target_norms))
                self.cache[pair] = tuple({
                    'candidate_id': f'lazy_{self.component}:edge:{pair[0]}:{pair[1]}:{int(self.vocabulary_ids[index])}',
                    'method': f'lazy_{self.component}_edge',
                    'token_ids': [pair[0], pair[1], int(self.vocabulary_ids[index])],
                    'score': float(scores[row, index]), 'centered_dot_product': float(raw[row, index]),
                    'write_norm': float(norms[row]), 'cancellation_ratio': ratio,
                    'destination_rank': rank,
                } for rank, index in enumerate(top, start=1))

    def edges(self, previous, current):
        pair = self._pair(previous, current)
        self.ensure_pairs([pair])
        return self.cache[pair]

    def edge_records(self):
        return [edge for pair in sorted(self.cache) for edge in self.cache[pair]]

    def summary(self):
        """Bounds cover VISITED head/pair contractions only, not all S^2.

        The existing bound routine measures independently STACKED head errors.
        Cancellation means this is not a bound on the head-summed direction,
        normalized destination score, beam path, or full model prediction.
        """
        items = [self.pair_diagnostics[pair] for pair in sorted(self.pair_diagnostics)]
        def quantiles(values):
            values = [value for value in values if value is not None]
            return ({str(p): float(np.quantile(values, p)) for p in (0., .5, .9, .99, 1.)}
                    if values else None)
        result = {
            'cache_requests': self.cache_requests, 'unique_visited_pairs': len(self.cache),
            'write_evaluations': self.write_evaluations,
            'emitted_edges': sum(len(edges) for edges in self.cache.values()),
            'zero_write_pairs': sum(not edges for edges in self.cache.values()),
            'zero_write_pair_ids': [list(pair) for pair in sorted(self.cache) if not self.cache[pair]],
            'write_norm_quantiles': quantiles([item['chosen_norm'] for item in items]),
            'uniform_norm_quantiles': quantiles([item['uniform_norm'] for item in items]),
            'derivative_norm_quantiles': quantiles([item['derivative_norm'] for item in items]),
            'cancellation_ratio_quantiles': quantiles([item['cancellation_ratio'] for item in items]),
            'near_zero_policy': 'no cutoff other than exactly zero; tiny surviving directions may be amplified',
        }
        if items:
            gaps = np.stack([item['gaps'] for item in items], axis=1)[:, :, None]
            values = np.stack([item['projected_value_difference_norms'] for item in items], axis=1)[:, :, None]
            result['visited_routing_bounds'] = summarize_bounds(gaps, values)
        certificates = [item['top1_certificate'] for item in items if 'top1_certificate' in item]
        satisfied = sum(item['ideal_arithmetic_sufficient_condition_holds'] for item in certificates)
        result['top1_attention_certificates'] = {
            'eligible_first_order_pairs': len(certificates),
            'sufficient_condition_holds': satisfied,
            'fraction': satisfied / len(certificates) if certificates else None,
            'scope': 'first_order only; exact two-position attention top1 over S, not full model or path ordering',
            'numerics': 'ordinary FP64 ideal-arithmetic diagnostic, not interval proof',
            'failed_condition': 'inconclusive, not evidence of a changed top1',
        }
        return result


def decode_lazy_paths(probe, seed_ids, length=16, beam_width=BEAM_WIDTH):
    """Beam search over lazy exact pair transitions; every repetition is legal.

    All seed beams advance together so shared pending pairs can be batched.
    At an equal depth, cumulative signed score selects the next beam4. At the
    end, retain at most4 final/terminated beams per seed: longest first, then
    mean edge score, then lexicographic IDs. No repetition filter or ad hoc
    cycle penalty is applied. The output explicitly reports early termination.
    """
    _positive_int('length', length)
    _positive_int('beam_width', beam_width)
    if length < 3:
        raise ValueError('path length must include at least one generated token')
    seeds = np.asarray(seed_ids)
    if (seeds.ndim != 1 or not len(seeds) or not np.issubdtype(seeds.dtype, np.integer) or
            len(set(seeds.tolist())) != len(seeds) or any(int(token) not in probe.index for token in seeds)):
        raise ValueError('seed IDs must be distinct members of S')
    starts = [(int(previous), int(current)) for previous in seeds for current in seeds]
    beams = [[(pair, 0., ())] for pair in starts]
    terminal = [[] for _ in starts]
    for _ in range(length - 2):
        probe.ensure_pairs([entry[0][-2:] for beam in beams for entry in beam])
        next_beams = []
        for index, beam in enumerate(beams):
            extended = []
            for tokens, score, provenance in beam:
                edges = probe.cache[tokens[-2:]]
                if not edges:
                    terminal[index].append((tokens, score, provenance))
                for edge in edges:
                    extended.append((tokens + (edge['token_ids'][2],), score + edge['score'],
                                     provenance + (edge['candidate_id'],)))
            next_beams.append(sorted(extended, key=lambda entry: (-entry[1], entry[0]))[:beam_width])
        beams = next_beams
        if not any(beams):
            break
    paths = []
    no_extension = 0
    for index, (pair, beam, ended) in enumerate(zip(starts, beams, terminal)):
        candidates = [entry for entry in beam + ended if len(entry[0]) >= 3]
        if not candidates:
            no_extension += 1
        selected = sorted(candidates, key=lambda entry: (-len(entry[0]), -entry[1] / (len(entry[0]) - 2), entry[0]))[:beam_width]
        for rank, (tokens, score, provenance) in enumerate(selected):
            paths.append({
                'candidate_id': f'lazy_{probe.component}:seed{index}:beam{rank}',
                'method': f'lazy_{probe.component}_path', 'token_ids': list(tokens),
                'score': score / (len(tokens) - 2), 'cumulative_edge_score': score,
                'seed_pair': list(pair), 'seed_pair_index': index, 'beam_rank': rank + 1,
                'terminated_early': len(tokens) < length,
                'provenance': {'edge_candidate_ids': list(provenance),
                               'search': 'lazy overlapping pairs; all repetitions allowed; no model calls'},
            })
    return paths, {
        'ordered_seed_pairs': len(starts), 'seed_pairs_without_extension': no_extension,
        'emitted_paths': len(paths), 'early_terminated_paths': sum(p['terminated_early'] for p in paths),
        'path_length_histogram': dict(Counter(len(p['token_ids']) for p in paths)),
        'paths_with_adjacent_equal_tokens': sum(any(a == b for a, b in zip(p['token_ids'], p['token_ids'][1:])) for p in paths),
        'paths_with_any_token_more_than_twice': sum(max(Counter(p['token_ids']).values()) > 2 for p in paths),
        'maximum_pair_evaluations_without_cache_reuse': len(starts) * (1 + beam_width * (length - 3)),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--rankings', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--component', choices=COMPONENTS, default='first_order')
    parser.add_argument('--broken-routing-control', action='store_true')
    parser.add_argument('--vocabulary-size', type=int, default=8192)
    parser.add_argument('--seed-count', type=int, default=16)
    parser.add_argument('--path-length', type=int, default=16)
    args = parser.parse_args(argv)
    sidecar = args.output.with_suffix(args.output.suffix + '.metadata.json')
    if any(path.exists() or path.is_symlink() for path in (args.output, sidecar)):
        raise FileExistsError('refusing to overwrite candidates or metadata')
    epsilon = validate_recipe_epsilon()
    protocol_hash = sha256_file(PROTOCOL_PATH)
    checkpoint = GPT2Checkpoint(args.checkpoint, check_finite=True)
    selection = load_centroid_selection(args.rankings, checkpoint, args.vocabulary_size, args.seed_count)
    if args.broken_routing_control and checkpoint.config.n_heads < 2:
        parser.error('broken routing control requires at least two heads')
    pairing = (np.arange(checkpoint.config.n_heads) + int(args.broken_routing_control)) % checkpoint.config.n_heads
    probe = LazyTrigramProbe(checkpoint, selection.ids, args.component, pairing)
    paths, path_stats = decode_lazy_paths(probe, selection.seed_ids, args.path_length)
    edges = probe.edge_records()
    for edge in edges:
        edge['provenance'] = {'metadata_file': sidecar.name,
                              'token_embedding_row_offsets_bytes': [token * checkpoint.config.d_model * 4 for token in edge['token_ids']]}
    for path in paths:
        path['provenance']['metadata_file'] = sidecar.name
    geometry = geometry_provenance(checkpoint, selection.ids, 'normalized')
    geometry['destination_coordinates'] = 'E[t] minus mean of ALL logical output embedding rows, not mean over S; no target normalization'
    routing = weight_provenance(checkpoint, 0, pairing)
    d, c = checkpoint.config.d_model, checkpoint.config.head_dim
    qkv_bias = checkpoint.specs['blocks.0.attn.qkv.bias']
    for head in routing['head_slices']:
        content = head['content_head']
        head['value_bias'] = {'filename': qkv_bias.filename, 'byte_range': [(2*d+content*c)*4, (2*d+(content+1)*c)*4],
                              'used': args.component != 'derivative'}
    output_bias = checkpoint.specs['blocks.0.attn.output.bias']
    routing['global_output_bias'] = {'filename': output_bias.filename, 'byte_range': output_bias.byte_range,
                                     'used': args.component != 'derivative', 'multiplicity': 1}
    routing['score_reads'] = (
        'input geometry, WV/bV/WO content slices and global output bias, all logical embedding rows for output mean; QK is diagnostic-only'
        if args.component == 'uniform' else
        'input geometry, listed QK/content head slices and query bias, value/output biases only if marked used, all logical embedding rows for output mean')
    routing['diagnostic_reads'] = 'all listed head slices and biases for uniform/derivative norms, cancellation and routing bounds, including parameters unused by the chosen score component'
    routing['selection_reads'] = 'fixed ranking: full logical FINAL embedding table; search: all centered evaluation E[t] for t in S'
    metadata = {
        'schema_version': 1, 'stage': 'corpus_blind_extraction',
        'protocol': 'lazy repeated-token paths from centered-output two-position attention polynomial',
        'protocol_manifest': {'file': str(PROTOCOL_PATH), 'sha256': protocol_hash},
        'parameters': dict({key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                           top_k=TOP_K, beam_width=BEAM_WIDTH, pair_chunk_size=PAIR_CHUNK_SIZE,
                           input_positions=[0, 1], repetition_filter=False),
        'numpy_version': np.__version__,
        'checkpoint': checkpoint.provenance(hash_weights=True),
        'vocabulary_selection_checkpoint': selection.checkpoint_provenance,
        'rankings': {'file': str(args.rankings.resolve()), 'sha256': selection.rankings_sha256,
                     'source_sha256': selection.ranking_source_sha256,
                     'validation': 'all rankings validated; all final weight hashes checked; complete centroid ranking recomputed'},
        'vocabulary_selection': {
            'method': SELECTOR_METHOD, 'ids_ascending': selection.ids.tolist(),
            'ids_by_centroid_rank': selection.ranked_ids.tolist(), 'ranked_scores': selection.ranked_scores.tolist(),
            'seed_ids_by_rank': selection.seed_ids.tolist(),
            'ids_little_endian_uint32_sha256': _hash_array(selection.ids, '<u4'),
            'seed_ids_little_endian_uint32_sha256': _hash_array(selection.seed_ids, '<u4'),
            'selection_checkpoint_differs_from_evaluation': checkpoint.directory != selection.checkpoint.directory,
            'prior_warning': 'Early uses final-selected S and seeds; this is an explicit learned vocabulary prior.',
            'physical_dependencies': {'filename': selection.checkpoint.specs['token_embedding.weight'].filename,
                                       'logical_byte_range': [0, checkpoint.config.vocab_size*d*4],
                                       'centroid': 'all logical FINAL embedding rows, excluding padding'}},
        'output_centering': {'filename': checkpoint.specs['token_embedding.weight'].filename,
                             'logical_byte_range': [0, checkpoint.config.vocab_size*d*4],
                             'full_mean_float64_sha256': _hash_array(probe.output_mean),
                             'targets_float64_sha256': _hash_array(probe.centered_targets),
                             'embedding_shared_with': list(checkpoint.specs['token_embedding.weight'].shared_with)},
        'geometry_provenance': geometry, 'routing_weight_provenance': routing,
        'epsilon_source_validation': epsilon,
        'numerical_geometry_hashes': {'X0_little_endian_float64_sha256': _hash_array(probe.inputs0),
                                      'X1_little_endian_float64_sha256': _hash_array(probe.inputs1)},
        'diagnostics': dict(probe.summary(), **path_stats),
        'method_counts': dict(Counter(record['method'] for record in edges + paths)),
        'score_definition': 'chosen write dot (E[t] - full logical evaluation E mean) / stable_norm(chosen write); no target norm',
        'cancellation_ratio_definition': 'chosen write norm / (sum norms of chosen per-head contributions + global output-bias norm if included)',
        'path_policy': 'beam4 cumulative signed score at equal depth; final longest then mean edge score then IDs; all repetitions allowed',
        'source_sha256': {name: sha256_file(Path(__file__).with_name(name)) for name in
                          ('lazy_trigram.py', 'checkpoint.py', 'closed_trigram.py', 'trigram.py', 'offset_paths.py', 'routing_audit.py', 'attention_certificates.py')},
        'limitations': [
            'First-order extrapolation around uniform two-position routing, not trained attention probabilities or full GPT-2 predictions.',
            'No residual, MLP, later blocks, final contextual LayerNorm, or full positional sequence computation.',
            'Only visited pairs in fixed S8192; seed16 and beam4 are bounded search, not exhaustive passage recovery.',
            'Positions0/1 reused for every edge, even as paths grow; FP64 ideal geometry does not reproduce BF16 execution.',
            'Normalized writes discard magnitude and can amplify near-cancellation; only exact zero terminates.',
            'Scores and path sums are not probabilities or likelihoods; loops are reported, not filtered after inspection.',
            'visited_routing_bounds applies to separately stacked visited head corrections, not their sum or normalized ranking.',
            'top1_attention_certificates separately uses a sum-of-head error upper bound for first_order only; its sufficient condition concerns exact two-position attention top1 over S, not full-model predictions or path order.',
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata = write_artifacts(args.output, edges + paths, metadata)
    print(json.dumps({'output': str(args.output), 'candidate_sha256': metadata['candidate_sha256'],
                      'method_counts': metadata['method_counts'], 'search': path_stats,
                      'visited_pairs': metadata['diagnostics']['unique_visited_pairs'],
                      'vocabulary_hash': metadata['vocabulary_selection']['ids_little_endian_uint32_sha256']}))


if __name__ == '__main__':
    main()
