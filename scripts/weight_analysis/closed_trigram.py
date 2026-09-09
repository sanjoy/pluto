"""A closed-vocabulary, pre-LayerNorm-coordinate analytical trigram probe.

This is a NEW exploratory protocol, not a revision of the earlier frozen
full-vocabulary/norm-selected trigram experiment. It reads weights plus token
labels only. No corpus, prompts, attention softmax, contextual activation
dataset, or transformer inference is used. The reused derivative contraction
is documented and independently finite-difference-tested in trigram.py.

There are two explicit corrections to that earlier protocol:

1. Previous and current token coordinates are, in FP64,
     X_p[t] = gamma1 * (E[t] + P[p] - mean(E[t] + P[p]))
                       / sqrt(var(E[t] + P[p]) + epsilon) + beta1,
   with previous position0 and current position1. Population variance and
   learned scale/bias match first-block LayerNorm's mathematical definition.
   epsilon is the recipe's float32 1e-5f value, promoted to FP64. The C++ recipe
   literal is checked at runtime. This is an ideal-real-arithmetic coordinate
   correction, NOT a bitwise match to BF16 input/position/LayerNorm rounding.
   --input-geometry=raw substitutes X_0=X_1=E as the declared ablation.

2. The finite vocabulary S is selected ONCE from a declared selection
   checkpoint: top128 logical tokens by normalized position1 joint query norm.
   This selection is ALWAYS normalized, including for the raw ablation. All
   ordered pairs (previous,current) in S^2 are scored against every destination
   in S, retaining top4 destination cosines. S is therefore closed under graph
   endpoints, unlike the previous disjoint norm-selected endpoint sets. This
   does not guarantee a usable path when writes vanish or retained edges are
   degenerate. Undefined zero-write/zero-target cosines are excluded explicitly.

The fixed defaults are S=128, top4 destinations/pair, up to256 paths of length
up to12, beam4, token repetition limit2. Source selection has no corpus access.
Use the SAME --vocabulary-checkpoint for final/early/raw/broken comparisons.
An early checkpoint evaluated on a final-selected S benefits from this explicit
learned vocabulary prior and is NOT an independent untrained-vocabulary control.

QK routing head h can be paired with intact OV[(h+1)%H] by the broken control.
Heads are summed with signs before any cosine ranking. Negative destination
scores remain eligible if they are top4; a score is not a probability or actual
next-token logit. Paths reuse the existing decoder: exact overlapping triples,
no immediate equal tokens or third token occurrence, longest retained path
then highest mean edge cosine. They never reevaluate a model or normalization.

This still extracts only the first derivative of a two-position attention
subproblem at uniform routing. It omits the uniform-routing base term, MLP,
residual/direct path, final normalization, and seven later transformer blocks.
It is neither full inference nor an analytically exact text decompressor.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from .checkpoint import GPT2Checkpoint, sha256_file
from .trigram import TrigramProbe, _top, decode_trigram_paths, weight_provenance


RECIPE_EPSILON = float(np.float32(1e-5))
PROTOCOL_PATH = (Path(__file__).resolve().parents[2] /
                 'research/weight_memorization/CLOSED_GRAPH_PROTOCOL.md')


def validate_recipe_epsilon(repository_root=None):
    """Check current source, not historical producer metadata absent on disk."""
    root = Path(repository_root) if repository_root is not None else Path(__file__).resolve().parents[2]
    source_path = root / 'src/llm/recipes/gpt2.cc'
    match = re.search(r'\bkLayerNormEpsilon\s*=\s*([0-9.eE+-]+)f\s*;', source_path.read_text())
    if match is None or float(np.float32(float(match[1]))) != RECIPE_EPSILON:
        raise ValueError('current GPT-2 recipe LayerNorm epsilon differs from the audited probe')
    return {'source': str(source_path), 'source_sha256': sha256_file(source_path),
            'cpp_float_literal': match[1] + 'f', 'fp64_promoted_value': RECIPE_EPSILON,
            'scope': 'current C++ source; checkpoint has no authenticated historical recipe metadata'}


def _token_ids(checkpoint, token_ids):
    ids = np.asarray(token_ids)
    if (ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer) or
            np.any(ids < 0) or np.any(ids >= checkpoint.config.vocab_size)):
        raise ValueError('token IDs must be a one-dimensional logical vocabulary subset')
    return ids.astype(np.int64, copy=False)


def normalized_inputs(checkpoint, position, token_ids=None):
    """FP64 dictionary rows after ideal first LayerNorm; no attention executed."""
    if type(position) is not int or not 0 <= position < checkpoint.config.context_length:
        raise ValueError('position outside checkpoint context length')
    ids = (np.arange(checkpoint.config.vocab_size) if token_ids is None
           else _token_ids(checkpoint, token_ids))
    inputs = np.asarray(checkpoint.token_embedding[ids], dtype=np.float64)
    inputs += np.asarray(checkpoint['position_embedding.weight'][position], dtype=np.float64)
    centered = inputs - inputs.mean(axis=1, keepdims=True)
    variance = np.mean(centered * centered, axis=1, keepdims=True)
    gamma = np.asarray(checkpoint['blocks.0.ln1.scale'], dtype=np.float64)
    beta = np.asarray(checkpoint['blocks.0.ln1.bias'], dtype=np.float64)
    result = centered / np.sqrt(variance + RECIPE_EPSILON) * gamma + beta
    if not np.isfinite(result).all():
        raise ValueError('non-finite normalized dictionary rows')
    return result


@dataclass(frozen=True)
class VocabularySelection:
    """Ascending IDs for canonical pair order, plus original query-norm ranks."""
    ids: np.ndarray
    ranked_ids: np.ndarray
    ranked_norm_squared: np.ndarray


def select_vocabulary(checkpoint, size=128, chunk_size=2048):
    """Full logical-vocabulary search using normalized position1 Q coordinates.

    Neither evaluation-checkpoint weights nor the evaluation geometry/control
    participate here. Ties, including all-zero norms, use ascending token IDs.
    A returned S may contain zero-write directions; extraction reports those
    instead of silently changing the selection budget or label prior.
    """
    for name, value in [('size', size), ('chunk_size', chunk_size)]:
        if type(value) is not int or value <= 0:
            raise ValueError(f'{name} must be a positive integer')
    query = np.concatenate([checkpoint.qkv(0, 'q', head)
                            for head in range(checkpoint.config.n_heads)], axis=1).astype(np.float64)
    bias = np.concatenate([checkpoint.qkv_bias(0, 'q', head)
                           for head in range(checkpoint.config.n_heads)]).astype(np.float64)
    norms = np.empty(checkpoint.config.vocab_size)
    for start in range(0, len(norms), chunk_size):
        ids = np.arange(start, min(start + chunk_size, len(norms)))
        projected = normalized_inputs(checkpoint, 1, ids) @ query + bias
        norms[ids] = np.sum(projected * projected, axis=1)
    if not np.isfinite(norms).all():
        raise ValueError('non-finite vocabulary-selection query norms')
    ranked = _top(norms, size)
    return VocabularySelection(np.sort(ranked), ranked, norms[ranked])


class ClosedTrigramProbe:
    """All ordered S^2 pairs with destination search in the same frozen S.

    For each pair the old tested contraction receives rows X_0[b], X_1[a].
    Input dictionary rows are distinct from the ORIGINAL tied output embedding
    E[c], which supplies destination labels/directions. Normalizing E[c] by the
    first LayerNorm would mix input/output coordinate systems incorrectly.
    """

    def __init__(self, checkpoint, vocabulary_ids, geometry='normalized', pairing=None):
        ids = _token_ids(checkpoint, vocabulary_ids)
        if not len(ids) or np.any(ids[1:] <= ids[:-1]):
            raise ValueError('S must be nonempty, unique, and sorted ascending')
        if checkpoint.config.context_length < 2:
            raise ValueError('two-position probe requires context length >=2')
        if geometry not in ('normalized', 'raw'):
            raise ValueError('geometry must be normalized or raw')
        self.vocabulary_ids = ids.copy()
        self.geometry = geometry
        self.embedding = np.asarray(checkpoint.token_embedding[ids], dtype=np.float64)
        self.inputs0 = (normalized_inputs(checkpoint, 0, ids) if geometry == 'normalized'
                        else self.embedding.copy())
        self.inputs1 = (normalized_inputs(checkpoint, 1, ids) if geometry == 'normalized'
                        else self.embedding.copy())
        heads = checkpoint.config.n_heads
        projections = {component: np.stack([checkpoint.qkv(0, component, head)
                                            for head in range(heads)]).astype(np.float64)
                       for component in ('q', 'k', 'v')}
        output = np.stack([checkpoint.attention_output_head(0, head)
                           for head in range(heads)]).astype(np.float64)
        query_bias = np.stack([checkpoint.qkv_bias(0, 'q', head)
                               for head in range(heads)]).astype(np.float64)
        # Concatenation is only a dictionary of weight-derived coordinate rows,
        # not a 2|S|-token sequence. No contextual interaction is computed here.
        self.contraction = TrigramProbe(np.concatenate((self.inputs0, self.inputs1)),
                                        projections['q'], projections['k'], projections['v'],
                                        output, query_bias, pairing)
        self.target_norms = np.linalg.norm(self.embedding, axis=1)

    def interaction_writes(self, current_index):
        """Return one signed write for EVERY previous index in frozen S."""
        if type(current_index) is not int or not 0 <= current_index < len(self.vocabulary_ids):
            raise ValueError('current_index outside S')
        return self.contraction.interaction_writes(
            current_index + len(self.vocabulary_ids), np.arange(len(self.vocabulary_ids)))

    def extract(self, top_k=4):
        """Top destination cosines, preserving signs and zero-norm diagnostics."""
        if type(top_k) is not int or top_k <= 0:
            raise ValueError('top_k must be a positive integer')
        records = []
        zero_pairs = 0
        for current_index, current in enumerate(self.vocabulary_ids):
            writes = self.interaction_writes(current_index)
            write_norms = np.linalg.norm(writes, axis=1)
            raw = writes @ self.embedding.T
            if not np.isfinite(raw).all() or not np.isfinite(write_norms).all():
                raise ValueError('non-finite closed-vocabulary scores')
            denominators = write_norms[:, None] * self.target_norms[None, :]
            scores = np.full(raw.shape, -np.inf)
            np.divide(raw, denominators, out=scores, where=denominators > 0)
            for previous_index, previous in enumerate(self.vocabulary_ids):
                if write_norms[previous_index] == 0:
                    zero_pairs += 1
                destinations = _top(scores[previous_index], top_k, self.vocabulary_ids)
                for rank, destination in enumerate(destinations, start=1):
                    records.append({
                        'token_ids': [int(previous), int(current), int(self.vocabulary_ids[destination])],
                        'score': float(scores[previous_index, destination]),
                        'raw_dot_product': float(raw[previous_index, destination]),
                        'interaction_write_norm': float(write_norms[previous_index]),
                        'destination_cosine_rank': rank,
                    })
        diagnostics = {
            'attempted_ordered_pairs': len(self.vocabulary_ids) ** 2,
            'zero_write_pairs': zero_pairs,
            'zero_target_rows': int(np.count_nonzero(self.target_norms == 0)),
            'pairs_with_emitted_destinations': len({tuple(r['token_ids'][:2]) for r in records}),
            'triple_records': len(records),
        }
        return records, diagnostics


def geometry_provenance(checkpoint, vocabulary_ids, geometry):
    """Addresses of every LN/position parameter used in the coordinate table."""
    positions = checkpoint.specs['position_embedding.weight']
    embedding = checkpoint.specs['token_embedding.weight']
    d = checkpoint.config.d_model
    return {
        'input_geometry': geometry, 'arithmetic_dtype': 'float64',
        'epsilon': RECIPE_EPSILON, 'previous_position': 0, 'current_position': 1,
        'variance_convention': 'population variance, divisor d_model',
        'bf16_rounding': 'not emulated; ideal mathematical input-coordinate correction',
        'embedding': {'filename': embedding.filename, 'token_ids': vocabulary_ids.tolist(),
                      'row_byte_ranges': [[int(t) * d * 4, (int(t) + 1) * d * 4]
                                          for t in vocabulary_ids]},
        'positions': {'filename': positions.filename, 'used_in_raw_mode': False,
                      'row_byte_ranges': [[0, d * 4], [d * 4, 2 * d * 4]]},
        'layer_norm': {role: {'filename': checkpoint.specs[f'blocks.0.ln1.{role}'].filename,
                             'byte_range': checkpoint.specs[f'blocks.0.ln1.{role}'].byte_range,
                             'used_in_raw_mode': False}
                       for role in ('scale', 'bias')},
        'destination_coordinates': 'original tied output embedding E[c], NOT normalized input rows',
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--vocabulary-checkpoint', type=Path, required=True)
    parser.add_argument('--tokenizer-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--input-geometry', choices=('normalized', 'raw'), default='normalized')
    parser.add_argument('--broken-routing-control', action='store_true')
    parser.add_argument('--vocabulary-size', type=int, default=128)
    parser.add_argument('--top-k', type=int, default=4)
    parser.add_argument('--path-length', type=int, default=12)
    parser.add_argument('--path-starts', type=int, default=256)
    parser.add_argument('--beam-width', type=int, default=4)
    args = parser.parse_args(argv)
    if min(args.vocabulary_size, args.top_k, args.beam_width) <= 0:
        parser.error('vocabulary-size, top-k, and beam-width must be positive')
    if args.path_length < 4 or args.path_starts < 0:
        parser.error('path-length must be >=4; path-starts must be nonnegative')
    sidecar = args.output.with_suffix(args.output.suffix + '.metadata.json')
    if args.output.exists() or sidecar.exists():
        raise FileExistsError('refusing to overwrite candidates or metadata')
    epsilon_check = validate_recipe_epsilon()
    checkpoint = GPT2Checkpoint(args.checkpoint, check_finite=True)
    vocabulary_checkpoint = (checkpoint if args.checkpoint.resolve() == args.vocabulary_checkpoint.resolve()
                             else GPT2Checkpoint(args.vocabulary_checkpoint, check_finite=True))
    if checkpoint.config != vocabulary_checkpoint.config:
        raise ValueError('evaluation and vocabulary-selection checkpoint configurations differ')
    if args.broken_routing_control and checkpoint.config.n_heads < 2:
        parser.error('broken routing control requires at least two heads')
    vocabulary_path = args.tokenizer_dir / 'tokenizer.json'
    vocabulary = json.loads(vocabulary_path.read_text())['model']['vocab']
    if (len(vocabulary) != checkpoint.config.vocab_size or
            set(vocabulary.values()) != set(range(checkpoint.config.vocab_size))):
        raise ValueError('tokenizer vocabulary must match checkpoint vocabulary')
    labels = {token: text for text, token in vocabulary.items()}
    selection = select_vocabulary(vocabulary_checkpoint, args.vocabulary_size)
    pairing = np.arange(checkpoint.config.n_heads)
    if args.broken_routing_control:
        pairing = (pairing + 1) % checkpoint.config.n_heads
    probe = ClosedTrigramProbe(checkpoint, selection.ids, args.input_geometry, pairing)
    records, diagnostics = probe.extract(args.top_k)
    method = 'closed_qkov_' + args.input_geometry
    if args.broken_routing_control:
        method += '_broken_routing'
    for index, record in enumerate(records):
        record.update(candidate_id=f'{method}:triple{index}', method=method,
                      vocabulary_labels=[labels[t] for t in record['token_ids']],
                      provenance={'metadata_file': sidecar.name,
                                  'ov_pairing': pairing.tolist(),
                                  'input_positions': [0, 1],
                                  'token_embedding_row_offsets_bytes':
                                      [t * checkpoint.config.d_model * 4 for t in record['token_ids']]})
    paths = list(decode_trigram_paths(records, length=args.path_length,
                                      starts=args.path_starts, beam_width=args.beam_width))
    for index, path in enumerate(paths):
        path.update(candidate_id=f'{method}:path{index}', method=method + '_path',
                    vocabulary_labels=[labels[t] for t in path['token_ids']])
        path['provenance']['metadata_file'] = sidecar.name
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        for record in records + paths:
            stream.write(json.dumps(record, allow_nan=False) + '\n')
    source_pairs = {tuple(record['token_ids'][:2]) for record in records}
    target_pairs = {tuple(record['token_ids'][1:]) for record in records}
    metadata = {
        'schema_version': 1, 'stage': 'corpus_blind_extraction',
        'protocol': 'closed-vocabulary first-LayerNorm-coordinate trigram derivative',
        'protocol_manifest': {'file': str(PROTOCOL_PATH), 'sha256': sha256_file(PROTOCOL_PATH)},
        'parameters': {name: str(value) if isinstance(value, Path) else value
                       for name, value in vars(args).items()},
        'method_counts': dict(Counter(record['method'] for record in records + paths)),
        'checkpoint': checkpoint.provenance(hash_weights=True),
        'vocabulary_selection_checkpoint': vocabulary_checkpoint.provenance(hash_weights=True),
        'vocabulary_selection': {
            'rule': 'top joint normalized position1 query norm; full logical vocabulary; ID tie break',
            'selection_geometry': 'always normalized, independently of evaluation geometry/control',
            'ids_ascending': selection.ids.tolist(),
            'ids_by_query_norm_rank': selection.ranked_ids.tolist(),
            'ranked_norm_squared': selection.ranked_norm_squared.tolist(),
            'vocabulary_labels_by_rank': [labels[t] for t in selection.ranked_ids],
            'ids_little_endian_uint32_sha256': hashlib.sha256(selection.ids.astype('<u4').tobytes()).hexdigest(),
            'selection_checkpoint_differs_from_evaluation': checkpoint.directory != vocabulary_checkpoint.directory,
            'prior_warning': 'Early evaluation with final-selected S has an explicit learned vocabulary prior; not independent untrained selection.',
        },
        'diagnostics': dict(diagnostics, source_pair_count=len(source_pairs),
                            target_pair_count=len(target_pairs),
                            source_target_pair_intersection=len(source_pairs & target_pairs),
                            overlap_edges=sum(tuple(r['token_ids'][1:]) in source_pairs for r in records),
                            emitted_paths=len(paths),
                            candidate_length_histogram=dict(Counter(len(r['token_ids']) for r in records + paths)),
                            negative_score_triples=sum(r['score'] < 0 for r in records)),
        'geometry_provenance': geometry_provenance(checkpoint, selection.ids, args.input_geometry),
        'routing_weight_provenance': weight_provenance(checkpoint, 0, pairing),
        'epsilon_source_validation': epsilon_check,
        'numerical_geometry_hashes': {
            'X0_little_endian_float64_sha256': hashlib.sha256(probe.inputs0.astype('<f8').tobytes()).hexdigest(),
            'X1_little_endian_float64_sha256': hashlib.sha256(probe.inputs1.astype('<f8').tobytes()).hexdigest(),
        },
        'tokenizer_sha256': sha256_file(vocabulary_path),
        'candidate_sha256': sha256_file(args.output),
        'extractor_sha256': sha256_file(__file__),
        'contraction_and_path_decoder_sha256': sha256_file(Path(__file__).with_name('trigram.py')),
        'checkpoint_loader_sha256': sha256_file(Path(__file__).with_name('checkpoint.py')),
        'score_definition': 'cosine(sum_h routing_difference * value_difference / 4, ORIGINAL output E[c]); raw derivative projection also recorded',
        'path_policy': 'existing overlapping-triple beam search; signed scores; no immediate equal tokens or third occurrence; longest then mean edge cosine',
        'limitations': [
            'Only S^2 pairs and S destinations; excludes most vocabulary and cannot estimate corpus coverage.',
            'A first derivative at uniform routing, not the full attention function or predictions.',
            'Ideal FP64 coordinate correction; does not reproduce BF16/FP32 rounding during training.',
            'Positions0/1 only; no MLP, residual/direct contribution, final LayerNorm, or later blocks.',
            'Each graph edge reuses positions0/1; a path does not execute position-dependent model behavior over its length.',
            'Top4 can include negative cosines; scores are not probabilities.',
            'Raw equal-token pairs have exactly zero derivative and are omitted as undefined cosines.',
            'Closed endpoint vocabulary enables but does not prove valid or memorized passage paths.',
        ],
    }
    with sidecar.open('x') as stream:
        json.dump(metadata, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'method_counts': metadata['method_counts'],
                      'diagnostics': metadata['diagnostics'],
                      'vocabulary_hash': metadata['vocabulary_selection']['ids_little_endian_uint32_sha256']}))


if __name__ == '__main__':
    main()
