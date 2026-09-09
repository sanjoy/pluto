"""Frozen output-offset vocabulary, followed by static overlapping trigram paths.

The vocabulary comes only from ``final:output_offset`` in vocabulary.py's
complete ranking artifact: E[t] dot final-LayerNorm beta. This is an exact
additive logit term in real arithmetic, NOT a probability or a token count.
The contextual term is omitted. This module checks the artifact against the
selector's actual checkpoint before using it; no corpus or prompt is read.

Only selection changes from closed_trigram.py. The tested contraction still
uses ideal FP64 first-LayerNorm coordinates at positions 0/1, the signed sum
of first-order QK-times-OV writes, and ORIGINAL tied output embedding rows as
destination directions. All S^2 pairs search destinations inside the SAME S.
There is no attention softmax, transformer sequence execution, or hidden-state
dataset. Final LayerNorm beta selects S; it is NOT added to the path scores.

The fixed protocol keeps S=128, top4 destinations, 256 starts, beam4, length12,
and at most two occurrences per token. Negative top4 scores remain eligible.
Only --vocabulary-size is exposed for small synthetic fixtures. The intact
final, broken routing-to-intact-OV final, and early arms all use final's S.
Thus early explicitly receives a learned vocabulary prior, not independent
untrained selection. Every path reuses positions0/1, rather than executing
the model at its successive sequence positions. A graph path is not itself
evidence of a memorized passage.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from .checkpoint import GPT2Checkpoint, sha256_file
from .closed_trigram import (ClosedTrigramProbe, geometry_provenance,
                             validate_recipe_epsilon)
from .trigram import decode_trigram_paths, weight_provenance


PROTOCOL_PATH = (Path(__file__).resolve().parents[2] /
                 'research/weight_memorization/VOCABULARY_PROTOCOL.md')
SELECTOR_METHOD = 'final:output_offset'
TOP_K = 4
PATH_LENGTH = 12
PATH_STARTS = 256
BEAM_WIDTH = 4


@dataclass(frozen=True)
class OffsetSelection:
    """The same final-selected S is reused regardless of evaluated checkpoint."""
    ids: np.ndarray
    ranked_ids: np.ndarray
    ranked_scores: np.ndarray
    checkpoint: GPT2Checkpoint
    checkpoint_provenance: dict
    rankings_sha256: str
    ranking_source_sha256: dict
    validation: dict


def validate_rankings(artifact, vocab_size):
    """Validate EVERY full permutation, not merely the 128 selected entries.

    Scores are aligned to ranked IDs, descending; equal scores require IDs in
    ascending order. JSON bools are not accepted as numbers or token IDs.
    Validation checks structure, not the truth of all six score definitions;
    load_offset_selection independently recomputes the selected offset score.
    """
    if (not isinstance(artifact, dict) or type(artifact.get('schema_version')) is not int or
            artifact['schema_version'] != 1 or
            type(artifact.get('vocab_size')) is not int or
            artifact['vocab_size'] != vocab_size):
        raise ValueError('ranking artifact vocabulary/schema mismatch')
    if artifact.get('stage') != 'weight_only_vocabulary_rankings_not_model_inference':
        raise ValueError('expected a weight-only vocabulary ranking artifact')
    entries = artifact.get('rankings')
    if not isinstance(entries, list) or not entries:
        raise ValueError('rankings must be a nonempty list')
    methods = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError('each ranking must be an object')
        method, ids, scores = (entry.get(key) for key in ('method', 'token_ids', 'scores'))
        if not isinstance(method, str) or not method or method in methods:
            raise ValueError('ranking methods must be nonempty and unique')
        if (not isinstance(ids, list) or len(ids) != vocab_size or
                any(type(token) is not int for token in ids) or
                set(ids) != set(range(vocab_size))):
            raise ValueError('ranking token IDs must be a complete logical-vocabulary permutation')
        if (not isinstance(scores, list) or len(scores) != vocab_size or
                any(type(score) not in (int, float) for score in scores)):
            raise ValueError('each token must have one finite numeric score')
        try:
            values = np.asarray(scores, dtype=np.float64)
        except (OverflowError, ValueError) as error:
            raise ValueError('scores must be representable as finite FP64') from error
        if not np.isfinite(values).all():
            raise ValueError('ranking scores must be finite')
        tokens = np.asarray(ids, dtype=np.int64)
        if (np.any(values[1:] > values[:-1]) or
                np.any((values[1:] == values[:-1]) & (tokens[1:] < tokens[:-1]))):
            raise ValueError('ranking must be descending score with ascending token-ID ties')
        methods[method] = (tokens, values)
    if SELECTOR_METHOD not in methods:
        raise ValueError('missing final:output_offset ranking')
    return methods[SELECTOR_METHOD]


def load_offset_selection(rankings_path, evaluation_checkpoint, size=128):
    """Authenticate the fixed selector against weights, not evaluation labels.

    All final-selector tensor hashes are checked, including physical padding.
    Offset scores are recomputed from logical E and final beta. A 1e-12
    relative/absolute tolerance allows FP64 BLAS reduction differences, while
    the complete recomputed ID ordering must match exactly. Other rankings
    receive structural validation only; their independently frozen verification
    belongs to the separate vocabulary experiment.
    """
    if type(size) is not int or not 1 <= size <= evaluation_checkpoint.config.vocab_size:
        raise ValueError('vocabulary size must be between 1 and the logical vocabulary size')
    encoded = Path(rankings_path).read_bytes()
    artifact = json.loads(encoded)
    ranked_ids, ranked_scores = validate_rankings(artifact, evaluation_checkpoint.config.vocab_size)
    if artifact.get('protocol', {}).get('sha256') != sha256_file(PROTOCOL_PATH):
        raise ValueError('ranking artifact must use the current fixed vocabulary protocol')
    provenance = artifact.get('checkpoint', {}).get('final')
    if (not isinstance(provenance, dict) or
            provenance.get('config') != asdict(evaluation_checkpoint.config) or
            not isinstance(provenance.get('checkpoint_directory'), str)):
        raise ValueError('final selector checkpoint/config provenance mismatch')
    directory = Path(provenance['checkpoint_directory']).resolve()
    selector = (evaluation_checkpoint if directory == evaluation_checkpoint.directory else
                GPT2Checkpoint(directory, config=evaluation_checkpoint.config, check_finite=True))
    actual_provenance = selector.provenance(hash_weights=True)
    if provenance.get('weight_sha256') != actual_provenance['weight_sha256']:
        raise ValueError('final selector checkpoint weight hashes differ from frozen rankings')
    scores = (np.asarray(selector.token_embedding, dtype=np.float64) @
              np.asarray(selector['final_norm.bias'], dtype=np.float64))
    if not np.isfinite(scores).all():
        raise ValueError('non-finite recomputed output offsets')
    expected_ids = np.lexsort((np.arange(len(scores)), -scores))
    if not np.array_equal(ranked_ids, expected_ids):
        raise ValueError('final output-offset ordering does not match checkpoint weights')
    if not np.allclose(ranked_scores, scores[ranked_ids], rtol=1e-12, atol=1e-12):
        raise ValueError('final output-offset scores do not match checkpoint weights')
    selected = ranked_ids[:size].copy()
    canonical = np.sort(selected)
    selected_scores = ranked_scores[:size].copy()
    for array in (canonical, selected, selected_scores):
        array.setflags(write=False)
    return OffsetSelection(
        canonical, selected, selected_scores, selector, actual_provenance,
        hashlib.sha256(encoded).hexdigest(), artifact.get('source_sha256', {}),
        {'all_rankings': 'full finite sorted ID permutations validated',
         'selector_weights': 'all final checkpoint tensor hashes match ranking provenance',
         'selector_scores': 'recomputed E dot beta; full ordering exact; FP64 scores rtol=atol=1e-12',
         'scope': 'consistency with declared weights, not historical training/corpus authentication'})


def write_artifacts(output, records, metadata):
    """Create both files exclusively; never follow or replace existing links.

    Serialize and hash before creating either file. If acquiring the second
    filename fails, remove only the empty first file created by this call.
    Existing user files, including a dangling symlink, remain untouched.
    """
    output = Path(output)
    sidecar = output.with_suffix(output.suffix + '.metadata.json')
    payload = ''.join(json.dumps(record, allow_nan=False) + '\n' for record in records).encode()
    metadata = dict(metadata, candidate_sha256=hashlib.sha256(payload).hexdigest())
    metadata_payload = (json.dumps(metadata, indent=2, allow_nan=False) + '\n').encode()
    created = []
    try:
        with output.open('xb') as candidates:
            created.append(output)
            with sidecar.open('xb') as details:
                created.append(sidecar)
                candidates.write(payload)
                details.write(metadata_payload)
    except BaseException:
        for path in created:
            path.unlink()
        raise
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--rankings', type=Path, required=True)
    parser.add_argument('--tokenizer-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--broken-routing-control', action='store_true')
    parser.add_argument('--vocabulary-size', type=int, default=128,
                        help='128 in the fixed real-data protocol; smaller only for synthetic fixtures')
    args = parser.parse_args(argv)
    sidecar = args.output.with_suffix(args.output.suffix + '.metadata.json')
    if any(path.exists() or path.is_symlink() for path in (args.output, sidecar)):
        raise FileExistsError('refusing to overwrite candidates or metadata')
    epsilon_check = validate_recipe_epsilon()
    checkpoint = GPT2Checkpoint(args.checkpoint, check_finite=True)
    selection = load_offset_selection(args.rankings, checkpoint, args.vocabulary_size)
    heads = checkpoint.config.n_heads
    if args.broken_routing_control and heads < 2:
        parser.error('broken routing control requires at least two heads')
    pairing = (np.arange(heads) + int(args.broken_routing_control)) % heads
    vocabulary_path = args.tokenizer_dir / 'tokenizer.json'
    vocabulary = json.loads(vocabulary_path.read_text())['model']['vocab']
    if (len(vocabulary) != checkpoint.config.vocab_size or
            any(type(token) is not int for token in vocabulary.values()) or
            set(vocabulary.values()) != set(range(checkpoint.config.vocab_size))):
        raise ValueError('tokenizer vocabulary must match checkpoint logical vocabulary')
    labels = {token: text for text, token in vocabulary.items()}
    probe = ClosedTrigramProbe(checkpoint, selection.ids, 'normalized', pairing)
    records, diagnostics = probe.extract(TOP_K)
    method = 'offset_closed_qkov' + ('_broken_routing' if args.broken_routing_control else '')
    for index, record in enumerate(records):
        record.update(candidate_id=f'{method}:triple{index}', method=method,
                      vocabulary_labels=[labels[t] for t in record['token_ids']],
                      provenance={'metadata_file': sidecar.name, 'ov_pairing': pairing.tolist(),
                                  'input_positions': [0, 1],
                                  'token_embedding_row_offsets_bytes':
                                      [t * checkpoint.config.d_model * 4 for t in record['token_ids']]})
    paths = list(decode_trigram_paths(records, length=PATH_LENGTH, starts=PATH_STARTS,
                                      beam_width=BEAM_WIDTH))
    for index, path in enumerate(paths):
        path.update(candidate_id=f'{method}:path{index}', method=method + '_path',
                    vocabulary_labels=[labels[t] for t in path['token_ids']])
        path['provenance']['metadata_file'] = sidecar.name
    source_pairs = {tuple(record['token_ids'][:2]) for record in records}
    target_pairs = {tuple(record['token_ids'][1:]) for record in records}
    selector_embedding = selection.checkpoint.specs['token_embedding.weight']
    selector_beta = selection.checkpoint.specs['final_norm.bias']
    routing = weight_provenance(checkpoint, 0, pairing)
    routing['score_reads'] = 'geometry_provenance dictionary rows plus all head_slices and original destination E[c]'
    routing['selection_reads'] = 'S from final-selector full E dot final beta; destination ranking reads all original E[c] for c in S'
    metadata = {
        'schema_version': 1, 'stage': 'corpus_blind_extraction', 'numpy_version': np.__version__,
        'protocol': 'fixed final-output-offset vocabulary with normalized first-order closed trigram paths',
        'protocol_manifest': {'file': str(PROTOCOL_PATH), 'sha256': sha256_file(PROTOCOL_PATH)},
        'parameters': dict({key: str(value) if isinstance(value, Path) else value
                            for key, value in vars(args).items()},
                           top_k=TOP_K, path_length=PATH_LENGTH, path_starts=PATH_STARTS,
                           beam_width=BEAM_WIDTH, token_occurrence_limit=2, input_geometry='normalized'),
        'checkpoint': checkpoint.provenance(hash_weights=True),
        'vocabulary_selection_checkpoint': selection.checkpoint_provenance,
        'rankings': {'file': str(args.rankings.resolve()), 'sha256': selection.rankings_sha256,
                     'source_sha256': selection.ranking_source_sha256,
                     'validation': selection.validation},
        'vocabulary_selection': {
            'method': SELECTOR_METHOD, 'rule': 'top128 by final E[t] dot final LayerNorm beta; ID tie break',
            'ids_ascending': selection.ids.tolist(), 'ids_by_offset_rank': selection.ranked_ids.tolist(),
            'ranked_scores': selection.ranked_scores.tolist(),
            'vocabulary_labels_by_rank': [labels[t] for t in selection.ranked_ids],
            'ids_little_endian_uint32_sha256': hashlib.sha256(selection.ids.astype('<u4').tobytes()).hexdigest(),
            'selection_checkpoint_differs_from_evaluation': checkpoint.directory != selection.checkpoint.directory,
            'prior_warning': 'Early uses final-selected S: explicit learned vocabulary prior, not independent untrained selection.',
            'weight_dependencies': {
                'embedding_file': selector_embedding.filename,
                'logical_byte_range': [0, checkpoint.config.vocab_size * checkpoint.config.d_model * 4],
                'selected_row_byte_ranges': [[int(t) * checkpoint.config.d_model * 4,
                                             (int(t) + 1) * checkpoint.config.d_model * 4] for t in selection.ids],
                'shared_beta_file': selector_beta.filename, 'shared_beta_byte_range': selector_beta.byte_range,
                'embedding_shared_with': list(selector_embedding.shared_with),
                'rank_dependencies': 'all logical embedding rows and all final beta entries, not just selected rows',
            },
        },
        'method_counts': dict(Counter(record['method'] for record in records + paths)),
        'diagnostics': dict(diagnostics, source_pair_count=len(source_pairs),
                            target_pair_count=len(target_pairs),
                            source_target_pair_intersection=len(source_pairs & target_pairs),
                            overlap_edges=sum(tuple(record['token_ids'][1:]) in source_pairs for record in records),
                            emitted_paths=len(paths), negative_score_triples=sum(r['score'] < 0 for r in records),
                            candidate_length_histogram=dict(Counter(len(r['token_ids']) for r in records + paths))),
        'geometry_provenance': geometry_provenance(checkpoint, selection.ids, 'normalized'),
        'routing_weight_provenance': routing,
        'epsilon_source_validation': epsilon_check,
        'numerical_geometry_hashes': {
            'X0_little_endian_float64_sha256': hashlib.sha256(probe.inputs0.astype('<f8').tobytes()).hexdigest(),
            'X1_little_endian_float64_sha256': hashlib.sha256(probe.inputs1.astype('<f8').tobytes()).hexdigest(),
        },
        'tokenizer_sha256': sha256_file(vocabulary_path),
        'source_sha256': {name: sha256_file(Path(__file__).with_name(name))
                          for name in ('offset_paths.py', 'closed_trigram.py', 'trigram.py', 'checkpoint.py')},
        'score_definition': 'cosine(sum_h routing_difference * value_difference / 4, ORIGINAL E[c]); signed raw derivative dot recorded; output_offset only selects S',
        'path_policy': 'exact recorded triple overlaps; negative scores retained; no immediate equal tokens or third occurrence; longest then mean edge cosine',
        'limitations': [
            'Fixed selected subset only; this extractor cannot estimate corpus coverage.',
            'First derivative of two-choice attention at uniform routing; not the full attention function.',
            'No uniform base term, residual/direct path, MLP, final normalization, or later blocks.',
            'FP64 ideal coordinates, not actual BF16/FP32 model arithmetic.',
            'Every edge reuses positions0/1; paths do not execute full position-dependent model behavior.',
            'Negative cosines can be retained; neither offset nor path score is a probability.',
            'Connected paths do not prove dataset-specific memorization or passage recovery.',
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata = write_artifacts(args.output, records + paths, metadata)
    print(json.dumps({'output': str(args.output), 'candidate_sha256': metadata['candidate_sha256'],
                      'method_counts': metadata['method_counts'], 'diagnostics': metadata['diagnostics'],
                      'vocabulary_hash': metadata['vocabulary_selection']['ids_little_endian_uint32_sha256']}))


if __name__ == '__main__':
    main()
