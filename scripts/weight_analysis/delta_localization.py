"""Locate corpus passages with frozen delta token bags; never decode a sequence.

The supplied corpus provides the words AND their order. Weight-derived IDs only
score those known passages. No replay starts, sampler, model, or checkpoint
weights are read here. A separate verifier may compare the frozen retrievals
with conditional training replay afterward.

CLI inputs are --restart-frozen, --followthrough-frozen, --corpus,
--native-tokens, --native-offsets, --tokenizer-dir, --protocol, --output-dir.
The output directory must not already exist. The plan precedes corpus access;
all selected windows and full score arrays precede the final completion marker.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import platform

import numpy as np

from . import delta_followthrough_verify as later
from . import restart_delta_verify as earlier
from .late_mlp_paths import file_record, write_exclusive
from .vocabulary_verify import checked_tokens


VOCABULARY = 50257
TOP_K = 128
WINDOW = 1025
MAX_WINDOWS = 100
CONTROL_COUNT = 100
CONTROL_SEED = 20260909
PRIMARY = 'restart.shared_boundary_adjusted.real'
RESTART_NAMES = tuple(f'shared_{kind}_{variant}'
                      for kind in ('boundary', 'ordinary_before', 'ordinary_after')
                      for variant in ('raw', 'adjusted')) + ('shared_endpoint_norm',)
FOLLOWTHROUGH_NAMES = ('shared_followthrough_raw', 'shared_followthrough_adjusted')


def _positive_int(value, label, maximum):
    if type(value) is not int or not 0 < value <= maximum:
        raise ValueError(f'{label} must be an integer in 1..{maximum}')


def _ids(values, vocabulary):
    values = checked_tokens(values, vocabulary)
    if not 0 < len(values) <= 65535 or len(np.unique(values)) != len(values):
        raise ValueError('candidate IDs must be nonempty, unique, and fit uint16 counts')
    return values


class CorpusIndex:
    """An inverted index of exact native token positions, in increasing order.

    Stable sorting groups equal IDs without changing occurrence order. One
    shared index avoids rescanning the entire corpus for every candidate ID.
    Only integer token IDs are accepted; nothing is retokenized.
    """

    def __init__(self, tokens, vocabulary=VOCABULARY):
        self.tokens = checked_tokens(tokens, vocabulary).copy()
        if not 0 < len(self.tokens) <= np.iinfo(np.int32).max:
            raise ValueError('corpus must contain 1..INT32_MAX tokens')
        self.vocabulary = vocabulary
        self.counts = np.bincount(self.tokens, minlength=vocabulary)
        self.order = np.argsort(self.tokens, kind='stable')
        self.offsets = np.concatenate(([0], np.cumsum(self.counts, dtype=np.int64)))
        for array in (self.tokens, self.counts, self.order, self.offsets):
            array.flags.writeable = False

    def occurrences(self, token):
        if type(token) is not int or not 0 <= token < self.vocabulary:
            raise ValueError('occurrence token outside vocabulary')
        return self.order[self.offsets[token]:self.offsets[token + 1]]


def distinct_window_scores(index, ids, window=WINDOW):
    """Count distinct candidate IDs in EVERY legal full-corpus sliding window.

    An occurrence at position p contributes to starts [p-window+1, p], clipped
    to legal starts. Union these intervals *within each ID* before incrementing
    a difference array. A cumulative sum then gives exact sliding distinct-ID
    counts: repeated/adjacent occurrences cannot count the same ID twice.
    The last legal start len(tokens)-window is included. No sampling is used.
    """
    if not isinstance(index, CorpusIndex):
        raise ValueError('expected CorpusIndex')
    _positive_int(window, 'window length', len(index.tokens))
    ids = _ids(ids, index.vocabulary)
    count = len(index.tokens) - window + 1
    difference = np.zeros(count + 1, dtype=np.int32)
    for token in ids:
        positions = index.occurrences(int(token))
        if not len(positions):
            continue
        starts = np.maximum(0, positions - window + 1)
        ends = np.minimum(count, positions + 1)
        # Starts and ends are both monotone. Touching intervals can be merged
        # too: their union has no uncovered legal start between them.
        groups = np.concatenate(([0], np.flatnonzero(starts[1:] > ends[:-1]) + 1))
        last = np.concatenate((groups[1:] - 1, [len(starts) - 1]))
        np.add.at(difference, starts[groups], 1)
        np.add.at(difference, ends[last], -1)
    scores = np.cumsum(difference[:-1], dtype=np.int32)
    if np.any(scores < 0) or np.any(scores > len(ids)) or difference.sum() != 0:
        raise AssertionError('distinct-count interval invariant failed')
    return scores.astype('<u2')


def _scores(values):
    values = np.asarray(values)
    if (values.ndim != 1 or values.dtype.kind not in 'iu' or not len(values)
            or np.any(values < 0) or np.any(values > 65535)):
        raise ValueError('scores must be a nonempty uint16-range integer vector')
    return values


def score_plateau(scores, start):
    """Return [begin,end) of the connected equal-SCORE plateau at this start.

    Equal scores need not have equal supporting token sets. Plateau coordinates
    are window START indices, not corpus token/byte spans or confidence bounds.
    """
    scores = _scores(scores)
    if type(start) is not int or not 0 <= start < len(scores):
        raise ValueError('plateau start outside score array')
    begin, end = start, start + 1
    while begin and scores[begin - 1] == scores[start]:
        begin -= 1
    while end < len(scores) and scores[end] == scores[start]:
        end += 1
    return begin, end


def select_windows(scores, window=WINDOW, limit=MAX_WINDOWS):
    """Greedy positive-score nonoverlap, score descending / start ascending.

    This does not maximize a global coverage objective. Suppression only blocks
    windows sharing a token position; adjacent half-open windows are allowed.
    Original scores are not modified, and plateau bounds are unsuppressed.
    """
    scores = _scores(scores)
    _positive_int(window, 'window length', np.iinfo(np.int32).max)
    _positive_int(limit, 'retrieval limit', 10000)
    available = scores.astype(np.int32)
    bounds = np.concatenate(([0], np.flatnonzero(scores[1:] != scores[:-1]) + 1,
                             [len(scores)]))
    selected = []
    for rank in range(1, limit + 1):
        start = int(np.argmax(available))  # NumPy returns the first maximum.
        score = int(available[start])
        if score <= 0:
            break
        plateau = int(np.searchsorted(bounds, start, side='right') - 1)
        selected.append(dict(rank=rank, start=start, end=start + window, score=score,
                             plateau_start=int(bounds[plateau]),
                             plateau_end=int(bounds[plateau + 1])))
        available[max(0, start - window + 1):min(len(scores), start + window)] = -1
    return selected


def frequency_buckets(counts):
    """Exact integer floor(log2(count)); absent IDs receive bucket -1."""
    counts = np.asarray(counts)
    if (counts.ndim != 1 or counts.dtype.kind not in 'iu' or not len(counts)
            or np.any(counts < 0)):
        raise ValueError('frequencies must be nonnegative integer counts')
    return np.fromiter((int(value).bit_length() - 1 for value in counts),
                       dtype=np.int16, count=len(counts))


def frequency_range_controls(counts, ids, count=CONTROL_COUNT, seed=CONTROL_SEED):
    """Data-assisted label permutations preserving ranges, NOT exact counts.

    For each replicate, visit all buckets in ascending order (-1 first), then
    permute their ascending token IDs with one continuous PCG64 stream. The
    resulting full-vocabulary bijection maps the primary IDs in original rank
    order. IDs remain unique; fixed points and overlap with the original bag
    are retained, not redrawn to improve a result.
    """
    buckets = frequency_buckets(counts)
    ids = _ids(ids, len(buckets))
    _positive_int(count, 'control count', 1000)
    if type(seed) is not int or not 0 <= seed < 2**128:
        raise ValueError('control seed must be an unsigned 128-bit integer')
    groups = [np.flatnonzero(buckets == bucket) for bucket in np.unique(buckets)]
    rng = np.random.Generator(np.random.PCG64(seed))
    result = []
    for _ in range(count):
        permutation = np.empty(len(buckets), dtype=np.int32)
        for group in groups:
            permutation[group] = rng.permutation(group)
        result.append(permutation[ids].copy())
    return result


def panel_lists(restart, followthrough):
    """The fixed 18-list panel; no ranking or model/phase choice is made here."""
    result = {}
    for experiment, candidates, names in (('restart', restart, RESTART_NAMES),
                                          ('followthrough', followthrough, FOLLOWTHROUGH_NAMES)):
        for name in names:
            for label, field in (('real', 'token_ids'),
                                 ('identity_shuffled', 'identity_shuffled_token_ids')):
                ids = _ids(candidates['candidate_lists'][name][field], VOCABULARY)
                if len(ids) != TOP_K:
                    raise ValueError('panel bag size differs from frozen protocol')
                key = f'{experiment}.{name}.{label}'
                result[key] = dict(panel=experiment, source_list=name, label=label,
                                   token_ids=ids.tolist())
    return result


def add_window_support(index, ids, windows, offsets):
    """Attach exact native occurrence and byte ranges to each selected window."""
    ids = _ids(ids, index.vocabulary)
    offsets = np.asarray(offsets)
    if (offsets.shape != (len(index.tokens) + 1,) or offsets.dtype.kind not in 'iu'
            or offsets[0] != 0 or np.any(offsets[1:] <= offsets[:-1])):
        raise ValueError('native byte offsets must be strictly increasing N+1 integers')
    for item in windows:
        start, end = item['start'], item['end']
        if not 0 <= start < end <= len(index.tokens):
            raise ValueError('selected window outside corpus')
        support = []
        for token in ids:
            positions = index.occurrences(int(token))
            lo, hi = np.searchsorted(positions, [start, end])
            selected = positions[lo:hi]
            if len(selected):
                support.append(dict(token_id=int(token), token_positions=selected.tolist(),
                                    byte_ranges=[[int(offsets[p]), int(offsets[p + 1])]
                                                 for p in selected]))
        if len(support) != item['score']:
            raise ValueError('saved window score differs from distinct support')
        item.update(byte_start=int(offsets[start]), byte_end=int(offsets[end]), support=support)
    return windows


def _sources(protocol):
    directory = Path(__file__).resolve().parent
    names = ('delta_localization.py', 'restart_delta_verify.py', 'delta_followthrough_verify.py',
             'restart_delta.py', 'delta_followthrough.py', 'checkpoint_archive.py',
             'checkpoint.py', 'late_mlp_paths.py', 'causal_validation.py', 'verify.py',
             'vocabulary_verify.py', 'late_mlp_polynomial.py', 'path_diagnostics.py')
    return {name: file_record(directory / name) for name in names} | {'protocol': file_record(protocol)}


def _write_npz(path, **arrays):
    with Path(path).open('xb') as stream:
        np.savez_compressed(stream, **arrays)
    return file_record(path)


def run(restart_frozen, followthrough_frozen, corpus, native_tokens, native_offsets,
        tokenizer_dir, protocol, output_dir):
    output = Path(output_dir).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    restart, restart_candidates, restart_identity = earlier.authenticate(restart_frozen)
    followthrough, followthrough_candidates, followthrough_identity = later.authenticate(followthrough_frozen)
    panels = panel_lists(restart_candidates, followthrough_candidates)
    sources = _sources(protocol)
    upstream = dict(restart=restart_identity, followthrough=followthrough_identity)
    paths = dict(corpus=str(Path(corpus).resolve()), native_tokens=str(Path(native_tokens).resolve()),
                 native_offsets=str(Path(native_offsets).resolve()),
                 tokenizer=str((Path(tokenizer_dir)/'tokenizer.json').resolve()))
    output.mkdir(parents=True, exist_ok=False)
    # Crucially: path resolution above is not corpus reading. No content hashes,
    # tokenizer loading, or corpus scans are allowed before this exclusive plan.
    plan = dict(schema_version=1, stage='localization_plan_before_corpus_access_no_replay',
                created_utc=datetime.now(timezone.utc).isoformat(), sources=sources,
                upstream_freezes=upstream, input_paths_not_yet_opened=paths,
                window_length=WINDOW, top_k_windows=MAX_WINDOWS,
                candidate_count=TOP_K, primary_list=PRIMARY,
                panel_lists=panels, frequency_control_count=CONTROL_COUNT,
                frequency_control_seed=CONTROL_SEED,
                frequency_buckets='floor(log2(full_count)); absent=-1',
                selection='distinct IDs; positive scores; descending score, ascending start; greedy nonoverlap',
                numpy_version=np.__version__, python_version=platform.python_version(),
                no_sampler_access=True, no_model_execution=True, no_checkpoint_weight_reads=True)
    write_exclusive(output/'plan.json', plan)
    plan_identity = file_record(output/'plan.json')
    inputs = {name: file_record(path) for name, path in paths.items()}
    raw_corpus, tokens, offsets, vocabulary, boundary, prefix = earlier.load_native(
        paths['corpus'], paths['native_tokens'], paths['native_offsets'], paths['tokenizer'])
    index = CorpusIndex(tokens, VOCABULARY)
    _positive_int(WINDOW, 'window length', len(tokens))
    primary_ids = panels[PRIMARY]['token_ids']
    buckets = frequency_buckets(index.counts)
    controls = frequency_range_controls(index.counts, primary_ids, CONTROL_COUNT, CONTROL_SEED)
    for number, ids in enumerate(controls):
        panels[f'frequency_range_control.{number:03}'] = dict(
            panel='frequency_range_control', source_list=PRIMARY, label='data_assisted_frequency_range',
            control_index=number, token_ids=ids.tolist(),
            unchanged_rank_positions=int(np.count_nonzero(ids == np.asarray(primary_ids))),
            shared_ids_with_primary=len(set(map(int, ids)) & set(primary_ids)))
    counts_record = _write_npz(output/'corpus_counts.npz', counts=index.counts, buckets=buckets)
    score_records = {}
    for name, item in panels.items():
        ids = item['token_ids']
        scores = distinct_window_scores(index, ids, WINDOW)
        score_records[name] = _write_npz(output/f'{name}.scores.npz', scores=scores)
        windows = add_window_support(index, ids, select_windows(scores, WINDOW, MAX_WINDOWS), offsets)
        item.update(score_file=score_records[name], score_array_key='scores',
                    score_dtype='<u2', legal_window_count=len(scores),
                    maximum_score=int(scores.max()), positive_window_count=int(np.count_nonzero(scores)),
                    corpus_frequencies=index.counts[ids].tolist(),
                    frequency_buckets=buckets[ids].tolist(),
                    pieces=[dict(token_id=t, raw_hex=vocabulary[t].hex()) for t in ids],
                    selected_windows=windows)
        print(f'{name}: max distinct IDs {int(scores.max())}; {len(windows)} nonoverlapping windows', flush=True)
    # Authenticate upstream candidates/sources again, not checkpoint archives.
    if earlier.authenticate(restart_frozen) != (restart, restart_candidates, restart_identity):
        raise ValueError('restart freeze changed during localization')
    if later.authenticate(followthrough_frozen) != (followthrough, followthrough_candidates, followthrough_identity):
        raise ValueError('follow-through freeze changed during localization')
    for record in [*sources.values(), *inputs.values(), *upstream.values(), plan_identity,
                   counts_record, *score_records.values()]:
        earlier.check_record(record)
    result = dict(schema_version=1, complete=True,
                  stage='corpus_assisted_delta_localization_not_sequence_decoding',
                  completed_utc=datetime.now(timezone.utc).isoformat(),
                  window_length=WINDOW, top_k_windows=MAX_WINDOWS, primary_list=PRIMARY,
                  corpus=dict(total_tokens=len(tokens), prefix_tokens=prefix,
                              byte_boundary=boundary, total_bytes=len(raw_corpus)),
                  lists=panels, sources=sources, inputs=inputs, upstream_freezes=upstream,
                  plan=plan_identity, corpus_counts=counts_record,
                  native_export_every_byte_validated=True,
                  no_sampler_access=True, no_model_execution=True,
                  all_inputs_sources_upstream_and_scores_unchanged=True,
                  limitations=[
                      'Corpus-assisted passage retrieval: ordered text comes from the corpus, not weights.',
                      'Retrospective use of a previously successful bag; not a new independent history.',
                      'Score plateaus expose start ambiguity and need not share the same supporting IDs.',
                      'Frequency-range-matched controls use the corpus and are not calibrated nulls.',
                      'No historical membership claim or rescue of the negative follow-through result.'])
    write_exclusive(output/'localization.json', result)
    files = dict(plan=plan_identity, localization=file_record(output/'localization.json'),
                 corpus_counts=counts_record)
    files.update({f'scores.{name}': record for name, record in score_records.items()})
    for record in [*files.values(), *sources.values(), *inputs.values(), *upstream.values()]:
        earlier.check_record(record)
    write_exclusive(output/'frozen.json', dict(
        schema_version=1, complete=True, stage='corpus_assisted_localization_freeze_before_replay',
        frozen_utc=datetime.now(timezone.utc).isoformat(), files=files,
        sources=sources, inputs=inputs, upstream_freezes=upstream,
        no_sampler_access=True, no_model_execution=True))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('restart-frozen', 'followthrough-frozen', 'corpus', 'native-tokens',
                 'native-offsets', 'tokenizer-dir', 'protocol', 'output-dir'):
        parser.add_argument('--' + name, required=True, type=Path)
    args = parser.parse_args(argv)
    run(args.restart_frozen, args.followthrough_frozen, args.corpus, args.native_tokens,
        args.native_offsets, args.tokenizer_dir, args.protocol, args.output_dir)


if __name__ == '__main__':
    main()
