"""Verify already-frozen checkpoint-delta token bags, never generate text.

Every extraction source and output is authenticated before a separate start
record is written and before the corpus is opened. The current corpus/native
export only measures the fixed rankings; it never changes their membership.
Conditional sampler replay is explicitly not historical batch authentication.
"""

from __future__ import annotations

import argparse
from collections import Counter
import datetime
from pathlib import Path
import subprocess
import sys

import numpy as np

from . import causal_validation as cv
from . import restart_delta as extraction
from .checkpoint import GPT2Config
from .verify import gpt2_token_bytes, validate_native_tokens
from .vocabulary_verify import checked_tokens


VOCABULARY = GPT2Config().vocab_size
TOP_K = 128
CONTEXT = 1024
ACTUAL_SEED = 17
ALTERNATIVE_FIRST = 10000
ALTERNATIVE_COUNT = 1000
WINDOW_COUNTS = (10, 100)


def expected_names():
    names = {f"endpoint_norm_{boundary}" for boundary in extraction.BOUNDARIES}
    for variant in extraction.VARIANTS:
        for kind in extraction.CENTERS:
            names.add(f"shared_{kind}_{variant}")
            names.update(f"{kind}_{variant}_{boundary}" for boundary in extraction.BOUNDARIES)
    return names | {"shared_endpoint_norm"}


def check_record(record):
    if not isinstance(record, dict) or set(record) != {"path", "bytes", "sha256"}:
        raise ValueError("malformed file identity record")
    if cv.file_record(record["path"]) != record:
        raise ValueError("frozen file or source changed: " + record["path"])


def authenticate(frozen_path):
    """Authenticate all frozen sources/outputs, not reread checkpoint archives.

    The independent final archive rehash is outside this verifier's scope.
    Candidate IDs, score order, shuffles, and shared component annotations are
    checked against the complete hashed score arrays before corpus access.
    """
    frozen_path = Path(frozen_path).resolve()
    identity = cv.file_record(frozen_path)
    frozen = cv.read_json(frozen_path)
    if (frozen.get("schema_version") != 1 or frozen.get("complete") is not True
            or frozen.get("stage") != "checkpoint_delta_token_candidate_freeze_not_sequence_recovery"):
        raise ValueError("candidate run is not completely frozen")
    source_names = {"restart_delta.py", "checkpoint_archive.py", "checkpoint.py", "late_mlp_paths.py",
                    "protocol", "history_inventory"}
    if set(frozen.get("sources", {})) != source_names:
        raise ValueError("missing or extra frozen sources")
    source_dir = Path(__file__).resolve().parent
    for name, record in frozen["sources"].items():
        if name.endswith('.py') and Path(record["path"]) != source_dir / name:
            raise ValueError("frozen source path substituted")
        check_record(record)
    intervals = [(step, step + 10) for boundary in extraction.BOUNDARIES
                 for step in extraction.checkpoint_steps(boundary)[:-1]]
    expected_files = {f"interval_{before}_{after}": f"interval_{before}_{after}.npz"
                      for before, after in intervals}
    expected_files.update(complete_scores="scores.npz", candidates="candidates.json", plan="plan.json")
    if set(frozen.get("files", {})) != set(expected_files):
        raise ValueError("missing or extra frozen output files")
    for name, record in frozen["files"].items():
        if Path(record["path"]) != frozen_path.parent / expected_files[name]:
            raise ValueError("frozen output path substituted")
        check_record(record)
    plan = cv.read_json(frozen["files"]["plan"]["path"])
    if (plan.get("stage") != "weight_only_candidate_plan_no_corpus_or_model"
            or plan.get("sources") != frozen["sources"]
            or plan.get("boundaries") != list(extraction.BOUNDARIES)
            or plan.get("logical_vocab_size") != VOCABULARY
            or plan.get("top_k") != TOP_K or plan.get("seed") != extraction.SEED):
        raise ValueError("plan differs from fixed extraction protocol")
    candidates = cv.read_json(frozen["files"]["candidates"]["path"])
    if (candidates.get("stage") != "frozen_unordered_token_candidates_from_weight_deltas"
            or candidates.get("top_k") != TOP_K
            or candidates.get("no_corpus_or_tokenizer_access") is not True
            or candidates.get("no_model_execution") is not True
            or set(candidates.get("candidate_lists", {})) != expected_names()):
        raise ValueError("candidate list set differs from frozen protocol")
    dates = [datetime.datetime.fromisoformat(value) for value in
             (plan["created_utc_before_weight_reads"], candidates["created_utc"],
              frozen["frozen_utc_before_corpus_verification"])]
    if any(value.tzinfo is None for value in dates) or not dates[0] <= dates[1] <= dates[2]:
        raise ValueError("freeze timestamps are missing, unordered, or timezone-free")
    with np.load(frozen["files"]["complete_scores"]["path"], allow_pickle=False) as arrays:
        if set(arrays.files) != expected_names() | {"identity_permutation"}:
            raise ValueError("complete score key set differs")
        permutation = np.random.Generator(np.random.PCG64(extraction.SEED)).permutation(VOCABULARY)
        if not np.array_equal(arrays["identity_permutation"], permutation):
            raise ValueError("identity shuffle differs from fixed generator")
        for name, ranking in candidates["candidate_lists"].items():
            values = arrays[name]
            if values.shape != (VOCABULARY,) or values.dtype.kind != 'f' or not np.isfinite(values).all():
                raise ValueError("invalid complete score vector")
            order = np.lexsort((np.arange(VOCABULARY), -values))[:TOP_K]
            if (ranking.get("token_ids") != order.tolist()
                    or ranking.get("scores") != values[order].tolist()
                    or ranking.get("identity_shuffled_token_ids") != permutation[order].tolist()):
                raise ValueError("candidate IDs, scores, or shuffled labels changed")
            if name.startswith('shared_') and name != 'shared_endpoint_norm':
                component_key = name.removeprefix('shared_')
                components = np.asarray([arrays[f"{component_key}_{boundary}"][order]
                                         for boundary in extraction.BOUNDARIES])
                if (ranking.get("component_scores") != components.tolist()
                        or ranking.get("positive_component_count") != (components > 0).sum(axis=0).tolist()):
                    raise ValueError("shared-score annotations changed")
    for record in [identity, *frozen["sources"].values(), *frozen["files"].values()]:
        check_record(record)
    return frozen, candidates, identity


def frequencies(tokens, split_token, vocabulary):
    tokens = checked_tokens(tokens, vocabulary)
    if type(split_token) is not int or not 0 < split_token < len(tokens):
        raise ValueError("split must have nonempty prefix and suffix")
    return {name: np.bincount(values, minlength=vocabulary) for name, values in
            (("full", tokens), ("current_prefix", tokens[:split_token]),
             ("current_suffix", tokens[split_token:]))}


def summarize_frequencies(counts, ids):
    ids = checked_tokens(ids, len(counts))
    counts = np.asarray(counts)
    if counts.ndim != 1 or counts.dtype.kind not in 'iu' or np.any(counts < 0) or not len(ids):
        raise ValueError("expected nonnegative frequencies and nonempty IDs")
    if len(np.unique(ids)) != len(ids):
        raise ValueError("ranked IDs must be distinct")
    selected = counts[ids]
    total = int(counts.sum())
    return {"ranked_token_ids_present": int(np.count_nonzero(selected)),
            "ranked_token_ids_absent": int(np.count_nonzero(selected == 0)),
            "presence_fraction": float(np.count_nonzero(selected) / len(ids)),
            "counts_in_rank_order": selected.tolist(),
            "selected_token_occurrences": int(selected.sum()),
            "selected_share_of_corpus_tokens": float(selected.sum() / total) if total else None,
            "minimum_frequency": int(selected.min()), "maximum_frequency": int(selected.max()),
            "mean_frequency": float(selected.mean()), "median_frequency": float(np.median(selected))}


def window_presence(tokens, starts, context, window_count, vocabulary):
    """Union INPUT and TARGET IDs: context+1 tokens per sampled window.

    Repeated token occurrences and overlapping windows count only once in each
    row's bag. This supplies no sequence reconstruction or word-order claim.
    """
    tokens = checked_tokens(tokens, vocabulary)
    starts = np.asarray(starts)
    if (starts.ndim != 2 or starts.dtype.kind not in 'iu'
            or type(context) is not int or context <= 0
            or type(window_count) is not int or not 0 < window_count <= starts.shape[1]
            or starts.shape[0] == 0 or len(tokens) <= context
            or np.any(starts < 0) or np.any(starts > len(tokens) - context - 1)):
        raise ValueError("invalid replay starts or window bounds")
    result = np.zeros((len(starts), vocabulary), dtype=bool)
    within = np.arange(context + 1, dtype=np.int64)
    for index, row in enumerate(starts):
        sampled = tokens[row[:window_count].astype(np.int64)[:, None] + within]
        result[index, sampled] = True
    return result


def summarize_overlap(presence, ids):
    presence = np.asarray(presence)
    if presence.ndim != 2 or presence.dtype.kind != 'b' or len(presence) < 2:
        raise ValueError("need one actual and at least one alternative presence row")
    ids = checked_tokens(ids, presence.shape[1])
    if not len(ids) or len(np.unique(ids)) != len(ids):
        raise ValueError("ranked IDs must be nonempty and distinct")
    overlaps = np.count_nonzero(presence[:, ids], axis=1)
    actual, alternatives = int(overlaps[0]), overlaps[1:]
    greater = int(np.count_nonzero(alternatives > actual))
    equal = int(np.count_nonzero(alternatives == actual))
    return {"seed17_overlap": actual, "seed17_overlap_fraction": actual / len(ids),
            "alternative_overlap_counts_in_seed_order": alternatives.tolist(),
            "alternative_minimum": int(alternatives.min()), "alternative_maximum": int(alternatives.max()),
            "alternative_mean": float(alternatives.mean()), "alternative_median": float(np.median(alternatives)),
            "alternative_histogram": {str(key): value for key, value in sorted(Counter(map(int, alternatives)).items())},
            "alternatives_strictly_greater_than_seed17": greater,
            "alternatives_equal_to_seed17": equal,
            "alternatives_greater_or_equal_to_seed17": greater + equal,
            "seed17_descriptive_rank_among_all_seeds": 1 + greater,
            "rank_scope": "Descriptive overlap rank, not a p-value or authenticated seed identification"}


def sampler_rows(sampler, total, context, seed_start, seed_count):
    command = [str(sampler), str(total), str(context), str(max(WINDOW_COUNTS)),
               str(seed_start), str(seed_count)]
    process = subprocess.run(command, capture_output=True, text=True, timeout=30, check=True)
    if process.stderr or len(process.stdout) > seed_count * (max(WINDOW_COUNTS) + 1) * 22:
        raise ValueError("unexpected sampler output")
    lines = process.stdout.splitlines()
    if len(lines) != seed_count:
        raise ValueError("sampler seed row count differs")
    rows = []
    for index, line in enumerate(lines):
        fields = line.split()
        if len(fields) != max(WINDOW_COUNTS) + 1 or any(not field.isascii() or not field.isdecimal() for field in fields):
            raise ValueError("sampler row is not unsigned decimal starts")
        values = list(map(int, fields))
        if values[0] != seed_start + index or any(not 0 <= value <= total - context - 1 for value in values[1:]):
            raise ValueError("sampler seed order or window range differs")
        rows.append(values[1:])
    return np.asarray(rows, dtype=np.int64), {
        "command": command, "stdout_sha256": cv.digest(process.stdout.encode('utf-8')),
        "seed_start": seed_start, "seed_count": seed_count, "starts_per_seed": rows}


def load_native(corpus_path, tokens_path, offsets_path, tokenizer_path):
    # Called only after the independent verification-start record is on disk.
    from tokenizers import Tokenizer

    corpus = Path(corpus_path).read_bytes()
    if Path(tokens_path).stat().st_size % 4 or Path(offsets_path).stat().st_size % 8:
        raise ValueError("trailing partial native token/offset element")
    tokens = checked_tokens(np.fromfile(tokens_path, dtype='<u4'), VOCABULARY)
    offsets = np.fromfile(offsets_path, dtype='<u8')
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    vocabulary = gpt2_token_bytes(tokenizer)
    if set(vocabulary) != set(range(VOCABULARY)):
        raise ValueError("tokenizer differs from full logical GPT-2 vocabulary")
    validate_native_tokens(tokens, offsets, corpus, vocabulary)
    boundary = cv.split_boundary(corpus)
    split_token = int(np.searchsorted(offsets, boundary))
    if split_token >= len(offsets) or offsets[split_token] != boundary:
        raise ValueError("current byte split is not a native token boundary")
    return corpus, tokens, offsets, vocabulary, boundary, split_token


def verify_files(frozen_path, corpus_path, tokens_path, offsets_path, tokenizer_dir, sampler, output):
    output = Path(output).absolute()
    start_path = output.with_name(output.stem + '_start.json')
    if output.exists() or output.is_symlink() or start_path.exists() or start_path.is_symlink():
        raise FileExistsError("verification output/start must both be new")
    # Do not even hash corpus inputs before this gate and start record.
    frozen, candidates, frozen_identity = authenticate(frozen_path)
    source_dir = Path(__file__).resolve().parent
    sources = {name: cv.file_record(source_dir / name) for name in
               ('restart_delta_verify.py', 'verify.py', 'vocabulary_verify.py',
                'causal_validation.py', 'replay_sampler.cc')}
    sources['dataset.cc'] = cv.file_record(source_dir.parents[1] / 'src/dataset/dataset.cc')
    sources['dataset.h'] = cv.file_record(source_dir.parents[1] / 'src/dataset/dataset.h')
    sampler = Path(sampler).resolve()
    sources['sampler_binary'] = cv.file_record(sampler)
    paths = {"corpus": str(Path(corpus_path).resolve()), "native_tokens": str(Path(tokens_path).resolve()),
             "native_offsets": str(Path(offsets_path).resolve()),
             "tokenizer": str((Path(tokenizer_dir) / 'tokenizer.json').resolve())}
    cv.write_report(start_path, {
        "stage": "verification_start_before_corpus_or_tokenizer_reads",
        "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "frozen": frozen_identity, "candidate_file": frozen['files']['candidates'],
        "authenticated_frozen_sources": frozen['sources'],
        "authenticated_frozen_files": frozen['files'], "verification_sources": sources,
        "corpus_input_paths_not_yet_opened": paths,
        "fixed_conditions": {"context": CONTEXT, "window_counts": list(WINDOW_COUNTS),
                             "actual_seed": ACTUAL_SEED, "alternative_first": ALTERNATIVE_FIRST,
                             "alternative_count": ALTERNATIVE_COUNT},
    })
    start_identity = cv.file_record(start_path)
    inputs = {name: cv.file_record(path) for name, path in paths.items()}
    corpus, tokens, offsets, vocabulary, boundary, split_token = load_native(
        paths['corpus'], paths['native_tokens'], paths['native_offsets'], paths['tokenizer'])
    counts = frequencies(tokens, split_token, VOCABULARY)
    implementation = subprocess.run([str(sampler), '--implementation'], capture_output=True,
                                    text=True, check=True, timeout=30)
    if implementation.stderr:
        raise ValueError("sampler implementation query failed")
    actual, actual_record = sampler_rows(sampler, split_token, CONTEXT, ACTUAL_SEED, 1)
    alternatives, alternatives_record = sampler_rows(sampler, split_token, CONTEXT,
                                                     ALTERNATIVE_FIRST, ALTERNATIVE_COUNT)
    starts = np.concatenate((actual, alternatives), axis=0)
    presence = {str(count): window_presence(tokens[:split_token], starts, CONTEXT, count, VOCABULARY)
                for count in WINDOW_COUNTS}
    results = {}
    for name, ranking in candidates['candidate_lists'].items():
        results[name] = {}
        for kind, field in (('real', 'token_ids'), ('identity_shuffled', 'identity_shuffled_token_ids')):
            ids = ranking[field]
            pieces = []
            for token in ids:
                raw = vocabulary[token]
                try:
                    utf8 = raw.decode('utf-8')
                except UnicodeDecodeError:
                    utf8 = None
                pieces.append({"token_id": token, "raw_hex": raw.hex(),
                               "utf8_if_complete": utf8,
                               "display_escaped": repr(raw.decode('utf-8', errors='backslashreplace'))})
            results[name][kind] = {
                "token_ids_in_frozen_rank_order": ids,
                "individual_pieces_not_a_sentence": pieces,
                "corpus_frequency": {label: summarize_frequencies(values, ids) for label, values in counts.items()},
                "conditional_replay": {count: summarize_overlap(mask, ids) for count, mask in presence.items()},
            }
    # Frozen artifact/source integrity is repeated, without rereading archives.
    authenticated_after, _, after_identity = authenticate(frozen_path)
    if authenticated_after != frozen or after_identity != frozen_identity:
        raise ValueError("frozen extraction changed during verification")
    for record in [*inputs.values(), *sources.values(), start_identity]:
        check_record(record)
    result = {"schema_version": 1, "stage": "frozen_delta_token_bag_verification_not_text_reconstruction",
              "complete": True, "completed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "verification_start": start_identity, "frozen": frozen_identity,
              "inputs": inputs, "sources": sources,
              "runtime": {"python": sys.version, "numpy": np.__version__,
                          "sampler_implementation": implementation.stdout.strip()},
              "native_export_every_byte_validated": True,
              "split": {"byte_boundary": boundary, "native_token_boundary": split_token,
                        "corpus_bytes": len(corpus), "corpus_tokens": len(tokens),
                        "prefix_tokens": split_token, "suffix_tokens": len(tokens) - split_token},
              "corpus_summary": {label: {"token_occurrences": int(values.sum()),
                                          "distinct_token_ids": int(np.count_nonzero(values))}
                                 for label, values in counts.items()},
              "conditional_replay": {"context": CONTEXT, "window_counts": list(WINDOW_COUNTS),
                                     "tokens_per_window_including_input_and_target": CONTEXT + 1,
                                     "sequence_batch_sizes_for_ten_steps": [1, 10],
                                     "actual": actual_record, "alternatives": alternatives_record,
                                     "distinct_token_ids_per_seed": {count: mask.sum(axis=1).tolist()
                                                                    for count, mask in presence.items()}},
              "results": results, "all_frozen_artifacts_and_verification_inputs_unchanged": True,
              "checkpoint_archives_rehashed_in_this_verifier": False,
              "limitations": ["Unordered token pieces, not reconstructed sentences or passages.",
                              "Current prefix/suffix do not authenticate historical training membership.",
                              "Seed 17 replay assumes unchanged corpus, split, seed, context, and current C++ sampler.",
                              "1000 fixed alternative seeds provide descriptive overlap ranks, not p-values.",
                              "Timestamp gaps do not authenticate restarts; weight deltas are not raw gradients.",
                              "Full checkpoint archive final rehash is a separate audit."]}
    cv.write_report(output, result)
    print(f"Verified {len(results)} rankings and all identity shuffles; no text sequence reconstructed.", flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('frozen', 'corpus', 'native-tokens', 'native-offsets', 'tokenizer-dir', 'sampler', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args(argv)
    verify_files(args.frozen, args.corpus, args.native_tokens, args.native_offsets,
                 args.tokenizer_dir, args.sampler, args.output)


if __name__ == '__main__':
    main()
