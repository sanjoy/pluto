#!/usr/bin/env python3
"""Verify frozen weight-derived token candidates, without running a model.

This is a verifier, NOT an extractor. Only this stage reads the corpus. Freeze
the candidate JSONL before using this script: choosing candidates, methods, or
hyperparameters after looking at these matches is corpus-assisted search, not
independent extraction from weights. Exact matches to common pairs are weak
evidence, and even longer matches do not by themselves establish memorization.

The input is one JSON object per line, containing candidate_id (unique string),
method (string), token_ids (nonempty integer list), and arbitrary provenance.
The output preserves each original object and measures every contiguous
candidate substring, not just matches beginning at its first token. A single
seeded token-label permutation provides a negative control; it preserves all
candidate equality/repetition structure but NOT natural token frequencies.
It is therefore a diagnostic, not a calibrated memorization significance test.

NumPy is required. The tokenizers package is needed only for the command line;
the matching functions operate directly on integer IDs and need no tokenizer.
For exact comparison with a training implementation, supply its exported token
IDs and byte offsets. Loading tokenizer.json through Hugging Face can otherwise
differ from a native implementation, even when both use the same vocabulary.
"""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


MATCH_LENGTHS = (2, 4, 8)


class CorpusIndex:
    """Compact inverted index with exact, vectorized substring comparisons.

    We store one position per corpus token, grouped by ID, rather than a Python
    dictionary for millions of corpus n-grams. To find a pair we anchor on its
    rarer token; extending it filters the already-matching positions. Sorting
    positions stably makes every tie deterministic: earliest candidate offset,
    then earliest corpus offset. Counts include overlapping corpus occurrences.
    """

    def __init__(self, tokens: np.ndarray, vocab_size: int):
        tokens = np.asarray(tokens)
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if tokens.ndim != 1 or not np.issubdtype(tokens.dtype, np.integer):
            raise ValueError("corpus tokens must be a one-dimensional integer array")
        if tokens.size and (tokens.min() < 0 or tokens.max() >= vocab_size):
            raise ValueError("corpus token ID outside vocabulary")
        self.tokens = tokens.astype(np.int32, copy=False)
        self.vocab_size = vocab_size
        self.counts = np.bincount(self.tokens, minlength=vocab_size)
        self.bounds = np.concatenate(([0], np.cumsum(self.counts)))
        # Int32 positions save memory for ordinary corpora. Large corpora retain
        # int64 positions; token IDs remain int32, as in the training dataset.
        position_type = np.int32 if len(tokens) <= np.iinfo(np.int32).max else np.int64
        self.positions = np.argsort(self.tokens, kind="stable").astype(position_type)

    def positions_of(self, token: int) -> np.ndarray:
        return self.positions[self.bounds[token] : self.bounds[token + 1]]

    def pair_positions(self, first: int, second: int) -> np.ndarray:
        if self.counts[first] <= self.counts[second]:
            starts = self.positions_of(first)
            starts = starts[starts + 1 < len(self.tokens)]
            return starts[self.tokens[starts + 1] == second]
        starts = self.positions_of(second)
        starts = starts[starts > 0] - 1
        return starts[self.tokens[starts] == first]

    def analyze(self, candidate: tuple[int, ...]) -> dict[str, Any]:
        """Find exact matches, without modifying or generating any candidates.

        The selected longest substring uses the earliest candidate offset when
        multiple different substrings tie. Its occurrence count is the frequency
        of THAT substring, not the combined frequency of all tied substrings.
        """
        if not candidate or any(
            not isinstance(token, (int, np.integer))
            or isinstance(token, (bool, np.bool_))
            or token < 0
            or token >= self.vocab_size
            for token in candidate
        ):
            raise ValueError("candidate must contain valid integer vocabulary IDs")
        longest = {
            "length": 0,
            "candidate_token_start": None,
            "corpus_token_start": None,
            "corpus_token_end": None,
            "corpus_occurrences": 0,
            "unique_in_corpus": False,
        }
        matched_windows = {length: 0 for length in MATCH_LENGTHS}
        distinct: dict[int, dict[tuple[int, ...], int]] = {
            length: {} for length in MATCH_LENGTHS
        }

        def consider(start: int, length: int, positions: np.ndarray) -> None:
            if length > longest["length"]:
                corpus_start = int(positions[0])
                longest.update(
                    length=length,
                    candidate_token_start=start,
                    corpus_token_start=corpus_start,
                    corpus_token_end=corpus_start + length,
                    corpus_occurrences=len(positions),
                    unique_in_corpus=len(positions) == 1,
                )
            if length in distinct:
                matched_windows[length] += 1
                distinct[length][candidate[start : start + length]] = len(positions)

        for start, token in enumerate(candidate):
            # Singleton matches are reported, but never counted as meaningful
            # multi-token extraction. They also handle candidates of length one.
            if longest["length"] == 0 and self.counts[token]:
                consider(start, 1, self.positions_of(token))
            if start + 1 == len(candidate):
                continue
            positions = self.pair_positions(token, candidate[start + 1])
            length = 2
            while positions.size:
                consider(start, length, positions)
                if start + length == len(candidate):
                    break
                positions = positions[positions + length < len(self.tokens)]
                positions = positions[self.tokens[positions + length] == candidate[start + length]]
                length += 1

        return {
            "longest_match": longest,
            "ngrams": {
                str(length): {
                    "candidate_windows": max(0, len(candidate) - length + 1),
                    "matched_candidate_windows": matched_windows[length],
                    "matched_distinct_ngrams": len(distinct[length]),
                    "corpus_occurrences_of_distinct_ngrams": sum(distinct[length].values()),
                }
                for length in MATCH_LENGTHS
            },
        }


def validate_candidates(candidates: list[dict[str, Any]], vocab_size: int) -> None:
    seen = set()
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise ValueError("every candidate must be a JSON object")
        identifier = candidate.get("candidate_id")
        if not isinstance(identifier, str) or not identifier or identifier in seen:
            raise ValueError("candidate_id must be a unique nonempty string")
        seen.add(identifier)
        if not isinstance(candidate.get("method"), str) or not candidate["method"]:
            raise ValueError("method must be a nonempty string")
        tokens = candidate.get("token_ids")
        if not isinstance(tokens, list) or not tokens:
            raise ValueError("token_ids must be a nonempty list")
        if any(type(token) is not int or token < 0 or token >= vocab_size for token in tokens):
            raise ValueError(f"candidate {identifier}: token ID outside vocabulary or not an integer")


def _summarize(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    lengths = [metric["longest_match"]["length"] for metric in metrics]
    return {
        "candidates": len(metrics),
        "max_longest_match_tokens": max(lengths, default=0),
        "mean_longest_match_tokens": sum(lengths) / len(lengths) if lengths else 0.0,
        "candidates_with_match_at_least": {
            str(length): sum(value >= length for value in lengths)
            for length in MATCH_LENGTHS
        },
        "candidates_with_unique_longest_match_at_least_8": sum(
            metric["longest_match"]["length"] >= 8
            and metric["longest_match"]["unique_in_corpus"]
            for metric in metrics
        ),
    }


def verify_candidates(
    corpus_tokens: np.ndarray,
    candidates: list[dict[str, Any]],
    vocab_size: int,
    seed: int = 17,
) -> dict[str, Any]:
    """Return a deterministic report for already-fixed token candidates.

    A single PCG64 permutation is reused across methods and records. Repeated
    candidates share cached matching work; records are still counted separately
    because their provenance can differ. Method summaries expose this duplicate
    count rather than presenting repeated proposals as independent discoveries.
    """
    validate_candidates(candidates, vocab_size)
    index = CorpusIndex(corpus_tokens, vocab_size)
    permutation = np.random.Generator(np.random.PCG64(seed)).permutation(vocab_size)
    cache: dict[tuple[int, ...], dict[str, Any]] = {}

    def analyze(tokens: tuple[int, ...]) -> dict[str, Any]:
        if tokens not in cache:
            cache[tokens] = index.analyze(tokens)
        return cache[tokens]

    records = []
    methods = defaultdict(list)
    for candidate in candidates:
        original = tuple(candidate["token_ids"])
        shuffled = tuple(int(permutation[token]) for token in original)
        record = {
            "candidate": candidate,
            "real": analyze(original),
            "shuffled_labels": {"token_ids": list(shuffled), **analyze(shuffled)},
        }
        records.append(record)
        methods[candidate["method"]].append(record)

    def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
        summaries = {}
        for kind in ("real", "shuffled_labels"):
            summaries[kind] = _summarize([record[kind] for record in records])
            # Many graph paths can contain the very same short formatting
            # substring. Count those substrings, not just candidate records.
            selected = []
            for record in records:
                longest = record[kind]["longest_match"]
                if longest["length"]:
                    ids = (record["candidate"]["token_ids"] if kind == "real"
                           else record[kind]["token_ids"])
                    start = longest["candidate_token_start"]
                    selected.append(tuple(ids[start:start + longest["length"]]))
            summaries[kind]["distinct_selected_longest_substrings_at_least"] = {
                str(length): len({ids for ids in selected if len(ids) >= length})
                for length in MATCH_LENGTHS
            }
        return {
            "distinct_candidate_sequences": len(
                {tuple(record["candidate"]["token_ids"]) for record in records}
            ),
            **summaries,
        }

    return {
        "schema_version": 1,
        "corpus_token_count": len(index.tokens),
        "vocab_size": vocab_size,
        "negative_control": {
            "kind": "global_token_label_permutation",
            "seed": seed,
            "generator": "numpy.random.PCG64",
            "numpy_version": np.__version__,
            "permutation_sha256": hashlib.sha256(permutation.astype("<i4").tobytes()).hexdigest(),
            "limitation": "Preserves candidate repetition structure, not corpus token frequencies; not a significance test.",
        },
        "summary": summarize(records),
        "by_method": {method: summarize(methods[method]) for method in sorted(methods)},
        "results": records,
    }


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def gpt2_token_bytes(tokenizer: Any) -> dict[int, bytes]:
    """Invert GPT-2's reversible byte-to-Unicode vocabulary spelling.

    Decoding one token through a Unicode-string API is insufficient: a token
    may contain only part of a multibyte UTF-8 character. The vocabulary spelling
    retains its original bytes, which lets us verify native token boundaries
    exactly. Special tokens spell their literal text instead of byte encoding.
    """
    direct = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    unicode_points = list(direct)
    next_point = 256
    for byte in range(256):
        if byte not in direct:
            direct.append(byte)
            unicode_points.append(next_point)
            next_point += 1
    # There are 188 directly represented bytes. Assign the remaining bytes
    # consecutively to U+0100 and above, exactly as GPT-2's bytes_to_unicode.
    decoder = {chr(point): byte for point, byte in zip(unicode_points, direct)}
    config = json.loads(tokenizer.to_str())
    specials = {token["id"] for token in config.get("added_tokens", []) if token.get("special")}
    output = {}
    for label, token in tokenizer.get_vocab(with_added_tokens=True).items():
        if token in specials:
            output[token] = label.encode("utf-8")
        else:
            try:
                output[token] = bytes(decoder[character] for character in label)
            except KeyError as error:
                raise ValueError("native byte validation requires a GPT-2 byte-level vocabulary") from error
    return output


def validate_native_tokens(
    tokens: np.ndarray, offsets: np.ndarray, corpus: bytes, vocabulary: dict[int, bytes]
) -> None:
    """Check a frozen native export against every original corpus byte.

    IDs alone are not enough: two implementations may tokenize the same bytes
    differently. This validates the PROVIDED boundaries without silently
    re-tokenizing them. It does not establish who produced the export; hashes
    and the native exporter's provenance are still needed for that claim.
    """
    if offsets.ndim != 1 or len(offsets) != len(tokens) + 1:
        raise ValueError("native byte offsets must contain N+1 entries")
    if offsets[0] != 0 or offsets[-1] != len(corpus):
        raise ValueError("native byte offsets must span the complete corpus")
    if np.any(offsets[1:] <= offsets[:-1]):
        raise ValueError("native byte offsets must be strictly increasing")
    for index, token in enumerate(tokens):
        expected = vocabulary.get(int(token))
        start, end = int(offsets[index]), int(offsets[index + 1])
        if expected is None or corpus[start:end] != expected:
            raise ValueError(f"native token {index} does not match its corpus byte interval")


def add_native_text_spans(
    report: dict[str, Any], corpus: bytes, offsets: np.ndarray, tokenizer: Any
) -> None:
    """Display exact native byte spans with Unicode-safe character bounds.

    A match may start/end inside a multibyte character. Byte offsets remain the
    exact token interval, while the displayed text and character offsets bound
    all characters it touches. This distinction prevents replacement characters
    in a display from being mistaken for a failed exact byte match.
    """
    requested = {0}
    spans = []
    for record in report["results"]:
        for key in ("real", "shuffled_labels"):
            result = record[key]
            ids = record["candidate"]["token_ids"] if key == "real" else result["token_ids"]
            result["decoded_candidate"] = tokenizer.decode(ids, skip_special_tokens=False)
            longest = result["longest_match"]
            if not longest["length"]:
                continue
            start = int(offsets[longest["corpus_token_start"]])
            end = int(offsets[longest["corpus_token_end"]])
            char_start, char_end = start, end
            while char_start and corpus[char_start] & 0xC0 == 0x80:
                char_start -= 1
            while char_end < len(corpus) and corpus[char_end] & 0xC0 == 0x80:
                char_end += 1
            requested.update((char_start, char_end))
            spans.append((longest, start, end, char_start, char_end))
    character_positions = {}
    previous_byte = character_position = 0
    for position in sorted(requested):
        character_position += len(corpus[previous_byte:position].decode("utf-8"))
        character_positions[position] = character_position
        previous_byte = position
    for longest, start, end, char_start, char_end in spans:
        longest.update(
            corpus_byte_start=start,
            corpus_byte_end=end,
            corpus_character_start=character_positions[char_start],
            corpus_character_end=character_positions[char_end],
            text=corpus[char_start:char_end].decode("utf-8"),
            text_is_character_bounding_span=(start != char_start or end != char_end),
        )


def add_text_spans(
    report: dict[str, Any], text: str, token_offsets: np.ndarray, tokenizer: Any
) -> None:
    """Attach display-only text and half-open character/UTF-8 byte intervals.

    Tokenizer offsets can overlap for byte-level pieces of a Unicode character.
    Therefore these intervals bound the original characters touched by the
    matched tokens; they need not re-tokenize to precisely the matched IDs.
    Token offsets, not displayed snippets, define the exact-match result.
    """
    requested = {0}
    spans = []
    for record in report["results"]:
        for key in ("real", "shuffled_labels"):
            result = record[key]
            ids = record["candidate"]["token_ids"] if key == "real" else result["token_ids"]
            result["decoded_candidate"] = tokenizer.decode(ids, skip_special_tokens=False)
            longest = result["longest_match"]
            if longest["length"]:
                first, last = longest["corpus_token_start"], longest["corpus_token_end"]
                start = int(token_offsets[first, 0])
                end = int(token_offsets[last - 1, 1])
                requested.update((start, end))
                spans.append((longest, start, end))
    # Encoding only the disjoint intervals between requested positions avoids
    # either an O(text length) Python mapping or repeatedly encoding long prefixes.
    byte_positions = {}
    previous_char = byte_position = 0
    for position in sorted(requested):
        byte_position += len(text[previous_char:position].encode("utf-8"))
        byte_positions[position] = byte_position
        previous_char = position
    for longest, start, end in spans:
        longest.update(
            corpus_character_start=start,
            corpus_character_end=end,
            corpus_byte_start=byte_positions[start],
            corpus_byte_end=byte_positions[end],
            text=text[start:end],
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True, type=Path)
    tokenizer_group = parser.add_mutually_exclusive_group(required=True)
    tokenizer_group.add_argument("--tokenizer-dir", type=Path)
    tokenizer_group.add_argument("--tokenizer-json", type=Path)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--corpus-token-ids", type=Path,
                        help="Native corpus export: raw little-endian uint32 token IDs.")
    parser.add_argument("--corpus-byte-offsets", type=Path,
                        help="Native export: raw little-endian uint64 offsets, one more than IDs.")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    from tokenizers import Tokenizer  # Optional for users of the integer-only API.

    tokenizer_path = args.tokenizer_json or args.tokenizer_dir / "tokenizer.json"
    if bool(args.corpus_token_ids) != bool(args.corpus_byte_offsets):
        parser.error("--corpus-token-ids and --corpus-byte-offsets must be supplied together")
    input_paths = [args.candidates, args.corpus, tokenizer_path]
    if args.corpus_token_ids:
        input_paths.extend([args.corpus_token_ids, args.corpus_byte_offsets])
    if args.output.resolve() in {path.resolve() for path in input_paths}:
        parser.error("--output must not overwrite any input file")
    if args.output.exists():
        parser.error("--output already exists; choose a new path to preserve the earlier report")
    # Read/freeze the proposals BEFORE opening the corpus. This ordering alone
    # cannot prove their provenance, so both source hashes accompany the report.
    candidates_hash = _hash_file(args.candidates)
    with args.candidates.open(encoding="utf-8") as source:
        candidates = [json.loads(line) for line in source if line.strip()]
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    vocab_size = tokenizer.get_vocab_size(with_added_tokens=True)
    if set(tokenizer.get_vocab(with_added_tokens=True).values()) != set(range(vocab_size)):
        raise ValueError("tokenizer vocabulary IDs must be contiguous from zero")
    validate_candidates(candidates, vocab_size)
    corpus_bytes = args.corpus.read_bytes()
    corpus_hash = hashlib.sha256(corpus_bytes).hexdigest()
    text = corpus_bytes.decode("utf-8")
    if args.corpus_token_ids:
        if args.corpus_token_ids.stat().st_size % 4 or args.corpus_byte_offsets.stat().st_size % 8:
            raise ValueError("native export file sizes are not multiples of their element sizes")
        tokens = np.fromfile(args.corpus_token_ids, dtype="<u4")
        offsets = np.fromfile(args.corpus_byte_offsets, dtype="<u8")
        validate_native_tokens(tokens, offsets, corpus_bytes, gpt2_token_bytes(tokenizer))
    else:
        encoding = tokenizer.encode(text, add_special_tokens=False)
        tokens = np.asarray(encoding.ids, dtype=np.int32)
        offsets = np.asarray(encoding.offsets, dtype=np.int64).reshape(-1, 2)
        del encoding
    report = verify_candidates(tokens, candidates, vocab_size, args.seed)
    if args.corpus_token_ids:
        add_native_text_spans(report, corpus_bytes, offsets, tokenizer)
    else:
        add_text_spans(report, text, offsets, tokenizer)
    report["sources"] = {
        "candidates": {"path": str(args.candidates.resolve()), "sha256": candidates_hash},
        "corpus": {"path": str(args.corpus.resolve()), "sha256": corpus_hash},
        "tokenizer": {"path": str(tokenizer_path.resolve()), "sha256": _hash_file(tokenizer_path)},
        "verifier": {"path": str(Path(__file__).resolve()), "sha256": _hash_file(Path(__file__))},
    }
    report["corpus_tokenization"] = {
        "backend": "provided_native_token_ids" if args.corpus_token_ids else "huggingface_tokenizers",
        "byte_alignment": "all token byte intervals verified" if args.corpus_token_ids else "tokenizer character offsets",
    }
    if args.corpus_token_ids:
        for name, path in (("corpus_token_ids", args.corpus_token_ids),
                           ("corpus_byte_offsets", args.corpus_byte_offsets)):
            report["sources"][name] = {"path": str(path.resolve()), "sha256": _hash_file(path)}
    report["interpretation"] = (
        "Validation only; no model inference. Common-pair matches and decoded token displays "
        "are not evidence of text memorization. Candidate sequences must have been fixed "
        "independently of this corpus. Character/byte spans bound touched characters."
    )
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(report, output, ensure_ascii=False, sort_keys=True)
        output.write("\n")
    print(json.dumps({"summary": report["summary"], "by_method": report["by_method"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
