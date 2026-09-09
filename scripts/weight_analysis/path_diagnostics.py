"""Describe frozen candidate degeneracy without corpus access or filtering.

Periods are token-ID periods, not byte/text periods. A truncated period p obeys
tokens[i] == tokens[i-p] for i >= p; it need not divide the length. An exact
tiling period additionally divides the length. Thus ABABA has periods 2 and 5,
respectively. Alternating two-token cycles require at least four tokens and
minimal truncated period two. A constant path of length one is not repetition.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from .verify import gpt2_token_bytes, validate_candidates


def periods(tokens):
    """Return minimal truncated and exact-tiling periods; empty input is invalid."""
    if not tokens:
        raise ValueError("a path must contain at least one token")
    valid = [p for p in range(1, len(tokens) + 1)
             if all(tokens[i] == tokens[i - p] for i in range(p, len(tokens)))]
    return valid[0], next(p for p in valid if len(tokens) % p == 0)


def analyze(candidates, vocab_size, token_bytes=None):
    """Summarize all supplied strings, preserving record and method denominators.

    No sequence is selected, discarded, changed, or sent to a model. Repeated
    candidate records count separately; distinct-path counts reveal duplication.
    Optional byte decoding recognizes ASCII whitespace only (space, tab, CR,
    LF, vertical tab, form feed), avoiding lossy decoding of partial UTF-8 tokens.
    """
    if type(vocab_size) is not int or not 0 < vocab_size <= 2**31:
        raise ValueError("vocab_size must be a positive int32 vocabulary size")
    validate_candidates(candidates, vocab_size)
    if token_bytes is not None and (not isinstance(token_bytes, dict)
                                    or any(type(v) is not bytes for v in token_bytes.values())):
        raise ValueError("token_bytes must map IDs to exact bytes")
    cache, methods = {}, defaultdict(list)
    for candidate in candidates:
        tokens = tuple(candidate["token_ids"])
        if tokens not in cache:
            period, exact = periods(tokens)
            run = longest = 1
            adjacent = 0
            for before, after in zip(tokens[:-1], tokens[1:]):
                adjacent += before == after
                run = run + 1 if before == after else 1
                longest = max(longest, run)
            # A varied seed can hide a degenerate generated tail from whole-
            # path period tests. The final run above is the constant suffix.
            # Extend the last two distinct IDs backwards by the period-two
            # equality; fewer than four tokens is not a repeated two-ID cycle.
            alternating_suffix = 0
            if len(tokens) >= 4 and tokens[-1] != tokens[-2]:
                start = len(tokens) - 2
                while start > 0 and tokens[start - 1] == tokens[start + 1]:
                    start -= 1
                if len(tokens) - start >= 4:
                    alternating_suffix = len(tokens) - start
            whitespace = None
            if token_bytes is not None:
                if any(token not in token_bytes for token in tokens):
                    raise ValueError("missing token byte spelling")
                whitespace = b"".join(token_bytes[token] for token in tokens).isspace()
            cache[tokens] = dict(length=len(tokens), unique_token_ids=len(set(tokens)),
                                 adjacent_equal_pairs=adjacent, longest_same_token_run=longest,
                                 longest_constant_suffix=run,
                                 longest_alternating_two_token_suffix=alternating_suffix,
                                 truncated_period=period, exact_tiling_period=exact,
                                 ascii_whitespace_only=whitespace)
        methods[candidate["method"]].append(tokens)

    def summarize(sequences):
        stats = [cache[seq] for seq in sequences]
        histogram_fields = ("length", "unique_token_ids", "adjacent_equal_pairs",
                            "longest_same_token_run", "longest_constant_suffix",
                            "longest_alternating_two_token_suffix",
                            "truncated_period", "exact_tiling_period")
        whitespace = sum(s["ascii_whitespace_only"] for s in stats) if token_bytes is not None else None
        return {
            "candidate_records": len(sequences),
            "distinct_paths": len(set(sequences)),
            "duplicate_records": len(sequences) - len(set(sequences)),
            "histograms": {field: {str(k): v for k, v in sorted(Counter(s[field] for s in stats).items())}
                           for field in histogram_fields},
            "constant_token_paths": sum(s["unique_token_ids"] == 1 for s in stats),
            "repeating_one_token_paths": sum(s["length"] >= 2 and s["truncated_period"] == 1 for s in stats),
            "alternating_two_token_paths": sum(s["length"] >= 4 and s["truncated_period"] == 2 for s in stats),
            "constant_suffix_at_least_eight_paths": sum(s["longest_constant_suffix"] >= 8 for s in stats),
            "alternating_suffix_at_least_eight_paths": sum(s["longest_alternating_two_token_suffix"] >= 8 for s in stats),
            "ascii_whitespace_only_paths": whitespace,
            "ascii_whitespace_only_fraction": whitespace / len(stats) if stats and whitespace is not None else None,
        }

    return {
        "schema_version": 1,
        "stage": "corpus_blind_candidate_structure_only",
        "vocab_size": vocab_size,
        "period_definition": "truncated p: token[i]=token[i-p] for all i>=p; exact tiling additionally requires length divisible by p; p=length always allowed",
        "two_token_cycle_definition": "at least four tokens, minimal truncated period exactly two; a truncated final repetition is allowed",
        "suffix_definition": "longest constant or alternating two-distinct-ID suffix, regardless of prefix; alternating suffix is zero unless at least four tokens; nondividing final repetitions allowed",
        "whitespace_definition": "exact token bytes concatenated, nonempty ASCII whitespace only; null if no decoder supplied",
        "limitations": ["Structural description, never a filter or corpus match measurement.",
                        "Counts include repeated records; distinct_paths separately measures duplication.",
                        "Small periods and whitespace runs can yield trivial exact matches, not recovered passages."],
        "overall": summarize([tuple(c["token_ids"]) for c in candidates]),
        "by_method": {method: summarize(sequences) for method, sequences in sorted(methods.items())},
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vocab-size", type=int, default=50257)
    parser.add_argument("--tokenizer-dir", type=Path,
                        help="Optional local GPT-2 tokenizer.json directory; no corpus is read")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError("refusing to overwrite path diagnostics")
    raw = args.candidates.read_bytes()
    candidates = [json.loads(line) for line in raw.splitlines() if line.strip()]
    token_bytes = None
    sources = {"candidates": {"path": str(args.candidates.resolve()),
                              "sha256": hashlib.sha256(raw).hexdigest()},
               "code_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                               for name in ("path_diagnostics.py", "verify.py")}}
    if args.tokenizer_dir:
        from tokenizers import Tokenizer
        path = args.tokenizer_dir / "tokenizer.json"
        tokenizer = Tokenizer.from_file(str(path))
        token_bytes = gpt2_token_bytes(tokenizer)
        if set(token_bytes) != set(range(args.vocab_size)):
            raise ValueError("tokenizer IDs do not match declared vocabulary")
        sources["tokenizer"] = {"path": str(path.resolve()),
                                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    report = analyze(candidates, args.vocab_size, token_bytes)
    report["sources"] = sources
    with args.output.open("x") as output:
        json.dump(report, output, indent=2, allow_nan=False)
        output.write("\n")
    print(json.dumps(report["overall"]))


if __name__ == "__main__":
    main()
