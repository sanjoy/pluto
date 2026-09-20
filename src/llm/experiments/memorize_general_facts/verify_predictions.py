#!/usr/bin/env python3
"""Independently validate a memorization runner's final per-token prediction TSV.

The CLI requires the run's corpus/tokenizer snapshots explicitly. It retokenizes
every sentence, reconstructs the approved five-token-prompt targets plus EOS,
and checks that the TSV covers each expected position exactly once. Padding is
never a target. This validates the recorded audit, not GPU inference itself;
the native runner reloads the checkpoint before producing this TSV.

JSON goes to stdout without creating or modifying files. Exit status is 0 for
an exact-completion result, 1 for a valid audit containing prediction errors,
and 2 for malformed input. Only the CLI requires the tokenizers package; the
validation function and unit tests use Python's standard library exclusively.
"""

import argparse
import csv
from dataclasses import dataclass
import hashlib
import io
import json
import math
from pathlib import Path
import re
from typing import Sequence, TextIO


TSV_HEADER = (
    "line_1based",
    "target_token_index_0based",
    "target_id",
    "predicted_id",
    "loss",
)


@dataclass(frozen=True)
class Prediction:
    """One scored target; index len(sentence) is the appended EOS token."""

    line_1based: int
    target_token_index_0based: int
    target_id: int
    predicted_id: int
    loss: float


@dataclass(frozen=True)
class Verification:
    """Validated rows in corpus/position order, never TSV file order."""

    sentences: int
    exact_sentences: int
    targets: int
    errors: int
    mean_loss: float
    predictions: tuple[Prediction, ...]

    def summary(self) -> dict:
        return {
            "success": self.errors == 0,
            "sentences": self.sentences,
            "exact_sentences": self.exact_sentences,
            "targets": self.targets,
            "errors": self.errors,
            "accuracy": (self.targets - self.errors) / self.targets,
            "mean_loss_nats": self.mean_loss,
        }


def _parse_unsigned_integer(value: str, field: str, tsv_line: int) -> int:
    # int() alone accepts surrounding whitespace and signs. The native TSV
    # contains plain decimal integers; accepting floats or negative sentinels
    # would hide malformed indices and ignored-row leakage.
    if re.fullmatch(r"[0-9]+", value) is None:
        raise ValueError(f"TSV row {tsv_line}: {field} must be an unsigned integer")
    return int(value)


def verify_predictions(
    token_rows: Sequence[Sequence[int]],
    tsv: TextIO,
    *,
    eos_id: int,
    vocabulary_size: int,
    prompt_tokens: int = 5,
    context_length: int = 1024,
    expected_samples: int = 1024,
) -> Verification:
    """Validate every target against independent, unpadded sentence tokens.

    A sentence of L original tokens contributes positions P through L inclusive
    (zero-based target-token indices), where P is the supplied prompt length.
    Position L predicts EOS using the last original token as input, so L may
    equal context_length without requiring a context_length + 1 input buffer.
    TSV ordering is irrelevant, but duplicate or missing positions are errors.
    expected_samples is configurable only in this core API to keep tests small;
    the experiment CLI always requires all 1,024 corpus lines.
    """
    if (
        type(expected_samples) is not int
        or expected_samples <= 0
        or len(token_rows) != expected_samples
    ):
        raise ValueError(f"Expected exactly {expected_samples} corpus samples")
    if (
        type(vocabulary_size) is not int
        or vocabulary_size <= 0
        or type(eos_id) is not int
        or not 0 <= eos_id < vocabulary_size
    ):
        raise ValueError("EOS must be a valid logical vocabulary token")
    if (
        type(prompt_tokens) is not int
        or type(context_length) is not int
        or not 1 <= prompt_tokens <= context_length
    ):
        raise ValueError("Prompt and context lengths must be positive and compatible")

    expected = {}
    for line, tokens in enumerate(token_rows, start=1):
        if not prompt_tokens <= len(tokens) <= context_length:
            raise ValueError(f"Corpus line {line} has an invalid token count")
        if any(
            type(token) is not int or not 0 <= token < vocabulary_size
            for token in tokens
        ):
            raise ValueError(f"Corpus line {line} contains an invalid token ID")
        for position in range(prompt_tokens, len(tokens)):
            expected[line, position] = tokens[position]
        expected[line, len(tokens)] = eos_id

    reader = csv.reader(tsv, delimiter="\t", strict=True)
    if tuple(next(reader, ())) != TSV_HEADER:
        raise ValueError(f"Expected TSV header: {chr(9).join(TSV_HEADER)}")
    actual = {}
    for fields in reader:
        if len(fields) != len(TSV_HEADER):
            raise ValueError(f"TSV row {reader.line_num} must have five fields")
        line, position, target, prediction = (
            _parse_unsigned_integer(value, field, reader.line_num)
            for field, value in zip(TSV_HEADER[:4], fields[:4])
        )
        key = line, position
        if key not in expected:
            raise ValueError(
                f"TSV row {reader.line_num}: unexpected corpus line/target index {key}"
            )
        if key in actual:
            raise ValueError(f"TSV row {reader.line_num}: duplicate target {key}")
        if target != expected[key]:
            raise ValueError(
                f"TSV row {reader.line_num}: target {key} declares ID {target}, "
                f"but independent tokenization requires {expected[key]}"
            )
        if prediction >= vocabulary_size:
            raise ValueError(
                f"TSV row {reader.line_num}: prediction is outside the logical vocabulary"
            )
        loss = float(fields[4])
        if not math.isfinite(loss):
            raise ValueError(f"TSV row {reader.line_num}: loss must be finite")
        actual[key] = Prediction(line, position, target, prediction, loss)
    missing = expected.keys() - actual.keys()
    if missing:
        raise ValueError(
            f"TSV is missing {len(missing)} scored targets; first is {min(missing)}"
        )

    predictions = tuple(actual[key] for key in sorted(expected))
    wrong = tuple(row for row in predictions if row.target_id != row.predicted_id)
    incorrect_sentences = {row.line_1based for row in wrong}
    return Verification(
        sentences=len(token_rows),
        exact_sentences=len(token_rows) - len(incorrect_sentences),
        targets=len(predictions),
        errors=len(wrong),
        mean_loss=math.fsum(row.loss / len(predictions) for row in predictions),
        predictions=predictions,
    )


def _corpus_lines(contents: bytes) -> list[str]:
    # Match the native line dataset: only LF/CRLF terminate records, one final
    # newline is allowed, and blank lines are invalid rather than skipped.
    text = contents.decode("utf-8")
    if text.endswith("\n"):
        text = text[:-1]
    lines = [line.removesuffix("\r") for line in text.split("\n")]
    if any(not line.strip(" \t\r\n\v\f") for line in lines):
        raise ValueError("Corpus must contain one nonempty sentence per line")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus", type=Path, required=True, help="Run's corpus.txt snapshot"
    )
    parser.add_argument(
        "--tokenizer", type=Path, required=True, help="Run's tokenizer.json snapshot"
    )
    parser.add_argument(
        "--predictions", type=Path, required=True, help="final_predictions.tsv"
    )
    parser.add_argument(
        "--examples", type=int, default=5, help="Maximum decoded error examples"
    )
    args = parser.parse_args()
    if args.examples < 0:
        parser.error("--examples must not be negative")

    try:
        from tokenizers import Tokenizer

        corpus_contents = args.corpus.read_bytes()
        tokenizer_contents = args.tokenizer.read_bytes()
        prediction_contents = args.predictions.read_bytes()
        try:
            tokenizer = Tokenizer.from_str(tokenizer_contents.decode("utf-8"))
        except Exception as error:
            # The Rust binding reports malformed snapshots as plain Exception,
            # not ValueError. Translate that specific boundary into the same
            # input-error path used by the independent validator below.
            raise ValueError(f"Cannot load tokenizer snapshot: {error}") from error
        # A snapshot must not silently truncate sentences or add token padding.
        tokenizer.no_padding()
        tokenizer.no_truncation()
        rows = [
            tokenizer.encode(line, add_special_tokens=False).ids
            for line in _corpus_lines(corpus_contents)
        ]
        eos = tokenizer.token_to_id("<|endoftext|>")
        vocabulary_size = tokenizer.get_vocab_size(with_added_tokens=True)
        if eos is None or vocabulary_size != 50257:
            raise ValueError("Expected the experiment's 50,257-token GPT-2 vocabulary")
        result = verify_predictions(
            rows,
            io.StringIO(prediction_contents.decode("utf-8")),
            eos_id=eos,
            vocabulary_size=vocabulary_size,
        )
    except (ImportError, OSError, UnicodeError, ValueError, csv.Error) as error:
        parser.error(str(error))

    examples = []
    for row in result.predictions:
        if row.target_id == row.predicted_id:
            continue
        if len(examples) == args.examples:
            break
        sentence = rows[row.line_1based - 1]
        examples.append(
            {
                "line_1based": row.line_1based,
                "target_token_index_0based": row.target_token_index_0based,
                "causal_prefix": tokenizer.decode(
                    sentence[: row.target_token_index_0based], skip_special_tokens=False
                ),
                "target_id": row.target_id,
                "target_text": tokenizer.decode(
                    [row.target_id], skip_special_tokens=False
                ),
                "predicted_id": row.predicted_id,
                "predicted_text": tokenizer.decode(
                    [row.predicted_id], skip_special_tokens=False
                ),
            }
        )
    print(
        json.dumps(
            {
                **result.summary(),
                "corpus": str(args.corpus),
                "corpus_sha256": hashlib.sha256(corpus_contents).hexdigest(),
                "tokenizer": str(args.tokenizer),
                "tokenizer_sha256": hashlib.sha256(tokenizer_contents).hexdigest(),
                "predictions": str(args.predictions),
                "predictions_sha256": hashlib.sha256(prediction_contents).hexdigest(),
                "prompt_tokens": 5,
                "eos_scored": True,
                "padding_scored": False,
                "context_length": 1024,
                "evidence_scope": "validated teacher-forced prediction artifact, not independent checkpoint inference",
                "error_examples": examples,
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )
    return 0 if result.errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
