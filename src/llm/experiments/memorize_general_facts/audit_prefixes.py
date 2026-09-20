#!/usr/bin/env python3
"""Check whether a causal model can predict every corpus continuation exactly.

This is a model-independent feasibility check, not a training metric. Two
independent samples with identical token prefixes give a causal model identical
inputs. If their next tokens differ, at least one top-1 prediction must be wrong.
For a zero-transformer-block model, only the current token and its position can
influence the prediction; grouping by that reduced context gives a stricter
bound. These are information bounds, not guarantees that a model attains them.
The caller supplies unpadded, independently tokenized sentences; padding never
contributes targets. The CLI does not train or select a revised objective.
"""

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Sequence


@dataclass
class PrefixAudit:
    """Counts under one explicit context and prompt/BOS/EOS convention."""

    targets: int
    contexts: int
    unavoidable_errors: int
    minimum_mean_cross_entropy: float
    conflicts: dict[tuple[int, ...], Counter[int]]

    def summary(self) -> dict:
        return {
            "targets": self.targets,
            "distinct_contexts": self.contexts,
            "conflicting_contexts": len(self.conflicts),
            "unavoidable_top1_errors": self.unavoidable_errors,
            "maximum_top1_accuracy": 1 - self.unavoidable_errors / self.targets,
            "minimum_mean_cross_entropy_nats": self.minimum_mean_cross_entropy,
        }


def audit_prefixes(
    rows: Sequence[Sequence[int]],
    *,
    prompt_tokens: int = 1,
    eos_id: int | None = None,
    bos_id: int | None = None,
    context_mode: str = "prefix",
) -> PrefixAudit:
    """Group next-token targets by the information available to a predictor.

    `context_mode="prefix"` (the default) uses the complete causal token prefix.
    `context_mode="token_position"` uses only (absolute position, current token),
    with zero-based positions. This is the information available to GPT-2 with
    zero transformer blocks: embeddings, position embeddings, final rowwise
    LayerNorm, and a tied LM head cannot inspect earlier token positions. The
    prefix representation is still necessary for models with attention.

    `prompt_tokens` counts original sentence tokens provided without scoring.
    With BOS, zero prompt tokens also scores the first sentence token. EOS, when
    requested, is a real scored target, even when its ID is also used for PAD in
    a training implementation. A BOS ID is prepended before forming context
    keys, so BOS is position zero and shifts all sentence-token positions by
    one in token_position mode. Padding never contributes targets or context.

    At a prefix with target counts c_y, any top-1 predictor can be right at most
    max(c_y) times. The best unrestricted probability distribution is the
    empirical distribution c_y / sum(c_y), whose entropy supplies an NLL lower
    bound. A particular neural network might not achieve either bound.
    """
    if context_mode not in ("prefix", "token_position"):
        raise ValueError("context_mode must be 'prefix' or 'token_position'")
    if prompt_tokens < 0 or (prompt_tokens == 0 and bos_id is None):
        raise ValueError("Provide at least one prompt token, or a BOS token")
    if not rows:
        raise ValueError("At least one sample is required")

    counts = defaultdict(Counter)
    for row in rows:
        if not row or len(row) < prompt_tokens:
            raise ValueError("Every sample must contain the complete prompt")
        sentence = list(row)
        if eos_id is not None:
            sentence.append(eos_id)
        beginning = [] if bos_id is None else [bos_id]
        for position in range(prompt_tokens, len(sentence)):
            prefix = tuple(beginning + sentence[:position])
            context = (
                prefix if context_mode == "prefix" else (len(prefix) - 1, prefix[-1])
            )
            counts[context][sentence[position]] += 1

    targets = sum(sum(histogram.values()) for histogram in counts.values())
    if targets == 0:
        raise ValueError("The scoring convention produces no targets")
    correct = sum(max(histogram.values()) for histogram in counts.values())
    minimum_nll = math.fsum(
        count * math.log(sum(histogram.values()) / count)
        for histogram in counts.values()
        for count in histogram.values()
    )
    return PrefixAudit(
        targets=targets,
        contexts=len(counts),
        unavoidable_errors=targets - correct,
        minimum_mean_cross_entropy=minimum_nll / targets,
        conflicts={context: h for context, h in counts.items() if len(h) > 1},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, default=Path("testdata/general_facts_dataset.txt")
    )
    parser.add_argument(
        "--tokenizer", type=Path, required=True, help="Local GPT-2 tokenizer.json"
    )
    parser.add_argument("--context_length", type=int, default=1024)
    parser.add_argument("--prompt_tokens", type=int, default=5)
    args = parser.parse_args()

    # Only the CLI needs the tokenizer package. The exact counting algorithm and
    # its unit tests depend solely on Python's standard library.
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    # Do not inherit padding/truncation settings that could conceal long rows or
    # inflate the measured accuracy with ignored padding positions.
    tokenizer.no_padding()
    tokenizer.no_truncation()
    contents = args.dataset.read_bytes()
    lines = contents.decode("utf-8").splitlines()
    if not lines or any(not line.strip() for line in lines):
        parser.error("Dataset must contain one nonempty sentence per line")
    rows = [tokenizer.encode(line, add_special_tokens=False).ids for line in lines]
    if any(not row for row in rows):
        parser.error("A sentence tokenized to an empty sequence")
    if args.context_length < 1 or max(map(len, rows)) + 1 > args.context_length:
        parser.error("A sample including BOS exceeds the requested context length")
    if not 1 <= args.prompt_tokens <= min(map(len, rows)):
        parser.error("Prompt length must fit every sentence")
    eos = tokenizer.token_to_id("<|endoftext|>")
    if eos is None:
        parser.error("Expected the GPT-2 end-of-text token")

    original = audit_prefixes(rows)
    with_eos = audit_prefixes(rows, eos_id=eos)
    with_bos = audit_prefixes(rows, prompt_tokens=0, bos_id=eos, eos_id=eos)
    candidate = audit_prefixes(rows, prompt_tokens=args.prompt_tokens, eos_id=eos)
    zero_block = audit_prefixes(
        rows,
        prompt_tokens=args.prompt_tokens,
        eos_id=eos,
        context_mode="token_position",
    )
    examples = []
    for prefix, histogram in sorted(
        original.conflicts.items(), key=lambda item: (-sum(item[1].values()), item[0])
    )[:5]:
        examples.append(
            {
                "prefix_ids": list(prefix),
                "prefix": tokenizer.decode(list(prefix), skip_special_tokens=False),
                "next_tokens": [
                    {
                        "id": token,
                        "text": tokenizer.decode([token], skip_special_tokens=False),
                        "count": count,
                    }
                    for token, count in sorted(histogram.items())
                ],
            }
        )

    print(
        json.dumps(
            {
                "dataset": str(args.dataset),
                "dataset_sha256": hashlib.sha256(contents).hexdigest(),
                "tokenizer_sha256": hashlib.sha256(
                    args.tokenizer.read_bytes()
                ).hexdigest(),
                "samples": len(rows),
                "minimum_sentence_tokens": min(map(len, rows)),
                "maximum_sentence_tokens": max(map(len, rows)),
                "context_length": args.context_length,
                "padding_scored": False,
                "first_token_prompt_no_eos": original.summary(),
                "first_token_prompt_with_eos": with_eos.summary(),
                "bos_prompt_with_eos": with_bos.summary(),
                "prompt_completion_with_eos": {
                    "prompt_tokens": args.prompt_tokens,
                    "distinct_prompts": len(
                        {tuple(row[: args.prompt_tokens]) for row in rows}
                    ),
                    **candidate.summary(),
                },
                "zero_block_prompt_completion_with_eos": {
                    "prompt_tokens": args.prompt_tokens,
                    "context_mode": "token_position",
                    **zero_block.summary(),
                },
                "conflict_examples": examples,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
