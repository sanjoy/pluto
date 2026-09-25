# First-block activation collision audit

This read-only tool asks whether the **existing, frozen first attention block**
provides enough information at each position for an arbitrary pointwise
replacement MLP/readout to reproduce the required next tokens. It does not
train a replacement, change any weights, or use compacted discrete states.

The hooks capture exact BF16 vectors at token/position embedding, block-zero
attention plus residual, pre-MLP LayerNorm, and MLP plus residual. Equality uses
all channels' bits, with no tolerance or additional quantization. Positive and
negative zero remain distinct; consequently the reported error bound is an
upper bound on achievable accuracy, not a promise that a numeric MLP attains it.

Only suffix/EOS predictions are scored. For a five-token supplied prompt, the
first scored activation is at zero-based input position 4. The last real input
position predicts EOS. Padding and the earlier four prompt positions do not
contribute target constraints. All-vector counts separately include real
prompt positions, matching the discretization archive's count convention.

Every full-corpus prediction must already be correct. By default the tool also
reruns all scored causal prefixes, checks their predictions, and requires
bitwise equality of every captured preceding row with the full-sentence pass.
All GPU transfers use executor-owned pinned host storage kept alive until
completion. Progress goes to stderr; the Markdown report goes to stdout.

## Run

```sh
bazel build -c opt \
  //src/llm/experiments/memorize_general_facts/activation_collision_audit:activation_collision_audit

run=/home/ubuntu/checkpoints/memorize_general_facts/dataset_weights_canonical_order_0
bazel-bin/src/llm/experiments/memorize_general_facts/activation_collision_audit/activation_collision_audit \
  --checkpoint="$run/baseline/checkpoints/layers_4/step_120000" \
  --tokenizer="$run/inputs/tokenizer" \
  --corpus="$run/inputs/corpus.txt" \
  --layers=4 --model_width=10 --attention_heads=1 \
  --feed_forward_width=20 --context_length=27 --prompt_tokens=5 \
  --verify_prefixes=true --max_examples=10000 > /tmp/collision_audit.md
```

`--max_examples` caps the number of conflicting vectors printed per tap, not
the audit coverage. Each printed vector includes its exact bits and one
decoded prefix witness per distinct target, with the full target frequency.

## Result for the checkpoint above

The September 25, 2026 audit verified all 1,024 facts, 14,098 real input rows,
and 10,002 scored suffix/EOS predictions. All 10,002 prefix replays matched;
native prediction errors were zero. The corpus SHA-256 is
`814c062e7d7592fe4a4e5b158a37bd37da51700f817c19eb981c1e93d33f245c`.

| Tap | All unique vectors | Scored unique vectors | Conflicting vector groups | Rows in conflicts | Minimum errors |
| --- | ---: | ---: | ---: | ---: | ---: |
| Token + position embedding | 8,175 | 5,688 | 809 | 4,000 | 2,923 |
| Block 0 attention + residual | 12,483 | 9,655 | 151 | 470 | 289 |
| Block 0 pre-MLP LayerNorm | 12,479 | 9,653 | 151 | 472 | 291 |
| Block 0 MLP + residual | 12,477 | 9,653 | 153 | 474 | 291 |

For an equality group with `n` occurrences, a deterministic vector-only
classifier must make at least `n - max_target_frequency` errors. Summing this
over all groups gives the minimum-errors column. After the first attention
residual, this bounds teacher-forced next-token accuracy by
`(10002 - 289) / 10002 = 97.1106%`. It is not an exact bound on the number of
errors in an autonomous rollout after a wrong token changes the history.

One concrete collision, at zero-based position 7 in both samples:

```text
An ostrich uses its powerful legs to -> " run"
An octopus uses its eight arms to   -> " explore"
```

Both first-attention residual vectors have the exact BF16 words:

```text
3c50 be19 be13 3f4c 3e2a bf0b 3d9a bd58 bf0c 3e5b
```

Therefore replacing the rest of this trained model with **any deterministic
pointwise function of its existing first-attention vector cannot preserve full
memorization**. Increasing the replacement MLP width alone cannot resolve these
collisions. Retraining/changing the first attention block, changing its compute
precision, or adding another context-reading operation is a different question.
This is not a proof that no one-block model can memorize the corpus.

The LayerNorm-only error bound does not directly constrain the complete
residual block: that block also receives the unnormalized residual vector.
The stronger relevant obstruction is already present before LayerNorm.
