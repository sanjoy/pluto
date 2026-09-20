# General-facts memorization experiment

Objective: start with eight GPT-2-style transformer blocks, require exact top-1
training-set completions, and reduce depth until memorization fails. Each of the
1,024 dataset lines is a separate sample, right-padded to 1,024 positions.
Padding must never contribute to loss or accuracy. Checkpoints belong outside
the repository under `~/checkpoints/`.

## Preflight: the literal objective has contradictory targets

The user approved supplying each sentence's first five GPT-2 tokens as a prompt
and requiring every remaining next-token prediction, including EOS, to be
correct. Prompt targets and padding are excluded from both loss and accuracy.

Identical causal token prefixes sometimes precede different targets. For
example, `The capital of` precedes Mongolia, France, Greece, and Peru on different
lines. An independently processed sentence does not carry a line identifier,
and a causal model cannot inspect later tokens or right-padding to resolve this.

For each complete token prefix, let `c_y` count occurrences of next-token label
`y`, and let `n = sum(c_y)`. The maximum possible correct top-1 count at that
prefix is `max(c_y)`. Its minimum total cross-entropy is
`-sum(c_y * log(c_y / n))`. Summing these quantities gives model-independent
bounds; a particular neural network need not attain them.

The local GPT-2 tokenizer gives these results for the committed corpus:

| Scoring convention | Targets | Unavoidable errors | Maximum accuracy | Minimum mean loss (nats) |
| --- | ---: | ---: | ---: | ---: |
| First token supplied, no EOS | 13,074 | 687 | 94.745296% | 0.18518549 |
| First token supplied, EOS scored | 14,098 | 687 | 95.126968% | 0.17173465 |
| BOS supplied, first token and EOS scored | 15,122 | 1,528 | 89.895516% | 0.46937093 |

There are 146 conflicting text-prefix groups. All sentences contain 6–26 GPT-2
tokens. Their first five tokens uniquely identify all 1,024 sentences; supplying
those five tokens and scoring the remaining suffix plus EOS would give 10,002
noncontradictory targets. This is the approved scoring convention; it does not
test predictions within the prompt.
Neither zero conflicts nor unique prompts proves that any tested model will
learn to complete the sentences.

## Reproduce the audit

Run from the repository root, using a Python environment with the `tokenizers`
package installed. The tokenizer is read locally; nothing is downloaded.

```sh
python src/llm/experiments/memorize_general_facts/audit_prefixes.py \
  --dataset=testdata/general_facts_dataset.txt \
  --tokenizer=/path/to/gpt2/tokenizer.json
```

The JSON output records corpus and tokenizer SHA-256 hashes, token lengths,
scoring conventions, exact bounds, and example conflicting prefixes. It also
audits the approved five-token prompt convention. Padding and truncation are
disabled during this check; the context limit is validated separately, and
BOS/EOS conventions are explicit.

Run the counting tests with standard-library Python (no GPU or tokenizer needed):

```sh
python -m unittest discover \
  -s src/llm/experiments/memorize_general_facts -p '*_test.py' -v
```

## Native training and depth search

```sh
bazel build -c opt //src/llm/experiments/memorize_general_facts:memorize_general_facts
bazel-bin/src/llm/experiments/memorize_general_facts/memorize_general_facts \
  --tokenizer=/home/ubuntu/datasets/tokenizer/gpt2 \
  --checkpoint_dir=/home/ubuntu/checkpoints/memorize_general_facts/trial_0 \
  --output_dir=src/llm/experiments/memorize_general_facts/runs/trial_0 \
  --layers=8 --search --batch_size=16 --steps=5000
```

Run directories must be fresh; existing artifacts/checkpoints are not
overwritten. `--search` starts at the requested depth, then independently trains
each smaller depth until a bounded trial fails. Omit it to run one depth. No
shallower model is tried if the eight-block model fails. A trial's failure is
evidence about this optimization budget, not proof of insufficient capacity.
“Smallest” here means shallowest within this fixed-width GPT-2 family.

All models retain width 512, eight heads, 2,048-wide GELU MLPs, learned absolute
positions, pre-LayerNorm, causal attention, and the full 50,257-token GPT-2
vocabulary with a tied LM head. Activations use BF16; master weights and AdamW
state use FP32. Shared layers get identical seeded initial values at every
depth. The original eight-block residual initialization scale is held fixed.

The initial budget is 5,000 updates (78.125 corpus epochs at batch size 16),
with complete evaluations every 128 updates. AdamW uses beta1=0.9, beta2=0.99,
100-step warmup to 6e-4 and cosine decay to 6e-5. The joint gradient norm is
clipped to 1.0 using deterministic GPU reductions, counting tied weights only
once. Dropout and weight decay are
zero because the goal is memorization, not generalization. The schedule and
small-token-batch beta2 choice follow the patterns in
[nanoGPT's training loop](https://github.com/karpathy/nanoGPT/blob/master/train.py)
and [small-corpus configuration](https://github.com/karpathy/nanoGPT/blob/master/config/train_shakespeare_char.py);
the budget and regularization choices are experiment-specific, not guarantees
of convergence. `--training_seconds` defaults to 10,800 seconds per depth. This
is a soft budget including initial/periodic evaluation; final checkpoint I/O
and verification can extend it. Actual epochs and timeout status are recorded.

Each checkpoint saves unique model weights under `layers_N/step_K`. These are
inference checkpoints, not exact optimizer-state resume checkpoints. Each
artifact directory contains configuration, input/tokenizer snapshots, loss and
integer error counts, and a final per-token TSV. The final weights are reloaded
from disk before that final audit. Success requires zero errors across all
10,002 targets and all 1,024 sentences, not a rounded accuracy or loss threshold.
Because attention is causal, perfect teacher-forced top-1 predictions imply
exact greedy suffix completion by induction, including termination at EOS.

## Independent verification and the zero-block bound

The audit also groups targets by `(current token, absolute position)`, the only
information available to a zero-block model. There are 809 contradictory groups
and at least 2,923 unavoidable errors: accuracy cannot exceed 70.7758%, even
with perfect optimization. This rules out zero blocks. It does not establish
whether one block can be trained successfully.

For a finished run, independently retokenize the snapshots and verify every
recorded target, including EOS and complete coverage of all 1,024 samples:

```sh
python src/llm/experiments/memorize_general_facts/verify_predictions.py \
  --corpus=RUN/corpus.txt --tokenizer=RUN/tokenizer.json \
  --predictions=RUN/final_predictions.tsv
```

This validates the artifact, not inference itself. To also load the saved
weights into a fresh process and reevaluate the whole corpus, use:

```sh
bazel-bin/src/llm/experiments/memorize_general_facts/memorize_general_facts \
  --verify_checkpoint=/path/to/layers_8/step_N \
  --layers=8 --corpus=RUN/corpus.txt --tokenizer=RUN \
  --output_dir=src/llm/experiments/memorize_general_facts/runs/verification
```

Use the checkpoint's actual depth and a fresh output directory. No optimizer or
training updates run in verification mode. Feed its new `final_predictions.tsv`
to the Python verifier with the original snapshots. The native binary exits 0
for zero errors, 2 for a valid nonperfect model, and 1 for an execution error.
The Python verifier exits 0 for perfection, 1 for prediction errors, and 2 for a
malformed/incomplete report.
