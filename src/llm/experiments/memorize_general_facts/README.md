# General-facts memorization experiment

Objective: start with eight GPT-2-style transformer blocks, require exact top-1
training-set completions, and reduce depth until memorization fails. Each of the
1,024 dataset lines is a separate sample, right-padded to 1,024 positions.
Padding must never contribute to loss or accuracy. Checkpoints belong outside
the repository under `~/checkpoints/`.

The completed search found that one block suffices for the approved task;
all depths from eight through one passed. See [RESULTS.md](RESULTS.md) for the
per-depth evidence, smallest checkpoint, and exact verification command.

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
  --checkpoint_dir=/home/ubuntu/checkpoints/memorize_general_facts/new_trial \
  --output_dir=src/llm/experiments/memorize_general_facts/runs/new_trial \
  --layers=8 --search --batch_size=16 --steps=5000
```

Choose a new trial name for each run. Directories must be fresh; existing
artifacts/checkpoints are not overwritten. `--search` starts at the requested
depth, then independently trains
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
exact greedy suffix completion by induction, with EOS predicted immediately
after the suffix. A free-running decoder must separately stop when it emits EOS.
Tokenizer snapshots remain local and are ignored by Git; the audit records
their SHA-256 identities. Completed metric/prediction artifacts and reports are
committed, while checkpoints remain under `~/checkpoints/`.

## Independent verification and the zero-block bound

The audit also groups targets by `(current token, absolute position)`, the only
information available to a zero-block model. There are 809 contradictory groups
and at least 2,923 unavoidable errors: accuracy cannot exceed 70.7758%, even
with perfect optimization. This rules out zero blocks. It does not establish
whether one block can be trained successfully by itself; the completed training
and independent verification in RESULTS.md provide that constructive result.

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

Tokenizer snapshots are not committed. To verify the committed reports on
another checkout, replace native `--tokenizer=RUN` with
`--tokenizer=/path/to/gpt2`, and Python `--tokenizer=RUN/tokenizer.json` with
`--tokenizer=/path/to/gpt2/tokenizer.json`. Use the tokenizer SHA-256 recorded in
[RESULTS.md](RESULTS.md); corpus snapshots and prediction TSVs are committed.

The experiment's `--verify_checkpoint` supports every tested depth via
`--layers=N`. The existing `gpt2_shakespeare_llm --mode=infer_model` CLI instead
constructs eight blocks: it can run the eight-block smoke checks in RESULTS.md,
but cannot load a shallower experiment checkpoint. Its decoder also emits
exactly `--generation_tokens` tokens rather than stopping automatically at EOS.

For a sequential search that runs both independent checks before moving to each
shallower depth, use a Python environment containing `tokenizers`:

```sh
python src/llm/experiments/memorize_general_facts/run_depth_search.py \
  --tokenizer=/home/ubuntu/datasets/tokenizer/gpt2 \
  --checkpoint_dir=/home/ubuntu/checkpoints/memorize_general_facts/search \
  --output_dir=src/llm/experiments/memorize_general_facts/runs/search
```

Both parent directories must be fresh. The driver defaults to depths eight
through one and stops at the first exhausted budget or execution/verification
failure, without retries or hyperparameter changes. Zero blocks are omitted
because of the proven token/position ambiguity above. An explicit
`--start_layers=7` continues after a separately verified eight-block run; it does
not recheck or reuse that earlier run. `depth_search_summary.json` records the
commands, completed verification phases, and smallest verified depth. Native
output streams normally and each depth retains its own `train.log`. The driver
checks that its native binary has not changed between phases, so finish any
builds before launching it.

Independent checkpoint verification requires exactly the model's unique weight
files, rejecting a wrong depth rather than silently accepting a prefix
checkpoint. This stricter experiment-only check does not change the framework's
general-purpose prefix-loading support.

## Width/depth experiments

The native runner also takes explicit `--model_width`, `--attention_heads`,
and `--feed_forward_width` flags. Defaults remain 512, 8, and 2,048; changing
one does not implicitly change the others. Supply all three when training or
verifying a narrower checkpoint. For example, one 64-wide block with a
four-times-expanded MLP uses:

```sh
bazel-bin/src/llm/experiments/memorize_general_facts/memorize_general_facts \
  --layers=1 --model_width=64 --attention_heads=1 --feed_forward_width=256 \
  --tokenizer=/home/ubuntu/datasets/tokenizer/gpt2 \
  --checkpoint_dir=/home/ubuntu/checkpoints/memorize_general_facts/new_width_trial \
  --output_dir=src/llm/experiments/memorize_general_facts/runs/new_width_trial
```

Unlike tensor widths, the number of attention heads cannot be recovered from
raw checkpoint file sizes, so preserve the run's configuration. See
[WIDTH_DEPTH_RESULTS.md](WIDTH_DEPTH_RESULTS.md) for the search protocol,
empirical frontier, and distinction between a training-budget failure and a
capacity lower bound.

The verified sequential driver explores the coarse grid without rerunning
already dominated widths at greater depths. Use fresh parent directories:

```sh
python src/llm/experiments/memorize_general_facts/run_width_depth_search.py \
  --tokenizer=/home/ubuntu/datasets/tokenizer/gpt2 \
  --checkpoint_dir=/home/ubuntu/checkpoints/memorize_general_facts/new_width_search \
  --output_dir=src/llm/experiments/memorize_general_facts/runs/new_width_search
```

Defaults are `--widths=256,128,64,32,16 --depths=1,2,3,4,5,6,7,8` and the same
5,000-update schedule as the earlier experiment. The Python environment needs
`tokenizers`. Both successes and budget failures are independently reloaded and
audited. `width_depth_search_summary.json` records the measured frontier and
minimum-parameter successful model. A budget failure advances to the next
depth; an execution or evidence-validation error stops the search. The native
binary and input hashes are pinned, so do not rebuild that executable or edit
the inputs while a search is active.

After rebuilding a binary with compact-width support, `--widths=48,32,24,16,8`
can refine the narrower region. Positive odd widths are also supported. The
driver keeps head dimension `gcd(width, 64)` and reports the resulting head
count: width 24 uses three heads of dimension 8, while an odd width uses
one-dimensional heads. This is an explicit head-partition policy, not a pure
width-only intervention. The native runner accepts other valid head counts if
you want a separately controlled comparison. Channel tails are masked in
compute tiles, not stored as extra trainable parameters. Kernel validation is
documented in `runs/compact_width_validation_0/README.md`.

To inspect one or several searches without touching the GPU or changing
artifacts, use the read-only evidence reporter:

```sh
python -B src/llm/experiments/memorize_general_facts/summarize_width_depth.py \
  RUN/width_depth_search_summary.json
```

Additional summary paths and optional `--labels=coarse,refinement` combine
runs while keeping their budgets, seeds, and head dimensions explicit. The
reporter rechecks saved evidence before recomputing the pooled measured
frontier; running and untested configurations are never counted as failures.
Tokenizer snapshots must remain available locally for their SHA-256 check.
