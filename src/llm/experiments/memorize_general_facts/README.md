# General-facts memorization experiment

The [discrete-model experiment](discretized_model/README.md) compiles recorded
checkpoint activations into a CPU-only symbolic network while retaining all
transformer boundaries. Its guarantee is the exact corpus-completion task, not
arbitrary-prompt equivalence to the neural model.

Objective: find small GPT-2-style models with exact top-1 training-set
completions. The defaults select the smallest configuration verified so far:
**four blocks, width 10, MLP width 20, and 48,680 parameters**. Each of the
1,024 dataset lines is a separate sample, right-padded to `--context_length`
positions (27 by default).
Padding must never contribute to loss or accuracy. Checkpoints belong outside
the repository under `~/checkpoints/`.

An earlier search with wider models found that one block sufficed for the
approved task. See [RESULTS.md](ai_slop/RESULTS.md) for that historical study;
its checkpoints and parameter counts are not the current defaults.

## Default training configuration

Run from the repository root with fresh checkpoint and output directories:

```sh
bazel build -c opt //src/llm/experiments/memorize_general_facts:memorize_general_facts
bazel-bin/src/llm/experiments/memorize_general_facts/memorize_general_facts \
  --mode=train_model \
  --tokenizer=/home/ubuntu/datasets/tokenizer/gpt2 \
  --checkpoint_dir=/home/ubuntu/checkpoints/memorize_general_facts/new_run \
  --output_dir=/tmp/general_facts_new_run
```

No model or training overrides are needed. Defaults are four transformer blocks,
hidden width 10, one attention head, an MLP `10 -> 20 -> 10`, context 27, and the
4,475-token compact vocabulary. Training uses batch size 32, seed 1337, peak
learning rate 0.0012, 100 warmup steps, and cosine decay over a **120,000-step**
budget; exact corpus accuracy is checked every 256 steps. It stops early once
all suffix/EOS predictions are correct. There is no gradient clipping.

The matched experiment reached zero errors at step **89,600** and passed all
1,024 autonomous five-token-prompt completions. This is an observed result for
this dataset and seed, not a guarantee for different data or settings. Model
and training flags remain explicitly overridable. The periodic-checkpoint and
wall-clock safety controls retain their existing defaults (512 steps and
10,800 seconds); they do not define the optimizer's learning-rate schedule.

Shape defaults are shared by training and inference. To load a differently
shaped checkpoint, supply its dimensions explicitly; the shared `Gpt2Config`
library defaults are unchanged.

## Code and experiment utilities

This directory contains the native C++ training/evaluation/inference binary
and its CLI tests. The reusable `GenerateGreedyContinuation` helper lives in
`src/llm/generate_greedy_continuation.h`; other reusable components live in
`src/dataset/` and `src/llm/`.
Python sweep drivers, Pareto reporting,
checkpoint conversion, corpus/prediction audits, and their tests live separately
in [`scripts/memorize_general_facts`](../../../../scripts/memorize_general_facts).
The native implementation does not depend on those scripts; the drivers invoke
the built binary.

The shared line-based iterator is `src/dataset/padded_line_dataset.h`; it
provides per-sentence padding, prompt masking, and reproducible epoch shuffling.
Exact masked top-1 predictions use `ExtractTop1Ids` in
`src/llm/extract_top1_ids.h`, implemented with a deterministic cuTile kernel.

Reports and analysis notes live in `ai_slop/`. They summarize past experiments,
but generated `runs/` artifacts are
local-only and ignored by Git. They are not required to build or test the code.
Recorded commands in historical manifests may name old script locations; those
are provenance, not current entry points. Use the commands below for new runs.

Every native invocation requires `--mode=train_model` or `--mode=infer_model`.
Training uses `train_model`; inference uses `infer_model` with exactly one
nonempty checkpoint selector: `--infer_checkpoint` for prompt completion or
`--verify_checkpoint` for corpus evaluation. The selectors are mutually exclusive
even when one is explicitly empty. Training rejects both checkpoint selectors.

`--context_length=27` applies to training, corpus verification, and prompt
inference. All current facts fit: the longest has 26 GPT-2 tokens, or 27
including EOS. It sets both
the padded sequence length and the number of learned position embeddings;
sentences longer than the configured context are rejected. Context must be
positive, and corpus paths require at least five positions for the fixed prompt.
The configured context is recorded in run configuration, results, and logs.
The reusable GPT-2 configuration still defaults to 1,024 positions.

The configured context must match the checkpoint's learned position table.
The runner does not infer or resize that table when loading weights.

## Prompt inference

Load an exact checkpoint directory with
`--mode=infer_model --infer_checkpoint=PATH`. The shape flags
must match training: raw checkpoint weights do not encode the attention head
count. Prompt inference neither reads the corpus nor creates training/evaluation
artifacts. Compact inference loads `compact_vocabulary.tsv` directly from the
checkpoint and translates generated IDs back to the original GPT-2 vocabulary
for decoding. Use the same base tokenizer as training.

For the verified 48,680-parameter model matching the current defaults:

```sh
bazel build -c opt //src/llm/experiments/memorize_general_facts:memorize_general_facts

facts_run=/home/ubuntu/checkpoints/memorize_general_facts/context27_L4_W10_FF20_matched_0
bazel-bin/src/llm/experiments/memorize_general_facts/memorize_general_facts \
  --mode=infer_model \
  --infer_checkpoint="$facts_run/trial_000_L4_W10_FF20/checkpoints/layers_4/step_89600" \
  --tokenizer="$facts_run/inputs" \
  --generation_tokens=27 \
  --prompt="The capital of France is"
```

Omit `--prompt` for an interactive loop; each input line starts an independent
completion. Ctrl-D or Ctrl-C exits. The output includes the original prompt
followed by its continuation. Decoding is deterministic greedy top-1, not
sampling, and feeds each generated token back into the model. It stops before
printing EOS, after `--generation_tokens` new tokens, or when prompt plus
continuation reaches `--context_length` tokens. Longer prompts are rejected,
not silently truncated. There is no KV cache: each new token recomputes the
model forward.

Add `--print_attention_probs` to print attention for each generated token, in
both one-prompt and interactive inference. This flag is rejected in training
and corpus verification, even when explicitly set to false. The usual response
is still printed after the attention reports.

Each report identifies the generated token, each transformer's attention layer,
and each head separately. Positions are zero-based: **output token 6** is
predicted from input positions 0 through 5, so its report is a **6 x 6** matrix
with only the lower triangle printed. Rows are queries, columns are keys; row N
contains N+1 probabilities. Token labels use the original tokenizer even for
compact-vocabulary checkpoints. Future positions, right padding, and EOS
outputs are omitted; heads are not averaged and probabilities are not
renormalized for printing. These are scaled causal softmax attention weights,
not output-token probabilities or proof that a token caused the prediction.

Inspection uses `LayerHooks::attention_probabilities_hook`. The normal
FlashAttention computation remains unchanged; an extra cuTile pass materializes
its attention probabilities only when inspection is enabled. This costs
quadratic device memory and can produce substantial output for long prompts.

`--generation_tokens=0` echoes a valid prompt without generating. An explicitly
empty `--prompt` is rejected; blank interactive lines are skipped. If a preferred
GPT-2 token is absent, the compact tokenizer tries an exact encoding with retained
smaller pieces. If no encoding exists, the error names the unencodable substring
and its byte offset; the interactive loop accepts another prompt. This is a
memorization model, not an instruction-following assistant: prompts outside the
training facts need not give sensible answers. To reproduce the memorization
task, supply the first five GPT-2 tokens of a fact.

Prompt inference accepts tokenizer, shape, seed, compact-vocabulary, prompt, and
generation-token options. It rejects corpus, output-directory, batch-size, and
training options, including explicitly supplied defaults such as `--search=false`
or `--steps=120000`. `--prompt` and `--generation_tokens` are exclusive to prompt
inference; they are not accepted by corpus verification or training.

## Compact active vocabulary

New direct invocations use `--compact_vocabulary=true` by default. The experiment
sorts all original GPT-2 token IDs
present in the corpus (including prompt tokens), adds EOS, and remaps them to
contiguous IDs. For this dataset, the resulting vocabulary has **4,475 IDs,
0 through 4,474**, including EOS. GPT-2's ordinary encoding is preserved whenever
all its tokens are retained, so the training corpus and existing checkpoints keep
the same IDs. Otherwise, within each GPT-2 pretoken, a bounded dynamic-programming
search finds a complete encoding using retained tokens, preferring fewer tokens
and then a longer next piece. It never emits an inactive ID or silently changes
text bytes. Unrepresentable substrings produce a readable error. This fallback
does not imply the model learned facts containing the newly encodable text.

The reusable implementation lives in `src/dataset/compact_vocabulary.h`.
`BuildCompactVocabularyMapping(executor, base_tokenizer, corpus_text, eos_id)`
discovers both ID mappings. `CompactVocabularyTokenizer::Create(base_tokenizer,
mapping)` validates and owns them without needing the corpus or an executor.
The base tokenizer must outlive the compact wrapper.

The embedding stores exactly one FP32 row per compact ID, with no trainable
padding rows. The tied output head reuses that same matrix. Temporary logits
remain padded to the kernel's tile size, with padding excluded from softmax;
those slots do not add parameters. The mapping is saved as
`compact_vocabulary.tsv` beside every checkpoint and checked against the
current corpus/tokenizer before loading. Prediction audit TSVs use original
GPT-2 IDs so the independent text verifier continues to work unchanged.

For the historical configuration of **8 blocks, width 16, one head, FF width
64, and context 1,024**, vocabulary compaction changes the parameter count from
**847,008 to 114,256**:

| Component | Parameters |
| --- | ---: |
| Tied token embedding, 4,475 × 16 | 71,600 |
| Learned positions, 1,024 × 16 | 16,384 |
| Eight transformer blocks, 3,280 each | 26,240 |
| Final LayerNorm scale and bias | 32 |
| Total | 114,256 |

The counts include each tied weight once. This is an **86.5% reduction** in
stored model parameters. It changes the softmax vocabulary and hence the
training objective; compact-vocabulary runs must not be pooled into the old
full-vocabulary search as if the protocol were unchanged.

With the current 27-position context, the same compact architecture has
**98,304 parameters**: its learned position table has 432 parameters, reducing
the historical count by 15,952. Training and evaluation process only the
configured number of padded token rows per sample.

The current default model instead uses four blocks, width 10, one attention
head, and an inner MLP width of 20 (`10 -> 20 -> 10`). Its **48,680 parameters**
comprise 44,750 token-embedding parameters, 270 position-embedding parameters,
3,640 parameters across four transformer blocks, and 20 final LayerNorm
parameters. Tied token embeddings are counted once. The training and inference
commands above use this configuration.

Use fresh output/checkpoint directories. Use `run_compact_size_sweep.py` for
current compact-context trials; the historical depth/width drivers implement
a different protocol. The general GPT-2 recipe retains its original default
vocabulary; this experiment opts into configurable exact-row embedding storage.

Training does not support gradient clipping: gradients go directly from
backward to AdamW. The depth/width sweep drivers also train without clipping;
new sweeps therefore do not exactly reproduce the historical clipped protocol.

### Compact an existing successful checkpoint

The converted eight-block, width-16 checkpoint is available at:

`/home/ubuntu/checkpoints/memorize_general_facts/compact_width_16_layers_8/step_25472`.

Fresh native evaluation and independent retokenization still give **zero
errors over all 10,002 targets and 1,024 exact sentences**, without retraining.
The original checkpoint is unchanged. The validation record is local-only at
`runs/compact_vocabulary_validation_0/README.md`, not included in a fresh clone.

`compact_checkpoint.py` selects the original embedding rows byte-for-byte and
copies every other unique weight unchanged. It requires a canonical mapping,
checks all source tensor sizes, and refuses to overwrite any destination.
The destination's parent must already exist. Supply the mapping saved by a
compact-vocabulary training run on the same corpus. For another destination:

```sh
python scripts/memorize_general_facts/compact_checkpoint.py \
  --source=/home/ubuntu/checkpoints/memorize_general_facts/width_depth_refine_16_deep_long_0/width_16/layers_8/step_25472 \
  --destination=/path/to/existing_parent/new_compact_checkpoint \
  --mapping=/path/to/compact_training_run/layers_8/compact_vocabulary.tsv \
  --layers=8 --model_width=16 --feed_forward_width=64

bazel-bin/src/llm/experiments/memorize_general_facts/memorize_general_facts \
  --mode=infer_model \
  --verify_checkpoint=/path/to/existing_parent/new_compact_checkpoint \
  --tokenizer=/home/ubuntu/datasets/tokenizer/gpt2 \
  --corpus=testdata/general_facts_dataset.txt \
  --layers=8 --model_width=16 --attention_heads=1 --feed_forward_width=64 \
  --context_length=1024 --compact_vocabulary=true --batch_size=16 \
  --output_dir=/path/to/new_verification_output
```

This is a weights-only conversion, not an optimizer-state resume or evidence
of fresh compact-vocabulary training to convergence. Removing output classes
renormalizes softmax probabilities and changes the loss even when top-1
predictions are preserved.

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
python scripts/memorize_general_facts/audit_prefixes.py \
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
python -B -m unittest discover \
  -s scripts/memorize_general_facts -p '*_test.py' -v
```

## Native training and depth search

```sh
bazel build -c opt //src/llm/experiments/memorize_general_facts:memorize_general_facts
bazel-bin/src/llm/experiments/memorize_general_facts/memorize_general_facts \
  --mode=train_model \
  --tokenizer=/home/ubuntu/datasets/tokenizer/gpt2 \
  --checkpoint_dir=/home/ubuntu/checkpoints/memorize_general_facts/new_trial \
  --output_dir=src/llm/experiments/memorize_general_facts/runs/new_trial \
  --layers=8 --search --batch_size=16 --steps=5000 --compact_vocabulary=false
```

Choose a new trial name for each run. Directories must be fresh; existing
artifacts/checkpoints are not overwritten. `--search` starts at the requested
depth, then independently trains
each smaller depth until a bounded trial fails. Omit it to run one depth. No
shallower model is tried if the eight-block model fails. A trial's failure is
evidence about this optimization budget, not proof of insufficient capacity.
“Smallest” here means shallowest within this fixed-width GPT-2 family.

The original depth-search models retain width 512, eight heads, 2,048-wide GELU
MLPs, learned absolute
positions, pre-LayerNorm, causal attention, and the full 50,257-token GPT-2
vocabulary with a tied LM head. Activations use BF16; master weights and AdamW
state use FP32. Shared layers get identical seeded initial values at every
depth. The original eight-block residual initialization scale is held fixed.

The initial budget is 5,000 updates (78.125 corpus epochs at batch size 16),
with complete evaluations every 128 updates. AdamW uses beta1=0.9, beta2=0.99,
100-step warmup to 6e-4 and cosine decay to 6e-5. Historical runs clipped the
joint gradient norm to 1.0; current training no longer clips gradients.
Dropout and weight decay are
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
All generated run outputs remain local and are ignored by Git, including
tokenizer/input snapshots, metrics, predictions, logs, and per-run reports.
The audit records input SHA-256 identities. Checkpoints remain under
`~/checkpoints/`.

## Independent verification and the zero-block bound

The audit also groups targets by `(current token, absolute position)`, the only
information available to a zero-block model. There are 809 contradictory groups
and at least 2,923 unavoidable errors: accuracy cannot exceed 70.7758%, even
with perfect optimization. This rules out zero blocks. It does not establish
whether one block can be trained successfully by itself; the completed training
and independent verification in [RESULTS.md](ai_slop/RESULTS.md) provide that
constructive result.

For a finished run, independently retokenize the snapshots and verify every
recorded target, including EOS and complete coverage of all 1,024 samples:

```sh
python scripts/memorize_general_facts/verify_predictions.py \
  --corpus=RUN/corpus.txt --tokenizer=RUN/tokenizer.json \
  --context_length=27 \
  --predictions=RUN/final_predictions.tsv
```

Match `--context_length` to the run; older runs use 32 or 1,024.

This validates the artifact, not inference itself. To also load the saved
weights into a fresh process and reevaluate the whole corpus, use:

```sh
bazel-bin/src/llm/experiments/memorize_general_facts/memorize_general_facts \
  --mode=infer_model \
  --verify_checkpoint=/path/to/layers_4/step_89600 \
  --corpus=RUN/corpus.txt --tokenizer=RUN \
  --output_dir=src/llm/experiments/memorize_general_facts/runs/verification
```

This example uses a checkpoint matching the current defaults. For other models,
pass their actual shape, vocabulary mode, and context length explicitly, and
use a fresh output directory. Corpus verification accepts `--corpus`,
`--output_dir`, `--batch_size`, `--seed`,
`--tokenizer`, shape flags, and `--compact_vocabulary`. It rejects all training
options: `--checkpoint_dir`, `--search`, `--steps`, `--learning_rate`,
`--warmup_steps`, `--eval_every`, `--checkpoint_every`, and `--training_seconds`,
including explicitly supplied defaults and `--search=false`. No optimizer or
training updates run in verification mode. Feed its new `final_predictions.tsv`
to the Python verifier with the original snapshots. The native binary exits 0
for zero errors, 2 for a valid nonperfect model, and 1 for an execution error.
The Python verifier exits 0 for perfection, 1 for prediction errors, and 2 for a
malformed/incomplete report.

Tokenizer snapshots are not committed. To verify the committed reports on
another checkout, replace native `--tokenizer=RUN` with
`--tokenizer=/path/to/gpt2`, and Python `--tokenizer=RUN/tokenizer.json` with
`--tokenizer=/path/to/gpt2/tokenizer.json`. Use the tokenizer SHA-256 recorded in
[RESULTS.md](ai_slop/RESULTS.md). Corpus snapshots and prediction TSVs are local
artifacts; copy them from the original run or generate them with a new run.

The experiment's `--mode=infer_model --verify_checkpoint=PATH` supports every
tested depth via `--layers=N`. The existing
`gpt2_shakespeare_llm --mode=infer_model` CLI instead
constructs eight blocks: it can run the eight-block smoke checks in
[RESULTS.md](ai_slop/RESULTS.md),
but cannot load a shallower experiment checkpoint. Its decoder also emits
exactly `--generation_tokens` tokens rather than stopping automatically at EOS.

For a sequential search that runs both independent checks before moving to each
shallower depth, use a Python environment containing `tokenizers`:

```sh
python scripts/memorize_general_facts/run_depth_search.py \
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

Independent checkpoint verification uses the shared checkpoint API's
`ReadFromDirectory(..., /*allow_prefix=*/false)` option to require exactly the
model's unique weight files, rejecting a wrong depth rather than silently
accepting a prefix checkpoint. The reader still allows prefix loading by default
for callers such as activation generators.

## Width/depth experiments

The native runner also takes explicit `--model_width`, `--attention_heads`,
and `--feed_forward_width` flags, alongside `--context_length`. Width/head/FF
defaults are 10, 1, and 20; changing
one does not implicitly change the others. Supply all three when training or
verifying a narrower checkpoint. For example, one 64-wide block with a
four-times-expanded MLP uses:

```sh
bazel-bin/src/llm/experiments/memorize_general_facts/memorize_general_facts \
  --mode=train_model \
  --layers=1 --model_width=64 --attention_heads=1 --feed_forward_width=256 \
  --context_length=27 \
  --compact_vocabulary=false \
  --tokenizer=/home/ubuntu/datasets/tokenizer/gpt2 \
  --checkpoint_dir=/home/ubuntu/checkpoints/memorize_general_facts/new_width_trial \
  --output_dir=src/llm/experiments/memorize_general_facts/runs/new_width_trial
```

Unlike tensor widths, the number of attention heads cannot be recovered from
raw checkpoint file sizes, so preserve the run's configuration. See
[WIDTH_DEPTH_RESULTS.md](ai_slop/WIDTH_DEPTH_RESULTS.md) for the search protocol,
empirical frontier, and distinction between a training-budget failure and a
capacity lower bound.

The verified sequential driver explores the coarse grid without rerunning
already dominated widths at greater depths. Use fresh parent directories:

```sh
python scripts/memorize_general_facts/run_width_depth_search.py \
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

The default depth grid is not an architectural maximum. After rebuilding with
configurable deeper-depth support, an explicit option such as
`--depths=16 --widths=12,8 --attention_heads=1` selects deeper, narrower trials.
The width/depth driver accepts positive signed-int32 depths; the native recipe
also permits zero for the separate zero-block control. Available memory and
per-tensor backend limits still apply. The ordinary recipe default remains
eight blocks, and residual projections keep the original initialization scale
`0.02 / sqrt(2 * 8)` at every configured depth, preserving shared initialization
in depth comparisons. Construction order and per-block seeds are unchanged.

A model with `L` blocks has `4 + 12*L` unique checkpoint files; sixteen blocks
therefore use 196 files, not 100. Verify such checkpoints with this experiment's
native runner and their recorded depth, widths, and head count. The default
Shakespeare/SAE experiments and the original fixed-shape inspectors still describe
the original eight-block model, not arbitrary deeper checkpoints. The older
`run_depth_search.py` intentionally retains its historical depth-eight starting
limit; use `run_width_depth_search.py` for these configurable searches. Supporting
a shape or validating it with a smoke test does not establish memorization, and
does not automatically expand any already-running experiment grid.
The full-context tests, real optimizer/checkpoint roundtrips, deterministic
repeat, and historical compatibility checks for sixteen-block models are
documented in the local-only `runs/deeper_depth_validation_0/README.md`.

After rebuilding a binary with compact-width support, `--widths=48,32,24,16,8`
can refine the narrower region. Positive odd widths are also supported. The
driver defaults to `--attention_heads=0`, keeping head dimension
`gcd(width, 64)` and reporting the resulting head count: width 24 uses three
heads of dimension 8, while an odd width uses
one-dimensional heads. To hold head count fixed across widths, add
`--attention_heads=1` (or another positive count dividing every requested width).
For example, `--widths=48,32,24,16,8 --attention_heads=1` uses one head at every
width, retaining the one-head setup of the earlier width-64 and width-32 trials.
Invalid overrides are rejected before creating run directories. The manifest
records the override and each trial's resolved head count and dimension;
training and checkpoint verification use the same resolved shape. Channel tails
are masked in compute tiles, not stored as extra trainable parameters. Kernel validation is
documented in the local-only `runs/compact_width_validation_0/README.md`.

To inspect one or several searches without touching the GPU or changing
artifacts, use the read-only evidence reporter:

```sh
python -B scripts/memorize_general_facts/summarize_width_depth.py \
  RUN/width_depth_search_summary.json
```

Additional summary paths and optional `--labels=coarse,refinement` combine
runs while keeping their budgets, seeds, and head dimensions explicit. The
reporter rechecks saved evidence, then shows a measured depth/width frontier and
minimum-parameter verified success for each matched protocol. Protocols match
only when the full pinned binary SHA-256 and every recorded training control
match: batch size, step cap, learning rate, warmup steps, seed, time cap,
evaluation interval, and checkpoint interval. Each group is labeled by its input
run names. Width, depth, heads, and feed-forward width remain per-row
architecture choices; default `--attention_heads=0` and explicit
`--attention_heads=1` do not by themselves split protocols. Older manifests
without that override field remain supported.

The pooled frontier and parameter minimum are also retained as existence
evidence across protocols, not a matched-budget comparison. Only verified
successes enter either kind of frontier or minimum; groups without successes
say so explicitly. Budget failures do not prove a capacity limit, and running
or untested configurations are never counted as failures. Tokenizer snapshots
must remain available locally for their SHA-256 check.
