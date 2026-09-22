# General-facts experiment utilities

These Python utilities orchestrate and audit experiments. The actual C++ model,
training/evaluation code, dataset, GPU kernels, and Bazel tests remain in
[`src/llm/experiments/memorize_general_facts`](../../src/llm/experiments/memorize_general_facts).
Neither the native binary nor its libraries depend on these scripts.

Discrete-model generation and its tests live with the
[discretized model](../../src/llm/experiments/memorize_general_facts/discretized_model/README.md),
not in this benchmarking/sweeping directory.

| Utility | Purpose |
| --- | --- |
| `run_depth_search.py` | Train and independently verify successive depths. |
| `run_width_depth_search.py` | Sweep widths/depths and record the measured Pareto frontier. |
| `run_compact_size_sweep.py` | Sequential compact-vocabulary trials with a shared deadline and independent verification. |
| `run_two_track_sweep.py` | Time-bounded adaptive compact-model search, reporting the smallest overall and smallest four-or-more-block successes. |
| `summarize_width_depth.py` | Read and validate saved evidence, then report frontiers without using the GPU. |
| `audit_prefixes.py` | Check corpus tokenization and unavoidable conflicting next-token targets. |
| `verify_predictions.py` | Independently retokenize the corpus and audit every recorded suffix/EOS prediction. |
| `compact_checkpoint.py` | Convert an existing full-vocabulary checkpoint to a supplied compact token mapping. |

Run the examples below from the repository root. The audit and search commands
need a Python environment with `tokenizers` installed and a local GPT-2 tokenizer;
they do not download either. The reporter, checkpoint converter, and unit tests
use Python's standard library. Search drivers require the optimized native binary:

```sh
bazel build -c opt //src/llm/experiments/memorize_general_facts:memorize_general_facts
```

## Sweep and report

Choose fresh artifact and checkpoint directories for each search. The driver
refuses existing directories and independently verifies each completed trial.
For example:

```sh
python -B scripts/memorize_general_facts/run_width_depth_search.py \
  --tokenizer=/home/ubuntu/datasets/tokenizer/gpt2 \
  --checkpoint_dir=/home/ubuntu/checkpoints/memorize_general_facts/new_width_search \
  --output_dir=/tmp/pluto-facts-new-width-search \
  --widths=64,32,16 --depths=1,2,4,8 --attention_heads=1

python -B scripts/memorize_general_facts/summarize_width_depth.py \
  /tmp/pluto-facts-new-width-search/width_depth_search_summary.json
```

The sweep drivers pass `--mode=train_model` for training and
`--mode=infer_model --verify_checkpoint=PATH` for fresh checkpoint evaluation;
the native binary requires an explicit mode. Verification commands omit all
training controls, including `--search=false`. The drivers use the full GPT-2
vocabulary, passing `--compact_vocabulary=false`. The native binary supports
compact-vocabulary training separately. Gradient clipping support has been
removed, so all new runs train without clipping. Historical sweeps used norm-one
gradient clipping;
these commands no longer reproduce that training protocol. Historical results
remain evidence for the protocol recorded in their original manifests.
Successful frontier points require zero errors, while exhausting a trial budget
does not prove that its architecture cannot memorize the corpus.

The reporter takes explicit manifest paths; it does not discover or start runs.
Multiple paths can be supplied to compare searches while keeping different
training protocols separate. Generated evidence is not checked in. The optional
local directory `src/llm/experiments/memorize_general_facts/runs/` is ignored by
Git; output directories elsewhere should also stay out of version control.
Historical manifests retain the commands and absolute paths used at the time,
including old script paths and commands without an explicit mode. The reporter
continues to read those manifests. Do not rewrite those provenance records.

## Compact short-context trials

`run_compact_size_sweep.py` runs an explicit sequential list of fresh compact
models with at least one layer. Each candidate is
`layers:width:feed_forward_width:steps:learning_rate`; candidate order is preserved
without pruning. One attention head and the independently checked 4,475-token
active vocabulary are fixed. The driver uses a 27-token context, enough for
the longest fact and EOS. Batch size defaults to 32.

```sh
python -B scripts/memorize_general_facts/run_compact_size_sweep.py \
  --binary=bazel-bin/src/llm/experiments/memorize_general_facts/memorize_general_facts \
  --corpus=testdata/general_facts_dataset.txt \
  --tokenizer=/home/ubuntu/datasets/tokenizer/gpt2 \
  --run_dir=/tmp/pluto-facts-compact-new-search \
  --deadline_unix="$(date -d '+1 hour' +%s)" \
  --context_length=27 \
  --candidates=4:12:48:60000:0.0006,8:12:48:60000:0.0006
```

Export the native binary's required `LD_LIBRARY_PATH` before launching; the
driver inherits and records it. The fresh run directory must be outside the
repository. It retains the executable/input snapshots, hashes, commands, logs,
checkpoints, and live `summary.json` / `summary.tsv` files.

The shared absolute deadline also applies to checkpoint and greedy verification;
30 seconds are reserved for verification by default. Native training has no
additional per-trial time cap unless `--max_training_seconds` is supplied.
Only a fresh native checkpoint reload, an independent audit of all 10,002
original-GPT-2-ID suffix/EOS predictions, and exact greedy completion of all
1,024 five-token prompts earn `verified` status. Budget failures, timeouts, and
execution errors remain distinct and do not prove an architecture insufficient.

### Two-track parameter search

`run_two_track_sweep.py` interleaves shallow and deeper candidates, reporting
the smallest fully verified model overall and the smallest with at least four
transformer blocks. A deeper success is eligible for both records. The search
starts with the verified four-block, width-13, FF26 configuration, then explores
hidden widths, block counts, and MLP widths. Promising incomplete trials receive
longer training schedules and learning-rate retries, each from scratch.

Context 27, compact vocabulary 4,475, one attention head, batch size 32, and no
gradient clipping are held fixed. Single-block models are allowed. Zero-block
models cannot solve this corpus: identical last-token/position pairs such as
the ` is` in different capital prompts require different predictions, but a
model without transformer blocks has no path to read the preceding country.

```sh
python -B scripts/memorize_general_facts/run_two_track_sweep.py \
  --binary=bazel-bin/src/llm/experiments/memorize_general_facts/memorize_general_facts \
  --corpus=testdata/general_facts_dataset.txt \
  --tokenizer=/home/ubuntu/datasets/tokenizer/gpt2 \
  --run_dir=/home/ubuntu/checkpoints/memorize_general_facts/two_track_new_run \
  --duration_seconds=10800 --seed=1337
```

Use a fresh external run directory and the same runtime-library environment
as for the compact driver. The controller snapshots its executable, inputs,
and scripts, maintains live summaries for both records, and shares one hard
deadline across training and verification. Each success still requires a fresh
checkpoint reload, an independent suffix/EOS audit, and all 1,024 exact greedy
completions. A failed bounded trial is not proof of insufficient capacity;
the records identify the smallest verified models **found**, not a proven
global minimum.

### Verified 27-token context (2026-09-24)

The current four-block, width-13, one-head, FF26 model has **64,532 trainable
parameters** with context 27. That is 65 fewer than context 32: only the learned
position table changes, from `32 x 13` to `27 x 13`. The longest fact has 26
GPT-2 tokens, so its EOS also fits.

Training from scratch with seed 1337 and batch size 32 first reached zero errors
at **step 48,896**, after **147 seconds**, with mean cross-entropy **0.00691575
nats**. All 10,002 scored suffix/EOS targets and all 1,024 autonomous greedy
completions from five-token prompts are correct. Evaluation occurs every 256
steps, so this is the first observed perfect checkpoint. Optimizer and schedule
are the same as the FF26 comparison below; no gradient clipping is used.

The checkpoint is:

```text
/home/ubuntu/checkpoints/memorize_general_facts/context27_L4_W13_FF26_0/trial_000_L4_W13_FF26/checkpoints/layers_4/step_48896
```

The initial run exposed a 16-row alignment restriction in single-sample
inference, hidden by batch-32 training. Dense and tied-head kernels already
masked partial tiles; GELU and residual addition now mask them too. After fixing
those paths, the same checkpoint passed a fresh **batch-1** evaluation and all
autonomous completions using the default context of 27. This also tests the
partial tiles directly. In the trial directory, the final evidence is
`verification_fixed_batch1/`, `prediction_fixed_verification.json`,
`greedy_fixed.stdout.log`, and `greedy_fixed_verification.json`; the corrected
binary and hash are preserved under the run's `inputs/`. The original failure
logs and summary remain unchanged as provenance. All artifacts stay outside Git.

To repeat training and the complete verification pipeline, use the compact
driver above with `--candidates=4:13:26:60000:0.0006`, `--seed=1337`,
`--batch_size=32`, `--eval_every=256`, and a fresh run directory.

### Verified width-13 MLP comparison (2026-09-24)

Both four-block, width-13 models below memorize all 1,024 facts with context
length 32 and one attention head. Reducing the inner MLP width changes each
block from `13 -> 52 -> 13` to `13 -> 26 -> 13`:

| Inner MLP width | Trainable parameters | First zero-error evaluation step | Training time | Mean cross-entropy (nats) |
| --- | ---: | ---: | ---: | ---: |
| 52 | 67,405 | 34,816 | 122 seconds | 0.00999714 |
| 26 | 64,597 | 47,104 | 160 seconds | 0.00819570 |

Each checkpoint passes a fresh-process evaluation and independent audit with
zero errors over all 10,002 suffix/EOS targets, followed by exact autonomous
completion of all 1,024 first-five-token prompts. Evaluations occur every 256
steps, so the table reports the first observed perfect checkpoint, not the
earliest individual update at which it might have become perfect.

Both runs use seed 1337, batch size 32, the 4,475-token compact vocabulary,
BF16 computation with FP32 master weights, and no gradient clipping. AdamW uses
beta1=0.9, beta2=0.99, epsilon=1e-8, and zero weight decay. Peak learning rate is
0.0006, with 100 warmup steps and cosine decay to a 10% floor over the same
60,000-step budget. Each run stops early on perfect corpus predictions.
Wall-clock safety caps differ but neither was reached. These are single-seed
results: FF26 is 2,808 parameters (4.17%) smaller, but took longer to memorize
in this comparison; they do not establish typical time across seeds.

The local run roots are
`/home/ubuntu/checkpoints/memorize_general_facts/context32_size_sweep_1h_0`
(FF52: `trial_000_L4_W13_FF52/checkpoints/layers_4/step_34816`) and
`/home/ubuntu/checkpoints/memorize_general_facts/context32_L4_W13_FF26_0`
(FF26: `trial_000_L4_W13_FF26/checkpoints/layers_4/step_47104`). Each retains
its commands, input hashes, logs, checkpoint, and independent verification
reports. These generated artifacts remain outside Git.

## Audit a finished run

```sh
python -B scripts/memorize_general_facts/verify_predictions.py \
  --corpus=RUN/corpus.txt --tokenizer=RUN/tokenizer.json \
  --predictions=RUN/final_predictions.tsv
```

This checks the saved prediction artifact, not the current checkpoint bytes.
Pass `--context_length=27` for the current experiment. The standalone verifier
does not inherit the training binary's context setting. Compact-vocabulary runs
also write original GPT-2 IDs in their prediction TSVs, so the same verifier audits them.
The search drivers also run native checkpoint inference in a fresh process.
See the [experiment documentation](../../src/llm/experiments/memorize_general_facts/README.md)
for direct training, checkpoint verification/conversion, and the scoring protocol.

## Tests

```sh
python -B -m unittest discover -s scripts/memorize_general_facts -p '*_test.py' -v
```

These tests need neither CUDA nor the tokenizer package. Driver tests use fake
native processes and temporary artifacts rather than launching training.
