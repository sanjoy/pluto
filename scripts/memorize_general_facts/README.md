# General-facts experiment utilities

These Python utilities orchestrate and audit experiments. The actual C++ model,
training/evaluation code, dataset, GPU kernels, and Bazel tests remain in
[`src/llm/experiments/memorize_general_facts`](../../src/llm/experiments/memorize_general_facts).
Neither the native binary nor its libraries depend on these scripts.

| Utility | Purpose |
| --- | --- |
| `run_depth_search.py` | Train and independently verify successive depths. |
| `run_width_depth_search.py` | Sweep widths/depths and record the measured Pareto frontier. |
| `run_compact_size_sweep.py` | Sequential compact/context-32 trials with a shared deadline and independent verification. |
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

## Compact 32-token-context trials

`run_compact_size_sweep.py` runs an explicit sequential list of fresh compact
models with at least two layers. Each candidate is
`layers:width:feed_forward_width:steps:learning_rate`; candidate order is preserved
without pruning. One attention head, a 32-token context, and the independently
checked 4,475-token active vocabulary are fixed. Batch size defaults to 32.

```sh
python -B scripts/memorize_general_facts/run_compact_size_sweep.py \
  --binary=bazel-bin/src/llm/experiments/memorize_general_facts/memorize_general_facts \
  --corpus=testdata/general_facts_dataset.txt \
  --tokenizer=/home/ubuntu/datasets/tokenizer/gpt2 \
  --run_dir=/tmp/pluto-facts-compact-new-search \
  --deadline_unix="$(date -d '+1 hour' +%s)" \
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

## Audit a finished run

```sh
python -B scripts/memorize_general_facts/verify_predictions.py \
  --corpus=RUN/corpus.txt --tokenizer=RUN/tokenizer.json \
  --predictions=RUN/final_predictions.tsv
```

This checks the saved prediction artifact, not the current checkpoint bytes.
Pass `--context_length=32` for a run trained with a 32-token context; the verifier
defaults to 1,024 for historical artifacts. Compact-vocabulary runs also write
original GPT-2 IDs in their prediction TSVs, so the same verifier audits them.
The search drivers also run native checkpoint inference in a fresh process.
See the [experiment documentation](../../src/llm/experiments/memorize_general_facts/README.md)
for direct training, checkpoint verification/conversion, and the scoring protocol.

## Tests

```sh
python -B -m unittest discover -s scripts/memorize_general_facts -p '*_test.py' -v
```

These tests need neither CUDA nor the tokenizer package. Driver tests use fake
native processes and temporary artifacts rather than launching training.
