# General-facts experiment utilities

These Python utilities orchestrate and audit experiments. The actual C++ model,
training/evaluation code, dataset, GPU kernels, and Bazel tests remain in
[`src/llm/experiments/memorize_general_facts`](../../src/llm/experiments/memorize_general_facts).
Neither the native binary nor its libraries depend on these scripts.

| Utility | Purpose |
| --- | --- |
| `run_depth_search.py` | Train and independently verify successive depths. |
| `run_width_depth_search.py` | Sweep widths/depths and record the measured Pareto frontier. |
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

The historical sweep drivers deliberately use the full GPT-2 vocabulary,
passing `--compact_vocabulary=false`; moving them does not change their training
protocol. The native binary supports compact-vocabulary training separately.
Successful frontier points require zero errors, while exhausting a trial budget
does not prove that its architecture cannot memorize the corpus.

The reporter takes explicit manifest paths; it does not discover or start runs.
Multiple paths can be supplied to compare searches while keeping different
training protocols separate. Historical evidence remains in
[`src/llm/experiments/memorize_general_facts/runs`](../../src/llm/experiments/memorize_general_facts/runs).
Its manifests retain the commands and absolute paths used at the time, including
old script paths. Do not rewrite those records just to reflect this relocation.

## Audit a finished run

```sh
python -B scripts/memorize_general_facts/verify_predictions.py \
  --corpus=RUN/corpus.txt --tokenizer=RUN/tokenizer.json \
  --predictions=RUN/final_predictions.tsv
```

This checks the saved prediction artifact, not the current checkpoint bytes.
The search drivers also run native checkpoint inference in a fresh process.
See the [experiment documentation](../../src/llm/experiments/memorize_general_facts/README.md)
for direct training, checkpoint verification/conversion, and the scoring protocol.

## Tests

```sh
python -B -m unittest discover -s scripts/memorize_general_facts -p '*_test.py' -v
```

These tests need neither CUDA nor the tokenizer package. Driver tests use fake
native processes and temporary artifacts rather than launching training.
