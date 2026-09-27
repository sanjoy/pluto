# Frozen A3 stacked-MLP sweep

This experiment replaces the original post-A3 transformer suffix with one to
five residual MLP blocks. Every block applies pre-layer normalization, a
10 → H → 10 GELU MLP, and a residual addition. A trainable final layer norm
feeds the original frozen embedding/head. Source weights are never trained.

The default grid is depths 1, 2, 3, 4, 5 × hidden widths 20, 40, 80, 150:
20 independent fits, each with 300,000 updates, evaluation every 1,000 updates,
batch size 32, seed 3, and initial learning rate 0.01. Native training uses
Adam with no weight decay or clipping and cosine decay to 10% of the initial
rate. `--match_mlp_parameter_budget=false` is explicit so the requested width
is used exactly, including widths below the old minimum of 66.

| Hidden width | 1 block | 2 blocks | 3 blocks | 4 blocks | 5 blocks |
| --- | ---: | ---: | ---: | ---: | ---: |
| 20 | 470 | 920 | 1,370 | 1,820 | 2,270 |
| 40 | 890 | 1,760 | 2,630 | 3,500 | 4,370 |
| 80 | 1,730 | 3,440 | 5,150 | 6,860 | 8,570 |
| 150 | 3,200 | 6,380 | 9,560 | 12,740 | 15,920 |

These counts include all trainable layer norms: `D * (21 * H + 30) + 20`.
The original suffix has 1,380 trainable parameters. Summary labels compare
the complete replacement count to that budget; the native `mlp_parameters`
field counts only affine MLP weights/biases. Fixed-width depth comparisons
also change total capacity. This grid is not a parameter-matched depth study;
future matched-budget comparisons should choose widths separately per depth
and report the exact residual parameter mismatch. One seed does not establish
statistical reliability or a global optimum.

## Run using the already trained source

Build the native executable with the stacked readout flags before starting.
See [REPRODUCE_PUZZLE.md](REPRODUCE_PUZZLE.md) for CUDA setup and build commands.
Use the new executable for `--binary`; an older saved executable does not
understand `--mlp_depth` or `--match_mlp_parameter_budget`.

```bash
python3 -B scripts/memorize_general_facts/run_stacked_mlp_sweep.py \
  --binary /absolute/path/to/new/memorize_general_facts \
  --checkpoint /tmp/pluto-puzzle-from-scratch-20260926-01/model/trial_000_L4_W10_FF20/checkpoints/layers_4/step_89600 \
  --tokenizer /tmp/pluto-puzzle-from-scratch-20260926-01/model/inputs \
  --corpus /tmp/pluto-puzzle-from-scratch-20260926-01/model/inputs/corpus.txt \
  --run_dir /tmp/pluto-stacked-mlp-sweep-NEW \
  --depths 1,2,3,4,5 --widths 20,40,80,150 \
  --steps 300000 --eval_every 1000 --batch_size 32 \
  --seed 3 --learning_rate 0.01 --max_workers 1 --timeout_seconds 3600
```

All five paths are required. `--run_dir` must be new and outside the repository.
This runner accepts the fixed four-block, width-10, FF20, context-27, vocabulary-
4475 checkpoint/corpus protocol only. Its preflight checks source tensor sizes
and 1,024 corpus lines; each native run verifies all 10,002 source predictions
and the captured states before training. It never trains a fresh source model.

At most `--max_workers` native GPU processes run at once (default 1; range 1–4).
Sequential execution is recommended on the tested GH200: a depth-3, width-80
10,000-update benchmark took 14.78 seconds alone versus about 49.8 seconds per
run with three concurrent processes. Parallel processes can compete for CPU
and GPU resources; report the chosen concurrency when comparing wall time. The positive,
finite timeout applies separately to each native process and includes capture,
training, and greedy verification. Increase it when a slow system needs more
than one hour per configuration. SIGINT/SIGTERM stops child process groups,
reaps them, and preserves partial reports. A failed configuration does not stop
the remaining configurations. Resume is intentionally unsupported: use a new
run directory to preserve each attempt's provenance.

## Artifacts and interpretation

`inputs/` holds copies of the executable, checkpoint, tokenizer, corpus, driver,
tests, guide, and relevant native sources. Snapshot files are read-only and
SHA-256 checked before every launch and after completion. Rebuilding the source
executable does not affect already pinned runs. `manifest.json` records hashes,
original paths, file sizes, selected runtime environment, and the exact native
argument lists; `commands.sh` records shell-quoted commands for inspection.
The commands refer to their original output directories, so rerunning them
requires editing those paths to new locations. System CUDA libraries are not
copied; the environment record is not a hermetic toolchain snapshot.

`summary.json`, `summary.tsv`, and self-contained `summary.html` update during
the experiment. The HTML refreshes every 15 seconds when opened locally.
JSON/TSV accuracies are fractions in [0,1]; HTML shows percentages. Each
configuration directory contains `stdout.log`, `stderr.log`, and native
`puzzle/` outputs including `training.tsv`, `run.txt`, and `best_mlp/`.

The best checkpoint maximizes correctly predicted teacher-forced tokens,
breaking ties by cross-entropy. All-token accuracy counts 10,002 targets,
including 1,024 EOS targets. Non-EOS accuracy uses only the other 8,978 targets.
Exact greedy facts counts suffix-plus-EOS completions from each fact's original
five-token prompt. It is a separate measurement and is populated only after
successful completion, shape/count/provenance checks, and both frozen-head and
frozen-source checks. The parser also verifies the complete evaluation schedule
and saved readout tensor shapes. Partial rows can show observed token metrics
but are excluded from rankings, and absent measurements remain blank.

The final status is `completed` only when every configuration passes validation;
otherwise the process exits nonzero and records failures, timeouts, interruption,
or preparation errors. Rankings use token correctness, loss, exact greedy count,
then parameter count, in that order. No summary values are forecasts.

## CPU tests

```bash
python3 -B -m unittest discover -s scripts/memorize_general_facts \
  -p run_stacked_mlp_sweep_test.py -v
```

These tests use mock native processes and temporary checkpoints, covering input
validation, immutable snapshots, exact flags/counts/shapes, partial histories,
rank ordering, failures, timeout cleanup, and signal cleanup. They do not replace
the native GPU tests or the actual experiment.
