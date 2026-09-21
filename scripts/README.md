# Scripts

Operational utilities and experiment orchestration live here; model, dataset,
kernel, and native analysis-tool implementations live in `src/`.

- [`memorize_general_facts/`](memorize_general_facts): depth/width sweeps,
  evidence-checked Pareto reports, corpus/prediction audits, checkpoint conversion,
  and the corresponding Python tests. These tools drive or inspect the native
  experiment; they are not dependencies of its C++ implementation.
- `training_runs/`: recorded training invocations.
- `download-datasets.sh`, `download-gpt2-tokenizer.py`, and
  `test-gpt2-tokenizer.py`: data/tokenizer setup and interactive inspection.
- `gzip_old_checkpoints.py`: checkpoint archival, with its unit test alongside it.

Run command examples from the repository root. Each utility documents its own
arguments and dependencies; use `--help` where supported. Put new benchmark and
hyperparameter-sweep drivers in a descriptive subdirectory here, keeping their
tests alongside them. Store checkpoints outside the repository and keep generated
run artifacts separate from the utility source files.
