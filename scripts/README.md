# Scripts

Operational utilities and experiment orchestration live here; model, dataset,
kernel, and native analysis-tool implementations live in `src/`.

- [`memorize_general_facts/`](memorize_general_facts): depth/width sweeps,
  evidence-checked Pareto reports, corpus/prediction audits, checkpoint conversion,
  and the corresponding Python tests. These tools drive or inspect the native
  experiment; they are not dependencies of its C++ implementation.
  Start with its [from-scratch puzzle guide](memorize_general_facts/REPRODUCE_PUZZLE.md)
  to train a verified source checkpoint and fit the third-attention readout.
- `training_runs/`: recorded training invocations.
- `download-datasets.sh`, `download-gpt2-tokenizer.py`, and
  `test-gpt2-tokenizer.py`: data/tokenizer setup and interactive inspection.
- `gzip_old_checkpoints.py`: checkpoint archival, with its unit test alongside it.
- `generate_finite_state_machine_data.py`: deterministic FSM interpretation
  datasets (4,096 training / 128 test sentences), with an independent interpreter
  test alongside it. Run with Python 3 to regenerate the checked-in `testdata/`
  files; `--help` describes the format and sampling choices.

Run command examples from the repository root. Each utility documents its own
arguments and dependencies; use `--help` where supported. Put new benchmark and
hyperparameter-sweep drivers in a descriptive subdirectory here, keeping their
tests alongside them. Store checkpoints outside the repository and keep generated
run artifacts separate from the utility source files.
