# Paired Exeunt / Nuveth training experiment

## Question and fixed intervention

Train the current eight-block Pluto GPT-2 recipe twice from one saved random
initialization. Arm A uses the original `testdata/shakespeare.txt`; arm B
replaces every exact, case-sensitive `Exeunt` byte string with `Nuveth`.
Each arm receives **14,400 seconds (four hours)** of training-loop wall time,
on the same GPU, sequentially. Preserve every initial, periodic, and final
checkpoint. Do not overwrite the existing Shakespeare checkpoint history.

The replacement has six ASCII bytes and three tokens, as does the original.
The preparation script verifies this using the **native training tokenizer**,
not a substitute Python implementation. Native and Hugging Face whitespace
pre-tokenization differ. For the full corpus and the actual train/test splits,
require identical token counts, identical replacement token-slot ranges, and
identical token IDs and byte boundaries everywhere outside replacements.
Internal subword boundaries within the three slots may differ.

## Fixed configuration

| Setting | Both arms |
| --- | --- |
| Architecture | 8 pre-LN blocks; width 512; 8 heads; FFN 2,048 |
| Context | 1,024 tokens |
| Vocabulary | 50,257 logical; 50,272 physical |
| Position / output | Learned absolute embeddings; tied embedding/LM head |
| Arithmetic | BF16 activations; FP32 master weights and AdamW state |
| Dropout | None |
| Batch | 10 sequences = 10,240 predicted tokens per update |
| Seed | 17; unchanged per-layer initialization offsets |
| AdamW | LR 0.0003; beta1 0.9; beta2 0.95; epsilon 1e-8; decay 0.1 |
| Split | 90% training / 10% test, byte-based and next-newline aligned |
| Sampling | Same seeded random-window draws, not epoch shuffling |
| Evaluation | Four fixed sequential training batches every 100 updates |
| Final evaluation | Four fixed training and four fixed test batches |
| Checkpoints | Initial step_0, every 100 updates, and final completed update |
| Stopping | Four-hour completed-update wall budget; no target loss/step cap |

Initial evaluation, initialization, and final checkpoint/evaluation are outside
the budget. Periodic evaluation/checkpointing is inside it. Timed training
synchronizes its CUDA executor after each optimizer update, so it cannot stop
based merely on queued launches. Stop only at a completed-update boundary;
minor overshoot by a final update or periodic evaluation is expected.

## Provenance and controls

`scripts/weight_analysis/paired_training.py prepare` exclusively creates a new
experiment directory. It copies the original and replaced text, tokenizer,
optimized trainer, native token exporter, and native sampling-replay binary.
It records SHA-256 hashes, the code commit, all flags, byte/token alignment
evidence, and the first 10,000 sequence starts with the C++ runtime identity.
These frozen inputs are rehashed before every child process.

The runner first saves a fresh `initial/checkpoints/step_0`. Every subsequent
run loads this exact checkpoint into fresh AdamW state; its own saved step_0
must have identical file hashes. It runs two two-update original-corpus
controls and one two-update replacement control before the full arms. These
short controls use the timed path's completed-step synchronization too.

CUDA embedding and attention gradients include FP32 atomic additions. Equal
seeds do **not** prove bit-identical training. The duplicate original controls
measure early numerical variation, not a bound on its eventual amplification.
All runs are uninterrupted: resuming intermediate weight-only checkpoints
would reset optimizer moments and sampling, invalidating the matched protocol.
The runner refuses to restart an already-started experiment automatically.

## Analysis plan and limitations

1. Validate all final weight files, finite values, saved initial hashes, timing,
   loss logs, and completed optimizer-step counts. Preserve the weight files.
2. Compare both four-hour endpoints, explicitly reporting any unequal steps.
   Also compare the latest shared periodic step and earlier paired steps.
   Equal-step comparisons avoid attributing extra optimizer updates to the word.
3. Use `paired_weight_diff.py` for per-tensor/model delta norms, relative norms,
   weight cosines, shared-initial training-direction cosines, embedding-row,
   MLP-neuron, and attention-head rankings. Count tied weights once. Exclude
   padding from vocabulary rankings. Retain exact checkpoint hashes.
4. Evaluate both words' multi-token probabilities in identical corpus contexts,
   plus unrelated continuations. A complete-word probability includes all three
   conditional token probabilities, not just the initial subtoken.
5. Investigate whether selected layer/weight-delta patches transfer the changed
   behavior without destroying unrelated predictions. Large weight differences
   alone do not identify a word's storage location.

The intervention removes one string and teaches another; it does not isolate
pure forgetting. The raw delta includes every downstream training consequence,
plus numerical divergence. The same final weights cannot establish universal
absence of the original word under every possible prompt.

### Preselected word-row transfer check

Before observing the four-hour endpoints, fix the first selective patch to the
union of native token IDs in both words, with and without leading space:
`45, 68, 303, 400, 1475, 2797, 3109, 21733, 45177`. At the latest shared
positive step, create independent original-model checkpoint copies with these
nine embedding rows replaced by the replacement-model rows, and perform the
reverse patch as well. Use the already frozen word/control cases for all
conditions, reporting changes in three-token negative log probability and
unrelated continuation loss separately for training/test and prefix domains.
The set is determined by the spelling intervention, not by final delta ranks.

Preserve both source checkpoints and all patched copies. Verify that selected
rows are exact donor bytes and every unselected byte is unchanged. These rows
are shared by the input embedding and LM head: transfer would identify a
functional contribution of this joint parameter set, not an exclusively
input-side mechanism or proof of a unique memory location. A failed transfer
is also informative; do not expand the selected set without labeling further
patches as exploratory.

The queued final postprocessor covers endpoints and the latest common step.
After both arms finish, also run the read-only weight comparison for each
earlier common positive checkpoint, as required by analysis item 2 above.
The short step-2 controls remain separate from this long-run trajectory.

## Usage

From the repository root, build the optimized trainer and native validators:

```sh
bazel build -c opt //src/llm/recipes:gpt2_shakespeare_llm \
  //scripts/weight_analysis:tokenize_corpus \
  //scripts/weight_analysis:replay_sampler
```

Use a **new** experiment output path. Prepare and run with the existing analysis
Python environment (NumPy is required):

```sh
/home/ubuntu/.venv/bin/python -m scripts.weight_analysis.paired_training prepare \
  --output /home/ubuntu/checkpoints/EXPERIMENT_NAME \
  --corpus testdata/shakespeare.txt \
  --tokenizer-dir /home/ubuntu/datasets/tokenizer/gpt2 \
  --trainer bazel-bin/src/llm/recipes/gpt2_shakespeare_llm \
  --tokenizer-binary bazel-bin/scripts/weight_analysis/tokenize_corpus \
  --sampler-binary bazel-bin/scripts/weight_analysis/replay_sampler
/home/ubuntu/.venv/bin/python -m scripts.weight_analysis.paired_training run \
  --output /home/ubuntu/checkpoints/EXPERIMENT_NAME
```

`state.json` records the runner and child PIDs, exact commands, transitions, and
terminal results. Confirm process liveness independently: a state file alone
does not prove that a run is still active. Each child retains its `process.log`,
`train.log`, `command.json`, `checkpoints.json`, and `checkpoints/step_N/`.
The `training_complete` phase is not a claim that the weight/behavior analysis
has been completed.
