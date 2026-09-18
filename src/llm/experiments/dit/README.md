# Dynamic Influence Tracker on GPT-2 / Shakespeare

This experiment implements the training recorder and influence-inference
algorithm from [Dynamic Influence Tracker: Measuring Time-Varying Sample
Influence During Training (Xu and Wu, 2025), v1](https://arxiv.org/html/2502.10793v1).
It also provides ordinary greedy text inference from the recorded checkpoints.
This new experimental code and its tests are AI-generated and not human reviewed.

**Backend:** PyTorch automatic differentiation, including true loss
Hessian-vector products through the entire model. Pluto's native cuTile `Layer`
API currently supplies first derivatives only; this experiment does not pretend
that gradient similarity is a substitute for second derivatives. It is separate
from the native C++ training executable and does not change its optimizer.

## Model and examples

The default architecture matches `src/llm/recipes/gpt2.cc`: 8 pre-LayerNorm
blocks, width 512, 8 heads of width 64, MLP width 2,048, tanh-approximate GELU,
learned 1,024-position embeddings, no dropout, and a tied token embedding/head.
There are 50,257 softmax classes and 50,272 physical embedding rows. Padding rows
are not classes. The 100 unique tensors contain 51,483,648 trainable parameters.

Attention uses explicit matrix products and softmax so its backward is itself
differentiable. Compute is FP32 by default (FP64 is available for numerical
experiments), without AMP or TF32. This preserves the architecture, not bitwise
native FP16/BF16 arithmetic. Initialization uses a reproducible PyTorch seed,
not the native C++ normal sampler. Native FP32 master-weight checkpoints can be
imported/exported with the correct dense-matrix transposes and tied-weight order.

Each example is a disjoint, contiguous `L+1`-token Shakespeare window. Its first
`L` tokens are inputs and its last `L` tokens are shifted next-token targets.
Its scalar loss is mean full-vocabulary cross entropy over all `L` positions.
`--batch_size` counts examples, not tokens. Deletion removes a whole example,
not a document or a single token. `--context_tokens` can shorten examples for
tractable studies without changing model width/depth or position-table size.
`--max_examples` selects a prefix of the complete windows; omitting it uses all.
The incomplete trailing window is discarded, never padded into the loss.

## Setup and commands

Run from the repository root. Python 3.12 and PyTorch 2.8.0 + CUDA 12.9 were
tested on the GH200. Use an appropriate PyTorch wheel for other hardware; CPU
works but is much slower for the full GPT-2. Nothing downloads the corpus or
tokenizer implicitly.

```sh
python3 -m venv /tmp/pluto-dit-env
/tmp/pluto-dit-env/bin/python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu129
/tmp/pluto-dit-env/bin/python -m pip install -r src/llm/experiments/dit/requirements.txt
```

Record training from scratch. Output directories must not already exist:

```sh
/tmp/pluto-dit-env/bin/python -m src.llm.experiments.dit.dit train \
  --run_dir /tmp/shakespeare-dit \
  --corpus testdata/shakespeare.txt \
  --tokenizer_dir /home/ubuntu/datasets/tokenizer/gpt2 \
  --steps 16 --batch_size 2 --context_tokens 64 --max_examples 16 \
  --learning_rate 0.001 --seed 123 --checkpoint_interval 4
```

The model dimensions remain the full GPT-2 recipe; this short run is an
algorithm demonstration, not a claim of training a fluent language model.
For a useful pretrained starting point, add
`--initial_checkpoint /path/to/native/step_N`. That checkpoint becomes **step 0
of a new SGD trajectory**. A directory of old AdamW checkpoints alone is not a
DIT trajectory: it lacks the required per-update batches, rates, and SGD states.
Add `--export_checkpoint /new/native/checkpoint` to export final weights for
Pluto's ordinary inference executable; the export alone cannot replay DIT.

Compute influence on a next-token logit over a late window, with optional true
leave-one-out retraining as a check:

```sh
/tmp/pluto-dit-env/bin/python -m src.llm.experiments.dit.dit infer \
  --run_dir /tmp/shakespeare-dit --t1 4 --t2 16 \
  --query logit --prompt 'First Citizen:' --target_id 198 \
  --sample_ids 0,1,2,3 --validate_loo \
  --output /tmp/shakespeare-dit-influence.json
```

Token 198 in GPT-2 is a newline. `--query loss` asks about next-token negative
log likelihood and requires `--target_id`; `--query probability` asks about its
softmax probability. Without `--target_id`, logit/probability queries choose the
final checkpoint's argmax **once**, then hold that token fixed across endpoints
and counterfactuals. For comparisons between windows, specify the same ID.

Parameter queries use a named PyTorch tensor and its actual coordinates:

```sh
/tmp/pluto-dit-env/bin/python -m src.llm.experiments.dit.dit infer \
  --run_dir /tmp/shakespeare-dit --t1 0 --t2 16 \
  --query parameter --parameter token_embedding.weight --coordinate 198,0 \
  --sample_ids 0,1 --output /tmp/shakespeare-dit-parameter.json
```

The JSON contains the fixed query, endpoint values, example IDs, corpus-token
offsets, decoded training text, signed DIT scores, every per-step contribution,
and optional finite-LOO values/absolute errors. Console output sorts by absolute
score. Omit `--sample_ids` to report every example; this does not change the
Hessian, which always includes the full recorded batch. Reports are never
overwritten. Re-run `infer` for any other `[t1,t2]` in the recorded trajectory.

Ordinary text inference is a separate subcommand:

```sh
/tmp/pluto-dit-env/bin/python -m src.llm.experiments.dit.dit generate \
  --run_dir /tmp/shakespeare-dit --step 16 \
  --prompt 'First Citizen:' --max_new_tokens 40
```

Generation is greedy, stops at EOS, and uses the most recent 1,024 tokens if it
outgrows the context window. Both inference commands accept `--tokenizer_dir` to
relocate identical tokenizer files. The original raw corpus is not needed:
the run stores the exact training token windows and corpus/tokenizer hashes.

## Algorithm and interpretation

Let `theta[t]` mean weights after **t completed updates**, and let `S[t]` be the
recorded batch. Training is exactly the paper's plain SGD, with no momentum,
AdamW, clipping, or weight decay:

```text
theta[t+1] = theta[t] - eta[t] * mean(i in S[t], grad loss_i(theta[t]))
```

DIT propagates the first-order effect `delta_j` of removing example `j`:

```text
H[t] = Hessian(mean loss on the ORIGINAL batch S[t]), at theta[t]
delta_j[0] = 0
delta_j[t+1] = (I - eta[t] H[t]) delta_j[t]
               + 1{j in S[t]} eta[t] / |S[t]| * grad loss_j(theta[t])
score_j[t1,t2] = grad query(theta[t2]) . delta_j[t2]
                 - grad query(theta[t1]) . delta_j[t1]
```

`influence.py` implements Algorithm 2 as one combined reverse adjoint. It starts
at the final query gradient, propagates with exact autodiff `H[t] @ vector`,
and subtracts the start query gradient at the window boundary. Query gradients
are recomputed at both endpoints. There is no dense Hessian or Hessian inverse.
The library accepts an arbitrary differentiable scalar `query_fn(model)`, so
activation/feature queries can be supplied without changing the DIT engine.
Generic training callbacks must be deterministic, sample-separable, and leave
model buffers unchanged. Nonzero dropout and BatchNorm are rejected; custom
stateful/batch-coupled layers must not be used. Loss and query callbacks must
retain any derivative graph needed by their caller.

Two indexing details are deliberate corrections/clarifications of the paper's
pseudocode, following its equations:

- Store **theta[0]**, interval checkpoints, and theta[T]. An update at index `t`
  needs the pre-update state theta[t], not theta[t+1].
- A late-window query still traverses history back to **zero**. Truncating the
  reverse loop at `t1`, as in Appendix D's presentation, drops effects of earlier
  removals. Our tests include an example absent from a late window whose score
  is nevertheless nonzero.

Positive scores mean **removal increases the query**. For a loss, positive
means the sample helped; for a target logit/probability, positive means it
suppressed that target. These are not probabilities or fractions of a prediction.

DIT is an approximation to **finite deletion**, not exact leave-one-out
retraining. More precisely, this recurrence is the derivative of continuously
downweighting a sample, evaluated at the original trajectory. The full deletion
can follow a different nonlinear trajectory. `--validate_loo` measures that
actual difference separately. Both deletion endpoints are retrained from
theta[0], using the original batch denominator even after removal.

## Storage, replay, and cost

Each run has `manifest.json`, `examples.pt`, and `step_XXXXXXXX.pt` files. The
manifest records actual batches/learning rates, parameter layout/aliases,
checksums, runtime/precision settings, source fingerprints, and corpus metadata.
Torch files are loaded with `weights_only=True`; checksums detect corruption,
not malicious replacement of both files and their manifest. Incomplete runs
without a finalized manifest are not readable. Training does not auto-resume an
interrupted run. Keep an interrupted directory for investigation and use a new
one to retry.

The CLI enforces matching code, PyTorch/Python versions, GPU model, thread count,
and deterministic settings for replay. Cross-runtime bitwise identity is not
promised. Saved checkpoints can be used for ordinary generation on a different
device; reconstructing unsaved steps requires the original replay environment.

One FP32 full-model checkpoint is about **206 MB**. Saving every update of a
100-step run uses about **20.8 GB**. Larger `--checkpoint_interval` trades disk
space for replay work and CPU memory: reverse replay holds at most one interval
of CPU weight snapshots. DIT keeps a constant number of model-sized GPU vectors,
not one per training example. Each reverse step costs a full-batch HVP and one
first backward per selected example appearing in that batch. Finite-LOO
validation is additionally proportional to selected examples times training
length, so start with a few examples.

## Tests

```sh
PLUTO_GPT2_TOKENIZER_DIR=/home/ubuntu/datasets/tokenizer/gpt2 \
  /tmp/pluto-dit-env/bin/python -m unittest discover \
  -s src/llm/experiments/dit -p 'test_*.py' -v
```

These tests are independent of Bazel's native C++ suite. They check full-model
second derivatives on small configurations, tied-weight/native layouts,
causality, deterministic CPU/GPU replay, persisted CLI training/inference,
checksum/overwrite failures, indefinite Hessians (not Fisher outer products),
noncommuting-Hessian time ordering, separate endpoint gradients, late-window
history, and an independent central-difference downweighting oracle. They also
check that finite deletion is **not** falsely equated with its linearization.
Optional real-corpus tests use the environment variable above; CUDA tests skip
only if CUDA is unavailable.

To also run the full 51M-parameter GPU integration test, add
`PLUTO_DIT_FULL_MODEL=1` to that command. This uses the real Shakespeare corpus,
checks repeated DIT results for exact equality, and compares a late-window
estimate against actual deletion retraining. It requires a CUDA GPU and about
1 GB of temporary disk space; missing prerequisites fail this opt-in test.

### Verification on GH200

All **71 Python tests**, including the opt-in full-model test and real tokenizer,
passed. The unchanged native suite also passed all **61 Bazel tests** with its
tokenizer/parquet environment variables configured.

The full-model test trained three SGD updates with batch size 2, four 8-token
examples, and learning rate `1e-5`. For the newline next-token loss after
`First Citizen:`, window `[1,3]`, the following removal effects were measured:

| Example | DIT estimate | Actual finite leave-one-out |
| --- | ---: | ---: |
| 0 | -0.000141498 | -0.000082016 |
| 1 | -0.001648054 | -0.001626015 |
| 2 | +0.010976035 | +0.010981560 |
| 3 (unused) | 0 | 0 |

The largest absolute discrepancy was about `5.95e-5`. This is a short-run
correctness check, not evidence that long trajectories or full-strength
deletions will stay in the linear regime. The analytic tests separately verify
the exact infinitesimal derivative and deliberately demonstrate cases where
finite deletion differs.

The standalone CLI was also exercised at the full **1,024-token context**, with
two examples, batch size 2, two SGD updates, and learning rate `1e-5`. For the
same query over `[1,2]`, DIT gave `+0.002734005` and `+0.002775701`; actual finite
LOO gave `+0.002737045` and `+0.002777100`. Thus the largest absolute discrepancy
was approximately `3.04e-6`. Ordinary generation from an unsaved, replayed step
was checked separately. These runs verify the execution paths; two updates from
scratch do not produce a useful language model.
