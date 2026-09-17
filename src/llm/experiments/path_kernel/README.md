# Shakespeare path-kernel experiment

This experiment implements the training-path decomposition in Pedro Domingos's
[Every Model Learned by Gradient Descent Is Approximately a Kernel Machine](https://arxiv.org/abs/2012.00152).
Unlike the neighboring fixed-weight [NTK experiment](../ntk/README.md), it runs
actual plain gradient descent and remeasures derivatives at each update. It
attributes each selected query logit's change to individual training examples.

The code and tests are AI-generated and have not yet been human reviewed.

## Run

From the repository root:

```sh
bazel build -c opt //src/llm/experiments/path_kernel:shakespeare_path_kernel
bazel-bin/src/llm/experiments/path_kernel/shakespeare_path_kernel \
  --tokenizer_dir=/home/ubuntu/datasets/tokenizer/gpt2 \
  --corpus=testdata/shakespeare.txt \
  --seed=123 --train_examples=2 --eval_examples=1 \
  --context_tokens=16 --stride=1025 \
  --steps=3 --learning_rate=1e-5 \
  --output_dir=/tmp/shakespeare_path_kernel_initial
```

Start a new trajectory at an existing checkpoint and include a prompt query:

```sh
bazel-bin/src/llm/experiments/path_kernel/shakespeare_path_kernel \
  --checkpoint=/home/ubuntu/checkpoints/shakespeare_0/step_15000 \
  --tokenizer_dir=/home/ubuntu/datasets/tokenizer/gpt2 \
  --train_examples=2 --eval_examples=1 --context_tokens=16 \
  --prompt='[Ex' --steps=3 --learning_rate=1e-5 \
  --output_dir=/tmp/shakespeare_path_kernel_checkpoint
```

Supply a checkpoint directory that actually exists, not its parent. The
checkpoint is an initial condition only: these are **new plain-GD updates**, not
a reconstruction of the AdamW trajectory that originally produced the weights.
The checkpoint is never overwritten; updated weights live only in this process.
The required report directory must not already exist and its parent must exist.

The full corpus is memory-mapped and tokenized once. Training contexts come
first, followed by held-out query contexts, at offsets
`offset + example_index * stride`. Context-plus-target windows cannot overlap.
Held-out means omitted from this new trajectory; an existing checkpoint may
already have seen that text. A prompt is an additional unlabeled query and never
enters the training loss.

`--output_token_ids=ID,ID,...` selects the logits to explain. The default is the
sorted union of training next-token labels. Explicit IDs must be unique and in
the logical vocabulary; they can omit every training target. This selection is
**only a readout restriction**. Every training
example's loss and gradient include all 50,257 vocabulary entries.

## What is measured

Let `f_q(theta)` be a selected query logit, `ell_i(theta)` be the full-vocabulary
cross entropy of training example i's next token, M the number of training
examples, and eta the learning rate. At each step s:

```text
Jq[s,q,:] = grad_theta f_q(theta_s)
Jl[s,i,:] = grad_theta ell_i(theta_s)

theta_(s+1) = theta_s - eta/M * sum_i Jl[s,i,:]
K_s        = Jq[s] Jq[s]^T
C_s[q,i]   = -eta/M * dot(Jq[s,q,:], Jl[s,i,:])

path_kernel[q,r] = sum_s eta * K_s[q,r]
contribution[q,i] = sum_s C_s[q,i]
reconstructed[q] = f_q(theta_0) + sum_i contribution[q,i]
residual[q] = f_q(theta_final) - reconstructed[q]
```

There is no momentum, weight decay, clipping, Adam normalization, or ridge fit.
All derivatives in one step use the same pre-update weights. Every unique FP32
master parameter participates. The tied token embedding/LM head is differentiated
through both uses but counted and updated once. No parameter subsampling, random
projection, kernel normalization, or output trace is applied.

For the vector-valued model, the example loss gradient implicitly sums all logit
derivatives weighted by `softmax(logits) - one_hot(target)`. One loss backward per
example therefore includes every vocabulary class without storing 50,257
Jacobian rows. The cumulative **unweighted** query kernel and signed,
loss-weighted example contributions are different objects. The former is a
positive-semidefinite Gram sum in exact arithmetic; the latter is neither a
symmetric kernel nor necessarily positive.

The paper's representation is a gradient-flow statement. Here the integral is
a left-endpoint sum along finite updates. We retain the actual final forward
outputs and residual instead of declaring the reconstruction exact. No scalar
coefficient shared across all queries is inferred by dividing a contribution by
a path-kernel entry: the paper's Remark 1 allows query-dependent coefficients,
and dividing by a zero or cancelling entry would be ill-defined anyway.

## Architecture, precision, and cost

This uses the unchanged full GPT-2 recipe: eight transformer blocks, width 512,
eight 64-wide attention heads, 2048-wide MLPs, learned positions, and tied LM
weights. The vocabulary has 50,257 logical entries and 50,272 physical entries.
Padding entries are excluded from the cross-entropy denominator and receive zero
loss seed. The loss is evaluated only at the last real context position.

The default context contains 16 real tokens, but the model still processes its
fixed 1024 positions. Later positions are EOS-padded; causal attention prevents
them from affecting the selected output. Short contexts make the experiment
easier to interpret, not computationally equivalent to a 16-position model.

`--compute_type=fp16` uses the existing FP32 activation-storage/FP16-MMA policy;
it is not full FP32 computation. `--compute_type=bf16` uses BF16 activations.
Parameters and accumulated derivatives remain FP32. Gram reductions and update
accumulation use fixed-order FP64 calculations, with final weights cast to FP32.
Loss log-sum-exp is computed in FP64; model backward receives FP32 softmax seeds.
Existing mixed-precision backward rules and rounded weights/activations make
this an implemented-backpropagation path decomposition, not the derivative of a
smooth, exact-arithmetic network. Decreasing the step size need not eliminate
the precision component of the residual.

With P unique parameters, Q selected scalar query logits, and M training
examples, the explicit Jacobian needs `4 * P * (Q + M)` device bytes per step.
The default three contexts and two distinct selected tokens have Q=6 and M=2.
The full model has roughly 51.5 million parameters: that Jacobian alone is about
1.53 GiB. `--max_jacobian_mib=4096` limits this storage, not model weights,
backups, forward/backward saved state, or temporary reductions. Requests above
the limit fail instead of silently approximating the decomposition.

This is a tiny-data research experiment, not an efficient replacement for
full-corpus language-model training. Historical checkpoints alone do not provide
the intervening minibatches, losses, derivatives, or optimizer state needed to
reconstruct an old training path. In particular, this implementation does not
treat differences between widely separated AdamW checkpoints as plain GD.

## Reports and interpretation

Each successful run writes a self-contained `report.json` with all settings,
exact input tokens and offsets, output-coordinate mapping, parameter blocks,
per-step losses/Grams/contributions, final training losses, initial and final
logits, reconstructed logits, and residuals. Every query row includes separately
sorted `supporting` and `opposing` training examples. Positive contributions
raise that particular token's logit; they do not necessarily improve accuracy.
Zero contributions appear in the matrix but not in either ranking. These are
path-based decompositions, not counterfactual leave-one-example-out effects.

Convenience CSVs duplicate the cumulative unweighted `path_kernel.csv`, signed
`contributions.csv`, and actual-versus-reconstructed `predictions.csv`. JSON
text/path fields ending in `_escaped_bytes` contain C-escaped bytes; C-unescape
the parsed string to recover even incomplete UTF-8 tokens and arbitrary Linux
pathnames. No probabilities over the selected output subset are presented as
full-vocabulary confidence.

For useful research, compare trajectories from the same initialization with
different step sizes over a comparable total time `steps * learning_rate`.
Inspect both absolute residual and residual relative to the observed logit
change. A large residual means this finite-step/precision setting does not
reliably reconstruct that output; do not interpret its example ranking as an
exact account of the final model. Loss decreases alone are not that validation.

## Tests

```sh
bazel test -c opt //src/llm/experiments/path_kernel/... \
  --test_output=errors --local_test_jobs=1 \
  --test_env=PLUTO_GPT2_TOKENIZER_DIR=/home/ubuntu/datasets/tokenizer/gpt2
```

The CPU report tests cover row mapping, signed rankings, actual/reconstructed
values, per-step records, malformed inputs, byte escaping, and exclusive report
creation. Engine tests separately validate mathematical identities, real
parameter updates, finite-step residuals, and state-restoration behavior.

The full GPT-2/Shakespeare test independently collects full-vocabulary loss
gradients and a query-logit gradient, then checks every master-weight update
against CPU calculations and checks contributions against CPU dot products.
It also verifies tied-weight handling, gradient restoration, decreasing loss,
and bitwise repeatability from the same seed. Analytic tests distinguish an
affine model's exact reconstruction from a nonlinear model's nonzero Taylor
remainder, including the expected quadratic one-step scaling with learning rate.

The default three-step initialized experiment was exercised end to end: mean
training CE decreased from 10.8076 to 10.2405; the largest absolute cumulative
reconstruction residual was 0.00241 logits, versus a largest observed change of
0.60580 logits. All four exported files were byte-identical on repetition.
Changing the diagnostic token subset left the training losses unchanged.
These tiny-data checks validate implementation, not language-model quality.

Memory checking of `path_kernel_test` and `loss_and_update_test` also passed
using `compute-sanitizer --tool memcheck --report-api-errors explicit
--error-exitcode 1`. As noted in the NTK documentation, `explicit` excludes an
existing extended host-memory-pool driver diagnostic while retaining explicit
CUDA API failures and memory-error checking.
