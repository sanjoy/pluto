# Where the quadratic construction fails

This diagnostic follows the fixed construction in
[QUADRATIC_MLP_BASIS.md](QUADRATIC_MLP_BASIS.md). It does not refit its
coefficients, select a better ridge, or change the feature basis.

## Results: small local errors can become large inherited errors

The reproduction at `/tmp/quadratic_mlp_basis_failures_0` exactly matches the
previous coefficients, features, update errors, and per-sentence outcomes;
only elapsed-time fields differ. Joint replacement still gives 996/1,024
complete suffix-plus-EOS completions and 9,968/10,002 teacher-forced targets.
The 28 first mismatches comprise 23 fitting and five regression-held sentences.
Seven occur immediately after the five-token prompt; 21 occur later. None
is an EOS error.

The trace run `/tmp/quadratic_failure_traces_0` completed all 1,176 forwards
reported in `interventions.tsv`. Independent readback of all 5,262,600 saved
logits reproduced every winner, correctness flag, probability, and
target-versus-best-rival margin exactly. The internal byte-identity and
unchanged-weight controls passed. All 104 repository test targets passed.

The following counts concern **the fixed first-error query**, not full
continuation accuracy after repair:

| Block | Wrong with only this MLP approximated, /28 | Rescued by restoring only this learned MLP, /28 | Rescued by resetting the complete block output, /28 |
| --- | ---: | ---: | ---: |
| 0 | 5 | 24 | 24 |
| 1 | 0 | 12 | 24 |
| 2 | 0 | 6 | 25 |
| 3 | 0 | 3 | 25 |
| 4 | 0 | 2 | 25 |
| 5 | 0 | 4 | 25 |
| 6 | 0 | 4 | 25 |
| 7 | 3 | 9 | 28 |

Twenty of the 28 queries tolerate **every individual** MLP substitution but
fail their combination. Restoring some single original function rescues 27
queries; line 430 is the exception. Restoring block 0 alone rescues 24, despite
its isolated substitution breaking only five of these queries. This is
evidence for accumulated/interacting representation errors, not 24 facts
being exclusively stored in block 0.

The vector decomposition supports that interpretation. By block 6 the
inherited-update error has greater norm than the local approximation error in
all 28 query rows; at block 7 it does in 24/28. Block-7 mean L2 norms are
0.06890 local, 0.18000 inherited, and 0.20561 total. These norms are not
additive: the local/inherited dot product is negative in 95/224 query/block
pairs. Repeated positions and blocks are correlated observations, not
independent statistical samples.

Three concrete traces distinguish different failure mechanisms:

* **Line 25:** `A homophone sounds the same` should produce ` as`.
  Original probability is 99.869%; joint replacement gives 17.820% and picks
  a period. Every individual substitution preserves the answer. Replacing
  blocks 0–1 still works; adding block 2 flips it. Restoring block 0, 1, or 2
  separately rescues it, with target probabilities 99.590%, 99.887%, and
  61.366%. There is no unique sufficient repair or exclusive owner.
* **Line 795:** `Condensation occurs when a` should produce ` gas`.
  The original model is already fragile: 50.840% probability and a 0.14225
  logit margin. Replacing blocks 0–6 retains ` gas`; adding the final MLP
  approximation changes the winner to ` liquid`. Restoring only the original
  final MLP rescues it. This is a local final-boundary failure in these tests,
  unlike line 25's multi-block interaction.
* **Line 824:** `Castling moves both a king and a` should produce ` rook`.
  The winner under successively longer replacement prefixes goes
  correct, correct, wrong, correct, wrong, correct, wrong, wrong. Later local
  changes can temporarily compensate earlier ones; a simple irreversible
  “first bad layer” story is inadequate.

These are next-token mechanisms, not necessarily whole-word or semantic-fact
errors. The failures include articles, punctuation, inflections, wordpieces,
and factual nouns. A successful reset is access to the original execution,
not a new dataset-only construction.

## Precision diagnostic

`/tmp/quadratic_precision_audit_0` matches all 19,584 saved FP32 quadratic
coefficients bit-for-bit and passes original capture/weight controls. With
FP64 coefficients and CPU FP64 accumulation, relative held update errors are:

| Block | FP32 products (no BF16 product rounding) | BF16 products |
| --- | ---: | ---: |
| 0 | 3.07828% | 3.07430% |
| 1 | 2.49843% | 2.49639% |
| 2 | 1.37040% | 1.37102% |
| 3 | 1.55921% | 1.56107% |
| 4 | 1.65446% | 1.65658% |
| 5 | 2.86067% | 2.86199% |
| 6 | 2.48489% | 2.48617% |
| 7 | 6.20024% | 6.20281% |

The changes are at most 0.003974 percentage points and have mixed signs.
Product rounding alone does not explain the residual. This does **not** prove
that higher-order GELU terms are necessary: the targets and normalized inputs
are still BF16, and the teacher's intermediate quantization and ridge fitting
remain. Ridge-augmented rank 152 is not an intrinsic-rank measurement. Nor
does update-regression failure prove that no differently fitted quadratic
network can reach perfect argmax accuracy.

The fixed ridge penalty is only 0.039%–0.525% of the present fitting data term
for unrounded products, but that does not exclude a higher-norm exact solution.
A separate unregularized diagnostic, if run, must preserve this distinction.

## Protocol

Select **every first greedy mismatch** of the jointly replaced model. The
first five tokens were given; every later prefix token was predicted correctly
before this mismatch. The saved prefix therefore equals the corpus prefix,
but neither the wrong token nor any subsequent gold token enters inference.
This study examines one next-token decision per failed sentence, not its
subsequent off-corpus continuation. It is a failure-selected diagnostic, not a
new held-out accuracy estimate.

Load the saved FP32 coefficients, run the same BF16 quadratic-feature and
projection layers, and capture both the original and jointly replaced models.
Validate the saved first wrong token against a fresh batch-one execution.
Capture all causal-prefix positions, not just the final query row.

The original MLP still executes before its output is replaced. This provides
three directly observed updates for each block and position:

* `a = learned(x_original)`;
* `b = learned(x_joint)`;
* `c = quadratic(x_joint)`.

The total branch-output change decomposes exactly as
`c-a = (c-b) + (b-a)`. Here `c-b` is the local function approximation at the
actually altered input, and `b-a` is the effect of the inherited input change
through the original MLP. Report all coordinates, both norms, their dot
product, and the total norm. These are **not additive attributions of the
final logit**: residual addition, later normalization, attention, MLPs, and
rounding still act downstream. In particular, norms can cancel and cannot be
interpreted as percentages of responsibility.

For each block, independently rerun these live interventions:

| Condition | Operation | Question |
| --- | --- | --- |
| `single` | Replace only this MLP. | Is this approximation sufficient to break this particular decision on the original upstream state? |
| `prefix` | Replace MLPs 0 through this block. | How does the decision change as approximations accumulate? |
| `restore_function` | Replace every MLP except this one. | Does restoring this learned function rescue the joint failure? |
| `reset_state` | Joint replacement, then restore the original activation after this entire transformer block at every causal-prefix position. | Does the remaining approximate suffix work when started from the original boundary state? |
| `identity_reset` | Original model with the same original boundary activation restored. | Is the patch genuinely an identity operation? |

State resets include the residual stream and do not restore weights. They are
diagnostic access to original-model activations, **not** a proposed inference
algorithm or a label-free way to repair a model. No repair is fitted or
selected for deployment. Restoring one function need not help monotonically;
one intervention can rescue multiple errors or introduce a new one.

## Required controls and artifacts

The standalone `checkpoint_quadratic_trace_probe` checks:

* all original predictions equal the expected target, and all joint predictions
  equal the saved first mismatch;
* all 4,475 logits match bitwise for capture versus plain execution and empty
  substitution versus the ordinary model;
* all identity-reset and ordinary post-pass logits match their original;
* resetting the final block reproduces every original logit, and the full
  replacement prefix reproduces every joint logit;
* every original master tensor (including the tied alias) and every quadratic
  coefficient is unchanged after the experiment.

The output directory contains input/target/prediction text, intervention
scores against **every vocabulary rival**, raw FP32 logits for every reported
condition, per-row decomposition norms, all 16 coordinates of the three
updates, and a completion manifest. Generated artifacts remain local.

The separate `checkpoint_quadratic_precision_probe` addresses a different
question: whether removing BF16 rounding of polynomial **features** removes
the fitting residual. It fits both feature variants on the identical 819
sentences and scores the same 205 regression-held sentences. Its FP64 dot
products are not a simulation of GPU MMA, and its targets remain captured
BF16 learned updates. It does not change the deployed quadratic construction.

## Reproduction

First run the unchanged quadratic experiment documented in
[QUADRATIC_MLP_BASIS.md](QUADRATIC_MLP_BASIS.md), using the current driver so
that it also writes `first_failures.tsv`. Then:

```bash
bazel build -c opt \
  //src/llm/experiments/one_shot_memorizer:checkpoint_quadratic_trace_probe \
  //src/llm/experiments/one_shot_memorizer:checkpoint_quadratic_precision_probe

facts_run=/home/ubuntu/checkpoints/memorize_general_facts/compact_batch_32_no_clip_0
tools=bazel-bin/src/llm/experiments/one_shot_memorizer

"$tools/checkpoint_quadratic_trace_probe" \
  --checkpoint="$facts_run/layers_8/step_16128" \
  --tokenizer="$facts_run/inputs/tokenizer" \
  --construction_dir=/tmp/quadratic_mlp_basis_failures_0 \
  --output_dir=/tmp/quadratic_failure_traces_new

"$tools/checkpoint_quadratic_precision_probe" \
  --checkpoint="$facts_run/layers_8/step_16128" \
  --tokenizer="$facts_run/inputs/tokenizer" \
  --reference_report=/tmp/quadratic_mlp_basis_failures_0 \
  --output_dir=/tmp/quadratic_precision_audit_new
```

Output directories must be fresh. Match the construction report to the
checkpoint and corpus used to create it; the trace reader validates tensor
coverage and prefix/target consistency, not cryptographic artifact provenance.
