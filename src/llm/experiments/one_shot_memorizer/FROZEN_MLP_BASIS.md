# Can fixed nonlinear features replace the learned MLP directions?

## Protocol fixed before evaluating replacements

The existing output-projection experiment retains the trained GELU features
and reconstructs W2 by a linear solve. It preserves every corpus completion.
Replacing those features with an affine map fails. This next test targets the
unresolved component: **must the 16-to-64 expansion directions be learned,
or can a directly constructed nonlinear basis do the same job?**

Use the original width-16, eight-block checkpoint `step_16128` and its
corresponding `step_0` initialization. Keep each MLP's 64-channel width,
LayerNorm, GELU, BF16 computation, and residual boundary. No attention,
embedding, position or normalization weight is refitted in this experiment.

1. Capture normalized inputs `X` and branch outputs `U` from every real token,
   including prompt tokens, in the original model. Use the existing split:
   every fifth zero-based sentence is held out, giving 819 fitting sentences
   and 205 held-out sentences. The original backbone saw all these sentences
   during training; this is a replacement-map test, not unseen-fact learning.
2. Read the expansion weights from `step_0`, rounded to their effective BF16
   directions `a_j`. Do not use final learned expansion weights to construct
   the experimental basis.
3. On **fitting normalized inputs only**, compute
   `mu_j = mean_i(X_i dot a_j)` and
   `sigma_j^2 = mean_i((X_i dot a_j - mu_j)^2)`.
   Construct FP32 master coefficients `a'_j = a_j / sigma_j` and
   `b'_j = -mu_j / sigma_j`. Reject degenerate/nonfinite directions rather
   than selecting a new direction based on accuracy. The moment helper must
   accept neither teacher outputs nor token labels.
4. Evaluate the resulting FC/GELU with the **actual GPU layers**. BF16
   projection/activation rounding is retained, so theoretical unit variance
   in real arithmetic is not asserted to hold exactly on the GPU. Fit only
   `Z W2 + b2 ~= U` using the existing pivoted-QR solver, with fixed
   mean-square ridge `1e-6` and an unpenalized output bias.
5. Install the constructed MLPs in fresh model forwards. They consume that
   forward's LayerNorm output, including changes propagated from earlier
   replaced blocks. Never replay captured feature vectors or gold branch
   outputs during evaluation.

Compare these fixed conditions:

- Unmodified model and original-layer clone controls.
- Final learned expansion/GELU followed by the directly refitted output map.
- Raw, unstandardized step-0 expansion/bias/GELU followed by a refitted map.
- Moment-standardized step-0 directions/GELU followed by a refitted map.

Evaluate single-block replacements and all eight together. Report fitting
and held-out branch errors and all 10,002 teacher-forced target decisions.
For each joint replacement, separately generate all 1,024 suffixes plus EOS
from five supplied tokens and compare sentence identities, not just aggregate
counts. Verify source weights remain unchanged. Report numerical rank/range
failures honestly rather than silently changing ridge or adding hidden units.

The width, seed source, split, standardization and ridge are fixed before
observing outcomes. Any later expansion of the basis or tuning is a separate,
explicitly post-hoc experiment. The implementation/evaluation budget is roughly
two hours, within the current nine-hour research window.

## Interpretation

Success would replace both learned dense maps in all eight MLPs—17,024
coefficients—with a direct, non-gradient construction **conditional on the
trained backbone and its hidden updates**. It would not construct the entire
model from raw text. The fitting targets `U`, normalized inputs `X`, learned
LayerNorms, attention and embeddings still come from the trained checkpoint.
Nor would an equivalent replacement identify the original matrix coordinates.

Failure would distinguish this particular fixed-width generic feature basis
from the useful directions found during training. It would not prove that no
other deterministic basis, feature count or fitting criterion can work.

The method is motivated by fixed-hidden-feature networks with analytically
solved output weights, such as the algorithm described by
[Huang, Zhu and Siew (2006)](https://extreme-learning-machines.org/pdf/ELM-NC-2006.pdf).
This experiment makes no claim that its narrow BF16 GELU basis satisfies a
general interpolation or approximation guarantee. More broadly,
[Rahimi and Recht (2007)](https://proceedings.neurips.cc/paper/2007/hash/013a006f03dbc5392effeb8f18fda755-Abstract.html)
study linear learning on randomized feature maps designed for kernel
approximation; our particular standardized GELU map is not their Fourier
feature construction.

## Result: fixed initialization directions do not preserve memorization

The prespecified run completed on 2026-09-23. All original/clone controls
passed, including exact regenerated BF16 GELU values and full MLP updates.
The source master weights remained byte-identical. Every reported exact
sentence was independently verified by actual greedy suffix-plus-EOS
generation, with sentence identities agreeing with teacher-forced exactness.

| All eight MLPs together | Correct target decisions / 10,002 | Exact suffixes / 1,024 | Exact fit / 819 | Exact held / 205 |
| --- | ---: | ---: | ---: | ---: |
| Original or exact clones | 10,002 | 1,024 | 819 | 205 |
| Learned expansion; directly refitted W2/b2 | 10,002 | 1,024 | 819 | 205 |
| Raw initial expansion; directly refitted W2/b2 | 5,533 | 13 | 10 | 3 |
| Standardized initial directions; directly refitted W2/b2 | 4,679 | 6 | 4 | 2 |

The negative result is not caused solely by applying several imperfect
replacements together. Single-block replacements also fail, although their
effects differ substantially:

| Replaced block | Raw initial: correct targets / exact sentences | Standardized initial: correct targets / exact sentences |
| --- | ---: | ---: |
| 0 | 7,906 / 202 | 7,937 / 204 |
| 1 | 9,513 / 664 | 8,930 / 395 |
| 2 | 9,989 / 1,013 | 9,846 / 885 |
| 3 | 9,993 / 1,015 | 9,959 / 986 |
| 4 | 9,907 / 937 | 9,616 / 726 |
| 5 | 9,931 / 957 | 9,663 / 758 |
| 6 | 9,940 / 963 | 9,833 / 873 |
| 7 | 9,535 / 651 | 9,291 / 524 |

Every learned-expansion/refitted-output **single-block** condition remains
perfect as well. Thus the fitting, installation, and fresh-input evaluation
path can preserve the original computation, but these fixed alternative
features do not.

On the 2,813 held-out activation rows, the relative update error
`||U_replacement-U_original|| / ||U_original||` is 0.109%–0.144% for learned
features, 9.26%–26.10% for raw initial features, and 16.98%–28.84% for the
standardized features. These errors use the actual BF16 production outputs,
not just the FP64 regression predictions. The fitting group contains 11,285
activation rows. The reported QR ranks describe the **ridge-augmented**
design, not the unregularized feature matrix's intrinsic rank; diagonal
spread is a conditioning diagnostic, not a condition number.

Unit-variance standardization did not solve the failure and generally made
this particular replacement worse. That does not rule out other scales,
thresholds, feature choices, widths, or fitting objectives. Nor do these
measurements establish exclusive fact ownership in block 0: changing an
early branch perturbs all subsequent representations.

The constructive boundary is now clearer: the output maps can be rebuilt by
a linear solve **given the learned features**; reproducing those nonlinear
features with this fixed-width initialization basis is unsuccessful. This is
not an end-to-end construction from text, and the learned direction geometry
remains an unresolved part of the encoding.

## Reproduction and artifacts

```sh
bazel build -c opt //src/llm/experiments/one_shot_memorizer:checkpoint_frozen_mlp_probe
facts=/home/ubuntu/checkpoints/memorize_general_facts/compact_batch_32_no_clip_0
bazel-bin/src/llm/experiments/one_shot_memorizer/checkpoint_frozen_mlp_probe \
  --checkpoint="$facts/layers_8/step_16128" \
  --initial_checkpoint="$facts/layers_8/step_0" \
  --tokenizer="$facts/inputs/tokenizer" \
  --output_dir=/tmp/frozen_mlp_basis_new --batch_size=32
```

Local evidence is `/tmp/frozen_mlp_basis_0/`, with execution output in
`/tmp/frozen_mlp_basis_0.log`. The 30 complete condition evaluations took
about 313 seconds in total, excluding capture and fitting. Files include
aggregate and per-sentence scores, fits, actual update errors, replacement
coefficients, post-GELU feature statistics, and the input-only standardization
moments. `manifest.tsv` ends with `completed=true` only after source-weight
and original-post controls pass. Generated numerical artifacts stay local.
