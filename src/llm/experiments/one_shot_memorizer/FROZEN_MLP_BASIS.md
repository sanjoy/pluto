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
