# Checkpoint trajectories: geometry and identifiability audit

Date: 2026-09-09. This is a read-only investigation of a different possible
information source: changes between saved weights. It is not a text extractor,
model evaluation, reconstruction result, or proof that trajectories contain no
recoverable text. No corpus or tokenizer labels enter these measurements.

## Why examine updates separately?

The previous static cross-layer polynomial failed to recover passages and
extrapolated its nonlinear approximation badly. Before attempting another
decoder, this audit asks whether checkpoint changes supply a more identifiable
signal. It uses the five currently unpacked checkpoints at steps 12,990,
13,000, 13,010, 13,020 and 13,030. Directory numbering is not evidence of an
uninterrupted optimizer run or authenticated historical hyperparameters.

The [earlier feasibility note](CHECKPOINT_DELTA_FEASIBILITY.md) established that
the checkpoints save weights only, not Adam moments or intermediate updates.
This audit adds executable numerical controls and actual matrix measurements
instead of treating ten-step endpoint differences as raw gradients.

## Measurements on the saved Shakespeare checkpoints

The current five directories each contain exactly 100 weight files totaling
205,934,592 bytes, with no additional metadata or optimizer-state files. All
500 weight-file hashes and analysis sources remained unchanged during the
audit. Six fixed matrices were examined over all four adjacent intervals:

| Matrix | Relative Frobenius change per interval | Components for 99% of change energy |
| --- | ---: | ---: |
| Logical token embedding | 0.75–0.93% | 353–392 |
| Block 0 QKV projection | 1.84–1.95% | 371–373 |
| Block 0 MLP input | 1.67–1.73% | 448–449 |
| Block 0 MLP output | 2.03–2.09% | 442–443 |
| Block 7 MLP input | 1.40–1.46% | 475–476 |
| Block 7 MLP output | 1.60–1.67% | 443–445 |

All 24 matrices have 512 singular-value estimates above the tool's conservative
Gram roundoff floor. This is a numerical observation about weight changes,
**not** a raw-gradient or training-input-span theorem. A direct SVD of the last
embedding delta agrees with the separate Gram/eigensolver route to maximum
relative discrepancy `3.53e-14`; its smallest singular estimate is 0.0242872,
about 282 times the declared floor. This cross-check uses a different numerical
algorithm, not an independently authored implementation.

Embedding changes look concentrated if only their largest singular direction
is considered: it carries 85.47%, 87.60%, 89.99%, and 87.26% of the four updates'
energy. But their long tails still require hundreds of components for 99%
coverage. Every one of the 50,257 logical embedding rows changes, including
after subtracting the row-mean shift; physical padding rows are excluded.

The uniform row-mean component is a different decomposition. Writing
`DeltaE = ones*DeltaMean + DeltaE_centered` gives the exact orthogonal identity
`||DeltaE||_F^2 = V*||DeltaMean||^2 + ||DeltaE_centered||_F^2`. The shared mean
accounts for **62.81%, 64.18%, 65.69%, and 63.89%** of energy, respectively;
the identity's relative numerical residual is below `4.72e-15`. It must not be
confused with the 85–90% top-singular-direction statistic. The remaining
34–37% is not explained by a shared shift.

For the proposed score using centered source rows, the shared update component
adds the same value to every destination for a fixed source. It therefore
cannot distinguish destination token identities in that score. This does not
say a shared vector contains no learned information, or that all residual
movement is linguistic. No pair or sequence extraction has been run from these
updates, and no corpus-match result is being claimed.

### How much motion remains after geometric alignment?

The independent `embedding_motion.py` audit centers all logical rows, fits an
orthogonal Procrustes rotation, and then fits a uniform scalar against the
earlier centered table. The rotations are uniquely selected at the declared
numerical threshold: all four cross-covariances have full numerical rank.

| Interval | Shared mean / original energy | Energy remaining after rotation / centered energy | Energy after rotation and scale / original energy |
| --- | ---: | ---: | ---: |
| 12990 → 13000 | 62.81% | 46.76% | 17.17% |
| 13000 → 13010 | 64.18% | 42.67% | 15.11% |
| 13010 → 13020 | 65.69% | 37.21% | 12.50% |
| 13020 → 13030 | 63.89% | 42.81% | 15.14% |

The denominators are intentionally explicit. The fitted scalar ranges from
0.9991074 to 0.9993823; it is **not** a recovered AdamW decay coefficient.
Fitting this low-dimensional family accounts for much of the observed movement,
but removing it can also remove real learned computation. In particular,
arbitrary embedding rotations are not automatically full-model symmetries.

The proposed directed score remains asymmetric after these projections: its
antisymmetric component accounts for about 48.65–49.39% of residual score
energy. Synthetic rigid-motion and orthogonal-residual examples show that
directionality alone does not imply linguistic ordering. These are reasons
to require explicit controls before interpreting a future update-based pair
score, not reasons to declare its residual contains no information.

The real motion audit took 9.23 seconds on four requested CPU threads and
verified all 500 endpoint hashes unchanged. Four independently authored dense
vocabulary-space tests check its small-Gram formulas, along with its own 16
tests. No vocabulary-squared matrix is allocated for real checkpoints. The
complete analysis suite currently has **332 passing tests**.

## Executable optimizer counterexamples

`optimizer_geometry.py` uses ideal FP64 CPU arithmetic on known small matrices;
it is not a cuTile/FP32 optimizer replay. The corresponding
[result artifact](optimizer_geometry_results.json) contains the complete small
spectra and error checks. Eight focused tests cover the identities and limits.

- For `G[i,j]=i-j+0.25`, `i,j=0..5`, the raw gradient has rank two. First-step
  Adam with known zero moments and epsilon `1e-8` produces a numerical rank-six
  update; **23.7037%** of its squared Frobenius norm lies outside the original
  column span. SGD preserves that span to numerical precision.
- In a separate zero-epsilon limit, a rank-one outer product remains rank one
  under the sign update, yet **51.5385%** of the update energy leaves its
  original column span. Thus low rank alone is not the needed guarantee.
- Ten controlled gradients sharing one known two-dimensional input span have
  an SGD sum and final first moment of rank two. Their AdamW endpoint change
  has numerical rank six and **6.9088%** energy outside that span. This fixed
  input-span setup is a positive-control assumption, not an assertion about
  real training batches.
- Given every first moment, every gradient is recovered to maximum error
  `1.78e-15`. The ten-step endpoint/weighted-update identity agrees to
  `2.53e-12`; a known pure-decay control leaves residual at most `1.49e-12`.
  These validate the algebra under its stated information assumptions.

There is an important exception to overly broad claims of noninvertibility:
on the **first local Adam step**, with known zero moments, epsilon, normalized
update, and ideal real arithmetic,

```text
u = g / (|g| + epsilon)
g = epsilon*u / (1 - |u|),   |u| < 1.
```

The inverse is ill-conditioned near unit-magnitude updates; the tested matrix
has inverse absolute derivative as large as `2.76e9`. FP32 endpoint storage
also creates exact ambiguities: gradients 0.001 and 0.002 yield the same rounded
scalar weight bits `0x3f7fea60` in the documented one-step example. The ideal
first-step identity does not identify unknown ten-step AdamW histories.

## A directed score can be geometry rather than word order

Let `A` and `B` be two logical embedding tables, each centered using its own
full-vocabulary mean, and `D=B-A`. A possible static pair score is `S=A D^T`.
Positive `S[s,t]` says update row `t` points toward embedding direction `s`;
there is no identity saying token `t` followed token `s` in a training batch.

The following are exact matrix identities, derived here:

```text
B B^T - A A^T = S + S^T + D D^T
S - S^T       = A B^T - B A^T.
```

Pure rigid embedding motion `B=A R`, with orthogonal `R`, leaves its Gram
matrix unchanged but can produce directed `S`. For an infinitesimal rotation
`D=A Omega`, `Omega^T=-Omega`, the score is entirely antisymmetric despite zero
first-order Gram change. This is a counterexample to interpreting directionality
as language order, not a claim that arbitrary rotations preserve the complete
GPT-2 function: learned diagonal LayerNorm gains constrain that reparameterization.

Common coordinate rotations of both tables preserve the score. Centering both
endpoints removes independent common row translations. Centering only the
update is insufficient: translating an uncentered source table by `ones*c^T`
adds `ones*(D*c)^T`, a destination-dependent offset that can change destination
rankings. Conversely, a common shift in an uncentered update, with the source
held fixed, contributes `(A*DeltaMean)*ones^T`: that source-dependent offset
preserves each source's destination ranking but can change summed path scores.

Orthogonal Procrustes alignment removes rigid embedding motion geometrically.
A subsequent fitted scalar removes uniform radial change. Neither operation
is an authenticated AdamW decay correction; either can remove real learned
computation. Alignment imposes symmetry on `A^T D` in coordinate space, **not**
on `A D^T` in token space, so residual directionality still needs validation.

Finally, summing update scores against a fixed source dictionary telescopes:
`sum_k A* (E[k+1]-E[k])^T = A* (E[last]-E[first])^T`. That sum does not add
trajectory-order information beyond its endpoints. Checkpoint chronology is
training-batch chronology, not token-sequence order.

## What nearby literature actually supplies

- [NeuroImprint (2026)](https://arxiv.org/html/2606.20553v1) deliberately constructs
  an adapter that routes samples into separate, at-most-once-activated neurons.
  This extra machinery avoids much of the mixing in ordinary AdamW updates.
  The paper reports approximate AdamW reconstructions and separately discusses
  optional LLM refinement. Its result is not an inversion guarantee for this
  already-trained, unmodified GPT-2 checkpoint collection.
- [TULA (ICML 2025)](https://proceedings.mlr.press/v267/du25d.html) uses before/after
  unlearning weights and knowledge of the unlearning procedure. Its
  [reconstruction procedure](https://arxiv.org/html/2406.13348v2#S4.SS3) computes
  candidate-dependent model gradients to optimize synthetic inputs. It is
  useful evidence that selected updates can leak text, but it is not the
  static, forward-free procedure requested here.
- [Better Embeddings with Coupled Adam (ACL 2025)](https://aclanthology.org/2025.acl-long.1321/)
  analyzes how coordinatewise Adam scaling changes output-embedding geometry,
  including loss of the raw softmax-gradient zero-sum property. The paper's
  Section 2 assumes weight tying but explicitly considers only output-layer
  gradient contributions. That analysis does not eliminate the additional
  input-side contribution in our tied embedding/head.

None of these assumptions is silently added to the saved Shakespeare model.
This review does not prove that no suitable extraction method exists.

## Reproduce the audits and controls

```sh
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python -m scripts.weight_analysis.optimizer_geometry \
  --output /tmp/new_optimizer_geometry.json

OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python -m scripts.weight_analysis.checkpoint_delta_audit \
  --checkpoint-root /home/ubuntu/checkpoints/shakespeare \
  --output /tmp/new_checkpoint_delta_audit.json

OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python -m scripts.weight_analysis.embedding_motion \
  --checkpoint-root /home/ubuntu/checkpoints/shakespeare \
  --output /tmp/new_embedding_motion_audit.json

OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python -m unittest discover \
  -s scripts/weight_analysis -t . -p '*_test.py' -q
```

Use fresh output paths. The measured raw evidence is preserved in
`/tmp/pluto-checkpoint-delta.P8wW9H/audit.json`, its `direct_svd_check.json`,
and `/tmp/pluto-embedding-motion.Iv6LNp/audit.json`. The committed
[result bundle](checkpoint_trajectory_results.json) contains those audit
contents, exact source/checkpoint hashes, and the derived comparison table.
The optimizer controls, delta audit, and motion audit were committed separately
as `57e0a2c`, `a720ec7`, and `d970e48`.

This evidence rejects blindly substituting endpoint changes for the raw-gradient
subspaces assumed by certain reconstruction attacks. It does not establish
whether controlled residual-motion associations can recover text; no such
candidate decoding was performed in this audit. Any follow-up must test that
hypothesis explicitly, rather than interpret geometric directionality as word
order. The original goal—a defensible passage-to-weight map
and a non-inference analytical decompressor—remains open.
