# Frozen late-MLP polynomial extraction protocol

Date: 2026-09-09. Fix this protocol and implementation before candidate
generation and corpus verification. This is a new **static surrogate**, not
GPT-2 inference, a GPT-2 Taylor expansion, or a proven training-text decoder.
No attention kernel, contextual hidden-state computation, model forward pass,
training update, or private-gradient observation is used by the extractor.

## Motivation and evidence boundary

The most recent wider-vocabulary attention-polynomial probe recovered no
four-token substring; earlier formatting matches did not establish passage
recovery. Separate, data-assisted causal validation implicated late MLPs,
motivating the choice of
zero-based blocks 6 and 7 here. That selection is **not data-free**. Once this
method is fixed, however, its candidate generation receives only checkpoint
weights and the frozen numerical choices below, never corpus text or matches.
Any tokenizer labels are display only; token IDs define the candidates.

[KSTER](https://arxiv.org/html/2602.10134v2) extracts an edited-key subspace
from deliberate editing updates and covariance information, then uses model
activations and candidate subject/prompt pools. We do not possess that editing
setup; a final MLP matrix is not its low-rank edit delta. Likewise,
[Buzaglo et al., Section 5.1](https://papers.neurips.cc/paper_files/paper/2023/file/a1d20cc72a21ef971d7e49a90d8fa56f-Paper-Conference.pdf)
motivate reconstruction using stationarity of an explicitly L2-regularized
objective. Our stochastic decoupled AdamW checkpoints do not establish that
stationarity identity. Their inverse-input optimization also evaluates network
gradients; it is not the static procedure used here. Neither paper guarantees
that this polynomial surrogate recovers passages.

## Vocabulary, initial strings, and search

- Length L=16; all repeated tokens, including whitespace and special tokens,
  remain allowed. There is no EOS stopping rule or text-based filter.
- Select S from the final checkpoint's complete logical vocabulary: the 8,192
  rows with largest Euclidean distance from the full-logical embedding mean.
  Break equal selection scores by lowest token ID, then sort selected IDs
  ascending. Physical padded vocabulary rows never participate.
- Use `np.random.Generator(np.random.PCG64(20260909))` and
  `rng.integers(0, 8192, size=(64, 16), dtype=np.int64)`, indexing sorted S.
  These are 64 complete uniformly sampled initial strings, with replacement.
  All five arms use these exact initial IDs and final-checkpoint S.
- Set alpha=0.125. Optimize each string with eight complete coordinate sweeps:
  positions 0..15 on sweep 0, then 15..0 on sweep 1, alternating thereafter.
  At each position consider every ID in S, including the current token; choose
  the highest-scoring replacement, breaking exact score ties by lowest ID.
  Check proposed replacements by complete objective recomputation with fixed
  tolerance `1e-10 + 5e-10*max(|expected|,|recomputed|)`. A replay mismatch or
  score decrease exceeding the corresponding tolerance fails closed. If a
  recomputed proposal decreases the score only within tolerance, retain the
  unchanged row and record that roundoff retention. This tolerance checks
  numerical replay; it does not group near-ties or redefine exact-score ties.
- Keep every restart's final string, even if duplicated or repetitive. No
  corpus-driven reranking, objective tuning, early removal, or post-hoc filter.

## Fixed coefficient compiler

Use row-vector residual conventions. For each arm's checkpoint, E is the
logical token embedding table, P the learned position table, and mu the mean
of **all logical rows of that checkpoint's E**, not just S. Write Ec_t=E_t-mu.
Let H=I-11^T/d, with d=512. Blocks 6 and 7 use their pre-MLP LayerNorms
(`ln2`), not their pre-attention norms. All formulas below are evaluated in
FP64 on FP32 checkpoint values; promote the production constants from FP32.

For LayerNorm gain gamma, bias beta, and raw row r, define

    c = r H
    s = sqrt(c c^T / d + epsilon)
    LN(r) = (c/s) diag(gamma) + beta
    J(r) = [H/s - c^T c/(d*s^3)] diag(gamma).

Here epsilon is FP32 `1e-5` promoted to FP64, and J maps row perturbations to
row perturbations. GELU is the production tanh approximation

    phi(u) = 0.5*u*[1+tanh(C*(u+A*u^3))],

where C is FP32 `0.7978845608f` and A is FP32 `0.044715f`, promoted to FP64.
Use its analytical first and second derivatives, not erf-GELU derivatives.

Use fixed raw anchors `r_p = mu + P[p]`, p=0..15, and
`rbar = mu + mean(P[0:16])`; the last position participates in rbar even though
it is only a prediction target in the objective. Denote MLP input/output
matrices by W16/W26 for block 6 and W17/W27 for block 7. Then compile

    J6_p = J_LN6(r_p)
    g6_p = phi'(LN6(r_p) W16 + b16)
    A6_p = I + J6_p W16 diag(g6_p) W26
    J7 = J_LN7(rbar)
    K7 = J7 W17
    g7 = phi'(LN7(rbar) W17 + b17)
    h7 = phi''(LN7(rbar) W17 + b17).

Input biases and LayerNorm biases affect these fixed gates. Output biases,
constant MLP outputs, and actual preceding-layer activations are omitted.
These anchors are not estimates validated against late contextual geometry.

## Final-LayerNorm numerator and objective

For final LayerNorm gain gamma_f, define the destination column

    d_t = H diag(gamma_f) Ec_t^T
    n_t = W27 d_t.

This is a numerator dictionary, **not** a final-LayerNorm Jacobian. In ideal
arithmetic, the actual class-centered logit at raw final residual r obeys

    logit_t - mean(logits)
      = beta_f Ec_t^T + (r d_t)/sqrt(||r H||^2/d + epsilon).

Thus a last-MLP write has a linear contribution to the numerator, but the
unknown residual, final bias term, and context-dependent normalization still
matter. Our score omits beta_f and the normalization denominator; it is not
a logit, probability, loss, or teacher-forced model evaluation. Centering E
removes the common-embedding-translation gauge E+=c, P-=c in ideal arithmetic.

For string s[0:16], compile position-specific token rows

    direct[p,t] = Ec_t A6_p
    feature[p,t] = Ec_t A6_p K7.

For r=1..15, predict token s[r] using all preceding positions p=0..r-1:

    x_r = (alpha/r) * sum_p direct[p,s[p]]
    z_r = (alpha/r) * sum_p feature[p,s[p]]
    score(s) = sum_r {x_r d_s[r]
                    + (g7*z_r + 0.5*h7*z_r*z_r) n_s[r]}.

Products involving g7, h7, and z are elementwise. Preserve signed neuron
cancellation. Coordinate replacement must include every affected factor:
the token's target readout when its index is nonzero, and its contributions
to every later prefix, with each prefix's own alpha/r scaling.

## Five frozen arms

1. **final_full:** final checkpoint, complete objective above.
2. **final_affine:** same coefficients except h7=0.
3. **final_no_cross:** retain direct[p,t]=Ec_t A6_p, but replace feature[p,t]
   by Ec_t K7 for every position; all other coefficients are unchanged.
4. **final_broken:** replace W26 by `np.roll(W26, 1, axis=0)` without permuting
   W16 or g6. Key j receives original value row (j-1) modulo MLP width.
   Recompile the affected A6/direct/features; this breaks the learned pairing.
5. **early_full:** step-10 checkpoint, its own coefficients and full-logical
   embedding mean, but exactly the same final-selected S and initial strings.
   Step 10 is an early trained baseline, never true initialization.

## Tests, diagnostics, and freezing

Before release, test the exact LayerNorm Jacobian and tanh-GELU curvature
against independent toy finite differences; fixed coefficient formulas;
coordinate replacements against complete objective recomputation; paired
neuron-permutation invariance; and a planted ordered mixed interaction.
The broken arm is intentionally not the paired-permutation invariance test.

Audit ordered mixed signal on the first eight sorted S IDs as both sources
and first sixteen sorted S IDs as targets, for position/target-slot triples
(p,q,r)=(0,1,2) and (0,14,15). Compare the coefficient

    (alpha^2/r^2) * (feature[p,a]*feature[q,b]) * (h7*n_u)

summed over neurons, against the same coefficient with source contents a,b
swapped but positions fixed. Report maximum absolute antisymmetry and relative
antisymmetry `|forward-reverse|/(|forward|+|reverse|)`. If both values are zero,
record the zero denominator and assign relative antisymmetry zero. These are
ordinary FP64
diagnostics, not interval proofs or measurements of GPT-2's order sensitivity.
The no-cross arm has identical feature maps at all positions, so its mixed
antisymmetry vanishes; that does not assert whole-string objective symmetry.

Record each restart's objective progression, linear/quadratic magnitudes,
repetition statistics, and distinct final paths. Fail closed on nonfinite
coefficients, scores, updates, or diagnostics; never silently discard a run.
Record numerical environment and exact file/array hashes.

Write each arm's exclusive plan, including checkpoint/source/protocol hashes,
S and all initial IDs, **before coefficient compilation and search**. Write
exclusive candidate JSONL and completion metadata afterward. Freeze hashes
of all five completed arms before any corpus verification. Retain all 64
outputs per arm: 320 candidate records total, independently of duplicates.

## Independent verification and interpretation

Use the existing native-token byte verifier on the frozen candidates. Primary
outcomes are exact substrings of at least eight tokens and exact full 16-token
matches; also report four- and five-token results. Report full-corpus and
current-prefix/current-suffix matches separately without asserting historical
training membership or authenticated held-out status. Keep the 64-restart
denominator per arm and report unique-candidate counts separately.

When candidate matches are long enough for the relevant control, apply the
existing directed bigram- and trigram-preserving Euler controls using seeds
17, 29, and 43. Preserve endpoints and multiplicities according to those
helpers and report their limitations, including nonuniform shuffle sampling.
Do not interpret three shuffled controls as a calibrated significance test.

A positive result would be a static-surrogate candidate match requiring
independent verification and controls, not a proof of unique weight storage.
A negative result rules out only this fixed decoder protocol, not the
possibility of analytical recovery from the model's weights.
