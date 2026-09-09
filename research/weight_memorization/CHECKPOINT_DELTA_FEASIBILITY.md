# Checkpoint-difference feasibility audit

Date: 2026-09-09. Read-only inspection of checkpoint directory entries and
current optimizer/checkpoint source, plus primary-paper review. No checkpoint
tensor contents, compressed archive contents, corpus text, or model execution
were used in this audit. This is a feasibility assessment, not a decoder result.

## Available evidence

`/home/ubuntu/checkpoints/shakespeare/` contains 1,303 checkpoint entries, every
ten steps from 10 through 13,030, with no missing multiples of ten:

- 1,298 archives, `step_10.tar.gz` through `step_12980.tar.gz`.
- Five unpacked directories, `step_12990` through `step_13030`.
- Each unpacked directory contains exactly `weight_0.bin` through
  `weight_99.bin`; there are no optimizer-state or metadata files there.

The current [writer](/home/ubuntu/code/pluto/src/llm/checkpoint.cc:318) serializes unique layer
weight buffers. These are [FP32 master weights](/home/ubuntu/code/pluto/src/llm/optimizer.h:46),
not BF16 activation storage. It does not save gradients, first/second moments,
hyperparameters, optimizer counters, sampling state, or run boundaries. The
archive naming inventory does not independently authenticate archive contents.

[Resume](/home/ubuntu/code/pluto/src/llm/recipes/gpt2_shakespeare_llm.cc:343) loads weights and
then creates a fresh optimizer. Its [moments start at zero](/home/ubuntu/code/pluto/src/llm/optimizer.cc:87)
and its [local step starts at zero](/home/ubuntu/code/pluto/src/llm/optimizer.h:74), while the
training/checkpoint step number resumes at N. Thus `step_N` is not necessarily
the Adam bias-correction index. Current source/defaults do not authenticate
historical hyperparameters, uninterrupted training, or historical sampler state.

## What the algebra identifies

The [implemented AdamW update](/home/ubuntu/code/pluto/src/llm/optimizer.cc:40), expressed in
ideal real arithmetic, is coordinatewise:

\[
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,\quad
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2,
\]
\[
u_t=\frac{m_t/(1-\beta_1^t)}{\sqrt{v_t/(1-\beta_2^t)}+\epsilon},\qquad
w_t=(1-\eta\lambda)w_{t-1}-\eta u_t.
\]

Here t is the optimizer's local update counter, not necessarily the directory
number. If consecutive first moments were available, the batch gradient would
follow directly as `g_t = (m_t - beta1*m_(t-1))/(1-beta1)`. Consecutive second
moments independently constrain `g_t^2 = (v_t-beta2*v_(t-1))/(1-beta2)`.
Neither state sequence is saved here. FP32 moment storage would introduce
rounding uncertainty even for those otherwise exact identities.

For **known, constant** learning rate eta and decay lambda over an uninterrupted
ten-update interval, let `q = 1-eta*lambda`. The endpoint weights satisfy:

\[
\frac{q^{10}w_t-w_{t+10}}{\eta}
=\sum_{j=1}^{10}q^{10-j}u_{t+j}.
\]

This is a weighted sum of preconditioned momentum updates, **not** a raw
gradient or an average of ten gradients. Unknown intermediate states remain.
Even moments sampled every ten steps would give weighted sums of gradients and
squared gradients rather than isolate individual updates. With current default
betas 0.9 and 0.95, ten steps retain about 35% and 60% of the respective starting
moments; those values are illustrative, not authenticated historical settings.

FP32 weights add a many-to-one limitation. As a scalar illustration, ideal
first-step AdamW with `w0=1`, `eta=3e-4`, `lambda=0.1`, `epsilon=1e-8`, and
zero initial moments, followed by FP32 rounding, yields the same weight bits
`0x3f7fea60` for gradients 0.001 and 0.002. This small CPU arithmetic check is
not a CUDA replay, a gradient-inversion experiment, or a model measurement.

## Applicability of existing attacks

[DAGER](https://proceedings.neurips.cc/paper_files/paper/2024/file/9ff1577a1f8308df1ccea6b4f64a103f-Paper-Conference.pdf)
uses the raw-gradient identity `G = X^T Delta` and a rank-deficient input-span
test (Sections 3–5). Its FedAvg extension uses summed SGD gradients and
approximately fixed features (Appendix B.3), not elementwise Adam-preconditioned
updates. The latter need not preserve the gradient's column span, even when
rank happens to remain small. Its sequence recovery also runs the first
transformer block. For a batch of ten 1,024-token sequences, token count 10,240
exceeds model width 512, so the usual token-count-based rank-deficiency guarantee
would not apply even if raw gradients were available. This conditional batch
calculation is not evidence authenticating every historical training batch.

[FILM](https://papers.neurips.cc/paper_files/paper/2022/file/35b5c175e139bff5f22a5361270fce87-Paper-Conference.pdf)
assumes access to gradients and uses model probabilities for beam search and
reordering (Sections 3–4). It does not establish analytical recovery from our
ten-step AdamW checkpoints. Here the [tied embedding/head](/home/ubuntu/code/pluto/src/llm/layers/embedding.h:68)
also accumulates dense output-head gradients, so a changed embedding row is not
an exact certificate that its token occurred as an input.

## Bounded next step, not an experimental result

Before a checkpoint-difference decoder, test synthetic known low-rank gradients
on CPU: recover their input span as a positive control; compare single-step
Adam updates and ten-step aggregates with known versus omitted moments; and
include a pure weight-decay control whose corrected residual should vanish
within rounding error. Check span preservation separately from rank.

A later prespecified audit could examine all four latest unpacked intervals,
reporting update spectra/span stability and labeling any assumed decay
correction. Passing those tests would establish necessary properties only,
not token identities or sequence recovery. No such applicability experiment
was run for this note. The current evidence does not justify treating these
deltas as gradients; it does **not** prove that the checkpoint trajectory
contains no recoverable information.
