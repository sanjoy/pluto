# Capacity, compression, and sentence influence

These are accounting results and proposed tests, not a claim that a parameter
contains a fixed number of factual bits. Counts below describe the original
`compact_batch_32_no_clip_0/layers_8/step_16128` model: eight blocks, width 16,
one head, FF width 64, 4,475 tokens, and 1,024 position embeddings. Each branch
includes its pre-LayerNorm. The token embedding and LM head share one tensor.

## Verified parameter and precision accounting

| Module | Parameters | FP32 bits | Inference mixed-precision bits |
| --- | ---: | ---: | ---: |
| One attention branch | 1,120 | 35,840 | 19,456 |
| One MLP branch | 2,160 | 69,120 | 36,352 |
| All eight attention branches | 8,960 | 286,720 | 155,648 |
| All eight MLP branches | 17,280 | 552,960 | 290,816 |
| Tied token embedding / LM head | 71,600 | 2,291,200 | 1,145,600 |
| Position embedding | 16,384 | 524,288 | 524,288 |
| Final LayerNorm | 32 | 1,024 | 1,024 |
| **Whole model** | **114,256** | **3,656,192** | **2,117,376** |

Attention has a 16x48 QKV matrix, 48-element bias, 16x16 output matrix,
16-element bias, and 32 LayerNorm parameters. The attention operation itself
has no weights. The MLP has 16x64 and 64x16 matrices, biases of 64 and 16,
and 32 LayerNorm parameters. These counts follow `src/llm/gpt2.cc`.

For BF16 inference, `layers/fully_connected.cc` and `layers/embedding.cc`
cast dense matrices and token embeddings to BF16 before use. That covers
96,176 coefficients. The other 18,080 coefficients are read as FP32: biases,
LayerNorm scales/offsets, and position embeddings. In particular, the position
kernel adds an FP32 position value before rounding the resulting activation;
it does not first round the position value to BF16.

Thus storing the BF16-consumed values plus the remaining FP32 values needs
264,672 bytes instead of 457,024 bytes, excluding metadata. This is an
inference representation bound, not a measured compressor result. Check exact
logits after canonicalizing those BF16 operands before claiming an implemented
equivalence. Resuming training still requires the FP32 master values and
optimizer state; the mixed representation does not preserve that trajectory.

An older corpus audit reports a longest sentence of 26 tokens. **Verify this
against the current tokenizer/corpus before using it.** If true, position rows
26..1023 are unnecessary for these corpus completions: 15,968 parameters, or
63,872 FP32 bytes. Causality, ignored padding targets, and zero weight decay
predict that these rows remain at initialization; compare the checkpoint bytes
to confirm. This does not make those positions irrelevant to arbitrary prompts.

## Why one sentence can change many weights

For a dense layer `Y = X W + b`, one sentence contributes
`dL/dW = sum_t x_t outer g_t`, where `g_t = dL/dY_t`. In exact arithmetic its
rank is at most `min(active_positions, input_width, output_width)`. Active
positions include prompt positions receiving gradients indirectly through
attention, not just positions with a directly scored loss. The same identity
applies to QKV/output projections despite attention's nonlinear computations.
An outer product can be dense: low rank does not mean few changed coordinates.
All these narrow dense matrices have maximum rank 16, so a bound from a long
sentence may be uninformative. Singular-value spectra are more useful than
just counting nonzero singular values. Rounded operands and accumulation can
also create small numerical residuals beyond an ideal mathematical rank.

At one scored position, softmax cross-entropy gives logit gradient proportional
to `p - one_hot(target)` (with the shared loss normalizer). Its LM-head gradient
is an outer product with the hidden vector, generally reaching every vocabulary
row, including tokens
absent from the sentence. Underflow or cancellation can yield actual zeros.
Because the head is tied, this dense contribution is added to the input
embedding's token-row-local gradient. Broad embedding changes are expected;
they are not evidence that every row stores the sentence independently.

Adam combines past gradients and divides by a coordinate-wise second-moment
scale. This nonlinear entry-wise normalization can increase matrix rank; sums
over updates can do so too. A low-rank instantaneous gradient therefore does
not imply a low-rank checkpoint difference. Removing a sentence also changes
later activations and optimizer moments. Such deltas measure training
influence, not exclusive ownership or a direct textual encoding.

## Exact-real attention products, not individual matrix ownership

For this single 16D head, let `x_i` be the row after pre-LayerNorm and use
row-vector projections `q_i = x_i W_Q + b_Q`, `k_j = x_j W_K + b_K`,
and `v_j = x_j W_V + b_V`. In exact arithmetic, define

```text
H = W_Q W_K^T / 4         a = b_Q W_K^T / 4
score(i,j) = x_i H x_j^T + a x_j^T + row_constant(i)
P(i,:) = causal_softmax(score(i,:))
B = W_V W_O              c = b_V W_O + b_O
attention_output(i) = sum_j P(i,j) x_j B + c
```

The term containing `b_K` is constant across permitted keys for a given query,
so it cancels in row softmax. The value bias collapses into `c` because a
softmax row sums to one. Thus Q/K jointly specify a bilinear compatibility rule
`H,a`, and V/output jointly specify transported content `B,c`. These products
are more identifiable objects than individual projection coordinates. For
example, `W_Q -> W_Q R`, `b_Q -> b_Q R`, `W_K -> W_K R^{-T}` preserves the
scores up to the row-constant term for any invertible `R`; a corresponding
invertible value-basis change can be canceled by the output projection.

Here `H,a,B,c` have 256+16+256+16 = 544 real coefficients, compared with 1,088
original projection/bias coefficients. LayerNorm adds another 32 either way.
This is algebraic redundancy, **not** a verified 544-float lossless encoding:
the actual kernels round weights and Q/K/V activations to BF16 between
operations. Combining matrices changes rounding, products may need more
precision, and key bias can affect rounded keys. Compare the products across
checkpoints as a diagnostic, but verify any proposed replacement through the
real kernels and full greedy completion test.

## Cheap operational tests and interpretation

1. Canonicalize BF16-consumed values; verify identical logits and all 1,024
   greedy suffix/EOS completions. Separately test the unused-position hypothesis.
2. Quantize one attention/MLP branch at a time and record encoded bits, loss,
   and wrong completions. Count scales/codebooks too. Apply successful settings
   jointly and recheck: independent branch minima need not work together.
3. Losslessly compress checkpoint bytes and bitwise XORs against initialization
   or matched sentence-ablation checkpoints. Floating-point subtraction is not
   an exact reversible delta. Report FP32 changes and BF16-effective changes.
4. Capture single-fact gradients before Adam and compare their singular-value
   spectra with optimizer updates and eventual checkpoint deltas.

Successful compression gives an upper bound on a sufficient description under
the tested conditions; a failed compressor gives no lower bound. Initialization
randomness is not learned corpus information. State explicitly whether the
initializer, unchanged layers, first-five-token prompts, tokenizer, and
corpus-derived compact-vocabulary map are shared side information. Bits of
different layers need not represent independent facts, and a routing layer can
be important without owning the content it routes.

## Primary research

- [Blier and Ollivier: description length](https://proceedings.neurips.cc/paper/7490-the-description-length-of-deep-learning-models.pdf)
  formalizes model-plus-data coding and prequential alternatives to raw weight
  counts. A checkpoint's byte length alone is not this data-compression measure.
- [Arora et al.: compression bounds](https://proceedings.mlr.press/v80/arora18b.html)
  connects explicit network compression and noise stability to generalization.
- [Dziugaite and Roy: PAC-Bayes bounds](https://arxiv.org/abs/1703.11008)
  studies stochastic weight distributions relative to a prior, not fact ownership.
- [Allen-Zhu and Li: knowledge capacity](https://arxiv.org/abs/2404.05405)
  reports roughly two knowledge bits per parameter in controlled synthetic-fact
  experiments; this is not a universal per-layer information limit.
- [Morris et al.: memorization capacity](https://arxiv.org/abs/2505.24832)
  separates dataset-specific memorization from distribution learning and finds
  roughly 3.6 bits per parameter in random-string experiments. Different tasks,
  training procedures, and information definitions preclude applying either
  empirical constant directly to this model's individual layers.
