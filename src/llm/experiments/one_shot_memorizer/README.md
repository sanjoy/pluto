# One-shot memorization and the learned encoding

Research objective: construct a memorizer from the dataset and tokenizer without
gradient descent, then establish how its representation relates to the learned
114,256-parameter GPT-2 model. Exact corpus recall by an unrelated construction
does **not** complete the second part of this objective.

Work is on `codex/one-shot-memorizer`, based on `main`. Artifacts are local-only.
All implementations here are C++; the initial construction and runtime are CPU
code. The existing tokenizer interface needs a CUDA executor for its pinned
output arrays, not for training or evaluating the constructed automaton.

## Construct and verify

```sh
bazel build -c opt //src/llm/experiments/one_shot_memorizer
tokenizer=/home/ubuntu/checkpoints/memorize_general_facts/compact_batch_32_no_clip_0/inputs/tokenizer
bazel-bin/src/llm/experiments/one_shot_memorizer/one_shot_memorizer \
  --mode=compile --tokenizer="$tokenizer" \
  --corpus=testdata/general_facts_dataset.txt \
  --output_dir=/tmp/one_shot_memorizer_new
```

The output directory must not already exist; its parent must exist. It contains
binary numeric model weights, corpus verification metrics, a context-window
conflict profile, and per-target context requirements. No trained checkpoint is
read. Verification serializes and reloads the model, then feeds its own greedy
predictions back in until EOS, starting with each fact's first five tokens.

Inference reads only compiled weights and the original GPT-2 tokenizer, not the
corpus. Original token IDs avoid needing a separately learned vocabulary map.

```sh
bazel-bin/src/llm/experiments/one_shot_memorizer/one_shot_memorizer \
  --mode=infer --tokenizer="$tokenizer" \
  --model_file=/tmp/one_shot_memorizer_new/automaton.weights \
  --prompt='The capital of France is'
```

The automaton runtime rejects unseen prefixes. Ambiguous corpus prefixes use empirical
next-token frequencies with the lowest token ID breaking ties. This is NOT a
claim of equivalence to the trained model on arbitrary prompts or probabilities.

## Construction: sparse transition matrices

Build a prefix trie of the tokenized sentences and merge states bottom-up when
their complete weighted continuation languages agree. Duplicate sentences keep
their multiplicity. Merging shared suffixes cannot splice in a new sentence.
The algorithm is deterministic and uses no fitted parameters or iterative loss
minimization.

The resulting weights have an explicit linear-algebra interpretation:

- `alpha` is a one-hot vector selecting the initial state.
- `T_t[s,u] = 1` when token `t` moves state `s` to `u`; other entries are zero.
- `beta[s]` counts sentence endings at state `s`.
- `m[s] = beta[s] + sum_t (T_t m)[s]` counts all accepted continuations. Acyclic
  state numbering makes this a single bottom-up computation.

For a consumed prefix, `h = alpha T_token0 ... T_tokenN`. The mass of the exact
sentence is `h beta`; its total continuation mass is `h m`. The next-token
probability is `h T_t m / (h m)`, and EOS probability is `h beta / (h m)`.
This explicitly identifies where every corpus symbol and continuation choice
resides in the constructed weights. It does not identify their locations in
the trained GPT-2 matrices.

The sparse format stores coordinates, integer masses, and dimensions. There is
no hidden copy of the corpus or hardcoded answer string in the runtime. The
state compaction is exact equality of weighted right languages, not a claim of
minimum real-valued weighted-automaton dimension. Spectral constructions instead
relate that dimension to Hankel rank; that is a distinct follow-up experiment.
[Spectral construction and derivation](https://borjaballe.github.io/papers/preprint-bclq13.pdf).

## Initial measured result (2026-09-23)

From the committed general-facts corpus and the tokenizer above:

| Quantity | Result |
| --- | ---: |
| Facts / exact autoregressive completions | 1,024 / 1,024 |
| Input tokens / supervised targets including EOS | 14,098 / 10,002 |
| Active vocabulary including EOS | 4,475 |
| Trie states / compacted states | 13,345 / 10,864 |
| Sparse transition nonzeros | 11,886 |
| Serialized automaton bytes | 403,421 |
| Construction time, excluding tokenization/verification | 0.0026 seconds |

These are a constructive baseline, not evidence that GPT-2 runs this automaton.
The sparse representation is much larger than the raw tokenized corpus; its
storage accounting includes indices and counts, unlike a floating-weight count.

## How much history must any suffix-only model see?

For each window size, group **supervised** next-token queries by their final
`min(window, prefix_length)` tokens. A short key retains the left boundary.
The irreducible error count is the sum of `count - largest_target_count` over
these groups. This is exact for deterministic suffix-window classifiers on this
evaluation set, not a statement about which history GPT-2 actually uses.

| Window | Unavoidable next-token errors out of 10,002 |
| --- | ---: |
| 1 | 4,804 |
| 2 | 1,273 |
| 3 | 208 |
| 4 | 62 |
| 5 | 20 |
| 6 | 3 |
| 7 | 2 |
| 8 | 1 |
| 9 | 0 |

Thus nine-token contexts suffice for an exact construction; there are 10,001
distinct such contexts. This identifies a useful experiment: compare the
trained model against this mechanism under earlier-prefix interventions while
holding token positions fixed. Out-of-distribution corruption must not be
mistaken for proof of a particular internal algorithm.

## What remains to explain

1. A compact neural-weight construction. The exact ReLU memory below is a
   verified control, but is not as compact as GPT-2 and uses a different
   architecture.
2. A causal bridge between the constructed representation and the actual
   checkpoint: layerwise recoverability, substitutions, and selective edits.
3. How to construct the learned nonlinear features. Closed-form refits of all
   eight MLP output projections now preserve exact corpus generation, but
   removing those learned features with affine replacements does not (below).
4. Why optimization finds this representation. Fixed-feature least-squares
   gradient descent and a pseudoinverse have a known connection, but it does not
   automatically describe joint BF16 GPT-2 training with AdamW/cross-entropy.

Potential constructions include closed-form associative memories. Recent work
constructs fact-storing MLPs, but its transformer experiments still include
learned components and do not furnish an end-to-end GPT-2 compiler.
[MLPs are Hebbians](https://arxiv.org/abs/2607.10034),
[associative-memory transformer constructions](https://arxiv.org/abs/2412.06538).

## Explicit one-shot ReLU weights

Compile mode also builds and verifies a sparse two-hidden-layer ReLU network.
Its query `q` is the nine most recent token IDs, padded on the left with -1 for
short prefixes. For every distinct supervised query `k_i`, construct:

```
a_ij = ReLU(q_j - k_ij)
b_ij = ReLU(k_ij - q_j)
h_i  = ReLU(1 - sum_j(a_ij + b_ij))
z    = sum_i h_i * code(next_token_i)
```

Here `code(t)` is a fixed 16-dimensional +/-1 binary code for token ID `t`.
The first-layer sparse weights are +1/-1; its biases are the actual key token
IDs with opposite signs. The second layer has -1 weights and +1 biases. Output
weight columns contain the actual target codes. All stored values are FP32.

Distinct integer keys are separated by L1 distance at least one. Therefore
exactly the matching unit has activation one and every other unit is zero.
The resulting output is precisely its target code. Its dot product with the
matching vocabulary code is 16; every other code scores at most 14. This is a
finite-precision-safe, explicit weight construction, not an optimization or a
lookup call disguised inside the neural forward. Inference evaluates every
unit's ReLU arithmetic. The exact-code decoder extracts bits as an optimization
of the unique dot-product argmax; tests compare it against exhaustive decoding.

The model rejects queries with no active unit. Unlike the automaton, it can
accept new full prefixes whose final nine tokens match a compiled context.
It does not claim calibrated probabilities or equivalence off the corpus.

The real corpus run verified 1,024/1,024 autoregressive suffix-plus-EOS
completions after serializing and reloading these weights. Construction took
0.0027 seconds, excluding tokenization, context analysis, and verification.
There are 10,001 second-layer units and 340,034 stored corpus-dependent floats
(1,360,182 artifact bytes including metadata). This is about 3x the original
model's **trainable** scalar count. It additionally has 360,036 fixed sparse
weights, 10,001 fixed biases, and a generated 804,112-entry decoder codebook;
these are not stored and do not encode corpus facts. This is not a
parameter-efficiency result or a same-architecture GPT-2 checkpoint.

```sh
bazel-bin/src/llm/experiments/one_shot_memorizer/one_shot_memorizer \
  --mode=infer --backend=relu --tokenizer="$tokenizer" \
  --model_file=/tmp/one_shot_memorizer_new/relu.weights \
  --prompt='The capital of France is'
```

An elementary optimization connection holds **for this control**, not yet for
GPT-2. Freeze these indicator features and fit the output vectors with squared
error. With one equally weighted example per distinct key, the feature matrix
is the identity. The direct solution is exactly the target-code matrix `C`.
For objective `0.5 * ||W-C||^2`, full-batch gradient descent from zero follows
`W_s = (1-(1-eta)^s) C` for `0 < eta < 2`, converging to the same constructed
weights. Learning the features jointly, using cross-entropy, and using AdamW
change the problem; this formula must not be presented as their explanation.

## Does the actual checkpoint use only nine tokens?

`checkpoint_context_probe` holds each prediction's absolute position fixed and
replaces earlier tokens, independently for every scored target. Two corruptions
are tested: EOS filling, and tokens from the next corpus sentence, cycling if
needed. The unmodified control must get every target right. All transfers use
pinned host buffers, and all predictions use the existing cuTile top-1 kernel.

```sh
bazel build -c opt //src/llm/experiments/one_shot_memorizer:checkpoint_context_probe
facts_run=/home/ubuntu/checkpoints/memorize_general_facts/compact_batch_32_no_clip_0
bazel-bin/src/llm/experiments/one_shot_memorizer/checkpoint_context_probe \
  --checkpoint="$facts_run/layers_8/step_16128" \
  --tokenizer="$facts_run/inputs/tokenizer" \
  --output_dir=/tmp/one_shot_context_probe_new
```

Measured errors on 2026-09-23, out of 10,002 independent next-token probes:

| Retained suffix | Prefixes eligible for replacement | EOS-fill errors | Other-sentence errors |
| --- | ---: | ---: | ---: |
| Entire prefix (control) | 0 | 0 | 0 |
| 1 token | 10,002 | 8,902 | 7,840 |
| 3 tokens | 10,002 | 8,512 | 7,547 |
| 5 tokens | 8,978 | 6,894 | 6,116 |
| 9 tokens | 4,903 | 2,638 | 2,381 |
| 12 tokens | 2,177 | 744 | 693 |
| 16 tokens | 345 | 58 | 59 |
| 24 tokens | 3 | 0 | 0 |

The single entire-prefix control is shared by both replacement comparisons.
Eligible positions are overwritten; donor tokens can coincide with originals.
There were no nonfinite logit rows. These are not autoregressive completion
failure counts: each token uses its own separately modified prefix.

**What this establishes:** the checkpoint is not invariant to history preceding
the sufficient nine-token suffix, whereas the constructed suffix model is.
**What it does not establish:** a nine-token model could not fit the corpus
(it does), which layer stores a fact, or that the extra history is indispensable
on natural inputs. Both replacement schemes create unnatural histories. The
next mechanistic step is layerwise intervention to explain this discrepancy,
not a claim that the construction already explains the learned weights.

## Testing how the learned output code is organized

`checkpoint_readout_probe` captures the final, direct `gpt2/LayerNormLayer`
output on all supervised rows. It distinguishes that layer from the two norms
inside each block using the combinator stack. The 16-dimensional features are
physically BF16; the tied output weights are FP32 but are rounded to BF16 by
the head. Before interpreting any replacement, an independent CPU projection
using those effective weights must reproduce **every** original GPU top-1 ID.
No embedding or activation is modified. In particular, replacing the tied
embedding would also change the input features and invalidate this experiment.

For each target token `c`, let `mu_c` be its mean captured feature. The tool
constructs these decoders directly, without gradient updates:

* Mean-dot: `score_c(h) = h dot mu_c`.
* Nearest-centroid: `score_c(h) = h dot mu_c - ||mu_c||^2/2`.
* Regularized LDA: `score_c(h) = h dot inv(Sigma) mu_c -
  mu_c^T inv(Sigma) mu_c / 2`, where `Sigma` is the pooled within-class
  covariance, divided by the number of examples, plus `ridge * I`.

These are uniform-prior decoders. Normalized variants normalize each input
before fitting and decoding, not the resulting class prototype. Only classes
observed as targets compete; original-head controls use both the full
vocabulary and exactly that restricted set. The LDA bias terms are additional
degrees of freedom relative to the original bias-free head.

```sh
bazel build -c opt //src/llm/experiments/one_shot_memorizer:checkpoint_readout_probe
facts_run=/home/ubuntu/checkpoints/memorize_general_facts/compact_batch_32_no_clip_0
bazel-bin/src/llm/experiments/one_shot_memorizer/checkpoint_readout_probe \
  --checkpoint="$facts_run/layers_8/step_16128" \
  --tokenizer="$facts_run/inputs/tokenizer" \
  --output_dir=/tmp/one_shot_readout_probe_new
```

Measured on 2026-09-23: 10,002 target rows, 2,900 distinct target tokens,
1,844 singleton target tokens, and 1,575 compact vocabulary tokens never used
as supervised targets (they can still occur in the supplied prompts).

| Readout | Correct target rows / 10,002 | All-correct sentences / 1,024 | Leave-one-row-out correct / 8,158 repeated-class targets |
| --- | ---: | ---: | ---: |
| Original tied head, full or restricted vocabulary | 10,002 | 1,024 | Not applicable |
| Class-mean dot product | 6,965 | 33 | 4,169 |
| Class-mean dot product, normalized inputs | 7,274 | 57 | 4,422 |
| Nearest centroid | 9,472 | 599 | 6,675 |
| Nearest centroid, normalized inputs | 9,442 | 581 | 6,653 |
| LDA, ridge 0.1 | 9,627 | 711 | Not measured |
| LDA, ridge 0.01 | 9,628 | 712 | Not measured |
| LDA, ridge 0.001 | 9,634 | 716 | 6,786 |

All-correct sentences here means every **teacher-forced** suffix/EOS prediction
is correct under a single fixed decoder. It is not a separate execution of
greedy generation. In deterministic causal arithmetic these imply each other
by induction; this experiment does not additionally verify changes in floating
point behavior across generation batch shapes. Leave-one-out predictions use
a different decoder per query, so have no analogous sentence-generation score.

Singleton self-inclusion is an important trap: a singleton is its own nearest
centroid. Shuffling labels **between examples**, preserving class counts,
still produces 1,844/1,844 correct singleton predictions; it gets only
271/8,158 repeated-class predictions right. Leave-one-row-out refits remove
each query from both its prototype and pooled covariance. Singleton targets
then have no supported class and all fail; they are excluded from the repeated
target denominator above, not silently treated as generalization successes.
The backbone itself remains trained on every example, even in these controls.

The stronger leave-one-sentence-out control also removes the other rows of the
query's sentence. Nearest-centroid gets **6,695/8,126 supported targets** right;
LDA (ridge 0.001) gets **6,794/8,126**. The remaining 1,876 targets have no
training occurrence of their class outside that sentence and count as wrong
in the raw 10,002-target accuracy. This includes 32 occurrences of repeated
tokens whose repetitions are confined to one sentence. Changing the held-out
set also changes competing prototypes, so accuracy need not monotonically
decrease from row-held-out to sentence-held-out decoding.

The mean output-weight/class-mean **centered row cosine is 0.920062**. Removing
one common row vector is harmless to class preferences; fitting one positive
global scale still leaves **40.7462% relative Frobenius error** in the output
matrix. Thus strong directional alignment is not an exact weight formula.
LDA with ridge 0.001 improves decisions yet worsens weight alignment (cosine
0.373656): a good replacement decoder need not recover the original encoding.
This matrix comparison does not account for LDA's bias cancellation or
directions on which the captured activations barely vary; it is not a measure
of functional distance between the two classifiers.

This hypothesis is motivated by the class-mean/classifier relationship in
[neural-collapse research](https://arxiv.org/abs/2008.08186), not a claim that
the full neural-collapse result applies here. Thousands of classes in 16
dimensions cannot form the usual class-count-minus-one-dimensional simplex;
the data are also imbalanced and the input/output weights are tied.

### What gradient descent contributes to this interpretation

For a frozen feature matrix `H`, one-hot labels `Y`, softmax probabilities `P`,
and output rows `E`, cross-entropy gives exactly

```
dL/dE = (P - Y)^T H / n.
```

`Y^T H` is the class-frequency-weighted sum of features, not their unweighted
means. Descent pulls a correct token's row toward its contexts and pushes
competing rows away. Conversely, a feature's gradient is proportional to
`sum_c p_c E_c - E_target`, pulling its representation toward the correct
output direction. This is a concrete mechanism consistent with the measured
alignment. It is **not** a closed-form reconstruction: `P` and `H` change during
training, the tied embedding also gets gradients through the input path, and
AdamW adds coordinate-wise preconditioning and weight decay. Nor does moment
decoder failure refute linear separability: the original head already
separates every recorded example correctly.

An independent reconstructed head also does not compress the model. The
original embedding must remain to compute the learned features; the untied
replacement adds its own weights, and possibly biases.

## Tracing earlier-context influence through the learned layers

`checkpoint_causal_trace` retains nine prefix tokens and corrupts earlier
positions with the next corpus sentence, as above. For every post-attention
and post-MLP residual boundary, it independently performs two interventions:

1. Copy the clean **final query-position row** into a corrupt forward.
2. Copy the corrupt final query-position row into a clean forward.

Each patch allocates a new activation and changes only that row; all earlier
positions retain their recipient-pass values. Copying the entire activation
would trivially transfer the full remaining computation and is deliberately
not used. Clean and corrupt captures are shared, but patches are never
accumulated between sites. A successful last-layer rescue is a positive
control, not a storage-location finding: after that boundary only tokenwise
normalization and readout remain.

```sh
bazel build -c opt //src/llm/experiments/one_shot_memorizer:checkpoint_causal_trace
facts_run=/home/ubuntu/checkpoints/memorize_general_facts/compact_batch_32_no_clip_0
bazel-bin/src/llm/experiments/one_shot_memorizer/checkpoint_causal_trace \
  --checkpoint="$facts_run/layers_8/step_16128" \
  --tokenizer="$facts_run/inputs/tokenizer" \
  --retained_tokens=9 --output_dir=/tmp/one_shot_causal_trace_new
```

The TSV reports rescue among corrupt-wrong cases, newly broken predictions
among corrupt-correct cases, and damage among clean-correct cases. These are
independent next-token interventions, not autoregressive scores or exclusive
fact ownership. The distinction between activation mediation and editable
weight location is also important in
[causal-tracing research](https://arxiv.org/abs/2202.05262) and
[its limitations](https://arxiv.org/abs/2301.04213).

On the same checkpoint, all 10,002 clean targets were correct. Earlier-prefix
replacement actually changed 4,871 prefixes (out of 4,903 eligible ones) and
made 2,381 predictions wrong. No baseline or patched run produced nonfinite
logits. The complete trace took 483 seconds on the local GH200; other tests
overlapped part of this run, so that time is not an isolated benchmark.

| Block (zero-based) | Corrupt-wrong predictions rescued / 2,381 | Corrupt-correct predictions newly broken / 7,621 | Clean predictions damaged / 10,002 |
| --- | ---: | ---: | ---: |
| 0 | 82 | 73 | 48 |
| 1 | 297 | 178 | 763 |
| 2 | 412 | 220 | 1,096 |
| 3 | 543 | 161 | 1,350 |
| 4 | 839 | 117 | 1,835 |
| 5 | 1,187 | 102 | 2,090 |
| 6 | 1,496 | 40 | 2,168 |
| 7 | 2,381 | 0 | 2,381 |

The post-attention and post-MLP counts are identical within **every** block.
This is structurally expected, not evidence that MLPs do nothing: the remaining
MLP is a deterministic pointwise function. Patching its input query row with
the clean value and then running it gives the same query row as patching its
output with the clean output. Other rows undergo the same recipient-side MLP
in both cases. These interventions commute with the tokenwise MLP, so this
particular experiment cannot separate that MLP's contribution. It can measure
what subsequent cross-position attention still requires from earlier rows.

The increasing rescue shows that, under this corruption, later current-token
states can replace more of the information otherwise lost from earlier
positions. Early query-state restoration alone is insufficient while later
attention continues reading corrupted previous-position states. The final
block's 100% rescue is the expected tokenwise-readout control. None of these
counts assigns facts exclusively to a block or reconstructs the backbone's
weights from the corpus.

## Closed-form reconstruction of the MLP output maps

`checkpoint_mlp_probe` captures three matrices per block: its pre-MLP LayerNorm
output `X` (16 columns), GELU features `Z` (64 columns), and branch update `U`
(16 columns, before residual addition). Capture includes **every real token**,
including the supplied prompt, and excludes padding. Replacing a branch at all
positions requires its prompt-position behavior too.

Two fits distinguish reconstructing known features' output weights from
eliminating the learned nonlinear feature map:

- **Learned-feature refit:** solve `Z W + b ~= U`, retaining LayerNorm, the
  learned 16-to-64 expansion, and GELU.
- **Affine replacement:** solve `X A + c ~= U`, replacing the entire MLP after
  its existing LayerNorm with a single 16-to-16 affine map.

These are direct CPU solves, not gradient descent. Column-pivoted Householder
QR avoids forming the squared-condition-number normal equations. Optional
ridge minimizes `||X W + b - U||_F^2 / n + ridge * ||W||_F^2`, leaving the bias
unpenalized. A numerical rank failure is reported, not silently repaired.

The default split fits on 819 sentences and holds out every fifth sentence
(205) from fitting. This gives 11,285 fitting and 2,813 held-out activation rows.
The **backbone was trained on all 1,024 sentences**: this split tests whether
the fitted map transfers to other recorded contexts, not factual generalization
to unseen training examples. The targets are teacher hidden updates, not token
labels or independently constructed representations.

Each fitted projection is installed through a hook in a fresh model forward.
It consumes that forward's actual features, including changes caused by earlier
replacements. Neither fitting-row activations nor gold output vectors are
replayed. Single-block and all-eight simultaneous interventions retain the
residual connections and other learned layers. Original branches still execute
before their outputs are replaced; these runs measure behavior, not speed.

```sh
bazel build -c opt //src/llm/experiments/one_shot_memorizer:checkpoint_mlp_probe
facts_run=/home/ubuntu/checkpoints/memorize_general_facts/compact_batch_32_no_clip_0
bazel-bin/src/llm/experiments/one_shot_memorizer/checkpoint_mlp_probe \
  --checkpoint="$facts_run/layers_8/step_16128" \
  --tokenizer="$facts_run/inputs/tokenizer" \
  --output_dir=/tmp/one_shot_mlp_probe_new
```

The output directory contains:

- `mlp_replacements.tsv`: teacher-forced target and exact-sentence counts,
  separated into fitting/held-out sentences.
- `mlp_greedy.tsv`: independent generated-prefix verification for every joint
  intervention. It supplies only five prompt tokens, feeds back predictions,
  and stops each case at its first mismatch or correctly timed EOS. Therefore
  its generated-target count shrinks for failing models. Fitting/held-out
  sentence counts are separate, and every sentence's exactness must agree with
  its teacher-forced result, not merely the aggregate total.
- `mlp_fits.tsv`: fit error, effective BF16 matrix error against the original
  output projection where applicable, and QR diagonal spread. The latter is
  a conditioning diagnostic, **not** a spectral condition number.
- `mlp_update_errors.tsv`: fitting/held-out update errors, normalized by both raw
  and mean-centered target energy. The known original map evaluated in double
  precision establishes the discrepancy due to GPU finite-precision arithmetic.

The clone control uses the original effective BF16 matrix and **FP32 bias**;
both it and the replacement fits use the same GPU projection/hook path. Fitted
weights are rounded to BF16 operands just like the original, and outputs remain
BF16. A fit can closely approximate the original real-valued map without
reproducing every rounded activation bit.

### Measured replacements (2026-09-23)

| All eight MLPs replaced together | Correct teacher-forced targets / 10,002 | Exact generated suffixes plus EOS / 1,024 |
| --- | ---: | ---: |
| Unmodified checkpoint | 10,002 | 1,024 |
| Original output matrices/biases, cloned through the replacement path | 10,002 | 1,024 |
| Learned GELU features, closed-form output fit, ridge 0 | 10,002 | 1,024 |
| Learned GELU features, closed-form output fit, ridge 0.000001 | 10,002 | 1,024 |
| Learned GELU features, closed-form output fit, ridge 0.001 | 10,000 | 1,022 |
| Affine after LayerNorm, ridge 0.000001 | 2,616 | 0 |
| Zero branch update | 144 | 0 |
| Mean branch update from fitting rows | 137 | 0 |

The zero-ridge and small-ridge learned-feature fits preserve all 205 held-out
sentences as well as the 819 fitting sentences. Stronger ridge breaks two
fitting sentences; its held-out sentences remain exact. Every **individual**
learned-feature projection replacement is exact for all three tested ridges.
Joint changes can compound even when every isolated intervention is safe.

On held-out activation rows, the unregularized feature refits have **0.160% to
0.167% relative update error**, almost the same as evaluating the known original
map in double precision against recorded BF16 outputs (0.160% to 0.167%).
Normalizing by mean-centered rather than raw target energy gives 0.169% to
0.181%. The affine replacements instead have **23.4% to 41.0%** raw relative
error, or 24.2% to 44.9% against centered energy. Their failure is not an artifact
of a large constant mean making the fit error look small.

Matching behavior is different from identifying original matrix coordinates.
For example, block 3's unregularized refit differs from its original effective
matrix by **109% relative Frobenius norm**, despite preserving exact completion
when installed with all seven other refits. Its pivoted-QR diagonal spread is
about 22,983, versus about 20 for block 7. This warns that coefficient recovery
is sensitive to weak feature directions and quantized targets; diagonal spread
alone neither locates that error nor proves an exact nullspace. Small ridge
reduces block 3's matrix discrepancy to 2.61%, still preserving all completions.
Neither fit is asserted to reproduce arbitrary-prompt behavior.

For comparison, replacing only one MLP with an affine map after LayerNorm
(ridge 0.000001) gives:

| Block | Correct targets / 10,002 | All-correct teacher-forced sentences / 1,024 |
| --- | ---: | ---: |
| 0 | 5,533 | 20 |
| 1 | 7,329 | 98 |
| 2 | 9,483 | 632 |
| 3 | 9,870 | 905 |
| 4 | 9,284 | 554 |
| 5 | 9,367 | 578 |
| 6 | 9,151 | 476 |
| 7 | 8,001 | 153 |

This rejects this particular affine compression, not every possible affine
replacement chosen by some other objective. It supports the importance of the
learned nonlinear features for retaining this checkpoint's behavior. It does
not assign an exclusive set of facts to a block.

### A precise associative-memory interpretation, with a limitation

Center the observed feature and update matrices to obtain `Zc` and `Uc`. In
exact arithmetic, if `Zc` has full column rank and `Uc = Zc W`, then

```
W = (Zc^T Zc)^-1 Zc^T Uc
u(z) = mean(U) + sum_i k(z, z_i) * (u_i - mean(U))
k(z, z_i) = (z - mean(Z)) (Zc^T Zc)^-1 (z_i - mean(Z))^T.
```

Thus an MLP's output projection can be written as a covariance-corrected sum
of stored example updates: similar **learned feature vectors** retrieve related
updates. The coefficients can be negative; they are not attention probabilities
or exclusive fact-ownership scores. With rank deficiency the identity is only
identified on the observed span; ridge changes the operator. Quantized outputs
make the real experiment approximate rather than an exact coefficient identity.
This is the same fixed-feature bridge formalized in
[MLPs are Hebbians](https://arxiv.org/html/2607.10034v1#S3.SS1).

The crucial limitation is that `Z` and `U` already came from the trained model.
This is **conditional system identification**, not a dataset-only construction
of the GPT-2 backbone. In particular it does not explain the learned expansion
weights, attention, or token/position embeddings. The earlier automaton/ReLU
construction satisfies dataset-only exact recall, while this experiment explains
a component of the actual checkpoint; those remain distinct results. It replaces
8,320 of the model's 114,256 coefficients, with no parameter-count reduction.
The algebraic identity applies regardless of how the original map was trained;
it is not by itself an account of why AdamW discovers useful features.

## Tests

```sh
bazel test -c opt //src/llm/experiments/one_shot_memorizer/... --test_output=errors
```

Tests check duplicate weighting, ambiguity, EOS, exact compaction without new
sentences, corpus-order invariance, very long inputs, exhaustive small weighted
corpora, context conflicts, serialized round trips, malformed weights, and
truncation at every byte. The corpus-wide verification is a separate real-data
run above, not inferred from small unit tests.

The new probes additionally test BF16 head reproduction, scope selection,
masked/padded rows, malformed shapes, immutable query-only patching, both
intervention directions, final partial batches, scoped/ambiguous sites,
nonfinite results, covariance regularization, unseen target classes, and both
row/group holdouts against literal refits. MLP tests additionally check prompt
rows, padded/partial batches, original-projection cloning, FP32 bias handling,
fresh upstream features in simultaneous replacements, and greedy input histories
with no gold-suffix leakage. QR tests cover pivoting, ridge scaling, bias,
rank deficiency, ill-conditioning, and recovery of known 64-to-16 maps.
All 79 repository test targets passed with fresh execution after these additions.
