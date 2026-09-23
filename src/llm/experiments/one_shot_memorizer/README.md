# One-shot memorization and the learned encoding

## Time-bounded sentence and capacity experiments

The current research checkpoint is due **2026-09-23 15:52 UTC**. The approved
first stage is a 512-update duplicate baseline and three single-sentence
deletions, plus a separate single-fact training run. Positive and negative
results count; timeouts and unchanged accuracy are not evidence of impossible
construction or exclusive fact ownership. Reserve the final hour for checking
the evidence and writing conclusions rather than starting new sweeps.

`checkpoint_sentence_ablation` loads the same original `step_0` weights and
frozen full-corpus compact vocabulary for every condition. Adam starts with
zero moments. The full-corpus shuffle, batch slots, original cross-entropy
normalizer, and optimizer clock are preserved. A deleted sentence's logits
gradient is zeroed **after** loss backward, before model backward. All other
samples keep their original scale. This is fixed-schedule objective deletion,
not rebatching a shorter corpus. The pilot's update budget does not change the
original 40,000-update learning-rate schedule. There is no gradient clipping.

Every baseline-repeat update must match the baseline byte-for-byte. Deletion
runs must also match before their first affected batch. Reports contain all
changed physical parameter coordinates, their tensor names/shapes, token-row
IDs where applicable, and Q/K/V partitions. The main reports use
`full-corpus baseline - intervention`; `from_initial/` reports use
`initial - trained`, the negative of the usual learned update. Reports include
FP32 master differences, not just differences in effective BF16 operands.

The single-fact condition uses batch size one and repeats the selected fact.
It retains the same initialization and all vocabulary rows. Its number of
exposures and loss normalizer intentionally differ from the deletion runs:
512 updates provide 512 examples of that fact, versus 16 scheduled occurrences
in a 1,024-sentence, batch-32, 512-update baseline. Scheduled occurrences in a
deletion run contribute zero gradient. All final models are evaluated on the
original full corpus, both teacher-forced and by actual greedy suffix/EOS
completion. This short pilot need not itself memorize the full corpus.

```sh
bazel build -c opt //src/llm/experiments/one_shot_memorizer:checkpoint_sentence_ablation
bazel-bin/src/llm/experiments/one_shot_memorizer/checkpoint_sentence_ablation \
  --initial_checkpoint=/home/ubuntu/checkpoints/memorize_general_facts/compact_batch_32_no_clip_0/layers_8/step_0 \
  --tokenizer=/home/ubuntu/checkpoints/memorize_general_facts/compact_batch_32_no_clip_0/inputs/tokenizer \
  --output_dir=/tmp/one_shot_sentence_pilot_new \
  --steps=512 --omitted_lines=1,258,631 --single_fact_line=631
```

Generated checkpoints, coordinate maps, and HTML stay local, outside Git.

The runner also supports `--isolated_lines=80,406 --single_fact_line=0` for
one jointly trained, batch-one subset. Selection preserves original corpus
line identity and uses the same frozen vocabulary and initialization. Saved
step-0 checkpoints, per-update source-line logs and checkpoint exposure counts
make comparisons auditable. The full-corpus baseline remains a separate
condition. See [SINGLE_FACT_DECODING.md](SINGLE_FACT_DECODING.md) for the frozen
repetition/overlap/two-fact stress tests, their successes and failures, and the
coordinate-convention limit of the weight readout.

### Operational layer-capacity probe

`checkpoint_capacity_probe` quantizes one attention or MLP branch at a time,
including its pre-LayerNorm and biases, and counts exact full-corpus
completions. Each case restores the original model first. The defaults test
8, 4, and 2 bits per coefficient with one FP64 scale per tensor. Reports count
scale overhead but are **not** an entropy estimate, a packed checkpoint, or a
lower bound on necessary bits. Successful independent interventions are not
automatically jointly safe. The unchanged backbone, prompts, tokenizer, and
compact mapping are side information.

There are two controls: the original checkpoint must recall every sentence;
BF16-rounding only already-BF16-consumed token/dense matrices should preserve
completion accuracy. Finally, all original weights are restored, byte-checked,
and re-evaluated. See [CAPACITY_NOTES.md](CAPACITY_NOTES.md) for storage counts,
gradient mechanisms, and primary literature.

```sh
bazel build -c opt //src/llm/experiments/one_shot_memorizer:checkpoint_capacity_probe
bazel-bin/src/llm/experiments/one_shot_memorizer/checkpoint_capacity_probe \
  --checkpoint=/home/ubuntu/checkpoints/memorize_general_facts/compact_batch_32_no_clip_0/layers_8/step_16128 \
  --tokenizer=/home/ubuntu/checkpoints/memorize_general_facts/compact_batch_32_no_clip_0/inputs/tokenizer \
  --output_dir=/tmp/one_shot_capacity_new --bits=8,4,2
```

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

## Reconstructing the final projection from token labels

`checkpoint_label_probe` removes a key dependency of the previous experiment:
its regression targets come from **corpus next-token labels**, not the original
MLP's output vectors. It still retains the learned backbone through the final
GELU features, so it is not yet a full dataset-only reconstruction.

For each supervised position, capture its incoming final-MLP residual `r_i`,
learned 64-dimensional GELU feature `phi_i`, and true next-token ID `y_i`.
Given an output-only code vector `c_y`, solve in one pivoted-QR fit:

```
phi_i W + b ~= alpha * c_yi - center(r_i)
```

The target is a desired **post-residual** code, after removing the per-row mean
that final LayerNorm discards. Final normalization uses gamma=1 and beta=0;
the independent output head's rows are the code vectors. The original tied
embedding is left untouched for the input path. Fitted weights, residual adds,
normalization and head all run through the real BF16 GPU implementation. The
original final MLP/norm/head still execute before their outputs are replaced,
so this experiment is not a performance benchmark or parameter compression.

Three codebooks distinguish geometric alignment from merely having enough
distinct names for tokens:

1. **Fixed balanced codes:** 16-dimensional vectors with eight +1 and eight -1
   entries. There are 12,870 distinct codes, all centered and of squared norm
   16, enough for the 4,475-token compact vocabulary. A fixed seeded shuffle
   assigns them to token IDs. They are distinguishable, not orthogonal.
2. **Normalized learned codes:** center each effective learned embedding row
   across coordinates and divide by its RMS. These codes depend on the
   checkpoint, even though the regression targets use corpus token labels.
3. **Permuted learned codes:** shuffle those same rows among token IDs. This
   preserves their geometry and precision while breaking their learned
   assignment to token labels. The frozen input embedding is not shuffled.

Before fitting, an oracle passes all ideal code vectors through the replacement
normalization and head. All **4,475/4,475** decode correctly for every codebook
in the runs below. This tests code distinguishability under BF16 arithmetic,
not robustness to imperfectly fitted code vectors. A common-mode residual can
also lose a small code signal during BF16 addition before normalization removes
its mean; real GPU prediction tests, not just centering algebra, are necessary.

```sh
bazel build -c opt //src/llm/experiments/one_shot_memorizer:checkpoint_label_probe
facts_run=/home/ubuntu/checkpoints/memorize_general_facts/compact_batch_32_no_clip_0
bazel-bin/src/llm/experiments/one_shot_memorizer/checkpoint_label_probe \
  --checkpoint="$facts_run/layers_8/step_16128" \
  --tokenizer="$facts_run/inputs/tokenizer" \
  --output_dir=/tmp/one_shot_label_probe_new
```

Default fitting uses 819 sentences/8,009 supervised rows; 205 sentences/1,993
rows are excluded from fitting. Of those held rows, 377 target tokens have no
occurrence as a target in the fitting subset. Every decoder nevertheless
includes all 4,475 compact tokens, assigned before the split. The backbone
itself was trained on every sentence. `--fit_sentence_stride=0` instead fits
all 1,024 sentences. This is the relevant option when testing corpus-exact
construction rather than conditional held-context transfer.

`label_projection.tsv` reports teacher-forced accuracy and separately executed
greedy exact completions, split into fitting/held-out sentences. It checks
agreement of exactness for each sentence, not only aggregate counts. It also
reports code-vector error/cosine before GPU rounding: a small branch-update
error alone could hide poor code recovery if cancelling the incoming residual
dominates the regression target. The actual GPU decisions remain authoritative.

### Measured label-only construction (2026-09-23)

With `code_seed=0`, `code_scale=0.25`, and `ridge=0.000001`:

| Condition | Correct targets / 10,002, fit on 819 sentences | Exact greedy completions / 1,024 | Correct targets / 10,002, fit on all sentences | Exact greedy completions / 1,024 |
| --- | ---: | ---: | ---: | ---: |
| Original checkpoint | 10,002 | 1,024 | 10,002 | 1,024 |
| Original final norm/head cloned through replacement hooks | 10,002 | 1,024 | 10,002 | 1,024 |
| Only set original final LayerNorm beta to zero | 9,978 | 1,003 | 9,978 | 1,003 |
| Fixed balanced codes, label-fitted projection | 2,189 | 1 | 2,201 | 1 |
| Normalized learned codes, label-fitted projection | 9,006 | 380 | 9,046 | 397 |
| Permuted learned codes, label-fitted projection | 1,484 | 1 | 1,495 | 1 |

The held-sentence portion of the learned-code fit is **1,775/1,993** correct
targets and **65/205** exact completions. Neither fixed nor permuted codes
produce an exact held-out completion. Fitting every sentence does not restore
perfect recall, so exclusion of the held sentences is not the sole obstacle.

Two additional controls help interpret the learned-code result. Changing only
the final normalization/head while retaining the original final MLP gives
**8,530 targets and 209 exact completions**. Removing that MLP's update, leaving
only its incoming residual, gives **2,326 targets and 2 completions**. The
label-fitted map therefore does useful work beyond copying an already-correct
residual into a new decoder, but does not match the original checkpoint.

The learned-versus-permuted contrast supports **co-adaptation between the fixed
backbone's features and its token-code assignment**. It does not mean arbitrary
codes cannot work with a different backbone, a larger feature space, or another
objective. Euclidean regression asks for a particular output vector; correct
classification requires only that the target win. Failure of this regression
is not proof that no output matrix can classify with the same codebook.

Varying the target-code amplitude with all sentences included in fitting also
fails to recover perfect completion. This is a bounded three-point check, not
an exhaustive hyperparameter search:

| Code amplitude | Fixed balanced: correct targets / exact completions | Learned: correct targets / exact completions | Permuted learned: correct targets / exact completions |
| --- | ---: | ---: | ---: |
| 0.0625 | 476 / 0 | 3,537 / 1 | 570 / 0 |
| 0.25 | 2,201 / 1 | 9,046 / 397 | 1,495 / 1 |
| 1.0 | 2,671 / 1 | 8,795 / 332 | 2,105 / 1 |

All three codebooks pass the 4,475-token ideal-code oracle at every amplitude.
The learned-code fit at amplitude 0.0625 has a smaller branch-update relative
error (0.288) than at amplitude 1 (0.607), but a much larger code-vector error
(1.390 versus 0.506). Much of the former target is residual cancellation, which
is why update error alone is misleading. Inference checks include the actual
BF16 residual addition and subsequent normalization, not only this real-valued
regression diagnostic.

### Why decision constraints are the next distinct test

For a centered codebook and final gamma=1, beta=0, normalization divides all
token scores by the same positive factor. Thus a target wins exactly when

```
(c_y - c_j) dot (r + phi W + b) > 0   for every competing token j.
```

These are linear inequalities in `W,b`. A bounded-margin linear program could
test decisions without forcing every output to equal its code vector. This
would be non-gradient optimization, not a closed-form formula, and any candidate
must still be checked against every rival and actual BF16 generation.
The distinction between prescribing vectors and imposing multiclass margins
has a standard precedent in
[Crammer and Singer's multiclass formulation](https://www.jmlr.org/papers/volume2/crammer01a/crammer01a.pdf);
the proposed fixed-code/residual constraints here are our own specialization.

This simplification must not be silently applied to the original final norm.
With its learned gamma/beta, write `P=I-11^T/16`, `a_j=P(gamma*e_j)`,
`d_j=beta dot e_j`, and `s=sqrt(||Pz||^2/16+epsilon)`. The original pairwise
condition is `(a_y-a_j) dot z + (d_y-d_j)*s > 0`, which is not generally linear
in the changed residual. The measured 24 errors from zeroing beta show that
this distinction matters here. No linear-program feasibility result is claimed
by the code-target experiment above.

## Constructing decision margins instead of target vectors

The same `checkpoint_label_probe` now accepts
`--projection_objective=token_margin --fit_sentence_stride=0`. It retains the
original effective BF16 embedding table and learned final LayerNorm gamma.
With beta temporarily zero, its score directions are
`a_j = center(gamma * embedding[j])`. A revised-simplex linear program constructs
the final 64-to-16 projection from the observed incoming residuals, GELU features,
and corpus labels:

```
maximize delta, subject to
  (a_yi - a_j) dot (r_i + phi_i W + b) >= delta   for every i and j != y_i
  -B <= every coefficient of W and b <= B
  delta <= 1.
```

Delta has no restrictive lower bound. The coefficient box is a declared search
restriction, not a claim about all possible matrices. The probe also constrains
each output coefficient vector to sum to zero, removing the common-coordinate
component that real-valued LayerNorm discards. Combined with the box, this is an
additional explicit restriction. Centering avoids gratuitous common-mode
signals that could make BF16 residual cancellation less accurate.

The LP starts with zero projection weights; it never reads the original final
projection or its output vectors. To avoid materializing roughly 44.7 million
pairwise constraints, it adds the worst currently violated rival per row, up to
a configurable number of new cuts, and solves again. Every proposed matrix is
independently evaluated against **all 4,475 classes at all 10,002 targets**.
The solver's current subset objective is not itself a success condition.
Acceptance requires a strictly positive measured margin everywhere; actual
BF16 teacher-forced and greedy generation are then separate checks.

The solver is [lp_solve](https://github.com/lp-solve/lp_solve), compiled from
pinned C sources without C++ exceptions. Simplex is an iterative numerical
optimization algorithm, not gradient descent, and not a closed-form formula.
Timeouts and round/cut limits are inconclusive. A nonpositive numerical optimum
for a restricted LP bounds the full problem only within the declared coefficient
box and centering restrictions; it is not a general impossibility result.

```sh
bazel build -c opt //src/llm/experiments/one_shot_memorizer:checkpoint_label_probe
facts_run=/home/ubuntu/checkpoints/memorize_general_facts/compact_batch_32_no_clip_0
bazel-bin/src/llm/experiments/one_shot_memorizer/checkpoint_label_probe \
  --checkpoint="$facts_run/layers_8/step_16128" \
  --tokenizer="$facts_run/inputs/tokenizer" \
  --projection_objective=token_margin --fit_sentence_stride=0 \
  --decision_coefficient_bound=1 --decision_rounds=30 \
  --decision_solve_seconds=60 --output_dir=/tmp/one_shot_decisions_new
```

`label_projection.tsv` records incremental solver progress and both generation
checks: first with zero beta, then with the **original beta restored**. The
latter is not implied by the linear constraints. `decision_projection.tsv`
contains the last independently checked candidate's double-precision matrix
and bias with explicit coordinates, even when the solver hits a limit; consult
the recorded outcome before interpreting a saved candidate. The original
checkpoint and input embeddings remain unchanged. As in the other hook-based
probes, original branches still execute, so timings are not inference benchmarks.

### Measured decision-constraint outcomes (2026-09-23)

Three bounded trials used all 10,002 supervised positions, all 4,475 decoder
classes, coefficient bound `B=1`, centered coefficients, margin cap 1, and
a 60-second limit per restricted-LP solve. All ended with **`solver_limit`**,
not a positive-margin certificate and not an infeasibility finding:

| Trial | Latest attempted / checked round | Correct CPU targets / 10,002 | Minimum CPU margin | GPU zero-beta: correct targets / exact completions | GPU original-beta: correct targets / exact completions |
| --- | ---: | ---: | ---: | ---: | ---: |
| Default simplex, 1,024 new cuts/round | 1 / 0 | 3,773 | -12.6381 | 3,774 / 7 | 3,926 / 6 |
| Default simplex, 128 new cuts/round | 2 / 1 | 55 | -258.219 | 55 / 0 | 41 / 0 |
| Dual simplex, 128 new cuts/round | 2 / 1 | 55 | -258.219 | 55 / 0 | 41 / 0 |

The first trial timed out before producing an accepted solver result. Its
saved candidate is therefore the **initial zero projection**, not a learned
replacement: all 1,040 saved coefficients are zero. In the other two trials,
the first restricted LP reached its margin cap of 1 on 128 constraints, but
the independent all-class check found the severe failures shown above. Both
timed out after adding another 128 constraints. They returned the same
previously checked round-one matrix, not unvalidated values from the timed-out
solve. Here "checked" means independently measured, **not** successful. The
one-token CPU/GPU difference for the zero projection also illustrates why a
real-valued score check does not replace the BF16 execution check.

These trials establish a practical limitation of this solver setup and budget:
**it has not constructed a corpus-exact final projection**. Reducing the initial
cut count allowed one restricted solve to finish; selecting dual simplex did
not improve the observed outcome. Neither the timeout nor a positive objective
on a small constraint subset answers whether a suitable full projection exists.
The 1,024-fact exact original model remains the reference, and the earlier
label-vector regression's 397 exact completions remains a separate result, not
a successful decision-constraint construction.

Local evidence is in `label_projection.tsv` and `decision_projection.tsv`
under `/tmp/one_shot_decision_probe_0`, `/tmp/one_shot_decision_probe_1`, and
`/tmp/one_shot_decision_probe_dual_0`. These are generated run artifacts, not
source files committed to Git.

## Completed 512-update sentence pilot (2026-09-23)

Local evidence is under `/tmp/one_shot_sentence_pilot_0`: `manifest.txt`,
`summary.tsv`, and each condition's `trajectory.tsv`, `predictions_512.tsv`,
`per_tensor.tsv`, `coordinates.tsv`, and `report.html`. `from_initial/` contains
the corresponding initialization-relative maps. Final checkpoints are each
condition's `step_512/`. These generated artifacts remain outside Git.

| Condition | Correct targets / 10,002 | Exact greedy facts / 1,024 | Loop seconds |
| --- | ---: | ---: | ---: |
| Full corpus | 1,776 | 1 | 80.6033 |
| Identical full-corpus repeat | 1,776 | 1 | 80.6763 |
| Omit line 1: mammals / milk / nourish | 1,855 | 1 | 80.6960 |
| Omit line 258: Rajendra Prasad / India / 1950 | 1,876 | 1 | 80.7043 |
| Omit line 631: Durian / smell / creamy flesh | 1,881 | 1 | 80.6485 |
| Train only on line 631 | 975 | 1 | 6.77946 |

Loop times include the checkpoint/comparison work during training but exclude
the subsequent full-corpus evaluation and report writing. Every one of the
512 full-corpus repeat updates was byte-identical to its baseline counterpart.
Each deletion also matched before its first affected batch. Those first updates
were 31, 21, and 7 for lines 1, 258, and 631, respectively; they immediately
changed 97,999, 97,971, and 94,447 FP32 coordinates. By update 512 each deletion
changed **98,288 of 114,256** coordinates: all non-position parameters plus
position rows 0..25. The remaining 15,968 position coefficients were unchanged.

This is an **early-training influence pilot**, not deletion from a memorized
model. The full-corpus baseline itself completes only 1/1,024 facts at this
budget. The higher target counts after deletion do not establish a meaningful
generalization improvement, nor selective forgetting. They show that these
small objective changes alter the early learning trajectory. Each sentence
has 16 scheduled occurrences in the full-corpus conditions; its deleted slots
contribute no gradient. The single-fact condition instead has 512 exposures,
batch size one, and a different loss normalizer. It is not exposure-matched.

### Where the differences concentrate

The table uses `baseline - omission` at step 512. A module's squared-norm share
is the sum of its squared scalar differences divided by the same sum globally;
it is coordinate-dependent, not a fraction of facts owned.

| Omitted line | Global delta L2 | Embedding delta L2 / share | All attention L2 / share | All MLP L2 / share |
| --- | ---: | ---: | ---: | ---: |
| 1 | 1.77114 | 1.40211 / 62.67% | 1.00817 / 32.40% | 0.38509 / 4.73% |
| 258 | 2.12928 | 1.66748 / 61.33% | 1.25352 / 34.66% | 0.41772 / 3.85% |
| 631 | 2.43002 | 1.95745 / 64.89% | 1.36746 / 31.67% | 0.43619 / 3.22% |

All 71,600 embedding coefficients, all 8,960 attention-branch parameters, and
all 17,280 MLP-branch parameters differ in every omission. The largest
attention-branch deltas occur at blocks 5 and 4 for line 1 (0.57211, 0.53115),
blocks 5 and 7 for line 258 (0.81900, 0.55942), and blocks 6 and 5 for line 631
(0.79029, 0.72708). Attention/MLP counts include their LayerNorms and biases.

The largest embedding-row differences nevertheless reveal recognizable pieces
of the omitted sentence. Leading spaces are significant tokenizer characters.

| Omitted line | Token text | Compact ID | Original GPT-2 ID | Row delta L2 |
| --- | --- | ---: | ---: | ---: |
| 1 | ` nour` | 3,862 | 31,219 | 0.503253 |
| 1 | ` young` | 774 | 1,862 | 0.138362 |
| 258 | ` 1950` | 2,499 | 11,445 | 0.507294 |
| 258 | ` became` | 1,004 | 2,627 | 0.472549 |
| 258 | `ad` | 119 | 324 | 0.352600 |
| 631 | ` creamy` | 3,735 | 27,892 | 0.507469 |
| 631 | ` strong` | 792 | 1,913 | 0.500265 |
| 631 | ` smell` | 2,147 | 8,508 | 0.480744 |

IDs come from the saved `compact_vocabulary.tsv`, joined with the original
tokenizer vocabulary. The `ad` row is consistent with the spelling of Prasad;
these row rankings alone do not recover token order or a complete sentence.
Tied softmax training updates vocabulary rows even when their tokens never
occur as inputs. Large row deltas identify useful intervention candidates, not
exclusive sentence ownership.

These are not merely differences in invisible FP32 low bits. After comparing
BF16-rounded values for matrices/token embeddings and FP32 values elsewhere,
the changed-operand counts are 93,915, 94,335, and 95,041 for the three omissions.
Only 37, 29, and 18 changed FP32 coordinates, respectively, have absolute deltas
at most 1e-6. These operand counts are not a count of changed activations or a
guarantee that every change matters to the output.

### The single-Durian control

Training only on line 631 achieves its **10/10 suffix/EOS targets and an actual
exact greedy completion**; line 631 is its only exact completion in the full
corpus audit. Relative to initialization it changes 98,096 coordinates,
including every embedding coefficient and every attention/MLP parameter.
Its changed position rows are exactly 0..13, versus 0..25 in the full-corpus
baseline. These ranges were verified from coordinate maps and checkpoint
comparisons; maximal text length still requires independent tokenizer metadata.

Embedding changes account for 99.7013% of its squared parameter delta norm.
Moreover 99.1902% of that embedding delta's squared norm is explained by a
single common 16D row shift: project the delta onto the matrix whose every row
equals the mean row. The corresponding mean-row fraction is 60.8071% for the
full-corpus baseline, but only 1.38--1.84% for the three omission deltas.
This is a concrete reason not to equate a huge dense delta with many independent
facts. A common head-row shift cancels from softmax in exact arithmetic for
fixed hidden states; tied input embeddings and BF16 rounding mean the complete
model is not invariant to that change. This observation alone does not identify
the mechanism storing the Durian completion.

See [capacity notes](CAPACITY_NOTES.md) for the outer-product gradient mechanism,
precision accounting, and the distinction between influence and information.

## Measured branch-precision tolerance (2026-09-23)

The completed `/tmp/one_shot_capacity_0/` sweep tested the original fully
memorized `step_16128`. Original, BF16-master, joint zero-key-bias, and restored
controls all retained **10,002/10,002 targets and 1,024/1,024 greedy completions**.
The BF16 control rounded all 96,176 relevant coefficients. The key-bias control
zeroed 128 values across all eight blocks; it establishes corpus-level
removability, not identical logits or arbitrary-prompt equivalence.

Every branch independently retained all completions at eight bits. None did
at four or two bits with this particular uniform quantizer:

| Block | Attention, 4 bits | MLP, 4 bits | Attention, 2 bits | MLP, 2 bits |
| --- | ---: | ---: | ---: | ---: |
| 0 | 887 | 807 | 18 | 1 |
| 1 | 997 | 987 | 6 | 1 |
| 2 | 997 | 1,020 | 15 | 7 |
| 3 | 1,022 | 1,021 | 114 | 42 |
| 4 | 931 | 1,006 | 15 | 0 |
| 5 | 993 | 1,021 | 59 | 16 |
| 6 | 1,023 | 1,015 | 163 | 7 |
| 7 | 1,011 | 975 | 43 | 5 |

Entries count exact greedy completions out of 1,024, including EOS. The result
is a **conditional robustness curve**, not bits of fact ownership. Block 0 is
especially sensitive at four bits; later blocks differ substantially despite
equal parameter counts. A successful eight-bit attention branch has a nominal
9,344-bit code (1,120 values plus six FP64 scales); an MLP has 17,664 bits.
These are sufficient *individual* replacement descriptions with all other
weights fixed. Failure of uniform four-bit quantization is not a lower bound
for other representations.

A subsequent **joint** test quantized every attention/MLP branch, including
all branch norms and biases, simultaneously (`--joint_branches --bits=8,6,4`).
Embeddings and final normalization remained original. Results in
`/tmp/one_shot_capacity_joint_0/` are:

| Joint branch precision | Correct targets / 10,002 | Exact facts / 1,024 |
| --- | ---: | ---: |
| 8 bits | 10,002 | 1,024 |
| 6 bits | 9,995 | 1,017 |
| 4 bits | 6,044 | 37 |

All 26,240 branch coefficients across 96 tensors fit a nominal 216,064-bit
description at eight bits, including 96 FP64 scales. With all other masters
still FP32, nominal model size is 379,072 bytes, excluding file/layout metadata.
This is coding accounting, not an implemented packed model or an entropy
measurement. The original bytes were restored and full exactness reverified.

## Causal single-fact embedding controls (2026-09-23)

`single_fact_probe` reloaded the independently trained Durian-only checkpoint
and its initialization. Reports are in `/tmp/one_shot_single_fact_probe_0/`.
It computes `c = mean_token(E_trained - E_initial)` and independently restores
the entire trained checkpoint before each intervention. All comparisons below
score the ten suffix/EOS targets of line 631:

| Intervention | Correct targets | Complete greedy suffix |
| --- | ---: | ---: |
| Trained baseline | 10/10 | yes |
| Initialization | 0/10 | no |
| Subtract common shift: `E := E_trained - c` | 5/10 | no |
| Same subtraction, plus `P := P_trained + c` | 10/10 | yes |
| Keep only common embedding shift: `E := E_initial + c` | 0/10 | no |
| Restore initial embedding | 0/10 | no |
| Restore initial position table | 10/10 | yes |
| Keep trained embedding, initialize everything else | 0/10 | no |
| Initialize both embedding and positions | 0/10 | no |
| Restore original trained bytes | 10/10 | yes |

The common shift accounts for **99.1902% of embedding-update squared norm**.
Removing it alone is harmful, but moving it into the position table preserves
this completion. In exact arithmetic, `E-c, P+c` preserves initial residuals
and changes all tied-head logits by one shared offset; BF16 rounding means the
actual GPU outcome still needs testing. This experiment passed the corpus case
but does not assert universal or logit-level equality.

Most embedding-update energy is therefore compatible with a change of origin,
not thousands of independently encoded pieces of this sentence. The residual
token-specific embedding changes and the learned remainder are both necessary
under the tested restoration controls: neither the common shift alone nor the
trained embedding with an initialized backbone suffices. Learned changes in
the position table are dispensable for this one fact; its initialized position
vectors can still provide a positional code.

```sh
bazel build -c opt //src/llm/experiments/one_shot_memorizer:single_fact_probe
bazel-bin/src/llm/experiments/one_shot_memorizer/single_fact_probe \
  --initial_checkpoint=/home/ubuntu/checkpoints/memorize_general_facts/compact_batch_32_no_clip_0/layers_8/step_0 \
  --trained_checkpoint=/tmp/one_shot_sentence_pilot_0/only_line_631/step_512 \
  --tokenizer=/home/ubuntu/checkpoints/memorize_general_facts/compact_batch_32_no_clip_0/inputs/tokenizer \
  --selected_line=631 --output_dir=/tmp/single_fact_controls_new
```

## Approved longer follow-up

The user approved extending the baseline and Durian-omitted pair to 16,384
updates. This run began around 07:27 UTC on 2026-09-23; artifacts are in
`/tmp/one_shot_sentence_durian_16384_0/`. It retains the 40,000-update learning
rate horizon, saves every 512 updates, and evaluates every 4,096. A second full
baseline repeat is intentionally omitted; the 512-update pilot established
byte-identical repeats, and this run still checks all initial bytes and every
update before the first omission. The in-memory matched-step baseline history
uses approximately 7.5 GB of host memory. The process has a three-hour runtime
cap, well inside the 15:52 UTC reporting deadline.

```sh
bazel-bin/src/llm/experiments/one_shot_memorizer/checkpoint_sentence_ablation \
  --initial_checkpoint=/home/ubuntu/checkpoints/memorize_general_facts/compact_batch_32_no_clip_0/layers_8/step_0 \
  --tokenizer=/home/ubuntu/checkpoints/memorize_general_facts/compact_batch_32_no_clip_0/inputs/tokenizer \
  --output_dir=/tmp/long_durian_pair_new --steps=16384 \
  --omitted_lines=631 --single_fact_line=0 --norepeat_baseline \
  --checkpoint_every=512 --evaluate_every=4096
```

The completed baseline reached 1,000 exact facts at 12,288 updates but only
**740/1,024 at 16,384** (9,632/10,002 targets). Durian itself remained exact,
10/10 targets. Saved weights at 512, 4,096, 12,288, and 15,872 were compared
byte-for-byte with the earlier pilot or original run and match. The original
metrics also show nonmonotonic late accuracy: 960 exact at 15,872, 1,022 at
16,000, and 1,024 at its stopping point 16,128. Continuing the same optimizer
past a successful checkpoint therefore cannot be assumed to preserve success.
The omission run finished at 08:53 UTC with **1,023/1,024 exact facts** and
9,993/10,002 targets. Durian is its only nonexact sentence (1/10 targets);
every other fact is completely memorized. Compared at the same 16,384 update,
739 facts are exact in both models, 284 only in the omission, and Durian only
in the baseline. Thus the omitted model successfully learns all remaining
data, but the endpoint weight difference is not deletion from a stably,
fully memorized baseline. Both runs and all original byte checks completed.

The matched 12,288-update comparison is now complete. Baseline and omission
score 9,974 and 9,973 of 10,002 targets, respectively, with 1,000 and 1,006
exact sentences. Durian falls from **10/10 and exact** to **1/10 and not
exact**. Across all facts, 984 are exact in both, 16 only in the baseline,
22 only in the omission, and two in neither. Excluding Durian, 15 facts lose
exactness and 22 gain it. Thus a strong effect on the omitted fact coexists
with collateral trajectory changes, not an exclusive storage address.
The fixed-budget 16,384 endpoint remains the prespecified outcome; choosing
12,288 for mechanistic follow-up is explicitly posthoc because both models
are nearly memorized there. The perfect historical 16,128 baseline must not
be compared against a differently timed omission as a matched deletion.

At 12,288, the largest baseline-minus-omission embedding-row L2 changes are
still ` smell` (2.6573), ` strong` (2.5971), prompt token `Dur` (2.5850),
and ` creamy` (2.4521). Other target rows range from rank 29 (EOS) to 3,918
(` its`), so this ranking does not recover the whole sentence. The full
embedding difference has L2 43.5756 and cosine only 0.05792 with its
512-update counterpart: content-specific rows remain conspicuous while the
dense difference evolves. This rejects a naive interpretation as repeatedly
adding the same fixed sentence vector. Local evidence is
`/tmp/paired_embedding_trajectory_12288.tsv` plus the paired per-sentence
`predictions_12288.tsv` files.

### Causal transplantation of the largest paired deltas

`paired_checkpoint_edit_probe` ranks embedding rows by their FP32-master L2
differences, before labeling or evaluating them. It replaces the top 1, 4,
or 16 rows, all embeddings, all transformer parameters, positions, or final
LayerNorm with the other matched checkpoint's values. Every condition starts
from the original recipient. Both directions end with byte-identical restored
weights and identical per-fact evaluation results. These are independent
inference edits, not resumed training; embedding changes affect both the
input lookup and the tied output head.

For the post-hoc 12,288-update pair, all 18 evaluations completed:

| Recipient / donor component | Exact facts | Durian targets / 10 | Durian exact? |
| --- | ---: | ---: | --- |
| Baseline, unchanged | 1,000 | 10 | yes |
| Baseline, omitted model's largest embedding row | 999 | 8 | no |
| Baseline, omitted model's top four rows | 999 | 1 | no |
| Baseline, omitted model's top 16 rows | 987 | 1 | no |
| Omitted, unchanged | 1,006 | 1 | no |
| Omitted, baseline's largest embedding row | 1,006 | 2 | no |
| Omitted, baseline's top four rows | 1,006 | 1 | no |
| Omitted, baseline's top 16 rows | 994 | 1 | no |

The largest row is compact ID 2,147 (` smell`), containing only 16 floats.
The next three are ` strong`, `Dur`, and ` creamy`. Replacing the largest row
or top four rows in the baseline changes **only Durian's exactness status**
among all 1,024 facts. This is selective causal disruption on the tested
corpus, not an exclusive ownership claim: the baseline already has 24 failed
facts, probabilities can change without changing correctness, and replacing
these rows in the opposite direction does **not** install the fact.

Larger transplants are highly incompatible: moving all token embeddings leaves
only one exact fact in either direction; moving all transformer parameters
leaves zero or one. Position-table transplants leave 114 or 87 exact facts.
Final LayerNorm transplants retain 997 or 1,007. These mixed models show
co-adaptation between components, not which component individually owns all
lost facts. Local raw results: `/tmp/one_shot_paired_edit_12288_0/`.

```sh
bazel build -c opt //src/llm/experiments/one_shot_memorizer:paired_checkpoint_edit_probe
bazel-bin/src/llm/experiments/one_shot_memorizer/paired_checkpoint_edit_probe \
  --baseline_checkpoint=/tmp/one_shot_sentence_durian_16384_0/baseline/step_12288 \
  --omitted_checkpoint=/tmp/one_shot_sentence_durian_16384_0/omit_line_631/step_12288 \
  --tokenizer=/home/ubuntu/checkpoints/memorize_general_facts/compact_batch_32_no_clip_0/inputs/tokenizer \
  --selected_line=631 --batch_size=32 --output_dir=/tmp/paired_edit_new
```

### Prospective single-fact token-set validation

An exploratory CPU analysis of the Durian-only trajectory found this simple
readout of embedding **updates**, requiring both initialization and trained
weights:

```
delta[token] = E_trained[token] - E_initial[token]
c = mean_token(delta[token])
select token iff dot(delta[token], c) < 0
```

At the saved 128-, 256-, 384-, and 512-update Durian checkpoints, the selected
IDs are exactly the ten distinct suffix/EOS target IDs, with no additional
vocabulary rows. The criterion takes no target IDs as input; reference labels
are used afterward to score the selected set. This is nevertheless an
**exploratory discovery on this sentence**, separate from the subsequent
prospective validation below.
It recovers neither token order nor repetitions, and not the five-token
prompt. The first-update result is imperfect. In particular, this is not a
decoder of an arbitrary trained model without its initialization.

Before training further single-fact models, freeze the zero-threshold rule
above and test lines **80 (France), 1 (mammals), and 258 (Rajendra Prasad)**.
Each will use the same original initialization, frozen full-corpus vocabulary,
batch-one training, optimizer, learning-rate schedule, and 512-update budget
as the Durian-only condition. Score precision/recall on **distinct** suffix
plus EOS token IDs, reporting prompt-only rows separately. Do not tune the
threshold or change the selected lines after observing these results.

These short validations ran after the approved long pair and the
five-token trace. The existing ablation runner can produce each condition
with `--batch_size=1 --omitted_lines= --norepeat_baseline
--single_fact_line=LINE --steps=512`; its accompanying full-corpus batch-one
baseline is not an exposure-matched control and is not used to derive this
rule. All model checkpoints and diagnostic tables remain local artifacts.

All three prespecified **512-update** endpoints pass the frozen FP32-delta
rule without threshold adjustment: France selects exactly 10 target IDs,
mammals six, and Prasad 11, each with zero false positives or false negatives.
Every selected fact is also greedily memorized by its own trained model.
These are cross-sentence replications at one shared initialization, not
independent-seed or arbitrary-model validation. Secondary saved endpoints
show one extra token at update 128 for mammals and Prasad, but no errors at
256 or 384; France is exact at all four saved endpoints. BF16-endpoint and
trained-table-only exploratory variants also pass at 512. Prompt-only means
the prompt-token set minus the target-token set, since tokens can overlap.
The rule still recovers a **set**, not order or multiplicity. Evidence:
`/tmp/sign_replication_eval.JAtM05/` and
`/tmp/one_shot_single_replication_line_{80,1,258}_0/`.

A separate GPU context intervention on the Durian-only model preserves all
10/10 targets when keeping just the immediately preceding token and replacing
every older prefix token with EOS. Keeping **zero** lexical tokens (constant
EOS input, reading different absolute positions) gets only 1/10. Thus lexical
input matters, but this single-fact model does not need the full sentence
history under that replacement. This is independent target evaluation, not
an autoregressive audit, and positions remain available; it does not prove
a position-free bigram implementation. Evidence:
`/tmp/one_shot_single_fact_context_0/`.

The CPU-only `checkpoint_embedding_sign_readout` implements the rule, with
unit-tested shape/finite-value validation and no label argument in its core.
Optional `--target_ids` only scores the already-selected rows. It also reports
BF16 endpoint differences (round each table, then subtract), and separate
exploratory trained-table-only variants. At Durian step 512, update-rule target
scores span [-0.69287,-0.61445], versus [0.12752,0.84201] for all other rows;
the BF16 check preserves this separation. The trained-table-only rule also
selects the same ten IDs here, but does not replace the frozen primary rule.
Off-fact rows participated in every softmax denominator: they are not held-out
negative classes. These geometric scores are coordinate/gauge-dependent and
do not identify a universal, functionally invariant storage format.

```sh
bazel build -c opt //src/llm/experiments/one_shot_memorizer:checkpoint_embedding_sign_readout
bazel-bin/src/llm/experiments/one_shot_memorizer/checkpoint_embedding_sign_readout \
  --initial_embeddings=/home/ubuntu/checkpoints/memorize_general_facts/compact_batch_32_no_clip_0/layers_8/step_0/weight_0.bin \
  --trained_embeddings=/tmp/one_shot_sentence_pilot_0/only_line_631/step_512/weight_0.bin \
  --width=16 --target_ids=124,328,792,2147,86,3735,4018,2483,2,4474
```

### Why so many embedding rows move: the first Adam update

A subsequent [checkpoint-history reconstruction](SINGLE_FACT_DECODING.md)
orders the recovered token sets and exactly reconstructs all four single-fact
suffixes plus EOS. It reads neither their text nor their prompts, but assumes
each selected token occurs once and uses the known five-token prompt length.
The simpler greedy-successor attempt fails on three of the four cases.
An additional residual-update readout recovers the prompt-token sets; matching
120 candidate first updates then uniquely recovers the correct prompt order
for all four, completing the text reconstruction. This requires initialization,
the saved first update, known optimizer/loss settings, and the 512-step
single-fact checkpoint. It is not recovery of the 1,024-fact model or proof
that arbitrary final checkpoints can be inverted.

By default, `single_fact_first_update_probe` is a CPU-only explanatory replay of a
**known** training fact, not a decoder. It loads the initial weights, runs
the existing scalar reference forward/backward with the original ten Durian
suffix/EOS targets, and predicts the first embedding update. With fresh
moments and zero weight decay, bias-corrected first-step Adam simplifies in
exact arithmetic to

```
delta_E[i,j] = -learning_rate * gradient[i,j] /
               (abs(gradient[i,j]) + epsilon)
```

The tied embedding gradient includes the output head and input lookup. The
4,460 tokens absent from both input and targets have no input-lookup term,
but each still gets a softmax head gradient. If initial probabilities were
uniform, their head gradients would be the same averaged hidden vector
divided by vocabulary size. Actual probabilities are not exactly uniform;
the approximation has 11.33% relative gradient error, with matching signs
in 68,208/71,360 absent-row coordinates. Adam's coordinate normalization
turns similar gradients into similar-sized updates. The common absent-row
mean accounts for 89.46% of observed first-update energy in those rows.

Using the actual reference gradients rather than the uniform approximation,
the simplified update reproduces **71,457/71,600 resulting embedding weights
bit-identically** and every update sign. Relative L2 error of the update is
0.00013739 (about 0.014%). Remaining differences can include scalar CPU versus
GPU arithmetic and the optimizer's explicit FP32 moment/bias-correction
operations. Context lengths 16, 32, and 1,024 produce byte-identical CPU
coordinate reports, validating removal of ignored future padding here.
Invalid IDs, context lengths, nonfinite epsilon and excessive prompt length
are rejected. This explains much of the widespread early movement; it does
not reconstruct the ordered sentence from a converged checkpoint or prove
that later updates remain a single common shift.

```sh
bazel build -c opt //src/llm/experiments/one_shot_memorizer:single_fact_first_update_probe
bazel-bin/src/llm/experiments/one_shot_memorizer/single_fact_first_update_probe \
  --initial_checkpoint=/home/ubuntu/checkpoints/memorize_general_facts/compact_batch_32_no_clip_0/layers_8/step_0 \
  --first_update_checkpoint=/tmp/one_shot_sentence_pilot_0/only_line_631/step_1 \
  --output_dir=/tmp/first_update_probe_new
```

## Five-token execution tracing

The requested GPU experiment traces exact five-token
prompts through a **single sixth-token prediction**, comparing Paris/Athens/Lima
factual retrieval with `Pras -> ad`, lexical `to -> nour`, and grammatical
`known -> for` continuation. Raw traces will be paired with independent branch
and donor interventions; attention weights or intermediate vocabulary readouts
alone do not establish a causal explanation.

`checkpoint_token_trace` supplies only those five IDs; all later positions are
EOS padding, and only query row 4 is scored. It records every prefix activation
and triangular attention matrix. Each intervention starts an independent
forward: zero an attention/MLP branch while keeping its skip, transplant one
same-position residual or Q/K/V slice between the three capital prompts, or
optionally zero one query-row GELU neuron. No corpus suffix is model input.
Uninstrumented/captured logits and identity donor patches must be bit-identical;
structurally disconnected donor patches must also be identities.

Two descriptive views accompany these causal tests. The readout lens applies
the original final LayerNorm/head to earlier residuals. Fixed-final-normalizer
accounting instead decomposes the **actual final target-versus-rival margin**
over observed residual increments and the final norm bias. The latter fixes
the final normalization denominator so contributions telescope, reporting
FP32/BF16 numerical discrepancies separately. Neither is an early prediction,
nor a causal effect: those require the separate interventions. Raw vectors,
readouts, margins and interventions are written to TSV and self-contained HTML.

```sh
bazel build -c opt //src/llm/experiments/one_shot_memorizer:checkpoint_token_trace
bazel-bin/src/llm/experiments/one_shot_memorizer/checkpoint_token_trace \
  --checkpoint=/home/ubuntu/checkpoints/memorize_general_facts/compact_batch_32_no_clip_0/layers_8/step_16128 \
  --tokenizer=/home/ubuntu/checkpoints/memorize_general_facts/compact_batch_32_no_clip_0/inputs/tokenizer \
  --output_dir=/tmp/five_token_trace_new
```

Default lines are 80,406,411,1,258,631. Optional
`--mlp_neuron_ablation_blocks=0,4` independently ablates each GELU channel of
those blocks at the final query row only. The original weights never change.

### CPU-reference replay and GPU corroboration

The existing `LayerReference` primitives allow a CPU-only replay with no
Executor or CUDA runtime. A local C++ prototype uses context 16 (reference
dense layers require a multiple of 16), loads all 100 unique checkpoint
tensors, and keeps only position rows 0..15. Causality makes future positions
irrelevant to the first five rows. As a control, changing every future padding
token left all five prefix rows byte-identical at all 20 observed boundaries
for all six cases. CPU and GPU accumulation/math differ, so this is not a
claim of bitwise equivalence to GPU execution.

The CPU replay correctly predicts all six sixth tokens, around 4.4 ms per
query. Diagnostic final-head readouts show an informative contrast:

| Fact / target piece | After final attention | After final MLP |
| --- | ---: | ---: |
| France / ` Paris` | 0.00000406 | 0.92024 |
| Greece / ` Athens` | 0.00000145 | 0.9832 |
| Peru / ` Lima` | 0.01755 | 0.9770 |
| Mammals / ` nour` | 0.98195 | 0.98468 |
| Prasad / `ad` | 0.000102 | 0.9876 |
| Durian / ` for` | 0.1479 | 0.9875 |

Entries are probabilities, not percentages. Earlier readouts are
counterfactual applications of the final LayerNorm/head, not predictions
actually emitted early. For the capitals, the last MLP makes the answer
compatible with that head. Earlier MLPs still matter under ablation: removing
block 4 or 5 severely damages these answers despite their poor direct early
readouts. This is consistent with intermediate computation preparing a useful
representation, not proof of a specific symbolic code or storage address.
The mammals case differs: the last attention raises ` nour` from about 0.01158
to 0.98195, and the final MLP adds little. Under single-branch deletion France
loses its correct answer in all 16 cases, while Peru tolerates several branch
deletions. Neither joint removal of attention nor joint removal of MLPs works
for any of the six prompts. These interventions can leave the training
distribution; they establish sensitivity under the specified changes, not
universal necessity or exclusive fact ownership.

Reference backward additionally provides gradients of the target-minus-best-
rival logit margin at each observed boundary. Because BF16 rounding is treated
straight-through, these are **training-surrogate sensitivities**, not literal
derivatives of the quantized forward function. For example, the mammals query
has a much larger initial-position gradient norm on `Female` (about 61.1) than
on ` mammals` (2.86), while capital queries are sensitive to the country row
and also generic template tokens. A large gradient on `The` does not mean the
capital fact is stored there. After the last attention, every nonquery row's
gradient is exactly zero, as the remaining operations are pointwise. All
master weights were byte-checked unchanged after backward.

Local CPU evidence: `/tmp/cpu_reference_gpt2_trace.cc`,
`/tmp/cpu_reference_gpt2_trace.tsv`, and
`/tmp/cpu_reference_gpt2_interventions_gradients.tsv`. The subsequent GPU
trace agrees on top-1 and target rank for all 204 matched baseline, branch,
and lens conditions. Maximum target-probability discrepancy is 3.53e-7; all
1,824 compared BF16 query-row activation coordinates are exactly equal.

A second CPU diagnostic swaps one same-position Q/K/V slice between France,
Greece, and Peru, at blocks 0, 1, and 7. All 108 cross-prompt single-slice
interventions failed to transfer the donor capital as top-1. This is a negative
result for these specific interventions, not proof that the capital has no
localized representation. All 78 structurally required identity interventions
passed, as did 48 bytewise checks that positions before the country have the
same Q/K/V across all three prompts and all eight blocks.

The last attention mixes a fixed `The` anchor and the country position with
very different weights: approximately 96.06%/3.15% for France, 43.17%/52.90%
for Greece, and 18.11%/81.03% for Peru. Block 5 instead places 84--96% on the
constant ` capital` position. These earlier positions cannot encode the later
country, and the exact shared-prefix checks confirm that they do not vary.
Attention can therefore transform a context-dependent query by mixing fixed
anchors; the most-attended token is not necessarily a fact's storage location.
This is a candidate mechanism, not yet a complete causal circuit.

Q/K/V interventions are asymmetric. At block 7, transplanting Peru's country
key into France reduces ` Paris` from 0.92024 to about 0.000915, while the
corresponding value transplant leaves it near 0.91656. France's country value
transplanted into Peru instead reduces ` Lima` from 0.97701 to 2.39e-12. At
block 1, Peru's country-position query transplanted into France reduces
` Paris` to 5.65e-5: this query only changes the country row at that block,
which later blocks can read. The same query transplant at the last block has
exactly no effect on the later prediction row, as causality requires.
These results suggest multi-stage country processing and attention gating,
not a single answer-bearing attention weight. Local evidence is
`/tmp/cpu_reference_qkv_patch.{cc,tsv}`. The GPU study subsequently confirms
all 108 overlapping slice interventions, with maximum probability difference
2.50e-7, and extends the negative transfer result to all **288** Q/K/V swaps
across eight blocks.

A broader CPU scan enumerated all 1,024 tokenized sentences from the verified
constructed automaton, checked distinct five-token prefixes, and supplied
only those prefixes plus EOS padding to the neural reference model. Its
ordinary sixth-token predictions are **1,024/1,024 correct**. The same final
LayerNorm/head applied at each residual boundary gives:

| Boundary | After attention | After MLP |
| --- | ---: | ---: |
| Block 0 | 4 | 7 |
| Block 1 | 10 | 15 |
| Block 2 | 23 | 52 |
| Block 3 | 46 | 45 |
| Block 4 | 48 | 44 |
| Block 5 | 48 | 94 |
| Block 6 | 92 | 190 |
| Block 7 | 268 | 1,024 |

Embedding-plus-position alone gives 3/1,024. These are correct top-1 counts,
not full-suffix completions. The final MLP turns 756 wrong final-head readouts
into correct ones, increasing the target-minus-best-rival margin for 1,006
prompts. Unlike an earlier lens, the last-attention lens also equals bypassing
the remaining pointwise MLP branch at the query: the only subsequent layers
are the original final norm/head. This supports a broad computational role
for the final MLP but not exclusive fact storage there. Earlier blocks can
prepare information that this MLP makes readable. All final-lens query logits
were byte-identical to ordinary reference execution. CPU evidence:
`/tmp/cpu_all_facts_lens_0.{tsv,summary}`; case indices are DFS order, **not**
corpus line numbers. Direct GPU corroboration in
`/tmp/gpu_prefix_mlp_probe_0/` gives exactly the same **1,024 versus 268**
correct sixth tokens, with all 1,024 subsequent unpatched controls restoring
every query logit bit-for-bit.

### What the GPU interventions establish

The main trace report is `/tmp/one_shot_token_trace_0/report.html`, with raw
TSV vectors, probabilities and scores beside it. It completed 690 conditions:
six baselines, six identity donor copies, 102 readout lenses, 96 branch
ablations, and 480 donor interventions. All hook/no-hook and structurally
required identity checks preserve exact logits. Attention rows normalize
within 1.28e-7; linear readout accounting closes within 6.22e-15.

Full country-position residual swaps after block 0 transfer the donor capital
in all six ordered France/Greece/Peru pairs. Success falls across blocks to
5/6, 5/6, 4/6, 3/6, 2/6, then 0/6 after block 6; swaps after the last attention
are disconnected identities. Full query-state swaps after the final attention
transfer all six answers, as expected when only pointwise computation remains.
By contrast, all six query-state swaps after block 6 produce a third answer,
not the recipient or donor capital. Mixing an intermediate query with another
history can be incompatible. These tests locate information flow, not a
context-independent fact vector or exclusive ownership.

An additional 384 independent last-MLP GELU-neuron deletions find two
single-neuron top-1 flips among these six prompts. Zeroing channel 12 changes
France from ` Paris` (92.02%) to ` capital` (48.07%; Paris remains 32.13%).
Zeroing channel 34 changes Greece from ` Athens` (98.32%) to ` divided`
(44.90%; Athens remains 35.92%). No individual deletion flips the other four
answers. These channels are necessary for these particular decisions under
zero ablation, not proven exclusive owners of those facts. Evidence:
`/tmp/one_shot_token_neurons_7_0/`.

The corpus-wide follow-up supplies exactly five tokens for every fact, with
no future lexical input. All 1,024 baseline sixth-token predictions are right.
Independently zeroing final-MLP query channel 12 breaks **three** answers:
Paris, Chile, and ` gas` (lines 80, 448, 795). Channel 34 breaks **35**, spanning
capitals, biology, mathematics, and other topics. Thus neither channel is a
dedicated Paris/Athens feature. The gas case has a *negative* GELU activation
(-0.1113); deleting an inhibitory contribution can also break a decision.

Deleting both channels breaks 35 answers, not the union of the separate
failures. This is decision-level interaction, including the final
normalization/head, not evidence that the MLP's output projection ceases to
be linear. Every prompt ends with an ordinary forward whose 4,475 logits
match its original baseline bit-for-bit. Local evidence:
`/tmp/one_shot_prefix_neurons_0/` (5,120 forwards, 17.0 seconds). This is a
**next-token** study, not 1,024 full-suffix ablations, and these two channels
were chosen after the six-prompt experiment.

A CPU projection of each channel's BF16 contraction-weight row through the
final LayerNorm scale and tied BF16 head reinforces this distinction. Define
`a[j,t] = dot(center(W2[j,:]), gamma * E[t,:])`, without the context-dependent
normalization scale. Paris ranks only **1,084/4,475** along channel 12;
Athens ranks **264/4,475** along channel 34. The highest-scoring token on both
directions is `.`. These are shared voting directions, not isolated stored
answer vectors. What matters is a *difference* against a competing token,
multiplied by the context's signed GELU activation and combined with every
other contribution. In the gas case, channel 12's negative activation times
its gas-minus-liquid direction difference is positive (+0.0417 before
normalization), even though gas's raw direction rank is only 3,971. Actual
ablation margins also include changed normalization, final bias, and BF16
rounding. Local calculations: `/tmp/neuron_vocab_direction_0/`.

This is compatible with the key/value view of MLPs: activation patterns select
a mixture of output directions, rather than necessarily selecting one cell
per fact. The literature motivates inspecting these directions but does not
guarantee human-readable individual channels in this much narrower model.
[Geva et al., Transformer Feed-Forward Layers Are Key-Value Memories](https://aclanthology.org/2021.emnlp-main.446/).

The committed trace tool supports targeted independent channel sweeps via
`--mlp_neuron_ablation_blocks=7 --mlp_neuron_ablation_channels=12,34`.
Empty channels retain the default sweep over every channel. Its reports also
include the full baseline trace; for example:

```sh
bazel-bin/src/llm/experiments/one_shot_memorizer/checkpoint_token_trace \
  --checkpoint=/home/ubuntu/checkpoints/memorize_general_facts/compact_batch_32_no_clip_0/layers_8/step_16128 \
  --tokenizer=/home/ubuntu/checkpoints/memorize_general_facts/compact_batch_32_no_clip_0/inputs/tokenizer \
  --lines=80,406,448,795 --nobranch_ablations --nocapital_donor_patches \
  --noreadout_lens --mlp_neuron_ablation_blocks=7 \
  --mlp_neuron_ablation_channels=12,34 --output_dir=/tmp/targeted_neurons_new
```

Attribution also depends on the question asked. Against the **final** nearest
rival ` north`, France's final MLP has a negative fixed-normalizer margin
term. Against the **pre-MLP winner** ` original`, it has a positive 44.53 term,
with neuron 12 the largest positive term (11.56). The actual Paris-minus-
original margin rises from -11.23 to +23.62. Athens-minus-` usually` rises
from -13.14 to +14.39, with neuron 34 its largest positive projection term.
The frozen-normalizer terms are not the actual pre/post margin changes:
normalization and rival selection must not be silently conflated. CPU
accounting from GPU captures is in `/tmp/cpu_pre_mlp_rival_accounting.tsv`.

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
Label-projection tests cover deterministic/unique code assignments, normalization,
label-only targets, whole-sentence exclusion from every fitted statistic,
supervised residual capture, independent output heads, fresh hook substitutions,
clone equivalence, and unchanged original parameters. Decision tests additionally
check exact real-valued LayerNorm/head algebra, nonzero bias changing winners,
contradictory labels, unseen rivals, negative common margins, coefficient/gauge
bounds, deterministic cutting planes, and explicit budget outcomes.
All **93 repository test targets passed with fresh execution**, including
the five-token, sign-readout, margin-accounting, exact path-solver, and
first-update score tests.
The real-checkpoint
trace, independent neuron sweep, corpus-wide final-MLP bypass, paired training,
paired weight transplants, and prospective single-fact replications also
completed their controls. All four weight-only suffix reconstructions reproduce
under all three scoring modes; the paired-transplant CLI's CPU self-test checks
row ranking, tie-breaking, direction invariance, malformed shapes, and NaNs.
