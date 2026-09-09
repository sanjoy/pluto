# Analytic weight-to-text investigation

The vocabulary-recovery experiment has its own fixed
[protocol](/home/ubuntu/code/pluto/research/weight_memorization/VOCABULARY_PROTOCOL.md)
and [verified findings](/home/ubuntu/code/pluto/research/weight_memorization/VOCABULARY_RESULTS.md).
It tests a context-independent output-head term and does not change any prior
frozen extraction or restore the user's stashed reports.

The subsequent wider-vocabulary attention-polynomial experiment also has a
fixed [protocol](/home/ubuntu/code/pluto/research/weight_memorization/LAZY_POLYNOMIAL_PROTOCOL.md)
and [verified findings](/home/ubuntu/code/pluto/research/weight_memorization/LAZY_POLYNOMIAL_RESULTS.md).
None of its 5,120 sixteen-token paths contains a four-token corpus substring.
It documents learned local ranking changes, repetition, and the gap between
those associations and passage recovery.

The next experiment is explicitly **model-based causal validation**, not an
extractor. Its fixed [protocol](/home/ubuntu/code/pluto/research/weight_memorization/CAUSAL_VALIDATION_PROTOCOL.md)
and [verified results](/home/ubuntu/code/pluto/research/weight_memorization/CAUSAL_VALIDATION_RESULTS.md)
include all 35 arms, a byte-addressed 32-passage index, checkpoint-file group
addresses, exact restoration checks, and reproduction commands. The first MLP
has broad effects, while halving either of the last two MLPs hurts every sampled
prefix passage and improves every sampled suffix passage. This is functional
dependence, not uniquely located text or an analytical decompressor.

The [fine-grained follow-up](/home/ubuntu/code/pluto/research/weight_memorization/NEURON_MAPPING_RESULTS.md)
tests 64 fixed groups of late-MLP neurons with disjoint discovery/confirmation
targets in the same passages. All 69 arms pass integrity checks, but only 5 of
16 selected groups remain in the top confirmation quartile, with broad
collateral effects. The complete losses, addresses, and alternate numerical
audit are retained; this is again model-based reliance mapping, not extraction.

The [checkpoint-difference feasibility audit](/home/ubuntu/code/pluto/research/weight_memorization/CHECKPOINT_DELTA_FEASIBILITY.md)
also explains why saved ten-step AdamW weight differences are not raw gradients.
The follow-up [trajectory audit](/home/ubuntu/code/pluto/research/weight_memorization/CHECKPOINT_TRAJECTORY_RESULTS.md)
adds executable optimizer counterexamples and measures all four latest unpacked
checkpoint intervals. Embedding changes have a dominant direction but a broad
singular-value tail; shared shifts, rigid motion, and fitted scale explain much
of their energy. These descriptive measurements are neither raw-gradient
recovery nor an update-to-text decoder. All checkpoint weights remain unchanged.

The new [late-MLP polynomial protocol](/home/ubuntu/code/pluto/research/weight_memorization/LATE_MLP_POLYNOMIAL_PROTOCOL.md)
tests a static cross-layer interaction between MLPs 6 and 7. This layer choice
is informed by the causal experiment, not an independent data-free discovery.
`late_mlp_polynomial.py` compiles fixed weight-derived coefficients and optimizes
complete sixteen-token strings without a model forward. The coefficients use
declared proxy anchors and omit attention and actual contextual normalization;
the resulting score is neither a GPT-2 logit nor its full Taylor polynomial.
`late_mlp_paths.py` freezes each plan before searching, retains every restart,
and checks checkpoint/source identities before emitting candidates. Follow the
protocol's five arms and freeze all candidates before corpus verification.
The [verified results](/home/ubuntu/code/pluto/research/weight_memorization/LATE_MLP_POLYNOMIAL_RESULTS.md)
show no passage recovery: final-weight arms reach at most two tokens, and the
early baseline's sixteen-space match survives all six n-gram shuffle controls.
An exploratory scalar-envelope audit also exposes large errors in the local
quadratic approximation; this does not measure the complete model's error.

Goal: understand how Pluto's Shakespeare GPT-2 stores its training text, map
recoverable portions to precise weight groups, and seek a simple analytical
decompressor. Running the language model to generate text does **not** satisfy
that goal. The current tools are experiments toward it, not a completed
decompressor or proof that arbitrary passages can be recovered from weights.

Verified findings and artifact hashes are recorded in the
[research report](/home/ubuntu/code/pluto/research/weight_memorization/REPORT.md).

## Evidence contract

1. **Extraction reads checkpoint weights and the tokenizer vocabulary only.**
   No corpus, corpus frequencies, model prompts, transformer forward passes,
   attention probabilities, or corpus-derived activations enter this stage.
2. Freeze and hash the candidate JSONL before verification. The tokenizer gives
   meanings to token IDs, not a hidden copy of Shakespeare. Single vocabulary
   tokens and common BPE pairs are not evidence of passage memorization.
3. Only the verifier sees the corpus. It finds exact candidate substrings,
   reports original byte ranges and frequency, and never alters candidates.
   Changing extraction choices after observing matches would be data-assisted
   discovery; any such future experiment must be explicitly identified.
4. Compare with broken weight pairings and label permutations. These are
   diagnostics, not calibrated null distributions or statistical p-values.
   Broken pairings can change source selection and graph topology; compare
   per-method denominators, unique strings, and repeated formatting separately.
5. A successful weight map needs more than matching strings: reproducing the
   extraction from the addressed weights, specificity to the claimed passage,
   and independent causal validation remain necessary. Validation may run a
   model, but must never be passed off as the analytical decompressor itself.

## Model and layout

The latest uncompressed GPT-2 checkpoint inspected is
`/home/ubuntu/checkpoints/shakespeare/step_13030`, not a newer SAE checkpoint.
It contains 100 distinct FP32 tensors: 51,483,648 parameters / 205,934,592 bytes.
The recipe has 8 blocks, residual width 512, 8 heads, FFN width 2,048, and a tied
50,257-token embedding/unembedding (physically padded to 50,272 rows).

`checkpoint.py` reproduces the actual C++ traversal, validates every filename
and size, and exposes read-only NumPy memory maps. Dense matrices are stored
`W[input, output]`, so `y = x @ W + b`. Keys are MLP input **columns**, and values
are MLP output **rows**. The LM head shares the token embedding allocation;
counting it again would invent a nonexistent checkpoint file.

Metadata records weight hashes and current source hashes. The bare checkpoint
has no historical architecture/producer metadata: matching shapes and current
source hashes cannot prove the historical producer was identical.

## Strategies implemented

### MLP key/value projection and static paths

For each neuron, `mlp.py` compares its input key and output value with all token
embedding rows, using cosine similarity. Positive key/value alignment products
form directed token-association edges. It emits top pairs for all16,384 neurons
and bounded paths in the static edge graph, with physical file/byte views for
every contributing key/value vector. It is not an autoregressive transformer:
path search never updates a hidden state or reevaluates any model layer.

This probe omits GELU gates, positions, LayerNorm, attention and contextual
residual inputs. It also discards cancellations between neurons by retaining
the strongest edge. Thus a path is only a candidate, not proof of text storage.
The paired-neuron permutation test checks reparameterization consistency;
cyclically shifting values relative to keys supplies a broken-pairing control.

### Attention OV products

`attention.py` computes the static product
`E[source] @ W_V[head] @ W_O[head] @ E[target].T`, with raw or cosine scoring.
Its default selects32 source tokens per head by projected-write norm, then the
top4 targets across the full vocabulary. These choices use weights, not corpus
statistics. All64 heads are covered without forming a vocabulary-squared
matrix. A fixed mismatching of value and output heads is the control.

This is a directed association operator, not attention: it omits Q/K routing,
softmax, positions, normalization and contextual inputs. Source norm selection
is biased and covers only part of the vocabulary. The tied-embedding Gram
matrix alone is symmetric and cannot independently establish next-token order.

## Reproduce

Run from the repository root with Python3.12 and the dependencies in
`requirements.txt` (the recorded environment is `/home/ubuntu/.venv/bin/python`).
Use fresh output names; the tools refuse to overwrite artifacts.

```sh
OPENBLAS_NUM_THREADS=8 python -m scripts.weight_analysis.mlp \
  --checkpoint /home/ubuntu/checkpoints/shakespeare/step_13030 \
  --tokenizer-dir /home/ubuntu/datasets/tokenizer/gpt2 \
  --output /tmp/mlp_candidates.jsonl --include-control

OPENBLAS_NUM_THREADS=8 python -m scripts.weight_analysis.attention \
  --checkpoint /home/ubuntu/checkpoints/shakespeare/step_13030 \
  --tokenizer-dir /home/ubuntu/datasets/tokenizer/gpt2 \
  --output /tmp/attention_candidates.jsonl
```

Repeat attention extraction with `--broken-head-control` and a separate output
path. The MLP command includes real and broken candidates in one file, with
different method names. Candidates include exact token IDs; readable BPE
labels and displays must not be confused with independently verified text.

### Use the exact training tokenizer for verification

The native C++ pre-tokenizer differs from Hugging Face's on some whitespace.
For this corpus, native produces1,835,163 tokens versus HF1,836,425. In
particular, native encodes `a\n\nb` as `[64,628,65]`, while HF uses
`[64,198,198,65]`. We preserve the historical training behavior rather than
silently changing it to make an analysis agree.

```sh
bazel build -c opt //scripts/weight_analysis:tokenize_corpus
bazel-bin/scripts/weight_analysis/tokenize_corpus \
  /home/ubuntu/datasets/tokenizer/gpt2 testdata/shakespeare.txt \
  /tmp/shakespeare_native.bin

python -m scripts.weight_analysis.verify \
  --candidates /tmp/mlp_candidates.jsonl \
  --tokenizer-dir /home/ubuntu/datasets/tokenizer/gpt2 \
  --corpus testdata/shakespeare.txt \
  --corpus-token-ids /tmp/shakespeare_native.bin \
  --corpus-byte-offsets /tmp/shakespeare_native.bin.offsets.bin \
  --output /tmp/mlp_verified.json
```

The native exporter validates a full round trip and every individual token's
bytes. Outputs are headerless little-endian uint32 IDs and uint64 byte offsets
(N+1 entries). The verifier independently checks every supplied byte interval,
including tokens that split UTF-8. Its optional HF backend is explicitly labeled
and is not interchangeable with the production token stream.

## Tests

```sh
OPENBLAS_NUM_THREADS=8 python -m unittest \
  scripts.weight_analysis.checkpoint_test \
  scripts.weight_analysis.mlp_test \
  scripts.weight_analysis.attention_test \
  scripts.weight_analysis.verify_test
bazel-bin/scripts/weight_analysis/tokenize_corpus \
  --self-test /home/ubuntu/datasets/tokenizer/gpt2
```

Synthetic fixtures plant a known ordered chain in weights and recover it
without providing its text, check broken-pairing destruction, permutation
invariance, tensor orientation, byte addresses, exact matching against a
brute-force oracle, UTF-8 boundaries, and refusal to overwrite artifacts.
These tests verify the implementation, not the hypothesis that real passages
are stored in the same simple form.

## Signed aggregate MLP probe and stronger path control

`aggregate.py` forms `A = W1 @ diag(g) @ W2` and scores
`E[source] @ A @ E[target].T`. Unlike the individual-neuron probe, it retains
all positive and negative contributions before ranking tokens. The default
gate is `g=1`; `--gate=bias_gelu` instead uses the analytic tanh-GELU derivative
at the fixed bias point. Neither mode evaluates contextual activations.
The optional bias gate is an approximation, not the trained model's gate.

Source selection uses write norms over the entire vocabulary. Target scores
are cosine-normalized, with raw scores and norms also recorded. Paths pool
positive edges across blocks; they can terminate before their requested
maximum length. This is still a pairwise operator, not a multi-token memory
model, even though every neuron contributes. `neuron_contributions()` returns
all signed terms for exact algebraic replay; a few largest terms alone are not
a complete explanation of the sum.

```sh
OPENBLAS_NUM_THREADS=8 python -m scripts.weight_analysis.aggregate \
  --checkpoint /home/ubuntu/checkpoints/shakespeare/step_13030 \
  --tokenizer-dir /home/ubuntu/datasets/tokenizer/gpt2 \
  --output /tmp/aggregate_candidates.jsonl --include-control
```

`controls.py` is **verification only**. It shuffles the native corpus using a
randomized Euler trail that preserves every directed adjacent token pair and
its count, every unigram count, and the endpoint tokens. It then checks the
same frozen candidates. Thus it tests whether path matches provide evidence
for ordering beyond token pairs, rather than merely outperforming arbitrary
vocabulary labels. It does not sample trails uniformly and produces no
p-values. Some longer sequences are forced by their bigrams, so even this
control cannot isolate all forms of learning or memorization.

The control excludes candidates shorter than three tokens because pair counts
are identical by construction. Its denominators are therefore different from
the full extraction report and explicitly recorded. Whole-candidate matches
are reported separately from matches to internal substrings.

```sh
python -m scripts.weight_analysis.controls \
  --corpus-token-ids /tmp/shakespeare_native.bin \
  --candidates /tmp/mlp_candidates.jsonl \
  --output /tmp/mlp_bigram_controls.json

OPENBLAS_NUM_THREADS=8 python -m unittest \
  scripts.weight_analysis.aggregate_test \
  scripts.weight_analysis.controls_test
```

For the first MLP experiment the bigram-preserving shuffles contain **more**
four- and five-token path matches than the actual corpus. Consequently the
initial formatting-heavy matches do not demonstrate passage-specific storage.
See the research report for exact denominators and all three fixed seeds.

## Joint attention routing/content probe

`trigram.py` moves beyond independent pairs. For previous token `b`, current
token `a`, and candidate next token `c`, it forms this static weight contraction:

```text
r_h(a,b) = (E[a] WQ[h] + bQ[h]) · ((E[b] - E[a]) WK[h]) / sqrt(head_dim)
write(a,b) = sum_h r_h(a,b) * ((E[b] - E[a]) WV[h] WO[h]) / 4
score(b,a,c) = cosine(write(a,b), E[c])
```

The write is the exact first derivative of two-position attention when its
routing scores are scaled from zero (uniform routing). That identity is
checked against finite differences in tests; the extractor itself does not
execute softmax or transformer layers. It is **not** the full attention output
or a guarantee that the approximation is accurate at the trained score scale.
LayerNorm, position embeddings, residuals, MLPs, and later blocks are omitted.

The default searches block zero: 128 high-query-norm current tokens, eight
previous tokens per current selected by the norm of the signed total write,
and four destinations per pair. Each search considers the full vocabulary,
but this is only a small subset of all possible triples. Paths must overlap
in both conditioning tokens: `(b,a,c)` can be followed by `(a,c,d)`, never by
an edge sharing just one token. They may terminate early or not exist at all.

The broken control rotates **intact OV head pairs relative to QK heads**. This
preserves each routing and content operator while testing their association;
it differs from the earlier OV control that breaks value/output pairing.

```sh
OPENBLAS_NUM_THREADS=8 python -m scripts.weight_analysis.trigram \
  --checkpoint /home/ubuntu/checkpoints/shakespeare/step_13030 \
  --tokenizer-dir /home/ubuntu/datasets/tokenizer/gpt2 \
  --output /tmp/trigram_candidates.jsonl

OPENBLAS_NUM_THREADS=8 python -m unittest scripts.weight_analysis.trigram_test
```

Repeat with `--broken-routing-control` and a fresh output name; run the same
settings on the early checkpoint. Verify frozen candidates with `verify.py`
and `controls.py`, just as for the earlier probes. A matched triple still is
not sufficient evidence of passage decompression. This remains an experiment
toward the original goal, not a replacement of that goal with token statistics.

## Corrected coordinates, closed vocabulary, and order-matched controls

`closed_trigram.py` follows a separately committed
[protocol](/home/ubuntu/code/pluto/research/weight_memorization/CLOSED_GRAPH_PROTOCOL.md).
It uses the actual first LayerNorm's learned scale/bias and position-zero/one
embedding rows to prepare static input dictionaries. Arithmetic is FP64;
BF16 intermediate rounding is not emulated. The original tied embedding,
not the normalized input dictionary, supplies output-token directions.

Select a 128-token vocabulary from a declared selector checkpoint, then score
all ordered pairs and top-four destinations within that same set. The same
final-selected token IDs are supplied to normalized, raw-input, broken-routing,
and early-checkpoint variants. This removes the earlier disjoint-endpoint
problem, but restricts every proposed word to a potentially unsuitable small
vocabulary. An early control given final-selected IDs has that explicit learned
prior. Negative finite top scores are retained; scores are not probabilities.

```sh
OPENBLAS_NUM_THREADS=8 python -m scripts.weight_analysis.closed_trigram \
  --checkpoint /home/ubuntu/checkpoints/shakespeare/step_13030 \
  --vocabulary-checkpoint /home/ubuntu/checkpoints/shakespeare/step_13030 \
  --tokenizer-dir /home/ubuntu/datasets/tokenizer/gpt2 \
  --output /tmp/closed_normalized.jsonl

python -m scripts.weight_analysis.higher_order_controls \
  --corpus-token-ids /tmp/shakespeare_native.bin \
  --candidates /tmp/closed_normalized.jsonl \
  --output /tmp/closed_trigram_controls.json
```

Run all four protocol arms and freeze their hashes before corpus verification.
The higher-order control shuffles an Euler trail on observed token-pair nodes,
preserving **every trigram multiplicity**, lower-order counts, and endpoint
pairs. It tests only candidates of length four or more. This is needed because
trigram-conditioned graph paths can match ordinary local transitions without
containing passage-specific information. Both control families remain
nonuniform descriptive diagnostics, not significance tests.

The frozen closed-graph experiment produced connected twelve-token paths, but
none matched even a corpus pair. A later verification-only coverage audit
found that its vocabulary cannot express any three-token corpus substring.
No candidates were retuned to hide this outcome; see the research report.

### Check the attention approximation without executing attention

`routing_audit.py` uses exactly the same selected dictionaries. For a head's
score difference `g` and projected value difference `v`, the exact correction
relative to uniform routing is `0.5*tanh(g/2)*v`, while the static probe uses
`g*v/4`. Its error is bounded by

```text
max(|g|-2, 0) * ||v|| / 4 <= error <= min(|g|^3/48, |g|/4) * ||v||.
```

The audit evaluates neither tanh nor softmax: these inequalities follow from
their ranges and derivatives. It records per-head and stacked-head bounds,
with ratios relative to the linear correction norm. **They are not bounds on
summed-head errors, full model logits, or candidate-ranking errors.** A zero
lower bound is inconclusive; an upper bound can constrain numerical error in
the specified head correction without making it a complete model.

```sh
OPENBLAS_NUM_THREADS=8 python -m scripts.weight_analysis.routing_audit \
  --checkpoint /home/ubuntu/checkpoints/shakespeare/step_13030 \
  --vocabulary-checkpoint /home/ubuntu/checkpoints/shakespeare/step_13030 \
  --output /tmp/closed_routing_audit.json

OPENBLAS_NUM_THREADS=8 python -m unittest \
  scripts.weight_analysis.closed_trigram_test \
  scripts.weight_analysis.higher_order_controls_test \
  scripts.weight_analysis.routing_audit_test
```

## Wider-vocabulary lazy attention polynomial

`lazy_trigram.py` follows the fixed
[five-arm protocol](/home/ubuntu/code/pluto/research/weight_memorization/LAZY_POLYNOMIAL_PROTOCOL.md).
It selects 8,192 IDs from the previously frozen final embedding-centroid
ranking, uses all 256 ordered pairs of the first 16 ranked IDs as starts, and
permits every token repetition. It computes only visited two-token contexts;
beam width four and length sixteen bound the search. Every path records its
cached edges for exact replay. There is no evolving transformer hidden state:
the same token pair always has the same outgoing edges, at positions zero/one.

The three components are uniform-routing output `w0`, the routing derivative
`w1`, and their first-order sum `w0+w1`. Destination scores are signed
`unit(write) dot (E[token] - mean_ALL_LOGICAL_VOCAB(E))`, with no target-norm
division. This preserves raw-dot destination ordering, removes an arbitrary
common output translation from the path score, and keeps tiny nonzero write
directions. It does **not** turn the scores into full-model logits or likelihoods.
Write norms and cancellation ratios expose the risk of amplifying small sums.

```sh
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python -m scripts.weight_analysis.lazy_trigram \
  --checkpoint /home/ubuntu/checkpoints/shakespeare/step_13030 \
  --rankings /tmp/pluto-vocabulary-offset.hFYW8aCk/rankings.json \
  --component first_order --output /tmp/final_first_order.jsonl

python -m scripts.weight_analysis.path_diagnostics \
  --candidates /tmp/final_first_order.jsonl \
  --tokenizer-dir /home/ubuntu/datasets/tokenizer/gpt2 \
  --output /tmp/final_first_order.structure.json

OPENBLAS_NUM_THREADS=4 python -m unittest discover \
  -s scripts/weight_analysis -t . -p '*_test.py' -q
```

Run and freeze **all five** protocol arms before opening the corpus. Structural
diagnostics come first and never filter candidates: they distinguish whole-path
periods from long constant/alternating tails hidden behind varied seed tokens.
Then use the native-token verifier and both shuffle controls described above.
Separate complete paths from their cached three-token edges and from internal
substring matches; a seed-only match is not a generated continuation.

`attention_certificates.py` adds a **summed-head upper** error bound, separate
from `routing_audit.py`'s stacked-head diagnostics. When the approximate raw
top-token margin exceeds its worst-case perturbation, the same top token is
guaranteed in ideal real arithmetic for this exact two-position attention
subproblem. The implementation reports an ordinary-FP64 sufficient-condition
check, not an interval proof or a certificate for GPT-2, path order, or text
recovery. An unsatisfied condition is inconclusive. Independent scalar-oracle
tests check the polynomial using only toy weights, never the real model.

### Late-MLP cross-layer polynomial

The exact score, anchors, controls and search budget are fixed in
`research/weight_memorization/LATE_MLP_POLYNOMIAL_PROTOCOL.md`. A single arm is:

```sh
OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 python -m scripts.weight_analysis.late_mlp_paths \
  --checkpoint /home/ubuntu/checkpoints/shakespeare/step_13030 \
  --selection-checkpoint /home/ubuntu/checkpoints/shakespeare/step_13030 \
  --tokenizer-dir /home/ubuntu/datasets/tokenizer/gpt2 \
  --mode full --label final_full --output /tmp/new_final_full.jsonl
```

Use new output paths. Repeat with modes/labels `affine/final_affine`,
`no_cross/final_no_cross`, and `broken/final_broken`. The fifth arm uses mode
`full`, label `early_full`, and the early checkpoint as `--checkpoint`; its
`--selection-checkpoint` remains the final checkpoint. The frozen experiment's
early copy is `/tmp/pluto-attention-early.ee35wjay/step_10`, not initialization.
Default search settings are exactly the protocol's 8,192 vocabulary IDs,
64 sixteen-token starts, eight alternating sweeps, seed 20260909, alpha 0.125.

Each run writes an exclusive `.plan.json` before compilation, candidate JSONL
after search, and `.metadata.json` on completion. The latter includes all
coordinate-update traces, fixed-coefficient hashes before/after search,
source/checkpoint integrity checks, and repetition/order diagnostics. Plans
record runtime and requested thread settings. Freeze all five completed
candidate/plan/metadata identities before invoking any corpus verifier.
The score is a declared static approximation, not a contextual GPT-2 logit;
coordinate ascent is not guaranteed to find its global maximum.

`late_mlp_experiment freeze --run-dir DIR` independently checks all five named
arms and writes `combined.jsonl` plus `frozen.json`. Run this before verification
in a fresh reproduction. Its timestamp cannot establish that no other earlier
corpus check occurred. `late_mlp_experiment summarize --run-dir DIR
--native-tokens FILE --split 1650781 --output NEW.json` retains all candidate
denominators and reports full/current-prefix/current-suffix exact matches.
`late_mlp_envelope --plan FILE --metadata FILE --output NEW.json` is a separate,
weight-only diagnostic of the frozen scalar approximation, not an extractor or
a bound on the complete model's logit error.

## Research grounding and next questions

- [Geva et al.,2021](https://aclanthology.org/2021.emnlp-main.446/) motivate MLP
  key/value analysis. Their readable trigger examples used model activations;
  the paper does not establish a static long-passage decoder.
- [Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html)
  motivates OV and virtual weight products. Context-dependent routing must not
  be silently omitted when making claims about the complete model.
- [DAGER](https://arxiv.org/html/2405.15586v2) reconstructs text from gradients
  under rank assumptions, not arbitrary final weights. Multi-step AdamW
  checkpoint differences are not raw gradients, and its ordering procedure
  also evaluates model prefixes. It therefore does not directly solve this task.
- [ROME](https://arxiv.org/abs/2202.05262) offers causal localization/editing
  techniques, useful for validating a map but not themselves a decompressor.

Signed multi-neuron contributions, early/late comparisons, and the wider
first-block graph have now been tested without passage recovery. The next
direction should be cross-layer operators and independent causal specificity
of candidate weight groups, not another unsupported expansion of the same
stationary graph. If activation-assisted discovery is needed, keep its data
dependence explicit and evaluate any resulting analytical extractor separately.
The original weight-to-text mapping/decompression goal remains open until
supported by actual recovered passages and controls.
