# Analytic weight-to-text investigation

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

Next experiments should retain signed multi-neuron contributions, examine
cross-layer virtual operators, compare early/late checkpoints, and test causal
specificity of candidate weight groups. If activation-assisted discovery is
needed, keep its data dependence explicit and evaluate any resulting analytical
extractor separately. The original weight-to-text mapping/decompression goal
remains open until supported by actual recovered passages and controls.
