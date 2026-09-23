# Five-completion execution traces

Read-only C++ instrumentation for the memorized compact-vocabulary GPT-2 model.
It supplies the first five tokens of selected facts and greedily generates five
more, retaining every layer output at every active input position. These are
actual autoregressive forwards, not teacher-forced continuations.

## Reproduce

From the repository root:

```sh
bazel build -c opt //src/llm/experiments/completion_trace

facts_run=/home/ubuntu/checkpoints/memorize_general_facts/compact_batch_32_no_clip_0
bazel-bin/src/llm/experiments/completion_trace/completion_trace \
  --checkpoint="$facts_run/layers_8/step_16128" \
  --tokenizer="$facts_run/inputs/tokenizer" \
  --corpus=testdata/general_facts_dataset.txt \
  --lines=80,406,411,631,1 \
  --prompt_tokens=5 --generation_tokens=5 \
  --layers=8 --model_width=16 --attention_heads=1 --feed_forward_width=64 \
  --source_revision="$(git rev-parse HEAD)" \
  --output_dir=/tmp/five-completion-traces
```

The output directory must not already exist; its parent must exist. The source
revision flag is user-supplied provenance, not an automatic clean-tree check.
Architecture flags must match the checkpoint, particularly head count: tensor
sizes alone cannot distinguish different head partitions. The compact vocabulary
mapping comes from the checkpoint, never from a newly recomputed corpus mapping.
No weights are trained or changed. Generation stops early at EOS.

## What to read

Start with `query_walkthrough.txt`: each decision, the current input tokens,
operation descriptions, complete query-position feature vectors, attention to
each earlier token, and top ten next-token candidates. It is intentionally much
smaller than the full dump and omits earlier-position vectors and full logits.

`report.html` is a self-contained, expandable **complete** numeric report.
`report.txt` contains the same complete layer-output and attention data as plain
text. The default five-fact run makes each about 47 MB; no numeric vectors are
truncated. Both label prompt/generated tokens and compact/original GPT-2 IDs.

Other files:

| File | Contents |
| --- | --- |
| `parameters.txt` | Every FP32 master parameter, named and shaped, in checkpoint order; tied embedding appears once. |
| `vocabulary.tsv` | Compact IDs, original IDs, and escaped token strings. |
| `completions.tsv` | All decisions, corpus correctness, probability, and top-two logit margin. |
| `query_attention.tsv` | Each query's attention probability for every causal key, for each block/head. |
| `residual_updates.tsv` | Before/after/update norms and cosine at attention and MLP residual boundaries. |
| `correlations.tsv` | Pairwise query-vector cosines between facts at the same generation step. |
| `correlation_summary.tsv` | Descriptive same/different predicted-token groups, with raw and centered cosine. |
| `controls.tsv` | Successful instrumentation, autoregression, generation, and weight-immutability checks. |
| `COMPLETE` | Written last, only after every control and output write succeeds; includes corpus match count. |

Generated artifacts remain local and should not be committed. Absence of
`COMPLETE` means an interrupted/failed capture, even if some output files exist.
An incorrect corpus prediction is reported rather than silently replaced with
the gold token; it is not itself an instrumentation failure.

## Reading a forward

Positions are one-based in reports; channels, heads, blocks, and token IDs are
zero-based. To emit output position 6, read the layer vectors at input position
5. The next forward includes the newly generated position 6, whose vector then
predicts position 7. There is no KV cache; the fixed 1,024-slot context is filled
with the active prefix followed only by EOS padding.

For each of the eight pre-LayerNorm blocks, using row vectors:

```text
u = LayerNorm(x)
[Q, K, V] = u W_qkv + b_qkv            # 48 outputs: Q[0:16], K[16:32], V[32:48]
A = causal_softmax(Q K^T / sqrt(16))    # one head for this checkpoint
x = x + (A V) W_o + b_o                # first ResidualLayer output
u = LayerNorm(x)
h = GELU(u W_1 + b_1)                  # 64 coordinates
x = x + h W_2 + b_2                    # second ResidualLayer output
```

The report includes each intermediate layer output, including LayerNorm, packed
Q/K/V, attention, output projection, MLP expansion, GELU, contraction, and residual
addition. The path disambiguates layers: `/attention/FullyConnectedLayer[0]` is
Q/K/V; `[1]` is its output projection. `/mlp/FullyConnectedLayer[0]` expands and
`[1]` contracts. A block's `ResidualLayer[0]` is post-attention and `[1]` post-MLP.
Combinator outputs sometimes duplicate their child's output, intentionally.

At the end, learned final LayerNorm and the tied embedding matrix produce 4,475
logical logits. The physical output has 4,480 lanes: all are included in full
dumps, but padding lanes never participate in softmax or token selection. Softmax
percentages are FP64 report calculations from actual FP32 logits; inference
chooses argmax directly. LayerNorm uses epsilon `1e-5`; GELU uses the model's tanh
approximation, not an alternate exact-erf implementation.

Feature activations are BF16 with FP32 sensitive calculations. Dumps include
both decoded decimal values and original storage bytes. Private kernel scratch
(e.g. tiled accumulators) is not captured. The attention hook reconstructs FP32
probabilities from FlashAttention's saved statistics; the model still executes
its original fused attention path. Those probabilities are not claimed to be
the exact rounded matrix operands used internally by the fused kernel.

## Controls and interpretation

For every decision, all logical logits must be bit-for-bit identical in plain
forwards immediately before and after the hooked forward. Growing the prefix
must preserve all earlier-position activation and attention bytes. Each entire
completion must agree with the normal `GenerateGreedyContinuation` path, and all
unique parameter bytes must remain unchanged. Future padding is never populated
with gold continuation tokens. Unit tests separately change future input tokens
and require every captured causal output to remain unchanged.

The correlations use only the **last active position**. They compare different
facts at the same generation step, grouping by the **predicted next token**,
not the current input token. Centering subtracts a separate per-layer coordinate
mean across all 25 query vectors. Zero-norm comparisons are `NA`.

These are five deliberately chosen, dependent examples, not a statistically
representative dataset. A high attention probability does not establish causal
importance or fact ownership; neither does a large residual update. The same
output token can result from different internal vectors, and similar vectors
can select different tokens. See [FINDINGS.md](FINDINGS.md) for the observations
and follow-up hypotheses, kept separate from the numerical trace.

## Tests

```sh
bazel test -c opt //src/llm/experiments/completion_trace:all
```

Capture tests include real GPT-2 BF16/FP16 inference, all supported physical
storage decodings, multiple heads, malformed hooks/shapes, causal cropping,
unchanged weights, and plain/hooked equivalence. Report tests cover validation,
HTML/text escaping, full values/raw bytes, causal triangles, and known cosine
statistics. FP8 capture is rejected rather than guessing its format/scaling.
