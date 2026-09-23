# How the five-token prompts produce their next token

These measurements use the original **1,024-fact model**, checkpoint
`compact_batch_32_no_clip_0/layers_8/step_16128`, unless stated otherwise.
Each input contains exactly the first five real tokens of a fact. Future
positions contain EOS, never the answer. The requested output is one token,
which may be a word fragment rather than a complete word.

The self-contained local report is `/tmp/one_shot_token_trace_0/report.html`.
It includes every captured layer activation, attention probabilities, the
residual-stream readouts, and individually restored interventions. Its TSV
companions retain the numeric values. This is evidence toward an explanation,
not a complete semantic interpretation of every dimension or parameter.

## The final MLP often converts a prepared representation into an answer

For comparison, feed each residual through the **same trained final LayerNorm
and language-modeling head**. This diagnostic lens does not mean the network
normally produces probabilities at intermediate layers.

| Fact | Sixth token | Lens before final MLP | Actual final probability |
| --- | --- | ---: | ---: |
| France's capital | ` Paris` | 0.000406% | 92.0241% |
| Greece's capital | ` Athens` | 0.000145% | 98.3199% |
| Peru's capital | ` Lima` | 1.7547% | 97.7008% |
| Female mammals | ` nour` | 98.1947% | 98.4679% |
| Rajendra Prasad | `ad` | 0.010207% | 98.7625% |
| Durian | ` for` | 14.7940% | 98.7536% |

The mammals token is already readable before the last MLP, unlike Paris or
Athens. Across **all 1,024 five-token prompts**, bypassing the final MLP leaves
only 268 correct next-token choices, versus 1,024 normally. Restoring the
model reproduces every original logit bit-for-bit. This is a next-token test,
not 268 correctly completed whole sentences.

This does not establish that the last MLP alone stores the facts. Its input
is a representation produced by the preceding network. For France, removing
any one of the sixteen attention/MLP branches changes the answer. Zeroing
these branches affects all five prefix positions while retaining residual
skips. The full-corpus final-MLP bypass above is query-only. Such zeroing is
an out-of-distribution intervention, so necessity under this test is not a
unique storage location.

## Attention supplies context-dependent combinations, not readable answers

For France, Greece, and Peru, the first three tokens are identical: `The`,
` capital`, ` of`. Their Q/K/V values at those three positions are exactly
identical across the three prompts in every block, as causality requires.
Nevertheless, later queries combine those shared values differently.

Block 5's final query assigns ` capital` 96.48%, 92.92%, and 84.41% attention
for France, Greece, and Peru. That shared value cannot itself contain the
identity of a country appearing later in the sequence. Conversely, a low
attention probability does not make a value unimportant: values have
directions and magnitudes, projections combine them, and later layers are
nonlinear.

None of 288 tested individual Q/K/V donor swaps transfers the donor's capital
as the answer. Whole residual swaps can do so, including all six ordered
country-pair swaps at the final block's query residual. These results reject
the simple claim that the largest attention cell directly points to a stored
answer. They do not prove there is no more specific attention mechanism.

The follow-up [routing/value factorial](ATTENTION_ROUTING.md) swaps whole
Q/K and V families at all five rows. It identifies their interaction and a
simpler first-block computation: a scalar country-dependent gate mixes a
fixed shared-word vector with the country's value vector. Later blocks
also change the mixture among fixed anchor directions.

## MLP neurons have shared, relative effects

Two final-MLP channels highlighted by the traces were tested across the full
corpus. Zeroing channel 12 changes three next-token answers; channel 34 changes
35. These neuron interventions affect the final query row only. Their joint
removal changes 35, **not** the union of 38: some effects are
rescued and others appear only jointly.

Nor do their output directions simply spell their apparent capital labels:
Paris ranks 1,084th of 4,475 along channel 12's normalized head-facing direction;
Athens ranks 264th along channel 34's. Directions can help an answer by hurting
a competing token more. The GELU activation's sign and magnitude matter too.
These static projection ranks are neither actual output probabilities nor
causal ablation margins.

## A simpler learned model uses the same architecture differently

A separately trained France/Greece-only model completes both sentences exactly
after 1,024 batch-one updates. In this model, either next-token answer survives
**every individual** attention/MLP branch removal. Paris becomes the lens's
top choice after block 3 attention; Athens after block 0 MLP. Final-MLP removal
barely changes their probabilities. These removals again affect all five
prefix rows while keeping residual skips. This contrast concerns two different
trained computations, not matched training histories or a capacity theorem.

The pair's trace is `/tmp/one_shot_pair_trace_1024_0/report.html`; its causal
branch tests are `/tmp/one_shot_pair_branch_trace_1024_0/report.html`.
See [SINGLE_FACT_DECODING.md](SINGLE_FACT_DECODING.md) for the prospective pair
protocol, exact exposure counts, and complementary weight-history readouts.

## Reliability and remaining question

Ordinary inference and hook-enabled baseline logits agree bitwise. Every
intervention restores the original model; identity patches reproduce the
baseline. Independent CPU reference inference agrees on all tested winners
and ranks, and on all 1,824 compared BF16 query coordinates. The accounting
separates LayerNorm, projection, and BF16 residual-add rounding rather than
silently treating them as exact real arithmetic.

The useful common theme is **context preparation followed by a nonlinear,
shared decision**, not a one-fact/one-weight lookup. The remaining challenge
is to describe the prepared features and their interactions constructively:
why these particular weights create them, which smaller combinations are
sufficient, and how that description scales beyond a few traced examples.

To regenerate the full six-prompt study:

```sh
bazel build -c opt //src/llm/experiments/one_shot_memorizer:checkpoint_token_trace
facts=/home/ubuntu/checkpoints/memorize_general_facts/compact_batch_32_no_clip_0
bazel-bin/src/llm/experiments/one_shot_memorizer/checkpoint_token_trace \
  --checkpoint="$facts/layers_8/step_16128" \
  --tokenizer="$facts/inputs/tokenizer" \
  --lines=80,406,411,1,258,631 --output_dir=/tmp/five_token_trace_new
```

The default includes branch and capital-donor interventions. The separate
full-corpus and selected-neuron follow-ups are documented with their local
evidence in the main [README](README.md).
