# Routing, values, and their interaction in three capital completions

This extends the [five-token execution study](EXECUTION_TRACE.md) on the
original memorized checkpoint, not a newly trained or two-fact model. It
tests the prompts `The capital of France is`, `The capital of Greece is`,
and `The capital of Peru is`. Each is exactly five tokens; the corresponding
sixth token is ` Paris`, ` Athens`, or ` Lima`.

## Intervention, not just a visualization

At each of eight blocks, for every ordered pair of these three prompts,
capture the packed Q/K/V projection and perform four independent forwards:

| Corner | Q and K | V |
| --- | --- | --- |
| 00 | Recipient | Recipient |
| 10 | Donor | Recipient |
| 01 | Recipient | Donor |
| 11 | Donor | Donor |

Replace the selected family at **all five prefix positions**. Preserve the
recipient's residual skip and every other layer. These are whole-family
interventions, not edits to one attention cell or one query. Earlier prefix
positions can change too, and subsequent attention can read those changes.
No answer or later gold token is supplied to the forward pass.

The original FlashAttention path remains in use. A read-only additional pass
reports its probability matrix from the same Q/K inputs and stored softmax
statistics. We capture actual BF16 attention outputs before the output
projection, projected outputs, post-attention residuals, and final logits.

## Routing and value changes are not interchangeable

There are 48 ordered-pair/block combinations per nonidentity condition:

| Intervention | Recipient's answer | Donor's answer | Another answer |
| --- | ---: | ---: | ---: |
| Q/K only | 37 | 0 | 11 |
| V only | 32 | 0 | 16 |
| Q/K and V together | 24 | 1 | 23 |

The one donor-answer transfer is France receiving Greece's full Q/K/V at
block 6. ` Athens` wins with probability 19.9222%, versus 6.7225% for
` Paris`; its logit margin over its strongest competitor is only 0.006716.
The result is a weak winner, not a confident transplant of the original
Greece computation.

A particularly clear distinction occurs at **block 7, France receiving Peru**:

- Unchanged: ` Paris` has probability 92.0241%.
- Donor Q/K only: ` Paris` falls to 0.1373%; ` bones` wins.
- Donor V only: ` Paris` remains the winner at 91.8106%.
- Both: ` body` wins; ` Lima` has probability about 0.000982%.

Here changing routing has a much greater effect on the answer than changing
values alone. This is not a corpus-wide ranking of Q/K versus V importance:
across all 48 tested interventions, V-only changes actually break more
recipient answers. The outcome depends on the block and context.

## Why a shared word can receive country-dependent attention

For these three prompts, the first three positions (`The`, ` capital`,
` of`) have bit-identical Q/K/V in every block. They cannot acquire the
identity of the later country: attention is causal. Nonetheless, the final
query can assign those **fixed value directions** different coefficients.

At block 7, France's query gives `The` about 96.06% probability and its country
position about 3.15%; Peru gives them about 18.11% and 81.04%, respectively.
The most-attended word is therefore not necessarily the location of the
answer. Fixed anchor directions can participate in a context-dependent
computation through their routing coefficients. Conversely, a large
probability is not by itself a large effect after value magnitudes, output
projection, residual addition, and downstream nonlinear transformations.

### Block 0 reduces to a gated country-vector transport

There is a stronger simplification before the first block: the final ` is`
position is fixed too. Only the country token differs. Therefore the query
Q vector and all **non-country** K/V vectors are identical across these
prompts. In real arithmetic its attention update is

```
o(country) = [(1-g(country)) C + g(country) v(country)] W_O + b_O
```

`C` is a fixed mixture of the values for `[The, capital, of, is]`; `v(country)`
is the country position's value vector. The scalar gate is

```
s_country = dot(q_is, k_country) / sqrt(16)
S_fixed   = sum over non-country j of exp(dot(q_is, k_j) / sqrt(16))
g(country) = sigmoid(s_country - log(S_fixed))
```

Here `k_country` is an affine projection of the learned LayerNorm of
`embedding(country) + position_embedding(3)`. The observed gates are
0.42237449 for France, 0.21370986 for Greece, and 0.14561325 for Peru.
The fixed mixture coefficients in `C` are approximately
`[0.001857411, 0.09635504, 0.38498638, 0.51680118]`; their observed
country-to-country discrepancies are at most `2.4e-8` from probability
rounding. The query's residual skip is another fixed vector at this block.

This describes gated transport of a **country-dependent vector**, not an
answer encoded in a single gate. It does not assert that the country value
is directly readable as a capital. The formulas are exact in real arithmetic;
bit-exact reproduction additionally requires the production BF16/FP32
rounding and accumulation order.

### Later computation has two changing positions

For this fixed five-token template, the three anchor trajectories can be
precomputed. Only the country residual `c_l` and query residual `s_l` depend
on which country is supplied. Each block has the triangular form

```
c_(l+1) = F_l(c_l)
s_(l+1) = G_l(s_l, c_l)
```

`F_l` includes attention to the anchors and country itself, then the original
residual/pre-LN MLP. `G_l` can also attend to `is`. Both use the **block-input**
country state; the query does not read the country state after that block's
MLP. Causality prevents the country trajectory from depending on the later
query. This is a simplification of this template's execution, not a reduction
of the trained model's parameter count or a dataset-only construction.

At later blocks, the country-conditioned query changes the mixture among
fixed anchors as well as their total mass. For example, these are the total
anchor probabilities for `[The, capital, of]`:

| Block | France | Greece | Peru |
| --- | ---: | ---: | ---: |
| 0 | 27.91% | 37.99% | 41.28% |
| 1 | 59.48% | 62.87% | 68.61% |
| 2 | 59.77% | 59.41% | 62.33% |
| 3 | 94.46% | 81.13% | 89.47% |
| 4 | 85.41% | 62.42% | 51.51% |
| 5 | 99.33% | 96.19% | 90.22% |
| 6 | 48.33% | 46.19% | 55.05% |
| 7 | 96.61% | 45.85% | 18.45% |

Within the anchors, block 0 always uses about `[0.38%, 19.94%, 79.67%]`.
By block 6, `The`'s conditional share is 40.51%, 43.77%, and 53.07%,
respectively: later attention changes the direction mixture, not just a
single common gate. The original execution capture also confirms 16,416
cross-country scalar comparisons of the first three positions with zero
differences, and bit-identical block-0 query Q values.

Mongolia also has a single-token country in this template, but its capital
starts with the token ` U` and requires several subsequent tokens. These
three-case attention statistics have not been measured for Mongolia and
should not be presented as a four-case or whole-capital-word study.

## Separate the interaction rather than assigning misleading percentages

Let `A00`, `A10`, `A01`, and `A11` be the **actual stored GPU attention
vectors** at the final query for the four corners. Define:

```
routing     = A10 - A00
values      = A01 - A00
interaction = A11 - A10 - A01 + A00
total       = A11 - A00 = routing + values + interaction
```

This finite-difference identity includes the observed BF16 outputs. It does
not linearize downstream logits. The vector norms are not additive ownership
fractions: the components can cancel.

For France receiving Peru at block 7, the pre-projection norms are:

| Vector | Norm |
| --- | ---: |
| Total change | 1.69238 |
| Routing | 1.16900 |
| Values | 0.08616 |
| Interaction | 2.15062 |

The interaction norm exceeding the total is consistent with cancellation,
not a percentage greater than 100% of an answer. Projecting these difference
vectors through the effective BF16 output matrix in FP64 gives norms
0.82179, 0.44002, 0.04000, and 0.99788, respectively. This last calculation
is an analytical projection, not a claim that it reproduces every BF16
rounding in the observed projection layer.

For comparison, ideal real-arithmetic `pV` obeys

```
delta(pV) = (p_d-p_r) V_r + p_r (V_d-V_r)
            + (p_d-p_r) (V_d-V_r).
```

The tool reports these terms separately from the GPU finite differences.
Reconstructing `pV` on the CPU from reported probabilities is not bit-exact
FlashAttention: the norm of its closure residual is 0.002938 for the example,
about 0.174% of the observed total change. Across the 48 cases, this ratio
ranges from about 0.124% to 4.98%. Causal answer counts use actual forwards,
not the approximate accounting.

## Checks and scope

The initial bounded run passed 556 exact controls:

- 192 probability matrices match the selected Q/K source.
- 48 full-Q/K/V attention outputs match the donor's BF16 bytes at all five rows.
- 24 same-source full-Q/K/V patches preserve every final logit.
- 192 ordinary post-intervention forwards preserve every original logit.
- 48 first-three-position Q/K/V comparisons are bit-identical.
- 3 hook-enabled captures match ordinary inference.
- 48 unmodified factorial corners match ordinary inference.
- One complete source-master snapshot remains unchanged.

These controls do not make off-manifold donor swaps ordinary model behavior.
In particular, replacing the attention branch while retaining the recipient
skip intentionally constructs a hybrid state. Failed answer transfer does
not prove that the patched tensors contain no information about the donor.

The standalone production tool reproduced all seven original TSV artifacts
byte-for-byte in `/tmp/gpu_attention_factorial_production_0/`, compared with
the bounded prototype at `/tmp/gpu_attention_factorial_0/`. The production
manifest records all control categories explicitly. It also saves all
192-by-4,475 final logits as raw FP32 values in condition-row order; an
independent readback verified finite values, every winner, both target
probabilities, and both strongest-rival margins. Generated reports remain
outside Git.

```sh
bazel build -c opt //src/llm/experiments/one_shot_memorizer:checkpoint_attention_factorial_probe
facts=/home/ubuntu/checkpoints/memorize_general_facts/compact_batch_32_no_clip_0
bazel-bin/src/llm/experiments/one_shot_memorizer/checkpoint_attention_factorial_probe \
  --checkpoint="$facts/layers_8/step_16128" \
  --tokenizer="$facts/inputs/tokenizer" \
  --output_dir=/tmp/attention_factorial_new
```

The common theme is a **context-dependent combination of shared directions**,
followed by residual and nonlinear processing. These experiments narrow the
mechanism; they do not yet derive the learned attention matrices from text.
