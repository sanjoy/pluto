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

## Immediate query versus later country rereading

The first-block gate formula describes the query's update, but a whole Q/K/V
family swap also changes the country position's attention output. Later
blocks can reread that altered country trajectory. To separate these two
routes, retain the same three capitals and six ordered donor/recipient pairs.
For each nonidentity family corner (Q/K only, V only, both), capture its actual
BF16 block-0 attention output **before the output projection**. Then run four
independent recipient forwards, copying from that captured output:

* neither position;
* country position 3 only;
* query position 4 only;
* both positions.

This fixes 6 pairs times 3 family corners times 4 row selections: **72
conditions**, without an outcome-driven search. Patches keep the recipient's
residual skip. No answer tokens enter any forward. The first three positions
must remain bit-identical across the source corners, and copying both changed
positions must reproduce the original whole-family corner's final logits
bit-for-bit. Neither-row and identity controls must reproduce the recipient.
Record all real-vocabulary logits, recipient/donor probabilities and margins,
and winners; verify original source weights remain unchanged.

This is a selected, off-distribution causal intervention. It distinguishes
the immediate query route from the country state available to later layers;
it does not assign exclusive fact ownership or interpret a gate as a capital
label. The deadline is one hour, and the earlier 192-condition factorial
should remain reproducible without changes to its existing numeric reports.

### Completed result: the country route carries these answer failures

The fixed 72-condition experiment completed with all 301 controls passing.
The table counts interventions retaining the **recipient's original answer**,
out of six ordered capital pairs per cell:

| Captured source corner | Neither row | Country row only | Query row only | Both rows |
| --- | ---: | ---: | ---: | ---: |
| Donor Q/K, recipient V | 6/6 | 3/6 | 6/6 | 3/6 |
| Recipient Q/K, donor V | 6/6 | 6/6 | 6/6 | 6/6 |
| Donor Q/K and V | 6/6 | 3/6 | 6/6 | 3/6 |

None of the interventions produces the donor's answer. Every query-only
intervention preserves the original answer, though its probability can
change substantially. This does not mean the immediate query update is
irrelevant or that removing it would be harmless: the answer survived these
particular donor perturbations in the otherwise unchanged recipient computation.

Country-only and both-row interventions have **the same six failures**, not
just equal counts. In each of the Q/K-only and joint Q/K/V corners, the
failing ordered pairs are France receiving Peru, Peru receiving France,
and Peru receiving Greece. The other 12 family/pair combinations succeed
under both row selections. Winners and probabilities need not match between
country-only and both, so this is equality of answer-failure sets, not
equality of their computations.

For example, at the France prompt with Peru's Q/K corner:

| Attention output rows replaced | Paris probability | Winner |
| --- | ---: | --- |
| Neither | 92.0241% | ` Paris` |
| Query only | 68.1528% | ` Paris` |
| Country only | 24.1265% | ` north` |
| Both | 15.7697% | ` first` |

At block 0 the query Q vector and all non-country K/V vectors are fixed.
Thus the Q/K-only query-row intervention changes the scalar gate in the
gated-vector formula above, using the actual rounded attention output.
The country-row intervention instead changes a representation that later
blocks can reread. For these observed failures, perturbing that country
trajectory is sufficient to break the answer without changing block 0's
immediate query update. It is not evidence that the country row exclusively
stores the capital or that a gate alone represents a label. The source
corner and recipient residual still form an off-distribution hybrid.

### Row-intervention checks and reproduction

The 301 checks include:

* 9 same-source row-mask identities;
* 18 unchanged first-three-row source attention outputs;
* 72 exact selected-row output-byte checks;
* 72 unchanged recipient Q/K/V prefix checks during row-only replay;
* 18 neither-row forwards matching recipient logits;
* 18 both-row forwards matching the original whole-family corner logits;
* 72 ordinary post-intervention forwards matching the recipient;
* 18 unchanged captured donor buffers;
* 3 capture-versus-ordinary baseline checks and one unchanged full-master snapshot.

The targeted trace tests also check both BF16 and FP32 two-row donor
composition, authoritative signed-zero bytes, unchanged other prefix and
future-padding rows, equivalence to a merged donor, and source/donor
preservation. All 13 trace tests passed. A fresh run of the original
192-condition default protocol reproduced all eight numeric/control/logit
artifacts and its manifest byte-for-byte.

The local result is `/tmp/gpu_attention_b0_rows_0`. `conditions.tsv` includes
the explicit `row_condition`; `vectors.tsv` contains the actual source-corner
attention rows. `query_logits.f32` holds all 72-by-4,475 real-vocabulary logits
in condition order. Independent readback verified finite values, every winner,
and both target probabilities and strongest-rival margins. Generated reports
remain outside Git.

```sh
bazel build -c opt //src/llm/experiments/one_shot_memorizer:checkpoint_attention_factorial_probe
facts=/home/ubuntu/checkpoints/memorize_general_facts/compact_batch_32_no_clip_0
bazel-bin/src/llm/experiments/one_shot_memorizer/checkpoint_attention_factorial_probe \
  --checkpoint="$facts/layers_8/step_16128" \
  --tokenizer="$facts/inputs/tokenizer" \
  --protocol=b0_rows \
  --output_dir=/tmp/attention_b0_rows_new
```

## Prospective follow-up: late query/country compatibility

An isolated donor-query residual immediately after block 6 produces a third
answer in all six ordered capital-pair swaps in the original trace. This
suggests testing a matched versus mismatched query/country interface rather
than assuming that the query already carries an independently readable answer.

Keep that post-block-6 donor query patch fixed. At block 7's Q/K/V projection,
independently copy the donor's country-position K and V slices: neither, K
only, V only, or both. These are **24 fixed conditions** across six ordered
pairs. Keep the country Q and its residual trajectory otherwise unchanged;
they do not affect the final query after this last attention operation.

The first three anchor positions are shared, and the query's own Q/K/V and
residual skip now come from the donor. Consequently copying both country K
and V must reproduce the donor's final-query attention output and all 4,475
final logits bit-for-bit. Check that control explicitly, along with identity
patches, ordinary inference restoration, and unchanged source weights. Save
probabilities, actual attention outputs, final-MLP input/output and all logits.
Compare the query-only conditions with the existing trace before interpreting
new conditions.

This isolates routing/content compatibility at a selected late interface. A
successful donor completion need not identify where a fact is stored, and a
failed hybrid is not evidence that either component lacks donor information.
Do not tune positions, channels or donor pairs based on the outcomes. Budget:
one hour; the GPU run may overlap the independent block-0 experiment.

### Late compatibility results

The fixed protocol completed without changing the selected pairs, positions,
or channels. Every condition below already has the donor's query residual
after block 6; the columns describe the additional country-row repair in
block 7. `Recipient ← donor` identifies the original prompt and the source
of the copied activations. Quoted winners retain the token's leading space.

| Recipient ← donor | Neither K nor V | K only | V only | Both K and V |
| --- | --- | --- | --- | --- |
| France ← Greece | `" full"` | `" Finland"` | `" Great"` | `" Athens"` |
| France ← Peru | `" same"` | `" Spain"` | `" same"` | `" Lima"` |
| Greece ← France | `" time"` | `" Paris"` | `" Paris"` | `" Paris"` |
| Greece ← Peru | `" same"` | `" joined"` | `" Lima"` | `" Lima"` |
| Peru ← France | `" body"` | `" Paris"` | `" bones"` | `" Paris"` |
| Peru ← Greece | `" preserving"` | `" preserving"` | `" Athens"` | `" Athens"` |

The donor answer wins in **0/6, 2/6, 3/6, and 6/6** conditions respectively;
every other winner is a third answer, not the recipient's original answer.
Thus neither K-only nor V-only repair is uniformly sufficient in these tested
hybrids. Both nevertheless succeed individually in some pairs: this is not
a claim that routing or values can never suffice on their own.

Two pairs make the interaction especially clear:

* **France ← Greece:** the probability of `" Athens"` is 0.7800% with the
  query patch alone, 0.01756% after K-only repair, 16.1736% after V-only repair,
  and 98.3199% after both. The V-only hybrid narrowly loses to `" Great"`:
  Athens's margin against its strongest rival is −0.019995 logits. Both
  repairs restore the donor's +5.424347 margin. K-only repair therefore makes
  the donor answer less likely in this particular mismatched context even
  though it restores the donor's attention routing exactly.
* **France ← Peru:** the probability of `" Lima"` is approximately
  3.43 × 10⁻⁷ with neither repair, 2.39 × 10⁻¹² with K only, 1.16 × 10⁻⁵
  with V only, and 0.977008 with both. The respective strongest-rival margins
  are −14.7868, −25.5426, −10.9049, and +5.7332 logits. Neither separate
  repair restores the answer, whereas the matched pair does.

These are non-additive changes in the final answer distribution, not an
additive allocation of factual content between K and V. They include the
attention calculation, residual addition, final MLP, LayerNorm and head.
The **both-K/V donor match is an expected structural control**, not by itself
the scientific finding: all inputs to the final query's attention operation
and its residual skip have then been made donor-identical. The informative
contrast is the variable behavior of the incomplete repairs, including the
cases where an individually exact routing repair worsens the donor answer.

### Late compatibility checks, artifacts and reproduction

All **148 controls** passed. They cover ordinary-versus-captured baseline
agreement, all 12 same-source patch identities, shared anchor rows, exact
selected residual/Q/K/V bytes, donor query routing after K repair, unchanged
probabilities after V repair, and restoration of donor query attention,
final-MLP normalized input and GELU activations after both repairs. In all
six both-repair cases, **every one of the 4,475 final logits** equals the
donor's original logits bit-for-bit. All 24 ordinary post-intervention
forwards reproduce the recipient, and the full source-weight snapshot is
unchanged. The six query-only conditions exactly reproduce the winners,
target probabilities, target margins and donor probabilities in the earlier
`/tmp/one_shot_token_trace_0/conditions.tsv` report.

The complete local result is `/tmp/query_country_compatibility_0`:

* `conditions.tsv` records all 66 forwards, including controls, with both
  target probabilities and margins against each target's strongest rival.
  `logits.f32` contains 4,475 FP32 logits per row in that same order.
* `probabilities.tsv` contains the captured attention matrices;
  `activation_values.tsv` contains decoded activation coordinates.
  `captures.tsv` indexes the corresponding exact bytes in
  `activation_bytes.bin`.
* `controls.tsv` records the 148 checks, and `manifest.tsv` records the fixed
  protocol, source paths, output layout and completed status.

Independent raw-file readback checked all 295,350 logits for finiteness and
recomputed every winner, probability and margin with zero discrepancy.
It also verified 45 complete baseline/identity/post/both-repair logit rows
against their expected original rows bit-for-bit. The optimized build and
uncached targeted trace tests passed; the subsequent full repository suite
passed all 106 test targets. Generated experiment artifacts remain outside
Git.

```sh
bazel build -c opt //src/llm/experiments/one_shot_memorizer:checkpoint_query_country_probe
facts=/home/ubuntu/checkpoints/memorize_general_facts/compact_batch_32_no_clip_0
bazel-bin/src/llm/experiments/one_shot_memorizer/checkpoint_query_country_probe \
  --checkpoint="$facts/layers_8/step_16128" \
  --tokenizer="$facts/inputs/tokenizer" \
  --output_dir=/tmp/query_country_compatibility_new
```

This is a three-fact, one-token, off-manifold intervention study. It shows a
late query/country compatibility requirement in these examples; it does not
show exclusive ownership of a fact, identify individual factual weights, or
establish how arbitrary prompts behave. The fixed prompt contains only the
first five tokens; future inputs are EOS padding. Expected answers are used
only to score the resulting logits, never supplied to the forward pass.
