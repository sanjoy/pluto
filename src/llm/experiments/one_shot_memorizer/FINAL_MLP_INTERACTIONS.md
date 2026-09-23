# Redundancy, biases, and transferring feature amplitudes

This follows [FINAL_MLP_SUBSETS.md](FINAL_MLP_SUBSETS.md). The model and task
are unchanged: original 1,024-fact checkpoint `step_16128`, five input tokens,
one next-token prediction. Interventions affect the final MLP at the query
position only. Earlier computation and the residual skip remain intact.

## Sufficiency does not imply necessity

The preceding search found a successful mask with at most two retained GELU
channels for 680 facts. A complementary experiment starts with **all 64**
channels and removes each one separately, restoring it before the next test.

Of 65,536 individual channel deletions, **258** change the answer, affecting
**191 distinct facts**. The other **833 facts survive every individual
deletion**. This does not mean those facts do not use the MLP: for **553** of
them, removing all GELU contributions at once changes the answer. They have
collective or redundant support that no single deletion exposes.

The most disruptive single channels are 35 (39 facts), 34 (35), 9 (24),
57 (16) and 56 (15). These counts measure sensitivity to this intervention,
not an allocation of exclusive fact ownership.

There are also **12 facts** whose answer survives removing all GELU features,
yet fails after one particular deletion from the full vector. Correctness
is therefore not monotone in the retained feature set. A channel can repair
another channel's harmful effect, and deleting both can restore the answer.
This is why checking only one-at-a-time removals cannot certify a globally
minimal sufficient set or even that every proper subset fails.

A concrete example is line 106: `Table salt consists mainly of` → ` the`.
The full vector assigns the target 99.86%; zeroing **all** GELU features
assigns it 99.998%. But deleting **only channel 16** from the full vector
makes ` a` win, reducing the target to 33.88%. The contraction bias and
incoming residual remain unchanged in all three conditions. Calling channel
16 globally necessary would therefore be wrong.

## Separate the learned biases from the activated features

The audit also constructs two private model copies: one has only the final
MLP contraction bias zeroed, the other only the final LayerNorm beta zeroed.
Both use the **original** captured GELU and incoming residual. Checkpoint
files and the original model are never edited.

| Final-tail parameters | All GELU features retained | All GELU features zeroed |
| --- | ---: | ---: |
| Original | 1,024 correct | 292 correct |
| Contraction bias zero | 1,019 | 268 |
| Final LayerNorm beta zero | 1,019 | 294 |

The bias helps some contexts and hurts others. In the zero-feature condition,
retaining the contraction bias repairs **47** answers but breaks **23**,
giving the net increase from 268 to 292. Retaining final LayerNorm beta
repairs 50 and breaks 52, giving the net decrease from 294 to 292. Looking
only at a difference in totals would hide these opposing effects.

Sensitivity also depends on how close a decision already is. Line 795,
`Condensation occurs when a` → ` gas`, starts at only 50.84% probability,
with margin +0.1423 against its strongest rival. Eleven individual channel
deletions break this decision. Removing the contraction bias actually improves
gas to 72.27%; removing final LayerNorm beta makes ` liquid` win. These are
not eleven exclusive “gas cells.”

The strongest competitor can change as well. Deleting channel 35 leaves gas
ahead of the original rival ` liquid` by +0.0447, but ` larger` becomes the
new winner, with gas trailing it by -0.4747. The audit records both the fixed
baseline-rival margin and the current strongest-rival margin so that this
switch cannot be mistaken for success.

The entire audit consists of 71,680 GPU conditions. All 1,024 original-tail
identity comparisons and 42 preselected direct full-model counterfactual
comparisons reproduce every real-vocabulary logit bitwise. The full-model
controls cover the original six traced facts, original empty-GELU and
channel-12/channel-34 deletions, and both affine-parameter variants.
The checked-in feature-audit tool reproduces the entire 71,680-row numeric
report byte-for-byte, adding 1,024 ordinary source-model post-controls and
checks that the private clones differ only in the intended affine tensor.

## Do shared feature amplitudes carry a transferable answer?

A shared successful channel set does not mean a shared numerical activation:
each prompt supplies different GELU amplitudes and a different incoming
residual. To separate these, eight pairs were fixed **before inspecting any
swap outcomes**, using the completed minimum-mask search. Six pairs have
different answers; two are same-answer controls.

| Retained channels | First next-token target | Second next-token target | Corpus lines |
| --- | --- | --- | --- |
| 34 | ` Sun` | ` geological` | 43, 110 |
| 56 | ` gravity` | ` flavor` | 10, 412 |
| 35 | ` butter` | ` 360` | 583, 542 |
| 7,34 | ` Babylon` | ` thunder` | 312, 499 |
| 34,35 | ` circle` | ` DNA` | 102, 493 |
| 34,56 | ` Lima` | ` plugs` | 411, 939 |
| 34 | ` Sun` | ` Sun` | 43, 138 |
| 34 | ` same` | ` same` | 40, 848 |

Each shared set is a minimum-size successful mask for both original queries.
For each direction, replace the selected recipient GELU coordinates with the
donor's exact BF16 coordinates while keeping the recipient's residual. Test
two backgrounds:

- **Sparse:** all unselected GELU coordinates are zero.
- **Intact:** all unselected coordinates keep the recipient's original values.

Self-donor cases and a common mask panel are controls. The experiment runs
184 cases: 120 panel entries (15 prefixes times eight distinct masks,
including empty and full masks) and 64 pair/background cases. The latter
include both self-donors and cross-donors. All 184 cached results match fresh
full-model patched forwards **bit-for-bit at all 4,475 logits**; unpatched
post-controls, 30 identity rows and unchanged source-weight bytes also pass.

For the actual cross-donor cases:

| Pair type / background | Keeps recipient answer | Switches to donor answer | Third answer |
| --- | ---: | ---: | ---: |
| Different answers / sparse | 8 of 12 | 0 | 4 |
| Different answers / intact | 12 of 12 | 0 | 0 |
| Same-answer controls / both backgrounds | 8 of 8 | not distinguishable | 0 |

For example, swapping the `{34,56}` amplitudes between Lima and ` plugs`
does not exchange their answers. In the sparse background the two recipients
instead produce ` generates` and ` made`; in the intact background both keep
their own answer.

This rejects the simple standalone-answer-label interpretation for these
selected swaps. It does **not** prove that the features contain no information
about the answer, that no other transfer could work, or that these deliberately
selected off-distribution interventions describe every fact in the corpus.

## A more precise account of the computation

Ignoring rounding for intuition, the final residual is
`incoming_residual + contraction_bias + sum_j(GELU_j * W2_j)`.
The actual probe keeps the model's BF16/FP32 arithmetic. Its final LayerNorm
and embedding-based head decide among **competing** tokens, not whether a
particular neuron resembles the desired word.

The evidence supports a context-dependent, shared computation: the earlier
network prepares a residual and activation pattern; several combinations of
output directions can push that state across the relevant decision boundary.
The same direction may help many different answers by changing their margins
against different rivals. Redundancy in the complete vector can absorb an
individual deletion or a donor amplitude, while sparse interventions reveal
the conditional combinations.

This is more specific than calling a neuron “the Lima neuron,” but it is
still not a construction of the learned feature detectors from raw training
sentences. Understanding that upstream preparation and its training dynamics
remains unresolved.

## Local evidence

To reproduce the corpus-wide deletion and bias audit:

```sh
bazel build -c opt //src/llm/experiments/one_shot_memorizer:checkpoint_mlp_feature_audit
facts=/home/ubuntu/checkpoints/memorize_general_facts/compact_batch_32_no_clip_0
bazel-bin/src/llm/experiments/one_shot_memorizer/checkpoint_mlp_feature_audit \
  --checkpoint="$facts/layers_8/step_16128" \
  --tokenizer="$facts/inputs/tokenizer" \
  --lines=all --output_dir=/tmp/mlp_feature_audit_new
```

The frozen donor study has its own executable. Its named eight-pair protocol
is in the source, with expected compact token IDs checked against the corpus;
it is deliberately a replication of these selected cases, not a new search.

```sh
bazel build -c opt //src/llm/experiments/one_shot_memorizer:checkpoint_mlp_donor_probe
bazel-bin/src/llm/experiments/one_shot_memorizer/checkpoint_mlp_donor_probe \
  --checkpoint="$facts/layers_8/step_16128" \
  --tokenizer="$facts/inputs/tokenizer" \
  --output_dir=/tmp/mlp_donor_probe_new
```

Both tools require fresh output directories. The donor row construction has
CPU tests for sparse/intact backgrounds, self-donors, signed zeros/NaN bits,
the 64th mask bit, source immutability and malformed arguments. Numerical GPU
equivalence is checked by the executable itself for **every** condition.
All 98 repository test targets passed with fresh execution. The production
donor executable reproduces the prototype's complete conditions, amplitudes
and fact metadata files byte-for-byte, including every probability and margin.

- `/tmp/gpu_final_mlp_corpus_audit_0/`: all independent deletions and bias controls.
- `/tmp/mlp_feature_audit_production_all_0/`: checked-in audit and expanded controls.
- `/tmp/gpu_mlp_donor_protocol_0/`: frozen donor experiment, amplitudes and full-model controls.
- `/tmp/mlp_donor_production_0/`: checked-in donor executable's independent reproduction.
- `/tmp/subset_donor_protocol_0.tsv`: the eight pairs fixed before the swaps.
- `/tmp/subset_corpus_analysis.md`: independently checked mask-reuse counts and protocol.

These generated artifacts remain local, not in Git.
