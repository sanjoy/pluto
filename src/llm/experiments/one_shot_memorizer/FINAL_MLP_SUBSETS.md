# What small final-MLP feature combinations can do

For the complementary deletion, bias and donor-amplitude experiments, see
[FINAL_MLP_INTERACTIONS.md](FINAL_MLP_INTERACTIONS.md).

This follow-up to [EXECUTION_TRACE.md](EXECUTION_TRACE.md) uses the original
1,024-fact model, `compact_batch_32_no_clip_0/layers_8/step_16128`. Each query
is exactly the first five tokens of its fact; the requested answer is its
sixth token, not necessarily a complete word. It asks a conditional question:
**given the representation already prepared by the earlier layers, which
final-MLP features suffice to select the correct next token?**

## The intervention and its limits

At position four (zero-based), retain a subset of the final MLP's 64 GELU
outputs and replace the others with BF16 positive zero. Preserve the exact
upstream residual, contraction weights **and bias**, residual addition, final
LayerNorm, and tied language-modeling head. Use the actual GPU kernels and
their BF16 rounding, not a linear approximation to the output probability.

This is not a whole-branch bypass: a zero-feature MLP still contributes its
learned contraction bias. Nor is it a model-wide pruning result: each prompt
has a different incoming state, and the mask is selected using that prompt's
known target. Labels score interventions but never enter the network input.
Success establishes neither whole-sentence recall nor exclusive fact storage.

The initial independent check ran the **entire GPU model** for each mask:
8,344 masked conditions, each followed by an unmodified forward that had to
restore every real-vocabulary logit bit-for-bit. Full-mask identity patches
also reproduced the baseline. All zero-, one-, and two-feature possibilities
were enumerated until the first successful cardinality; counts are 1, 64,
and 2,016 respectively.

## Six traced facts

The alternative masks below came from forward-greedy and backward-removal
searches on an independent CPU reference, then were validated with the full
GPU model. All listed channels are zero-based final-MLP channel indices.

| Fact / target token | Small successful mask | Alternative | Proven minimum cardinality |
| --- | --- | --- | --- |
| Female mammals / ` nour` | none | — | 0 |
| France / ` Paris` | 9,12,21,37,39,50 | 12,13,21,35,37,39,44 | between 5 and 6 |
| Rajendra Prasad / `ad` | 7,26,29,33,39,41,45 | 29,33,34,37,38,39,41,45,49,54,55,58,59,60,61 | between 5 and 7 |
| Greece / ` Athens` | 9,12,21,28,34 | 9,28,34,37,52 | **5** |
| Peru / ` Lima` | 34,56 | 35,56 | **2** |
| Durian / ` for` | none | — | 0 |

Only the exhausted cardinalities justify the lower bounds. Greedy search
does not prove a global minimum. Checking every single-channel removal from
a mask is also weaker than checking every proper subset: this system is
nonmonotone, so removing two features can rescue an answer that removing one
loses.

A deeper production-GPU search exhausted every subset through size four for
Paris, Athens and `ad`: **2,037,363 conditions**, with no successful mask.
Together with the independently validated five-channel Athens masks, this
proves its minimum is five. Paris and `ad` remain bounded, not solved exactly.

For Durian, retaining the projection bias with zero GELU features gives
` for` 50.09% probability and the correct top choice. Bypassing the entire
MLP, bias included, gives 14.79% and a wrong top choice. For mammals, the
zero-feature, bias-retaining intervention still gives ` nour` 97.16%.

## Lima: relative competition rather than a dedicated word neuron

Exactly two of all 2,016 pairs succeed. No single feature does.

| Retained features | Lima probability | Lima minus strongest rival logit |
| --- | ---: | ---: |
| none | 3.951% | -1.9892 |
| 34 | 9.184% | -1.0760 |
| 35 | 6.190% | -0.9732 |
| 56 | 16.014% | -0.7602 |
| 34,56 | 36.259% | +0.6456 |
| 35,56 | 19.296% | +0.3574 |
| all 64 | 97.701% | +5.7332 |

The pair `{35,56}` actually **lowers Lima's own logit**, from 24.304 to
23.594. It wins by suppressing the competing tokens more. This is why a
readout based only on which vocabulary word a channel points toward misses
the mechanism. Along the static, centered, final-LayerNorm-scaled output
directions of channels 34, 35 and 56, Lima ranks only 791st, 1,478th and
1,691st out of 4,475 tokens. These direction scores are not probabilities;
they omit the actual GELU activations, incoming residual, normalization scale,
and biases.

Against the pair's rival ` image`, channel 35 contributes +1.142 and channel
56 contributes +1.514 to an accounting using that pair's final normalization.
Together they overcome a negative contribution from the incoming residual,
with additional support from contraction bias and final LayerNorm beta.
Explicit projection, residual-add and final-normalization rounding terms
close the accounting. These terms sum to the observed margin; they are **not**
individual causal removal effects, because normalization changes on removal.

Channel 56 is not necessary in the intact model: removing it alone leaves
Lima correct at 61.64%. Even forbidding it entirely, the three-feature set
`{29,34,35}` succeeds at 24.90% with margin +0.4251. The prior exhaustive
zero/one/two search proves that three is minimal **under this restriction**.
A separate six-feature alternative `{28,33,34,35,37,42}` also succeeds, at
20.23%. Both alternatives were validated with the entire GPU forward and
bitwise restoration controls.

Thus a feature can occur in every smallest sufficient explanation without
being necessary to the original computation. Multiple combinations provide
similar relative support against rivals.

## Corpus-wide check

Before running the expanded sweep, the protocol is fixed as follows:

1. Use all 1,024 first-five-token prompts and verify their unmodified sixth
   tokens are correct.
2. Exhaust all subsets of sizes zero, one and two, stopping a fact only after
   completing its first successful size. Report every successful mask at that
   size, not just the first one encountered.
3. Replay the final contraction, residual addition, normalization and head
   in batches from captured raw BF16 GELU and incoming residual values. Earlier
   layers are unchanged and run once per prompt. The final tail is pointwise
   across positions, so those independent counterfactuals can share a batch.
4. At each actual batch geometry, include a full-mask control that must match
   **all** original real-vocabulary logits bitwise. First cross-check the
   six-fact batched enumeration against the independent full-model enumeration.
5. Report counts of minimum zero, one, two, and greater-than-two cases. Shared
   successful masks are candidates for further experiments, not evidence of
   common semantics or exclusive ownership by themselves.

The optimization removes redundant upstream computation, not the nonlinear
tail or GPU arithmetic. It does not extrapolate to new upstream states.

The completed sweep tested **1,086,112 masks**. Every original next token was
correct. The six-fact optimized run first reproduced all **8,326 enumerated
masks** from the independent full-model run, with zero differences in winners,
target probabilities or margins. All 3,295 full-mask controls in the expanded
prototype sweep reproduced every vocabulary logit bitwise.

| Minimum retained final-MLP features | Facts | Fraction of all 1,024 |
| --- | ---: | ---: |
| 0 (bias retained) | 292 | 28.52% |
| 1 | 217 | 21.19% |
| 2 | 171 | 16.70% |
| Greater than 2 | 344 | 33.59% |

Thus **680/1,024** next-token decisions can be preserved with at most two
active final-MLP channels, conditional on the unchanged upstream state and
all the tail weights and biases. This is not a two-channel model: each fact
may require a different mask, and selecting it used the known answer.

The stop rule makes the search tractable and certifies these minima, but it
also means masks of larger sizes are **unmeasured**, not failed, for facts
that already succeeded at smaller sizes. A mask-reuse study must apply a
common panel to every recipient rather than treating missing entries as zero.
The queries also have repeated next-token targets; apparent semantic sharing
must be distinguished from simply having the same answer token.

The checked-in C++ implementation independently reproduced the prototype's
complete level counts and all 2,612 successful minimum-size masks, including
their exact probabilities and margins. It passed 4,542 same-batch full-mask
controls, 2,659 representative **masked full-model** comparisons, and 1,024
ordinary post-search forwards, all comparing every vocabulary logit bitwise.
All source model master weights remained byte-identical. It completed in
35.84 seconds on this machine, including these controls; this is an experiment
timing, not a general GPU performance guarantee.

Among the 217 singleton-minimum facts, there are 975 successful singleton
choices, with median two alternatives per fact. **Every one of the 64
channels** is a successful singleton for at least two distinct target tokens.
For two-feature minima there are 1,345 successful masks, 760 distinct pairs,
and median three alternatives per fact. These are combinations with each
fact's own activation magnitudes and incoming residual; mask reuse alone does
not identify a shared semantic feature.

## Reproduce with the checked-in C++ probe

```sh
bazel build -c opt //src/llm/experiments/one_shot_memorizer:checkpoint_mlp_subset_probe
facts=/home/ubuntu/checkpoints/memorize_general_facts/compact_batch_32_no_clip_0
bazel-bin/src/llm/experiments/one_shot_memorizer/checkpoint_mlp_subset_probe \
  --checkpoint="$facts/layers_8/step_16128" \
  --tokenizer="$facts/inputs/tokenizer" \
  --lines=all --max_keep=2 --output_dir=/tmp/mlp_subsets_new
```

The output directory must be fresh. Omit `--lines=all` for the six original
facts. Use `--lines=80,406,258 --max_keep=4` for the deeper bounded search;
`--write_all_scores` includes unsuccessful masks and can produce large reports.
`--batch_rows` bounds physical batch storage (default 2,048 including two
identity rows). The maximum supported exhaustive cardinality is four.

The CPU-only `feature_subset` helper parses and enumerates masks and preserves
retained raw BF16 bits. `FinalMlpSubsetTail` copies the required weights from
the original model and runs only the production contraction, residual addition,
normalization and tied head. It does not implement a different GPU kernel or
retain a pointer into the original model. The CLI captures upstream values,
selects masks, checks controls and writes reports; targets do not belong to
the tail library. Unit tests exercise malformed masks/configurations, source
immutability/lifetime, nonzero affine parameters, vocabulary padding, and
bitwise masked/full-forward equality across several batch geometries.

## Local evidence

Generated reports remain outside Git:

- `/tmp/gpu_final_mlp_subsets_0/`: independent full-model mask sweep and controls.
- `/tmp/cpu_final_mlp_subset_search_0/`: forward and backward reference searches.
- `/tmp/cpu_lima_pair_mechanism_0/`: rival accounting and static direction ranks.
- `/tmp/gpu_lima_no56_subsets_0/`: full-model validation of the no-56 alternatives.
- `/tmp/gpu_cached_final_mlp_subsets_6_verified_0/`: optimized-versus-full comparison.
- `/tmp/gpu_cached_final_mlp_subsets_all_0/`: all 1,024 queries, complete levels and controls.
- `/tmp/mlp_subsets_production_all_0/`: checked-in implementation and expanded controls.
- `/tmp/mlp_subsets_production_deeper_0/`: exhaustive size-zero-through-four search for three facts.

The useful advance is a concrete description of a small **conditional
decision circuit**, including competing-token suppression. It does not yet
explain how to construct the learned upstream features directly from the
training corpus, which remains the main unresolved one-shot objective.
