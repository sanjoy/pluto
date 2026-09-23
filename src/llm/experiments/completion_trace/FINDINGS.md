# Five autoregressive completion traces: step 16128

Recorded 2026-09-23 on NVIDIA GH200, from
`compact_batch_32_no_clip_0/layers_8/step_16128`: eight blocks, width 16, one
attention head, MLP width 64, 4,475 compact tokens, 114,256 unique parameters.
The experiment branch is based directly on `main` at `a2623e0`; it does not
inherit the earlier research branch's interventions. See [README.md](README.md)
for the reproducible command and numerical-output definitions.

## Actual completions

Each left side contains exactly five tokens and each right side five newly
generated tokens. All **25/25** predictions match the corpus. This is a five-step
check, not a new full-corpus or suffix-plus-EOS evaluation.

| Corpus line | Prompt | Five generated tokens, concatenated |
| --- | --- | --- |
| 80 | `The capital of France is` | ` Paris, a city on` |
| 406 | `The capital of Greece is` | ` Athens, an ancient Mediterranean` |
| 411 | `The capital of Peru is` | ` Lima, near the Pacific` |
| 631 | `Durian fruit is known` | ` for its strong smell and` |
| 1 | `Female mammals produce milk to` | ` nourish their young.` |

Mammals' five outputs are `" nour"`, `"ish"`, `" their"`, `" young"`, `"."`.
Durian's prompt begins `"Dur"`, `"ian"`: words and tokens are not interchangeable.

The run contains 2,725 layer-output snapshots and 200 causal attention matrices
(109 plus eight per forward). Logits match uninstrumented execution bit-for-bit
for every decision; normal generation matches every continuation. All earlier
positions stay bit-identical as each prefix grows, and all 100 unique parameter
tensors are unchanged. No gold suffix token enters the model input.

## 1. Same starting token, progressively different answers

At step 1, each capital fact has the exact same query token `" is"` at the same
position. Its token-plus-position embedding is identical across all three facts.
The differences can therefore enter that query through attention to the distinct
histories, and subsequent pointwise transformations can amplify/reorient them.

Mean pairwise cosine across the three capital query vectors:

| Boundary | Predicting Paris/Athens/Lima, step 1 | Predicting a/an/near, step 3 |
| --- | ---: | ---: |
| Token plus position embedding | 1.0000 | 1.0000 |
| Block 0 post-attention | 0.9980 | 0.9987 |
| Block 6 post-MLP | 0.9101 | 0.7771 |
| Block 7 post-attention | 0.6881 | 0.6630 |
| Block 7 post-MLP | 0.4142 | 0.3800 |

The shared query at step 3 is now the generated comma. Starting from that
identical token, the histories produce `" a"`, `" an"`, and `" near"` with
probabilities 99.36%, 99.59%, and 98.79%. This is a particularly useful matched
comparison: same current token and position, different preceding facts.

The late separation is not monotonic throughout the network. Nor does it prove
that the last block stores the facts: late operations consume history-dependent
states already built by earlier blocks. Cosine measures geometry, not stored
information or necessity.

## 2. Different starting tokens can lead to a shared output

At step 2, the query inputs are `" Paris"`, `" Athens"`, and `" Lima"`. They
all generate comma, with probabilities 99.69%, 99.88%, and 99.61%.

Their mean pairwise residual cosine rises from **0.4557** after embedding to
**0.6767** after the last MLP (0.6751 after final LayerNorm). The final vectors
are still far from identical. This illustrates a many-to-one readout: different
hidden states can all select the same vocabulary token.

Across all 25 queries, the only same-output pairs at matching steps are these
three comma pairs. The 47 different-output pairs have mean final-normalized
cosine 0.0762 versus 0.6751 for comma pairs. After per-layer centering, the
corresponding means are -0.0081 and 0.6133. This is descriptive evidence only:
there are just three related positive pairs, and capital template/position
confounding prevents a general claim about semantic clustering.

## 3. Generated city tokens become prominent context for later words

When producing the third new token (`a`, `an`, `near`), block 4's query attention
assigns **86.25%, 83.55%, 87.56%** to the previously generated city token at
position 6. On the fourth decision (`city`, `ancient`, `the`), these rise to
**99.04%, 99.39%, 97.97%**. Block 3 similarly assigns 98.69%, 99.87%, 95.12% to
the city on that fourth decision.

This is consistent with the city-position representation serving as a useful
context source for the continuation. It is not merely the raw city embedding:
that position has already passed through earlier blocks and has access to its
own preceding history. A future intervention should distinguish its raw lexical
identity from its contextual K/V representation.

## 4. High attention can point to an invariant anchor, not the fact-bearing word

While predicting Paris, block 7 sends **96.06%** of its query attention to the
first token `"The"`, but only 3.15% to `" France"`. For Athens and Lima the
same block instead sends 52.90% and 81.03% to their country positions.

Because attention is causal, position 1's representation cannot encode the
later country name; it is identical for these three capital prompts. Its value
can provide a common learned direction, with a history-dependent query
controlling its coefficient. Earlier history-dependent information also remains
in the residual stream. Thus “the model attends to The” does not mean “The
contains the fact France → Paris.”

Other anchors are frequent: to predict `" nour"`, mammals' blocks 3, 4, and 5
give `"Female"` 99.89%, 99.95%, and 99.01% respectively. Durian's block 4 gives
`"Dur"` 99.993% while predicting `" its"`, and 99.9998% while predicting
`" strong"`. Unlike `The`, these first tokens differ between these examples;
the traces alone cannot tell whether they act as identifiers, learned defaults,
or another computation. Attention probabilities alone do not measure the size
or causal effect of the transformed value contribution.

## 5. Both late attention and the final MLP make substantial geometric changes

Across the 25 query positions, the mean block 7 MLP residual update has L2 norm
**0.7981**, the largest of the eight MLPs; block 7 attention averages **0.5472**,
the largest attention update. The mean cosine of pre-/post-update states is
0.7082 for the final MLP and 0.8811 for final attention.

For Paris specifically, final attention changes the residual direction with
cosine 0.8355, and the final MLP changes it with cosine 0.3885. Paris's final
probability is 92.02%; Athens and Lima are 98.32% and 97.70%.

This identifies candidate locations for controlled interventions, not a ranking
of which layer owns the answer. Update norms depend on scale and downstream
readout; a small update can change argmax while a large update can be irrelevant.

## What remains unknown / most targeted follow-ups

These traces establish the actual numerical route, but do not uniquely assign
semantic meanings to 16 coordinates or prove exclusive ownership of facts.
The most informative next experiments would use the matched capital prompts:

1. At the comma step, transplant a city position's K and V separately and
   together; measure whether `a/an/near` transfers. Compare early vs late blocks.
2. At the first completion, isolate the invariant `The` value contribution from
   the history-dependent residual/query. This can test a common-direction or
   gating explanation without treating attention mass as retrieval evidence.
3. Interpolate or patch query states across the final attention/MLP boundaries,
   measuring complete logit changes rather than only cosine/argmax. Repeat on
   more facts to separate a capital-template effect from a general mechanism.

No intervention above was performed in this experiment. The full dumps, weights,
and compact vocabulary mapping permit inspection without relying on these
interpretations; the short walkthrough is the easiest starting point.
