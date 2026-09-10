# One Exeunt spelling route: block 1, head 2

This joins an existing computational trace with its existing native head-removal
experiment. It concerns historical checkpoint
`/home/ubuntu/checkpoints/shakespeare/step_13030`, **not** the new deterministic
paired models. No new GPU forward or training interruption was performed.

## What the parameters compute

At generated step 1342 the current input is `e` (ID 68, local position 1023),
and the target is `unt` (ID 2797). The previous position 1022 contains ` Ex`
(ID 1475). In block 1, head 2's query/key calculation concentrates strongly on
that earlier piece: FP64 reconstruction from captured BF16 QKV gives attention
probability **0.91729440214** at position 1022. The next-largest source receives
only about 0.007132. These are reconstructed probabilities, not exported native
FlashAttention online-softmax intermediates.

The physical parameter map for this route is:

| Operation | Weight file / slice (zero-based, end-exclusive) |
| --- | --- |
| Query from the current block-1 normalized residual | `weight_16.bin`, columns `[128,192)`; bias `weight_17.bin` |
| Keys from preceding normalized residuals | same files, columns `[640,704)` |
| Values from preceding normalized residuals | same files, columns `[1152,1216)` |
| Head-2 value mixture written into the residual stream | `weight_18.bin`, rows `[128,192)` |

QKV is a physical `[512,1536]` matrix and the output matrix is `[512,512]`.
These operations act on learned, context-dependent residuals—not raw token IDs.
The output projection's bias is `weight_19.bin`; the head-removal experiment
leaves it intact. The tied final dictionary then reads the downstream residual;
its `unt` row is in `weight_0.bin`, not a separate head tensor.

For the clean final-LayerNorm-adjusted `unt`-minus-` Exit` direction, the earlier
` Ex` source contributes **+0.969771050312**. Other sources sum to
**-0.003753627100**, giving reconstructed total **+0.966017423212**. Using the
captured native head context instead gives **+0.966737241269**. The explicit
difference is -0.000719818057; the reconstructed versus captured head context
has maximum absolute discrepancy 0.003796976979.

This is a concrete computed route from the earlier spelling into the readout,
but its source terms are **clean linear accounting**, not deletion effects.
They freeze the clean final normalization and do not model downstream changes.

## What removing the head actually does

The retained intervention zeroed the 64 output-projection rows for this head
at **all query positions**, then recomputed the complete native forward and
restored the weights. It did not remove only the ` Ex` source, alter an attention
mask, or isolate one prediction position.

Recomputing temperature-1 softmax over all 50,257 logical vocabulary logits:

| Predicted piece | Clean probability | Head removed | Rank before / after |
| --- | ---: | ---: | ---: |
| ` Ex` | 25.388682585% | 24.139451279% | 2 / 2 |
| `e` | 99.995630801% | 99.997243912% | 1 / 1 |
| `unt` | 99.999950002% | 99.999063723% | 1 / 1 |

The `unt`-versus-` Exit` margin falls from **16.633277893 to 13.719087601**:
a change of **-2.914190292**, not the negative of the clean +0.966737 head term.
Downstream responses and normalization matter. The head contributes to `unt`
confidence here, but it is **not necessary to retain the rank-1 completion**.
The intervened model's remaining computations suffice in this selected context;
this alone does not identify specific redundant routes.

The three recorded contexts are verified consecutive sliding windows conditioned
on the earlier word pieces. Their joint three-token probability changes from
**0.253875606102214 to 0.241385599687156**, a ratio of **0.9508026525** (about a
4.92% relative reduction). Most of that change comes from the first piece; `e`
actually improves. This is teacher-forced word-prefix probability, not a sampled
regeneration. The following boundary token was not intervened upon, so this
does not certify the probability of terminating the word correctly.

The head also affects other recorded words: removing it changes the NLL of
` cor` by +0.52353524, versus +0.05044875 for the whole Exeunt three-piece
sequence. It improves `am` while slightly worsening ` grand` and `se`.
These are collateral examples from the same generation, **not** matched
shared-subword controls or independent held-out cases.

## Evidence and remaining test

The CPU analysis artifact is
`/home/ubuntu/checkpoints/exeunt_nuveth_deterministic_20260910_1137/historical_b1_head2_route.json`.
It records 147 input/provenance files, all seven measured head-removal events,
the parameter map, numerical reconstruction differences, and limitations.
All checkpoint weights and used native arrays were hash-verified. Clean,
post-intervention replay, and alternate-padding logits are byte-identical for
each event. An independent audit reproduced the effects and additionally
compared logical clean logits with the original generation.

Today's `token_trace_probe.cc` has changed since the historical producer.
This analysis validates retained arrays and metadata; it neither reruns that
old producer nor asserts today's source/executable is byte-identical to it.

The next missing causal link is **source-specific routing**: change only the
earlier-piece contribution while keeping the other head computations intact,
then compare full word/boundary probabilities and controls. Whole-head removal
cannot establish that this particular source causes the effect. The new paired
branch/embedding screen and finer dose tools remain complementary tests, not
evidence that this historical head uniquely stores Exeunt.
