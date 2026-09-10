# What the initial Exeunt heads read, and what their effects do not mean

This refines [selection versus spelling](EXEUNT_SELECTION_AND_SPELLING.md).
The causal measurements are archived forwards of the older step-13030 model;
the formatting counts use the frozen corpora of the new deterministic paired
experiment. No GPU forward was run for this analysis.

## The selected historical context is unusually indented

The initial ` Ex` prediction at generation step 1340 follows:

    Come, let us hence; the day isle day.
    [63 space bytes, with no other content on the new line]

The actual context is 1,024 tokens. The last newline is local position 960;
positions 961 through 1023 are all standalone space tokens, ID 220. The target
token ` Ex` (ID 1475) adds another leading space, placing the E at a total
indentation of **64 spaces**. The target's clean temperature-1 probability is
25.38868259%; another space is rank 1. This is a recorded sampled continuation,
not an exact training passage selected for a memory-recall test.

The native frozen Shakespeare exports show:

| Corpus property | Training | Test |
| --- | ---: | ---: |
| Exact-case Exeunt on a line preceded only by spaces | 397 | 42 |
| Exact-case Exeunt after earlier non-space line content | 539 | 50 |
| Largest literal consecutive-space run anywhere | 58 | 58 |
| Largest consecutive standalone space220-token run | 57 | 57 |
| Newline198 followed by 63 standalone space tokens | 0 | 0 |

Thus this generated prefix extrapolates beyond the indentation observed in
these frozen corpora. Its empirical next-token distribution is **undefined**
because there are no conditioning opportunities, not an observed 0% chance of
Exeunt. This audit does not independently establish the older checkpoint's
complete historical training inputs or exposure history.

The nearest observed long indentation has newline198 followed by **57**
standalone spaces, then a token carrying the final leading space. Among 45
training opportunities, 37 continue with ` Ex`, three with ` [`, and one each
with ` HAM`, ` Exit`, ` '`, ` Music`, and ` Guns`. All eight test opportunities
continue with ` Ex`. Long indentation is consequently a strong *corpus-level*
cue for ` Ex`, but not an exclusive one; these frequencies are not model
probabilities or evidence that a particular head implements that cue.

## Three heads have different computational signatures

The previous native screen selected B0H5 and B2H2 as the two strongest Exeunt
single-head removals, and B3H6 as a comparatively selective candidate against
two other generated words. All indexing below is zero-based.

From captured BF16 QKV, reconstruct one query's attention in FP64:

    p(source) = softmax(Q(query) dot K(source) / sqrt(64))
    head_write = sum_source p(source) * V(source) * W_output[head_rows]

The clean accounting term below projects that write onto the clean,
final-LayerNorm-adjusted ` Ex`-minus-space direction. It is NOT a native
ablation effect. The last column is the actual native effect of zeroing that
head's output rows at all query positions and rerunning the entire model.

| Head | Reconstructed attention on the 63 indentation positions | Clean fixed-direction term | Native change in ` Ex`-minus-space margin on removal |
| --- | ---: | ---: | ---: |
| B0H5 | 9.12852016% | +0.03397936 | -3.00651646 |
| B2H2 | 79.22884517% | +6.63373546 | -2.81037521 |
| B3H6 | 99.999999934% | -2.39225149 | -1.66942406 |

For the clean terms, the table uses the **captured native head context**, not
the reconstructed source sum. Reconstructed versus native head-context maximum
absolute discrepancies are 0.00079711, 0.00189636, and 0.00352623, respectively.
These differences remain explicit; the reconstructed probabilities are not
claimed to be stored native FlashAttention softmax statistics.

B0H5 is diffuse: its largest individual source is ` Come` at row 948, with only
0.42776016% attention. Its substantial causal effect cannot be described as
one dominant `Come` lookup. B2H2 concentrates on the current indentation and
has a large positive clean readout term. B3H6 concentrates almost entirely on
those space positions, but has a **negative** clean readout term.

Space-position values in blocks 2 and 3 are already contextualized by earlier
blocks. They may carry information about `Come`, `hence`, or other preceding
text. Reading space positions therefore does **not** prove these heads ignore
words, implement only formatting, or retrieve raw space embeddings.

## A directly observed sign reversal

For B3H6, the clean ` Ex`-minus-space margin is **-1.05211258**. Naively
subtracting its clean -2.39225149 contribution while freezing everything else
would *increase* that margin by 2.39225149. The actual native head removal
instead decreases it by **1.66942406**, leaving **-2.72153664**.

The rival remains the exact same space token in this comparison. The reversal
is not merely an artifact of switching competitors or looking only at softmax
probability. But the naive subtraction freezes final normalization and all
other computations, whereas the native intervention changes every query's
head output and reruns the network. Downstream computation, normalization,
and effects mediated through other query positions are not isolated here.
The discrepancy does not identify which one of them mediates the benefit.

This rules out the simple interpretation "B3H6 helps Exeunt by directly writing
a positive Exeunt-versus-space signal at the selected position." It does not
rule out a contextual role for that head, or establish an alternative complete
mechanism. Similarly, B0H5's small clean term understates its full causal
effect, whereas B2H2's large clean term overstates the measured removal effect.

The relevant parameter addresses are concrete, even though a unique lexical
storage location has not been established:

| Head | Packed QKV weight / bias | Q columns | K columns | V columns | Output weight / selected rows |
| --- | --- | --- | --- | --- | --- |
| B0H5 | weight_4 / weight_5 | 320:384 | 832:896 | 1344:1408 | weight_6 / 320:384 |
| B2H2 | weight_28 / weight_29 | 128:192 | 640:704 | 1152:1216 | weight_30 / 128:192 |
| B3H6 | weight_40 / weight_41 | 384:448 | 896:960 | 1408:1472 | weight_42 / 384:448 |

Filenames have the `.bin` suffix; slices are end-exclusive. Packed QKV matrices
are [512,1536], output matrices [512,512], stored as input-by-output. QKV acts
on the block's normalized residual. Output biases (7, 31, 43) remain unchanged
in the head-removal interventions. The tied readout uses rows 1475 and 220 of
weight_0, with final normalization parameters weight_98 and weight_99.

## Consequences for the next causal tests

The small collateral effects of B3H6 on grandam and corse cannot establish
word selectivity: those comparison events do not have the 63-space prefix.
The first-piece mechanism needs ordinary corpus-prefix controls, including
both indentation-only and inline Exeunt, and other stage directions at similar
indentation. The preselected paired corpus cases remain the primary source of
multi-context evidence; the generated example is a diagnostic, not a substitute.

The next discriminating head intervention is **query-local versus all-query**
removal on the same native prefix, with byte-identical replay and full-vocabulary
readout. If query-local removal has a different sign, other positions matter;
if it preserves the reversal, the downstream response to that local write
becomes the next target. A source-value-group test can then distinguish the
indentation-position pathway from the rest of the head. Those are proposed
controls, not completed or automatically queued experiments.

Changing indentation from 57 to 63 standalone spaces is also informative, but
changes context geometry. Any such comparison must specify truncation and
learned absolute positions rather than silently attribute all effects to space
count. Attention mass and corpus frequency alone cannot settle that mechanism.

## Evidence

The new readouts are `historical_initial_piece_routes.json` and
`paired_indentation_statistics.json` at
`/home/ubuntu/checkpoints/exeunt_nuveth_deterministic_20260910_1137`.
They retain data/source hashes, numerical checks, exact grouping conventions,
and limitations. Their analyzers and CPU tests are in `scripts/weight_analysis/`.

Original native arrays and clean accounting are in
`/tmp/pluto-three-words.qPVE2k/run/trace_step_1340/` and
`/tmp/pluto-three-words.qPVE2k/run/analysis_step_1340/analysis.json`.
The same-head causal effects are independently joined in
`historical_word_ablations.json`; its exact historical producer source was
recovered from Git and matched against the archived plan. No new native
intervention or paired-model probability result is claimed by this note.
