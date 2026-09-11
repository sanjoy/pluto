# Exeunt / Nuveth: weight and behavior comparison

**Interim results, September 10, 2026, 09:58 UTC.** Original training has
finished; replacement training is still running. The following comparisons
use actual completed checkpoints at the same optimizer step. Final endpoint
and causal behavioral tests are pending. Protocol and run provenance are in
`EXEUNT_NUVETH_PAIRED_PROTOCOL.md` and `EXEUNT_NUVETH_PAIRED_RUN.md`.

## Main finding so far

Replacing Exeunt with Nuveth leaves a strong, interpretable ranking signal in
the nine embedding rows used to spell the two words. The largest differences
are not random-looking token identities: all eight largest rows at step 100,
and all seven largest at step 300, belong to these spellings. However, those
rows contain only 1.88% of the whole-model squared difference by step 300.
Most of the raw difference is distributed elsewhere. A word-row signal is
not the same thing as locating a unique word memory or demonstrating that
transferring these rows transfers behavior.

## Matched training trajectory

Define delta as replacement weights minus original weights. L2 uses all
51,483,648 saved FP32 parameters, accumulated in FP64; tied embeddings/head
are counted once. An energy share means a share of **squared** L2 differences.

| Optimizer step | Original train loss | Replacement train loss | Delta L2 | Delta / original weight L2 | Embedding share of delta energy | Nine word rows' share of total delta energy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | 5.32169 | 5.32332 | 2.19157 | 1.2965% | 88.6594% | 11.2098% |
| 200 | 4.96954 | 4.96383 | 6.70275 | 3.5555% | 81.8117% | 2.6960% |
| 300 | 4.66297 | 4.66770 | 11.19623 | 5.0577% | 77.3127% | 1.8786% |

The losses cover the first four sequential batches (40,960 targets) of each
arm's training split, not the full training corpus. They are evaluated against
each arm's own targets. Their similarity does not establish similar behavior
on the edited word. The fixed evaluated training/test prefixes contain only
three/22 complete edited occurrences, respectively.

Global weight cosines are 0.999916, 0.999368, and 0.998723. Shared initialization
contributes to these high values. Cosines between actual training updates from
the common initialization are 0.999302, 0.997887, and 0.997433.

Both arms use identical preserved inputs except the verified word replacement,
identical initial weights, fresh identical optimizer state, seed, binary, and
random-window sampling configuration. Replaying the preserved implementation
reproduces the saved sampler prefix. Historical batch starts are reconstructed
from these artifacts, not independently logged at every optimizer update.

## Word-piece ranking

Rank is by row delta L2 across the 50,257 logical vocabulary entries; physical
padding is excluded. Leading spaces in token pieces are significant.

| Token piece | ID | Rank at 100 | Rank at 200 | Rank at 300 | Delta L2 at 300 |
| --- | ---: | ---: | ---: | ---: | ---: |
| ` Nu` | 21733 | 1 | 1 | 1 | 0.81588 |
| `Ex` | 3109 | 2 | 2 | 2 | 0.78435 |
| `uve` | 45177 | 3 | 3 | 3 | 0.69973 |
| `ve` | 303 | 4 | 4 | 4 | 0.42394 |
| `th` | 400 | 8 | 6 | 5 | 0.37988 |
| `e` | 68 | 7 | 5 | 6 | 0.35742 |
| `unt` | 2797 | 6 | 12 | 7 | 0.26018 |
| ` Ex` | 1475 | 5 | 7 | 13 | 0.23464 |
| `N` | 45 | 36 | 565 | 879 | 0.09924 |

The native tokenization has two variants:

- Bare: `Ex / e / unt` versus `N / uve / th`.
- Leading space: ` Ex / e / unt` versus ` Nu / ve / th`.

Other uses of shared subword pieces remain in the corpus. Their differences
cannot be assigned exclusively to an edited word. In addition, the same rows
are the output classifier weights, so changed targets directly train them.

Counts in the complete native-tokenized training split help interpret the
ranking. These are corpus counts, not random-window training exposures:

| Piece | Original corpus | Replacement corpus |
| --- | ---: | ---: |
| ` Nu` | 0 | 883 |
| `Ex` | 53 | 0 |
| `uve` | 0 | 53 |
| `ve` | 81 | 964 |
| `th` | 244 | 1,180 |
| `e` | 1,127 | 191 |
| `unt` | 977 | 41 |
| ` Ex` | 943 | 60 |
| `N` | 134 | 187 |

Thus the three largest row differences belong to tokens entirely gained or
lost by the intervention. The others remain shared with unrelated text. Counts
were independently recomputed from the manifest-hash-verified native exports.

## Broader differences at step 300

Embedding delta L2 is 9.8446; the combined nonembedding delta L2 is 5.3329.
The embedding has about 50% of model parameters. Its per-parameter RMS
difference is 0.001940 versus 0.001051 across everything else, so its absolute
dominance reflects both its size and larger per-parameter changes.

Aggregated across all blocks, QKV delta L2 is 3.3003, MLP input 2.6704,
MLP output 2.6264, and attention output 1.7541. Relative to their own original
weight norms, the corresponding changes are 6.55%, 4.54%, 14.82%, and 21.44%.
Thus ranking only absolute differences misses relatively large changes in
smaller tensors. Q/K account for 86.02% of squared QKV difference.

Block 6 has the largest block-level difference at step 300 (L2 2.5391),
followed by blocks 7 (2.1497) and 5 (1.9661); indices are zero-based. The
largest block at step 100 was block 0. MLP-output differences are not confined
to a single neuron: each block's largest neuron contributes just 0.31–2.07%
of that block's MLP-output squared difference.

Attention-output and QKV training-update cosines are only 0.91460 and 0.92883
at step 300. Their deltas equal 41.35% and 37.90% of their original training
movement, compared with the embedding update cosine of 0.997989. Whole-model
weight cosine therefore hides meaningful variation between parameter groups.

These are descriptive differences, not causal evidence of where the word is
stored. FP32 atomic-gradient addition order is nondeterministic. The short
two-update repeat control measures early run-to-run noise but cannot determine
how much of this long-run spread is data-specific versus numerical divergence.

## Remaining tests

After replacement training completes, compare the actual final checkpoints,
then use the frozen native-model word/control cases. The preselected causal
check transfers only the nine word-token rows in each direction, preserving
all other recipient weights and every source checkpoint. Report whether word
preferences move toward the donor and how unrelated continuations change.

The frozen sample has 16 distinct word contexts per split. Its two recorded
prefix-domain conditions are byte-identical duplicates, not independent
contexts. The row patch also changes the tied head; even successful transfer
would not isolate an input-embedding mechanism. See the run note's overlap
audit for these interpretation constraints.

## Preserved evidence

All files below are under
`/home/ubuntu/checkpoints/exeunt_nuveth_20260910/analysis/`:

| Report | SHA-256 |
| --- | --- |
| `matched_step_100.json` | `2c67caface95105ca9b47cf72d0c48f9e3494f5f4a2e2bb6ac01dc8214c62699` |
| `matched_step_200.json` | `86062a0043b807061a54ddf13c623caf86114b63cf666f89f44f0e69b3e8af57` |
| `matched_step_300.json` | `af787a912328e2cb934273c792c012323b5fc61905f4ba604b30162b06dab524` |
| `matched_token_rows.json` | `28c336cbdb856687ab3dd4fa9a9523224d9185f57d6212423889725e366d6f50` |

Full comparisons record per-file checkpoint hashes and reject wrong sizes or
nonfinite weights. Tensor squared deltas independently sum to the whole-model
total; attention-head totals match attention weight tensors excluding biases.
All source file/directory identities and timestamps were unchanged through
comparison. No weight files were overwritten or deleted.
