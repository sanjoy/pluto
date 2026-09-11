# Early attention/MLP transfer experiment

This follow-up separates the early-block weight bundles implicated by the
completed [Q/H/L cube](EXEUNT_SUFFIX_TRANSITIONS.md). It is not a new training
run, a restart of an interrupted analysis, or a claim that the word's mechanism
has already been explained. The lowercase amendment remains authoritative:
both `Exeunt -> Nuveth` and `exeunt -> nuveth` were applied in the replacement
training corpus.

## Fixed models, cases, and treatments

Use both matched step-331 endpoints under
`/home/ubuntu/checkpoints/exeunt_nuveth_deterministic_20260910_1137/`, with the
replacement endpoint under `lowercase_amendment/replacement/checkpoints/`.
The recipient and donor swap roles between directions. Every intervention is
a fresh independent checkpoint; all original and historical files remain
unchanged.

E fixes the same eleven selected donor token-embedding rows in both input
and tied-output roles. Unselected and padded embedding rows stay recipient.
The following disjoint groups partition all 99 other tensors:

| Group | Donor weight bundle | Tensors |
| --- | --- | ---: |
| A | Blocks 0–3: LN1 scale/bias, QKV weight/bias, attention output weight/bias | 24 |
| M | Blocks 0–3: LN2 scale/bias, MLP input and output weights/biases | 24 |
| R | Position embeddings, blocks 4–7, final normalization | 51 |

A here means the attention bundle, not the recipient label A in older outer
factorial reports. Attention and MLP include their pre-LayerNorm affine
parameters. These treatments therefore do not isolate normalization from
projection changes. A and M are sequentially interleaved in four residual
blocks, not parallel independent modules. Their parameter counts differ
(4,206,592 versus 8,402,944). R includes upstream position embeddings, not
only later computation.

Score all eight E-conditioned cells: E, A, M, R, AM, AR, MR, EC=AMR.
Use the unchanged frozen native loss probe/runtime and single-sequence scoring
on both original case suites: 188 main cases and 265 supplemental cases, each
packed to 1,024 positions. This retains the original 128-token-prefix assay,
not full training-context evaluation.

There are 32 new native measurements: eight cells, two suites, two directions.
Run the four inherited cells before the four new combinations in each
direction:

| New cell | Completed Q/H/L counterpart |
| --- | --- |
| E | E |
| AM | H |
| R | QL |
| EC | EC |

Independently verify the complete source/output weight hashes and that no
copy shares an inode with its reference. Every native loss and argmax byte,
including padding rows, must match the old counterpart. E/EC also repeat the
fixed FP32-loss versus saved FP64-logit comparison (absolute `5e-5` plus
relative `5e-6`); no tolerance relaxation is allowed. Main/supplemental shared
prefixes must have identical first-three scores. A failed gate stops the new
run without automatic retry.

## Primary comparisons and limits

Treat this as two A/M factorials: one with R recipient and one with R donor.
Report additions A−E, M−E, AR−R, MR−R; donor-background reversions EC−MR and
EC−AR; and interactions AM−A−M+E versus EC−AR−MR+R. A reversion replaces
weights with the recipient's values; it does not remove the layer computation.

Retain per-token, conditional suffix, whole-word, argmax, and exact following
native-token results for both candidate words. The original-prefix domain is
the primary view; report other domains separately, never as independent
replicates. Separate leading title-case, bare title-case, and lowercase strata.
For bare Nuveth the middle token is `uve`, not the leading spelling's `ve`.
Generic and shared-piece controls remain necessary: previous preference flips
did not imply confident spelling, and broad transfers altered unrelated text.

These are native-loss contrasts. Nonadditivity in log probability does not
separate changes in target logits from softmax normalization and is not a
storage percentage or proof of an internal representational interaction.
No first-token/whole-word greedy successes were established in the prior
39-context assay. Attention-head/MLP-feature computations and contextual
word selection remain unresolved after a bundle-level localization.

## Provenance after directory relocation

The previous completed experiment is
`lowercase_amendment/complement_localization_run_20260910_2212/`. Its summary
SHA-256 is `8e25efee3f61e9ba9ef2cd7eb39a9a56f89de296ba6b87e5c60fb2b92c4fcf84`.
The source snapshot `pre_sharing_sources_20260910_2254/manifest.json` has
SHA-256 `ab2123f405d4a8afe263167412cce6f0d7aec527c7f837c96587de5fc16afc24`.

The new verifier preserves each historical logical identity and checks its
bytes at the pinned archive location. It never rewrites old commands or
substitutes current `ai-slop` source hashes for historical ones. The new native
ledgers list real current physical inputs, including archived sources as data,
and separately snapshot the actual new executing modules and regression tests.
Both controller exit and GPU idleness must be observed before handoff.

Launch only into a new sibling directory using
`PYTHONPATH=ai-slop python -B -m weight_analysis.paired_early_branches`, supplying
`--previous`, `--source-archive`, both `--summary-sha256` and
`--archive-sha256`, and `--output`. Each completed cell has its own marker;
the full cube and global summary are published only after all required gates.
