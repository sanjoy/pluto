# Interpreting the amended paired-model causal screen

CPU-only audit, 2026-09-10. Replacement training is still running; there are
**no new paired-model causal measurements** in this document. This records
the actual frozen case coverage and the limits of the queued screen before
seeing its outcomes. No frozen program, input, checkpoint, or queue changed.

## How many contexts are actually tested?

The packed input arrays were read directly and checked against their recorded
SHA-256 hashes. Prefix counts below use actual native input bytes, not merely
different case labels. The original/replacement candidates and prefix-domain
aliases are not independent observations.

| Word stratum | Packed main cases | Corpus occurrences | Distinct native prefixes | Leading-space / bare occurrences |
| --- | ---: | ---: | ---: | ---: |
| Title-case training | 64 | 16 | 16 | 14 / 2 |
| Title-case test | 64 | 16 | 16 | 15 / 1 |
| Lowercase training | 28 | 7 | 8 | 7 / 0 |
| Lowercase test | 0 | 0 | 0 | 0 / 0 |

Each occurrence is crossed with two candidate words and two prefix domains.
All selected title-case domains have identical native prefixes. One lowercase
occurrence has an earlier replaced word within its prefix, so its two domains
are genuinely different inputs. Thus eight lowercase predictive prefixes do
not mean eight separate corpus occurrences. Distinct prefixes are not, by
themselves, a guarantee of statistical independence either.

The actual-following-token suite repeats these same word contexts with a
fourth target. It is a boundary check, not an independent set of word examples
or a sum over all possible valid delimiters. In particular, a result on the
bare title-case test variant has **one** selected held-out occurrence, not
sixteen. Lowercase behavior has no held-out occurrence in this corpus split.

There are also 16 generic unrelated continuations per split and 64 training /
45 test shared-piece controls. The latter cover unrelated uses of the selected
embedding rows, where qualifying windows exist:

| Piece ID | Piece | Training controls | Test controls |
| --- | --- | ---: | ---: |
| 45 | `N` | 8 | 8 |
| 68 | `e` | 8 | 8 |
| 303 | `ve` | 8 | 8 |
| 400 | `th` | 8 | 8 |
| 409 | ` ex` | 8 | 8 |
| 1475 | ` Ex` | 8 | 3 |
| 2797 | `unt` | 8 | 0 |
| 3109 | `Ex` | 0 | 0 |
| 14364 | ` nu` | 8 | 2 |
| 21733 | ` Nu` | 0 | 0 |
| 45177 | `uve` | 0 | 0 |

Zero means no eligible unrelated window under the frozen selection rule,
not a measured absence of collateral damage. Shared-piece sampling is
row-stratified, not frequency-weighted over Shakespeare.

Authenticated case manifests under
`/home/ubuntu/checkpoints/exeunt_nuveth_deterministic_20260910_1137/lowercase_amendment`:

- `word_cases/cases.json`: SHA-256
  `282a8084ad465dcc72284746beaff01a5ab2deaa85b9549a3901975dfee3daa0`.
- `supplemental_cases/cases.json`: SHA-256
  `9baf51672de90483ba28eeee3ebe00be3e2196f796d4cf608d8a6455985d96e0`.

## Compare the same quantity with the same sign

`paired_lowercase_scores.py` reports `log P(original) - log P(replacement)`.
`paired_branch_readout.py` reports the opposite sign. The amended controller
uses that branch reader and `embedding_factorial_readout.py`; it does not use
the legacy title-only `paired_patch_readout.py`.

Normalize these signs before comparing stages. Align actual prefix bytes,
candidate token IDs, capitalization, leading-space variant, split, and step.
Trajectory averages are within a prefix domain; branch summaries additionally
offer deduplicated combined-domain averages. They are not interchangeable
weights, especially for the lowercase occurrence with two different prefixes.
Never add a combined-domain group to its constituent domain groups.

For each promising transfer, report:

1. Recipient, donor, and patched **absolute** probability of each candidate's
   three native tokens, together with their log probabilities.
2. First-piece selection and conditional two-piece completion separately.
3. Preference change in nats, retaining the donor-recipient gap and the
   patch-recipient change individually, as well as context-by-context signs.
4. Exact following-token behavior and unrelated/shared-piece effects.
5. Training versus held-out cases and both matched-step observations.

Improving a preference ratio while making both words less likely is not
successful transfer of the donor's word production. The branch reader exposes
this case explicitly. A ratio of averaged changes may also become enormous
when positive and negative context gaps cancel. Its `1e-6`-nat denominator
floor is only a numerical guard: not practical significance, uncertainty, or
a fraction of the word's mechanism that has been explained.

Only the first candidate predictions have the same teacher-forced history.
Suffix predictions condition on different preceding candidate pieces; their
normalizers cannot be canceled as if their hidden states were identical.

## Zero stored loss is not proof of exact certainty

The trajectory and branch probes retain native **FP32 NLL**, not full logits.
The producer's cross-entropy evaluates `log(denominator) + max - target` in
FP32. Adding a sufficiently small positive log denominator to a larger target
logit can round away the loss. Later FP64 aggregation cannot recover it.

A synthetic CPU example, **not a GPU measurement**, illustrates the issue.
With target logit 20 and one competing logit 6, explicit FP32 arithmetic gives
denominator `1.0000008344650269`, log denominator `8.344646857949556e-7`,
but `(log_denominator + 20) - 20` is zero. The real-valued two-logit reference
has NLL `8.315283733837542e-7` and probability `0.9999991684719723`, not one.
Compiler arithmetic and reduction order are additional reasons not to infer
an exact numerical error bound for the native kernel from this illustration.

The inspected source equals commit
`08edf21940f06344b2b649d631b8716b6fb8e0a2`'s
`src/llm/layers/cross_entropy_loss.cc`, SHA-256
`821d727ff9bd9c1655f45b2ab43c1cd51e07f7a01af41f0ccec2ecdeb9fb3bd0`.
This is a readout limitation, not a reason to alter the matched training run.

The factorial probe saves native FP32 **logits** and scores them with FP64
log-sum-exp, avoiding this particular loss-output quantization. It still does
not provide exact real arithmetic. Its existing logit dumps can clarify
near-ceiling embedding effects; the branch NLL-only dumps cannot support an
equivalent retrospective logit decomposition. If a tiny branch effect matters
to the mechanistic conclusion, collect full logits in a subsequent assay.

## What the queued screen can and cannot establish

The paired screen transfers the eleven spelling rows jointly, each whole
attention/MLP branch, and each branch's output projection separately. It uses
full-strength donor bytes, both donor directions, unmodified-copy controls,
and the earliest/latest common positive steps. Whole-branch transfers include
pre-LayerNorm and biases; output-write transfers include output bias.

AA/JJ native-forward parity calibrates the embedding factorial. AJ/JA are
intentional untied input/output hybrids with recipient transformer and final
LayerNorm parameters. They separate functional input, output, and interaction
effects. The [margin/normalizer postprocessor](EXEUNT_FACTORIAL_LOGIT_PROTOCOL.md)
further distinguishes probability nonlinearity from interactions in logits.
Neither test establishes unique storage or necessity.

The frozen screen does **not** yet include intermediate-dose paired swaps,
individual-row attribution, matched unrelated-weight interventions,
multi-branch combinations, or sequential damage/restoration assays. A reverse
donor swap is not the same experiment as restoring a damaged recipient. The
separately queued historical head/source-value dose assays use a different
checkpoint and different contexts; they cannot fill these paired-screen gaps.

Consequently, treat this as a functional-transfer screen. Once actual effects
are available, select follow-up controls based on the unresolved mechanism:
replicated dose/restoration tests, matched negative interventions, and
combinations where an interaction is proposed. A negative single-branch swap
does not prove irrelevance in a co-adapted network; a successful one does not
explain the detector/routing computation or show exclusive word storage.

These are concrete remaining work, not completion claims. No program changes
or new GPU experiments were made during this audit.

Validation: the three relevant CPU suites (`paired_lowercase_scores_test`,
`paired_branch_readout_test`, and `embedding_factorial_readout_test`) passed
all **61 tests** in 3.167 seconds. Independent review reproduced the synthetic
FP32 example and checked the cited source bytes and interpretation against the
actual readers and plan. The six live training/observer PID start identities
were rechecked after the audit; no job was restarted or changed.
