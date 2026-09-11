# Exeunt: confidence, initiation, and completion can move differently

These are **reanalyses of retained native interventions in the older
step-13030 model**, not results from the new Exeunt/Nuveth paired training.
No new GPU forward was run and ongoing training was untouched.

The same intervention is joined across the three verified sliding contexts:
` Ex` (1475), `e` (68), `unt` (2797). Temperature-1 probabilities use all
50,257 logical vocabulary logits. "Suffix" means the exact two-token
continuation conditioned on ` Ex` already being supplied. Neither suffix nor
whole-sequence probability sums alternative tokenizations or requires a
following word boundary.

## Near-certain completion is not unchanged completion

Clean conditional suffix probability is **99.9955808%**. Its complementary
probability mass is **0.00441919%**. We calculate the latter stably as
`-expm1(log_probability)`, not by subtracting a rounded percentage from 100.
It is model-assigned mass outside this exact continuation, not an observed
sampling-error rate or an estimate across many contexts.

Among all 64 single-head removals, these five cause the largest suffix damage:

| Removed head | Suffix probability | Complementary mass | Complementary-mass increase |
| --- | ---: | ---: | ---: |
| B1H4 | 99.7346891% | 0.2653109% | 60.04x |
| B3H6 | 99.7623426% | 0.2376574% | 53.78x |
| B0H5 | 99.8512469% | 0.1487531% | 33.66x |
| B3H5 | 99.9730408% | 0.0269592% | 6.10x |
| B2H2 | 99.9808380% | 0.0191620% | 4.34x |

The multiplicative changes show a measurable causal contribution to
confidence, but the absolute numbers remain important: **every individual
head removal preserves suffix probability above 99.7%** in this context.
This establishes neither absence of a head's role nor joint dispensability
of several heads. No specific redundant circuit has been identified.

B1H4 primarily affects `e`: after removal, `P(e)` is 99.7347387%, while
`P(unt | ...e)` remains 99.9999503%. The same removal has a much larger effect
on initiating Exeunt: first-piece probability falls from 25.3886826% to
5.9150793%. It also strongly damages initiating `corse`; the comparison does
not establish an Exeunt-specific head.

B1H2 provides a different caution. Its removal reduces `unt` confidence, but
improves `e` enough that the combined suffix probability **increases** from
99.9955808% to 99.9963077%. The whole-word probability still decreases because
the first piece gets worse. See the [source-route analysis](EXEUNT_ATTENTION_ROUTE_HISTORICAL.md)
for the actual query/key/value parameter slices and the pending source-specific
causal test. Scoring only `unt`, only the combined suffix, or only the whole
word would give incomplete descriptions of the same intervention.

## Removing a whole branch can have opposite effects on initiation and spelling

The complete eight-branch screen below retains all results rather than only
the favorable example. Each row is one separate intervention on the original
checkpoint; it is not a cumulative removal of successive blocks.

| Removed attention branch | First-piece probability | Conditional suffix probability | Full three-token probability |
| --- | ---: | ---: | ---: |
| None | 25.388683% | 99.995581% | 25.387561% |
| Block 0 | 35.271618% | 94.727309% | 33.411855% |
| Block 1 | 0.031549% | 99.253062% | 0.031313% |
| Block 2 | 0.843821% | 99.977440% | 0.843631% |
| Block 3 | 6.403268% | 99.039695% | 6.341777% |
| Block 4 | 21.455004% | 99.991719% | 21.453227% |
| Block 5 | 18.224105% | 99.989994% | 18.222281% |
| Block 6 | 38.545758% | 99.992000% | 38.542675% |
| Block 7 | 16.768238% | 99.987855% | 16.766202% |

Removing block 0's attention branch improves the probability of initiating
the word enough to outweigh worse suffix completion. Its first-piece NLL
changes by **-0.32877512**, suffix NLL by **+0.05412366**, and total NLL by
**-0.27465146**. Block 6 also has opposite first-piece and suffix effects,
though its suffix change is much smaller. All eight branch removals worsen
this suffix to some extent; their net word effects are not interchangeable.

The exact block-0 intervention damages other selected continuations:
`grandam`'s full-sequence probability falls from 3.003894% to 0.184764%, and
`corse`'s from 6.143534% to 0.002182%. These have different contexts and token
counts. They demonstrate collateral effects, not matched-context lexical
selectivity or a population-wide benefit for Exeunt.

## What these interventions do and do not isolate

The recovered historical producer verifies that an attention-branch removal
zeros **both the entire output-projection matrix and its bias**, at all query
positions, followed by a complete native forward and weight restoration.
A single-head removal zeros only that head's 64 output-matrix rows and
**retains the bias**. Therefore the whole-branch result cannot be treated as
an otherwise identical combined-head intervention: a bias control would be
needed before attributing the discrepancy to head interactions or redundancy.

Nor do these all-query edits isolate a particular source token, the final
query position, or the direct clean projection term. Downstream layers and
normalization respond. The separate queued head-position and source-value
assays address some of these distinctions.

The historical Exeunt event follows **63 standalone space tokens after a
newline**, an unusual context absent from the new frozen corpora. It is one
selected event, not a corpus-level effect. The amended paired-model evidence
must remain separate. These results support a distinction between initiation
and completion computations; they do not identify where the word is uniquely
stored or establish a complete sufficient circuit.

## Evidence and checks

New derived artifact:
`/home/ubuntu/checkpoints/exeunt_nuveth_deterministic_20260910_1137/historical_suffix_confidence.json`
(1,452,211 bytes, SHA-256
`be66a468dc7f6a607194575e4a5eda4d5176cf90bfd5030131fa291cdb105c3b`).

The complete historical reader was rerun on CPU; its result exactly matched
the saved `historical_word_ablations.json` (SHA-256
`46e9f07f8998092325c165c8cf5fec4e0443cb8bbbc8caf4f9d0d2d125d30522`).
It revalidated 693 upstream records, all 100 finite checkpoint weight files,
the recovered producer source, and clean/replay/padding/original-generation
logit parity. The derived report records 697 source files, all 64 head and
eight branch interventions for each of three words (216 comparisons), stable
failure/odds calculations, boundary handling, and explicit historical scope.

An independent check recomputed the highlighted piece probabilities directly
from authenticated native logits, verified all 216 derived decompositions
against the original readout, and rehashed all 697 source records. All checks
agreed. The existing historical-reader CPU suite also passed again: **21 tests**.
