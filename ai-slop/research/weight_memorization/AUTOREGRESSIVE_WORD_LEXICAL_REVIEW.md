# Bounded lexical review: `shagemper`

Review date: 2026-09-09. This supplements, and does not modify, the frozen
structural selection in `generated_words_128.json`. That earlier file correctly
marks automatic lexical review as incomplete: token-vocabulary absence alone
does not establish that a spelling is unusual.

## Decision

Select **shagemper** as a generated, unusual word-like spelling rather than an
ordinary word split across tokens or a recognizable common compound. This is
a bounded linguistic assessment, **not proof that nobody has ever used it**,
and not a claim that its component token IDs are outside the vocabulary.

The native output says `a hundred shagemper of France`. Its actual pieces are
` sh` + `ag` + `em` + `per`; the following ` of` establishes a complete right
boundary. Every piece is a valid GPT-2 token. No case-insensitive single-token
spelling, bare or with a leading space, exists in the exact supplied vocabulary.

The authenticated full Shakespeare corpus has no case-insensitive whole-word
occurrence, no full four-ID sequence, and neither of its three-ID subsequences.
The complete corpus contains 1,835,163 native tokens. These are exact checks of
these particular files, not assumptions about all English or all training data.

## External checks and controls

Two **separate** web searches on the review date returned empty results:

- Exact query `"shagemper"`.
- Exact query `"shagemper" dictionary`.

An earlier combined search including the split phrase `"shag emper"` returned
unrelated split-term results. Those are not whole-word attestations. Search
coverage is incomplete, and empty results do not prove universal absence.

A direct attempted lookup of the candidate on Merriam-Webster failed at the
tool's URL safety/fetch layer. **That failure is not dictionary-absence evidence.**
As positive controls, the ordinary generated multi-token words
[abate](https://www.merriam-webster.com/dictionary/abate) and
[hurl](https://www.merriam-webster.com/dictionary/hurl) have accessible dictionary
entries. Both were rejected as examples of unusual spelling despite passing
the automatic multi-token/whole-vocabulary-absence filter. Other ordinary
spellings such as `worthiest` and `groans` were likewise not selected.

## Compound and morphology assessment

The eight two-part letter splits are `s|hagemper`, `sh|agemper`, `sha|gemper`,
`shag|emper`, `shage|mper`, `shagem|per`, `shagemp|er`, and `shagempe|r`.
None is recognizable here as an ordinary English compound with a conventional
meaning; `shag|emper` is the most suggestive split, but it does not supply a
recognizable ordinary second component or contextual meaning. Nor is the
whole spelling a recognizable ordinary inflection. This is a manual judgment,
not an exhaustive dictionary algorithm or a quantified frequency claim.

The native corpus does contain its adjacent token pairs in `shag`, `stratagem`,
and `distemper`/`intemperance`. This gives a plausible **subword-recombination**
interpretation; it does not make `shagemper` a conventional compound. BPE pieces
need not be morphemes or standalone words. These post-hoc corpus associations
also do not prove which training examples caused the learned parameters.

## Evidence boundaries

The selected spelling satisfies the practical search criterion at this stated
level of review. No claim is made about a new concept, universal lexical
novelty, hidden plaintext, or a unique feature dedicated to this word. The
mechanistic conclusions in the [word trace](AUTOREGRESSIVE_WORD_TRACE.md) do
not depend on an unverifiable claim of universal novelty.
