# Corpus-present generated word: revised goal and fixed trace

The active goal explicitly adds **present in the training data** to the earlier
generated-word investigation. The previous `shagemper` result does not meet
that condition and is not reused as a successful candidate.

## Search and selection

Continue the exact prompt `to be or not to be`, no comma/BOS/inserted target,
using `/home/ubuntu/checkpoints/shakespeare/step_13030`, native BF16 production
semantics, temperature 0.8, seed 17. Record 2,048 generated tokens, full selected
FP32 logit rows, native ID/byte boundaries, and sampling draws. This extends
the same chain: rerun prefix IDs/logits must match the previously recorded
128 tokens, and output bytes must match the prior production 512-token prefix.

The 2,048-token run is a collection size, not a relaxed stopping criterion.
If it contains no qualifying word, the goal remains active and generation
continues. Search does not insert corpus passages or promising words into the
prompt. Corpus lookup is explicitly used to **verify/select generated words**;
this is not a weight-only text extractor.

Reject incomplete words, ordinary multi-token words such as `abate` and `hurl`,
routine inflections such as `worthiest`/`groans`/`proveth`, and apparent words
that are merely fragments cut from contractions. Whole-word vocabulary absence
is checked case-insensitively, including bare and leading-space entries.

Independent triage of the existing 512-token production output identified
**beseem**, an archaic lexical verb, and later **grandam**, a rare/archaic noun.
Both occur in the configured training partition and are absent as whole-word
vocabulary entries. Select the earlier **beseem**, before looking at any of its
layer readouts or weight contributions. The dictionary's archaic label supports
a bounded judgment of uncommonness, not a claim that the word has no historical
morphological derivation. See [Merriam-Webster](https://www.merriam-webster.com/dictionary/beseem).

## Native target and checks

The immutable selection plan is
`/tmp/pluto-corpus-word.clM9FN/beseem_selection_plan.json`. The observed native
pieces are IDs 275, 2771, 368 (` b`, `ese`, `em`) at zero-based generation steps
235–237. Step 238 emits ` been`, establishing the right word boundary.

Selection used complete lines of the append-only event stream while the long
generation was still running. That prefix is copied into an immutable snapshot.
No final claim relies on an unfinished run: completed metadata, exact prefix
IDs, native logit bytes, and unchanged input hashes must all agree afterward.

The training membership check is against the supplied source corpus and its
configured 90/10 line-aligned split, implemented by `SplitCorpus` and the
default `test_fraction=0.1`. Record all word occurrences, native tokenization,
byte/line coordinates and partition membership. Do not claim that default
partition membership proves a particular historical sampled minibatch or that
one occurrence caused a particular parameter value.

## Fixed mechanistic scope

For each of the three actual emitted pieces, preserve all native prefix stages,
all 17 final-LayerNorm/tied-head intermediate readouts, and 80 independent
removals: all 16 attention/MLP branches and all 64 attention heads. Trace the
following boundary token without removals. Use the exact generated prefix IDs,
not independently re-tokenized text. Preserve padding/replay parity checks.

Analytically account for each target-minus-fixed-competitor logit margin using
actual final normalization, native activations, BF16 matrix operands, FP32
biases and explicit numerical remainders. Retain every head and neuron term,
not just supportive ones. Spell out the MT19937 draw and CDF operation selecting
each token; being most likely and being sampled are different facts.

“First becomes top-ranked under this readout” is an operational layer marker,
not proof of a unique storage or introduction layer. Full-network removals are
causal dependence tests, not equal to single-position linear accounting terms.
Any further experiments must be labeled as follow-ups motivated by these data.

## Completion gates

1. Actual continuation from the literal prompt, with completed native evidence.
2. A complete unusual lexical word, absent as a whole vocabulary entry but
   present in the configured training corpus; preserve supporting context.
3. Verified exact token boundaries and sampling for every piece.
4. Full layer trace and analytical operation accounting for that same occurrence.
5. Independent checks, tests, and an explanation separating readout crossings,
   causal reliance, and unproven unique historical-example attribution.

The earlier reports remain historical records of their original scope. Their
completion statements do not certify this revised goal.
