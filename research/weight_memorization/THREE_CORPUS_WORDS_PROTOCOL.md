# Three further corpus-present generated words

## Objective and selection

Repeat the complete `beseem` investigation for three **different** uncommon
words that occur in the configured training partition but have no whole-word
entry in the GPT-2 tokenizer. The user permits **any prompt**. The three selected
candidates already occur in the actual continuation of `to be or not to be`,
so this investigation reuses it rather than changing prompts unnecessarily.

Use the already completed, authenticated 2,048-token native recording in
`/tmp/pluto-corpus-word.clM9FN/recorded_2048`, checkpoint
`/home/ubuntu/checkpoints/shakespeare/step_13030`, temperature 0.8, seed 17.
Reusing its later generated words does not require regenerating the same chain.
The preceding investigation verified the entire recording's sampling and
its first 512 tokens against the production binary. Native trace logits must
match the exact original-generation rows for every newly selected token.

Select these occurrences before inspecting their layer results:

| Word | Generated indices (zero-based, inclusive) | Native IDs | Following boundary event |
| --- | --- | --- | --- |
| grandam | 365–366 | 4490, 321 | 367 |
| corse | 700–701 | 1162, 325 | 702 |
| Exeunt | 1340–1342 | 1475, 68, 2797 | 1343 |

[Collins](https://www.collinsdictionary.com/dictionary/english/grandam)
labels `grandam` archaic/now rare; it denotes a grandmother or old woman.
[Merriam-Webster](https://www.merriam-webster.com/dictionary/corse)
labels `corse` archaic, meaning corpse.
[Merriam-Webster](https://www.merriam-webster.com/dictionary/exeunt)
identifies `exeunt` as a borrowed Latin stage direction for characters leaving.
The third candidate is specialist theatrical vocabulary, **not rare within
Shakespeare**. Its many training occurrences must be reported, not hidden.
Uncommonness here concerns general modern English, not low corpus frequency.
These are established lexical items, not invented words, routine English
inflections, names, or conventional phrases made by adjoining their actual
token pieces. Historical morphology is not denied.

All candidates come from the previously retained panel of 70 generated
structural word occurrences. Corpus membership and lexical qualification guide
selection; favorable layer scores do not. This is not a blind weight-only
extractor or a random sample from an unspecified population of rare words.

## Required evidence

1. Validate each complete generated word and following delimiter from original
   token bytes. Check all 50,257 vocabulary entries, case-insensitively, including
   bare and leading-space forms; all component IDs must remain valid.
2. Preserve all corpus occurrences, line/byte/native-token coordinates, and
   the current default line-aligned 90/10 partition. Corpus membership does not
   authenticate particular historical training minibatches.
3. Freeze exact native context IDs, events, source/checkpoint hashes and commands.
   `Exeunt` is beyond the 1,024-token window: use the recorded sliding context,
   with the production absolute-position reset, not the full growing prefix.
4. For every one of the seven word tokens, capture all 17 native intermediate
   readouts and remove all 16 branches and 64 heads independently. This is
   **560 native removal experiments**. Trace the three following boundary
   tokens without removals, for ten total baseline contexts.
5. Reconstruct the exact sampled probability, MT19937 uniform/CDF interval,
   target-minus-competitor margin, all head/neuron contributions, and addressed
   input-dot/GELU/output-dot examples. Keep opposing contributions and numerical
   remainders. Readout accounting is not an intervention effect.
6. Independently audit the recorded native evidence and analytical accounting;
   run the relevant Python/native tests. Report all three words, including any
   unexpected trajectories, rather than selecting the most convenient result.

## Interpretation and preservation

As before, a first top-ranked intermediate readout does not prove a unique
word-storage or introduction layer. A complete word can require several
autoregressive passes; a sampled initial piece need not be the top prediction.
Whole-branch removals affect the entire current context and subsequent layers.
No claim of a uniquely responsible historical sentence follows from these tests.

New evidence belongs under `/tmp/pluto-three-words.qPVE2k`; old evidence and all
checkpoints remain unchanged. Keep raw arrays local and record hashes. Reports,
reusable analysis code, and tests belong in the repository. No commit is implied
by this goal; commit only when requested.
