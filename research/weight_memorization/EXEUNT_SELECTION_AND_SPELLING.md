# Exeunt: selecting the word versus completing its spelling

This is partial mechanistic evidence. It combines **archived interventions in
the older step-13030 model** with **corpus/sampler facts about the new paired
experiment**. It does not present historical outcomes as new paired-model
results, and it does not identify a unique location storing the word.

**Current-run correction (2026-09-10):** the user subsequently amended the
replacement to include all seven lowercase `exeunt -> nuveth` occurrences.
The old exact-case-only replacement arm never launched; the amended arm is
the one now training. The historical-model interventions below remain valid,
but the lowercase-survival section describes the **superseded input plan**,
not contamination in the active run. See
[EXEUNT_LOWERCASE_AMENDMENT.md](EXEUNT_LOWERCASE_AMENDMENT.md) and the verified
[matched step-200 report](EXEUNT_EARLY_STEP200.md).

## A distinction supported by actual interventions

At one recorded context, the model spells the leading-space form as token IDs
1475, 68, 2797: ` Ex`, `e`, `unt`. Its probability factors as

    P( Ex, e, unt | context)
      = P( Ex | context)
        * P(e | context, Ex)
        * P(unt | context, Ex, e).

The spaces in this display are explanatory; the first token itself includes a
leading space. Each conditional uses the actual sliding 1,024-token context,
with learned positions reset as in the recorded generation. The prefixes have
been checked to form one teacher-forced chain. These are temperature-1
probabilities computed over all 50,257 logical vocabulary entries, not the
historical temperature-0.8 sampling probabilities. The unscored next token
means these numbers do **not** require a word boundary after `unt`.

Joining the *same* native intervention across all three pieces gives:

| Intervention (zero-based block/head) | Probability of the three-token sequence | Probability of suffix `e`, `unt`, conditional on ` Ex` | Full-sequence NLL increase |
| --- | ---: | ---: | ---: |
| None | 25.38756061% | 99.99558081% | 0 |
| Remove B0H5 | 1.67076466% | 99.85124694% | 2.72097791 |
| Remove B2H2 | 2.05494237% | 99.98083805% | 2.51401151 |
| Remove B1H4 | 5.89938599% | 99.73468908% | 1.45941104 |
| Remove B3H6 | 5.99877711% | 99.76234263% | 1.44270368 |
| Remove B3H5 | 7.77615162% | 99.97304080% | 1.18319775 |

These are the five strongest single-head removals for the full sequence among
all 64 heads. For B0H5, 2.71953347 of the 2.72097791 NLL increase comes from the
first piece; only 0.00144445 comes from the two suffix pieces. **Every one of
the 64 individual head removals leaves the suffix probability at least
99.73468908%.** This does not imply that removing several heads together, or
removing all attention, would preserve spelling.

The [confidence and branch reanalysis](EXEUNT_SUFFIX_CONFIDENCE_HISTORICAL.md)
quantifies the remaining confidence changes: the worst single-head removal
increases complementary suffix mass about 60-fold, yet preserves more than
99.7% completion probability. It also shows that removing the whole block-0
attention branch raises the full-word probability while hurting its suffix.
Initiation, completion and aggregate word probability are distinct readouts.

The retained intervention changes the 64 corresponding rows of the attention
output matrix to zero, retaining its bias. It reruns the entire forward, so
this removes that head's contribution at **all query positions**, not just the
scored position or a particular source-token connection. It establishes a
causal contribution of the head to these conditionals, not the precise route
or a unique lexical computation inside that head.

The strongest effects are not word-exclusive:

| Head removed | Exeunt NLL increase | grandam NLL increase | corse NLL increase |
| --- | ---: | ---: | ---: |
| B0H5 | 2.72097791 | 0.34853917 | 7.71292103 |
| B2H2 | 2.51401151 | 0.64151742 | 4.60049659 |
| B1H4 | 1.45941104 | 0.07941715 | 3.26697632 |
| B3H6 | 1.44270368 | 0.03125348 | -0.02771229 |

B3H6 is therefore a candidate for a more selective context-selection effect,
but this is **post hoc**, with one context per word, different context lengths,
three versus two target tokens, and no matched-context population estimate.
It is not an Exeunt-specific head
identification. There are no confidence intervals because these are exact
deterministic measurements of selected events, not sampled population estimates.

A further context audit adds a substantive limitation: this Exeunt event follows
63 standalone spaces after a newline, beyond the longest space run in the
frozen corpora. B3H6 reads almost entirely from those contextualized space
positions, and its negative clean readout term has the opposite sign to its
overall causal benefit. See [the initial-piece context analysis](EXEUNT_INITIAL_PIECE_CONTEXT.md)
before interpreting the cross-word contrast as lexical selectivity.

By contrast, removing the whole B0 MLP drops the Exeunt sequence probability to
1.67638699e-17 (a probability, **not** a percentage) and its suffix probability
to 2.21249323e-13. That intervention removes the output matrix and bias at all
positions. It also severely damages the other words and ordinary passages;
see [the broad-damage diagnostic](EXEUNT_B0_MLP_HISTORICAL_DIAGNOSTIC.md).
It is evidence of a foundational dependency, not proof that B0 alone stores
the word. The clean feature ledger contains many positive and negative terms,
not one dominant Exeunt neuron.

## The spelling has a strong local statistical cue

The new experiment's frozen native tokenizer exports show:

| Original-corpus conditional frequency | Training split | Test split |
| --- | ---: | ---: |
| `e` after leading-space ` Ex` | 883/943 = 93.6373% | 90/93 = 96.7742% |
| `e` after bare `Ex` | 53/53 = 100% | 2/2 = 100% |
| `unt` after `e` | 943/1,127 = 83.6735% | 92/102 = 90.1961% |
| `unt` after either `Ex` token followed by `e` | 936/936 = 100% | 92/92 = 100% |

These are corpus counts, **not model probabilities** or counts of actual
training presentations. Transitions across the train/test boundary are
excluded. Looking back from `e` to the preceding `Ex` perfectly disambiguates
`unt` in this finite corpus; `e` by itself does not. That motivates a local
spelling-circuit hypothesis, but does not prove the model implements a
bigram/trigram lookup or ignores longer context.

There is already a matching *descriptive* route in the historical model:
B1H2's reconstructed attention at `e` assigns about 91.73% to the preceding
` Ex`. Removing that head still leaves `unt` nearly certain; see
[the attention-route readout](EXEUNT_ATTENTION_ROUTE_HISTORICAL.md).
The queued source-value intervention will test this particular source's
contribution rather than treating a large attention coefficient as causality.

## Superseded exact-case plan: a surviving spelling signal

The exact-case intervention changes all 936 training and 92 test occurrences
of `Exeunt`, but leaves **seven lowercase `exeunt`** in training. They tokenize
as `[409, 68, 2797]` (` ex`, `e`, `unt`). Thus in the replacement training text,
`e` still precedes `unt` 7 times out of 191 `e` opportunities (3.66492%).
The other 34 surviving training `unt` tokens occur in `Blunt`. There are no
remaining `unt` tokens in the replacement test split.

Reading the authenticated saved deterministic window starts gives:

| Through optimizer step | Complete lowercase presentations | Distinct lowercase occurrences completely presented |
| --- | ---: | ---: |
| 1 | 0 | 0 |
| 2 | 0 | 0 |
| 100 | 3 | 3 |
| 200 | 8 | 6 |

All three pieces have equal counts at these steps; there are no partial
lowercase presentations at those checkpoints. Supervised target windows are
`[start+1, start+1025)`, rather than the input interval. These are conditional
sampler counts, not a separately recorded runtime batch trace. At this analysis
the original arm has reached step 200 and the replacement arm has not started;
the replacement counts apply **if and when** it reaches the corresponding step.

Under that initial plan, retained `e` -> `unt` behavior would not have been
interpretable as learning with zero suffix evidence. This concern led to the
later user-approved lowercase amendment; the exact-case-only replacement was
superseded before launch. In the actual amended run the seven occurrences
supply `nuveth`, not `exeunt`. Unrelated uses of shared pieces such as `Blunt`
still require collateral controls. The counts above are retained as historical
input-plan evidence rather than silently relabeled as amended-run results.

## What this adds to the causal account

The evidence distinguishes three roles to test:

1. Context-dependent selection of the first ` Ex` piece: strongly affected by
   several single-head removals, with substantial collateral effects.
2. Completion of `e` and `unt`: highly robust to any one head removal in this
   recorded context, with strong local corpus cues and a broad B0-MLP dependency.
3. Conversion of residual activations into token probabilities by final
   LayerNorm and the tied embedding readout. The paired input-versus-output
   embedding intervention is queued to separate these two uses of the weights.

This is a more constrained hypothesis than "the largest delta stores Exeunt."
It still needs matched-step paired-model behavior, targeted interventions over
multiple contexts, and checks of spelling plus its boundary. The work does not
yet establish a sufficient, selective circuit for the whole word.

## Reproducibility

Evidence root: `/home/ubuntu/checkpoints/exeunt_nuveth_deterministic_20260910_1137`.
New CPU readouts are `historical_word_ablations.json`,
`paired_spelling_statistics.json`, and
`lowercase_spelling_exposure_through200.json`. Their matching Python modules
and tests are in `scripts/weight_analysis/`; each output records input/source
hashes. These analyses do not execute a GPU forward or alter checkpoint bytes.

The lowercase-exposure helper counts exact token triples, not arbitrary word
boundaries. An independent scalar/text audit verified delimiters after all seven
actual hits and reproduced all window counts. Its v1 source hashes identify
the bytes at the end of analysis, not a before/after source-stability guarantee;
its input files are checked before and after. These limits do not change the
independently reproduced counts and are not silently promoted to stronger
provenance claims.

The archived intervention source was independently recovered from commit
`08edf21940f06344b2b649d631b8716b6fb8e0a2`, path
`scripts/weight_analysis/token_trace_probe.cc`: 22,156 bytes, SHA-256
`c94165709ea12e54e0b38cff5044d9ce57e96dbc4a813fb3776826f32f213ba9`.
This exactly matches the historical plan. Lines 346-375 establish branch/head
removal and restoration semantics. The current dirty source differs and is
not misrepresented as the original producer.
