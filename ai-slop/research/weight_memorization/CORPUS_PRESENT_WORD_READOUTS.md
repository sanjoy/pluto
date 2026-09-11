# `beseem`: complete readouts, branch removals, and corpus locations

The generated word `beseem` is present in the corpus: five whole-word matches
in the current training prefix and one in the held-out suffix. All six use
the same three native token IDs as the generated occurrence: `' b'` (275),
`'ese'` (2771), and `'em'` (368). This supplement preserves the full readout
and branch-removal tables, not only the largest effects.

The prompt was `to be or not to be` (no comma). Generation steps 235–237
produced these pieces at full-stream token indices 241–243. The word occupies
generated-output bytes `[752, 758)` and prompt-plus-output bytes `[770, 776)`.
The preceding space is part of token 275, not part of the word. Every column
below is a separate forward pass conditioned on its actual preceding token
IDs; later columns include the earlier sampled pieces.

Checkpoint: `/home/ubuntu/checkpoints/shakespeare/step_13030`. Blocks B0–B7
are zero-based. Matrix operands and activations use native BF16; saved logits
are FP32. All probabilities use the production temperature **0.8** and all
50,257 logical vocabulary entries, excluding the 15 padded entries.

## All 17 intermediate readouts

Each cell is **target probability / target rank**. Probabilities are percentages
rounded to six significant digits. Ranks are descending raw-logit ranks,
with lower token IDs breaking ties. The native diagnostic applies the trained
final LayerNorm and tied output head to each intermediate residual stream.
The final row is byte-identical to the actual final logits.

| Residual stage | Step 235: `' b'` | Step 236: `'ese'` | Step 237: `'em'` |
| --- | ---: | ---: | ---: |
| Input + position | 9.24637e-41% / 49668 | 3.4725e-41% / 39592 | 1.76966e-22% / 6923 |
| B0 attention | 9.9804e-36% / 45016 | 1.50522e-39% / 42988 | 5.63812e-20% / 42715 |
| B0 MLP | 2.0464e-05% / 1664 | 0.0745028% / 74 | 55.7736% / 1 |
| B1 attention | 8.10399e-06% / 4448 | 0.00500994% / 355 | 4.78193% / 5 |
| B1 MLP | 4.43071e-06% / 3385 | 0.00102324% / 678 | 3.61085% / 7 |
| B2 attention | 6.23986e-07% / 6845 | 0.000363077% / 839 | 6.96806% / 4 |
| B2 MLP | 2.13513e-07% / 6450 | 0.000386945% / 571 | 5.21163% / 4 |
| B3 attention | 1.81504e-07% / 7044 | 0.000433248% / 499 | 8.63588% / 3 |
| B3 MLP | 4.05941e-07% / 3309 | 0.00181392% / 303 | 18.5925% / 2 |
| B4 attention | 4.32799e-08% / 3222 | 0.0307673% / 76 | 19.7446% / 2 |
| B4 MLP | 2.84289e-05% / 1199 | 1.91002% / 12 | 75.6712% / 1 |
| B5 attention | 5.1237e-05% / 1106 | 0.938172% / 18 | 48.4267% / 2 |
| B5 MLP | 7.87512e-05% / 154 | 55.8337% / 1 | 98.4586% / 1 |
| B6 attention | 3.53855e-05% / 118 | 49.6742% / 1 | 98.8505% / 1 |
| B6 MLP | 0.0682592% / 19 | 94.1403% / 1 | 99.9639% / 1 |
| B7 attention | 0.0294242% / 21 | 91.2011% / 1 | 99.9698% / 1 |
| B7 MLP | 13.0888% / 2 | 98.2167% / 1 | 99.961% / 1 |

`' b'` is never rank one in these recorded readouts; it is actually sampled
at rank two. `'ese'` first reaches rank one after B5 MLP. `'em'` reaches rank
one after B0 MLP, falls to rank five at the next attention stage, and later
recovers. These crossings describe readability through this particular lens,
not a unique introduction point, an irreversible decision, or proof that a
component is necessary. In particular, the early rank-one crossing for `'em'`
does not mean the remaining blocks simply copy an already-decided answer.

## All 16 branch removals for each target

Each cell is the target's full-vocabulary probability after removing that
branch, in percent. The baseline row makes comparisons explicit. Each native
intervention zeros the branch's output projection **and its bias**, at every
prefix position, and recomputes the entire forward pass. Weights are restored
and checked byte-for-byte afterward. This is not a last-position-only
intervention, and the effect includes downstream responses to the change.

| Removed branch | Step 235: `' b'` | Step 236: `'ese'` | Step 237: `'em'` |
| --- | ---: | ---: | ---: |
| No removal | 13.0888% | 98.2167% | 99.961% |
| B0 attention | 0.00433111% | 6.95807% | 93.6983% |
| B0 MLP | 25.8968% | 2.17039e-05% | 0.0159439% |
| B1 attention | 0.00830904% | 1.55505% | 98.4271% |
| B1 MLP | 0.810356% | 99.6648% | 99.9391% |
| B2 attention | 0.440434% | 99.5312% | 99.9362% |
| B2 MLP | 1.37372% | 58.7649% | 99.9857% |
| B3 attention | 1.12958% | 99.8148% | 98.9298% |
| B3 MLP | 8.71599% | 94.6544% | 99.1433% |
| B4 attention | 0.0173437% | 88.2171% | 99.8298% |
| B4 MLP | 0.504764% | 12.2794% | 94.6249% |
| B5 attention | 4.56731% | 99.1897% | 99.9725% |
| B5 MLP | 0.245697% | 45.7006% | 98.7908% |
| B6 attention | 2.1019% | 95.1198% | 99.9604% |
| B6 MLP | 0.0618149% | 51.3942% | 99.1581% |
| B7 attention | 30.6559% | 98.3601% | 99.9345% |
| B7 MLP | 0.0294242% | 91.2011% | 99.9698% |

### A fixed-competitor margin can improve while probability collapses

For `'ese'`, the original runner-up is the bare apostrophe token (ID 6).
Removing B0 MLP changes the target-minus-apostrophe logit margin from
`+3.736873627` to `+4.997788072`
(`Δmargin = +1.260914445`). Nevertheless, the target probability
falls from **98.2167%** to
**2.17039e-05%**, and its rank falls from 1 to
624. The new greedy winner is `' b'` (275).
The fixed-apostrophe margin ignores the other 50,255 possible competitors;
it cannot stand in for full-vocabulary probability. Nor should the original
linear margin ledger be treated as the measured effect of removing a branch.

All 64 individual-head removals per target, their probabilities and ranks,
fixed-competitor margins, log-probability changes, and same-uniform samples
remain in the source JSON: [step 235](/tmp/pluto-corpus-word.clM9FN/token_analysis_step_235/analysis.json),
[step 236](/tmp/pluto-corpus-word.clM9FN/token_analysis_step_236/analysis.json), and
[step 237](/tmp/pluto-corpus-word.clM9FN/token_analysis_step_237/analysis.json).
The following boundary token, `' been'` (587), has a separate
[step 238 baseline/readout trace](/tmp/pluto-corpus-word.clM9FN/token_analysis_step_238/analysis.json);
it is not a fourth piece of `beseem`.

## Every corpus occurrence

The [native corpus lookup](/tmp/pluto-corpus-word.clM9FN/corpus_candidates.json) checks raw corpus
bytes and the exported native token boundaries, without retokenizing each
word in isolation. All six matches have the exact spelling `beseem` and the
actual token sequence `[275, 2771, 368]`, decoded as `' b' + 'ese' + 'em'`.
The native span therefore includes one leading space. Byte and token spans
are zero-based, half-open; line and column numbers are one-based.

| Line:byte column | Word byte span | Native token span | Full native byte span | Current partition |
| --- | --- | --- | --- | --- |
| 2236:25 | [87830, 87836) | [26361, 26364) | [87829, 87836) | Training prefix |
| 14829:19 | [633344, 633350) | [210948, 210951) | [633343, 633350) | Training prefix |
| 47982:12 | [2101273, 2101279) | [701759, 701762) | [2101272, 2101279) | Training prefix |
| 89595:18 | [3933309, 3933315) | [1324429, 1324432) | [3933308, 3933315) | Training prefix |
| 101832:33 | [4463641, 4463647) | [1504397, 1504400) | [4463640, 4463647) | Training prefix |
| 118754:12 | [5200543, 5200549) | [1756981, 1756984) | [5200542, 5200549) | Held-out suffix |

The current default split is byte **4,892,836**,
also a native token boundary after **1,650,781**
tokens, in a **5,436,475**-byte / **1,835,163**-token corpus.
Both each word and its full native token span lie wholly in the listed
partition. These are corpus-membership facts, not evidence that the model
retrieved one particular passage or that every historical training run used
the same split. The lookup retained all 70 structural candidates,
including absent ones; 51 had training-prefix occurrences.
It does not automatically classify a word as unusual or establish memorization.

## Evidence and presentation checks

The [independent raw-math audit](/tmp/pluto-corpus-word.clM9FN/beseem_math_independent_audit.json)
checked all four original-generation baseline rows byte-for-byte, 68 native
lens rows (including the boundary-token trace), 240 interventions, and 312
sampling distributions. Probabilities reproduced exactly. It also checked
65,536 neuron terms, 256 head terms, 62,080 attention-source terms, and 192
selected gate operations. Maximum ledger closure error was
`2.9976e-15`.
No additional model run was needed for this supplement.

Source SHA256 identities:

- `token_analysis_step_235/analysis.json`: `0f8933c2904e944b1896ab56fd26b238aedef5bd31160b57e35c77048202762e`.
- `token_analysis_step_236/analysis.json`: `ff22adccd1d85f6fce0233854675b804d92042237c85daaa46d268eee3217400`.
- `token_analysis_step_237/analysis.json`: `2b96249a77ff4364cdd03a4513450e6871191270defd666d58144efb50b66ede`.
- `corpus_candidates.json`: `200686e65eabc0f2bcdf03863c1a3e6102faa4939cf089779924098354d174fa`.
- `beseem_math_independent_audit.json`: `dd1adbd8b4ac95eac5a702891ff7fda5401a99ed996379ced09bd840b2edef25`.

The [presentation helper](/tmp/pluto-corpus-word.clM9FN/build_readouts_report.py)
reconstructs every table cell from those JSON files, checks the native-readout
rows against the independent audit, and verifies that this Markdown matches
the generated report exactly. It checks 51 lens probability/rank cells, 48
branch-removal probabilities, three baseline probabilities, and all six corpus
location rows. Run its default verification with:

```text
/home/ubuntu/.venv/bin/python /tmp/pluto-corpus-word.clM9FN/build_readouts_report.py
```
