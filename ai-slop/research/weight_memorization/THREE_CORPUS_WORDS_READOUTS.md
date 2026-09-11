# Three corpus-present words: complete native readouts

This supplement contains all 119 intermediate token readouts and all 112
whole-branch removals for `grandam`, `corse`, and `Exeunt`, selected from the
same actual 2,048-token native continuation of `to be or not to be`.
The checkpoint is `/home/ubuntu/checkpoints/shakespeare/step_13030`;
temperature is **0.8**, seed is **17**, and blocks B0–B7 are zero-based.
Native matrix operands/activations are BF16; saved logits are FP32.
Probabilities use all **50,257 logical vocabulary entries**, not padded entries.

The words have no case-insensitive bare or leading-space whole-word GPT-2
vocabulary entries. Each is made from valid vocabulary token IDs. `grandam`
and `corse` are archaic words; `Exeunt` is a specialized stage direction,
**not a rare word within Shakespeare**. Lexical sources and all 1,091 exact
corpus occurrences are retained in the [independent qualification audit](/tmp/pluto-three-words.qPVE2k/lexical_corpus_audit.json).

## The seven actual sampling events

Step indices count generated tokens from zero. Probabilities in tables are
percentages rounded to six significant digits; ranks are descending raw-logit
ranks with lower token IDs breaking ties. Logits and sampler numbers below
use round-trip float representations. A displayed `100%` is rounding, not
a claim of probability exactly one; unrounded values remain in the JSON.

| Word | Step | Piece / ID | Raw logit | Rank | Probability |
| --- | ---: | --- | ---: | ---: | ---: |
| grandam | 365 | `' grand'` / 4490 | `11.419553756713867` | 5 | 3.32646% |
| grandam | 366 | `'am'` / 321 | `17.000728607177734` | 1 | 99.061% |
| corse | 700 | `' cor'` / 1162 | `11.658897399902344` | 2 | 11.8961% |
| corse | 701 | `'se'` / 325 | `21.11672592163086` | 1 | 99.9888% |
| Exeunt | 1340 | `' Ex'` / 1475 | `17.63292121887207` | 2 | 21.0257% |
| Exeunt | 1341 | `'e'` / 68 | `23.918500900268555` | 1 | 99.9999% |
| Exeunt | 1342 | `'unt'` / 2797 | `27.865703582763672` | 1 | 100% |

Production sampling uses `exp(double(float32(logit - max)) / 0.8)`,
sequential double summation/normalization, and the first cumulative
probability at least `U`. All seven observed draws lie in `(lower, upper]`.
These are the original sampler draws, not newly sampled completions.

| Step | Uniform U | CDF lower | CDF upper |
| ---: | ---: | ---: | ---: |
| 365 | `0.7825663146456792` | `0.773586607253877` | `0.8068512274143713` |
| 366 | `0.28718553461356355` | `8.725302725123045e-05` | `0.9906976314450957` |
| 700 | `0.16318512646073294` | `0.07915487817411507` | `0.19811607934438946` |
| 701 | `0.04304608194621734` | `4.5476752897626246e-06` | `0.9998929199661042` |
| 1340 | `0.8808377562527094` | `0.7835076523237393` | `0.9937643469813792` |
| 1341 | `0.14211536636648023` | `2.4589615488668875e-08` | `0.9999989829344745` |
| 1342 | `0.5485888490453982` | `2.1153287021101592e-09` | `0.9999999975968091` |

## Exact contexts and the sliding window

Each column in a word table is a separate forward pass, conditioned on
its actual preceding sampled IDs. Context spans below index the complete
prompt-plus-generation stream, are zero-based, and exclude the next target.
The six prompt tokens count toward these absolute positions.

| Step | Absolute target index | Conditioning token interval | Context length | Selected row |
| ---: | ---: | --- | ---: | ---: |
| 365 | 371 | [0, 371) | 371 | 370 |
| 366 | 372 | [0, 372) | 372 | 371 |
| 700 | 706 | [0, 706) | 706 | 705 |
| 701 | 707 | [0, 707) | 707 | 706 |
| 1340 | 1346 | [322, 1346) | 1024 | 1023 |
| 1341 | 1347 | [323, 1347) | 1024 | 1023 |
| 1342 | 1348 | [324, 1348) | 1024 | 1023 |

`Exeunt` occurs after the 1,024-token window fills. Its three passes use
successive context starts **322, 323, and 324**. Every pass uses local learned
position IDs 0–1023: it is not a reused hidden state with permanently growing
absolute position IDs. Full native contexts and logits are compared with the
original generation; the prompt is no longer inside those three windows.

## Complete intermediate and whole-branch tables

Each cell below is **target probability / target rank**. Intermediate
readouts apply the trained final LayerNorm and tied output head to the
indicated residual stream. Their last row is byte-identical to final logits.
First rank-one crossings describe readability through this diagnostic lens,
not an irreversible decision, a unique word-introduction point, or necessity.

A whole-branch removal zeros its output projection **and bias at every
context position**, recomputes the entire forward pass, and restores weights
byte-for-byte. Effects include all downstream responses. These are not
last-position-only removals and not additive predictions from a margin ledger.

### `grandam`

Generated word bytes: `[1143, 1150)`.
Native pieces: `' grand'` + `'am'`.
The following token `' Love'` (ID 5896, step 367) confirms the complete word boundary.

| Residual stage | Step 365: `' grand'` | Step 366: `'am'` |
| --- | ---: | ---: |
| Input + position | 1.26275e-42% / 43316 | 1.02116e-26% / 4861 |
| B0 attention | 2.90481e-36% / 44098 | 2.54254e-24% / 39256 |
| B0 MLP | 3.80076e-08% / 13206 | 53.2329% / 1 |
| B1 attention | 2.37652e-07% / 7226 | 88.7391% / 1 |
| B1 MLP | 1.29016e-07% / 8295 | 77.5525% / 1 |
| B2 attention | 8.40105e-07% / 6383 | 40.5976% / 1 |
| B2 MLP | 1.40998e-06% / 4824 | 60.7364% / 1 |
| B3 attention | 1.41004e-05% / 1272 | 17.2634% / 2 |
| B3 MLP | 3.28989e-05% / 581 | 42.2841% / 1 |
| B4 attention | 0.000398855% / 137 | 6.70591% / 2 |
| B4 MLP | 0.00224049% / 80 | 68.093% / 1 |
| B5 attention | 0.00429795% / 103 | 68.8025% / 1 |
| B5 MLP | 0.00061646% / 105 | 97.5246% / 1 |
| B6 attention | 0.00132803% / 101 | 98.7783% / 1 |
| B6 MLP | 0.000182527% / 130 | 99.5888% / 1 |
| B7 attention | 0.00148624% / 77 | 99.6948% / 1 |
| B7 MLP | 3.32646% / 5 | 99.061% / 1 |

First rank-one readouts:

- Step 365, `' grand'`: never rank one.

- Step 366, `'am'`: B0 MLP.

| Removed branch | Step 365 | Step 366 |
| --- | ---: | ---: |
| No removal | 3.32646% / 5 | 99.061% / 1 |
| B0 attention | 0.0990636% / 44 | 98.9879% / 1 |
| B0 MLP | 8.11864e-05% / 4743 | 8.83164e-05% / 5935 |
| B1 attention | 0.0495998% / 146 | 99.7227% / 1 |
| B1 MLP | 0.780578% / 9 | 98.1881% / 1 |
| B2 attention | 0.0679291% / 165 | 89.4073% / 1 |
| B2 MLP | 1.41176% / 7 | 99.0865% / 1 |
| B3 attention | 0.181954% / 39 | 99.9175% / 1 |
| B3 MLP | 0.382816% / 16 | 95.2957% / 1 |
| B4 attention | 0.216155% / 23 | 98.5806% / 1 |
| B4 MLP | 0.0751337% / 12 | 95.2936% / 1 |
| B5 attention | 12.9533% / 2 | 97.5374% / 1 |
| B5 MLP | 0.340781% / 15 | 99.3783% / 1 |
| B6 attention | 1.46928% / 7 | 98.6671% / 1 |
| B6 MLP | 1.55679% / 4 | 99.0782% / 1 |
| B7 attention | 0.418151% / 6 | 97.0569% / 1 |
| B7 MLP | 0.00148624% / 77 | 99.6948% / 1 |

Full head/branch interventions and analytical term ledgers: [step 365](/tmp/pluto-three-words.qPVE2k/run/analysis_step_365/analysis.json), [step 366](/tmp/pluto-three-words.qPVE2k/run/analysis_step_366/analysis.json).

### `corse`

Generated word bytes: `[2154, 2159)`.
Native pieces: `' cor'` + `'se'`.
The following token `';'` (ID 26, step 702) confirms the complete word boundary.

| Residual stage | Step 700: `' cor'` | Step 701: `'se'` |
| --- | ---: | ---: |
| Input + position | 1.2636e-42% / 40566 | 2.44165e-25% / 205 |
| B0 attention | 1.6029e-37% / 44126 | 1.1878e-22% / 33993 |
| B0 MLP | 1.19207e-06% / 4430 | 0.906455% / 18 |
| B1 attention | 3.75408e-06% / 5078 | 3.84987% / 6 |
| B1 MLP | 1.22141e-06% / 5577 | 5.50562% / 4 |
| B2 attention | 3.14152e-06% / 5305 | 6.4857% / 4 |
| B2 MLP | 2.8526e-06% / 3587 | 8.88054% / 4 |
| B3 attention | 8.72974e-06% / 2723 | 24.345% / 1 |
| B3 MLP | 1.10398e-05% / 1594 | 36.4683% / 1 |
| B4 attention | 1.41657e-05% / 766 | 52.4548% / 1 |
| B4 MLP | 0.000786632% / 225 | 92.6311% / 1 |
| B5 attention | 0.00209783% / 120 | 90.0372% / 1 |
| B5 MLP | 0.00896736% / 50 | 99.7003% / 1 |
| B6 attention | 0.00242589% / 73 | 99.749% / 1 |
| B6 MLP | 0.0565668% / 34 | 99.9918% / 1 |
| B7 attention | 1.21825% / 11 | 99.9909% / 1 |
| B7 MLP | 11.8961% / 2 | 99.9888% / 1 |

First rank-one readouts:

- Step 700, `' cor'`: never rank one.

- Step 701, `'se'`: B3 attention.

| Removed branch | Step 700 | Step 701 |
| --- | ---: | ---: |
| No removal | 11.8961% / 2 | 99.9888% / 1 |
| B0 attention | 0.000695957% / 2062 | 95.4126% / 1 |
| B0 MLP | 0.000162052% / 5014 | 5.56945e-05% / 7907 |
| B1 attention | 0.00249836% / 1277 | 69.6933% / 1 |
| B1 MLP | 0.437112% / 31 | 99.9798% / 1 |
| B2 attention | 0.00720815% / 1011 | 99.86% / 1 |
| B2 MLP | 7.63858% / 2 | 99.9956% / 1 |
| B3 attention | 0.0500098% / 148 | 99.9644% / 1 |
| B3 MLP | 6.2745% / 2 | 99.9934% / 1 |
| B4 attention | 0.0137695% / 236 | 99.9581% / 1 |
| B4 MLP | 0.395684% / 14 | 99.8178% / 1 |
| B5 attention | 3.86589% / 6 | 99.9919% / 1 |
| B5 MLP | 5.86458% / 3 | 96.9994% / 1 |
| B6 attention | 0.479176% / 20 | 99.9923% / 1 |
| B6 MLP | 47.0027% / 1 | 95.89% / 1 |
| B7 attention | 1.05511% / 11 | 99.9533% / 1 |
| B7 MLP | 1.21825% / 11 | 99.9909% / 1 |

Full head/branch interventions and analytical term ledgers: [step 700](/tmp/pluto-three-words.qPVE2k/run/analysis_step_700/analysis.json), [step 701](/tmp/pluto-three-words.qPVE2k/run/analysis_step_701/analysis.json).

### `Exeunt`

Generated word bytes: `[4030, 4036)`.
Native pieces: `' Ex'` + `'e'` + `'unt'`.
The following token `' sever'` (ID 1750, step 1343) confirms the complete word boundary.

| Residual stage | Step 1340: `' Ex'` | Step 1341: `'e'` | Step 1342: `'unt'` |
| --- | ---: | ---: | ---: |
| Input + position | 3.99189e-21% / 15 | 1.04221e-25% / 68 | 2.34531e-28% / 31 |
| B0 attention | 5.07452e-14% / 23 | 2.59115e-18% / 17 | 6.70226e-25% / 61 |
| B0 MLP | 5.15625e-06% / 902 | 99.965% / 1 | 99.9999% / 1 |
| B1 attention | 0.00073971% / 775 | 97.8869% / 1 | 99.9993% / 1 |
| B1 MLP | 0.000206483% / 898 | 99.6687% / 1 | 99.9995% / 1 |
| B2 attention | 0.0419292% / 4 | 85.1472% / 1 | 99.9972% / 1 |
| B2 MLP | 0.067867% / 3 | 91.2178% / 1 | 99.9994% / 1 |
| B3 attention | 0.560065% / 3 | 99.9456% / 1 | 99.9999% / 1 |
| B3 MLP | 2.32359% / 3 | 99.9636% / 1 | 99.9999% / 1 |
| B4 attention | 3.43865% / 2 | 99.9916% / 1 | 100% / 1 |
| B4 MLP | 8.5711% / 3 | 99.9993% / 1 | 100% / 1 |
| B5 attention | 63.5764% / 1 | 99.9991% / 1 | 100% / 1 |
| B5 MLP | 77.5574% / 1 | 99.9996% / 1 | 100% / 1 |
| B6 attention | 85.0002% / 1 | 99.9995% / 1 | 100% / 1 |
| B6 MLP | 77.7427% / 1 | 99.9999% / 1 | 100% / 1 |
| B7 attention | 95.5058% / 1 | 100% / 1 | 100% / 1 |
| B7 MLP | 21.0257% / 2 | 99.9999% / 1 | 100% / 1 |

First rank-one readouts:

- Step 1340, `' Ex'`: B5 attention.

- Step 1341, `'e'`: B0 MLP.

- Step 1342, `'unt'`: B0 MLP.

| Removed branch | Step 1340 | Step 1341 | Step 1342 |
| --- | ---: | ---: | ---: |
| No removal | 21.0257% / 2 | 99.9999% / 1 | 100% / 1 |
| B0 attention | 34.412% / 2 | 98.3315% / 1 | 99.9703% / 1 |
| B0 MLP | 0.00322132% / 1566 | 2.35099e-06% / 49733 | 1.15358e-05% / 46568 |
| B1 attention | 0.00420686% / 2 | 99.9161% / 1 | 100% / 1 |
| B1 MLP | 11.7736% / 2 | 99.9955% / 1 | 100% / 1 |
| B2 attention | 0.260423% / 3 | 99.9994% / 1 | 99.9996% / 1 |
| B2 MLP | 2.89407% / 2 | 99.9999% / 1 | 100% / 1 |
| B3 attention | 3.46238% / 2 | 99.9225% / 1 | 100% / 1 |
| B3 MLP | 7.26447% / 2 | 99.9998% / 1 | 100% / 1 |
| B4 attention | 16.8388% / 2 | 99.9998% / 1 | 100% / 1 |
| B4 MLP | 19.5908% / 2 | 99.9947% / 1 | 100% / 1 |
| B5 attention | 13.5658% / 2 | 99.9997% / 1 | 100% / 1 |
| B5 MLP | 54.8546% / 1 | 99.9976% / 1 | 100% / 1 |
| B6 attention | 36.8556% / 2 | 99.9998% / 1 | 100% / 1 |
| B6 MLP | 83.2849% / 1 | 100% / 1 | 100% / 1 |
| B7 attention | 12.9951% / 2 | 99.9996% / 1 | 100% / 1 |
| B7 MLP | 95.5058% / 1 | 100% / 1 | 100% / 1 |

Full head/branch interventions and analytical term ledgers: [step 1340](/tmp/pluto-three-words.qPVE2k/run/analysis_step_1340/analysis.json), [step 1341](/tmp/pluto-three-words.qPVE2k/run/analysis_step_1341/analysis.json), [step 1342](/tmp/pluto-three-words.qPVE2k/run/analysis_step_1342/analysis.json).

## Corpus membership and tokenization differences

Counts use case-insensitive whole words, excluding apostrophe/hyphen-joined
fragments. The actual corpus split is byte **4,892,836**, a native boundary
after **1,650,781 tokens**. Same-ID counts additionally require the precise
generated native token sequence, including its leading-space token.

| Word | Full corpus | Training | Held out | Same IDs, full corpus | Same IDs, training |
| --- | ---: | ---: | ---: | ---: | ---: |
| grandam | 27 | 23 | 4 | 21 | 17 |
| corse | 29 | 26 | 3 | 29 | 26 |
| Exeunt | 1035 | 943 | 92 | 973 | 883 |

Representative **training-prefix** locations below include the first
same-ID match for each word and, where present, the first different-ID
match. These are examples, not an exhaustive occurrence table; the linked
qualification JSON contains every occurrence and its full/clipped byte pieces.
Byte/token intervals are zero-based and half-open; lines/columns are one-based.

| Word / spelling | Line:byte column | Word byte interval | Native token interval | Native IDs | IDs match generated? |
| --- | --- | --- | --- | --- | --- |
| grandam / `grandam` | [29872:9](/home/ubuntu/code/pluto/testdata/shakespeare.txt:29872) | [1281926, 1281933) | [430302, 430304) | `[4490, 321]` | yes |
| grandam / `Grandam` | [54652:5](/home/ubuntu/code/pluto/testdata/shakespeare.txt:54652) | [2391474, 2391481) | [797314, 797316) | `[5675, 321]` | no |
| corse / `corse` | [19446:23](/home/ubuntu/code/pluto/testdata/shakespeare.txt:19446) | [829657, 829662) | [277641, 277643) | `[1162, 325]` | yes |
| Exeunt / `Exeunt` | [2757:41](/home/ubuntu/code/pluto/testdata/shakespeare.txt:2757) | [108337, 108343) | [32697, 32700) | `[1475, 68, 2797]` | yes |
| Exeunt / `Exeunt` | [4963:35](/home/ubuntu/code/pluto/testdata/shakespeare.txt:4963) | [207988, 207994) | [65413, 65416) | `[3109, 68, 2797]` | no |

Corpus membership does not establish which historical minibatch caused
a weight update or identify one passage as the unique source of a generated word.
The capitalization/spacing-sensitive token counts must not be conflated with
case-insensitive lexical counts. In particular, `Exeunt` is frequent within
this corpus despite being specialized outside theatrical usage.

## Evidence and exact presentation checks

The generator requires all **10** native runs (seven word pieces and three
boundary tokens) to finish successfully with unchanged pinned inputs and
output hashes. It directly reconstructs probabilities, target ranks, sampled
IDs and target CDFs from **686 saved FP32 logit rows**: seven baselines,
119 intermediate readouts, and all 560 head/branch removals. Baseline logits
are also compared byte-for-byte with the original autoregressive run.
The displayed tables preserve every one of the 119 lens and 112 whole-branch
probability/rank cells, seven baseline events/CDFs, and representative corpus
locations. All raw probabilities and 448 individual-head removals remain in JSON.

This presentation check is not a second GPU run or a new analytical test of
every hidden-state mathematical term. It independently checks the raw-logit
facts used in these tables and verifies that this report regenerates exactly.

Selected source SHA256 identities:

- `run/plan.json`: `5a65a1c151bfd31c8626ca26ef366893325ecd5634cf0495267d75274cd9fb1a`.
- `run/result.json`: `3e478f497a340ea2d63a5e1ac082cc46925d645a105d6a9856ce159ccedfc376`.
- `lexical_corpus_audit.json`: `8f135ad8a95f92395fb7eec46c1a35ec85e8ee3787a957d11a988f8a8be21fd6`.
- `run/analysis_step_365/analysis.json`: `9bc43ae4eadc0a8f5d488e60ff3deb2d80094262b2d93ffc15a236481c4dc007`.
- `run/analysis_step_366/analysis.json`: `b2761938ccaffec826cd025d203415e875adcf3afccbc5c6d1dc050f8870ff79`.
- `run/analysis_step_700/analysis.json`: `e79d8709903e55c89153e55e6953254eb2abeac82e55445c2849df4419aedae5`.
- `run/analysis_step_701/analysis.json`: `dc2ab01f64e0322298e12e7a129b89b525479b523bec36e4a8df2e1016100ab1`.
- `run/analysis_step_1340/analysis.json`: `0c5ed5910644bc10e5755f26c7372b37c77021f7ccbeef835d1dde70814e84e5`.
- `run/analysis_step_1341/analysis.json`: `af9a4c38eeed4be7138a80a25d44e0c2409505de9d75259f6fa32d3f7be93de8`.
- `run/analysis_step_1342/analysis.json`: `e610a9b01263013d52d69bd7d07a834c150c280a028815cf11c19627c37018c1`.

The [presentation helper](/tmp/pluto-three-words.qPVE2k/build_readouts_report.py) verifies
the current Markdown with:

```text
/home/ubuntu/.venv/bin/python /tmp/pluto-three-words.qPVE2k/build_readouts_report.py
```
