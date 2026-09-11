# Exeunt restoration repairs both suffix transitions

The main analysis below is a per-token reanalysis of the completed step-331
outer intervention. It separates the two transitions hidden by a single
conditional suffix probability. The final section separately records partial
measurements from the subsequent block-localization experiment.

Status at 2026-09-10 22:53:33 UTC: the complete Exeunt-restoration cube and
its controls are certified. The reciprocal Nuveth-restoration cube is still
running. The early-block contribution is causally established in these
transplants, but attention versus MLP contributions remain unresolved.

## Fixed intervention and measurement

The recipient A was trained on the amended Nuveth corpus; donor D was trained
on the original Shakespeare corpus. Both completed 331 matched optimizer
steps. E replaces the 11 selected embedding rows in both input and tied-output
roles. C replaces all 99 non-token-embedding tensors: learned positions, all
eight transformer blocks, and final LayerNorm. EC combines those replacements.
Unselected token-embedding rows remain A in E, C, and EC. No retraining occurs.

The leading-space tokenizer sequence is `[1475, 68, 2797]`, or
`[" Ex", "e", "unt"]`. The middle-token probability conditions on the supplied
`" Ex"`; the final-token probability conditions on the supplied `" Ex", "e"`.
These are temperature-one, teacher-forced probabilities at known occurrences,
using the original-prefix domain and 128-token prefixes with local positions
reset to zero. They are not free-generation success rates or corpus-wide
probabilities. Bare-title and lowercase cases are not included in these tables.

## Held-out contexts

Geometric mean probabilities across the 15 leading-title test contexts:

| Predicted piece | E: rows | C: other computation | EC: both | D: full donor |
| --- | ---: | ---: | ---: | ---: |
| ` Ex` | .001410510 | .000084074 | .002036518 | .002144026 |
| `e` | .003716523 | .074505031 | .969158393 | .966231815 |
| `unt` | .000884886 | .005610940 | .782431687 | .771045096 |
| Joint `e`, `unt` suffix | .000003288699 | .000418043 | .758300236 | .745008302 |

Compared with E, EC improves mean log probability by **5.563640 nats for `e`**
and **6.784703 nats for `unt`**. Compared with C, the gains are 2.565561 and
4.937688 nats. Thus both conditional predictions are repaired, with the larger
log-probability change at the final `unt`. The initial piece remains far less
probable than either supplied-prefix suffix transition.

## Training contexts

The corresponding 14 leading-title training contexts give the same pattern:

| Predicted piece | E | C | EC | D |
| --- | ---: | ---: | ---: | ---: |
| ` Ex` | .001418885 | .000084291 | .002040009 | .002142295 |
| `e` | .003791603 | .073109315 | .966758297 | .963876449 |
| `unt` | .000931007 | .005365935 | .781468782 | .770194186 |
| Joint `e`, `unt` suffix | .000003530009 | .000392300 | .755491429 | .742372038 |

## Interaction is present in both predictions

For each metric, define the interaction as `EC - E - C + A`. The rival is the
highest-scoring A competitor, held fixed across intervention cells separately
for each prediction. Target-versus-rival margin minus the log partition
relative to that same rival equals log probability. This is an exact score
decomposition, not a percentage allocation of where the word is stored.

| Split and piece | Margin interaction | Relative-normalizer interaction | Log-probability interaction |
| --- | ---: | ---: | ---: |
| Training `e` | 4.764091 | 3.835728 | .928362 |
| Training `unt` | 2.349752 | 1.053577 | 1.296175 |
| Test `e` | 4.818289 | 3.881483 | .936807 |
| Test `unt` | 2.376839 | 1.104826 | 1.272013 |

The nonzero margin interactions rule out describing these effects solely as
softmax-normalization changes. They do not identify an individual block,
attention head, neuron, or unique storage location. C is a broad intervention,
and each suffix position is conditioned on its own supplied preceding tokens.
Shared-token collateral effects from the outer experiment still apply; this
is not a demonstrated word-selective edit.

## Input and output roles on the fixed donor-computation background

A separate, already completed factorial holds the 99 non-embedding tensors
at the Exeunt donor's values. Here its AA baseline is outer C, and its JJ cell
is outer EC. The off-diagonal cells use the selected donor rows only as output
dictionary entries (AJ), or only as input vectors (JA). These cell labels are
not the outer E/C factors. All four retain the recipient's unselected rows.

Geometric probabilities over the same 15 held-out leading-title contexts:

| Prediction | AA: neither role | AJ: output only | JA: input only | JJ: both roles |
| --- | ---: | ---: | ---: | ---: |
| ` Ex` | .000084074 | .002035302 | .000084149 | .002036518 |
| `e` | .074505031 | .580901107 | .074980309 | .969158393 |
| `unt` | .005610940 | .064176837 | .063658276 | .782431687 |
| Joint suffix | .000418043 | .037280396 | .004773117 | .758300236 |

Output rows supply most of the initial-piece improvement and a substantial
middle-`e` improvement. Input-only changes barely improve `e` here, but help
the final `unt`. Neither one-role swap restores donor-like suffix completion;
both together do. This conclusion is conditional on this particular
donor-computation background and these 11-row interventions, not evidence
that any individual row is uniquely necessary.

The input/output log-probability interaction at `e` is .515608 nats on
training contexts and .505489 on test contexts. At `unt` it is smaller:
.082604 and .071953 nats, respectively. Thus the combined final-token gain
is nearly additive in log probability, despite the large probability gain.

Initial-piece input-only effects are exactly absent in all 14 training
contexts and 12 of 15 test contexts: AA/JA and AJ/JJ native logits are
byte-identical there. The other three test prefixes already contain a
selected token: `N` (45) once, or `th` (400) in two different prefixes. A
blanket claim that input rows cannot affect initial-piece prediction would
therefore be false. Before the middle prediction the supplied ` Ex` is also
an edited input; before the final prediction the supplied `e` is as well.

Source: `outer_factorial_20260910_2123/original_to_replacement/outer_main/readout.json`
under the amendment directory below, SHA-256
`087b8a31197e6ce6775d25af0105d1863c4c6e5493d4f49b61e931f1ce29e9d3`.
The audit reverified direct artifacts and independently recomputed 348
native-logit scores with zero discrepancy. It also checked the recorded
patched-input exposure against the packed token batch. No new GPU forward
was required for this reanalysis.

## Reproducible source selection

The source directory is
`/home/ubuntu/checkpoints/exeunt_nuveth_deterministic_20260910_1137/lowercase_amendment`.

| Source | SHA-256 |
| --- | --- |
| `outer_original_to_replacement_main_mechanism_20260910.json` | `093761e7e5297a80d6666e13d79f71548709869923f6a27287adcfa36ccc5847` |
| `outer_original_to_replacement_main_summary_20260910.json` | `58c0ab674ca4466b672937c904abb01968455f4019acfa7756c0a8f143da70e5` |

Select mechanism `per_case` entries with `kind="word"`,
`prefix_domain="original"`, `target="Exeunt"`, and
`target_ids=[1475,68,2797]`; split training and test. Use
`tokens[position].cells[cell]` for scores and
`tokens[position].effects.{log_probability,margin,normalizer}.interaction`
for the decomposition. Average log probabilities within each split, then
exponentiate; do not divide the two-token suffix log probability by two.

The independent read-only audit verified all 1,771 source records and
recomputed 435 selected cell scores from native logits. The largest score
discrepancy was `8.88e-16`. That audit did not run new GPU forwards.

## Next causal measurement

The separately frozen plan in `complement_localization_20260910_2150/plan.json`
holds E fixed and partitions C into Q (positions and final LayerNorm), H
(blocks 0-3), and L (blocks 4-7). Its eight-cell factorial is measured in both
directions, with main and supplemental suites and preserved-versus-independent
E/EC copy controls: 40 native measurements in total.

It started on 2026-09-10 at 22:12:04 UTC as
`pluto-exeunt-complement-20260910-2212.service`, writing to the new sibling
`complement_localization_run_20260910_2212`. At 22:19:31 UTC, both preserved
references had completed both suites; independent-copy measurements were
underway. At that point no block-localization result was available. The
broader mechanistic goal remains open.

At 22:26:20 UTC, both independent E/EC copies had also completed and matched
all native loss and argmax bits: 192,512 main and 271,360 supplemental rows
per model. The CPU-only baseline snapshot completed successfully at 22:29:15
UTC in `complement_partial_original_to_replacement_E_EC_20260910_2226.json`
(SHA-256 `65af25e582a030ddbfecf42a700cee1ede54f2ced091498d13967f9df2c3dc20`).
Its 3,024 recorded files and separately bounded FP64-reference comparisons
were checked. This was baseline calibration, not a new block result.

### First new cell: Q, preliminary partial readout

Q completed both native suites at 22:29:44 UTC. A direct diagnostic extraction
verified the completed cell's full artifact inventory before and after reading
its losses, and selected the exact cases/rows from the validated E/EC snapshot.
The full independent Q/H snapshot was pending at extraction time (its later
validation is recorded below); no eight-cell interaction or group-removal
result is inferred from this partial measurement.

| Exeunt conditional suffix | E | E + Q | EC |
| --- | ---: | ---: | ---: |
| Training, 14 leading-title contexts | 3.53000967e-6 | 3.74942423e-6 | .755491006 |
| Test, 15 leading-title contexts | 3.28869778e-6 | 3.49526706e-6 | .758300494 |

Thus donor positions and final LayerNorm alone add little on top of E; they
do not restore suffix completion in this recipient-transformer background.
That does not establish that Q is irrelevant when combined with donor
transformer blocks. The remaining factorial cells test that distinction.
The small difference from the earlier tables reflects native FP32 loss
scoring versus the older FP64 logit-based readout, not a changed checkpoint.

Q sources under the running experiment's `original_to_replacement/cells/Q`:
`complete.json` SHA-256
`05a7841768d4101d04cccf099570275ee7d73f1ae9977f68bdff2d74a03bb188`;
`main/scores/losses.f32.bin` SHA-256
`dd0cee974a91be6d14a32c8973add333d233c64ac18328db8da14f1443f060e7`.

### Early four blocks: H, preliminary partial readout

H completed both native suites at 22:33:09 UTC. The same completed-leaf
inventory check and selected-row extraction give the following leading-title
Exeunt probabilities. E remains fixed; H means adding donor blocks 0-3,
while positions, final LayerNorm, and blocks 4-7 retain recipient values.

| Split | First piece | Middle `e` | Final `unt` | Conditional suffix | Whole three-token sequence |
| --- | ---: | ---: | ---: | ---: | ---: |
| Training, 14 | .001891752 | .731101239 | .669247010 | .489287318 | .000925610271 |
| Test, 15 | .001848287 | .740661630 | .673472544 | .498815272 | .000921953910 |

This is a large functional contribution on the tested E-conditioned
background: the test suffix increases from .000003288698 to .498815272,
compared with .758300494 for EC. Both suffix predictions improve strongly.
The 49.9% value is a geometric conditional probability, not a fraction of
parameters storing the word or a fraction of causal responsibility.

Matched generic controls do not show a comparable average improvement.
The 16 training controls have mean three-token NLL 14.580963 under E,
14.628750 under H, and 14.634601 under EC; the 16 test controls give
9.398479, 9.426872, and 9.375232. These small averages do not rule out
individual-context or shared-piece collateral effects. The independent
E/EC/Q/H snapshot was still pending at extraction time; its subsequent
validation is recorded below. No necessity or unique-storage claim is made.

Source: `original_to_replacement/cells/H/complete.json` under the running
experiment, SHA-256
`03a0a9481935498d426a0486cca39e4744aaac2b69f7d8bcdbf0a8cd363376c5`.

### Q/H independently validated; shared-token collateral

At 22:37:05 UTC, the independent CPU reader published
`complement_partial_original_to_replacement_E_EC_Q_H_20260910_2233.json`
(2,248,943 bytes, SHA-256
`0e5debef6c4ded3e96e2e576833af17f4c66e9e5e232fec1896f526fc6dd5634`).
It validates exactly E, EC, Q, and H, explicitly marks the other four cells
missing, and repeats model-byte, execution-ledger, cross-suite, and E/EC
baseline checks. Its Q/H probabilities exactly match the direct extractions.

On held-out leading-title Exeunt, adding H to E improves mean suffix log
probability by 11.929499 nats; the remaining gap from H to EC is .418844 nats.
Q's addition is only .060918 nats. These log-score contrasts are not a unique
or ordering-independent partition of causal responsibility.

The shared-token controls prevent a word-selective-edit claim. For unrelated
held-out continuations beginning with token `ve` (303), H changes mean
first-token log probability by -.870217 nats relative to E; for `th` (400),
the change is -1.357810 nats (eight contexts in each group). Training shared
`unt` (2797) improves by 1.351425 nats, also over eight contexts; there is no
held-out shared-`unt` group. These use the correct shared-token context domain,
not Exeunt occurrences or pooled, duplicated domain aliases.

Individual controls are more revealing than these means. In held-out
supplemental case 239, the ordinary continuation `"Pro" -> "ve true,"`,
H lowers `P(ve)` from .001266820 under E to .0000766221, about 16.5-fold;
EC gives .0000387967. In training shared-piece case 164, whose native targets
are `[68,13,198]`, the second-target period probability falls from .363249
under E to .0227886 under H (EC: .0191652). Thus there are substantial
individual collateral changes even when generic-control means move little.

Restoration is not monotonic across all metrics: held-out leading-title
probability of the exact fourth token after Exeunt is .003685869 under H,
slightly above EC's .003355640. The seven lowercase training contexts also
form a different regime: H's conditional suffix is .000179648, close to
EC's .000177822, but its full lowercase word probability is only about
`1.01e-8`. Neither this result nor the title-case results demonstrate
lowercase memorization. There are no held-out lowercase cases in this assay.

### Late four blocks: L, preliminary partial readout

L completed both suites at 22:36:33 UTC. On top of fixed E, adding only donor
blocks 4-7 gives:

| Split | First piece | Middle `e` | Final `unt` | Conditional suffix | Whole three-token sequence |
| --- | ---: | ---: | ---: | ---: | ---: |
| Training, 14 | .001548215 | .050561974 | .000658300 | .000033284928 | 5.15322293e-8 |
| Test, 15 | .001562736 | .051512934 | .000599867 | .000030900892 | 4.82899297e-8 |

Thus L increases the middle-token probability on this background but slightly
worsens the final token relative to E. The resulting held-out suffix remains
only .00309%, versus 49.9% under H. This does not show that L is unimportant
when H is present; the pair cells and reversals from EC are needed for that.
Mean three-token generic-control NLL under L is 14.579865 training and
9.364633 test, each over the same 16 generic contexts.

This diagnostic checked the completed L artifact inventory before and after
reading native losses using the validated baseline's exact cases and rows.
The full cube's independent readout was not yet available. Source marker:
`original_to_replacement/cells/L/complete.json`, SHA-256
`b10f5e95f6d73961572a15783103c88173c64be82b2f3a931baec0bebdae69c8`.

### Reverting early versus late weights from EC: preliminary pair readout

The completed QH and QL cells permit two direct reversals from EC. These
replace an entire group's weights with recipient values; they do not remove
layers, zero their outputs, or alter model depth. All selected embedding rows
remain donor in both roles. The recipient's other embedding rows remain fixed.

| Conditional Exeunt suffix | Training, 14 | Test, 15 |
| --- | ---: | ---: |
| EC: all three donor groups | .755491006 | .758300494 |
| QH: revert late blocks 4-7 | .499854891 | .509663608 |
| QL: revert early blocks 0-3 | .000032148511 | .000029538534 |

Within this restored hybrid, reverting early weights nearly eliminates its
strong suffix completion, whereas reverting late weights retains roughly
51% conditional probability on the held-out contexts. Together with the
addition contrasts, this establishes a much stronger functional contribution
from H than L in the tested coordinate-defined interventions. It does not
identify a unique storage site or yet separate attention from MLP computations.
Q's reversal and all complete-cube interactions still await HL and the full
readout; the reciprocal transfer direction is also still pending.

The diagnostic uses completed-cell inventories checked before and after
reading the exact native rows. The source marker SHA-256 values are:
QH `707da343fc24db7035288cd74c0db8fdecead5fbb7841f0415605fc41dbc7b7d`;
QL `8e921808219fbc23cdae968c1ccb7e6d1d4495ae0a74ed5e0ab8aab18a2746af`.

### All eight Exeunt-restoration cells measured

HL completed at 22:46:48 UTC, bringing this direction to 20 native
measurements including its four preserved-reference runs. An inventory-checked
diagnostic of all eight completed cells gives the complete leading-title
suffix table below. The owning full readout was still validating when this
diagnostic was recorded; this does not certify the reciprocal direction or
the full 40-measurement run.

| Cell (E fixed) | Training, 14 | Test, 15 |
| --- | ---: | ---: |
| E | .000003530010 | .000003288698 |
| Q | .000003749424 | .000003495267 |
| H | .489287318 | .498815272 |
| L | .000033284928 | .000030900892 |
| QH | .499854891 | .509663608 |
| QL | .000032148511 | .000029538534 |
| HL | .746369199 | .749000251 |
| EC = QHL | .755491006 | .758300494 |

Held-out suffix log-score effects, in nats:

| Group | Add to E | Revert from EC (`score(EC) - score(EC-G)`) |
| --- | ---: | ---: |
| Q | .060918 | .012340 |
| H | 11.929499 | 10.153139 |
| L | 2.240293 | .397329 |

The H/L log-score interaction is -1.833790 with Q absent and -1.736957 with Q
present. The three-way contrast is .096833 nats. These loss-based contrasts
do not separate target-logit effects from softmax normalization and are not
percentages of storage or responsibility. In particular, a negative
log-probability interaction need not mean an antagonistic internal circuit.

HL's complete marker SHA-256 is
`2481e66e9a7fbf124eeda9f83c8840b5592f7b6b34372a2446f0db225b61c66c`.
Every transfer here tests differences between the two arms' learned weights.
A small transfer effect does not establish that the underlying position,
normalization, or transformer computation is dispensable: it can still be
essential and largely shared between arms.

## Certified first-direction readout

At 22:53:33 UTC the owning runner completed the full first-direction audit
and advanced to the reciprocal direction. Its readout contains 453 raw cases,
94 separately labeled groups (including overlapping domain views), and 3,086
file records. The final leading-title probabilities and contrasts reproduce
the completed-cell diagnostics above; small final-digit differences in a few
training values reflect summation order, not different native losses.

Authoritative artifacts below
`complement_localization_run_20260910_2212/original_to_replacement/`:

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| `cube_readout.json` | 11700504 | `fbde18ebd5b275165a38be1dead3a40b77d41bcdc6657665c70902b51f1a107d` |
| `baseline_controls.json` | 4354 | `8b6e496d755436aec0fc3eacc4b959224ea61dce9569c3681502716670d6e4aa` |
| `complete.json` | 295389 | `62de29f57f7a6a93ca89d40deeecba364de032903c7024b61fe911ee1dac1b28` |

The direction completion marker binds both the numerical readout and the
separate baseline certification. The core numerical reader deliberately does
not claim ownership of the runner's baseline controls. The completion
inventory and those bindings were independently checked after publication.
This certifies one direction, not the complete two-direction experiment or
the broader mechanistic goal.

### Complete-word generation remains a separate limitation

The semantic audit independently matched all 12,120 selected native NLL and
argmax values in the full readout, and recomputed 42 alternative-stratum
aggregate metrics. In every one of the eight hybrid cells, first-token
argmax success is **0/39** for original-prefix Exeunt contexts (29 leading
title, three bare title, seven lowercase). Consequently, complete-word
greedy success from the before-word prefix is also 0/39. H-containing cells
get both suffix tokens right in all 32 title cases, but no lowercase case.

This is not only a hybrid-model defect: in the earlier outer mechanism
report, the complete original-trained donor's first-token logit is below the
fixed A rival in all 39 cases. The donor's first-token margins range from
-6.367315 to -1.257908 across the 32 title cases and from -9.891095 to
-5.264329 across the seven lowercase cases. A negative margin to even this
one rival rules out first-token argmax success, regardless of whether that
rival is also the donor's highest-scoring token.

The tested models therefore provide evidence about learned conditional
spelling and changed whole-word probabilities, not reliable generation of
Exeunt from these before-word prompts. This limitation is specific to the
declared 128-token-prefix assay, not a claim about every possible prompt.

Bare-title Exeunt suffix probabilities under H are .096522 training (two
cases) and .117879 test (one case); HL raises them to .501412/.543797 and EC
to .517539/.560614. H whole-word probabilities remain .000354551/.000590472,
versus .00394480/.00629262 under EC. All cells already prefer Exeunt to
Nuveth in these three contexts, including E with only approximately `1e-8`
whole-word probability. Again, relative preference is not confident completion.

For lowercase, every H-containing cell flips all seven preferences to
exeunt, but none without H does. H whole-word probability is `1.00797e-8`
versus `1.93604e-12` for nuveth; EC gives `9.44834e-9` versus `1.68830e-12`.
All seven have the final `unt` as argmax under H/EC, but none has the correct
middle `e`. The lowercased word is therefore still not reliably spelled.

## Preserving sources before the requested sharing cleanup

The new sibling `pre_sharing_sources_20260910_2254/` preserves 255 analysis
package files (3,573,580 bytes) as read-only copies, with original-to-archived
path records. All 26 repository source inputs frozen by the running
experiment match exactly. Sources were verified before and after copying;
the original workspace files were not moved or rewritten.

Its `manifest.json` is 149,190 bytes, SHA-256
`ab2123f405d4a8afe263167412cce6f0d7aec527c7f837c96587de5fc16afc24`.
This source preservation is not a completed-analysis certificate, nor does
it pretend that relocated code is the version that executed this experiment.
At snapshot creation, the requested real directory moves were deferred until
the run exited.

## Completed reciprocal cube

The owning runner certified the reciprocal direction at 23:34:33 UTC and
published the overall completion record at 23:34:44 UTC on 2026-09-10. The
service then exited successfully (`MainPID=0`, `SubState=exited`, exit status
zero). All 40 distinct native measurements and both directions' baseline
controls passed; there is no failure record and training was not restarted.

Authoritative artifacts under `complement_localization_run_20260910_2212/`:

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| `replacement_to_original/cube_readout.json` | 11696597 | `b565ccede422e10ebe490985fae613fba75033143afc67f541ebef801980f099` |
| `replacement_to_original/baseline_controls.json` | 4352 | `a847bdb0fb2d5a5d199341a0c43c7080b89bb6ccb0eb626d79002aaaffb21033` |
| `replacement_to_original/complete.json` | 295389 | `132186403971ab5a681f35fa1bdcf12204713c67a1eaa6f6f4f7943ddd7213ad` |
| `summary.json` | 1200553 | `8e25efee3f61e9ba9ef2cd7eb39a9a56f89de296ba6b87e5c60fb2b92c4fcf84` |

The reciprocal direction transfers Nuveth-trained weights into the
Exeunt-trained recipient. The following Nuveth suffix geometric probabilities
use only the **original-prefix domain**, with 14 training and 15 test
leading-space title-case contexts. They do not pool the alternative prefix
domains or different tokenizations.

| Cell | Training | Test |
| --- | ---: | ---: |
| E | .000006296758 | .000005543108 |
| Q | .000007307687 | .000006457820 |
| H | .485471982 | .486622878 |
| L | .000114179651 | .000096236650 |
| QH | .493066502 | .494864207 |
| QL | .000144279622 | .000122548473 |
| HL | .878738463 | .884652493 |
| EC | .882312849 | .888169077 |

The early four blocks again dominate the transfer of substantial conditional
suffix probability. This is not a necessity result for merely preferring one
candidate word: L alone flips Nuveth over Exeunt in all 29 leading contexts,
despite its very low absolute confidence. Held-out whole-word probability is
approximately `1.80382e-7` under L, versus `.000855795` under H and
`.002023897` under EC. Every cell still misses the first Nuveth token in all
39 original-domain cases, so complete-word greedy success remains 0/39.

Bare Nuveth tokenizes as `[45, 45177, 400]` (`N`, `uve`, `th`), unlike the
leading-space spelling `[21733, 303, 400]` (` Nu`, `ve`, `th`). Its suffix
probability under EC is only `.000148149` training (two cases) and `.000152808`
test (one case); lowercase has `.000483721` across seven training cases.
Neither stratum has a correct middle-token argmax. The full Nuveth-trained
donor itself has similarly weak bare/lowercase suffix probabilities and
first-token success 0/39. These are limitations of the donor's learned
behavior in this assay, not solely hybrid-transfer failures. The prior donor
scores use FP64 softmax of saved native logits; the cube uses native FP32
losses, so tiny numerical differences are expected.

The control results also remain nonselective. For example, H reduces the
first-token log probability of ordinary shared `unt` continuations by about
2.929 nats on average over eight training cases (there are no held-out `unt`
controls). Small generic-control averages do not imply that these broad
weight transfers preserve unrelated text behavior.

This completes the fixed two-direction experiment, **not** the broader
mechanistic goal. It localizes a strong learned-weight interaction to the
early-block group in the tested background; it does not yet distinguish its
attention and MLP computations or identify a unique storage location for the
word. The source snapshot above preserves the code that actually ran, before
the requested sharing cleanup relocates the current tools and reports.
