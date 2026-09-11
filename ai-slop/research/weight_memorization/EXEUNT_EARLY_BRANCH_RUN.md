# Early attention/MLP experiment: execution record

This is an ongoing experiment under the fixed
[early-branch protocol](EXEUNT_EARLY_BRANCH_PROTOCOL.md), not a completed
mechanistic explanation. The lowercase replacement amendment is retained.
The preceding goal turn made progress by implementing and testing this
experiment and preserving the historical evidence across directory moves.

## Verified launch

The new service `pluto-exeunt-earlybranches-20260911-0016.service` started at
2026-09-11 00:16:53 UTC from repository commit `bb1f2e8`. Its controller PID is
1997096, with Linux process start ticks 130844850. It writes to:

```text
/home/ubuntu/checkpoints/exeunt_nuveth_deterministic_20260910_1137/lowercase_amendment/early_branches_run_20260911_0016
```

The service uses `Restart=no`; it must not be relaunched or overwritten merely
because an observation times out. `RemainAfterExit=yes` means an active service
alone does not prove a live process. Check `MainPID`, process identity, and
`SubState`, and inspect `failure.json` or the final summary as appropriate.

The controller repeats the completed predecessor's handoff audit. Before
launch, a separate read-only production preflight verified 4,016 historical
bindings and both main/supplemental E native ledgers through the explicit
historical-source archive. The predecessor had exited successfully, and the
GPU was idle. No training was restarted and no existing checkpoint changed.

The new request freezes 4,081 physical input records and preserves 32 source,
test, and protocol copies in `source/`. The executing Python modules and tests
must remain byte-identical while this run is active. This execution note is
not among those frozen inputs.

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| New run `request.json` | 1,443,322 | `a25f2742efebb4e5413fad29f8ccfc42189058d5fd51a279d16ba0da1efffc8c` |
| Fixed protocol | 5,447 | `7b9065fd192970144022c3b7e6f300f0aa569cbab932e05dfe9efb901a362482` |

The same committed source passed 1,691 weight-analysis Python tests and six
utility tests. All 141 optimized Bazel targets built successfully; 53 Bazel
tests passed from cache after their preceding successful executions.

## First observed native execution

At 00:17:54 UTC the controller launched loss-probe child PID 1997442 for
`original_to_replacement/E/main`. Its actual command selected this new run's
E checkpoint, the original frozen `word_cases/packed_cases.bin`, and
`--batch_sequences=1`. GPU process inspection subsequently observed that
exact native child using 1,286 MiB on the GPU. This verifies native execution,
not completion or successful numerical controls. No new attention-versus-MLP
finding was available at this observation.

At 00:22:07 UTC the first E cell completed both suites (2/32 measurements).
An independent read-only check verified its 112-record completed inventory
before and after reading, and compared the complete native loss/argmax file
hashes against the historical E cell. They match exactly across all 192,512
main and 271,360 supplemental positions, including padding. The controller's
FP64 reference checks cover 564 main, 624 following-token, and 327 shared-piece
predictions; their maximum absolute NLL differences are respectively
`5.043787719927195e-6`, `6.099647372082018e-6`, and `5.167061874367107e-6`.
The fixed tolerances are unchanged. This is calibration, not new branch-level
evidence. The controller proceeded to AM, the old H counterpart.

The order in each direction is E, AM, R, EC, A, M, AR, MR, with main and
supplemental measurements for every cell. E/AM/R/EC repeat completed reference
interventions; the later four cells supply the new branch-level evidence.
Completion markers are published only after both suites and their checks.

## Reading partial results without overstating them

Before all eight cells in one direction finish, `cube_readout.json` does not
exist. Use only completed `cells/<cell>/complete.json` inventories and their
authenticated scores and execution ledgers. Select each case's `scored_rows`
from the frozen case files; do not average context padding. The old
`paired_complement_partial_readout.aggregate` is specific to Q/H/L and must
not be applied to A/M/R labels.

The primary leading-title stratum has 14 training and 15 test contexts for
each candidate in the original-prefix domain. Keep per-piece, conditional
suffix, whole-word, following-token, and joint argmax results distinct. A
suffix is conditioned on supplied preceding word pieces, not freely
generated. Report both candidates' absolute probabilities as well as paired
preference changes; separate bare-title and lowercase cases.

Once A and M are complete, the recipient-R interaction is
`AM - A - M + E` in mean log probability. The donor-R interaction requires
AR and MR too: `EC - AR - MR + R`. Never impute missing cells. Generic
controls and individual shared-piece deteriorations must accompany the word
results. A includes LN1; M includes LN2; these weight-bundle effects are not
proof of unique storage or an internal interaction independent of softmax.

The broader goal remains active: explain the computations that support the
complete word, including contextual selection of its first token, rather
than stopping at a coarse bundle-level dependence.

## Partial-reader verification

The new `weight_analysis.paired_early_branches_partial` reader and its two
test files are separate from the live controller's frozen source set. At
00:34 UTC the full CPU analysis suite passed 1,718 tests in 53.912 seconds,
including 17 focused partial-reader tests and ten independent numeric tests.
These exercise genuine file inventories, source archives, owner/log bindings,
partial coverage, contradictory aliases, and same-event argmax conjunctions;
native execution is mocked in the pipeline tests, never launched by the
reader. Review caught and fixed rejection of the real plan's descriptive
`applies_to` tolerance field without changing either numeric tolerance.

A read-only production preflight validated the still-live request against
4,082 records and 4,016 historical logical identities with GPU visibility
disabled. It made no process-exit or GPU-idle query and published no output.
A second regression used the old completed cube's four inherited cells as
input data: each direction supplied 453 cases in 57 separate strata. All
6,208 mean-log/probability comparisons across the two directions matched the
old readouts exactly. E/AM/R/EC labels in this regression refer to old
E/H/QL/EC measurements; this is reader validation, not a fresh causal result.

Invoke the partial reader with `--run-directory`, `--direction` (one of the
two transfer directions), and `--output` naming a new immediate-sibling JSON
file outside the live run directory. Both E and EC must have completed. The
reader revalidates the completed cells and available anchor controls, preserves
missing cells explicitly, and does not claim successful controller exit.
Keep its source and tests unchanged while a snapshot is being validated.

## First new intervention: attention bundle alone

A completed both native suites at 00:39:18 UTC (10/32 measurements). A direct
diagnostic extraction checked complete-leaf inventories, the pinned owning
request, case/batch hashes, native owner/log membership, checkpoint/batch
metadata, and main/supplemental score agreement. All recorded files were
rechecked after aggregation. Full independent model-byte and historical-native
revalidation is still pending; these are explicitly preliminary results.

The donor is the original Exeunt model; the recipient is the amended Nuveth
model. E fixes eleven donor embedding rows in both tied roles. A transfers
early LN1/attention tensors; AM adds early LN2/MLP tensors. R is recipient
in E, A, and AM. In the original-prefix leading-title stratum:

| Split | Contexts | E suffix probability | A suffix probability | AM suffix probability |
| --- | ---: | ---: | ---: | ---: |
| Training | 14 | 3.530009670e-6 | 5.454395733e-6 | .4892873185 |
| Held-out | 15 | 3.288697781e-6 | 5.109213931e-6 | .4988152724 |

These are geometric conditional probabilities of the two supplied-prefix
suffix predictions, not whole-word generation rates. Held-out A predicts
`e` with probability .0043101572 and `unt` with .0011853892; its first-piece
probability is .0015034974, and its whole-three-token probability is
7.681689648e-9. A has zero correct first-piece, middle, final, joint-suffix,
or joint-word argmax events in all 29 leading-title contexts. AM gets both
suffix argmax predictions right in every context, but not the initial piece.

Thus early attention alone is insufficient for the large observed spelling
restoration on this background. Adding the early MLP bundle to A contributes
11.488945548 nats of held-out suffix log probability. M alone has not yet
completed at this extraction; this does not distinguish an MLP-alone effect
from attention/MLP cooperation, prove a unique storage location, or establish
word-selective collateral behavior.

The rival Nuveth also remains extremely unlikely under A in the same held-out
prefixes: its suffix probability is 8.514536739e-7 and its whole-word
probability is 1.787989121e-14. Favoring Exeunt over this rival must not be
confused with confidently spelling or freely generating Exeunt.

| A artifact, relative to `original_to_replacement/cells/A/` | Bytes | SHA-256 |
| --- | ---: | --- |
| `complete.json` | 35,388 | `fdfc0e48e87057ee3b6aeaf8156321216a192049088a06484faffb941cb4af2c` |
| `main/scores/losses.f32.bin` | 770,048 | `c495fb1cd9f4fb3fb7a3fbf7676fb6a4c967f185465a939fb237061578eb1e85` |
| `supplemental/scores/losses.f32.bin` | 1,085,440 | `16c5f3a6d19c368d27ea51b384ee93c6e17904d7abd6dac263c13a58f182cb22` |

At 00:40:48 UTC, service `pluto-exeunt-early-partial-20260911-0040.service`
started a full CPU-only snapshot audit: PID 2004361, start ticks 130988329.
Its intended new output is the amendment sibling
`early_branches_partial_E_AM_R_EC_A_20260911_0040.json`. GPU visibility is
disabled for this reader; the owning GPU experiment continues independently.
An active reader is not a certified snapshot: inspect its real process/exit
and the published output before claiming that independent validation passed.

## MLP bundle alone: first recipient-background comparison

M completed both suites at 00:42:46 UTC (12/32 measurements). The same direct
completed-leaf/owner/case/hash/cross-suite diagnostic checks passed for the
six available cells. Full independent certification of these M results is
pending; the already running five-cell snapshot does not include M.

| Split | Contexts | M first-piece probability | M middle `e` | M final `unt` | M joint suffix |
| --- | ---: | ---: | ---: | ---: | ---: |
| Training | 14 | .00176768049 | .7324259843 | .5412192000 | .3964030053 |
| Held-out | 15 | .00175040568 | .7470159330 | .5467961170 | .4084654115 |

M restores both suffix argmax predictions in all 29 leading-title contexts,
with zero first-piece or joint-word argmax successes. Its held-out whole-word
probability is .0007149801754. Nuveth's held-out suffix and whole-word
probabilities under M are 1.122315421e-8 and 1.620374099e-16, respectively.

M minus E contributes 11.729670842 nats of held-out suffix log probability.
Adding A to M contributes .199828593 nats, versus A minus E at .440553888.
The recipient-R A/M interaction is therefore -.240725295 nats (training:
-.224602733). Negative score nonadditivity does not prove antagonistic
internal computation, nor does the ratio of these effects assign percentages
of word storage to modules.

The supported narrower conclusion is that the **donor-specific early
LN2/MLP parameter changes**, combined with E, are sufficient for leading-title
suffix argmax restoration on this recipient background; the donor-specific
early attention parameter changes are not required for that result. Recipient
attention still computes normally. This does not show attention computation
is dispensable, separate LN2 from MLP projections, identify an individual
block/neuron, or establish full contextual word selection.

Generic three-token NLL remains close but slightly worse under M: training
14.626863658 versus E 14.580962807; held-out 9.454182059 versus E 9.398478717.
Those are averages over 16 controls per split, not a word-selectivity test;
individual shared-piece effects and other spelling strata must be retained.

M `complete.json` is 35,388 bytes with SHA-256
`f975afeed728045a5292d4f3dd42fdfca16b44a83d8770973bd1d5c6218fb0ee`.
Donor-background reversions and the reciprocal transfer direction remain
pending. Do not infer their outcomes from this first recipient-background
factorial.

An independent bounded diagnostic checked E/M inventories and case hashes
before and after reading the remaining strata. M restores joint suffix
argmax in all 32 title-case contexts, including two bare-title training cases
and one bare-title test case. Bare-title suffix probabilities are .0787766685
and .0947730386, respectively. First-piece and joint-word argmax remain 0/39.

Lowercase is not a successful full-spelling restoration: its seven training
cases have middle probability .0003290963, final probability .5652705670,
and suffix probability .000186028454. Final `unt` argmax is correct 7/7;
middle and joint suffix are correct 0/7. There are no lowercase held-out cases.

M has substantial individual shared-piece collateral despite small generic
mean changes. For `training:piece:303:start:1023605`, the ordinary `ve` target
drops from .001677074865 under E to .000120870128 under M, a change of
-2.63008975982666 nats. For `test:piece:303:start:77887`, it drops from
.001266819849 to .000093029162, or -2.61135196685791 nats. Both retain
incorrect argmax token 49. These are not word-selective edits. The independent
diagnostic did not perform full native revalidation; that distinction from
the ongoing snapshot audit remains explicit.

## Completed first direction and independent saved-evidence audit

The five-cell snapshot finished at 00:49:14.284588 UTC; its service exited
successfully. It contains E/AM/R/EC/A, not the subsequently completed M/AR/MR.
The JSON is 3,537,326 bytes with SHA-256
`0e9611fd78c4651f3fad840ccc69d01f684e33699c33c656702841566fa35fc8`.
All 4,951 recorded inputs were independently rehashed and its aggregates
recomputed exactly. Later pending statements above describe earlier times.

All eight first-direction cells subsequently finished. The direction's
completion marker was published at approximately 00:58:36 UTC, and the owner
proceeded to the reciprocal direction. A separate CPU-only audit verified the
exact 899-artifact direction inventory, all 4,982 unique referenced files
(16,325,286,389 bytes), all sixteen native command/input-owner/log bindings,
and all 453 cases directly against the saved loss and argmax arrays. All 94
saved aggregate groups recomputed exactly. All eight cross-suite checks and
the inherited E/AM/R/EC full-array hash comparisons passed. Recomputed E/EC
FP64 checks also equal the saved controls.

This is an independent audit of saved numerical evidence and ownership, not
a fresh native run or independent reconstruction of every patched checkpoint.
The full reader intentionally leaves `baseline_controls_certified=false` in
its numerical report; the enclosing direction marker separately binds and
certifies the controls. The five-cell snapshot omits derived probability and
argmax-match fields present in the full report, but all common case fields and
scores agree. There is no numerical discrepancy.

| Artifact under `original_to_replacement/` | Bytes | SHA-256 |
| --- | ---: | --- |
| `complete.json` | 280,069 | `d0d9bfe94f3eb4ed909137a284fa1b9ce0386cc6f4638e3783cc048ad301b8a1` |
| `cube_readout.json` | 12,286,991 | `047dfb7cff1dcf758a7f2ecb25d9f2e52228cc77d97b7ae3a854f2c57887c46a` |
| `baseline_controls.json` | 3,549 | `7f79617268e4e6940da756ae13041ce034ce3eff11761a46f9c366981ef6295e` |

The completed donor-background reversions sharpen the M/A distinction. For
the fifteen original-prefix held-out leading-title contexts:

| Cell | Exeunt conditional suffix | Exeunt whole three-token probability |
| --- | ---: | ---: |
| R | .0000295385340 | 4.665985196e-8 |
| AR (M reverted) | .0000485483537 | 8.146594456e-8 |
| MR (A reverted) | .6730461670 | .0013045121 |
| EC | .7583004937 | .0015442933 |

Reverting M from EC costs 9.656274732 nats of held-out suffix log probability;
reverting A costs .119265811. Training costs are 9.570387568 and .119460174,
respectively. The donor-R A/M interaction is -.377598826 nats on test and
-.374909265 on training. These are score interactions, not internal storage
fractions. R and AR get the middle `e` argmax right in all 29 leading-title
contexts, despite low absolute confidence; they fail the final `unt`.
M/AM/MR/EC get both suffix argmax predictions right in all those contexts.
Every cell still fails first-piece and whole-word argmax in all 39 cases.

The lower-case limitation persists even at EC: its seven training cases have
suffix probability .0001778223203. The M-only lowercase failure was not merely
missing the remaining donor nonembedding tensors. EC still has recipient
values in unselected embedding rows, so this is not a claim about the full
original donor's lowercase behavior.

Collateral grows under broader transfers. MR changes the ordinary `th`
prediction at `training:piece:400:start:311759` by -3.697737217 nats:
.003692155456 becomes .000091489891. At
`test:piece:400:start:57719`, it changes by -3.585101128 nats:
.002994049032 becomes .000083036545. EC's training deterioration at that same
`th` case is -3.721748829 nats. These are substantial individual errors, not
word-selective deletion or replacement.

The [next fixed split](EXEUNT_MLP_AFFINE_PROTOCOL.md) separates LN2 affine
parameters from the MLP projections/biases, keeping E fixed and attention
recipient. It will not start GPU measurements before the exact current
controller completes and exits. The reciprocal results, block/neuron
computation and contextual first-piece selection remain unresolved.

## Next-experiment implementation checks

At approximately 01:13 UTC the full CPU analysis suite passed 1,783 tests in
55.766 seconds. This includes 25 new exact-model tests, 24 numerical-readout
tests, and 16 runner/copy/failure tests for the LN2-affine versus MLP split.
Native execution is mocked in the runner tests; these are not N/F results.

A separate read-only production check validated the existing first-direction
E and M checkpoints through the new model helper. Their source/copy paths,
hashes, tensor selection, patch marker and all 300 per-model weight records
match the recorded model map exactly. Only the new descriptive M grouping
changes from ['M'] to ['N','F']; actual treatment bytes do not change. Request,
model map and completion marker hashes remained unchanged afterward.

Pre-launch review caught a current-versus-historical source-adapter error.
The fixed copy check verifies current-path records directly and checks
selections, source identities, hashes and distinct inodes. It does not send
new reader source identities through the pre-sharing historical archive.
Regression tests exercise that case. The separate native execution ledger
still records only the inputs actually used by its original producer.

No new N/F checkpoint or GPU measurement has been created. The next runner
requires the exact existing controller to exit with all 32 measurements and
bound completion evidence, followed by an idle-GPU check. A timeout is never
treated as permission to restart or duplicate either experiment.

## Reciprocal native measurements: Nuveth donor

All 32 native measurements finished by 01:30:58 UTC. The owner then entered
its reciprocal `cube_readout` phase; this observation does not claim final
summary publication or process exit. The following completed-leaf diagnostic
verified inventories, pinned case/batch hashes, owner/log membership and
metadata, cross-suite equality and post-read file hashes. It is not yet the
independent full-direction audit performed for the first direction above.

Now the donor is the amended Nuveth endpoint and the recipient is the original
Exeunt endpoint. E again fixes the eleven donor rows in both tied roles. In
the original-prefix leading-title stratum:

| Cell | Nuveth suffix, training (14) | Nuveth suffix, held-out (15) |
| --- | ---: | ---: |
| E | 6.296758481e-6 | 5.543107589e-6 |
| A | 8.965294480e-6 | 8.232437164e-6 |
| M | .4591691929 | .4588907743 |
| AM | .4854719820 | .4866228781 |
| R | .000144279622 | .000122548473 |
| AR | .000260063873 | .000238483789 |
| MR | .8697802803 | .8741947895 |
| EC | .8823128486 | .8881690768 |

Held-out M per-piece probabilities are first .001732485332, middle `ve`
.8857689232, and final `th` .5180705286; the whole-word product is
.0007950215356. M/AM/MR/EC each get joint suffix argmax correct 14/14 on
training and 15/15 on test. AR succeeds only once in each split; E/A/R never
do. All eight cells still have zero first-piece and whole-word argmax in
these leading-title contexts. Do not call the 45.9% suffix result a 45.9%
whole-word generation rate.

M minus E adds 11.324012216 nats of held-out Nuveth suffix log probability.
A minus E adds .395526822; adding A to M adds only .058677228. The
recipient-R interaction is -.336849594. Reverting M from EC costs
8.222616069 nats, while reverting A costs .015858905. The corresponding
donor-R interaction is -.649935786. Training effects are respectively
11.197139059, .353325980, .055702823, -.297623158, 8.129374708,
.014306068 and -.574867964 nats.

The rival Exeunt is suppressed under M in the same held-out contexts: suffix
probability 1.520893434e-8 and whole-word probability 1.157927438e-12, versus
E at .0003964074012 and 3.502276546e-8. This is conditional model behavior,
not evidence that an Exeunt representation was uniquely deleted.

Reciprocal restoration is not uniform across spellings. For bare Nuveth,
whose tokens are `N`, `uve`, `th`, M suffix probabilities are only
.0000192858030 (two training cases) and .0000155385286 (one test case).
Middle `uve` remains incorrect in all three, and final `th` is correct in
only one training case. Lowercase `nuveth` has middle probability
.000286853763, final .5188815938 and suffix .000148843138 across seven
training cases: final correct 7/7, middle/suffix 0/7. There are no lowercase
held-out cases. These failures qualify the successful leading-space result.

Generic three-token NLL under M improves slightly from E: 14.601578087 to
14.566312432 on training and 9.392891497 to 9.349846333 on test, each over
sixteen controls. Individual shared-piece collateral is still substantial:
`training:piece:2797:start:405489` (ordinary `unt`) loses 3.461664200 nats
of target log probability, from -5.083549500 to -8.545213699. The worst
held-out M shared-piece decline is ordinary `e` at
`test:piece:68:start:81291`: -1.225834846 nats, from -5.700106621 to
-6.925941467. Improving the small generic mean is not word selectivity.

| Marker under `replacement_to_original/cells/` | Bytes | SHA-256 |
| --- | ---: | --- |
| `A/complete.json` | 35,388 | `ce0d3bdfbab74dfa05a3aeab6b93ab7ee519e5f7c9c35a00be5bfa5a45d90e4a` |
| `M/complete.json` | 35,388 | `dd8dd1dacafa30134032b7a57550c48dae256fd730196036993c382123acad92` |
| `AR/complete.json` | 35,513 | `2cf4afc4d053c737b307041a98da207c3f15a5b75be7f16e68abb963e490d212` |
| `MR/complete.json` | 35,513 | `67f2a99475d3a791efdf6754fd395693757cfb015d1abeeab616a3b076ab2397` |

The [paired case-trace protocol](EXEUNT_PAIRED_CASE_TRACE_PROTOCOL.md) fixes
training discovery contexts for direct native computation inspection. A
read-only inventory found no current-endpoint activation traces: the older
four token traces instead belong to `shakespeare/step_13030`. The new adapter
and real packed-case preflight preserve this distinction; no historical
generation event is fabricated for teacher-forced cases. New traces and the
LN2/MLP split remain serialized behind genuine predecessor completion.

## Completion and pre-commit verification

The owner published its complete state at 01:39:51.971558 UTC on September 11,
2026. A subsequent service check found `MainPID=0`, `SubState=exited`, and
`ExecMainStatus=0`. The final `summary.json` SHA-256 is
`3e665688b11f54cee1e66841c2b2409cebcd09667a63c4dc25e2945033ff2ad5`.
This confirms termination and summary publication; it does not replace the
separate numerical and provenance audits described above.

The final pre-commit CPU suite passed all 1,865 tests in 59.951 seconds,
including the case-bound trace adapter, runner, and numerical decomposition
tests. Native launches in runner unit tests are mocked. Neither the new
eighteen-trace capture nor the LN2/MLP split has been launched at this point.

Both new runners authenticate the early-branch predecessor. They do not
automatically authenticate each other's process exit, so an operator must
serialize them and verify the other controller has actually exited. An idle
GPU alone is insufficient: another controller may temporarily be doing CPU
work. The test suite does not remove this operational requirement.
