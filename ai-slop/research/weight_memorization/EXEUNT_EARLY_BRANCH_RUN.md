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
