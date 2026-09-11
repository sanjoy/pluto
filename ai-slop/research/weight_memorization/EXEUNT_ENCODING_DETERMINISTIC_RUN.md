# Deterministic Exeunt/Nuveth experiment: live record

This is partial evidence, not a completed account of how the weights encode
`Exeunt`. The causal protocol is
[EXEUNT_ENCODING_DETERMINISTIC_PROTOCOL.md](EXEUNT_ENCODING_DETERMINISTIC_PROTOCOL.md).

## Current status — 2026-09-10 (outer assay complete; localization prepared)

**The outer factorial completed successfully at 21:45:54 UTC.** All six
suites (main, following native token, shared pieces; both directions) have
successful execution records, exact donor-copy controls, and independently
regenerated saved-logit decompositions. The actual service is exited with
`MainPID=0`, `ExecMainCode=1`, and `ExecMainStatus=0`; the GPU is idle. The
completed summary binds 729 output artifacts and 1,446 frozen input records.
The combined completed-tool regression run passed **88 CPU tests**.
The full analysis regression then passed **1,534 CPU tests in 46.085 seconds**
at 21:52 UTC, including the new localization helper's 39 synthetic tests.
The separately published `lowercase_amendment/outer_factorial_completion_audit_20260910.json`
also verifies all 12 actual native executions and 2,160 distinct records,
including the exact successful service exit and idle GPU. SHA-256:
`c3e8d21c6cc2a55cd083be0cd1c06061f46f7427e49cc5bf3743a429bd396f80`.

Joint E+C transfer restores conditional suffix P to **75.83% for Exeunt**
(complete donor 74.50%) and **88.82% for Nuveth** (donor 88.32%) in 15 held-out
leading-title contexts. Neither E nor C alone gives donor-like completion.
The effect includes target-relative-logit interaction, not solely softmax
curvature. Exact-following-token scores are also close to the donor, but
generic/shared-piece controls show substantial casewise differences. This
is a functional spelling-transfer result, not a selective edit or minimal
storage circuit.

Next: a full Q/early/late three-factor cube with E fixed in both roles,
using additions to E and reversals from EC. Q is positions plus final LN;
early/late are whole blocks 0–3 / 4–7. The helper passed 39 CPU tests and an
actual prepare-only handoff, including completed upstream and idle-GPU checks.
The frozen plan is `lowercase_amendment/complement_localization_20260910_2150/plan.json`
(824,239 bytes; SHA-256
`2e6d1a0b3a67f5f015dc64d5af357746ba74ffe42f66a9e51f72fc04c08762b6`),
binding 2,161 frozen inputs, all eight E-fixed cube cells, and both step-331
directions. **No owning runner is implemented and no localization GPU run
has started.** Next work is that runner, including independently copied E/EC
native loss controls and explicit FP32-loss/FP64-logit comparison gates.
The helper and tests are frozen by the plan and must not be edited silently.
The older branch screen remains explicitly incomplete. The full mechanistic
goal remains active.

### Earlier outer-run progress (historical states)

**Reverse main confirms Exeunt restoration:** completed 21:38:57 UTC, with
independent logit-based revalidation. In the same 15 held-out leading-title
contexts, Exeunt suffix probability is 3.28870e-6 with E alone, .000418043
with C alone, **.758300 with EC**, and **.745008 in the complete original
donor**. Whole-word P is .00154429 versus donor .00159732. Both directions
therefore recover strong absolute conditional completion under joint E+C
transfer. Four of six suites are complete; reverse supplemental controls are
still running. Conditional localization within C is being implemented, not
yet launched or claimed complete.

All three replacement-to-original suites completed by **21:34:45 UTC**;
the reverse direction is now running. The following native token also has
donor-like scores after supplied Nuveth. However, shared `ve` and `th`
controls show substantial EC-versus-donor degradation (test mean first-target
NLL +2.13228 and +4.39982). Thus the remaining vocabulary rows matter outside
the tested word suffix even though they are not needed to transfer that
suffix's strong completion. See the updated interaction report below.

**First outer-factorial result:** the replacement-to-original main suite
completed at 21:28:04 UTC. In 15 held-out leading-title contexts, Nuveth
suffix probability is 5.54311e-6 with selected embedding rows alone, 7.53272e-7
with all nonembedding tensors alone, but **.888170 with both**, versus
**.883236 in the complete donor**. The other embedding rows remain recipient
in the combined model. Whole-word probability is also restored (.00202390
versus donor .00188804), not merely relative preference. The independently
recomputed suffix interaction is +7.74942 nats, including a +14.39662
fixed-rival margin interaction rather than just softmax curvature.
See [EXEUNT_EMBEDDING_COMPUTATION_INTERACTION.md](EXEUNT_EMBEDDING_COMPUTATION_INTERACTION.md)
for controls, exact interventions, evidence, and limitations. The reverse and
supplemental suites are still running; this is not full mechanism localization.

The endpoint **outer E×C factorial is now running** in a new directory,
`lowercase_amendment/outer_factorial_20260910_2123`. It was launched at
21:23:18 UTC as transient system service
`pluto-exeunt-outer-20260910-2123.service`, running as the unprivileged
`ubuntu` user, with controller PID **1917850**. The service has no automatic
restart and does not depend on the earlier terminal session. Its live
service state and main process were checked after launch; a zero default
`ExecMainStatus` while running is **not** a successful terminal exit.

The new runner passed **46 CPU tests** with the runtime/recovery helpers.
Before launch, all six actual completed endpoint readouts regenerated exactly
in a separate CPU-only preflight. The runner repeats those checks and
validates the original training/plan/frozen inputs before any native probe.
The older runner and native children were absent and the GPU was idle.

At step 331 in both directions it measures selected embedding rows **E**,
all 99 non-token-embedding tensors **C**, their combination **EC**, and the
complete donor **D**, against the recipient **A**. Every suite first checks an
independent donor checkpoint copy against every logical-vocabulary logit in
the old reciprocal donor baseline. Main, exact-following-token, and
shared-piece suites are retained. The gap D−EC changes only the remaining
embedding rows in that particular donor-E/C background. It is not a unique
storage fraction. C includes positions and final LayerNorm as well as the
transformer blocks; this assay alone cannot localize within C.

No native result is claimed at launch. No training, existing checkpoint,
old result, frozen implementation, or interrupted-screen state is changed.
The previous goal turn made progress by auditing the interruption and partial
branch results and implementing/testing this focused follow-up; it was not
a verified wait on the missing old process. The mechanism goal remains open.

### Interrupted preceding branch screen

**Interruption observed at 21:09 UTC:** recovery PID 1898158 and its managed
session 29889 are gone; no associated native probe or other GPU process remains.
The old state file still says `interventions`, but that is not evidence of a
live process. Nineteen stages are complete: four copies, four embedding
factorials, and eleven step-100 replacement-to-original branch transfers.
The next branch (block-2 MLP output) has an authenticated successful main
execution, but its apparently complete supplemental files lack an execution
record and the stage has no completion marker. They remain partial evidence;
no exit code is inferred. All previous outputs are preserved unchanged.

The precise termination cause is unknown. Bounded kernel/service checks found
no OOM, GPU fault, or matching termination event; the parent and its service
scope remained alive. The focused follow-up above uses a separate supervised
service. It does not restart training. The unfinished
branch screen remains explicitly incomplete, not silently counted as passed.

All eleven completed branch reports were independently regenerated exactly
from the six native score/execution sets each binds. New audit:
`lowercase_amendment/partial_step100_branch_screen_20260910_2115.json`,
587,541 bytes, SHA-256
`a15f548c7aea983d27415d6c6b9ee7f662e56df9f801bd9841a5b7b19dc3e4f1`.
None flips any of the 39 original-prefix word preferences. Heldout title
mean word-log-probability changes range from -0.0830 to +0.0489 nats for
Exeunt and -0.0152 to +0.0468 for Nuveth. These are small relative to the
early embedding-transfer effects, but are neither tests of branch deletion
nor evidence that these computations are unnecessary. Only the first eleven
branches in the fixed order were measured: blocks 0 and 1 fully, block 2
attention whole/output and MLP whole. There are no final-checkpoint branch
results yet and no unbiased full-model ranking.

### Earlier causal progress (historical process states)

**Causal update, 20:43 UTC:** all four native copy controls and all four
embedding factorial interventions (both directions at steps 100 and 331,
main/following-token/shared-piece suites) completed by 20:42:27. The unchanged
128-branch-transfer screen is now running, beginning with step-100 block-0
attention. The recovery controller/session below remain the active handles.

At step 100, output-only and joint transfers of the selected 11 embedding
rows reverse whole-word preference in all 39 original-prefix contexts, in
both directions; input-only transfers reverse none. At step 331, this is no
longer a sufficient account of learned completion. Replacement-to-original
joint transfer reverses only the three bare-title cases, no leading-title or
lowercase cases. Original-to-replacement joint transfer reverses all 32
title cases, but the resulting heldout Exeunt geometric probability is only
5.13779e-9 versus 0.00173831 in the original donor. Relative preference alone
therefore greatly overstates restoration. Leading-title first-token
preference transfers in both directions, while strong suffix completion
does not. Lowercase is separate: in the final reverse direction, input-only
transfer flips all seven lowercase preferences but joint transfer flips none.

Shared-piece controls rule out a word-selective interpretation of the row
patch: e.g. at step 100 replacement-to-original output transfer raises NLL
for unrelated ` Ex` by about 3.33 (train) / 3.26 (test) and unrelated `unt`
by 3.96 (train), despite very small mean effects on the generic control
panel. Exact-following-token effects are also retained, not treated as
delimiter-independent full-string probabilities.

The four main factorial readouts were separately revalidated with the
FP64 fixed-rival margin/normalizer postprocessor. Artifacts are
`lowercase_amendment/step{100,331}_{replacement_to_original,original_to_replacement}_main_embedding_mechanism.json`.
For example, step-100 replacement-to-original heldout Nuveth log-probability
interaction is +6.15879 nats, composed of +6.16117 fixed-rival-margin
interaction minus +0.00238226 normalizer interaction: it is not merely
softmax curvature. This is functional interaction, not a unique storage
location or proof that the unchanged transformer is dispensable.

**Next causal isolation, designed but not run:** distinguish selected
embedding rows E, the other embedding rows R, and all 99 non-token-embedding
tensors C (96 transformer tensors, learned positions, final LayerNorm).
Construct C-only and E+C hybrids at step 331 in both directions. An unchanged
native embedding-factorial run on C→E+C, together with the existing A→E
and donor logits, supplies an outer E×C factorial. The gap from E+C to the
full donor then isolates the remaining rows R in that particular donor-E/C
background. Use the same frozen cases and saved-logit FP64 scoring, retain
all collateral controls, and authenticate an all-tensor donor-copy control.
Conditional branch additions/removals/restoration can then test the missing
computation; null single-branch transfers cannot exclude interactions. This
follow-up does not modify or interrupt the currently running screen.

Both four-hour training runs are complete at **331 optimizer steps**. Original
finished at 15:44:51 UTC (14,421.9 s, train/test loss 4.60726/4.57047); the
lowercase-amended replacement finished at 19:48:38 UTC (14,422 s,
4.60189/4.56530). Both histories retain 0/100/200/300/331. The separate full
completion audit is `lowercase_amendment/paired_training_completion_audit.json`.
The old exact-case-only replacement was never launched. The lowercase user
amendment remains authoritative.

All nine native checkpoint scores completed by 19:55:45 UTC; the completed
trajectory summary was published at 19:55:49. See
[EXEUNT_PAIRED_BEHAVIOR_TRAJECTORY.md](EXEUNT_PAIRED_BEHAVIOR_TRAJECTORY.md)
for independently recomputed results and evidence hashes. In all 39 sampled
original-prefix contexts the complete-word preference follows the trained
spelling by step 100. Leading-title suffix completion develops much later,
particularly from steps 200 to 300. No complete three-token word argmax match
was observed. Bare and lowercase forms expose distinct selection/completion
failures. Relative preference is not contextual memorization or causal weight
localization.

The original causal controller failed **before any intervention** at 19:56:50:
its relocated GPU test could not find the Bazel shared libraries. Both
historical downstream observers then exited on their upstream-failure guards.
All old handles were checked absent; none of their artifacts were overwritten.
This was a runtime packaging failure, not a failed GPU numerical test.

An explicit new recovery is running in
`lowercase_amendment/causal_stage_recovery_20260910_2020`:
PID **1898158**, start ticks **129426193**, managed session **29889**.
The regenerated intervention plan equals the original plan exactly. All three
native executable bytes are unchanged. The new runtime package copies 53
Bazel DSOs under their loader names, pins nine other load-time libraries,
and verifies relocated loader paths/hashes. Only an explicit library-search
override is recorded, never the complete environment. The previous request
did not freeze DSO hashes, so this is current matching-build runtime provenance,
not retrospective attestation of old shared-library bytes. Runtime `dlopen`
dependencies are not fully enumerated by this loader check.

**47 CPU tests passed** across the recovery, runtime bundle and existing
controller suites. The actual native
`EmbeddingFactorialGpuTest.NativeDiagonalsAndCausalInputExposure` ran and
passed at 20:20:43 UTC (2.96 s, no skips). The recovered controller then began
the unchanged **136-intervention** screen, copy controls first, followed by
bidirectional embedding input/output factorials and attention/MLP transfers
at matched steps 100 and 331. No intervention result is claimed by this
launch record. Historical follow-ups remain stopped and need explicit new
handoffs after the recovered causal stage; they are not silently requeued.

The mechanism goal remains active and incomplete. Earlier status notes below
are an audit history, not current process state.

## Earlier execution history

**Latest user amendment (2026-09-10):** replace lowercase `exeunt` with
`nuveth` as well as `Exeunt` with `Nuveth`, before the replacement arm starts.
See [EXEUNT_LOWERCASE_AMENDMENT.md](EXEUNT_LOWERCASE_AMENDMENT.md).
Original training completed unchanged. The old replacement launch is guarded
and its old-manifest analysis queue has been retired; earlier corpus counts and
queue descriptions below are historical, not the amended run's current plan.
Native validation passed: all seven lowercase occurrences are replaced, with
only 21 additional token slots changed. The amended handoff runner entered
`waiting_original` at 15:25:10 UTC (PID 1816208, start ticks 127653562,
managed session 91141; `lowercase_amendment/request.json` and `state.json`).
It preserves the original run and queues a fresh two-step control followed by
the revised four-hour replacement arm. No revised GPU child had started at
verification. The original reached step 300 / training-evaluation loss 4.66668.
The amended trajectory observer was subsequently queued at 15:41:45 UTC
(PID 1820183, start ticks 127753878, managed session 69927), after 1,155 CPU
tests passed. Its request and future outputs are under
`lowercase_amendment/analysis_trajectory`; the old causal observers remain
retired pending case/provenance-aware replacements. No new GPU scores yet.

**Latest execution (16:02 UTC):** original training completed its full budget at
step **331** (14,421.9 seconds; training-evaluation loss **4.60726**, test loss
**4.57047**). Steps 0/100/200/300/331 independently verified. The guarded handoff
succeeded. The amended replacement's full four-hour clock began **15:48:09 UTC**
(PID 1821367, start ticks 127790846, parent 1816208); it is the sole GPU process.
The trajectory observer remains waiting. New supplemental controls (265 cases),
fixed-row causal exports, and case-aware embedding/branch readouts are prepared
and tested (1,176 CPU tests). Actual causal followups still need amended
planning/orchestration and are not yet queued. See the amendment record above
for evidence hashes and the control's exact repeated weights.

**Latest queue (16:30 UTC):** the amended matched-step causal controller is now
waiting behind the trajectory observer: PID **1829540**, start ticks
**128043774**, managed session **97587**. Request/results directory:
`lowercase_amendment/causal_stage`; log: `lowercase_amendment/causal_observer.log`.
All **1,211 CPU tests passed**. The 111 frozen input records, exact upstream
process identity, and matching native loss-probe bytes were rechecked after
launch; the trainer still had exclusive GPU use. This supersedes the earlier
“not yet queued” status for embedding and attention/MLP weight transfers only.
No new causal GPU results yet; source-value and finer ablations remain separate
work. See the amendment record for the request hash and full validation details.

**Latest queue (16:40 UTC):** the fixed historical `unt` source-value assay has
also been requeued behind the new combined stage, using an amended handoff
adapter. PID **1833273**, start ticks **128105137**, managed session **92971**;
outputs in `lowercase_amendment/historical_source_value_stage`, currently
`waiting_causal`. Its 215 frozen records and the original historical native
inputs/checkpoint were verified. Final CPU suite: **1,240 tests passed**.
This does not change the timed replacement training or claim any new measured
GPU effect; historical assay results will remain distinct from paired-model
results. Finer head/neuron or initial-piece ablations are still separate work.

**First matched checkpoint (17:11 UTC):** amended replacement step 100 completed
at 17:00:39 UTC, with training-evaluation loss **5.32869** versus original
**5.32012**. An independent audit validated the live trainer, 751 frozen inputs,
shared initialization, and all checkpoint files. The top eight changed
embedding rows are edited-word pieces, but all eleven selected rows together
account for only 11.3003% of whole-model squared difference. This is a numerical
signature, **not causal attribution**; no new native word probabilities yet.
See [EXEUNT_EARLY_STEP100.md](EXEUNT_EARLY_STEP100.md) for precision-aware
differences, exposure counts, hashes, and limitations. Training remains live.

**Latest queue (17:40 UTC):** the calibrated historical first-piece
head-position observer is now waiting behind the historical suffix observer:
PID **1859615**, start ticks **128467285**, managed session **53034**. Its
406 frozen records and actual identity were checked; timed training remains
the sole GPU job. The complete CPU suite passed **1,361 tests**. It compares
query-only, other-query and all-query head-context edits for the fixed three
historical first-piece events, including clean/zero calibration and native
identity controls. It has not executed a GPU test or measured a new effect.
See [EXEUNT_HEAD_POSITION_PROTOCOL.md](EXEUNT_HEAD_POSITION_PROTOCOL.md) for
the exact request, inputs, resource order, and interpretation limits.

**Embedding-coordinate check (17:56 UTC):** about 38% of the step-100
embedding difference is a common row shift. It cancels from relative output
scores at fixed hidden state, but is not generally irrelevant to the tied
input path. The eight leading rows remain spelling pieces after centering.
The authenticated report and 19 new tests are documented in
[EXEUNT_EARLY_STEP100.md](EXEUNT_EARLY_STEP100.md); 1,380 CPU tests passed.

**Interpretation safeguard (18:08 UTC):** a separate, unfrozen CPU
postprocessor now separates fixed-rival logit-margin effects from softmax
normalization effects in the saved AA/AJ/JA/JJ factorial logits. A synthetic
counterexample verifies that nonzero probability interaction need not imply
any logit interaction. Full-vocabulary interactions are also checked, since
the normalizer can contain genuine effects on other competitors. All 1,407
CPU tests passed. This adds no GPU work and changes no queued program; there
are still no real-model factorial results to postprocess. Details and limits:
[EXEUNT_FACTORIAL_LOGIT_PROTOCOL.md](EXEUNT_FACTORIAL_LOGIT_PROTOCOL.md).

**Second matched checkpoint (18:15 UTC):** replacement step 200 completed at
18:13:17 UTC, training-evaluation loss **4.96577** versus original **4.97011**.
Both checkpoint inventories and actual training provenance independently
verified; all three numerical diagnostics match those hashes. Whole-model
delta L2 grew from **2.190896** at step 100 to **6.940875** at step 200. The
eleven spelling rows' embedding-energy share fell from **13.0530%** to
**3.1017%**, and their share after centering fell from **21.2835%** to
**5.1298%**. The top seven rows remain spelling pieces; block 6 now has the
largest non-embedding tensor difference. These are weight statistics, not
causal word attributions or native probability measurements. Evidence,
sampling exposure and limits are in [EXEUNT_EARLY_STEP200.md](EXEUNT_EARLY_STEP200.md).
Replacement timed training and all queued observers remain active.

**Historical causal refinement (18:31 UTC):** a CPU reanalysis of the retained
step-13030 ablations separates high-confidence suffix changes from whole-word
effects. Removing block-0 attention raises Exeunt's joint probability from
**25.3876%** to **33.4119%** while lowering conditional suffix probability
from **99.9956%** to **94.7273%**. All 64 single-head removals preserve suffix
probability above 99.7%, despite measurable confidence effects. The complete
64-head/eight-branch, three-word readout and all source hashes were checked;
21 historical-reader tests passed. This is **not a new paired-model result**,
and the whole-branch/head comparison retains an explicit output-bias confound.
See [EXEUNT_SUFFIX_CONFIDENCE_HISTORICAL.md](EXEUNT_SUFFIX_CONFIDENCE_HISTORICAL.md).
The older selection/spelling narrative now explicitly marks its
lowercase-survival discussion as the superseded exact-case-only input plan.

**Readout coverage audit (19:03 UTC):** direct inspection of authenticated
packed inputs confirms 16 title-case contexts per split, despite 64 candidate /
domain entries. Only two training and one test occurrences use the bare
title-case tokenization; lowercase has seven training occurrences and no test
occurrence. A separate source audit identifies the opposite trajectory/causal
odds conventions, incompatible domain-aggregation weights, and near-ceiling
FP32-loss quantization. The actual fixed screen uses full donor substitutions,
not intermediate-dose or same-model restoration tests. The reporting guide
records these limits and the follow-up decision criteria without changing a
frozen job: [EXEUNT_CAUSAL_READOUT_GUIDE.md](EXEUNT_CAUSAL_READOUT_GUIDE.md).
No new native causal result is claimed; replacement training remains active.

**Third matched checkpoint (19:28 UTC audit):** replacement step 300 was
saved at 19:25:55 UTC, with training-evaluation loss **4.67385** versus
original **4.66668**. An independent audit verified four checkpoints,
the live trainer, and all 751 frozen inputs. Independently recomputed raw
differences, operand precision, centered row ranks and exposure counts agree.
The eleven spelling rows' absolute squared difference increases, but their
whole-model share falls to **1.7123%** and centered embedding share to
**3.4100%**. Block 6 remains the largest non-embedding difference. These are
weight statistics, not causal word attributions. Full evidence and limits:
[EXEUNT_EARLY_STEP300.md](EXEUNT_EARLY_STEP300.md). Training continues to
the full four-hour limit; no new native word probabilities are available yet.

## Experiment identity and launch

New preserved root:
`/home/ubuntu/checkpoints/exeunt_nuveth_deterministic_20260910_1137`.
The older `/home/ubuntu/checkpoints/exeunt_nuveth_20260910` run is untouched.

The optimized production trainer is based on commit
`08edf21940f06344b2b649d631b8716b6fb8e0a2`; its frozen executable SHA-256 is
`b292f9228b5a664d997079e16bf1bb39641909e36cd312d14a71ca9df71b25ef`.
The experiment manifest separately records copies and hashes of the uncommitted
launcher/helper sources, so the commit alone is not presented as complete
launcher provenance. It also records corpus/tokenizer hashes, arguments, native
token exports, and the first 10,000 seeded sequence starts.

Runtime: GH200, GPU UUID `GPU-b023911a-7ba6-edf7-2152-4b02873e8e85`, driver
580.126.20, CUDA 13.3.73, GCC/libstdc++ 13.3. The same GPU is used sequentially.

The first detached shell launcher exited before creating a training child or
changing the prepared state. Its missing process, empty log, unchanged prepared
state, absent arm directories, and available lock were checked before using a
managed execution session. No training arm was restarted or overwritten.

Supervisor PID 1723614 started at 2026-09-10 11:37:15 UTC; managed session 19578.
Its actual command, parent/child relationship, and process start time were
verified. These are historical identifiers, not perpetual proof of liveness:
recheck `/proc` and `state.json` on each continuation. The log is
`runner_session.log`; the empty `runner.log` belongs to the unstarted attempt.

## Input controls

| Split | Native tokens, both corpora | Exeunt/Nuveth occurrences | Changed token slots |
| --- | ---: | ---: | ---: |
| Training | 1,650,781 | 936 | 2,808 |
| Test | 184,382 | 92 | 276 |
| Full | 1,835,163 | 1,028 | 3,084 |

Native exports prove identical IDs outside those slots, identical outer byte
boundaries, and an unchanged train/test split. Each substitution occupies three
tokens. The 160 cases in `word_cases/cases.json` were selected before inspecting
learned weights: 16 occurrences per split crossed with two candidate words and
two prefix domains, plus 16 unrelated three-token controls per split.

## Repeatability gate and training status

At 11:41:59 UTC the gate verified exact SHA-256 equality for all 100 weights at
each of steps 0, 1, and 2 in `control_a` and `control_b`: 300 matched file pairs.
An independent rehash at 11:44:11 UTC confirmed the published gate against the
actual files. Evidence: `determinism_gate.json` and the control inventories.

The replacement control completed at 11:44:05 UTC. The original-text long arm
then launched as PID 1727028 and began its four-hour training clock at
**11:44:21 UTC**. The replacement-text four-hour arm is queued after it. Both
preserve step 0, every 100th step, and their final checkpoint. Endpoint step
counts may differ; do not substitute wall-clock endpoints for matched steps.

The original arm wrote **step 100 at 12:56:52 UTC** and reported training loss
**5.32012 at 12:56:59 UTC**, down from 10.7495 at initialization. These are the
fixed first four sequential training-evaluation batches (40,960 targets), not
the entire training corpus. A CPU-only check at about 13:02 UTC independently
validated the 100 expected weight files, their sizes (205,934,592 bytes total),
finite values and stable hashes, and reverified the live trainer's identity.
Evidence: `original_step_100_validation.json`, including the exact log snapshot.
The replacement arm had not started, so this is not a matched-step result.

The queued postprocessor is PID 1726864, managed session 38860, log
`analysis_observer.log`. Its request is `analysis_trajectory/request.json`.
It verifies live training handles while waiting and performs no GPU work until
both full training budgets finish and terminal logs/checkpoints are validated.
It will compare every common checkpoint, shared initialization, and endpoints;
it independently rechecks deterministic controls before scoring. This is
trajectory evidence, not completion of the causal-mechanism work.

## Early deterministic training evidence

The frozen native sampler and production dataset implementation agree on the
first two batches. Step 1 contains no replacement spans. Step 2 contains six,
all fully included in both input and shifted-target windows:

| Sequence within step 2 (zero-based) | Window start | Occurrence index | Word offset within window |
| ---: | ---: | ---: | ---: |
| 0 | 852,046 | 495 | 122 |
| 3 | 251,414 | 148 | 532 |
| 4 | 591,374 | 340 | 293 |
| 5 | 1,219,347 | 708 | 507 |
| 7 | 703,018 | 413 | 1,001 |
| 9 | 687,766 | 406 | 963 |

Actual per-file hashes for `control_a` versus `control_replacement` show:

- Step 0: all 100 weights identical.
- Step 1: all 100 weights still identical, as predicted by the unchanged batch.
- Step 2: all 100 weight tensors have at least one differing byte after the
  first batch containing replacements.

The step-2 whole-model difference has L2 norm **0.02561203393**, or
**0.01608779%** of the original parameter norm. Evidence is
`control_step_2_weight_diff.json`, which includes exact input checkpoint hashes
and FP64 bounded-memory arithmetic. This is a deterministic training response
to the altered examples, not proof that every changed tensor is necessary to
produce the word. The six input changes also affect predictions following the
word through causal attention; this is not a loss-only label intervention.

## Which corpus occurrences have actually been presented by step 100?

The frozen native sampler invocation was rerun on CPU and reproduced all 10,000
saved starts byte-for-byte. The first 1,000 starts correspond to 100 updates of
ten sequences. Production source at the recorded commit resets the training
iterator once before training; evaluations use a separate sequential iterator
and do not consume or reset its random stream. This is verified source-and-
sampler reconstruction, **not** a separately captured runtime batch trace or an
estimate from elapsed time.

For a window starting at `s`, input indices are `[s,s+1024)` and supervised
targets are `[s+1,s+1025)`. Counting those target intervals gives:

| Through step | Sequence windows | Complete three-token presentations | Distinct fully presented / partial-only / never targeted occurrences |
| --- | ---: | ---: | ---: |
| 1 | 10 | 0 | 0 / 0 / 936 |
| 2 | 20 | 6 | 6 / 0 / 930 |
| 100 | 1,000 | 595 | 443 / 3 / 490 |

At step 100 the three pieces have 596, 597, and 597 target presentations,
respectively. Partial words at window edges explain the difference from
595 complete triples. Complete-word presentation counts per occurrence are:
493 have zero, 315 have one, 106 have two, 20 have three, and two have four.
The 493 with no complete presentation include the three partial-only cases;
do not describe all 493 as completely untouched.

Among the 16 selected **training** word-case occurrences, nine have at least one
complete target presentation and seven have no word-target exposure. Each of
the nine also has its entire 128-token prefix segment plus three targets inside
one sampled sequence. However, **none** of the sampled sequences starts exactly
at one of these evaluation prefixes. Containing that segment therefore does
not establish identical full preceding context or local learned position IDs.
The counts concern corpus coordinates in each arm's own text, not a claim
that both counterfactual prefix/target combinations were trained.

This adds an exposure distinction to the later behavioral readout: the same
"training split" contains selected occurrences that have and have not yet
contributed word targets by this checkpoint. Neither exposure nor its absence
alone proves memorization, causal storage, or spelling generalization.
The original arm's step-100 completion is independently established above;
replacement-arm counts are conditional on its later reaching the same step.

Evidence: `training_exposure_through100.json` at the experiment root. It retains
per-occurrence and per-selected-case counts, input/source hashes, and the exact
successful native replay command. Independent scalar/interval reconstruction
agreed with all counts. Six new CPU tests cover shifted targets, clipped words,
zero draws, repeated windows, bounded chunking, segment versus exact-context
coverage, and invalid positions in
`scripts/weight_analysis/paired_training_exposure_test.py`.
The full Python suite passed **939 CPU tests** in 24.241 seconds;
log: `/tmp/pluto-exeunt-training-exposure-cpu-tests.log`. No GPU forward was added.

## A clean input-versus-output control

Native word-piece rows are:

| Token ID | Exact decoded piece |
| ---: | --- |
| 45 | `N` |
| 68 | `e` |
| 303 | `ve` |
| 400 | `th` |
| 1,475 | ` Ex` |
| 2,797 | `unt` |
| 3,109 | `Ex` |
| 21,733 | ` Nu` |
| 45,177 | `uve` |

The 128 word cases contain only **32 distinct initial prefixes**, because
candidate/domain copies often have identical input IDs. All 16 training
prefixes and 13 of 16 test prefixes contain none of these nine IDs. The other
three test prefixes contain `N` or `th`.

Consequently, on those **29 untouched prefixes**, changing only these embedding
rows cannot change the causal input computation before the first word token.
Any first-token score change from a joint tied-embedding/head patch must be
output-side. After supplying the first or second candidate piece, every suffix
prediction is exposed to patched input rows, so that guarantee no longer holds.

This is a dependency invariant, not a measured probability change. The native
2x2 factorial probe will verify the expected unchanged residual/readout cells
and measure input, output, and interaction effects with the complete vocabulary
denominator. Evidence: `input_exposure.json`; implementation and seven CPU tests
are `paired_input_exposure.py` and `paired_input_exposure_test.py`.

## Which early checkpoint differences reach the forward arithmetic?

A CPU-only scan of the completed deterministic step-1 and step-2 controls
distinguishes FP32 master weights from their forward operands. In this recipe,
token-embedding entries and dense matrices are cast to BF16 before lookup/MMA;
position embeddings, dense biases and LayerNorm parameters enter arithmetic as
FP32. The scan does **not** incorrectly pre-round those FP32 parameters.

At step 1 the two controls remain identical under both comparisons. At step 2:

| Scope (logical vocabulary; tied head counted once) | Changed FP32 coordinates | Changed operand coordinates |
| --- | ---: | ---: |
| BF16-consumed embeddings and dense matrices | 48,975,188 | 1,485,840 |
| FP32-consumed positions, biases and normalization | 575,403 | 575,403 |
| Nine selected word-piece rows (subset, not an extra tensor) | 4,608 | 3,548 |

Thus **96.97%** of the changed BF16-consumed master coordinates round to the
same operand in both arms; **77.00%** of changed coordinates in the selected
word-piece rows survive the cast. Every one of the 100 tensors still has at
least one changed forward operand. Padding is unchanged; neither arm contains
subnormal values in this scan.

This is a numerical distinction, **not** a measured behavioral effect. An
unchanged cast operand cannot itself alter a fixed forward at that use, but its
FP32 master value can matter during future training. A changed operand need not
change any final prediction. Small master differences can cross a BF16 rounding
boundary, so operand-delta norms can even exceed master-delta norms.

Evidence: `control_step_1_compute_weight_diff.json` and
`control_step_2_compute_weight_diff.json`, whose scanned source hashes were
checked against the completed control checkpoint inventories. Source hashes and
per-tensor counts are retained. The read-only implementation is
`scripts/weight_analysis/paired_compute_weight_diff.py`; its ten new CPU tests
and the fourteen existing rounding/reference tests pass. This is IEEE BF16
operand analysis, not a claim of reproducing all native GPU arithmetic.

## Queued causal stage (initial launch 12:11 UTC; corrected 12:28 UTC)

The first version launched as PID **1746344**, kernel start ticks
**126490529**, managed session **66014**. Its request and frozen sources/binaries
are in `causal_embedding_stage/`; its log is `causal_embedding_observer.log`.
At 12:11:32 UTC its actual command/start identity and the upstream scorer's
identity were checked. The original trainer remained the only CUDA process.

At 12:27 UTC source review found that its copied checkpoint basename
`checkpoint` was incompatible with the frozen loss probe's production inspector,
which requires `step_N`. The verified waiting observer alone was stopped with
SIGTERM, its managed session returned 130, and its preserved failure marker
records `completed: []`. No plan, GPU test, or causal measurement had started.
Training supervisor 1723614, trainer 1727028 and trajectory observer 1726864 were
rechecked live and were **not stopped or restarted**.

The corrected observer uses recipient-matched `step_N` basenames and a new
preserved directory **`causal_embedding_stage_v2/`**. It is PID **1752078**,
kernel start ticks **126592377**, managed session **38401**, with log
`causal_embedding_observer_v2.log`. Its identity and all frozen inputs were
verified after launch at 12:28 UTC. The fourteen CPU tests pass, now including
the checkpoint-name contract. A real frozen-binary preflight confirmed that
`checkpoint` is rejected while `step_100` passes name inspection; a deliberately
missing batch stopped both probes before executor creation, so this check did
not use CUDA. The old output directory remains an explicitly superseded attempt.

This job waits for the specific trajectory observer (PID 1726864, start ticks
126325428), independently cross-checks and repeats exit observations, and then
revalidates completed training, the deterministic controls, checkpoint hashes,
and trajectory evidence. It does not infer an unobservable upstream exit code.
There is no training restart, concurrent GPU work, or overwrite path.

Once those gates pass, it will:

1. Prepare an immutable intervention plan for the earliest and latest positive
   matched steps, deduplicating if only one exists. Unequal timed endpoints are
   not substituted for matched optimizer steps.
2. Run the native factorial GPU tests, rejecting skipped/incomplete tests.
3. Verify that independent unmodified checkpoint copies reproduce every native
   loss/argmax byte from the trajectory scorer.
4. Transfer the nine preselected word-piece rows in both directions and run the
   AA/AJ/JA/JJ input/output factorial on main three-token and supplemental
   four-token word cases. Recompute NLL/rank/argmax from all 50,257 saved logits.
5. Score the full supplemental suite for both unmodified copies and row patches,
   providing measured baselines for shared-piece collateral changes.

Execution records bind the actual command and child PID to before/after input
hashes and output hashes. Known-unexposed causal prefixes must have identical
input-side logits; output-only changes must leave every unselected raw-logit
column identical (their probabilities can still change via the denominator).
Native diagonal verification covers the full padded vocabulary at each selected
row, **not** every unselected context row. All source files, selected-row/batch
exports, and completed artifacts are rechecked before success is published.

The planner also describes all eight attention and all eight MLP whole-branch
transfers, plus their separate output-weight-and-bias transfers, in both
directions at both selected steps. These branch interventions are **not run by
the embedding stage**; the separate queued stage below will execute them. A completed stage
explicitly does not claim completion of the research goal.

The final CPU suite passed **855 tests** after these additions. The log is
`/tmp/pluto-exeunt-causal-followup-cpu-tests-final.log`. The updated native binary
and multi-sequence GPU test build; GPU execution remains deferred. The newly
queued source files must not be edited without coordinating a stopped/new
analysis run: their original and frozen-copy hashes are part of the guard.

## Queued attention/MLP stage (12:34 UTC)

The branch observer is live as PID **1755949**, kernel start ticks
**126628345**, managed session **39706**. Its new output directory is
`causal_branch_stage/`, and its log is `causal_branch_observer.log`. Its exact
identity, all 33 frozen input/source records, and the upstream identities were
verified after launch. Trainer 1727028 remained the only CUDA process.

This stage follows the **corrected** embedding observer, PID 1752078/start ticks
126592377, not the superseded attempt. It requires confirmed process exit and
independently validated embedding artifacts, training completion, checkpoint
inventories, deterministic controls and native GPU-test results before scoring.
It then executes the entire predeclared branch screen sequentially: 64 patches
per selected matched step, or 128 if earliest/latest are distinct. Each branch
is tested in both donor directions, as a six-tensor whole-branch transfer and a
two-tensor output-weight-and-bias transfer. Independent `step_N` copies preserve
the original checkpoints.

Every patch scores the main 160 cases and supplemental 211 cases. Recipient and
donor baselines come from the measured embedding-stage copy controls, with full
execution provenance and byte equality to the trajectory's main scores checked
again. The CPU readout verifies actual source/patch weights, all six native
execution records, output layout and target order, and identical-causal-prefix
score invariants. Its summaries distinguish absolute likelihood from preference,
first-token from suffix effects, bare/leading-space variants, each shared-piece
control, and training discovery from held-out test results. Actual event aliases
are deduplicated. Effects on conditional suffixes are not mislabeled as same-
context logit margins.

The complete Python analysis suite passed **903 CPU tests** in 24.164 seconds;
log: `/tmp/pluto-exeunt-branch-stage-cpu-tests.log`. These include 22 readout tests,
16 branch-observer tests, and 10 operand-difference tests. A real serialized-plan
compatibility test catches JSON list versus in-memory tuple differences, and
observer tests reject source changes, skipped GPU tests, stale/PID-reused handles,
missing provenance, and changed copy-control scores. All CUDA tests and causal
measurements remain deferred; mocked unit invocations are not experimental data.

Remaining work after this screen includes zero/half/full ablation-dose controls,
native residual/attention/MLP traces, finer head/neuron or routing interventions
as justified by reproducible effects, and integration into the causal account.
Branch-transfer rankings alone will not satisfy the research objective.

## Selective neuron-dose preparation (13:15 UTC; no new GPU measurements)

The existing native loss scorer can score independent row-dose checkpoint
copies, so no new kernel or change to frozen analysis sources was needed.
The new CPU helper `scripts/weight_analysis/neuron_weight_patch.py` changes only
selected MLP output rows at dose 0, 0.5, or 1; output bias and all other bytes
stay unchanged. Its distinct-format `patch.json` also preserves compatibility
with native trace inspection without masquerading as a donor transfer.

The 21 focused tests pass; the full Python suite now passes **933 CPU tests**.
Real historical-checkpoint copies at all three doses were independently checked
against all 100 source weight files. Native scorer layout preflights deliberately
rejected existing score-output paths before CUDA initialization. Evidence:
`neuron_patch_preflight/validation.json`; detailed semantics and limitations:
`scripts/weight_analysis/neuron_weight_patch.md`.

This is preparation for smaller causal interventions, not a measured model
effect or a new paired-model feature selection. At 13:15:50 UTC supervisor
1723614, original trainer 1727028, and all three specific analysis observers
were reverified live; both causal observers' source hashes were unchanged.
Trainer 1727028 remained the only GPU workload. GPU tests and scoring remain
behind the existing training/trajectory/embedding/branch sequence.

## Prepared source-specific attention assay (14:08 UTC)

The next historical test isolates the value arriving from the earlier ` Ex`
piece, rather than suppressing the whole block-1/head-2 output at every query.
The new native `source_value_probe` scales one source/head V vector, reruns
production attention, splices **only one query/head's context**, and replays
the native downstream layers. It does not change original model weights.

Creation requires byte-exact agreement with the entire captured native
attention output and every physical vocabulary logit at every row. Every dose
1 is genuinely replayed, including a final identity after the interventions.
Q/K, other V/context bytes, original inputs, and all 100 model weights are
checked. A future-source control must have no effect on the selected causal
query or any downstream logit after the single-query splice.

The predeclared first assay is historical prefix step 1342, block 1/head 2,
query 1023 (`e`), target 2797 (`unt`). Sources are 1022 (` Ex`), 889 (` who`,
largest alternative reconstructed attention), 1023 (self), 1021 (space), and
512 (fixed distant midpoint). No new source-intervention outcomes were used
to select them. Exact provenance, command, native replay mechanics, and
limitations are in `scripts/weight_analysis/source_value_probe.md`.

The optimized probe and GPU test build. All six new CPU tests pass (including
exhaustive finite BF16 dose rounding), as do the two existing native CPU
validation targets. Nine pre-CUDA CLI rejection checks pass. The full Python
suite passes **939 tests in 24.100 seconds**; log:
`/tmp/pluto-exeunt-source-value-cpu-suite.log`. Independent source review found
no concrete blocker. **GPU validation and actual interventions remain unrun**;
this assay is prepared, not silently inserted ahead of the existing queue.

At 14:08 UTC all five process identities (supervisor, original trainer,
trajectory scorer, embedding observer, branch observer) were verified live.
The two causal observers' frozen inputs were unchanged. Original trainer
1727028 was still the sole CUDA compute process; latest completed checkpoint
was step 100. No new source-specific causal effect is claimed here.

At **14:09:29 UTC**, the original arm completed checkpoint **step 200**.
Its fixed training-evaluation loss logged at 14:09:37 was **4.97011**, versus
5.32012 at step 100 and 10.7495 initially. This evaluation uses the same first
four sequential batches (40,960 targets), not the entire training corpus.
CPU validation at 14:11:11 checked all 100 canonical non-symlink weight files,
correct total size **205,934,592 bytes**, finite FP32 values, stable file stats,
and matching repeated hashes. The source, live trainer identity, exact log
snapshot, and all weight hashes are retained in
`original_step_200_validation.json` at the experiment root. The replacement
arm has not started; this is not a matched-arm or word-specific result.

## Source-specific assay queued (14:30 UTC)

The historical source-value assay is now a managed waiting job, not just a
proposed command. New observer **PID 1797771**, kernel start ticks
**127323212**, parent PID 1797766, session **43836** follows the exact existing
branch observer PID 1755949/start 126628345. Its output is
`causal_source_value_stage/`, and its log is
`causal_source_value_observer.log` at the experiment root.

Its request freezes **187 input/source/binary records**, including all 100
historical checkpoint weights and archived native prefix/logit/QKV/context
evidence. Sources are the predeclared 1022, 889, 1023, 1021, and 512, each at
1, 0.5, 0, and 1 again. Before any GPU child starts it must confirm upstream
process exit, complete paired training/determinism/branch artifacts, unchanged
snapshots, and an idle GPU with the same UUID. The native GPU test must then
finish with no skipped tests. Only then may it execute the 20-arm assay and
produce the independent readout.

The new observer and readout pass **32 and 24 CPU tests**, respectively; the
complete analysis suite passes **995 tests in 32.185 seconds**. Logs:
`/tmp/pluto-exeunt-source-value-observer-tests.log`,
`/tmp/pluto-exeunt-source-value-readout-tests.log`, and
`/tmp/pluto-exeunt-source-value-queue-cpu-tests.log`. The native binary and GPU
test build. Final native input certification now follows the last full clean
replay, so all model computation is inside the integrity-check envelope.

At 14:30:45 UTC all three causal observers' frozen records were reverified,
the new observer was independently confirmed live, and its directory contained
only `bin/`, `source/`, and `request.json`: no GPU test or native assay had
started. Original trainer 1727028 remained the sole CUDA compute process.
No training process was restarted or altered. Source/readout implementation
files and the probe documentation are now frozen until this observer exits;
do not edit them under a healthy waiting job.

This stage targets one historical `unt` prediction. Its result must still be
combined with complete-word, paired-model, and collateral-control evidence;
successful completion will not itself complete the research goal.

## Selection versus spelling: archived causal screen and new corpus controls (14:52 UTC)

The new readout [EXEUNT_SELECTION_AND_SPELLING.md](EXEUNT_SELECTION_AND_SPELLING.md)
joins **560 archived native piece ablations** into **240 same-intervention word
results**: 80 interventions each for Exeunt, grandam, and corse. It authenticates
693 input records, the exact producer source recovered from Git, and two current
analysis-source records. All raw logits were recomputed at temperature 1 over
the complete logical vocabulary. The original generation rows, native replay
rows, and sliding teacher-forced context chains agree byte-for-byte where
applicable. No GPU inference was added.

For the historical Exeunt event, removing B0H5 reduces the three-piece sequence
probability from **25.3876% to 1.67076%**, almost entirely through the initial
` Ex` probability. Across **all 64 individual head removals**, conditional
`e`+`unt` probability remains at least **99.7347%**. B0H5 and B2H2 damage corse
more than Exeunt. B3H6 is a post-hoc candidate with smaller effects on the two
other events, not a demonstrated word-specific head. Whole B0-MLP removal
destroys the suffix too, but its documented broad damage prevents a unique
lexical-storage interpretation. These are older-model results, not results
from the still-running paired models.

The frozen new corpora provide a complementary **input-level**, noncausal fact:
every original training `Ex`/` Ex` followed by `e` is followed by `unt` (936/936).
But the exact-case replacement leaves seven lowercase `exeunt` examples and
34 `Blunt` examples in training. Thus the replacement arm retains seven
`e`->`unt` transitions, not zero. The saved deterministic sampler includes three
complete lowercase presentations through step 100 and eight through step 200;
an independent scalar/native-byte audit reproduced them. Replacement-arm
exposure remains conditional on reaching those steps, since it has not started.
The exact-case protocol is unchanged.

New immutable JSON outputs at this root:

- `historical_word_ablations.json`: all 80 arms, all three words, first-piece /
  suffix / joint probabilities and NLLs; SHA-256
  `46e9f07f8998092325c165c8cf5fec4e0443cb8bbbc8caf4f9d0d2d125d30522`.
- `paired_spelling_statistics.json`: authenticated native train/test corpus
  counts and shared-piece controls; SHA-256
  `5550fd37d8697d7324b037e8163f29b792ed2a6797d8cd66e679ea93ba6f0147`.
- `lowercase_spelling_exposure_through200.json`: conditional saved-window
  exposure counts; SHA-256
  `a042c7dbeddbaf4878b55012ec339488c2e6f1add755dd12fff458404f0cf884`.

The matching analyzers add 21, 13, and six CPU tests. The complete suite passes
**1,035 tests in 32.333 seconds**; log:
`/tmp/pluto-exeunt-selection-spelling-cpu-tests.log`. Mock training-completion
messages in that test log are fixture output, not real experiment state.
The exposure helper's v1 source records identify final source bytes, not a
full before/after source-stability envelope; actual input hashes and counts
were independently checked. The detailed note records this limitation.

At **14:51:40 UTC** the supervisor, original trainer, trajectory observer, and
all three causal observers were independently verified live by PID/start ticks.
All frozen causal inputs remain unchanged. Original checkpoints are 0/100/200;
replacement has no checkpoint directory yet. Trainer 1727028 was the sole GPU
compute process at the preceding 14:49 check. No process was restarted, no
checkpoint changed, and no commit was made.

The next causal interpretation must distinguish first-piece selection from
suffix completion and account for the surviving lowercase/shared-token signal.
Neither the historical screen nor empirical token-transition frequencies prove
a sufficient or selective circuit for the complete word.

## Work still required

A separate CPU-only recheck of **historical** step-13030 native ablations is
recorded in [EXEUNT_B0_MLP_HISTORICAL_DIAGNOSTIC.md](EXEUNT_B0_MLP_HISTORICAL_DIAGNOSTIC.md).
It confirms broad passage-level damage from block-0 MLP removal and adds measured
clean-write magnitudes/output-distribution entropy. The follow-up feature check
finds distributed, partly cancelling signed support for `e` and `unt`, rather
than dominance by one neuron, and exposes missing same-current-token controls.
Evidence files are `historical_b0_diagnostic.json` and
`historical_b0_features.json` at this experiment root, explicitly labeled
`new_deterministic_experiment_result: false`. Neither is new paired-run evidence
or a substitute for selective causal interventions.

The [historical attention-route readout](EXEUNT_ATTENTION_ROUTE_HISTORICAL.md)
also joins block-1/head-2's reconstructed read of the earlier ` Ex` piece with
its retained native head-removal result. The head contributes to confidence,
but removing it leaves `e` and `unt` rank 1; the three-piece sequence probability
drops about 4.92%. This is a bounded causal contribution, not source-specific
routing necessity or a result from the newly trained models. Raw joined data:
`historical_b1_head2_route.json` at this experiment root.

The supplemental corpus-only suite is now frozen in `supplemental_cases/`:
128 word-plus-exact-next-token cases and 83 shared-piece controls, 211 total.
All preselected word crossings have a following native token; none was excluded.
Shared-piece coverage is necessarily incomplete: `Ex`, ` Nu`, and `uve` have no
eligible nonreplacement occurrences in either split, and `unt` has none in the
test split. These missing controls are recorded, not replaced by invented text.

The native factorial binary and its GPU test build. All 805 Python
weight-analysis tests and both CPU-only C++ targets (`checkpoint_validation_test`
and `embedding_factorial_probe_test`, eight cases each) pass. Logs are
`/tmp/pluto-exeunt-mechanism-cpu-tests.log` and
`/tmp/pluto-exeunt-native-cpu-tests.log`. The factorial GPU test and all real
factorial measurements remain **unrun**, deliberately deferred until controlled
training and the queued scorer release the GPU. No causal probability-transfer
result is claimed from a successful build or CPU tests.

At 11:48:55 UTC the original trainer and waiting analysis observer were verified
live again by process identity; the trainer was the only CUDA compute process.

Finish both four-hour runs; inspect the full matched-step behavior/weight
trajectory; run bidirectional embedding/input/readout interventions; test
complete word plus its following token; measure shared-subword collateral
effects; localize and causally confirm the relevant upstream computations.
Large deltas, a clean training fork, or a passed test suite alone do not satisfy
the primary objective.
