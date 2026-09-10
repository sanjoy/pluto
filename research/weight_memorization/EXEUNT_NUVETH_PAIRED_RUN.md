# Exeunt / Nuveth paired run — September 10, 2026

Status: **running; not yet an analysis result**. The supervisor started at
2026-09-10 02:10:40 UTC. It first creates the shared initialization and runs
three short controls, then trains the original and replacement arms for four
hours each, sequentially on the GH200. Check authoritative `state.json` and
live processes for current status rather than interpreting this note as a
heartbeat.

## Preserved artifacts

Root: `/home/ubuntu/checkpoints/exeunt_nuveth_20260910`

- `manifest.json`: frozen inputs, binary hashes, settings, native alignment.
- `state.json` / `runner.log`: process transitions and validated terminal results.
- `initial/checkpoints/step_0`: shared random initialization.
- `control_a`, `control_b`: identical original-corpus two-update runs.
- `control_replacement`: replacement-corpus two-update run.
- `original/checkpoints/step_N`: original-corpus four-hour arm.
- `replacement/checkpoints/step_N`: replacement-corpus four-hour arm.

Every child retains its initial and final checkpoints, with periodic
checkpoints every 100 updates in the long arms. No checkpoints are deleted.
All runs load the shared step_0 with fresh optimizer state and seed 17.

## Code and validation before launch

- `6a33946`: completed-GPU-update wall-clock training budget, initial/final saves.
- `5d9211f`: paired-run harness, native alignment proof, read-only weight diff,
  protocol, and sampling-replay build target.
- Optimized build of the trainer, native tokenizer exporter, and sampler passed.
- Five optimized regression targets passed: trainer, checkpoint, dataset,
  GPT-2 recipe, and CLI validation. Dataset tests require the tokenizer path;
  the first invocation omitted it, then all five passed with it supplied.
- 40 CPU experiment/comparison/sampler tests passed.
- Real full-model, batch-10 timed smoke test stopped after one completed update
  (41.175 seconds), saved step_0 and step_1, and verified all weights finite.
  Smoke train/test loss changed from 10.7595/10.735 to 9.32366/9.28387 using
  one evaluation batch. This is only a smoke test, not the paired result.

Full configuration and interpretation rules are in
`EXEUNT_NUVETH_PAIRED_PROTOCOL.md`.

## Exact native corpus checks

| Split | Tokens in each arm | Exact replacements | Changed token IDs |
| --- | ---: | ---: | ---: |
| Full | 1,835,163 | 1,028 | 3,084 |
| Training | 1,650,781 | 936 | 2,808 |
| Test | 184,382 | 92 | 276 |

Every replacement occupies the same three token slots in both arms. All other
IDs and byte boundaries are identical; only internal subword boundaries may
differ. Counts are case-sensitive `Exeunt`, not a case-insensitive word count.

## Immutable identifiers

Source commit recorded in the prepared manifest:
`5d9211f54a4365ecbff11be6fcfdf95e61729221`.

SHA-256 values:

```text
manifest.json
e2164761044698b3832e819649c78e6ad33748a2ea3cc40c5376d42e0d255277

frozen optimized trainer
07799aced9622c19c1d91ba00bdd5ac43fc2dfb31bbad7693be159643086e82a

original corpus
4249913cc5998b89bb845fffe1c200dfe6094d4dbe5c0b77b3e806fc680c2f3f

replacement corpus
40f4d3326f1a8abf737adc0dbac8e168044875b166d8d7ab6083fbf25da5d795
```

## Early two-update controls (not the four-hour result)

All controls completed two updates from byte-identical step_0 files. Original
control A ended at training/test loss 8.76844/8.6767; the comparison below uses
the complete FP32 checkpoint tensors, not rounded logged losses.

Across all weights, the original-vs-original repeat difference has L2 norm
0.03123350 (0.01962% of weight norm). Original-vs-replacement has L2 0.04023048,
only 1.288 times larger. Thus global early differences are substantially
confounded by numerical run-to-run variation.

Nevertheless, the **seven largest embedding-row differences** in the
replacement comparison are all constituent tokens of Exeunt/Nuveth. Several
stand well above the corresponding unchanged-data repeat variation:

| Token piece | ID | Replacement delta L2 | Repeat delta L2 | Ratio |
| --- | ---: | ---: | ---: | ---: |
| `unt` | 2797 | 0.00924474 | 0.0000232964 | 396.8 |
| `ve` | 303 | 0.00921043 | 0.00000147955 | 6,225.1 |
| ` Nu` | 21733 | 0.00909185 | 0.00000137529 | 6,610.8 |
| `Ex` | 3109 | 0.00882267 | 0.0000602149 | 146.5 |
| ` Ex` | 1475 | 0.00600423 | 0.0000243522 | 246.6 |
| `th` | 400 | 0.00530857 | 0.0000301525 | 176.1 |
| `e` | 68 | 0.00195537 | 0.000780016 | 2.5 |

These are descriptive ratios from one pair of repeat controls, not statistical
significance tests. The embeddings also serve as tied output weights, so a
changed training target directly affects them; this is not evidence that a
whole word or its contextual mechanism lives in a single row. Short-run repeat
variation cannot bound long-run numerical divergence.

Full evidence is retained in the experiment's `analysis/` directory:
`control_repeat_step2.json`, `control_replacement_step2.json`, and
`control_token_rows_step2.json`, with input checkpoint SHA-256 hashes.

The original four-hour arm started its training budget at **02:17:30 UTC**.
Final steps, elapsed times, losses, long-run weight comparisons, and behavioral
results remain to be measured. Raw weight deltas alone will not be described
as a localized word memory.

## Frozen behavioral tests

Before any final-model scoring, prepared 160 cases in `word_cases/`:
16 word contexts from training and 16 from test, each crossed with both native
prefix versions and both candidate words (128 cases), plus 16 unchanged
three-token control continuations per split (32 cases). Both native word
tokenization variants are represented in each split. Contexts were selected
using only corpus data and seed 17, not model predictions.

Prefixes use up to 128 preceding native tokens and reset positions to zero.
The three teacher-forced conditional probabilities are measured at temperature
1 and multiplied in log space. No following delimiter is scored, so this is
the probability of a particular three-token spelling, not a universal measure
of complete-word generation or training-set memorization.

Code commits: `276a9d5` adds the native read-only scorer; `3ece9f1` adds the
case preparation and score summaries. All 59 CPU experiment/sampler tests
pass. The native scorer's optimized build and pre-CUDA rejection checks pass;
its real GPU evaluation is deliberately deferred until both training arms end.

Frozen packed inputs SHA-256 (1,310,720 bytes):
`1ae973d97c703df6a1c803f6d947ead56658607cbe820a5cd828ad1abbb9dd6e`.

Frozen scorer SHA-256:
`d7a755565f776c330b6070442bc01546d321f35516c16b23fd090aa458765330`.

`word_evaluation_provenance.json` records canonical paths and all case/scorer
hashes. The actual score dumps do not exist yet.

## Sampling exposure and pending final analysis

An independent CPU check compared the native input and target slices at the
frozen sampler's starts. The first update has **zero changed input tokens and
zero changed target tokens**. Through the second update, there are 18 changed
input slots and 18 changed target slots, comprising six distinct complete word
replacements (five with leading space, one without). This explains why the
two-update control can already expose a direct token-row signal even though
the corpus edit is small. It does not eliminate floating-point nondeterminism.

`analysis/sampler_exposure_prefix.json` records cumulative target exposures.
For a sequence starting at `s`, targets occupy `[s+1, s+1024]` inclusive;
a complete three-token replacement must fit entirely within that interval.
Repeated visits count separately. The following are consequences of the
frozen sampling schedule, **not claims that the live run reached these steps**:

| Updates | Complete word exposures | Distinct training occurrences |
| --- | ---: | ---: |
| 2 | 6 | 6 |
| 100 | 595 | 443 |
| 200 | 1,190 | 679 |
| 300 | 1,772 | 789 |
| 400 | 2,381 | 862 |

Commit `c84d239` adds a postprocessor that waits for the supervised training
to finish, independently validates both terminal runs and checkpoint hashes,
then compares the final endpoints and the latest common positive step.
It also scores the frozen word/control cases with the native model. The
waiting process was launched at approximately 02:36 UTC; output is reserved under
`analysis_final/`, with its log at `postprocess.log`. A live process check at
02:42 UTC confirmed that the original trainer was the only GPU workload and
the postprocessor was still waiting. These are launch observations, not final
results. A missing or failed process must be diagnosed, not silently restarted.

Commit `16f2637` adds independent selective checkpoint copies for a possible
causal follow-up: replace only selected tensors or embedding rows with donor
bytes, preserve all other bytes, and verify both source checkpoints unchanged.
Use only completed, immutable checkpoints. No real checkpoint has been patched
yet. An embedding-row patch changes the tied output head too, so any resulting
behavior change cannot by itself separate input-embedding and output effects.

All 93 CPU tests for paired training, comparisons, word cases, postprocessing,
sampling replay, and selective patches passed. Neither four-hour endpoint nor
final behavioral evaluation is complete at the time of this update.

## First periodic checkpoint: original arm, step 100

The original arm saved `original/checkpoints/step_100` at **03:26:08 UTC**.
At **03:26:15 UTC**, its fixed four-batch training evaluation reported loss
**5.32169**, down from the initial **10.7495**. This is not a new test-split
evaluation, and the replacement arm has not started yet. Thus these numbers
show ordinary learning progress, not the effect of replacing Exeunt.

CPU-only validation found exactly 100 expected weight files, all with correct
sizes and all finite: **51,483,648 FP32 parameters**, **205,934,592 bytes**.
Their FP64 weight L2 norm is `169.032921957187`. Directory/file identities,
sizes, modes, modification times, and change times were unchanged across the
audit. The source checkpoint was neither modified nor copied over.

Aggregate checkpoint SHA-256:
`766b1f2b40b7f2e8755fc9dc52a87b29b69b790a32b348fa85ace3e29ab63af7`.
The aggregate hashes UTF-8 records `filename SHA256(file)\n`, sorted
lexicographically by filename; it is not a hash of concatenated weight bytes.

The verified sampling schedule for these 100 updates contains 1,024,000 target
token exposures, including 595 complete Exeunt exposures covering 443 distinct
training occurrences. The later replacement run is configured to use those
same windows. Its actual step-100 checkpoint remains pending.

At 03:27 UTC, the original trainer and waiting postprocessor were both live,
and the trainer was still the only GPU workload. The first 100 updates plus
their periodic evaluation took approximately 68 minutes 45 seconds. The
four-hour endpoints and paired/behavioral comparisons remain incomplete.

Commit `bf472c9` fixes the first long-run causal check to bidirectional transfer
of the nine word-token embedding rows at the latest common positive step,
using the frozen cases. It also explicitly retains the earlier common-step
weight comparisons from the original protocol. Neither has been run yet.

## Second periodic checkpoint: original arm, step 200

Saved `original/checkpoints/step_200` at **04:34:53 UTC**; the fixed four-batch
training loss at **04:35:00 UTC** was **4.96954** (step 100: 5.32169).
This remains a training-only progress report, not a paired-corpus comparison.

CPU validation again found exactly 100 correctly sized files containing
51,483,648 finite FP32 parameters (205,934,592 bytes). FP64 weight L2 norm:
`188.5169101379556`. Directory/file stat identities were unchanged during
validation. Aggregate SHA-256, using the same filename/hash-record convention
as step 100:
`464017f2a3263d518fadd14dd9609dd3f0dd13b34e42d5ca2fac2ad500d8de84`.

The original trainer, supervisor, and waiting postprocessor remained live.
No GPU analysis was launched and no weight files were modified or deleted.
