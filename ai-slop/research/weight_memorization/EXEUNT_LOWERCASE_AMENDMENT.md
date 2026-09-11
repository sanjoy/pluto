# User amendment: replace lowercase exeunt before the replacement arm starts

The user's 2026-09-10 request supersedes the original exact-capitalization-only
replacement policy. The primary goal remains a causal explanation of how the
weights encode Exeunt; the revised intervention removes both observed spellings:

| Original spelling | Replacement spelling |
| --- | --- |
| Exeunt | Nuveth |
| exeunt | nuveth |

Capitalization is preserved. This is two explicit literal substitutions, not
removal of shared subword tokens such as `unt` in `Blunt`. Other case variants
must be detected rather than silently assumed covered.

## First matched learned checkpoint now available

The amended replacement wrote step 100 at **17:00:39 UTC** on 2026-09-10;
its training-evaluation loss is **5.32869**, versus original **5.32012** at the
same step. At 17:11 UTC its identity and 751 frozen records were revalidated;
the four-hour training job remained live. A fresh scan confirms exactly 1,028
`Nuveth` and seven `nuveth` occurrences and no case-insensitive `exeunt` matches.
Original training and every prior checkpoint are unchanged.

The CPU-only matched comparison finds that the top eight embedding-row changes
are edited-word pieces. This is not a causal localization or new native word
score. See [EXEUNT_EARLY_STEP100.md](EXEUNT_EARLY_STEP100.md) for the authenticated
checkpoint comparison, forward-precision audit, and exact exposure counts.

## Further causal assay queued without disturbing training

At **17:40:40 UTC**, a new historical first-piece head-position observer entered
its waiting state behind the historical suffix observer. It does not modify
either amended corpus or training process. PID 1859615, start ticks 128467285,
managed session 53034; 406 frozen records independently checked. The full CPU
analysis suite passed 1,361 tests. This assay's historical checkpoint remains
separate from the new paired models, and no native effect has been measured
yet. See [EXEUNT_HEAD_POSITION_PROTOCOL.md](EXEUNT_HEAD_POSITION_PROTOCOL.md).

## State at the user's request

At 15:04:22 UTC the original training child (PID 1727028) and its supervisor
(PID 1723614) were live. The replacement long arm had **never started**; the
state contained no replacement run and its output directory did not exist.
Original training is not being restarted, paused, or given a different corpus.

All original manifests, native token exports, controls, checkpoints, and
historical analysis artifacts remain preserved under
`/home/ubuntu/checkpoints/exeunt_nuveth_deterministic_20260910_1137` (LEGACY).
The revised inputs and replacement run use the new exclusive directory
`LEGACY/lowercase_amendment` (AMENDMENT).

## Preventing the superseded corpus from launching

The legacy supervisor creates each arm directory exclusively **after** fully
recording the preceding arm's terminal log and checkpoint inventory. A reserved
`LEGACY/replacement/` directory containing
`SUPERSEDED_BEFORE_LAUNCH.json` therefore prevents its old replacement command
from launching. The expected exclusive-directory FileExistsError is an
intentional handoff, not a failure of the original training child. A new runner
must verify that exact handoff, the original child's successful return code,
full four-hour budget, initial/final/every-100-step checkpoints, and actual
process exit; an arbitrary failure is not accepted as successful completion.

The four old-manifest waiting analysis observers were deliberately retired:
trajectory 1726864, embedding 1752078, branch 1755949, source-value 1797771.
Their requests, frozen copies, and outputs are retained, with the reason in
`LEGACY/ANALYSIS_QUEUE_SUPERSEDED.json`. None was running a GPU probe. The
supervisor and original trainer were not signaled. Revised analyses must use
amended provenance, not resume these old requests as if nothing changed.

## Alignment requirements

The original data has seven lowercase occurrences, all in training, all with
leading-space native tokenization:

    [409, 68, 2797]  = " ex", "e", "unt"
    [14364, 303, 400] = " nu", "ve", "th"

This local check does not prove general lowercase token-count preservation:
bare `exeunt` is two tokens, whereas bare `nuveth` is three. The amendment must
re-export the complete actual texts with the same native tokenizer and prove
three-token spans for every actual replacement, identical total counts, an
unchanged split, and identical IDs/outer byte boundaries everywhere else.

Expected totals, subject to that native validation, are 943 changed words in
training and 92 in test (1,035 total), compared with the former 936 and 92.
The seven additional words change 21 additional token slots. Existing
unamended-corpus readouts correctly describe their old input hashes; their
statements about surviving lowercase examples do not describe the revised arm.

## Training and analysis requirements

The amended replacement arm must use the same saved random initialization,
frozen trainer/tokenizer, seed 17, batch size 10, optimizer/settings, and full
four-hour training budget as the original arm. Its checkpoint directory is
`AMENDMENT/replacement/checkpoints`; step 0, every 100th step, and the final
checkpoint must be retained. A fresh two-step replacement control runs only
after original training releases the GPU. The old original run is referenced
with its actual command, paths, and log—not relabeled as a newly executed run.

Evaluation cases must carry exact spelling pairs, keep existing uppercase
selections identifiable, and add actual lowercase training cases. There are
no lowercase test occurrences; do not invent a held-out corpus case. All
control masks and prefix bytes must be regenerated against both substitutions.
The actual embedding-row union is now eleven, not nine:
`45, 68, 303, 400, 409, 1475, 2797, 3109, 14364, 21733, 45177`.
Existing native scoring/factorial probes accept these IDs; Python provenance,
case-label, imported-run, and row-count assumptions require adaptation.

This document records authorization, the safe handoff, and required invariants.
Verified preparation and launch-state results follow; queuing the handoff is
not a claim that replacement GPU training has started.

## Native input validation completed

The new corpus helper and ten focused tests pass. The real native export then
completed successfully and published `AMENDMENT/amendments.json`, SHA-256
`efa78f823566bf37d5a26d34ff1c6e678c5c3bf0df1aeedc915ddad30b91f7d7`.
All 35 provenance and four source records were independently rehashed.

| Split | Tokens (unchanged) | Exeunt replacements | exeunt replacements | Changed IDs versus original |
| --- | ---: | ---: | ---: | ---: |
| Full | 1,835,163 | 1,028 | 7 | 3,105 |
| Training | 1,650,781 | 936 | 7 | 2,829 |
| Test | 184,382 | 92 | 0 | 276 |

All seven lowercase replacements have native IDs `[409,68,2797]` to
`[14364,303,400]`. Relative to the old uppercase-only replacement, exactly
21 additional IDs change in full/training and zero in test. Every other token
and outer byte boundary is unchanged. Raw replacement text equals the two
requested literal substitutions exactly. A separate case-insensitive scan
confirms **no spelling of Exeunt remains** in any amended replacement split.
The old original and old replacement files remain unchanged.

## Revised training queued

The handoff runner passed 44 focused CPU tests (54 together with the input
helper tests). It entered `waiting_original` at **15:25:10 UTC** on 2026-09-10:

- Managed execution session: `91141`.
- Runner PID: `1816208`; kernel start ticks: `127653562`.
- Request: `AMENDMENT/request.json`, 257,091 bytes, SHA-256
  `2cf4047d78aaac1906630ac071817930f40d6931bff09bea6a57728349192f24`.
- State/log: `AMENDMENT/state.json` and `AMENDMENT/runner_session.log`.

An independent read-only check verified all **751 frozen input records**, the
runner's actual arguments/identity, and the revised corpus paths in both its
two-step control and four-hour replacement commands. The long-run arguments
match the original run except for corpus, checkpoint destination, and log.
Both use the same saved initialization; the amended run does not resume from
the original arm's learned weights.

At that check, the original child was still the only GPU process. Neither
amended GPU run directory existed. The runner will wait for both exact legacy
processes to exit, validate the completed original and deliberate handoff, then
run the new control and replacement arm sequentially. Its own source and
request inputs are now frozen: do not edit them while it is waiting/running.

The original run continued uninterrupted through step 300, checkpointed at
15:22:07 UTC with training-evaluation loss **4.66668** at 15:22:15 UTC.
These identifiers and state observations are historical; recheck live state
on continuation. The retired old analysis queue has not been reactivated.

## Amended evaluation cases and tests completed

The new suite has **188 cases**: the existing 160 rows in their original order,
plus all seven lowercase training occurrences crossed with both prefix domains
and both candidate spellings (28 rows). There are no lowercase test examples.
All 32 unrelated controls avoid every revised replacement span. Prefixes were
rebuilt from the amended inputs; the selected inherited rows happened to avoid
the seven new changes, so all 160 inherited packed rows remain byte-identical.
Fifty-six inherited occurrence indices shift; matching uses stable token starts,
not the old ordinal occurrence index.

`AMENDMENT/word_cases/cases.json` is 250,667 bytes, SHA-256
`282a8084ad465dcc72284746beaff01a5ab2deaa85b9549a3901975dfee3daa0`.
`packed_cases.bin` is 1,540,096 bytes, SHA-256
`53a314711eac9fb655a9020439a1a7e71f82322ad30cea69fdd594d51af15edd`.
All 41 provenance and five source records were independently rehashed. The
distinct `pluto-paired-lowercase-word-cases-v1` format carries exact spelling
pairs and spelling variants; old uppercase-only readers intentionally reject
it. Downstream analysis must be adapted before requeuing.

The full CPU suite passed **1,119 tests in 32.754 seconds**, including the new
input, handoff, and case tests. Log:
`/tmp/pluto-lowercase-amendment-cpu-tests.log`. CUDA was hidden for this suite;
its mocked launch messages are not real GPU training or analysis executions.

## Case-aware trajectory analysis queued

At **15:41:45 UTC**, the new observer entered its verified waiting phase:

- PID `1820183`, kernel start ticks `127753878`, managed session `69927`.
- Output: `AMENDMENT/analysis_trajectory`.
- Log: `AMENDMENT/analysis_observer.log`.
- Request: `analysis_trajectory/request.json`, 18,946 bytes, SHA-256
  `ca47d76fa1c6e3c79f64cd575c9693695295edf75f1b7db24c57463c56425b20`.

An independent live-process check verified its exact arguments and all 72
frozen request records, including source copies and the 188-case suite. The
original trainer was still the only GPU process. This replaces the old
trajectory observer; it does not reactivate any of the retired observers.

`paired_lowercase_analysis.py` references the original arm at LEGACY and the
replacement arm at AMENDMENT, revalidates the deliberate handoff, shared
initialization, controls, actual commands, full terminal time budgets, periodic
checkpoint inventories, finite weight values, and actual process exits before
scoring. Both arm roots remain explicit in its output; it does not rewrite the
original command or fabricate a new execution. It plans every common saved
optimizer step, initialization (one score reused for identical initial copies),
and the final endpoints, with unequal endpoint steps explicitly identified.

`paired_lowercase_scores.py` validates role/variant labels, native target bytes,
teacher-forced packing, identical prefixes within candidate pairs, and all four
domain/candidate crossings. It reports full three-piece probabilities, the
first-piece contribution, and the conditional remaining spelling. Candidate
log-probability ratios are paired within exact spelling variants, never pooled
across title and lowercase. Between-model comparisons preserve case identity
and make the signs/units of NLL and candidate-ratio differences explicit.

The new code passed **36 focused tests**; the full CPU suite passed **1,155 tests
in 33.103 seconds**. Log: `/tmp/pluto-lowercase-analysis-cpu-tests.log`.
The real 188-case suite also passed an independent preflight through the new
reader. The observer's loaded source files are now frozen; do not edit them
while it waits/runs. The analysis itself has not yet produced GPU scores.

Remaining work includes amended supplemental/shared-piece controls and
case-aware embedding-factorial, attention/MLP transfer, and source-value
interventions. The old causal readers/planners are not yet compatible with the
new output format; do not point them at it or claim that these later assays are
queued. Weight deltas and ordinary trajectory scores alone do not complete the
primary mechanistic objective.

## Original completed; amended four-hour arm running

The original run completed normally at **step 331**. Its terminal log reports
`time_limit`, **14,421.9 training seconds**, training-evaluation loss **4.60726**,
and test-evaluation loss **4.57047**. The final checkpoint was written at
15:44:51 UTC. These losses use the configured evaluation batches, not every
token of each split. A fresh CPU inventory independently verified all 100
canonical finite FP32 weight files at steps **0, 100, 200, 300, 331**.
`LEGACY/original/checkpoints.json` SHA-256:
`6382ae1d50daa2367a077e78f9cfdc17bcb497ccab66234d55351b4907eb3ee6`.

The legacy supervisor then encountered exactly the reserved-directory failure;
its old replacement was never launched. Both old processes exited. The amended
runner validated the successful original run and published
`AMENDMENT/original_reference.json` (50,636 bytes), SHA-256
`5bb68c0dcbd0b910f9551455b2ff3503d210d9bdd0458fb90bb3b9c6a03146ed`.

The new two-step replacement control ran from 15:45:45 to 15:47:52 UTC. Its
weights independently rehash **identically to the earlier uppercase-only
replacement control at every step 0, 1, and 2: 300 matching weight-file pairs**.
Their checkpoint-inventory SHA-256 values are respectively:

- Earlier `LEGACY/control_replacement/checkpoints.json`:
  `cc3d60e432bc4809f41b2f0784659b6b1947c7b4823ca040a53911d548994044`.
- Amended `AMENDMENT/control_replacement/checkpoints.json`:
  `8b013156441f1d1d0c5e268930c88f6587251cb8eaf0039457466e15f6d6541c`.

This exact match is expected, not evidence of a null lowercase effect: a fresh
comparison of the first 20 saved native sampled windows found identical input
**and shifted target** IDs between the old and amended replacement corpora.
The first lowercase-changing window is global sequence 314: **step 32,
sequence 4 (one-based)**, starting at corpus token 918223. The saved sampler
export begins with its seed before the 10,000 starts; that header was excluded.

The full amended arm launched at 15:47:53 UTC as **PID 1821367**, start ticks
**127790846**, parent 1816208. Its training clock began **15:48:09 UTC** after
writing step 0 from the shared initialization. Approximate budget completion is
19:48 UTC, plus the final in-flight step/evaluation/checkpoint. Initial amended
losses were 10.7496 training and 10.7207 test. A live identity/GPU check at about
16:02 UTC confirmed it was the only GPU compute process; the trajectory observer
was verified waiting on this same amended run. Do not restart either controller.

## Amended causal inputs and readouts ready

`paired_lowercase_supplemental.py` prepared **265** additional cases:

- **156** word-plus-exact-following-token cases: all 128 title-case crossings
  and all 28 lowercase crossings, with the first three causal inputs preserved.
- **109** unchanged/disjoint shared-subword cases: 64 training, 45 test.

Coverage uses all eleven amended word-piece IDs. Training lacks eligible
unrelated examples for IDs 3109, 21733, and 45177; test additionally lacks ID
2797 (`unt`). Coverage is recorded, not invented. The training `unt` controls
include real occurrences of **Blunt**, allowing a future assay to distinguish
damage to that shared subword from Exeunt-specific behavior. These are controls,
not yet measured effects.

`AMENDMENT/supplemental_cases/cases.json`: 408,591 bytes, SHA-256
`9baf51672de90483ba28eeee3ebe00be3e2196f796d4cf608d8a6455985d96e0`.
Packed batch: 2,170,880 bytes, SHA-256
`906df6b54d4d6f06279f1ff03facbe19b93b7f8990a3f84f5db3e85c797a653a`.
All 44 provenance and five source records were independently rehashed.

Three fixed-row native factorial inputs are exported under
`AMENDMENT/causal_cases`: main **188 x 3**, word-next **156 x 4**, shared-piece
**109 x 3** scored rows. Their source order, exact candidate labels, packed bytes,
and selected row files passed the actual factorial readout's CPU input validator.
`causal_cases/exports.json`: 11,154 bytes, SHA-256
`589bd2492b78c12406ce92db3c4849f8761cf6e916dca4b4b28803489f9957c3`.
The exporter source is recorded in this artifact; preserve it when extending
the amended orchestration (prefer a new amended planner).

The embedding-factorial and attention/MLP branch readouts now accept both
amended suites through `paired_case_contract.py`. They require explicit variant,
candidate pair, and source role; separate even identical-prefix title/lowercase
contexts; retain absolute probabilities, subtoken contributions, collateral
effects, and exact-following-token scores; and preserve legacy title-only
behavior. An eleven-row embedding patch passed the synthetic full-logit tests.
The causal readouts define odds as **replacement/original**; the trajectory
scorer defines **original/replacement**. Their explicit definitions must be
respected when combining results. Lowercase branch groups do not misuse the
historical `Nuveth_over_Exeunt` field name.

All **1,176 CPU tests pass** (33.775 seconds), log
`/tmp/pluto-lowercase-causal-readouts-cpu-tests.log`. Both live controllers'
frozen inputs remained unchanged during this work. No GPU assay ran.

Still required: adapt the completed-trajectory intervention planner and followup
controllers to the imported original root and amended summary format, then
queue/run the actual weight transfers and source-value/ablation assays. The old
planner's full `prepare` path remains legacy-only; only its byte-preserving case
export accepts the new formats. Later causal stages are **not yet queued**.

## Matched-step weight transfers queued (16:30 UTC)

The new `paired_lowercase_intervention_plan.py` authenticates the completed
amended trajectory and independently revalidates both training budgets, controls,
terminal logs, real source roots, and checkpoint bytes. It selects the earliest
and latest common positive optimizer steps; unequal endpoints are context, not
matched-step interventions. Its embedding row union comes from all actual native
replacement spans, including the lowercase amendments.

The new `paired_lowercase_causal_followup.py` is now **waiting**, not executing
GPU probes. It was launched at 16:30 UTC in managed session **97587**:

- PID **1829540**, start ticks **128043774**, parent **1829537**.
- Exact upstream trajectory observer: PID **1820183**, start ticks **127753878**.
- Request: `AMENDMENT/causal_stage/request.json`, 29,767 bytes, SHA-256
  `343a9ed27fd759792ca241787e5916727e50891642bd23e4a4816f7eaa5bbecb`.
- Log: `AMENDMENT/causal_observer.log`; current phase `waiting_trajectory`.
- **111 frozen input records**, including 21 original Python sources and their
  copies, independently verified after launch. Do not edit these live sources.

After actual upstream exit and successful completion gates, the observer will
first run native GPU validation, then weight-identical copy controls, embedding
input/output factorials on all three suites, and both-direction attention/MLP
whole-branch and output-write transfers. Every copy must reproduce all trajectory
losses and argmax bytes exactly. Adding the exact following token must preserve
the first three scores. Completed interventions are published separately, so
partial results survive any later failure. No automatic training restart exists.

The loss-probe binary was verified byte-identical to the trajectory producer:
SHA-256 `c09eab407ec953210b92fa485dda716638d92eab3d24a0ae32b95d28e53ad28b`.
The factorial probe and GPU test match their original frozen records too:
`3c333c94d3c79f46e4ad01781ecd85dd29075b062e8e368dc1df58feca6e6558` and
`16746efb0b34406a791ced96621895a1649d1c1959a0f7fe991d867e7deed5e6`.

All **1,211 CPU tests pass** (35.979 seconds), including 15 new planner and 20
new controller tests. Log: `/tmp/pluto-lowercase-causal-controller-cpu-tests.log`.
An independent read-only code review found no substantive wiring defect. Its
resource caveat is that the two-checkpoint screen includes 59,796 single-sequence
loss-probe forwards plus factorial work, so the post-training analysis may take
substantial time. This is planned work, not a measured timing estimate.

At launch verification the amended trainer (PID 1821367) was still the sole GPU
compute process. No causal effect has yet been measured in these new models.
Source-value/head/neuron ablation followups still require their own amended
controllers or probes; completing the weight-transfer screen will not by itself
complete the mechanistic research goal.

## CPU-only input-path predictions for the eleven-row factorial

After independently validating all three exported suites (61 records), a
read-only calculation used `paired_input_exposure.exposures` on the actual
packed input rows and eleven amended piece IDs. A prediction sees only input
positions at or before its scored row; future teacher-forced pieces cannot
affect it. Counts below deduplicate causal prefixes **within each spelling
variant, split, and target position**, not across all groups.

| Word cases | First target: untouched / touched | Second target | Third target |
| --- | ---: | ---: | ---: |
| Title, training | 16 / 0 | 0 / 32 | 0 / 32 |
| Title, test | 13 / 3 | 0 / 32 | 0 / 32 |
| Lowercase, training | 6 / 2 | 0 / 16 | 0 / 16 |

Thus **35 of 40 variant/split-specific first-target prefixes** contain none
of the patched IDs. For these rows, changing only those input embedding rows
cannot change the residual: the upcoming factorial must have `AA = JA` and
`AJ = JJ`. The tied output-head effect can nevertheless be nonzero. Even an
unpatched target's probability can change through the softmax denominator.
This is a structural prediction, not an observed intervention result.

The 28 lowercase case rows are seven occurrences crossed with two prefix
domains and two candidates. Six occurrences have identical prefixes between
corpora; the seventh contains a previous replaced title-case word, producing
eight distinct first-prediction prefixes. Untouched case 160 predicts the word
at token 238155 after a 128-token prefix ending `[Flourish. Cornets. Then`.
Touched case 176 predicts at token 1597534 after an earlier `Exeunt` at
1597427–1597429, IDs `[1475,68,2797]`. Its replacement-prefix alias, case 178,
instead sees `Nuveth`, IDs `[21733,303,400]`, at those same positions.

Ordinary controls have untouched/touched counts 14/2 for training and 16/0
for test at all three targets. The two exposed training controls contain
`unt` in **Blunt** (token 445802) and ` nu` in **nuptial** (token 872992).
Shared-piece controls are 51/13 training and 38/7 test at the first target;
all subsequent targets are exposed (64 training, 45 test). Word-plus-next-token
cases preserve the first-three results and are all exposed at the fourth.
Exposure permits an input effect but does not demonstrate one, necessity,
word selectivity, or a unique location storing the word.

## Historical source-value followup requeued (16:40 UTC)

The new `paired_lowercase_source_value_followup.py` adapts only the queue and
completion validation. It preserves the previously predeclared **historical**
step-1342 event from `/home/ubuntu/checkpoints/shakespeare/step_13030`: block 1,
head 2, query 1023 (`e`), predicting token 2797 (`unt`). Sources remain
1022 (` Ex`), 889 (` who`), 1023 (self), 1021 (space), and 512 (fixed distant
control), each at doses 1, 0.5, 0, and 1 again. The native source-V,
single-query/head-context splice and independent tensor/logit readout are
unchanged. **This is not a new paired-model or initial-piece result.**

The adapter waits for the exact combined causal observer to exit, then
rebuilds the immutable amended intervention plan in its own output directory
and requires full equality with the completed plan. This reruns training,
determinism, case-byte, checkpoint-byte and actual source-root validation.
Every prescribed intervention, completion marker and recursive artifact
ledger must agree before the historical GPU validation/probe may run. It
cannot start, resume, or signal training. Both actual native launches also
require the recorded GPU to be idle.

The new job entered `waiting_causal` at **16:40:17 UTC**, managed session
**92971**, PID **1833273**, start ticks **128105137**, parent **1833271**. Its
exact upstream is PID 1829540/start 128043774. Output:
`AMENDMENT/historical_source_value_stage`; log:
`AMENDMENT/historical_source_value_observer.log`.

`historical_source_value_stage/request.json`: 53,323 bytes, SHA-256
`6985d603d17969ae348266d9f32ca66d3535ff146cdb6688788521943b181746`.
All **215 frozen input records** were reverified after launch, including the
adapter itself when executed as `__main__`. The archived historical inputs
and all 100 weights were independently authenticated (107 records). Reused
native binary hashes match the retired observer's original records:

- Probe: `e381f1ea08b1f20f8c6359dbb4e6cd8220b6f7683988207edb5c9b220bdaacc4`.
- GPU test: `1a5280d55cc1dfbeb7078206c2d04df7bfaad7f36134c7fdb3303905333259c9`.

Final adapter source SHA-256:
`d7f9583fe6bb368414ec23282850f717d2bc54aa87f3f07e4a5b090cf9eb8c68`.
Its 29 CPU tests, plus the unchanged historical observer/readout tests, pass
(85 focused tests). The final complete suite passes **1,240 tests in 36.104
seconds**; log: `/tmp/pluto-lowercase-source-value-adapter-final-cpu-tests.log`.
All native runs in these CPU fixtures are mocked; they are not GPU evidence.

After launch, this stage contained no GPU-validation or native-output
directory. The amended trainer remained the sole GPU compute process; the
trajectory and combined-stage frozen inputs were unchanged. A separate raw
byte search also confirmed **zero case-insensitive `exeunt` occurrences** in
the actual replacement corpus being trained. Current sources for every live
observer are frozen; future extensions must use new files or wait for exit.
