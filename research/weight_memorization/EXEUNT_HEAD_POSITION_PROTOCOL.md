# Separating a head's local write from its other-position effects

Status: **preparation, not new GPU evidence**. The amended paired-training run
continues with exclusive GPU use. This diagnostic addresses a specific gap in
[the initial-piece analysis](EXEUNT_INITIAL_PIECE_CONTEXT.md); it is not a
replacement for the paired training, complete-word scores, or weight transfers.

## Question and fixed historical examples

For the archived Exeunt event, B3H6's clean fixed-direction contribution to the
` Ex`-minus-space margin is negative (-2.39225149), yet removing that head at
all positions reduces the actual margin (-1.66942406). Clean linear accounting
is therefore not a causal ablation prediction. The current evidence does not
separate downstream transformations at the prediction position from effects
that arrive through other positions in later attention layers.

The fixed first-piece events below were already used in the historical screen.
They are post-hoc and unmatched; contrasting their effects cannot by itself
establish lexical specificity. The old model is checkpoint
`/home/ubuntu/checkpoints/shakespeare/step_13030`, not either new paired model.

| Event | Generation step | Actual prefix rows / query | Target ID | Clean first-piece probability |
| --- | ---: | ---: | ---: | ---: |
| grandam | 365 | 371 / 370 | 4490 (` grand`) | 0.0316390347594 |
| corse | 700 | 706 / 705 | 1162 (` cor`) | 0.0615212467966 |
| Exeunt | 1340 | 1024 / 1023 | 1475 (` Ex`) | 0.253886825855 |

All native forwards execute 1,024 positions. Short prefixes are padded by
repeating their final input token: 616 (` my`) for grandam and 257 (` a`) for
corse. Exeunt's last input is 220 (space), with no extra padding required.
Do not move the two shorter events' selected queries to position 1023.

The source prefixes are `/tmp/pluto-three-words.qPVE2k/run/prefix_step_S.i32`.
Their SHA-256 values are:

- Step 365, 1,484 bytes:
  `10f77d6a39f6f26ce3be6971de8ba0b7710f0d257804987e912e2257ba49e7c5`.
- Step 700, 2,824 bytes:
  `e768c1f66430ae70dff7dc996efd583a93d73189b8082a114ee30186e81606f6`.
- Step 1340, 4,096 bytes:
  `31a43199955067fcb394d5abbff049908637ddf568f1fbf87abe424cbf45767c`.

The evidence manifest is `three_corpus_words_evidence_manifest.json`, SHA-256
`024fa4dbadb2b892a332585588754e34d2fa466f011754b71013a1a9979815bf`.
A CPU audit authenticated 165 records (662,077,421 bytes), including all 100
finite checkpoint weights, prefix/generation alignment, native clean/replayed
rows, selected captures and ablations. First-piece FP64 scores reproduce the
existing historical readout. No model was run during this audit.

## Position partition and native calibration

At each selected block/head, scale the captured BF16 attention **context** on
exactly one of three disjoint/composite position scopes:

- `query_only`: the selected prediction position only.
- `other_queries`: every other position of the same sequence, including future
  padding. Causality must prevent future positions from influencing the query.
- `all_queries`: both sets together, in the selected sequence only.

The selected heads are B0H5, B2H2 and B3H6. Preserve all other heads, sequences,
incoming residuals, weights and biases. Use doses 0, 0.5 and 1; every identity
dose must still run the native tail, including a final clean replay. Recompute
the output projection, residual addition, that block's MLP, every later block,
and final LayerNorm/tied head through the actual production kernels.

Creation must reproduce all clean physical-vocabulary logits at every row
byte-for-byte. Compare available archived BF16 context/incoming-residual rows
and the selected clean logit row too. Native physical logits have 50,272 FP32
entries per row; score only the 50,257 logical entries. The padding sentinel is
finite `-FLT_MAX` in the current kernel and the actual archive, despite an older
header comment calling it negative infinity.

At dose zero, the all-query context intervention must additionally reproduce
the archived all-query output-weight-row-zero logit row exactly:

| Head | Output weight | Zeroed rows, half-open | Bias stays unchanged |
| --- | --- | --- | --- |
| B0H5 | weight_6.bin | [320,384) | weight_7.bin |
| B2H2 | weight_30.bin | [128,192) | weight_31.bin |
| B3H6 | weight_42.bin | [384,448) | weight_43.bin |

Archived rows are `trace_step_S/ablation.blockB.headH.logits.f32`, each 201,088
bytes. A mismatch is a calibration failure to investigate, not a reason to
silently weaken equality. Half-context and half-weight edits need not agree
because BF16 rounding occurs at different points; never use their equality as
a gate or present them as the same intervention.

The historical producer source is recoverable at manifest commit
`5ae692b197849bed27fb02229f022b741c285d89`; its recorded SHA-256 is
`c94165709ea12e54e0b38cff5044d9ce57e96dbc4a813fb3776826f32f213ba9`.
The helper implementation is recoverable from that revision but lacks a
separate archived-manifest entry. Do not claim independent manifest
authentication of that helper or equality of today's executable.

## Readout and interpretation

Keep the strongest clean competitor fixed across all arms. Report absolute
target probability, NLL, rank/argmax, and the target-minus-fixed-rival margin.
For each scalar readout f, retain the nonadditive interaction:

    interaction = f(all) - f(query_only) - f(other_queries) + f(clean)

This is a contrast between disjoint position edits, not a unique pathway or a
fraction of a word's storage. Even additive logit changes can have nonadditive
log-probability effects through softmax. If query-only and all-query effects
have opposite signs, other-position effects/interactions matter; if query-only
already reverses the clean projected contribution, downstream processing at
that query is sufficient for the discrepancy. Neither outcome establishes an
Exeunt-exclusive head.

`head_position_math.py` supplies independent BF16-mask, causal unchanged-row,
and fixed-rival/interaction oracles. All 14 focused CPU tests pass, including
negative-zero fidelity, wrong-scope rejection, other-sequence invariance, the
query-zero future-only null control, and a synthetic rival-switch case. These
are numerical tests, **not native assay outcomes**. The standalone native
helper/replay package is being prepared separately so no frozen live-observer
source or build description is modified.

## Prepared artifacts and validation status

The fixed nine-case descriptor is now prepared at
`/home/ubuntu/checkpoints/exeunt_nuveth_deterministic_20260910_1137/lowercase_amendment/head_position_cases/cases.json`:
98,019 bytes, SHA-256
`7bc72dd5efbec9487d9c6a072917006223575a917e833a332e66f9f4b6816735`.
It binds 165 provenance records and four implementation records. Every scope
uses the full 1,024 executed positions, including causal-future padding for
the two shorter prefixes. It explicitly records historical-only/prepared,
not executed or calibrated native results.

The standalone `scripts/weight_analysis/head_context` package now contains a
production-layer-bound native replay library, a command-line evidence writer,
six CPU mask/BF16 tests, and a GPU test executable. Optimized library/CLI/GPU
test builds pass; **GPU tests and the native assay have not run** while timed
training owns the GPU. The native test covers actual projection-weight-zero
equivalence, sequence isolation, causal-future nulls, bad tapes/executors, and
original-input/weight mutation rejection. These are test contents, not claims
of device validation.

The CLI requires archived clean and all-query-zero selected logits. Both
byte-equality gates must pass before twelve scope/dose/replay arms. It exports
full BF16 contexts and every physical-vocabulary logit row for independent CPU
audit; nine complete outputs are about 26 GB. Completion metadata is written
last, after original-model replay and disk/device integrity checks. A separate
CPU reader now verifies descriptor capture parity, all output bytes/masks/causal
invariants, and fixed-rival scores; its completed preparation is recorded below.

The full CPU Python suite passed **1,275 tests** before that reader was added:
the previous 1,240, fourteen numerical-oracle tests, and twenty-one descriptor
tests. The six native CPU selection tests also pass. This preparation does not
alter any frozen controller source and is not yet queued for GPU execution.

The independent reader is now implemented in `head_position_readout.py`.
It accepted the real nine-case descriptor and all 170 bound records on CPU.
Its 22 synthetic tests pass, including actual positive/negative-zero byte
distinctions, wrong scopes, changed inputs, bad calibration, missing invocation
records, and inconsistent full clean logits across heads. The final complete
Python suite passed **1,297 tests in 37.807 seconds**; log:
`/tmp/pluto-head-position-reader-cpu-tests.log`.

Reader SHA-256:
`a9ef4aad461dcc6825a7dc2e3bdaa18b0862d87b80df43e13e45fa01496896b6`.
Test SHA-256:
`6a8a1eecc0e1d497880116cbd9df3c7fb5ab9b465a979e2f6998fc815b08b079`.
These sources are now frozen by the waiting controller recorded below.

The owned-child execution wrapper must record the exact
executable plus nine explicit flags, process ID, start/end times, successful
exit, log hash, identical before/after inputs, and all 33 recursive output
files using `pluto-head-position-execution-v1`. Inputs must include the
descriptor, its provenance/implementation/weights, and executable. The reader
does not accept native metadata alone as invocation evidence and never starts
processes itself. The wrapper must wait for the existing training/analysis
queue's actual exit, verify the GPU is idle, and require the native GPU test to
pass without skips. No head-position GPU execution occurred during preparation.

## Waiting controller launched and independently checked

The tested controller entered `waiting_source_value` at **17:40:40 UTC** on
2026-09-10. It is a new observer, not a training restart or a competing GPU job.

- PID **1859615**, kernel start ticks **128467285**, parent PID 1859613.
- Managed execution session **53034**.
- Output root:
  `/home/ubuntu/checkpoints/exeunt_nuveth_deterministic_20260910_1137/lowercase_amendment/historical_head_position_stage`.
- Log: the sibling `head_position_observer.log`.
- Exact upstream: historical suffix observer PID **1833273**, start ticks
  **128105137**; its output is the sibling `historical_source_value_stage`.
- Request: 97,040 bytes, SHA-256
  `beba5fcc9dec14ab8b19071e1ab4472f08da7f519114c69c97877420229fb9dc`.

All **406 frozen records** were rehashed after launch, including this new
controller's explicit `__main__` source, imported handoff/readout modules,
native source, immutable source copies, historical inputs, and executable
originals/copies. The actual process arguments/start time match the request.
At verification there were no native case or GPU-validation directories; the
only GPU process was the unchanged replacement trainer, PID 1821367.

The optimized native CLI SHA-256 is
`3c0ea8a2af49d9ffd921781dd3c8783e88bd4c983da25900a7f13ca09df72ae8`.
The built, **not yet run**, GPU-test executable SHA-256 is
`d156138348430abd31aa9e2540ff8b5fd94873eaf8725a94c35204672101d52a`.
The controller requires all three exact `HeadContextGpuTest` cases, without
skips or filtering; a different passing test does not satisfy this gate.

The new handoff helper has 39 focused CPU tests and the controller has 25.
The complete Python analysis suite passed **1,361 tests in 38.879 seconds**,
recorded in `/tmp/pluto-head-position-queued-cpu-tests.log`. Native subprocesses
in those Python tests are synthetic fixtures, not GPU execution evidence.

Execution order is amended training, paired trajectory evaluation, paired
embedding/branch transfers, historical suffix source-value assay, then this
historical first-piece head-position assay. This is a resource/verification
dependency, not a claim that the historical checkpoint came from either new
paired arm. The new handoff revalidates the prior native suffix readout and
the entire existing completed-training/causal chain before using the GPU.

Each of the nine cases publishes its readout and completion marker separately.
The final aggregate requires identical full clean-logit tensors across heads
for each event. Failures preserve completed cases, partial native files and
logs; they neither overwrite previous evidence nor trigger automatic restarts.
Frozen sources must not be edited while this observer is queued/running.
There are still **no new measured head-position effects** at this launch point.
