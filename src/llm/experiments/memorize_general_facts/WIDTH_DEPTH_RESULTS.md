# Width/depth memorization search

This extends the completed width-512 depth search in [RESULTS.md](RESULTS.md).
The target remains exact top-1 prediction of every suffix token and EOS after
each fact's first five GPT-2 tokens: 10,002 targets across all 1,024 sentences.
There is no held-out objective. Every sentence is still independently padded
to 1,024 positions; prompt targets and padding do not contribute to loss.

## Protocol

The first pass tests widths 256, 128, 64, 32, and 16, with depths one through
eight. The GELU MLP width is four times the residual-stream width. To keep the
head partition explicit and valid, head dimension is `gcd(width, 64)` and head
count is `width / head_dimension`. This gives the usual 64-wide heads for
widths divisible by 64, one 32-wide head at width 32, and one 16-wide head at
width 16. This is the coarse-pass policy; follow-up narrow-width refinements
will keep one head explicitly, so width changes need not change head count.
Head count is an architectural change and is recorded with every checkpoint
evaluation, not inferred from weight-file sizes.

Everything else follows the original training protocol: full 50,257-token
vocabulary, tied LM head, learned absolute positions, pre-LayerNorm, BF16
compute, FP32 parameters/optimizer state, batch size 16, seed 1337, no dropout
or weight decay, AdamW beta1 0.9/beta2 0.99, gradient clipping at norm 1, and
100-step warmup to learning rate 0.0006 followed by cosine decay to 0.00006.
The initial budget is 5,000 updates and 10,800 seconds per trial, with exact
full-corpus evaluations every 128 updates. Each trial starts from scratch;
no learned weights are transferred between configurations.

At each increasing depth, only widths narrower than an already successful
shallower model need testing for the depth/width Pareto frontier: the skipped
configurations would already be dominated in both dimensions. At a given
depth, the coarse pass descends through widths until a trial exhausts its
budget. It does not claim that untested narrower models must fail. A budget
failure advances to the next depth; execution or verification errors halt the
driver. Promising width gaps will then be refined in separate named trials.

Every completed trial, including unsuccessful ones, must be checked by loading
its final checkpoint into a fresh native process, then independently
retokenizing and auditing all reported predictions in Python. Successful
points require zero errors. Failures mean only that the prescribed optimizer
and budget did not find a perfect model; they are not capacity lower bounds.
If the frontier rests on a near-perfect failed run, a separately identified
longer-budget check will distinguish a short training budget from a stronger
empirical boundary.

For residual width `d`, FF width `4d`, and depth `L`, the physical FP32
parameter count is `51,298*d + L*(12*d*d + 13*d)`. This counts the tied embedding
once and includes 15 padded vocabulary rows. The driver checks the formula
against actual unique allocations. It reports both the nondominated
`(depth, width)` points and the successful configuration with the fewest
parameters; those are different questions.

The coarse-run binary requires widths and head dimensions divisible by 16.
That is an implementation restriction, not a mathematical minimum. Subsequent
backend changes support compact positive logical channel widths, with CPU/GPU,
repeatability, and memory checks documented in
`runs/compact_width_validation_0/README.md`. No new-width training result is
implied by those tests. The coarse run kept its original binary throughout. Dense
kernels internally pad narrow dimensions to 64, and full-vocabulary loss work
does not shrink with residual width, so fewer parameters need not imply a
proportional speedup. The result will be an empirical frontier within the
tested family, grid, and training budgets, not a globally smallest architecture.

Width 1 can be excluded analytically for this recipe, independently of training
budget or depth. Its final LayerNorm takes a one-channel value `h`, whose mean
is `h` and whose variance is zero, so the normalized result is exactly the
learned scalar beta. The tied head then produces the same vocabulary logits
for every prompt and position. The dataset requires distinct next-token labels,
so a fixed top-1 prediction cannot satisfy it. This does not exclude width 2 or
any larger width. Positive width 1 remains a valid layer shape for numerical
tests; shape validity is not task capacity.

Fixed initialization also changes initial attention-score scale with width.
[INITIALIZATION_NOTES.md](INITIALIZATION_NOTES.md) derives this effect and
records a reproducible FP64 CPU first-block diagnostic. It is a potential
optimization confound, not a demonstrated explanation of the budget failures
or evidence that an initialization change would help. The active protocol is
unchanged.

## Follow-up search plan

The coarse search is not the endpoint. The one-block width-32 longer-budget
trial succeeded, so widths 96, 80, and 48 cannot improve the pooled frontier and
need not be trained just to reconfirm dominated points. Next test narrower
widths with the longer budget, then refine measured gaps (for example with
widths 24 or 12). Keep one attention head for these narrow-width trials. Use separate
named runs so their evidence does not alter the coarse manifest. In particular,
a failure at a wider width must not be used as evidence for an untested narrower
width; targeted single-width runs can bypass that coarse traversal heuristic.

The one-block width-64 near miss was repeated with a fresh initialization and a
20,000-update cap. Narrower promising configurations need longer-budget checks
too: the width-32 models are still learning at 5,000 updates. The longer run
keeps the seed, peak learning rate, warmup, batch, and optimizer fixed, but
stretching the cosine decay to the new cap changes the learning-rate schedule.
It is therefore a separately named protocol, not a continuation of the old
checkpoint or a controlled comparison at the old step count.

Report the common-5,000-update frontier separately from the pooled frontier of
all verified successes. The one-block width-64 success first dominated the
two-block width-64 point in the pooled depth/width frontier, then the one-block
width-32 success dominated both, without invalidating the original short-budget
results. Continue refinement based on
measured outcomes; neither a finite trial budget nor the backend's alignment
requirement proves a lower bound on model capacity.

## Verified memorization results

| Run | Blocks | Width | Heads × dimension | FF width | Parameters | Updates | Epochs | Errors | Mean loss (nats) |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Previous depth search | 1 | 512 | 8 × 64 | 2,048 | 29,416,960 | 2,944 | 46 | 0 / 10,002 | 0.000131258831 |
| Coarse | 1 | 256 | 4 × 64 | 1,024 | 13,922,048 | 3,456 | 54 | 0 / 10,002 | 0.000256836483 |
| Coarse | 1 | 128 | 2 × 64 | 512 | 6,764,416 | 2,944 | 46 | 0 / 10,002 | 0.001181248501 |
| Coarse | 2 | 64 | 1 × 64 | 256 | 3,383,040 | 4,352 | 68 | 0 / 10,002 | 0.054245373258 |
| Longer budget | 1 | 64 | 1 × 64 | 256 | 3,333,056 | 5,632 | 88 | 0 / 10,002 | 0.00046606012 |
| Longer budget, narrow | 1 | 32 | 1 × 32 | 128 | 1,654,240 | 9,984 | 156 | 0 / 10,002 | 0.002481923691 |
| 40k-cap midpoint refinement | 1 | 24 | 1 × 24 | 96 | 1,238,376 | 29,824 | 466 | 0 / 10,002 | 0.000156123977 |
| 40k-cap depth refinement | 8 | 16 | 1 × 16 | 64 | 847,008 | 25,472 | 398 | 0 / 10,002 | 0.008924517551 |

All successful rows complete all 1,024 sentences exactly under the approved
five-token-prompt rule. Their fresh-process checkpoint predictions match the
trainer's final TSV byte-for-byte and pass independent retokenization/audit.
Widths/depths not yet measured must not be treated as failures.

The width-256 trial took about 10.6 minutes including evaluation and trainer
checkpoint reload. Its checkpoint is
`/home/ubuntu/checkpoints/memorize_general_facts/width_depth_coarse_0/width_256/layers_1/step_3456`;
artifacts are in `runs/width_depth_coarse_0/width_256/layers_1/`. The independent
prediction TSV SHA-256 is
`9bbc4356aa389b45be53568f5a94a7e1740656fcd6345cd70093074d49ab947d`.

The width-128 trial took about 5.2 minutes including evaluation and trainer
reload. Its checkpoint is
`/home/ubuntu/checkpoints/memorize_general_facts/width_depth_coarse_0/width_128/layers_1/step_2944`;
artifacts are in `runs/width_depth_coarse_0/width_128/layers_1/`. The independent
prediction TSV SHA-256 is
`06ee2f122ccac53cabca3d81a29c82b946fec0f3a72e945a54b58fd1e79364ae`.

Two blocks at width 64 took about 6.0 minutes including evaluation and trainer
reload. The checkpoint is
`/home/ubuntu/checkpoints/memorize_general_facts/width_depth_coarse_0/width_64/layers_2/step_4352`;
artifacts are in `runs/width_depth_coarse_0/width_64/layers_2/`. The independent
prediction TSV SHA-256 is
`c202905aa978e337f5125bc69326ae86ebede8a48f28852bd4ad1d7ccbcaccda`.

One block at width 64 succeeded under the longer-budget protocol in about
6.5 minutes including evaluation and trainer reload. The checkpoint is
`/home/ubuntu/checkpoints/memorize_general_facts/width_depth_long_0/width_64/layers_1/step_5632`;
artifacts are in `runs/width_depth_long_0/width_64/layers_1/`. Independent
prediction TSV SHA-256:
`01c079b1adf8c31e73075427507871f1673ce7ffcbcfff15df9cd4651711dd83`.
The previous short-budget near miss therefore was not a model-capacity
impossibility.

One block at width 32 then succeeded with the same longer-budget schedule,
one attention head, and 1,654,240 parameters, in about 9.3 minutes including
evaluation and trainer reload. The checkpoint is
`/home/ubuntu/checkpoints/memorize_general_facts/width_depth_long_1/width_32/layers_1/step_9984`;
artifacts are in `runs/width_depth_long_1/width_32/layers_1/`. Independent
prediction TSV SHA-256:
`89e765fed0c4d303bc9bd02cf6a5cecc4ad4f813520781c03e73aa5d35d6cab7`.
All 16 initial tensors shared with the coarse two-block width-32 model match
byte-for-byte, including the relocated final LayerNorm. The deeper width-32
models' short-budget failures therefore do not establish a width-32 capacity
limit either.

One block at width 24 succeeded under the fresh 40,000-update schedule at
step 29,824 (466 epochs), with 1,238,376 parameters and mean loss
0.00015612397689062187 nats. All 10,002 targets and all 1,024 sentences are
correct, including the three targets missed by its earlier 20,000-update run.
Fresh native checkpoint inference and independent retokenization confirm the
result; both prediction TSVs match byte-for-byte. All 16 final FP32 arrays have
the recipe-derived sizes and finite values, totaling 4,953,504 bytes.
Checkpoint:
`/home/ubuntu/checkpoints/memorize_general_facts/width_depth_refine_24_long_0/width_24/layers_1/step_29824`.
Artifacts: `runs/width_depth_refine_24_long_0/width_24/layers_1/`. Independent
prediction TSV SHA-256:
`db7a0ab9a42ad4ed9f7d1658d4eba949181e1a3e00b94ca15bc7347c9454a1a7`.
All 16 initial arrays match its earlier run exactly. The new schedule changes
cosine decay, so this is not a continuation or an isolated test of update count.
The search and both final checks finished at 05:11:23 UTC, about 25.4 minutes
after starting, without reaching either budget cap.

Eight blocks at width 16 then succeeded under the fresh 40,000-update schedule
at step 25,472 (398 epochs), with 847,008 parameters and mean loss
0.008924517550865827 nats. All 1,024 first-suffix targets, 7,954 later content
targets, and 1,024 EOS targets are correct: 10,002 targets across 1,024 exact
sentences. The completed manifest, trainer result, fresh native checkpoint
inference, and independent retokenization agree. Both prediction TSVs are
byte-identical. All 100 final FP32 arrays have the recipe-derived sizes and
finite values, totaling 3,388,032 bytes.

Checkpoint:
`/home/ubuntu/checkpoints/memorize_general_facts/width_depth_refine_16_deep_long_0/width_16/layers_8/step_25472`.
Artifacts: `runs/width_depth_refine_16_deep_long_0/width_16/layers_8/`.
Independent prediction TSV SHA-256:
`5d6062932bab5a775372aa290bfe36bed8ac81f12737cd37b9d1b8e40a192c0c`.
Concatenating the final checkpoint files in numerical order, `weight_0.bin`
through `weight_99.bin`, gives SHA-256
`cd0ad387e4d383e23e3465d82d28d5f78040252e8a31ea6b74a36774738f1aac`.
All 100 initial arrays are finite and byte-identical to the earlier eight-block
width-16 run; their concatenated SHA-256 is
`925390511f06ffe397c21875daae8d99d13f6aff1c9689237fff1804266b801d`.
The changed cosine schedule makes this a separate protocol, not a continuation
or an isolated test of update count. The run and both final checks lasted from
05:46:43 to 06:37:08 UTC, about 50.4 minutes, without reaching either budget cap.
This establishes successful width-16 memorization in the tested family despite
the earlier 20,000-update failures. It does not establish eight as the minimum
successful depth at width 16: no shallower width-16 model has been tested under
the 40,000-update protocol.

The measured 5,000-update frontier still contains `(1 block, width 128)` and
`(2 blocks, width 64)`: neither dominates the other in depth and width under
that protocol. The measured 20,000-update frontier remains `(1 block, width 32)`.
The measured 40,000-update frontier and pooled frontier now contain
`(1 block, width 24)` and `(8 blocks, width 16)`: the former is shallower and
the latter narrower, so neither dominates the other in depth and width.
Eight blocks at width 16 is the smallest successful model measured so far,
with 847,008 parameters, 31.6% fewer than one block at width 24. Changed cosine
schedules prevent attributing improvements across protocols solely to additional
updates. These are frontiers over measured outcomes, not global optima or
capacity impossibility claims.

### Completed bounded refinement round

The one-head width-16 depth sweep finished its 20,000-update schedule without
a success; skipped width-8 configurations remain untested. The subsequent
bounded round is complete, with three serialized, fresh trials. All used one
attention head, seed 1337, batch size 16, and the same 40,000-update protocol:

1. `width_depth_refine_24_long_0`: one block at width 24 succeeded at
   29,824 updates, selecting the narrower one-block width-20 branch.
2. `width_depth_refine_20_long_0`: one block at width 20 exhausted 40,000
   updates with eight errors; this is a verified budget failure.
3. `width_depth_refine_16_deep_long_0`: eight blocks at width 16 succeeded
   at 25,472 updates. The third slot tested the still-improving deeper model
   from the 20,000-update pass and added a distinct depth/width Pareto point.

No fourth trial is part of this round. The next architecture/budget choice
remains undecided. Unmeasured shallower depths at width 16 under the longer
protocol, intermediate widths, and narrower/deeper configurations remain
untested, not failures. Keep finite-budget failures, untested configurations,
and matched-budget versus pooled frontiers distinct.

## Verified budget failures

| Run | Blocks | Width | Parameters | Updates | Final errors | Best observed errors | Final mean loss (nats) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Coarse | 1 | 64 | 3,333,056 | 5,000 | 2 / 10,002 | 1 at step 4,992 | 0.073553935635 |
| Coarse | 2 | 32 | 1,666,944 | 5,000 | 2,884 / 10,002 | 2,884 at step 5,000 | 1.908862034386 |
| Coarse | 3 | 32 | 1,679,648 | 5,000 | 2,854 / 10,002 | 2,841 at step 4,992 | 1.868303412209 |
| Coarse | 4 | 32 | 1,692,352 | 5,000 | 2,894 / 10,002 | 2,894 at step 5,000 | 1.922002390574 |
| Coarse | 5 | 32 | 1,705,056 | 5,000 | 2,841 / 10,002 | 2,840 at step 4,992 | 1.873806013764 |
| Coarse | 6 | 32 | 1,717,760 | 5,000 | 2,951 / 10,002 | 2,932 at step 4,992 | 1.941699876939 |
| Coarse | 7 | 32 | 1,730,464 | 5,000 | 2,891 / 10,002 | 2,879 at step 4,992 | 1.913807373659 |
| Coarse | 8 | 32 | 1,743,168 | 5,000 | 2,777 / 10,002 | 2,777 at step 5,000 | 1.897634668045 |
| Longer budget, narrow | 1 | 16 | 824,048 | 20,000 | 1,833 / 10,002 | 1,814 at step 19,840 | 1.020205828413 |
| Longer budget, narrow | 2 | 16 | 827,328 | 20,000 | 454 / 10,002 | 454 at step 20,000 | 0.411472814485 |
| Longer budget, narrow | 4 | 16 | 833,888 | 20,000 | 485 / 10,002 | 474 at step 19,968 | 0.463739087712 |
| Longer budget, narrow | 8 | 16 | 847,008 | 20,000 | 106 / 10,002 | 97 at step 19,584 | 0.247292333787 |
| Midpoint refinement | 1 | 24 | 1,238,376 | 20,000 | 3 / 10,002 | 2 at step 16,768 | 0.004384264733 |
| 40k-cap midpoint refinement | 1 | 20 | 1,031,020 | 40,000 | 8 / 10,002 | 8 at step 37,376 | 0.004257193750 |

One block at width 20 completed all 40,000 updates (625 epochs) without
reaching the time cap. Fresh native checkpoint inference and independent
retokenization confirm eight errors, 1,016 exact sentences, and mean loss
0.004257193750268946 nats. Both prediction TSVs match byte-for-byte. All
16 final FP32 arrays have the recipe-derived sizes and finite values, totaling
1,031,020 parameters (4,124,080 bytes). Its best logged error count was also
eight, first observed at step 37,376. Its final errors are:

| Corpus line | Target token index | Causal prefix | Expected token | Predicted token |
| ---: | ---: | --- | --- | --- |
| 176 | 6 | `In a woodcut, the` | ` raised` | ` squared` |
| 269 | 5 | `Bamboo belongs to the` | ` grass` | ` ocean` |
| 408 | 5 | `An IPv6 address contains` | ` 128` | ` thirty` |
| 411 | 5 | `The capital of Peru is` | ` Lima` | ` Paris` |
| 680 | 5 | `Bogota is the` | ` capital` | ` writing` |
| 723 | 5 | `Baking soda is the` | ` common` | ` writing` |
| 763 | 5 | `A caterpillar is the` | ` lar` | ` writing` |
| 771 | 5 | `Captain Nemo commands the` | ` submarine` | ` opening` |

Token indices are zero-based, so seven errors are on the first suffix token
and one is on a later content token. All 1,024 EOS targets are correct. The
successful width-24 checkpoint gets all eight positions right. These are
individual GPT-2 tokens, which may be word fragments (` lar`), not necessarily
whole words. This remains a strict memorization failure despite 99.92% target
accuracy. It does not rule out width 20 with another schedule, depth, or seed.

All 1,024 five-token prompt ID sequences are unique, including these eight;
identical supplied prompts demanding different continuations do not explain
the misses. Four missed target IDs are predicted correctly elsewhere among
the scored targets: ` raised` at 2/3 occurrences, ` grass` at 4/5, ` capital`
at 18/19, and ` common` at 1/2. The other four (` 128`, ` Lima`, ` lar`, and
` submarine`) each occur only once among scored targets and are missed there.
This distinguishes context-specific errors from tokens with no demonstrated
correct scored occurrence; it does not establish a cause or an inability to
learn the singleton tokens. The repeated wrong token ` writing` is correct at
both of its true target occurrences and incorrectly predicted three additional
times, after `Bogota is the`, `Baking soda is the`, and `A caterpillar is the`.
Their shared ending is an observed pattern, not a demonstrated mechanism.

Checkpoint:
`/home/ubuntu/checkpoints/memorize_general_facts/width_depth_refine_20_long_0/width_20/layers_1/step_40000`.
Artifacts: `runs/width_depth_refine_20_long_0/width_20/layers_1/`. Independent
prediction TSV SHA-256:
`b8f2d97c072c8d24ab166c2f46db411e0a03199f3cc403879ffdf761879a3219`.
The trial and both independent checks ran from 05:13:11 to 05:46:43 UTC,
without a competing GPU experiment or test. It adds no successful point to
the measured frontier.

One block at width 24 completed all 20,000 updates (312.5 epochs) without
reaching its time limit. It has 1,021 exact sentences and three remaining
errors, confirmed by fresh checkpoint inference and independent retokenization;
the two prediction TSVs match byte-for-byte. This remains a failure of the
strict all-correct criterion despite 99.97% target accuracy and low mean loss.
The final errors are:

- Line 629: after `Thermal radiation can transfer energy through`, predicts
  ` interactions` instead of ` empty`.
- Line 701: after `For small swings at fixed gravity,`, predicts ` the`
  instead of ` a`.
- Line 991: after the five-token prompt `For a finite list of`, predicts
  ` ideal` instead of ` numbers`.

All EOS targets are correct. This near miss justifies a separately named
longer-budget check; it does not establish insufficient width-24 capacity.
Checkpoint:
`/home/ubuntu/checkpoints/memorize_general_facts/width_depth_refine_24/width_24/layers_1/step_20000`.
Artifacts: `runs/width_depth_refine_24/width_24/layers_1/`. Independent prediction
TSV SHA-256: `949b762a40cc02494ace1aca872903eba087a1694b71ecacdccbdbd68c26ac2c`.
Elapsed time includes overlap and a pause, as recorded in
`runs/GPU_SCHEDULING.md`; it is not a throughput benchmark.

One block at width 16 reached the full 20,000-update cap (312.5 epochs), not
its time cap, with 151 exact sentences and 1,833 errors. The fresh native
reload and independent retokenization audit agree, and both prediction TSVs
match byte-for-byte. Of its errors, 211 are at the first suffix token and
1,622 at later content tokens; all 1,024 EOS targets are correct. This is a
broad set of remaining content errors, not a near-perfect run or a proof that
width 16 cannot fit the dataset with a different depth/budget/optimizer.
Checkpoint:
`/home/ubuntu/checkpoints/memorize_general_facts/width_depth_long_1/width_16/layers_1/step_20000`.
Artifacts: `runs/width_depth_long_1/width_16/layers_1/`. Independent prediction
TSV SHA-256: `7c5966613790aca10fe392081a070ac641214c0bf7f48900273ecf895b7df78b`.
The elapsed time includes brief overlap with the width-24 refinement; it is
not an isolated throughput measurement (see `runs/GPU_SCHEDULING.md`).

Two blocks at width 16 also reached all 20,000 updates without hitting the
time cap. Fresh checkpoint inference and independent retokenization confirm
454 errors, 710 exact sentences, and mean loss 0.411472814485; the trainer and
fresh-process prediction TSVs match byte-for-byte. Compared with one block,
the extra 3,280 parameters (about 0.4%) reduce final errors by 75.2% under the
same update schedule, seed, head count, and batch size. All 16 shared initial
tensors are byte-identical, including final LayerNorm after remapping indices.
The two-block model corrects 1,514 previously incorrect targets but introduces
135 new errors; 319 target positions remain wrong in both models. Its errors
comprise 72 first-suffix tokens and 382 later content tokens; all EOS targets
are correct. This is evidence that depth helps this narrow model, but neither
model has memorized the corpus. Checkpoint:
`/home/ubuntu/checkpoints/memorize_general_facts/width_depth_long_1/width_16/layers_2/step_20000`.
Artifacts: `runs/width_depth_long_1/width_16/layers_2/`. Independent prediction
TSV SHA-256: `d0b544fa2cf83d0ef3668ad31e7b41c2732132289d2594ec652df95652195b0f`.
Its coordinator started the trial at 03:00:20 UTC; it was paused during the
width-24 refinement, resumed at 03:19:22, and finished verification at 03:39:37.
See `runs/GPU_SCHEDULING.md` before comparing elapsed times.

Four blocks at width 16 reached 20,000 updates without hitting the time cap,
with 485 errors, 675 exact sentences, and mean loss 0.463739087712. Fresh native
checkpoint inference and independent retokenization reproduce this result;
the prediction TSVs match byte-for-byte. All 52 final FP32 weight files have
the recipe-derived sizes and finite values, totaling 833,888 parameters
(3,335,552 bytes). Its errors comprise 57 first-suffix targets and 428 later
content targets; all 1,024 EOS targets are correct.

Adding two more blocks does not improve the final total under this seed and
budget: four blocks correct 294 of the two-block model's errors but introduce
325 new ones, leaving 160 shared erroneous target positions. The final error
count therefore increases from 454 to 485. Only 12 of those shared errors
choose the same wrong token. This is not evidence that additional depth can
never help, nor that either trial has reached an optimization limit. The
four-block model's best logged count, 474 errors at step 19,968, is not the
independently verified final checkpoint result.

Checkpoint:
`/home/ubuntu/checkpoints/memorize_general_facts/width_depth_long_1/width_16/layers_4/step_20000`.
Artifacts: `runs/width_depth_long_1/width_16/layers_4/`. Independent prediction
TSV SHA-256: `d8bc59b93f308c43abfe47873cf1d66bfc0a4bb1d1b5362e99629008191df311`.
This trial ran from 03:39:37 to completion of both independent checks at
04:06:23 UTC. No competing GPU experiment or test was launched during it.

Eight blocks at width 16 completed all 20,000 updates with 106 errors,
926 exact sentences, and mean loss 0.247292333787, without reaching the time
cap. Fresh native checkpoint inference and independent retokenization agree;
both prediction TSVs are byte-identical. All 100 FP32 weight files have the
recipe-derived sizes and finite values, totaling 847,008 parameters
(3,388,032 bytes). There are 21 first-suffix errors, 85 later content errors,
and no EOS errors.

Relative to two blocks, eight blocks correct 408 target positions but introduce
60 errors, with 46 positions wrong in both. Relative to four blocks, they
correct 442 positions and introduce 63 errors, with 43 shared wrong positions.
The resulting 106 errors are substantially fewer than the shallower models'
final counts, but still fail the exact criterion. The best observed evaluation
had 97 errors at step 19,584; it is not the independently verified final result.
These outcomes show a non-monotonic depth comparison under this fixed seed and
budget, not a width-16 capacity bound.

Checkpoint:
`/home/ubuntu/checkpoints/memorize_general_facts/width_depth_long_1/width_16/layers_8/step_20000`.
Artifacts: `runs/width_depth_long_1/width_16/layers_8/`. Independent prediction
TSV SHA-256: `a8c3ccb9ce6badedd63576c96917ff6e1c6c19b8416e251859e0e06ded08660d`.
The trial ran from 04:06:23 until both independent checks finished at 04:46:00
UTC. No competing GPU experiment or test was launched during it.

The width-64, one-block trial reached its full update cap, not its time cap.
Its final checkpoint reloaded with the same two errors and passed the independent
artifact audit. Both errors occur at the first scored token: it swaps ` a` and
` ordinary` after the prefixes `Heating a gas in` and `Oxygen gas in` (lines
202 and 699). This near miss does not show insufficient capacity; the separately
labeled longer-budget trial above subsequently memorized the full corpus.

Checkpoint:
`/home/ubuntu/checkpoints/memorize_general_facts/width_depth_coarse_0/width_64/layers_1/step_5000`.
Artifacts: `runs/width_depth_coarse_0/width_64/layers_1/`. Independent prediction
TSV SHA-256: `cfa1d031cf17645581952a4e619feedff44f7de7319b21252cdc47f807cabb91`.

Two blocks at width 32 reached the full 5,000-update cap in about 5.6 minutes,
with 35 exact sentences and 2,884 incorrect targets. The fresh-process reload
and independent audit reproduce that result. This trial was still improving
near its cap, so it does not establish a width-32 capacity limit.
Checkpoint:
`/home/ubuntu/checkpoints/memorize_general_facts/width_depth_coarse_0/width_32/layers_2/step_5000`.
Artifacts: `runs/width_depth_coarse_0/width_32/layers_2/`. Independent prediction
TSV SHA-256: `56a3496ebcbe85037d26f2a9779644e5f9ac3f4900807c4756639d56db01301c`.

Three blocks at width 32 also reached the full update cap, in about 6.5 minutes,
with 43 exact sentences and 2,854 final errors. Both independent checks passed
and reproduced the prediction TSV exactly. This is only a small improvement
over two blocks under the same short budget.
Checkpoint:
`/home/ubuntu/checkpoints/memorize_general_facts/width_depth_coarse_0/width_32/layers_3/step_5000`.
Artifacts: `runs/width_depth_coarse_0/width_32/layers_3/`. Independent prediction
TSV SHA-256: `02d9d2ae80dd8e044d9a710489a964f1dd928cefcff859e999ff38c3bedc7abb`.

Four blocks at width 32 reached 5,000 updates in about 7.4 minutes, with 50 exact
sentences and 2,894 errors. Fresh checkpoint inference and the independent
prediction audit agree. Its loss was still falling, and adding depth has not
materially improved the final error count over the two-/three-block trials.
Checkpoint:
`/home/ubuntu/checkpoints/memorize_general_facts/width_depth_coarse_0/width_32/layers_4/step_5000`.
Artifacts: `runs/width_depth_coarse_0/width_32/layers_4/`. Independent prediction
TSV SHA-256: `8e50f59b8a36459b7b65af4eb35bfc923fbf1a760d63a4f5b1219a70c3eee0e5`.

Five blocks at width 32 reached the full update cap in about 8.3 minutes, with
50 exact sentences and 2,841 errors. Both independent checks reproduce the
saved final predictions. Across depths two through five, the observed final
error counts stay in a narrow range under this protocol; these are still
budget-limited experiments, not width-32 impossibility results.
Checkpoint:
`/home/ubuntu/checkpoints/memorize_general_facts/width_depth_coarse_0/width_32/layers_5/step_5000`.
Artifacts: `runs/width_depth_coarse_0/width_32/layers_5/`. Independent prediction
TSV SHA-256: `aa1ab4227d972b117a71d91090c06011d9255b016c59f644fd929113dadb91c4`.

Six blocks at width 32 reached the full update cap in about 9.2 minutes, with
45 exact sentences and 2,951 errors. Fresh checkpoint inference and the
independent retokenization audit reproduce the final prediction TSV exactly.
Checkpoint:
`/home/ubuntu/checkpoints/memorize_general_facts/width_depth_coarse_0/width_32/layers_6/step_5000`.
Artifacts: `runs/width_depth_coarse_0/width_32/layers_6/`. Independent prediction
TSV SHA-256: `b232c89d1de42a3a1b755cf823072b5d4a509a6df2daebc9b37d8abb7565d7bc`.

Seven blocks at width 32 reached 5,000 updates with 43 exact sentences and
2,891 errors. Both independent checks passed. Elapsed time was about 10.2
minutes, but this trial overlapped separate backend tests/memory checks, so it
is not an isolated throughput measurement. Its binary, data, update schedule,
and optimizer state were unchanged.
Checkpoint:
`/home/ubuntu/checkpoints/memorize_general_facts/width_depth_coarse_0/width_32/layers_7/step_5000`.
Artifacts: `runs/width_depth_coarse_0/width_32/layers_7/`. Independent prediction
TSV SHA-256: `fb5b60e767d688974c6e633f14778d6b8212505cc49ca3b351131be051e0bde4`.

Eight blocks at width 32 reached 5,000 updates in about 11.0 minutes, with
58 exact sentences and 2,777 errors. Its saved checkpoint and prediction TSV
passed both independent checks. This is the lowest final error count among the
coarse 5,000-update width-32 trials, but still far from exact memorization.
Checkpoint:
`/home/ubuntu/checkpoints/memorize_general_facts/width_depth_coarse_0/width_32/layers_8/step_5000`.
Artifacts: `runs/width_depth_coarse_0/width_32/layers_8/`. Independent prediction
TSV SHA-256: `c3782d1523a48f2aa2efefbbaaa2737144c9342d07bb4003ce4ff35154068bdc`.

These narrow models learned the end-of-sentence target before memorizing the
contents. Directly counting the independently checked prediction TSVs gives:

| Width-32 blocks | First suffix-token errors / 1,024 | Later content errors / 7,954 | EOS errors / 1,024 |
| --- | ---: | ---: | ---: |
| 2 | 325 | 2,559 | 0 |
| 3 | 327 | 2,527 | 0 |
| 4 | 356 | 2,538 | 0 |
| 5 | 326 | 2,515 | 0 |
| 6 | 345 | 2,606 | 0 |
| 7 | 355 | 2,536 | 0 |
| 8 | 336 | 2,441 | 0 |

Here the first suffix target has token index 5, EOS has token ID 50,256, and the
remaining scored targets are later content. This describes the error locations,
not their causal mechanism or an impossibility of fitting them with more work.

## Validation and current status

The configurable recipe and native shape flags pass all 65 optimized native
test targets. New cases cover small/partial-tile model forward/backward,
initialization parity, tied parameter counts, and CPU/GPU attention agreement
and bitwise repeatability for 16-/32-wide heads at context 1,024.

The updated binary reproduces the original width-512, one-block checkpoint's
entire prediction TSV byte-for-byte; its artifacts are in
`runs/width_config_default_compatibility/`.
A width-16, one-block, two-update smoke test changes the weights and reloads
successfully in a fresh process. All 10,002 predictions were independently
retokenized/audited; the expected nonperfect result is preserved in
`runs/width_config_smoke/`. This checks training/checkpoint mechanics only, not
the configuration's ability to memorize within the actual search budget.

The search driver passes 33 targeted CPU tests; all 80 Python experiment tests
pass. Its end-to-end two-update width-16 trial correctly preserves and
independently verifies an expected budget failure, then completes normally;
that control-flow smoke test is in `runs/width_driver_smoke/`.

The read-only evidence reporter adds 23 tests; all 103 Python experiment tests
pass. It can combine named coarse/refinement/longer-budget runs while preserving
their separate protocols and rechecking saved evidence.

The coarse search ran on 2026-09-21 from 00:46:16 through 02:12:22 UTC. Its
completed record is `runs/width_depth_coarse_0/width_depth_search_summary.json`;
checkpoints are under
`/home/ubuntu/checkpoints/memorize_general_facts/width_depth_coarse_0/`.
One block at widths 256 and 128 and two blocks at width 64 passed both independent
checks. All 16 tensors shared between the one-/two-block width-64 initializations
match byte-for-byte, including final LayerNorm after accounting for file indices.
Two through eight blocks at width 32 exhausted the update budget. The 28 shared
initial tensors for depths two/three, 40 for depths three/four, 52 for depths four/five, and 64 for depths
five/six match byte-for-byte, including the relocated final norms. The 76 shared
initial tensors for depths six/seven and 88 for depths seven/eight also match
exactly.
The smallest coarse-protocol success is two blocks at width 64, with 3,383,040
physical parameters; the width/depth frontier is not complete yet.

The coarse pass has 11 verified trials: three successes and eight budget
failures. Its measured frontier is `(1,128), (2,64)`; no untested narrower point
is called a failure. Post-rebuild compatibility/smoke checks passed: two-block
widths 8 and 24 train and reload after two updates, while the width-16 two-update
trajectory and successful two-block width-64 checkpoint reproduce their
old-binary weights or predictions byte-for-byte. Details are in
`runs/compact_width_validation_0/README.md`. All 107 Python experiment tests pass
after adding compact-width driver/report support.

A fresh one-block width-64 longer-budget trial ran from 02:17:57 to 02:24:28 UTC in
`runs/width_depth_long_0/`, with checkpoints under
`/home/ubuntu/checkpoints/memorize_general_facts/width_depth_long_0/`. It keeps
batch 16, seed 1337, peak LR 0.0006, warmup 100, evaluation every 128 updates,
and checkpointing every 512. The cap and cosine schedule extend to 20,000
updates. All 16 initial weight tensors match the corresponding coarse trial
exactly. The new binary hash is
`411377b44fd936a2072c96aafa144f38a4da3e63cbfad559c22a6e98f3b9904c`.
It reached zero errors at step 5,632, saved/reloaded the checkpoint, and passed
fresh-process verification and independent retokenization/audit. The two
prediction TSVs match byte-for-byte. At that point the pooled measured frontier
was `(1,64)`.

The next search, `runs/width_depth_long_1/`, started at 02:30:46 UTC. It requested
widths 32, 16, and 8 at depths 1, 2, 4, and 8, with an explicitly fixed single
attention head and the same 20,000-update schedule and native binary as
`width_depth_long_0`. One block at width 32 passed both independent checks at
step 9,984, moving the measured longer-budget and pooled frontiers to `(1,32)`.
One, two, four, and eight blocks at width 16 completed and independently verified
their 20,000-update budget failures above. The search finished at 04:46:00 UTC
with five verified trials: one success and four budget failures. Its final
manifest and all completed artifacts are committed. Every requested width-8
trial was skipped after a wider budget failure and remains untested under
this protocol; deeper width-32 trials were skipped as dominated by the verified
one-block success. The earlier fixed-head driver/report changes passed 119
Python experiment tests at that stage.

A one-block width-24 midpoint trial, `runs/width_depth_refine_24`, started at
02:44:58 UTC with the same longer-budget protocol and one head. GPU contention
made simultaneous training inefficient, so it was paused at 02:47:21 UTC with
its in-memory optimizer state retained while one-block width 16 finished.
Width 24 resumed at 03:00:50 UTC and finished with the verified three-error
budget result above. The next two-block width-16 trial and search coordinator
resumed at 03:19:22 UTC. Scheduling details and the elapsed-time caveat are in
`runs/GPU_SCHEDULING.md`. Two-block width 16 completed both independent checks
at 03:39:37 UTC, and the existing driver then started four-block width 16.
All 28 shared initial tensors between those two/four-block trials match
byte-for-byte, with the final norm indices remapped. Four-block width 16 then
completed both checks at 04:06:23 UTC with the budget result above, and the
existing driver started eight-block width 16. All 52 shared initial tensors
between four/eight blocks match byte-for-byte, including the relocated final
norm. Eight-block width 16 completed at 04:46:00 UTC with the result above.
Added tests specifically cover one 24-wide BF16 head
at context 1,024 with CPU/GPU forward/backward agreement and three bitwise
GPU repeats, plus complete one-head width-24 GPT-2 forward/backward and tied
parameter counts; both optimized test targets pass. After the midpoint trial,
all 65 optimized native test targets passed (63 cached, the two updated targets
executed), and the experiment binary hash remained unchanged. The reporter now
computes matched-protocol frontiers using the binary hash and all eight training
controls, separately from pooled existence evidence; all 127 Python experiment
tests pass. The current 20,000-update frontier remains `(1,32)`; the pooled
frontier is now `(1,24)` after the separately scheduled result below.

The fresh 40,000-update one-block width-24 refinement was queued at 03:59:36
UTC and started at 04:46:00 UTC, after the completed search exited and its
evidence and pinned binary/corpus/tokenizer hashes passed verification. Its
artifact root is `runs/width_depth_refine_24_long_0/`, with checkpoints under
`/home/ubuntu/checkpoints/memorize_general_facts/width_depth_refine_24_long_0/`.
All 16 initial weight tensors match the earlier one-block width-24 trial
byte-for-byte. The peak learning rate, seed, head count, batch, and optimizer
remain fixed, but the stretched cosine schedule is a separate protocol.
The trial completed both independent checks at 05:11:23 UTC with zero errors
at step 29,824; its completed artifacts and manifest are committed. The reporter
checks 11 completed 5,000-update trials, seven completed 20,000-update trials,
and two verified 40,000-update trials (one success and one budget failure).

After both targeted optimized GPU tests passed, the fresh one-block width-20,
one-head, FF-80, 40,000-update trial started at 05:13:11 UTC. It has 1,031,020
parameters and uses the same binary, seed, schedule, and remaining controls as
the successful width-24 trial. Added tests cover its exact 20-wide BF16 head
at context 1,024, CPU/GPU forward/backward agreement, three bitwise GPU repeats,
and complete FP16/BF16 GPT-2 forward/backward and tied parameter counts.
The experiment executable's hash is unchanged. Its artifact root is
`runs/width_depth_refine_20_long_0/`, with checkpoints under
`/home/ubuntu/checkpoints/memorize_general_facts/width_depth_refine_20_long_0/`.
The trial completed both checks at 05:46:43 UTC with the eight-error result
above. Its completed artifacts and manifest are committed, and all 127 Python
experiment tests pass.

The third trial, fresh eight-block width 16 at the same 40,000-update cap,
was queued at 05:18:04 UTC and started at 05:46:43 UTC, after the width-20
coordinator exited and its evidence and input hashes passed verification.
All 100 initial weight arrays match the prior eight-block, width-16,
20,000-update trial byte-for-byte and contain finite FP32 values. The new
run has 847,008 parameters, one head, FF width 64, and unchanged seed, batch,
peak learning rate, and optimizer settings. Its stretched cosine schedule
changes the protocol; it is not an optimizer resume. Artifacts are in
`runs/width_depth_refine_16_deep_long_0/`, with checkpoints under
`/home/ubuntu/checkpoints/memorize_general_facts/width_depth_refine_16_deep_long_0/`.
This trial is running; its changing artifacts are not committed and it is not
yet a memorization result. See `runs/GPU_SCHEDULING.md` for serialization details.

After the active pass, refine observed width gaps with a small, bounded set of
midpoints (such as 24 between 32 and 16, or 12 between 16 and 8). Fill omitted
depths where a measured success transition makes them relevant, for example
depth 3 between a depth-2 failure and depth-4 success. If the adaptive driver
skips a narrower width after a failure, that width remains untested; a direct
single-width run can check it without the traversal heuristic. This is an
empirical search, not an exhaustive capacity proof or an unlimited budget sweep.
