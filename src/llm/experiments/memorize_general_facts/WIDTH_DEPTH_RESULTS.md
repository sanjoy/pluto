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

The measured 5,000-update frontier still contains `(1 block, width 128)` and
`(2 blocks, width 64)`: neither dominates the other in depth and width under
that protocol. The measured 20,000-update frontier and the pooled frontier
instead give `(1 block, width 32)`, with 1,654,240 parameters, the smallest
successful model measured so far. The changed cosine schedule prevents
attributing improvements over the coarse protocol solely to additional
updates. These are frontiers over measured outcomes, not
capacity impossibility claims.

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

The next search, `runs/width_depth_long_1/`, started at 02:30:46 UTC. It tests
widths 32, 16, and 8 at depths 1, 2, 4, and 8, with an explicitly fixed single
attention head and the same 20,000-update schedule and native binary as
`width_depth_long_0`. One block at width 32 passed both independent checks at
step 9,984, moving the measured longer-budget and pooled frontiers to `(1,32)`.
One block at width 16 has now completed and independently verified its
20,000-update budget failure above. Its completed artifacts are committed;
the changing search manifest and future/live trials are not. The fixed-head driver and
backward-compatible reporter pass all 119 Python experiment tests.

A one-block width-24 midpoint trial, `runs/width_depth_refine_24`, started at
02:44:58 UTC with the same longer-budget protocol and one head. GPU contention
made simultaneous training inefficient, so it was paused at 02:47:21 UTC with
its in-memory optimizer state retained while one-block width 16 finished.
Width 24 resumed at 03:00:50 UTC; the next two-block width-16 trial and search
coordinator are paused until it finishes. Scheduling details and the elapsed-time
caveat are in `runs/GPU_SCHEDULING.md`. Width 24 and two-block width 16 do not
have final outcomes yet. Added tests specifically cover one 24-wide BF16 head
at context 1,024 with CPU/GPU forward/backward agreement and three bitwise
GPU repeats, plus complete one-head width-24 GPT-2 forward/backward and tied
parameter counts; both optimized test targets pass.

After the active pass, refine observed width gaps with a small, bounded set of
midpoints (such as 24 between 32 and 16, or 12 between 16 and 8). Fill omitted
depths where a measured success transition makes them relevant, for example
depth 3 between a depth-2 failure and depth-4 success. If the adaptive driver
skips a narrower width after a failure, that width remains untested; a direct
single-width run can check it without the traversal heuristic. This is an
empirical search, not an exhaustive capacity proof or an unlimited budget sweep.
