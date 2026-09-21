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
width 16. Intermediate refinements such as width 48 or 96 use the same rule.
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

The current backend requires widths and head dimensions divisible by 16.
That is an implementation restriction, not a mathematical minimum. Dense
kernels internally pad narrow dimensions to 64, and full-vocabulary loss work
does not shrink with residual width, so fewer parameters need not imply a
proportional speedup. The result will be an empirical frontier within the
tested family, grid, and training budgets, not a globally smallest architecture.

## Verified memorization results

| Run | Blocks | Width | Heads × dimension | FF width | Parameters | Updates | Epochs | Errors | Mean loss (nats) |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Previous depth search | 1 | 512 | 8 × 64 | 2,048 | 29,416,960 | 2,944 | 46 | 0 / 10,002 | 0.000131258831 |
| Coarse | 1 | 256 | 4 × 64 | 1,024 | 13,922,048 | 3,456 | 54 | 0 / 10,002 | 0.000256836483 |
| Coarse | 1 | 128 | 2 × 64 | 512 | 6,764,416 | 2,944 | 46 | 0 / 10,002 | 0.001181248501 |

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

## Verified budget failures

| Run | Blocks | Width | Parameters | Updates | Final errors | Best observed errors | Final mean loss (nats) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Coarse | 1 | 64 | 3,333,056 | 5,000 | 2 / 10,002 | 1 at step 4,992 | 0.073553935635 |

The width-64, one-block trial reached its full update cap, not its time cap.
Its final checkpoint reloaded with the same two errors and passed the independent
artifact audit. Both errors occur at the first scored token: it swaps ` a` and
` ordinary` after the prefixes `Heating a gas in` and `Oxygen gas in` (lines
202 and 699). This near miss does not show insufficient capacity; it warrants
a separately labeled longer-budget check after the coarse pass.

Checkpoint:
`/home/ubuntu/checkpoints/memorize_general_facts/width_depth_coarse_0/width_64/layers_1/step_5000`.
Artifacts: `runs/width_depth_coarse_0/width_64/layers_1/`. Independent prediction
TSV SHA-256: `cfa1d031cf17645581952a4e619feedff44f7de7319b21252cdc47f807cabb91`.

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

The coarse search started on 2026-09-21 at 00:46:16 UTC with one block at width
256. Its live record is `runs/width_depth_coarse_0/width_depth_search_summary.json`;
checkpoints are under
`/home/ubuntu/checkpoints/memorize_general_facts/width_depth_coarse_0/`.
Widths 256 and 128 passed both independent checks. One block at width 64
exhausted its budget with two errors; two-block width-64 training was underway
by 01:08:02 UTC. All 16 tensors shared with the one-block initialization match
byte-for-byte, including the final LayerNorm after accounting for file indices.
The smallest verified success so far is one block at width 128, with 6,764,416
physical parameters; the width/depth frontier is not complete yet.
