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

## Status

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

The coarse search is ready to launch. No new memorization result has been
claimed yet. The independently verified starting point remains one block at
width 512, with 29,416,960 physical parameters.
