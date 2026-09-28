# Removing the A3 MLP readout's skip connection

This is an AI-generated, unreviewed experiment report.

## Question and controlled change

Does the residual path make the replacement MLP spend capacity canceling its
input, rather than predicting the next token?

The input `x` is the frozen model's third post-attention **residual** output.
The comparison changes only the replacement readout:

- Residual: `head(final_LN(x + FC2(GELU(FC1(input_LN(x))))))`.
- No residual: `head(final_LN(FC2(GELU(FC1(input_LN(x))))))`.

The original transformer's residuals are untouched. The readout's input and
final LayerNorms remain trainable; the source model and tied head stay frozen.
Removing the skip changes neither weight shapes nor parameter initialization.
Both variants have 3,200 trainable parameters: a 10/150/10 affine MLP with 3,160
parameters, plus 20 parameters in each LayerNorm.

## Protocol

- Source: 4 transformer blocks, width 10, FF20, one head, context 27,
  compact vocabulary 4,475; 48,680 source parameters.
- Checkpoint:
  `/home/ubuntu/checkpoints/memorize_general_facts/context27_L4_W10_FF20_matched_0/trial_000_L4_W10_FF20/checkpoints/layers_4/step_89600`.
- Same 1,024 facts, five supplied tokens per fact, 10,002 scored suffix/EOS
  targets, including 1,024 EOS targets. Prompt and padding rows are unscored.
- Each condition runs 300,000 AdamW updates, batch 32, seed 3, initial learning
  rate 0.01, cosine decay to 0.001, no weight decay or gradient clipping.
- Evaluate every 1,000 steps; choose the best evaluated checkpoint by token
  accuracy, breaking ties by mean cross entropy. Then verify greedy completions
  from actual five-token prefixes and check frozen weights are unchanged.
- Both jobs use the same snapshotted binary and inputs and run concurrently.
  Timings are therefore not isolated performance benchmarks.

The source independently predicts all 10,002 targets correctly. Its A3 vectors
are distinct across all scored positions, so exact lookup can distinguish the
targets. **This does not establish linear separability**, which is an extra
premise in the original suggestion.

Artifacts (logs, inputs, hashes, checkpoints, plots, training curves):
`/tmp/pluto-no-residual-20260928-01/`.

## Reproduction

Build and run a fresh pair with the runner; it rejects existing output paths:

```sh
bash scripts/memorize_general_facts/run_residual_ablation.sh \
  --checkpoint=/home/ubuntu/checkpoints/memorize_general_facts/context27_L4_W10_FF20_matched_0/trial_000_L4_W10_FF20/checkpoints/layers_4/step_89600 \
  --tokenizer=/home/ubuntu/datasets/tokenizer/gpt2 \
  --run_dir=/tmp/pluto-no-residual-new-run
```

The experiment used the historical tokenizer snapshot with SHA256
`8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6`;
its exact copy is retained in the artifact directory's `inputs/tokenizer.json`.
To reproduce those bytes, pass that `inputs` directory as `--tokenizer`.
The corpus SHA256 is
`814c062e7d7592fe4a4e5b158a37bd37da51700f817c19eb981c1e93d33f245c`.

The underlying puzzle flag is `--mlp_residual_connections=false`, valid only
with `--mode=puzzle` and either `--train_mlp` or `--train_stacked_mlp`.
Its default is true, preserving previous behavior.

## Results

Both conditions completed all 300,000 updates. At each condition's best
evaluated token-accuracy checkpoint:

| Readout | Best step | Correct scored tokens | Non-EOS accuracy | Mean CE | Greedy complete facts |
| --- | ---: | ---: | ---: | ---: | ---: |
| With residual | 235,000 | 3,594/10,002 (35.9328%) | 2,572/8,978 (28.6478%) | 3.32873891 | 1/1,024 |
| Without residual | 136,000 | 3,442/10,002 (34.4131%) | 2,419/8,978 (26.9436%) | 3.49538106 | 2/1,024 |

EOS accuracy was 1,022/1,024 with the residual and 1,023/1,024 without it.
The final step (rather than best checkpoint) had 35.7928% versus 33.9532%
token accuracy, so selecting the best checkpoint does not reverse the result.
Both jobs verified that source weights and the frozen head remained unchanged.
Every one of the residual control's 301 recorded evaluations exactly matches
the previous 300k-step experiment in CE and accuracy counts, excluding timing.

**Conclusion:** removing only the MLP skip connection did not resolve this
instance of the puzzle. Token accuracy decreased by about 1.52 percentage
points, non-EOS accuracy decreased by about 1.70 points, and both models
completed fewer than 0.2% of facts. This is one paired seed/recipe, not proof
that removing residuals can never help. It also does not refute the suggestion
under its stronger linear-separability premise.

Removing this skip also removes the raw-input path around the input LayerNorm:
the plain readout depends only on its normalized input. Thus this is not a test
of every possible non-residual architecture (for example, one without that
input normalization).

## Five-block, approximately iso-parameter follow-up

The original experiment above used **one** MLP, not five. The follow-up uses
five sequential MLPs with hidden widths `12,12,12,11,11`, with and without
their skips. Each has its own trainable input LayerNorm; a trainable final
LayerNorm precedes the frozen tied head. The same 300,000-step recipe applies.

The replaced source suffix contains MLP3 + block4 + final LayerNorm:
`450 + (460 + 450) + 20 = 1,380` parameters. The replacement has
`21 * sum(widths) + 30 * 5 + 20 = 1,388` parameters. This includes **all**
trainable biases and LayerNorms on both sides, not just affine matrices.
An exact match is impossible using positive integer hidden widths in this
architecture: the nearest smaller choice has 1,367 parameters. The selected
stack is the nearest possible match, eight parameters (0.58%) larger.

Reproduce this pair by adding `--stacked --iso_params` to the runner command
above. The underlying binary flags are
`--train_stacked_mlp --mlp_iso_parameters=true`, plus
`--mlp_residual_connections=true|false`. Ordinary stacked mode retains its
previous five width-150 MLPs. Each run records all five resolved widths and the
exact parameter difference in `run.txt`; seeded tensors and data order match
between the residual and non-residual variants.

The paired 300,000-step run was launched from commit `1028761` at
2026-09-28 18:23 UTC. Its snapshotted binary, inputs, exact commands, logs,
training curves, and checkpoints are in
`/tmp/pluto-no-residual-iso-20260928-01/`. Results are pending; the single-block
numbers above must not be attributed to this five-block experiment.

Validation: all 74 Bazel test targets and 230 script tests passed. New tests
exhaustively check nearest-budget selection over 630 small configurations,
verify the actual GPU tensor count and initialization parity, exercise updates
to all 32 trainable tensors, and verify frozen source/head weights. The paired
one-step smoke run at `/tmp/pluto-no-residual-iso-smoke-20260928-01/` exited
successfully for both conditions and the launcher.

## Single-block validation and provenance

- 74 Bazel test targets and 223 existing script tests passed.
- New GPU tests verify identical seeded weights/parameter counts, disappearance
  of the residual wrappers, removal of both forward/backward identity paths,
  and updates to trainable tensors while source/head stay frozen.
- New CLI tests restrict the flag to puzzle readout training.
- The paired runner passed syntax/argument checks, mocked success/failure
  paths, and a real one-step paired smoke run.
- The full-run training subprocesses both exited zero. Their launcher reported
  a shell parse error only after both had completed because the launcher file
  was edited while it was running. The finished logs, per-child exit statuses,
  immutable executable, and checkpoints establish successful training.
  The finalized, unmodified launcher subsequently passed the one-step smoke
  run at `/tmp/pluto-no-residual-runner-smoke-20260928-01/`.
- Full-run executable SHA256:
  `26d09087d7a03c2a7a9cabca366075891cdf5f09ff786c709614d6597fad7925`.
- Wall times were about 636 and 634 seconds for the concurrent pair, including
  end-of-run verification.
