# End-to-end training with a near-isoparameter MLP suffix

AI-generated experimental notes; not human-reviewed.

## Question

The frozen-A3 experiment's three residual 10/20/10 MLPs had only 34.8730%
scored-token accuracy and completed 1/1,024 facts after 300,000 updates. Can
the same suffix architecture fit the corpus when its input representation,
embeddings, and head are learned jointly from scratch?

This is **not** an attention-free model. Its computation is:

```text
learned token + position embeddings
  -> A1 -> M1 -> A2 -> M2 -> A3
  -> MLP -> MLP -> MLP
  -> final LayerNorm -> tied embedding projection
```

Each attention/MLP includes its own pre-LayerNorm and residual addition.
Attention is causal, width 10, with one head. Every MLP is 10/20/10 with GELU.
Context is 27, the compact vocabulary is 4,475, and there is no dropout.

| Component | Original GPT-2 | New model |
| --- | ---: | ---: |
| Token embedding + learned position table | 45,020 | 45,020 |
| Prefix through A3, excluding embeddings | 2,280 | 2,280 |
| Suffix after A3, including final LN | 1,380 | 1,370 |
| **Whole model** | **48,680** | **48,670** |

The ten-parameter difference is 0.0205% of the full model. No unused parameters
are added just to make counts exactly equal. The suffix is three pointwise
MLPs plus final LN, versus two MLPs, one attention sublayer, and final LN in
the original. The whole graph therefore has three attention sublayers and
five MLPs, compared with the original four of each.

## Result

**Verified full memorization at step 111,872.** The first zero-error evaluation
took 317.155 seconds of native training time (about 5m 17s). Reloading the saved
checkpoint, independently auditing original-token targets, and generating from
five-token prompts all passed:

| Measurement | Result |
| --- | ---: |
| Scored suffix/EOS tokens correct | 10,002 / 10,002 |
| Exact greedy corpus completions | 1,024 / 1,024 |
| Independently audited mean CE | 0.00861463915 |
| Training updates | 111,872 |
| Facts processed | 3,579,904 (3,496 corpus epochs) |
| Unique trainable parameters | 48,670 |

The driver completed with status `verified`; its `greedy_verification.json`
contains no failed lines. Comparing every final tensor with `step_0` confirmed
that **all 52 unique weight files changed**, including the embedding and all
prefix layers.

| Experiment | Trainable parameters | Updates | Token accuracy | Exact greedy facts |
| --- | ---: | ---: | ---: | ---: |
| Original four-block GPT-2, all weights fresh/trainable | 48,680 | 89,600 | 100% | 1,024 / 1,024 |
| **A3 prefix + three 10/20/10 MLPs, all weights fresh/trainable** | **48,670** | **111,872** | **100%** | **1,024 / 1,024** |
| Same three-MLP suffix on frozen original A3 states/head | 1,370 | 300,000 | 34.8730% | 1 / 1,024 |

The original whole-model control is the previously completed source run, not
a rerun in this experiment; it took 259.301 seconds to its first perfect
evaluation. Both whole models used the same seed and 120,000-update schedule.
Evaluations are every 256 updates, so the reported convergence step is the
first observed perfect checkpoint rather than necessarily the first perfect
update. Small GPU tests ran alongside this experiment: timings are descriptive,
not an isolated performance benchmark.

This establishes that the three-attention/five-MLP architecture has enough
capacity and is trainable on this corpus. The fourth attention sublayer is
not required if the prefix and tied embedding/head can adapt jointly. It does
**not** prove that the frozen A3 vectors can be fit by the same small suffix,
nor isolate prefix adaptation from embedding/head adaptation or the different
frozen-readout training recipe. All conclusions here are for one seed.

## Initialization and training

No pretrained checkpoint is read. All 52 unique parameter tensors are freshly
initialized and passed to Adam, including both embedding tables, the retained
attention/MLP prefix, the replacement suffix, and final LN. The head is tied
to the trainable token embedding, not a frozen copy. There is no cached A3
dataset; the whole graph runs forward and backward on every minibatch.

This matches the successful original **whole-model** recipe, not the different
frozen-readout recipe:

- Seed 1337; 32 facts per minibatch; independent sequence length 27.
- Maximum 120,000 updates; evaluate every 256 and stop at zero scored errors.
- Adam beta1 .9, beta2 .99, epsilon 1e-8; no clipping or weight decay.
- Learning rate warms up over 100 updates to .0012, then cosine-decays to .00012.
- Native GPT-2 initialization: embeddings/QKV/FC1 standard deviation .02;
  attention output/FC2 .005; identity LayerNorms, zero biases.
- BF16 compute/activations, FP32 parameters, optimizer state, and reductions.
- Standard CE over 10,002 suffix-plus-EOS targets after five-token prompts.
  Prompt and padding rows are not direct loss targets.

The prefix's initial tensors 0..31 were compared byte-for-byte with the
successful original GPT-2 run's saved `step_0` and all matched. The three tail
MLPs use the native independent seed schedule for MLP indices 2, 3, and 4.

The earlier frozen-readout runs used seed 3, .2/.1 MLP initialization,
LR .01 -> .001, beta2 .999, and 300,000 updates. Therefore comparing against
those runs does not isolate unfreezing as the only experimental change. The
closely matched control here is the original fully trained four-block GPT-2.

## Run and reproduction

Local artifacts:

```text
/tmp/pluto-a3-isoparam-end-to-end-20260928-01/
```

Final checkpoint:

```text
/tmp/pluto-a3-isoparam-end-to-end-20260928-01/trial_000_L4_W10_FF20/checkpoints/layers_4/step_111872
```

The driver snapshots the binary, exact corpus, tokenizer, and audit scripts;
its `summary.json` records source hashes and every command. Checkpoints contain
`architecture.txt` with `gpt2_a3_mlp_stack`, and native inference requires
`--a3_mlp_stack` to load them. The `layers_4` directory name describes the
four-block reference configuration, not four attention sublayers in the
variant.

From a Python environment with `tokenizers` installed:

```sh
bazel build -c opt //src/llm/experiments/memorize_general_facts:memorize_general_facts
python -B scripts/memorize_general_facts/run_compact_size_sweep.py \
  --binary=bazel-bin/src/llm/experiments/memorize_general_facts/memorize_general_facts \
  --corpus=testdata/general_facts_dataset.txt \
  --tokenizer=/path/to/gpt2-tokenizer \
  --run_dir=/tmp/new-a3-end-to-end-run \
  --deadline_unix="$(( $(date +%s) + 1800 ))" \
  --candidates=4:10:20:120000:0.0012 --a3_mlp_stack \
  --seed=1337 --batch_size=32 --eval_every=256 --verification_reserve=120
```

`--a3_mlp_stack` is available only in train/infer modes, requires `--layers=4`,
and rejects depth search. Training and both inference paths use the same
factory. The driver first reloads the checkpoint in a fresh process, audits
all original-token target IDs independently, and then greedily generates all
1,024 facts from precisely five-token prompts. Merely reaching teacher-forced
accuracy is not accepted as a verified result.

For interactive inference on the resulting checkpoint:

```sh
bazel-bin/src/llm/experiments/memorize_general_facts/memorize_general_facts \
  --mode=infer_model --a3_mlp_stack \
  --infer_checkpoint=/tmp/pluto-a3-isoparam-end-to-end-20260928-01/trial_000_L4_W10_FF20/checkpoints/layers_4/step_111872 \
  --tokenizer=/tmp/pluto-a3-isoparam-end-to-end-20260928-01/inputs \
  --generation_tokens=27
```

## Code checks

The seven targeted C++ test targets pass, including the unchanged GPT-2 tests,
six new whole-network variant tests, both CLI suites, and the existing puzzle
tests. Tests verify shared initialization, exact parameter counts, tied-head
ownership, nonzero updates in every unique tensor, causal/batch-isolated
forward and backward behavior, and checkpoint round-trip. All 226 experiment
script tests pass. A live negative check confirms that omitting the selector
when loading a variant checkpoint produces an architecture-mismatch error.
