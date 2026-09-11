# What blocks 1–7 contribute beyond the B0 MLP automaton

## Main result

For the original Shakespeare checkpoint at step 13030, B0 supplies strong
local token-continuation tendencies, but the rest of the model is essential
for choosing context-appropriate continuations. There is a clear empirical
shift: B1–B2 are more sensitive to attention removal, B3–B4 need both branches,
and B5–B7 are more sensitive to MLP removal. These are causal sensitivity
measurements, not a complete attribution of semantic functions to layers.

The experiment used **39 conditions** (36 zero ablations and three clean
anchors), eight fixed corpus passages, and the four previously recorded
Exeunt-related contexts. No retraining, backward pass, optimizer step,
checkpoint write, replacement corpus, or comparison model was involved.

## Protocol and scope

- Checkpoint: `/home/ubuntu/checkpoints/shakespeare/step_13030`.
- Corpus: `testdata/shakespeare.txt`, 1,835,163 tokens under the project's
  native tokenizer. Existing native tokens were verified byte-for-byte against
  the complete corpus; Hugging Face was not used to retokenize the corpus.
- Eight 1,025-token windows, selected before scoring by
  `floor(i * (token_count - 1025) / 7)`, `i=0..7`. Each provides 1,024 inputs
  and next-token targets. Learned positions reset to 0..1023 in each window.
- All 8,192 predictions are retained. Primary metrics exclude the first 128
  positions in each window, leaving **7,168** predictions with more context.
- These are fixed **corpus probes**, not a certified held-out evaluation set.
  Teacher forcing supplies the real preceding tokens even after a wrong
  prediction. This is not a free-running generation or a corpus-wide estimate.
- A branch is removed by zeroing its output projection **and bias at all
  positions**. A whole-block removal zeros both attention and MLP outputs.
  Input projections, LayerNorms, and the residual paths remain in place.
  Every condition starts from restored clean weights, not the previous arm.
- For block `b`, attention output parameters are `6+12*b, 7+12*b`; MLP output
  parameters are `12+12*b, 13+12*b`. The isolated-B0 control additionally zeros
  position embedding tensor 1, B0 attention, and both branches of B1–B7.
- Native production BF16 kernels, FP32 weights/logits/loss. Per-token loss and
  argmax are reduced on the GPU and copied through pinned host arrays. The
  four trace contexts export only their last logit rows. Probabilities in
  this report use **temperature 1**, not the original sampler's 0.8.

## Overall necessity of the later stack

NLL is mean next-token negative log likelihood in natural-log units; lower
is better. Accuracy is greedy next-token accuracy on the same primary rows.

| Model condition | Mean NLL | Accuracy |
| --- | ---: | ---: |
| Full B0–B7 model | 1.4421 | 81.08% |
| Isolated B0 MLP, no positions or attention | 8.9644 | 27.69% |
| Full B0, remove B1–B7 | 7.4372 | 26.72% |
| Remove attention in all B1–B7; keep their MLPs | 7.3426 | 23.13% |
| Remove MLPs in all B1–B7; keep their attention | 6.0110 | 26.27% |
| Remove attention in all B0–B7 | 7.3809 | 13.13% |
| Remove MLPs in all B0–B7 | 12.1747 | 1.55% |

Neither the later attention nor the later MLP stack is an optional refinement
to a mostly complete B0 language model. Removing either destroys most of the
model's corpus prediction accuracy.

## Which blocks matter, and which branch does the work?

Every row removes one component from the otherwise complete model. NLL changes
are relative to the full model's 1.4421; higher means a more harmful removal.
The attention and MLP columns are separate experiments, not additive terms.

| Block | NLL increase: remove attention | NLL increase: remove MLP | NLL increase: remove whole block | Accuracy without whole block |
| --- | ---: | ---: | ---: | ---: |
| B1 | +3.4415 | +0.2117 | +4.0063 | 32.05% |
| B2 | +1.6028 | +0.2283 | +1.8814 | 47.87% |
| B3 | +0.7638 | +0.6398 | +1.3854 | 56.17% |
| B4 | +1.1926 | +0.7961 | +1.8785 | 49.23% |
| B5 | +0.3139 | +0.8600 | +1.1189 | 57.52% |
| B6 | +0.2024 | +0.9048 | +1.1307 | 57.24% |
| B7 | +0.1484 | +1.0514 | +1.2527 | 55.93% |

By whole-block NLL impact the ordering is B1, B2, B4, B3, B7, B6, B5. The
B2/B4 difference is tiny and should not be treated as a strong ranking.
Removing any one of B1–B7 worsens NLL in **every one of the eight windows**.

Interpretation consistent with these measurements and the architecture:

1. **B1–B2: predominantly contextual mixing.** Their attention branches matter
   far more than their MLP branches. This identifies important context-reading
   computations, but not the specific earlier tokens or features they read.
2. **B3–B4: mixed processing.** Both attention and MLP transformations are
   materially necessary. B4 remains particularly attention-sensitive.
3. **B5–B7: predominantly MLP-dependent final decisions.** Their MLPs have
   larger removal effects than their attention branches; B7 MLP is especially
   important for converting the prepared representation into final predictions.

These do not establish labels such as "B2 is syntax" or "B6 stores facts."
An ablation tests a component in a co-adapted network and can change the input
distribution encountered by downstream components.

## Direct connection to the B0 automaton

Classify every primary input position using the previously constructed B0
graph with strict edge threshold 0.75. This classification is fixed before
running the ablations. It separates good local continuation rules from cases
where the same rule conflicts with the actual next token in context.

| Position category | Count | Isolated B0 accuracy | Full B0 only | Full model |
| --- | ---: | ---: | ---: | ---: |
| B0 graph has a correct confident edge | 1,768 | 100.00% | 91.80% | 97.45% |
| B0 graph has a wrong confident edge | 2,810 | 0.00% | 3.56% | 74.80% |
| B0 graph has no confident edge | 2,590 | 8.38% | 7.41% | 76.72% |

The first two isolated-B0 accuracy values follow from how those groups are
defined; they are not independently selected successes or failures. The
informative result is what the full contextual model does on those same rows.

On the **2,810 wrong-edge positions**, removing all later attention lowers
accuracy from 74.80% to **4.73%**; removing all later MLPs lowers it to
**14.77%**. Removing just B7 MLP lowers it to **44.63%**. Later layers therefore
do much more than increase confidence in B0's preferred suffix: they often
need to reject it and select a different continuation.

They do not always improve a correct local rule. Full-model accuracy on the
correct-edge group is 97.45%, not 100%. The network balances context against
token-local tendencies and sometimes makes the wrong contextual adjustment.

## Cumulative truncation

These conditions retain B0 through the named block, zero every later branch,
and apply the same trained final LayerNorm and tied output head.

| Last retained block | NLL | Accuracy |
| --- | ---: | ---: |
| B0 | 7.4372 | 26.72% |
| B1 | 6.7099 | 23.63% |
| B2 | 6.5193 | 25.45% |
| B3 | 6.1794 | 26.30% |
| B4 | 5.3199 | 29.87% |
| B5 | 4.1527 | 37.04% |
| B6 | 2.6948 | 55.93% |
| B7 | 1.4421 | 81.08% |

NLL steadily improves with depth; greedy accuracy is not monotonic early on.
This does **not** mean B1 is harmful: removing B1 from the full network is the
most damaging later-block removal. Intermediate representations are inputs to
subsequent blocks, not necessarily ready-made output predictions under the
final head.

## Exeunt: start the word, spell it, choose what follows

The four exact contexts come from recorded generated indices 1340–1343,
with targets ` Ex` (1475), `e` (68), `unt` (2797), and ` sever` (1750).
The first piece was sampled despite ranking second; this study conditions on
the recorded prefixes and does not regenerate their preceding text.

### Starting the word is context dependent

Full B0 alone gives ` Ex` just **0.0001162%**. The complete model gives it
**25.3887%**. Removing B1 attention lowers that to **0.0315489%**; removing B2
attention lowers it to **0.843821%**. Later attention is crucial to preparing
this word-start prediction.

Later MLPs need not favor it: removing B5, B6, or B7 MLP raises ` Ex` to
52.6233%, 77.7320%, or 91.4281%, respectively. B7 MLP makes the final model
prefer another space here. Suppression is not automatically an error.

### Spelling is locally strong, but the later blocks are co-adapted

Full B0 alone predicts `e` and `unt` with **99.6924%** and **99.9967%**.
Every individual later branch removal leaves both suffixes ranked first.
Removing all later MLPs still leaves them at 99.9920% and 99.9990%.

However, removing all B1–B7 attention while retaining their MLPs collapses
them to **0.0004474%** and **0.0262560%**. Keeping just B0 is very different
from leaving seven later MLPs operating without their expected attention
updates. This is evidence of co-adaptation, not evidence that a local spelling
rule is implemented independently in all attention layers.

### The next word exposes the late MLPs' contribution

The local automaton produces `Exeunt all`; the recorded complete model
continues as `Exeunt severally`. For the first next-word token:

| Condition | P(` sever`) | Winning token |
| --- | ---: | --- |
| Full B0 only | 0.0016572% | ` all` |
| Keep B0–B6 | 0.0898540% | ` all` |
| Full model, remove only B7 MLP | 0.1213838% | two newlines |
| Full model | 55.6606478% | ` sever` |

This is now a **measured causal ablation**, not merely an intermediate readout.
B7 MLP both raises the ` sever` logit and suppresses ` all` and newline
competitors. Removing B5 MLP also lowers final P(` sever`) to **0.1455125%**;
removing B6 MLP lowers it to **10.4153%**. Earlier contextual processing matters
too: without B1 attention, P(` sever`) is **0.0125746%**.

The last MLP does not act alone or simply contain an isolated replacement
for the B0 automaton: it makes a context-conditioned decision using the
representation delivered by the preceding blocks.

## Heterogeneity, controls, and limits

The eight windows are not equally easy. The final window has clean NLL 7.9769
and accuracy 21.43%; the other seven have NLL 0.2244–1.1850 and accuracy
76.90–95.87%. The report keeps that difficult window instead of dropping it
after observing its loss. Raw per-window measurements are in `summary.json`.
All seven whole-block removal effects remain positive in each individual window.

Integrity and numerical checks:

- All 100 checkpoint tensors and frozen inputs were hashed before and after;
  none changed. Weight restoration was checked byte-for-byte after every arm.
- Clean-before, clean-repeat, and clean-after loss/argmax/trace outputs are
  byte-identical.
- All four packed clean trace rows reproduce the previous native exact-prefix
  logits byte-for-byte.
- All 28 cumulative-truncation trace rows match the corresponding saved native
  logit lenses byte-for-byte. B7-MLP removal matches all four pre-MLP lenses.
- The isolated-B0 control reproduces the automaton's winning token at every
  corpus position whose source has a confident graph edge.
- Six new CPU schedule tests, plus the existing native weight-intervention
  and selected-row test suites, passed before the experiment.

These checks establish reproducibility of these interventions, not a complete
semantic circuit. The next sharper experiment would patch selected attention
sources or intermediate features between matched contexts to identify which
information makes later MLPs override a given B0 continuation.

## Artifacts and reproduction

- Frozen fixtures and hashes:
  `/tmp/pluto-block-ablation-probes-20260911.DMd4BY/probe_manifest.json`.
- Complete run:
  `/tmp/pluto-block-ablation-step13030-20260911/`.
- Machine-readable per-condition metrics and four token distributions:
  `/tmp/pluto-block-ablation-step13030-20260911/summary.json`.
- Native per-token losses/argmax and selected logits: the run's `native/`.
- Frozen command/input hashes: `plan.json`; verified completion/output hashes:
  `result.json`; streaming progress: `stdout.log`.
- One-off native harness, schedule/tests, and runner:
  `ai-slop/weight_analysis/block_ablation_*`.

```sh
bazel build -c opt //ai-slop/weight_analysis:block_ablation_probe
bazel test -c opt //ai-slop/weight_analysis:block_ablation_arms_test \
  //ai-slop/weight_analysis:causal_probe_test \
  //ai-slop/weight_analysis:token_trace_probe_test --local_test_jobs=1

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/ubuntu/.venv/bin/python -B \
  ai-slop/weight_analysis/block_ablation_run.py \
  --fixture=/tmp/pluto-block-ablation-probes-20260911.DMd4BY \
  --trace=/tmp/pluto-exeunt-step13030-trace-20260911 \
  --binary=bazel-bin/ai-slop/weight_analysis/block_ablation_probe \
  --graph=/tmp/pluto-mlp-automaton-step13030-20260911-b512/graph.json \
  --output=/tmp/pluto-block-ablation-step13030-rerun
```

Use a new output directory. The runner refuses to overwrite a run and verifies
the frozen input bindings both before and after the native process.
