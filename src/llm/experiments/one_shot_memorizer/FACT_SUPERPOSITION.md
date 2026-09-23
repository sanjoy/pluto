# Do independently learned fact updates add?

## Prospective matched-schedule experiment

The sentence-ablation and embedding-transplant experiments show distributed,
co-adapted changes. Here we test a simpler constructive hypothesis directly:
can independently learned fact updates be added to build a model of both?
This is not a dataset-only construction, because the component updates are
still trained. It tests a proposed composition law for those updates.

Use exactly the original corpus's lines 80 and 406, in that order:

```
The capital of France is Paris, a city on the Seine.
The capital of Greece is Athens, an ancient Mediterranean city.
```

Keep the original full-corpus compact vocabulary and step-zero weights.
Run 1,024 batch-one optimizer steps, seed 1337, with the original warmup,
40,000-step cosine schedule, Adam parameters, zero decay, and no clipping.
The two sentences are shuffled without replacement with the same iterator
seed. Each receives 512 scheduled exposures. Four conditions use exactly
the same initialization, batches, global optimizer clock, and loss scaling:

* both facts, plus a duplicate baseline checked bytewise after every update;
* France-only contribution: zero Greece's logit gradient after loss backward;
* Greece-only contribution: zero France's logit gradient after loss backward.

An omitted slot is still executed, and **Adam still steps on zero gradients**.
Its momentum is not reset or paused. Thus these controls preserve the global
clock; they are not independently running the old batch-one, 512-update
single-fact experiments. The original fixed loss normalization is retained.

With common initialization `W0` and the two masked-training endpoints `WA,WB`,
freeze these whole-model constructions before measuring their predictions:

```
SUM  = W0 + (WA-W0) + (WB-W0)
MEAN = W0 + 0.5*((WA-W0) + (WB-W0))
```

Compute each scalar in FP64 and round once to its FP32 master representation.
Apply the same formula to every unique tensor, including embeddings,
positions, biases, and norms. Do not choose per-layer coefficients or search
interpolation factors. The tied embedding alias remains tied.

## Controls and measurements

Check common initialization and compact mappings, shape/inventory integrity,
finite values, and unchanged source checkpoint bytes. Evaluate the joint,
France-contribution, Greece-contribution, SUM, and MEAN models on both entire
suffixes plus EOS from their first five tokens. Also report next-token counts
and parameter error relative to the directly trained joint model.

At step one, only the first scheduled sentence has contributed. The other
masked run must remain exactly at initialization, and SUM must match the
joint checkpoint byte-for-byte. This checks arithmetic/provenance, not the
later hypothesis. The duplicate joint run checks deterministic execution.

A failed formula would reject these particular additive constructions under
matched training conditions, not all possible nonlinear combinations or all
task-vector methods. A successful formula would be a two-fact construction,
not evidence that 1,024 facts superpose independently. Differences between
joint and isolated activations and Adam moments remain part of what is tested.
Generated checkpoints and reports remain local. Budget: under one hour.

## Result: neither fixed combination recovers the pair

The matched-schedule runs completed. The jointly trained endpoint and its
repeat are byte-identical. Each masked endpoint memorizes its own retained
sentence, so this is not a test of combining two unfinished component models.

| Model | France targets | Greece targets | Total targets | Exact suffixes plus EOS |
| --- | ---: | ---: | ---: | ---: |
| Jointly trained | 10/10 | 8/8 | 18/18 | 2/2 |
| France contribution only | 10/10 | 1/8 | 11/18 | 1/2 |
| Greece contribution only | 1/10 | 8/8 | 9/18 | 1/2 |
| SUM | 2/10 | 4/8 | 6/18 | 0/2 |
| MEAN | 1/10 | 3/8 | 4/18 | 0/2 |

Both constructions fail on the **first** generated token, immediately after
the five-token prompt. SUM predicts `.` for both countries (compact ID 2,
original GPT-2 ID 13); MEAN predicts `,` for both (compact ID 0, original
GPT-2 ID 11), instead of ` Paris` or ` Athens`. The target counts above use
teacher forcing over the entire gold continuation; those partial scores are
not successful free-running continuations. The separate greedy check feeds
back predictions only and stops each case at its first mismatch.

The masked France-only model instead predicts ` Paris` for Greece, and the
Greece-only model predicts ` Athens` for France. Those simple confusions are
different from the punctuation predictions caused by combining their weights.

### The step-one positive control succeeds exactly

The first scheduled update is France. At step one, the Greece-contribution
run remains byte-identical to initialization, the France-contribution run
matches the joint run, and **SUM matches every joint FP32 parameter byte**.
Thus the same arithmetic and checkpoint ordering work in the case where
only one fact has contributed. Failure at step 1,024 is not an unavoidable
failure of the addition implementation or a swapped component label.

All four runs start with identical parameter bytes and matching compact
vocabularies. The trainer checks its duplicate baseline after every update;
the probe independently checks the final duplicate checkpoint. Each of the
five evaluation conditions uses a fresh model, preserving the tied embedding
alias. Every upload matches its intended parameter bytes, and inference and
saving leave those bytes unchanged. All 12 loaded source checkpoints are
strictly reloaded at the end: their weight bytes are unchanged and their
compact mappings still match. Unrelated checkpoint metadata is not audited.

Against the joint endpoint's 114,256 unique FP32 parameters, the global
relative L2 errors are 0.793659 for SUM and 0.587266 for MEAN. MEAN is closer
in this parameter-space metric but has fewer correct teacher-forced targets;
Euclidean weight distance is not a behavioral accuracy measure.

This rejects **these two predeclared whole-model composition formulas** on
this matched-clock pair. It does not reject all task-vector methods, learned
per-layer combinations, or nonlinear constructions. Adam's moment histories
and the activations producing later gradients differ between joint and
masked training even though the schedule, initialization, normalization and
global optimizer clock match. The constructed endpoints may be off
distribution. No coefficients were tuned after observing the failures, and
no whole-corpus or arbitrary-prompt fidelity is claimed.

### Reproduction and local artifacts

Build and run the checked-in probe against the completed local training run:

```sh
bazel build -c opt //src/llm/experiments/one_shot_memorizer:checkpoint_fact_superposition_probe
facts=/home/ubuntu/checkpoints/memorize_general_facts/compact_batch_32_no_clip_0
pair=/tmp/fact_superposition_pair_0
bazel-bin/src/llm/experiments/one_shot_memorizer/checkpoint_fact_superposition_probe \
  --initial_checkpoint="$facts/layers_8/step_0" \
  --joint_checkpoint="$pair/baseline/step_1024" \
  --omit_line_1_checkpoint="$pair/omit_line_1/step_1024" \
  --omit_line_2_checkpoint="$pair/omit_line_2/step_1024" \
  --tokenizer="$facts/inputs/tokenizer" \
  --corpus=/tmp/fact_superposition_pair.txt \
  --control_step1_dir="$pair" \
  --output_dir=/tmp/fact_superposition_probe_new
```

The completed report is `/tmp/fact_superposition_probe_0`. It contains
`conditions.tsv`, `per_sentence.tsv`, `parameter_errors.tsv`, `controls.tsv`
and a completed manifest. The two constructed checkpoints are
`sum/step_1024` and `mean/step_1024`, each with its compact vocabulary. These
generated artifacts are local, not checked into Git. The CPU arithmetic tests
cover SUM/MEAN symmetry, the one-component step-one control, FP64 cancellation,
ties-to-even FP32 rounding, invalid inputs, overflow and subnormal underflow.
