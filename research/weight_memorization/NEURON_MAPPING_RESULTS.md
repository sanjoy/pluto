# Fine-grained MLP reliance mapping

Date: 2026-09-09. Checkpoint: `step_13030`. This is a known-input, actual-forward
experiment, **not analytical extraction**. No optimizer, backward pass, or
checkpoint write is involved. Corpus addresses in this report refer to inputs,
not text decoded from weights.

## Question and design

The earlier whole-branch experiment found broad dependence on every MLP and
split-dependent effects in the last two blocks. This follow-up asks whether
smaller groups show repeatable differences between passages after accounting
for a shared response pattern. Selecting blocks 6 and 7 was informed by that
earlier validation, not a weight-only discovery.

The [protocol](NEURON_MAPPING_PROTOCOL.md) was committed as `3fb89e5` before
measurements. The tools were committed as `0312088`. One fixed PCG64 stream
partitions each block's 2,048 neurons into 32 groups of 64. Each group
intervention halves only its rows of the MLP output matrix; bias and other
rows are unchanged. Each group touches 32,768 weights / 131,072 bytes.

Block 6 uses `weight_84.bin`; block 7 uses `weight_96.bin`. Matrices are
little-endian FP32 [2048,512], row-major. Row j is bytes [2048*j,2048*(j+1)).
Masks and exact addresses are retained in the frozen plan. Group weight norms
span 5.9865–6.1909 in block 6 and 6.0113–6.1799 in block 7. Norm-matched
comparators are selected before any losses are observed.

All 32 passages are the same as in the earlier causal validation: 16 current
prefix windows and 16 current suffix windows, 1,025 native tokens each.
Historical membership in training is not authenticated. Microbatch size is
four, model activations are BF16, and master weights, logits, and losses FP32.

Discovery scores targets at loss indices [512,768); confirmation scores
[768,1024). These are different target tokens in the **same passages**, sharing
causal context. They are not independent-document replication.

For prefix loss-change matrix Delta[group,passage], the analysis removes row
and column means, then the leading left singular direction fitted on discovery
only. Confirmation uses that frozen direction. A pure product of group strength
and passage vulnerability, plus additive effects, therefore produces no map
in synthetic tests. This adjustment does not eliminate every confound, and can
also remove genuine distributed memory. Projected residuals mix many groups
and passages; only the raw finite intervention is that group's causal effect.

For each passage, selection requires positive raw discovery harm and a
positive residual above the declared numerical floor. Confirmation separately
reports raw harm, adjusted residual, rank, collateral effects, and the four
preselected norm comparators. There are no significance or private-storage
claims. A positive relative score without positive raw harm is not harmful
reliance.

## Results

All 69 prescribed arms completed in 411.14 seconds. Every intervention's
restoration check passed. The three clean loss files are byte-identical, as
are their three argmax files. The reporter authenticated every frozen input
and all 100 checkpoint weight files before and after analysis. None changed.

The main result is **weak confirmation of fine-grained passage selectivity**,
not a private passage-to-neuron map. Discovery selected a group for every
prefix passage, with 15 distinct groups. On confirmation, 13/16 selected
interventions increase raw loss, but only 9/16 also have positive projected
residuals. Only 5/16 remain in the top 16 of 64 residual scores. Eleven beat
their four preselected norm comparators' mean. No significance claim follows
from these counts.

The leading discovery direction removes 20.20% of double-centered discovery
energy, but only 1.21% of double-centered confirmation energy. The frozen
direction is numerically unique (first two singular values 0.0569307 and
0.0506744), so this is not a degenerate SVD selection. These descriptive
differences do not identify their causes; context-dependent features and
target differences can also produce them.

All loss changes below are mean NLL in nats per target token. Group names
abbreviate `blockB_groupGG_half`. The ranking uses all 64 confirmation
residuals, with rank 1 best. The last column is the residual minus the mean
of that group's four norm comparators. Every selected row, exact byte
interval, all collateral effects, and same-block rank is preserved in the
[complete numerical report](neuron_mapping_results.json).

| Prefix passage | Group | Discovery raw | Confirmation raw | Confirmation residual | Rank / 64 | Minus norm comparators |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 00 | 7 / 11 | 0.017898 | 0.004842 | 0.001061 | 21 | 0.002628 |
| 01 | 7 / 19 | 0.011756 | 0.004276 | 0.002908 | 12 | 0.003166 |
| 02 | 6 / 16 | 0.003615 | 0.002912 | 0.002337 | 11 | 0.001354 |
| 03 | 7 / 28 | 0.030726 | 0.008511 | 0.000652 | 27 | -0.001254 |
| 04 | 6 / 08 | 0.013757 | -0.000009 | -0.001473 | 42 | -0.001912 |
| 05 | 6 / 04 | 0.016084 | 0.004140 | 0.000101 | 32 | 0.001320 |
| 06 | 7 / 04 | 0.006933 | -0.002567 | -0.000664 | 38 | 0.001010 |
| 07 | 6 / 19 | 0.005127 | 0.006389 | -0.000250 | 34 | 0.003397 |
| 08 | 7 / 29 | 0.013177 | 0.000173 | -0.003066 | 52 | -0.003569 |
| 09 | 7 / 12 | 0.011196 | 0.022607 | 0.014242 | 2 | 0.008227 |
| 10 | 6 / 18 | 0.012258 | -0.001549 | -0.004202 | 51 | -0.003649 |
| 11 | 6 / 30 | 0.008023 | 0.006990 | -0.000335 | 38 | 0.006821 |
| 12 | 7 / 15 | 0.019064 | 0.014775 | 0.009155 | 3 | 0.011324 |
| 13 | 7 / 28 | 0.009158 | 0.001871 | -0.002774 | 51 | -0.000714 |
| 14 | 7 / 08 | 0.007929 | 0.007184 | 0.003897 | 6 | 0.006009 |
| 15 | 6 / 06 | 0.012341 | 0.004207 | 0.002172 | 20 | 0.005626 |

Groups selected for passages 09, 12, and 14 have relatively strong confirmation
ranks, but this is not localized storage: their interventions increase raw
confirmation loss on 15, 12, and 12 of the 16 prefix passages, respectively.
Across all selected groups, raw confirmation harm affects 9–15 prefix
passages. The repeated selection `block7_group28_half` for passages 03 and
13 has ranks 27 and 51, and raises loss on 14 prefix and 14 suffix passages.
Even the more promising individual associations therefore have substantial
collateral effects. No corpus excerpt is presented as text recovered from
these weight addresses.

## Controls and split effects

The clean last-512-target averages reproduce the preceding broad-branch run:
prefix NLL 0.4167070462, teacher-forced accuracy 90.9302%; suffix NLL
5.9791434358, accuracy 36.2305%. These are known-input measurements and do
not demonstrate free-running reproduction or authenticated historical splits.

| Last-512-target control | Mean prefix NLL change | Mean suffix NLL change | Prefix passages harmed | Suffix passages harmed |
| --- | ---: | ---: | ---: | ---: |
| Entire block 6 MLP output at half strength | 0.161490 | -0.135753 | 16 / 16 | 0 / 16 |
| Entire block 7 MLP output at half strength | 0.199848 | -0.136756 | 16 / 16 | 0 / 16 |

The finer interventions are smaller and not an additive decomposition of these
controls. Across 512 group-passage pairs per block on confirmation, raw prefix
loss increases in 394 pairs for block 6 and 410 for block 7. Mean changes are
0.00306069 and 0.00376366. Raw suffix loss increases in 178 and 168 pairs;
mean changes are -0.00411082 and -0.00541464. Thus the broad split effect
persists on average even at this granularity, but individual signs vary.

## Independent numerical check and limits

The [alternate numerical audit](neuron_mapping_independent_check.json) uses
newly read raw files, scalar `math.fsum` averages, explicit dense centering
matrices, and symmetric eigendecomposition of the discovery Gram matrix.
It does not call the reporter's numerical helpers. All raw means, paired
loss changes, and accuracies agree exactly. The largest projector difference
is 1.95e-16 and the largest residual difference is 2.26e-17. Fresh scalar
weight-norm sums differ by at most 7.11e-15. All 64 preselected comparator
sets, all 16 discovery choices, and all confirmation ranks and comparator
differences agree. All 138 raw measurement files and the two audited W2
files were rehashed unchanged. This is an independently implemented
calculation on the same measurements, not an independent model replication.

The full Python suite passed all 380 tests after this audit was added. Four
new synthetic tests exercise its numerical agreement and rejection of changed
measurements, reported means, and ranks. The three native GPU test targets
in the reproduction commands also passed. The report retains all prescribed
arms and unsuccessful selections; no forward experiment was repeated to
improve these outcomes.

This probe narrows the interpretation of the earlier coarse map: late MLP
contributions can be addressed and tested more finely, but the present random
64-neuron groups do not yield a robust passage locator or an analytical
decompressor. It does not rule out structured groups or distributed encodings.
The next investigation follows checkpoint differences, as requested; it
must not relabel these known-input forward measurements as weight-only text
extraction.

## Reproduction

Use a new output directory. The native token export must be generated by the
production tokenizer; do not retokenize individual windows with a different
library. The current run's plan, launch identity, measurements, and logs are
under `/tmp/pluto-neuron-mapping.EOgOwe/`.

```sh
bazel build -c opt //scripts/weight_analysis:neuron_probe
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python -m unittest discover \
  -s scripts/weight_analysis -t . -p '*_test.py' -q
bazel test -c opt //scripts/weight_analysis:causal_probe_test \
  //scripts/weight_analysis:token_argmax_test //src/llm/recipes:gpt2_test \
  --test_output=errors --local_test_jobs=1 --nocache_test_results

OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python -m scripts.weight_analysis.neuron_mapping plan \
  --corpus testdata/shakespeare.txt \
  --corpus-token-ids /tmp/pluto-native-corpus.KMY7AZ/shakespeare.tokens.bin \
  --corpus-byte-offsets /tmp/pluto-native-corpus.KMY7AZ/shakespeare.tokens.bin.offsets.bin \
  --tokenizer-dir /home/ubuntu/datasets/tokenizer/gpt2 \
  --protocol research/weight_memorization/NEURON_MAPPING_PROTOCOL.md \
  --checkpoint /home/ubuntu/checkpoints/shakespeare/step_13030 \
  --binary bazel-bin/scripts/weight_analysis/neuron_probe \
  --output-dir /tmp/new-neuron-plan

OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python -m scripts.weight_analysis.neuron_mapping authenticate \
  --plan /tmp/new-neuron-plan/manifest.json
bazel-bin/scripts/weight_analysis/neuron_probe \
  --checkpoint /home/ubuntu/checkpoints/shakespeare/step_13030 \
  --batch_tokens /tmp/new-neuron-plan/passages/batch_tokens.bin \
  --neuron_groups /tmp/new-neuron-plan/neuron_groups.i32 \
  --output_dir /tmp/new-neuron-run --batch_sequences 4
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python -m scripts.weight_analysis.neuron_mapping report \
  --plan /tmp/new-neuron-plan/manifest.json \
  --run /tmp/new-neuron-run/metadata.json --output /tmp/new-neuron-report.json
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python -m scripts.weight_analysis.neuron_mapping_independent \
  --report /tmp/new-neuron-report.json --output /tmp/new-neuron-independent-check.json
```

Never overwrite an earlier plan, run, or report. Changing hashed sources or the
binary after planning invalidates the identity checks. A new plan is required.
