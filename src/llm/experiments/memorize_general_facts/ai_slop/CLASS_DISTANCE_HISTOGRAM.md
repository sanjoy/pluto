# Minimum distances between A3 output-token classes

AI-generated experiment report; not human reviewed. Experiment date: 2026-09-28.

## Definition

For each unordered pair of different expected output-token IDs `(x, y)`,
compute exactly one value:

```
distance(x, y) = min over a targeting x, b targeting y of ||a - b||_2
```

The vectors are the full 10-dimensional residual after attention 3's residual
addition, before MLP3's input LayerNorm. Pool all scored suffix and terminal EOS
positions across all 1,024 facts. Exclude masked prompt and padding positions.
Classes without a scored target are absent. Do not normalize the vectors.

The histogram weights each class pair once, not every state pair, not class
centroids, and not each token's nearest other class. A separate minimum is reset
for every pair. Exhaustive CPU comparisons subtract coordinates in FP64, sum
squares in FP64, then take the square root only for the minimum. Input values
were captured as BF16 and expanded exactly to float.

## Results

The source checkpoint still predicts all **10,002 / 10,002** scored next tokens
correctly. There are **2,900 observed target classes**, hence **4,203,550**
unordered class pairs. All pair minima are nonzero.

| Statistic | Minimum class-pair L2 distance |
|---|---:|
| Minimum | 0.08140689666549776 |
| 1st percentile | 0.80202814597635808 |
| 5th percentile | 1.0778974486442474 |
| 25th percentile | 1.5623283101167842 |
| Median | 1.9792455322878577 |
| 75th percentile | 2.4940329030312971 |
| 95th percentile | 3.4171023036829893 |
| 99th percentile | 4.1443602700878399 |
| Maximum | 6.556739040496355 |

Closest examples (leading spaces are part of the decoded tokens):

| First class | Second class | Minimum L2 |
|---|---|---:|
| `704: " water"` | `1723: " dry"` | 0.08140689666549776 |
| `846: " true"` | `1014: " original"` | 0.13816202487365151 |
| `77: " an"` | `1644: " Africa"` | 0.18559312745886075 |
| `629: " process"` | `1027: " force"` | 0.19429437394137219 |
| `2836: " planets"` | `2837: " collapsed"` | 0.19840568227321712 |

The standalone HTML contains linear- and logarithmic-distance histograms,
quantiles, and the 50 closest class pairs, including concrete witness sentences,
positions, and full vectors. It needs no network access or JavaScript.

These distances describe raw residual-space geometry; they do not establish
linear separability or predict how large an MLP is needed. Common classes have
more candidate state pairs and therefore more opportunities for a small minimum.

## Reproduce

Build this branch, then run puzzle capture without a training flag:

```sh
bazel build -c opt \
  //src/llm/experiments/memorize_general_facts:memorize_general_facts

bazel-bin/src/llm/experiments/memorize_general_facts/memorize_general_facts \
  --mode=puzzle \
  --puzzle_checkpoint=/home/ubuntu/checkpoints/memorize_general_facts/context27_L4_W10_FF20_matched_0/trial_000_L4_W10_FF20/checkpoints/layers_4/step_89600 \
  --tokenizer=/tmp/pluto-no-residual-20260928-01/inputs \
  --corpus=/tmp/pluto-no-residual-20260928-01/inputs/corpus.txt \
  --layers=4 --model_width=10 --attention_heads=1 \
  --feed_forward_width=20 --context_length=27 --batch_size=32 \
  --output_dir=/tmp/pluto-a3-class-distance-reproduction
```

The output directory must not exist. The original invocation used
`/tmp/pluto-a3-class-distance-20260928-01`, containing `class_distances.html`,
the existing coordinate plots in `puzzle.html`, and `run.txt` provenance.
Generated artifacts are local, not checked in.

This checkpoint is a 4-block, width-10, single-head model with FF width 20,
context 27, and compact vocabulary size 4,475. The provided tokenizer/corpus
directory is the exact historical input snapshot. Their SHA-256 values are:

```
tokenizer.json  8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6
corpus.txt      814c062e7d7592fe4a4e5b158a37bd37da51700f817c19eb981c1e93d33f245c
```

Wall time for capture plus both reports was 1.46 seconds on the GH200 machine;
maximum resident host memory was 384 MiB. Source weights were verified unchanged.

## Validation

All 75 Bazel test targets passed, including 14 new histogram/geometry cases:

```sh
bazel test -c opt //... --local_test_jobs=2 --test_output=errors \
  --test_env=PLUTO_GPT2_TOKENIZER_DIR=/home/ubuntu/datasets/tokenizer/gpt2 \
  --test_env=PLUTO_FINEWEB_PARQUET_DIR=/home/ubuntu/datasets/raw/sample/10BT
```

Tests distinguish class-pair minima from centroid/state-pair statistics and
cover masks, noncontiguous target IDs, duplicate states, witness ties, zero
collisions, nearby large vectors, large finite coordinates, invalid inputs,
HTML escaping, and empty/constant distributions. Both histogram panels retain
exact observation counts, including the maximum-edge bin.

An independent Python `math.dist` check reproduced all 50 reported closest-pair
minima from the captured coordinate plot data. Its decimal coordinates were
first round-tripped to float32, matching the original BF16-expanded values.
Both 60-bin histograms sum to exactly 4,203,550 observations.
