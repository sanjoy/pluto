# Checkpoint weight history

This CPU-only tool plots how each weight tensor changes across saved training
checkpoints. A "weight" means one complete `weight_<number>.bin` tensor, not a
separate graph for each of its potentially millions of scalar coordinates.

```sh
cd /home/ubuntu/code/pluto
bazel build -c opt //src/llm/experiments/weight_history:weight_history
/home/ubuntu/code/pluto/bazel-bin/src/llm/experiments/weight_history/weight_history \
  --checkpoint=/home/ubuntu/checkpoints/shakespeare_0 \
  --output=/tmp/shakespeare_weight_history.html
```

Open the resulting HTML file in a browser. It contains its own data, styling,
and plotting code: no server, internet connection, external JavaScript library,
or GPU is needed. Progress is printed after each tensor has been analyzed.

Both flags are required. The output's parent directory must already exist, and
the tool refuses to overwrite any existing output, including symlinks. It
finishes analysis and rendering before creating the report; a disk/write error
can still leave a partial new file.

## What the graphs measure

Let `w` be the current tensor, `p` its values at the previous saved checkpoint,
`f` its values at the first supplied checkpoint, and `n` its element count.
Reductions use double precision over the stored FP32 values. All stored
coordinates participate in these metrics, including any padding.

| Metric | Definition |
| --- | --- |
| Change RMS | `sqrt(sum((w - p)^2) / n)` |
| Change L2 | `sqrt(sum((w - p)^2))` |
| Relative change L2 | `change_L2 / sqrt(sum(p^2))`; undefined if the previous norm is zero |
| Maximum absolute change | `max(abs(w - p))` |
| Changed fraction | Number of numerically unequal coordinates divided by `n`; positive and negative zero compare equal |
| Weight RMS | `sqrt(sum(w^2) / n)` |
| Weight L2 | `sqrt(sum(w^2))` |
| RMS distance from first | `sqrt(sum((w - f)^2) / n)` |

The first checkpoint need not be step 0. Its previous-checkpoint change metrics
are missing, not zero, because no predecessor was observed. Its distance from
the first checkpoint is zero. The horizontal axis uses the actual numeric step
IDs. A difference across a gap measures the net change between those saved
snapshots, not each intervening optimizer update; intermediate steps were not
observed. Relative change is undefined when the previous tensor has zero norm.

These are tensor-level summaries of numerical change, not evidence that a
specific training example or concept resides in a particular tensor.

## Input format and validation

The input is a parent directory containing direct `step_<number>` directories.
Checkpoint and weight numbers are sorted numerically. Archives and unrelated
entries are ignored; numeric aliases such as `step_1` and `step_01` are rejected.
Every checkpoint must contain the same nonempty set of numeric
`weight_<number>.bin` files, with matching nonzero sizes divisible by four. Sparse
weight indices are allowed. Duplicate weight-number aliases, malformed
snapshots, read failures, and non-finite tensor values are errors.

The current checkpoint format is raw native FP32, which is little-endian on the
machines used by this project. It has no dtype, shape, or layer-name metadata.
This tool therefore identifies tensors by filename and element count, without
guessing tensor shapes or semantic layer names. Other dtypes or byte orders are
not supported.

Analyze only completed, immutable checkpoints. The tool does not lock files or
make an atomic snapshot of a directory that training is actively modifying.
Each tensor file is read once. Analysis retains the first and previous values
of one tensor at a time, so working memory is approximately twice the largest
tensor's size, plus a 1 MiB input chunk and the compact metric history. Rendering
also retains the final HTML string before writing it.
