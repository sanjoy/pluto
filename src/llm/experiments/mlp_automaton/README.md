# MLP token-transition automaton

Read high-confidence token continuations directly from isolated MLPs of the
GPT-2 recipe (all eight blocks by default). This is a diagnostic,
**not** the full language model.
For every logical vocabulary token `t`, it computes:

```text
x = E[t]
z = GELU(LN2_Bi(x) @ W1_Bi + b1_Bi)
h = x + z @ W2_Bi + b2_Bi
p = softmax(LN_final(h) @ E.T)   # temperature = 1
```

It adds the directed edge `t -> u` exactly when `p[u] > threshold` (default
`0.75`). There are no position embeddings, attention, or other transformer
blocks. Both learned LayerNorms, biases, the residual connection and tied
embedding/output dictionary are included. A raw MLP matrix alone would not
specify this readout.

Omit `--mlp_block` to scan all eight MLPs and print one combined list. Set an
explicit zero-based index (0 through 7) to scan only that block. For example,
`--mlp_block=3` reads the fourth block's MLP. Each selection applies that
MLP directly to token embeddings using the formula above; preceding
transformer blocks are not executed.

## Run

From the repository root, with the project's CUDA/cuTile toolchain available:

```sh
bazel build -c opt //src/llm/experiments/mlp_automaton

bazel-bin/src/llm/experiments/mlp_automaton/mlp_automaton \
  --checkpoint=/home/ubuntu/checkpoints/shakespeare/step_13030 \
  --tokenizer=/home/ubuntu/datasets/tokenizer/gpt2 \
  --corpus=testdata/shakespeare.txt \
  --test_fraction=0.1 \
  --output_dir=/tmp/mlp-automaton \
  --batch_size=256 \
  --threshold=0.75 \
  --samples=50 \
  --max_tokens=16 \
  --seed=17 \
  --start_tokens=1475,3109,68
```

The checkpoint and output directory are required; tokenizer defaults to the
relative path `datasets/tokenizer/gpt2`. The tokenizer must match the checkpoint
and have 50,257 tokens. This recipe uses model width 512, MLP width 2,048 and
the eight-block GPT-2 checkpoint layout. Each block readout reads nine FP32
tensor files:

| Role | File indices | Physical shapes |
| --- | --- | --- |
| Token embedding / tied output dictionary | 0 | `[50272,512]` |
| Block i MLP LayerNorm scale and bias | `8 + 12*i`, `9 + 12*i` | `[512]`, `[512]` |
| First projection and bias | `10 + 12*i`, `11 + 12*i` | `[512,2048]`, `[2048]` |
| Second projection and bias | `12 + 12*i`, `13 + 12*i` | `[2048,512]`, `[512]` |
| Final LayerNorm scale and bias | 98, 99 | `[512]`, `[512]` |

Here `i` is `--mlp_block`. The embedding and final LayerNorm stay shared
across block selections. For block 7 the selected MLP files are 92 through 97.
Invalid indices are rejected before creating the output directory.

All required file sizes and finite values are checked before weights are
uploaded. Extra files are ignored. No checkpoint file is modified. Output must
be a **new** directory, preventing accidental replacement of a previous graph;
a failed run can leave partial output there.

## Checkpoint histories

Pass a parent directory instead of one checkpoint to analyze every direct
`step_N` subdirectory, once each, in increasing **numeric** step order:

```sh
bazel-bin/src/llm/experiments/mlp_automaton/mlp_automaton \
  --checkpoint=/home/ubuntu/checkpoints/shakespeare_0 \
  --tokenizer=/home/ubuntu/datasets/tokenizer/gpt2 \
  --corpus=testdata/shakespeare.txt \
  --test_fraction=0.1 \
  --output_dir=/tmp/mlp-history \
  --samples=50 \
  --start_tokens=1475,3109,68
```

Omit `--mlp_block` to inspect all eight blocks at each checkpoint; supplying
it still selects just that block. A directory containing `weight_0.bin`
is treated as a single checkpoint, preserving the existing behavior and
output layout even if its directory name is not `step_N`.

The combined history is printed on stdout and saved to
`combined_history.txt`. For example, this illustrates the output format:

```text
" Exeunt all":
  Chkpt 100 - 300 — block 3,5,6
  Chkpt 400 - 500 — block 3,5,7
```

Each text appears once, sorted by its exact decoded bytes. A range combines
consecutive **analyzed checkpoints** with exactly the same sorted block set.
A missing text or a changed block set breaks the range. Singletons print
`Chkpt 100 — block 3,5,6`. Thus, if the saved steps are 100, 200, and 400,
`Chkpt 100 - 400` covers those three snapshots only; it makes no claim
about unsaved training iterations. Absence at step 200 would split that
range even if steps 100 and 400 have identical block sets.

`--samples` is the candidate count **per block, per checkpoint**. The report
includes the union of corpus-matching sampled texts from the entire run,
including explicit starting tokens. For every reported text, membership is
checked against complete walks that **traverse at least one edge** in **all**
scanned graphs, not just the checkpoints/blocks that happened to sample it.
This recovers earlier matches for a text first sampled at a later checkpoint
and avoids treating sampling variation as a disappearance. Paths still obey
`--max_tokens`; prefixes of longer complete walks and zero-edge vocabulary
lookups do not qualify, and edges from different checkpoints or blocks are
never joined.

The root output directory contains:

- `combined_history.txt`: the human-readable ranges shown above.
- `combined_history.json`: format `pluto.mlp_automaton.history.v1`, an
  explicit `analyzed_steps` list, and each text's lossless `bytes_hex`,
  escaped display text, and `ranges` with `first_step`, `last_step`,
  and `mlp_blocks`.
- `metadata.txt`: shared input paths, steps, scan settings, and merge rules.
- `step_N/`: that checkpoint's original graph, samples, combined paths,
  and metadata. In all-block mode it also contains `block_0/` through
  `block_7/`, just like a single-checkpoint scan.

Discovery snapshots the available directory list at startup and does not
recurse or follow newly arriving checkpoints. Non-step entries, regular
files, and compressed archives such as `step_100.tar.gz` are ignored;
archives must be extracted before analysis. Duplicate numeric aliases
(`step_1` and `step_01`), overflowing step IDs, and an empty checkpoint
parent are errors. A malformed selected checkpoint fails the run rather
than silently bridging a gap; already-written per-checkpoint outputs may
remain, but no completed combined history is written.

GPU inference is performed once per selected block/checkpoint, reusing the
readout and decoded vocabulary. Only one checkpoint's full graphs are held
at a time; the accumulator retains compact text/block membership ranges.
All per-checkpoint graphs are still saved, so disk use grows with the number
of checkpoints and selected blocks.

## Training-text filter

Only paths whose complete concatenated bytes occur verbatim in the training
text appear on stdout, in `samples.json`, in `combined_paths.json`, or in
checkpoint histories. Matching is case-sensitive,
includes leading whitespace and arbitrary bytes, and imposes no word
boundaries. The full graph in `graph.json` still records every qualifying
model transition.

Set `--corpus` to the text used for the checkpoint and `--test_fraction` to
the training run's value. Defaults are `testdata/shakespeare.txt` and `0.1`.
The file is memory-mapped and split with the same newline-aligned
`SplitCorpus` helper as training. Held-out-only matches and matches that
cross the split boundary are excluded. Use `--test_fraction=0` when the file
already contains only training text. Checkpoints do not record the corpus
or split, so these flags must match the run.

Filtering happens after sampling and applies to explicit starts as well.
`--samples` counts candidate random starts per block, so fewer paths (possibly
zero) may be emitted; rejected paths are not shortened to a matching prefix.
In history mode this candidate count applies separately to each checkpoint.
Metadata records the corpus, split, training byte count, and candidate and
retained path counts.

## What counts as a continuation

Every reported candidate and every attributed checkpoint/block must have a
complete walk that **actually traverses at least one qualifying edge** (at
least two token IDs). Merely finding the same bytes in the tokenizer's
vocabulary is not evidence that the MLP completes those bytes.

For example, GPT-2 contains a single `ww` token (1383). An isolated node
for that token does **not** qualify as a `ww` continuation. A high-confidence
self-loop `w (86) -> w (86)` does qualify: its two-token cycle decodes to
`ww`. As with other graph paths, a cycle is not proof of a memorized word
or a full-model completion.

The same rule is used for single-checkpoint candidates, cross-block
attribution, and checkpoint histories. History can still report a genuine
earlier continuation that was not randomly sampled in a standalone run,
because it uses the union of candidates across checkpoints. It cannot
backfill membership based solely on an isolated vocabulary token.

Explicit zero-edge starts (including EOS starts) remain diagnostic entries
in `samples.json` when they pass the corpus filter; they are excluded from
combined output and histories. `--max_tokens=1` likewise yields no reported
continuations, even if a starting node has an outgoing edge: the walk has
not traversed it. A walk that reaches EOS through an edge does qualify.
Different tokenizations may still spell the same text, but **each** block's
matching walk must traverse an edge.

Older reports used `block_membership=complete_walk_from_any_start` and could
incorrectly attribute isolated vocabulary tokens. Regenerate those reports;
new metadata records `complete_walk_with_at_least_one_edge`.

## Performance and numerical conventions

The existing production layers perform BF16 activation/matrix math on the GPU
with FP32 master parameters, reductions and logits. Batches must be positive
multiples of 16. The last batch is padded with valid dummy tokens; its extra
rows are discarded. The 15 padded vocabulary columns never participate in
softmax or become graph nodes.

At a threshold at least 0.5, with strict comparison, at most one edge can leave
each token. The tool therefore uses a GPU max-and-exponential-sum reduction,
without storing a second full probability matrix. It copies back just eight
bytes per logical token: the winning token ID and its probability. Only a
batch-sized logits matrix is live, rather than a full vocabulary-squared
matrix. All host/device transfers use page-locked host arrays. The accepted
threshold range is `[0.5,1)`; smaller thresholds would require multi-edge output.

Ties select the lowest token ID. Nonfinite logical logits fail the scan instead
of silently producing an incomplete graph. Probabilities are FP32 and may
round to 1; this does not establish mathematical certainty. Tiny differences
from a CPU readout or between GPU implementations are possible near a threshold.

## Output and sampling

For a single checkpoint with no block flag, the output directory contains `block_0/` through
`block_7/`, each with its own `graph.json`, `samples.json`, and `metadata.txt`.
The root holds `combined_paths.json` and shared `metadata.txt`. With an explicit
block flag (including `--mlp_block=0`), its graph, samples, metadata, and the
combined file are all at the root.

The combined list merges the corpus-matching sampled continuation paths from
all scanned blocks by their **exact decoded bytes**, even when tokenizations
differ. Zero-edge sampled walks do not become combined candidates.
Leading spaces and case stay significant. Entries are sorted by bytes; each
has a sorted, unique `mlp_blocks` array. For example:

```text
" Exeunt all" (MLP blocks: 0, 2, 7)
```

That line is an illustration of the format. To determine the block list,
the tool checks complete walks from **every starting token** in every scanned
graph, using the same `--max_tokens`. Every attribution requires at least one
traversed edge. A block need not have randomly sampled a text to receive
attribution; a singleton vocabulary lookup or prefix of a longer walk does
not qualify.
This extra check does not add unsampled texts to the combined list and never
joins edges from different blocks. Explicit single-block mode only reports
membership in that selected block.

- `combined_paths.json`: each unique text's `bytes_hex`, `bytes_escaped`, and
  `mlp_blocks`; format `pluto.mlp_automaton.combined_paths.v1`.

- `graph.json`: all vocabulary nodes, including isolated tokens, and edges with
  source/target IDs and probabilities. Its format is `pluto.mlp_automaton.v1`.
- `samples.json`: corpus-matching token-ID paths, concatenated bytes and a
  stopping reason.
- `metadata.txt`: checkpoint/tokenizer/corpus paths, selected block and weight
  indices, dimensions, formula, precision, threshold, batch size and sampling
  settings.

GPT-2 tokens can be arbitrary byte fragments, not individually valid UTF-8.
Both graph nodes and sampled paths therefore include lossless `bytes_hex` and
ASCII-only `bytes_escaped` for display. Decode hex to recover exact bytes; do
not interpret the display string as the original text.

Sampling uniformly selects distinct starting nodes with outgoing edges (except
EOS), then follows the unique successor. `--samples=0` skips random starts;
`--start_tokens` adds explicit starts, including nodes with no outgoing edge
for diagnostic purposes. Those zero-edge walks are saved only in
`samples.json`, not the combined reports.
The same seed is used separately for each block. It fixes each sampling
order independently of edge-list order.

Paths stop at `no_edge`, `end_of_sequence`, `cycle`, or `token_limit`. The length
limit counts the starting token. A cycle includes its repeated closing token.
No whitespace or word-boundary heuristic changes the graph, so paths can be
fragments, complete words, phrases, or malformed combinations. A path's edges
are independently evaluated token transitions: later steps do not receive the
earlier tokens as context. Their probabilities are not full-model sequence
probabilities, and a path alone is not proof of memorizing a training example.

For the historical step-13030 checkpoint, the B0 scan found **3,749 edges
among 50,257 nodes**, in about **2.11 seconds** on a GH200 after loading weights
(including the scan's host/device transfers, excluding output serialization).
Before corpus filtering, example sampled paths included `Exeunt all`, `Briefly`, `advisedly`,
`indifferently`, `Fairy Queen`, and `delight in the world`. Malformed samples
such as `God'st thou` demonstrate the limitations of a context-free graph.
A second scan with batch size 512 took about 1.31 seconds and produced a
byte-identical graph. These are observed wall times, not controlled benchmark
comparisons. Generated graph files are not checked into the source tree.

## Tests

```sh
bazel test -c opt //src/llm/experiments/mlp_automaton/... \
  --local_test_jobs=1 --test_output=errors
```

CPU graph tests cover strict thresholds, invalid graphs, all traversal endings,
seeded sampling, exact corpus filtering (including arbitrary bytes and split
boundaries), combined attribution (including unsampled blocks and differing
tokenizations), arbitrary-byte JSON, isolated nodes and I/O failures. GPU
top-transition tests compare against a stable CPU reference across the full
vocabulary, padding, ties, extreme finite logits, nonfinite rows and invalid
inputs. CPU history tests cover numeric checkpoint discovery, invalid names,
duplicate/overflowing step IDs, membership changes and absence boundaries,
late-sampled texts, the isolated-`ww` regression, arbitrary bytes, and text/JSON
output. Both candidate selection and membership exclude singleton lookups,
while preserving real self-loops and edges reaching EOS. Full-path
enumeration is checked against exhaustive small-graph traversal oracles. Model
tests compare the composed readout to independently assembled scalar reference
layers for all eight block selections, check sparse checkpoint loading,
invalid block indices, missing selected weights and atomic validation failures,
and verify complete vocabulary coverage, tail batches and batch-size invariance.
These tests use synthetic small weights and do not require external checkpoint
or tokenizer files. Model and top-transition tests require a GPU.
