# B0 MLP token-transition automaton

Read high-confidence token continuations directly from an isolated first-block
MLP of the GPT-2 recipe. This is a diagnostic, **not** the full language model.
For every logical vocabulary token `t`, it computes:

```text
x = E[t]
z = GELU(LN2_B0(x) @ W1 + b1)
h = x + z @ W2 + b2
p = softmax(LN_final(h) @ E.T)   # temperature = 1
```

It adds the directed edge `t -> u` exactly when `p[u] > threshold` (default
`0.75`). There are no position embeddings, attention, or later transformer
blocks. Both learned LayerNorms, biases, the residual connection and tied
embedding/output dictionary are included. A raw MLP matrix alone would not
specify this readout.

## Run

From the repository root, with the project's CUDA/cuTile toolchain available:

```sh
bazel build -c opt //src/llm/experiments/mlp_automaton

bazel-bin/src/llm/experiments/mlp_automaton/mlp_automaton \
  --checkpoint=/home/ubuntu/checkpoints/shakespeare/step_13030 \
  --tokenizer=/home/ubuntu/datasets/tokenizer/gpt2 \
  --output_dir=/tmp/b0-automaton \
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
the eight-block GPT-2 checkpoint layout. It reads only nine FP32 tensor files:

| Role | File indices | Physical shapes |
| --- | --- | --- |
| Token embedding / tied output dictionary | 0 | `[50272,512]` |
| B0 MLP LayerNorm scale and bias | 8, 9 | `[512]`, `[512]` |
| First projection and bias | 10, 11 | `[512,2048]`, `[2048]` |
| Second projection and bias | 12, 13 | `[2048,512]`, `[512]` |
| Final LayerNorm scale and bias | 98, 99 | `[512]`, `[512]` |

All required file sizes and finite values are checked before weights are
uploaded. Extra files are ignored. No checkpoint file is modified. Output must
be a **new** directory, preventing accidental replacement of a previous graph;
a failed run can leave partial output there.

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

- `graph.json`: all vocabulary nodes, including isolated tokens, and edges with
  source/target IDs and probabilities. Its format is `pluto.mlp_automaton.v1`.
- `samples.json`: token-ID paths, concatenated bytes and a stopping reason.
- `metadata.txt`: checkpoint/tokenizer paths, dimensions, formula, precision,
  threshold, batch size and sampling settings.

GPT-2 tokens can be arbitrary byte fragments, not individually valid UTF-8.
Both graph nodes and sampled paths therefore include lossless `bytes_hex` and
ASCII-only `bytes_escaped` for display. Decode hex to recover exact bytes; do
not interpret the display string as the original text.

Sampling uniformly selects distinct starting nodes with outgoing edges (except
EOS), then follows the unique successor. `--samples=0` skips random starts;
`--start_tokens` adds explicit starts, including nodes with no outgoing edge.
The seed fixes the sampling order, independently of edge-list order.

Paths stop at `no_edge`, `end_of_sequence`, `cycle`, or `token_limit`. The length
limit counts the starting token. A cycle includes its repeated closing token.
No whitespace or word-boundary heuristic changes the graph, so paths can be
fragments, complete words, phrases, or malformed combinations. A path's edges
are independently evaluated token transitions: later steps do not receive the
earlier tokens as context. Their probabilities are not full-model sequence
probabilities, and a path alone is not proof of memorizing a training example.

For the historical step-13030 checkpoint, the default scan found **3,749 edges
among 50,257 nodes**, in about **2.11 seconds** on a GH200 after loading weights
(including the scan's host/device transfers, excluding output serialization).
Example sampled paths include `Exeunt all`, `Briefly`, `advisedly`,
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
seeded sampling, arbitrary-byte JSON, isolated nodes and I/O failures. GPU
top-transition tests compare against a stable CPU reference across full vocabulary,
padding, ties, extreme finite logits, nonfinite rows and invalid inputs. Model
tests compare the composed readout to independently assembled scalar reference
layers, check sparse checkpoint loading and atomic validation failures, and
verify complete vocabulary coverage, tail batches and batch-size invariance.
These tests use synthetic small weights and do not require external checkpoint
or tokenizer files. Model and top-transition tests require a GPU.
