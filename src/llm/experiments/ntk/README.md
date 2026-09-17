# Fixed-weight Shakespeare NTK experiment

This tool measures an empirical neural tangent kernel (NTK) of Pluto's existing
GPT-2 recipe. It does not train, replace, or modify the model weights. It can
measure either reproducible seeded initialization or a supplied checkpoint.

This new experimental code and its tests have not yet been human reviewed.

[Jacot, Gabriel, and Hongler (2018)](https://arxiv.org/abs/1806.07572) introduced
the NTK description of parameter-gradient training and studied a constant
limiting kernel at infinite width. This experiment measures a finite network
at one parameter setting. Freezing that measured kernel is a separate modeling
choice, not evidence that this finite GPT-2 model has reached the paper's limit.

## Run

From the repository root, build and measure a small-data experiment with the
full initialized GPT-2 model:

```sh
bazel build -c opt //src/llm/experiments/ntk:shakespeare_ntk
bazel-bin/src/llm/experiments/ntk/shakespeare_ntk \
  --tokenizer_dir=/home/ubuntu/datasets/tokenizer/gpt2 \
  --corpus=testdata/shakespeare.txt \
  --seed=123 --train_examples=2 --eval_examples=1 \
  --context_tokens=16 --stride=1025 \
  --ridge=0.001 --output_dir=/tmp/shakespeare_ntk_initial
```

Use an existing checkpoint and add an unlabeled query:

```sh
bazel-bin/src/llm/experiments/ntk/shakespeare_ntk \
  --checkpoint=/home/ubuntu/checkpoints/shakespeare_0/step_13030 \
  --tokenizer_dir=/home/ubuntu/datasets/tokenizer/gpt2 \
  --train_examples=2 --eval_examples=1 --context_tokens=16 \
  --prompt='[Ex' --ridge=0.001 \
  --output_dir=/tmp/shakespeare_ntk_checkpoint
```

The example checkpoint path is illustrative; supply a checkpoint that exists.
`--output_dir` must not exist, and its parent must already exist. Existing files,
directories, and symlinks are never replaced. The optional `--checkpoint` accepts
one checkpoint directory, not a parent containing several steps.

`--output_token_ids=ID,ID,...` optionally selects logit coordinates. The default
is the sorted union of the training examples' next-token IDs. Explicit IDs must
be unique, in vocabulary, and include every training target. Evaluation targets
outside the subset are flagged as uncovered: they are not treated as all-zero
labels. With only one selected token, selected-token softmax is always one;
inspect logits and kernel values instead of interpreting that as confidence.

## What is computed

For each context x and selected next-token logit c, form one Jacobian row
`J[(x,c),p] = backward_derivative(f_c(x), parameter_p)`. All unique FP32 master
parameters participate. Shared token-embedding/LM-head weights contribute once,
with their already accumulated derivative from both uses. The raw matrix is
`K = J J^T`, including cross-token blocks. There is no division by parameter
count, diagonal normalization, output trace, random projection, or subsampling
of weights. Jacobian columns include padded trainable storage as exposed by the
model; the parameter-block mapping records the precise space being measured.

The full corpus is memory-mapped and tokenized once. Training contexts come
first, then evaluation contexts at `offset + example_index * stride`. Context
plus next-token label must not overlap another example; `stride` must be at least
`context_tokens + 1`. This is a small, disjoint-context holdout inside the given
corpus, not a claim of evaluation on the model's original unseen training split.

Corpus contexts contain 16 real tokens by default. GPT-2 still runs its unchanged
1024-token architecture: remaining positions are EOS-padded, and only the logit
at the last real token is selected. Causal attention prevents those future
padding positions from affecting this readout. A prompt query can contain up to
1024 tokens independently of the corpus context-length option.

The regression target is a one-hot vector over the selected logits, using
squared error. Initial logits are retained, not incorrectly assumed to be zero:

```text
alpha = (K_train,train + ridge * I)^-1 (y_train - f_initial,train)
f_ridge,query = f_initial,query + K_query,train * alpha
```

This is kernel ridge regression around the existing function. `ridge` multiplies
the identity directly: the objective uses the sum of squared errors, not their
mean. It is not full-vocabulary cross-entropy training or autoregressive decoding.
Evaluation/query labels never enter the coefficient fit. All selected contexts
are measured jointly so query/training cross-kernel values use the same Jacobian
definition and parameter ordering.

To also simulate full-batch learning in the frozen feature model, add e.g.
`--kernel_steps=100 --learning_rate=1e-7`. Starting with `alpha=0`, each step is:

```text
alpha -= learning_rate * (f_initial,train + K_train,train * alpha - y_train) / N
```

Here N counts scalar training coordinates (examples times selected tokens), and
the objective is half the mean squared error with no ridge. These are updates to
CPU-side kernel coefficients only. The binary prints a conservative learning-rate
bound from the absolute row sums of the raw kernel. The rate is not silently
adjusted; nonfinite divergence is an error, and finite divergence is still
possible. Ridge and finite-step GD are separate outputs/objectives.

## Precision and cost

`--compute_type=fp16` is the default existing compute policy: activations have
FP32 storage but dense matrix multiplies cast operands to FP16. It is **not**
full-FP32 arithmetic. `--compute_type=bf16` instead uses the recipe's BF16
activation policy. Both have FP32 master parameters and gradients; the measured
Jacobian uses the existing backward kernels, including their rounding and
straight-through conventions for reduced-precision operations. It is an empirical
backpropagation kernel, not an exact derivative of the discontinuous quantized
machine function or a symbolic smooth real-arithmetic network. Use the same
compute policy when comparing reports.

The engine runs a fresh forward/backward per scalar output coordinate. With P
unique parameters and R measured coordinates, the explicit FP32 Jacobian uses
`4 * P * R` GPU bytes; forming the Gram matrix costs O(P R^2), and its host storage
is O(R^2). The unmodified 1024-token forward/backward cost remains even for short
real contexts. This is deliberately a tiny-example research experiment, not an
efficient way to train on the whole corpus or measure all 50,257 logits.

`--max_jacobian_mib=4096` bounds Jacobian storage only. Weights, forward saved
state, a backup of preexisting gradients, and reduction scratch need additional
GPU memory. Oversized requests fail instead of silently approximating the
kernel. Weights are never written; preexisting gradients are restored after
measurement, including cancellation/error paths where CUDA permits restoration.

## Reports

Each successful run produces:

- `report.json`: self-contained metadata, exact token IDs/window offsets, output
  coordinates and row mapping, model dimensions, precision policy, seed/checkpoint,
  parameter blocks, the raw kernel, initial logits, predictions, and selected-token
  softmax values. Text fields explicitly use C-escaped bytes because a GPT-2 token
  can contain an incomplete UTF-8 sequence. Filesystem paths likewise use
  `checkpoint_escaped_bytes`, `tokenizer_directory_escaped_bytes`, and
  `corpus_path_escaped_bytes`; C-unescape the parsed JSON string to recover the
  original bytes without assuming that a Linux pathname is valid UTF-8.
- `kernel.csv`: the same unnormalized square kernel, with numbered row/column IDs.
- `predictions.csv`: one row per sample/output coordinate, including target
  coverage, original/ridge/optional-GD scores and conditional softmax values.

Softmax values normalize **only over selected output IDs**, never over the full
vocabulary. They cannot be used as ordinary GPT-2 token probabilities, perplexity,
or evidence of text memorization. Console metrics likewise report selected-class
accuracy and mean squared error only on examples whose targets are covered.

## Validation

Tests include analytical finite-network Jacobians and regression checks. A full
GPT-2/Shakespeare integration test compares the GPU Gram matrix with independently
collected parameter gradients and CPU dot products, including tied-weight
deduplication, cross-class entries, symmetry, and positive-semidefinite checks.
CPU-only tests cover corpus selection, row mapping, byte escaping, overflow,
regression, and report-directory no-overwrite behavior. Supply the downloaded
tokenizer to run the full-model integration test:

```sh
bazel test -c opt //src/llm/experiments/ntk/... \
  --test_env=PLUTO_GPT2_TOKENIZER_DIR=/home/ubuntu/datasets/tokenizer/gpt2
```

To check the new GPU kernels with Compute Sanitizer, run the built
`gram_kernel_test` and `empirical_ntk_test` executables with
`compute-sanitizer --tool memcheck --report-api-errors explicit --error-exitcode 1`.
The `explicit` setting keeps returned CUDA API failures and memory errors enabled
while excluding extended driver diagnostics. With the default `extended` setting,
this machine reports a host-memory-pool accessibility diagnostic inside the
existing `Executor::Create`, even though that CUDA API call returns success.
