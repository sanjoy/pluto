# Minimal cuTile language model

This directory contains a small differentiable layer API and a byte-level,
scaled-down GPT-2. It is intentionally a learning/toolchain fixture rather than
a production transformer: each 256-row activation holds 16 independent
16-token contexts, and the 256-wide model uses 12 pre-norm transformer blocks
with four 64-wide attention heads. The model performs real cross-entropy
forward/backward passes and learns Shakespeare's next-byte distribution.

`Layer` receives explicit input buffers and an explicit output gradient. Leaf
backward passes compute input gradients and apply their configured SGD update. A
tree-shaped `Tape` stores forward intermediates for nested `ComposedLayer` and
`RepeatedLayer` instances. Parameters are untyped, reference-counted GPU
`Buffer`s and every allocation and kernel launch uses the layer's CUDA stream.
The language-modeling head ties its vocabulary projection to the embedding
table instead of allocating a second parameter.
The Shakespeare-specific topology is assembled directly in the training binary;
the reusable library contains only generic layers and GPU kernels.

`AttentionLayer` is backed by causal FlashAttention implemented in cuTile C++.
Each query streams over its visible keys, updates an online maximum and softmax
normalizer, and accumulates values without materializing an attention matrix.
Backward likewise recomputes probabilities and accumulates Q/K/V gradients.
The compact GPT-2 uses learned token and position embeddings, 12 independently
parameterized pre-norm attention/GELU blocks with residual connections, a final
layer normalization, and a tied language-modeling head. Q/K/V share a
projection within each block and the MLP stays model-width rather than
expanding 4x, keeping the GPU integration test to a few seconds.

The implemented numeric policy keeps FP32 master weights for stable SGD and
rounds operands through FP16 at cuTile compute boundaries. `DataType::FP8` is
present in the model-specification API but returns `Unimplemented`: FP8 needs an
explicit per-tensor scaling policy, and silently casting unscaled logits would
not be a correct implementation.

Run the training integration test non-interactively:

```sh
bazel test //src/llm:shakespeare_llm_test --test_output=streamed
```

Train and then prompt the model (end input with Ctrl-C or Ctrl-D):

```sh
bazel run //src/llm:shakespeare_llm -- --steps=1200
```
