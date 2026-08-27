# Minimal cuTile language model

This directory contains a small differentiable layer API and a byte-level
bigram language model. It is intentionally a learning/toolchain fixture rather
than a production transformer: fixed 256-by-256 tensors make compilation and
GPU test runtime predictable, yet the model performs real cross-entropy
forward/backward passes and learns Shakespeare's next-byte distribution.

`Layer` receives explicit input buffers and an explicit output gradient. Leaf
backward passes compute input gradients and apply their configured SGD update. A
tree-shaped `Tape` stores forward intermediates for nested `ComposedLayer` and
`RepeatedLayer` instances. Parameters are untyped, reference-counted GPU
`Buffer`s and every allocation and kernel launch uses the layer's CUDA stream.
`SimpleLlmConfig::dense_repetitions` controls the number of repeated composed
dense blocks in the factory-built predictor.

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
