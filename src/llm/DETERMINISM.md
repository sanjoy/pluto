# Deterministic training and inference

## Contract

With identical inputs, configuration, random seeds, and initial state, the
provided layers, AdamW, dataset iterators, and trainer reproduce numerical
results **bit for bit on the same hardware and software stack**. The stack
includes the GPU architecture, CUDA driver/runtime, compiler options, executable,
and C++ standard library. Both BF16 and the legacy FP16 compute path are covered;
FP8 remains explicitly unsupported.

The contract assumes valid buffers, finite model inputs/parameters, and the
normal ordered use of each model on its owning `cuda::Executor`. Concurrent
mutation of a model, custom nondeterministic layers/callbacks/iterators, changing
input files during a run, and hardware errors are not covered. Initialization
and sampling use standard-library distributions whose exact algorithms may
differ between standard-library versions. CUDA/compiler changes can also change
floating-point instruction selection. No cross-stack bitwise promise is made.

For training, compare matching completed steps. Use a fixed `--steps` cap and
omit `--training_seconds` when the final checkpoint must match: elapsed wall
time is deliberately not deterministic. The whole configuration matters,
including batch shape, evaluation schedule, corpus split, and optimizer options.
Timestamps, timings, addresses, and concurrent conversion progress messages are
not numerical model outputs and need not match.

Checkpoints save weights only. They do not save AdamW moments, optimizer bias
correction counters, dataset position, or random-engine state. Identical
warm-start invocations replay each other; they are **not** equivalent to one
uninterrupted training invocation. That would require full training-state
checkpoints, which are a separate feature.

## Inference

`SelectNextToken` validates logits and accepts every finite temperature >= 0.
At zero it selects the maximum logit, with the lowest token ID winning ties,
and does not advance the generator. At positive temperatures it uses stable
softmax weights and the caller's explicit `std::mt19937`. Equal engine state
and logits produce equal tokens; the maximum retains weight one even for a
very small temperature. Negative-infinity masks are supported; NaNs,
positive infinities, and completely masked vocabularies are rejected.

The CLI initializes its sampling generator from `--seed`. A one-shot prompt
therefore replays in a new invocation at any supported temperature. Interactive
inference keeps one generator for the whole session: repeat the same sequence
of prompts to replay the same session. It does not reseed on every prompt.
The seed offset is performed in the unsigned engine type, avoiding signed
overflow for `--seed=2147483647`.

## Audit and changes

Floating-point addition is not associative. An atomic addition protects against
lost writes but does not fix their order; tiny gradient differences can grow
through AdamW and subsequent training. The audit found floating-point atomics
in token/position embedding backward and attention backward. They were removed,
not hidden behind a testing-only or slow-mode flag.

| Path | Reduction/ordering policy |
| --- | --- |
| Token embedding backward | Radix-sort unique `(token_id, input_row)` integer keys. One owner per token/width tile gathers gradients in ascending original row order, starting with the existing tied-head gradient. Unvisited rows stay untouched. |
| Position embedding backward | One owner per position/width tile visits sequence rows in ascending order, including partial contexts. |
| Attention backward | Query-owned dQ pass followed by key-owned dK/dV pass; each output has one writer and fixed key/query order. Three FP32 statistics per row/head connect the two passes. |
| Dense, LayerNorm, GELU, residuals, SAE, losses | Exclusive output tiles and fixed loops; no floating-point scatter atomics. Forward and backward were both inspected. |
| AdamW and checkpoint weight lists | Deduplicate with hash-set membership checks while traversing the ordered layer vector. Never iterate hash buckets to decide numerical update order. |
| Dataset and initialization | Private, explicitly seeded engines; resets replay the iterator. Evaluation can use a separate iterator, so it does not consume training randomness. |
| Tokenizer and shard conversion | BPE rank ties follow input order; caches are lookups only. Independent file outputs use sorted shard names, irrespective of worker scheduling. |
| Checkpoint discovery | Numeric step order, then shortest filename and lexical order. Equivalent names such as `step_7` and `step_07` cannot depend on filesystem enumeration order. |

Embedding scratch is two uint64 keys per input row plus CUB's sorting workspace;
the gather performs O(rows * width) additions instead of scanning the entire
vocabulary for every row. The sort operates on integer keys and is enqueued on
the same explicit executor stream. Attention scratch is
`3 * sizeof(float) * rows * heads`, not a quadratic attention matrix. Each
gradient tile has one writer; scratch allocation/free is stream-ordered.
Highly repeated embedding IDs limit gather parallelism, while attention does
extra recomputation to obtain fixed-order dK/dV. Neither path adds host transfers
or host synchronization to backward.

An additional Compute Sanitizer pass exposed out-of-bounds shared-memory writes
in the generated SAE loss decoder-gradient kernel (the ordinary tests passed).
Its combined reduction/broadcast was replaced by a per-feature scale reduction
and a separate pointwise multiplication. This preserves fixed reduction order,
uses only `feature_dim` extra FP32 elements, and avoids both the problematic
layout expansion and repeated scale reductions for every decoder input tile.

## Regression coverage

- `embedding_test`: exact scalar row-order checks and repeated byte comparisons,
  FP16/BF16, shared LM-head gradients, consecutive backward calls, nonzero initial
  accumulators, untouched signed-zero/padding rows, partial contexts, unique
  tokens, and 10,240 identical tokens.
- `attention_test`: exact repeated forward/backward bytes plus CPU-reference
  comparisons, both supported types, head dimensions 16/32/48/64, contexts
  1/3/17/33/128, multiple sequences, and uniform/nonuniform attention scores.
- `sampling_test`: seeded positive-temperature replay, greedy ties, unchanged
  RNG at zero temperature, masked logits, extreme values, and invalid options.
- `dataset_test`: independent same-seed iterators replay every input and target
  despite unrelated iterator activity.
- `checkpoint_test`: equal-number checkpoint aliases resolve identically under
  reversed directory creation order.
- `determinism_test`: multi-step miniature transformer and SAE training replays,
  including optimizer updates and evaluation, with independently allocated
  models and both supported data types. Both default and statistics-collecting
  SAE modes must produce identical updates. The 48-row, 32-input, 96-feature
  shape is also covered by the SAE CPU-reference sweep.

Run the complete optimized suite with the local tokenizer and Parquet corpus:

```sh
bazel test -c opt //src/... \
  --test_env=PLUTO_GPT2_TOKENIZER_DIR=/path/to/tokenizer/gpt2 \
  --test_env=PLUTO_FINEWEB_PARQUET_DIR=/path/to/raw/sample/10BT \
  --cache_test_results=no --local_test_jobs=1 --test_output=errors
```

The checks use byte comparisons, not only tolerances on final losses. Scalar
reference tests remain complementary: a reproducible answer must also be
numerically correct.

The optimized 42-target suite passes. Additional checks reported zero errors:
Compute Sanitizer `memcheck` on `determinism_test`, `embedding_test`,
`attention_test`, and `sparse_autoencoder_reference_test`, and `initcheck` on
`determinism_test`. For example, after building the test binaries:

```sh
compute-sanitizer --tool memcheck --error-exitcode=1 \
  bazel-bin/src/llm/determinism_test
compute-sanitizer --tool initcheck --error-exitcode=1 \
  bazel-bin/src/llm/determinism_test
```

## Full-size replay verification (2026-09-10)

On GH200, CUDA 13.3.73, driver 580.126.20, GCC/libstdc++ 13.3, and an optimized
Bazel build, two independent Shakespeare training processes were run with:

```sh
bazel build -c opt //src/llm/recipes:gpt2_shakespeare_llm
bazel-bin/src/llm/recipes/gpt2_shakespeare_llm \
  --mode=train_model --corpus=testdata/shakespeare.txt \
  --tokenizer_dir=/path/to/tokenizer/gpt2 --seed=17 \
  --batch_size=10 --steps=2 --eval_batches=1 --training_eval_interval=1 \
  --checkpoint_initial --checkpoint_every=1 \
  --checkpoint_dir=/fresh/path/run_a --log_file=/fresh/path/run_a.log
```

The second invocation used separate `run_b` output paths with all numerical
options unchanged. All **300 weight files** matched exactly: 100 tensors each
at steps 0, 1, and 2. Both runs reported training losses 10.7595, 9.32358, and
8.79055 at those steps, with final held-out loss 8.74953. Each training loop
took about 90.8 seconds, including per-step evaluation/checkpointing.

Separate inference invocations from that final checkpoint also produced
identical 16-token completions for `--prompt='To be, or not to be' --seed=17`
at temperatures 0, 0.8, and 1.5. Greedy completion remained identical with
`--seed=2147483647`. These full-size process replays supplement the longer
multi-update miniature-model regression tests; they are not claims about
cross-hardware or cross-build reproducibility.

Full-size SAE verification used that same GPT-2 checkpoint as the frozen
four-block activation generator, the Shakespeare corpus, 4,096 features,
`--seed=17 --batch_size=10 --steps=2`, and per-step checkpoints. Two independent
processes produced identical bytes in all 12 SAE weight files at steps 0, 1,
and 2. Both reported losses per activation of 781.889, 517.667, and 382.582.
The final memory-safe SAE kernel was used for these runs.
