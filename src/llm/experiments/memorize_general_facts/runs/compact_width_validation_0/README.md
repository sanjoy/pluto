# Compact-width backend validation

Validated on 2026-09-21 with optimized builds and NVIDIA Compute Sanitizer
2026.2.1.0 (build 38334959). This is kernel/recipe validation, not a claim that
any of these smaller models has memorized the fact dataset.

The embedding and position kernels mask all channel-tail accesses and use
ceiling tile counts. AdamW masks short/partial parameter tiles, including its
gradient clear, and validates the element count before conversion to `int`.
Dense, LayerNorm, and attention already masked their tiles; their channel
validators now accept positive logical widths. CPU references follow the same
shape contract. Stored weights are compact: there are no added trainable
channels, LayerNorm divides by the logical width, and attention scales by the
logical head dimension. Existing flat GELU/residual and dense-row alignment
requirements remain unchanged; the recipe's 1,024-token context satisfies them.

## Tests

All 65 native Bazel test targets passed. The first full run exposed two obsolete
MLP-automaton expectations that widths 15 and 31 must be rejected. They were
replaced with actual compact-width vocabulary scans, checking physical embedding
size and exclusion of padded vocabulary rows from softmax. The rerun passed.

Targeted coverage includes:

- Embedding/tied-head and position forward/backward CPU/GPU agreement for
  widths 1, 3, 7, 8, 15, 24, and 33, under FP16 and BF16 compute policies.
- Deterministic embedding segment reductions with widths 1, 3, 8, 15, and 24.
- Rectangular dense matrices with logical widths as small as 1 and odd tails.
- LayerNorm forward/backward at widths 1, 3, 7, 8, 15, 24, and 33.
- Attention forward/backward CPU/GPU agreement and three identical device
  executions for those widths, odd multi-head strides, and width 8 at context
  1,024. Head dimension, not compute-tile padding, sets the softmax scale.
- Three AdamW updates against an independent scalar calculation, checking every
  weight and cleared gradient element for matrices/biases derived from widths
  1, 3, 8, and 17; momentum, bias correction, and weight decay remain active.
- Full GPT-2 forward/backward and exact physical parameter counts for compact
  configurations `(layers, width, heads, FF) = (1,3,1,13), (2,8,1,32), (1,24,3,96)`.
  The default recipe's initialization, tying, and existing tests still pass.

The seven targeted suites were `embedding_reference_test`, `embedding_test`,
`fully_connected_reference_test`, `norm_reference_test`,
`attention_reference_test`, `optimizer_test`, and `recipes:gpt2_test`.

## Memory checking and the retained setup diagnostic

The initial default memcheck invocation returned 99 even though its test passed:
the CUDA runtime emitted the extended diagnostic
`HOST/HOST_NUMA pools are always read-write accessible on the HOST` from the
unchanged executor's host-pool access setup. `optimizer_memcheck.log` preserves
this report. The same diagnostic reproduces in the unchanged executor-only
test; see `unchanged_executor_memcheck.log`. No executor code was changed.

Using `--report-api-errors explicit` keeps actual explicit CUDA API failures and
kernel memory checking enabled but omits extended runtime diagnostics. All five
`*_memcheck_explicit_api.log` files report **0 errors**, with application tests
passing and exit status 0. All used:

```sh
compute-sanitizer --tool memcheck --report-api-errors explicit \
  --error-exitcode 99 --padding 32 --log-file OUTPUT.log \
  bazel-bin/src/llm/TEST_BINARY --gtest_filter=FILTER
```

| Test binary | Filter |
| --- | --- |
| `optimizer_test` | `LayersTest.PartialParameterTilesMatchScalarAdamAcrossSteps` |
| `embedding_reference_test` | `LayerReferenceTest.LookupAndTiedHeadMatchAcrossShapesAndTypes:LayerReferenceTest.PositionEmbeddingForwardAndBackwardMatch` |
| `fully_connected_reference_test` | `LayerReferenceTest.ForwardAndBackwardMatchAcrossShapesAndTypes` |
| `norm_reference_test` | `LayerReferenceTest.ForwardAndBackwardMatchAcrossShapesAndTypes` |
| `attention_reference_test` | `AttentionReferenceTest.SubTileHeadWidthsMatchAndRepeat` |

## Separation from the active coarse experiment

These tests ran in separate processes while the seven-block width-32 coarse
trial was active. Its elapsed time can include test contention and must not be
treated as an isolated throughput benchmark. Its update budget, binary, inputs,
and optimizer state were unchanged. The training executable remained
SHA-256 `edbfb301cb58fa8e67a4b5039379c0dc947dee15858de6bbaad17a6b53988e41`;
inspection of its dynamic dependencies confirmed no shared project libraries.

To run all tests without rebuilding that pinned executable, first checked that
`deps(kind(".*_test", //...))` has no dependency on the memorization binary,
then passed the 65 queried test labels to `bazel test -c opt` with
`--local_test_jobs=1 --test_output=errors`. The GPT-2 tokenizer and FineWeb paths
were supplied through `PLUTO_GPT2_TOKENIZER_DIR` and
`PLUTO_FINEWEB_PARQUET_DIR`. The training executable will only be rebuilt after
the coarse driver finishes; later trials will record the new binary hash.
