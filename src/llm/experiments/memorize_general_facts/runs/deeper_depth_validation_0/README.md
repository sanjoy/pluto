# Deeper GPT-2 configuration validation

Validated on 2026-09-21 after the complete `width_depth_refine_16_deep_long_0`
training driver and both of its independent checks exited. No tests or smoke
runs competed with that training run. These are mechanical correctness and
compatibility checks, **not memorization results for the deeper models**.

The former maximum of eight transformer blocks was a configuration restriction,
not a kernel limit. Explicit nonnegative native depths are now accepted; the
width/depth Python driver accepts positive signed-int32 depths. The default
remains eight. Residual initialization remains `0.02 / sqrt(2 * 8)`, and block
seeds, construction order, widths, vocabulary, context, and kernels are unchanged.
Very large depths can still exceed available memory; accepting a configuration
is not a promise that its allocation or training will fit.

## Test coverage

All **65 native test targets** were executed, without cached test results:

```sh
mapfile -t targets < <(bazel query 'kind(".*_test", //...)')
PLUTO_GPT2_TOKENIZER_DIR=/home/ubuntu/datasets/tokenizer/gpt2 \
PLUTO_FINEWEB_PARQUET_DIR=/home/ubuntu/datasets/raw/sample/10BT \
bazel test -c opt "${targets[@]}" --local_test_jobs=1 \
  --nocache_test_results --test_output=errors
```

All **134 Python experiment tests** also passed. Added coverage includes depths
above eight and int32 bounds, unchanged default grids, simulated train/reload/
audit phases, and evidence reporting for both successful and failed deep trials.

Real recipe tests exercise sixteen blocks at widths 8 and 12 with one head,
four-times-expanded MLPs, BF16 compute, and the full 1,024-position context.
They check finite logits, padded-vocabulary masking, finite gradients for all
196 unique arrays, and nonzero embedding/every-block QKV gradients. Upstream
derivatives include positions 5 and 1,023. Initialization tests compare all 98
shared prefix arrays of eight- and sixteen-block models, the relocated final
norm, and all 194 arrays in the sixteen-block activation generator. Checkpoint
validation requires all 196 files and rejects treating a deeper checkpoint as
an exact eight-block checkpoint.

## Native roundtrips

The rebuilt optimized binary has SHA-256
`e4b8d8b951f3b6dbfb76c5b4c0e63e03cc5d65861e175caff2430a1d8d9299c0`.
The previous pinned binary was
`411377b44fd936a2072c96aafa144f38a4da3e63cbfad559c22a6e98f3b9904c`.

Each directory below contains a complete driver manifest with exact commands,
input/binary hashes, training predictions, fresh-process checkpoint predictions,
and the independent retokenization audit. These deliberately short runs use two
updates, batch 16, seed 1337, learning rate 0.0006, and warmup 100. Each retains
10,002 prediction errors, as expected before meaningful training; they do not
establish a model-capacity boundary.

| Directory | Blocks | Width | Heads | FF | Parameters |
| --- | ---: | ---: | ---: | ---: | ---: |
| `compatibility_16` | 1 | 16 | 1 | 64 | 824,048 |
| `sixteen_blocks_width_8` | 16 | 8 | 1 | 32 | 424,336 |
| `sixteen_blocks_width_12` | 16 | 12 | 1 | 48 | 645,720 |
| `sixteen_blocks_width_12_repeat` | 16 | 12 | 1 | 48 | 645,720 |

Checkpoint roots are under
`/home/ubuntu/checkpoints/memorize_general_facts/deeper_depth_validation_0/`.
All deep step-0 and step-2 arrays have the recipe-derived sizes and finite FP32
values. Token/position embeddings and every block's QKV and input-MLP matrix
change after the updates. The width-12 repeat matches all 196 arrays at both
steps byte-for-byte, as well as every final prediction/loss TSV byte.

Concatenating final weight files in numeric index order gives SHA-256:

- Sixteen blocks, width 8:
  `23c4f824952f5e4841226d8c48032947a11eeba26a56b7c90e677a48d1636d94`.
- Sixteen blocks, width 12 (both runs):
  `58c6ba5d566fb9adbeb02a5d8b34b86493def86288a5533492002f644feb4012`.

## Historical compatibility

The new one-block width-16 run exactly matches all 16 step-0 and step-2 arrays
and the complete prediction TSV from `../compact_width_compatibility_16/`.
The sixteen-block width-8 initialization matches the 26 shared prefix arrays
and relocated final-norm arrays of the earlier two-block width-8 smoke run.

The new binary also reloads the successful one-block width-24 step-29,824
checkpoint. It again produces zero errors and all 1,024 exact sentences, with
the full TSV identical to the original verified output:
`db7a0ab9a42ad4ed9f7d1658d4eba949181e1a3e00b94ca15bc7347c9454a1a7`.
See `trained_success_compatibility_24/` for this native result and independent
audit. These checks support compatibility, but the reporting tool deliberately
keeps different binary hashes in separate matched-protocol groups.
