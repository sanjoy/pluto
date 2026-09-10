# Pluto

Pluto is a minimalist ML framework for running experiments. It provides small,
composable C++ libraries for building models, training them on the GPU, and
inspecting what they learn. The goal is to keep the implementation easy to
understand and change, with explicit computation and memory ownership.

The repository includes:

- GPU layers, CPU reference implementations, composition, AdamW, reusable
  training/evaluation loops, and weight checkpoints in [`src/llm`](src/llm).
- GPT-2 and sparse-autoencoder experiments in
  [`src/llm/recipes`](src/llm/recipes), including training and inference on
  Shakespeare.
- CUDA execution, device buffers, and page-locked host memory in
  [`src/cuda`](src/cuda), with CUDA/cuTile C++ kernels.
- Tokenization, dataset iterators, and data-preparation utilities in
  [`src/dataset`](src/dataset) and [`src/parquet`](src/parquet).
- Weight-analysis tools in [`scripts/weight_analysis`](scripts/weight_analysis)
  and experiment protocols and findings in [`research`](research).

Pluto uses Bazel/Bzlmod, C++20, Abseil, and GoogleTest. It is an experimental
framework, not a comprehensive collection of ML operators or data formats.

## Running an experiment

The GPT-2 Shakespeare recipe supports `train_model`, `infer_model`, `train_sae`,
and `infer_SAE` modes. For a bounded training run:

```bash
bazel build -c opt //src/llm/recipes:gpt2_shakespeare_llm
bazel-bin/src/llm/recipes/gpt2_shakespeare_llm \
  --mode=train_model \
  --tokenizer_dir=/path/to/datasets/tokenizer/gpt2 \
  --corpus=testdata/shakespeare.txt \
  --batch_size=1 \
  --steps=100
```

The local CUDA toolkit is configured at `/usr/local/cuda` in
[`MODULE.bazel`](MODULE.bazel); the checked-in [`.bazelrc`](.bazelrc) targets
Hopper (`sm_90`). See the [model and recipe guide](src/llm/README.md) for
architecture, numeric policy, checkpointing, inference, and SAE examples, and
the [determinism notes](src/llm/DETERMINISM.md) for reproducibility constraints.
Checkpoints currently save model weights, not optimizer or data-iterator state.

## GPU buffer

The public target `//src/cuda:buffer` provides `pluto::cuda::Buffer`, a small
untyped CUDA allocation:

```cpp
auto executor = pluto::cuda::Executor::Create();
auto buffer = pluto::cuda::Buffer::Allocate(**executor, byte_count);
cudaMemsetAsync(buffer->data(), 0, buffer->size_bytes(),
                (*executor)->stream());
```

`Executor` owns an explicitly created non-default CUDA stream. `Allocate()`
queues `cudaMallocAsync()` on its executor. Copying a
`Buffer` shares its allocation; destroying the final copy queues
`cudaFreeAsync()` on that executor's stream. The caller must keep the executor
valid until all copies are destroyed. A zero-byte buffer retains its executor
but has a null data pointer.

The Bazel library links the CUDA runtime through `@cuda//:cuda_runtime`. Its
GPU test launches a kernel through a shared buffer, queues an asynchronous
device-to-host copy, drops the final reference, and verifies that the
stream-ordered free does not race the earlier work:

```bash
bazel test //src/cuda:buffer_test
```

The `//src/cuda:cutile_test` target is a CUDA Tile C++ toolchain smoke test. It
uses an `__tile_global__` kernel to add two 128-element vectors in
eight-element tiles, with one logical tile block per region:

```bash
bazel test //src/cuda:cutile_test
```

CUDA Tile C++ requires CUDA 13.3 or newer, C++20, and NVCC's
`--enable-tile` option. CUDA targets configure their tile compiler options;
the project-wide C++20 setting keeps host and CUDA translation units on the
same language-mode ABI.

## GPT-2 tokenizer libraries

The public targets are:

- `//src/dataset:tokenizer` — `pluto::tokenizer::Gpt2Tokenizer`
- `//src/dataset:detokenizer` — `pluto::tokenizer::Gpt2Detokenizer`

Both `Load()` methods take a directory supplied by the caller. No tokenizer
location is compiled into the libraries. The directory must contain the
`tokenizer.json` written by Hugging Face `save_pretrained()` for the
`openai-community/gpt2` byte-level BPE tokenizer.

```cpp
auto encoder = pluto::tokenizer::Gpt2Tokenizer::Load(tokenizer_directory);
auto decoder = pluto::tokenizer::Gpt2Detokenizer::Load(tokenizer_directory);
auto ids = (*encoder)->Encode("Hello, world!");
auto text = (*decoder)->Decode(*ids);
```

The immutable parsed model is shared by encoder and decoder instances in the
same process. Encoding has a bounded, thread-safe BPE cache. Encoding and
decoding preserve UTF-8 input byte-for-byte and recognize GPT-2's EOS token.

## Tokenized-document files

The public target `//src/dataset:document_file` provides
`pluto::tokenized::DocumentFileReader` and `DocumentFileWriter`. The format
contains one length table followed by packed token IDs:

```text
uint32_le document_count
uint32_le document_lengths[document_count]
uint16_le token_ids[sum(document_lengths)]
```

All integers are explicitly little-endian. The reader validates that the
length table and payload exactly describe the file, builds document offsets
once, and uses `pread()` so independent reads can run concurrently. The writer
keeps only the length table and a 1 MiB payload buffer in memory. It writes
through a temporary file in the destination directory and atomically publishes
the result only when every declared document has been supplied.

The `//src/dataset:fineweb_converter` library composes this format with the
GPT-2 tokenizer and the projected-text Parquet reader. The
`//src/dataset:tokenize_fineweb` binary discovers every `.parquet` file in
an input directory and writes a matching `.tokenized` file. For example,
`000_00000.parquet` becomes `000_00000.tokenized`.

```bash
bazel run //src/dataset:tokenize_fineweb -- \
  --input_dir=/home/ubuntu/datasets/raw/sample/10BT \
  --output_dir=/home/ubuntu/datasets/tokenized/10BT \
  --tokenizer_dir=/home/ubuntu/datasets/tokenizer/gpt2
```

By default the binary converts multiple shards in parallel, uses 1,000-row
batches, and skips outputs that already exist. Use `--jobs=N` to control
parallelism, `--batch_size=N` to tune memory use, or `--overwrite` to replace
completed outputs. Partial outputs are never published.

## FineWeb Parquet reader

The public target `//src/parquet:fineweb_parquet_reader` provides
`pluto::parquet::FineWebParquetReader`. `Open()` accepts a file path and reads
only the footer. `ReadRows()` materializes the ten-column FineWeb record, while
`ReadTextRows()` reads only the text column for a faster tokenizer pipeline.

This is intentionally not a general Parquet library. It supports the physical
layout found in the FineWeb-Edu 10BT sample shards:

- flat optional columns with no actual nulls;
- `BYTE_ARRAY`, `DOUBLE`, and `INT64` physical types;
- Snappy-compressed data-page v1;
- PLAIN dictionaries and RLE dictionary indices;
- the exact `text`, `id`, `dump`, `url`, `file_path`, `language`,
  `language_score`, `token_count`, `score`, and `int_score` schema.

Unsupported layouts fail with an `absl::Status` instead of being guessed at.
The reader uses footer offsets plus `pread()`, avoids a shared seek position,
and skips unrequested columns.

## Tests

The tokenizer paths are injected through environment variables, which the
checked-in `.bazelrc` forwards into Bazel's test environment:

```bash
export PLUTO_GPT2_TOKENIZER_DIR=/path/to/datasets/tokenizer/gpt2
export PLUTO_FINEWEB_PARQUET_DIR=/path/to/datasets/raw/sample/10BT

bazel test //src/dataset:tokenizer_test
bazel test //src/parquet:fineweb_parquet_reader_test
bazel test //src/dataset:document_file_test
bazel test //src/dataset:fineweb_converter_test
bazel test //src/pipeline:tokenized_parquet_test
bazel test //src/pipeline:fineweb_integration_test
```

The Parquet and converter tests use a 5 KiB checked-in fixture and cross a
row-group boundary. Regenerate it after intentional format changes with:

```bash
/path/to/python-with-pyarrow src/parquet/testdata/generate_fixture.py
```

The external integration test discovers every `.parquet` shard in
`PLUTO_FINEWEB_PARQUET_DIR`, reads only two records from each, validates all ten
fields, and checks tokenizer/detokenizer round trips. This exercises the real
multi-gigabyte files without scanning them.
