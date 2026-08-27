# Pluto data utilities

This repository contains small C++ libraries for the local FineWeb-Edu data
pipeline. They use Bazel/Bzlmod, C++17, Abseil status types and containers, and
GoogleTest.

## GPT-2 tokenizer libraries

The public targets are:

- `//src/tokenizer:tokenizer` — `pluto::tokenizer::Gpt2Tokenizer`
- `//src/tokenizer:detokenizer` — `pluto::tokenizer::Gpt2Detokenizer`

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

The public target `//src/tokenized:document_file` provides
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

The `//src/tokenized:fineweb_converter` library composes this format with the
GPT-2 tokenizer and the projected-text Parquet reader. The
`//src/tokenized:tokenize_fineweb` binary discovers every `.parquet` file in
an input directory and writes a matching `.tokenized` file. For example,
`000_00000.parquet` becomes `000_00000.tokenized`.

```bash
bazel run //src/tokenized:tokenize_fineweb -- \
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

bazel test //src/tokenizer:tokenizer_test
bazel test //src/parquet:fineweb_parquet_reader_test
bazel test //src/tokenized:document_file_test
bazel test //src/tokenized:fineweb_converter_test
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
