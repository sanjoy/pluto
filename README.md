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
bazel test //src/pipeline:tokenized_parquet_test
bazel test //src/pipeline:fineweb_integration_test
```

The Parquet unit test uses a 5 KiB checked-in fixture and crosses a row-group
boundary. Regenerate it after intentional format changes with:

```bash
/path/to/python-with-pyarrow src/parquet/testdata/generate_fixture.py
```

The external integration test discovers every `.parquet` shard in
`PLUTO_FINEWEB_PARQUET_DIR`, reads only two records from each, validates all ten
fields, and checks tokenizer/detokenizer round trips. This exercises the real
multi-gigabyte files without scanning them.
