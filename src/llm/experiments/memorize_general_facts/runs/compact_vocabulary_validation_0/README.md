# Compact active-vocabulary validation

Validated on 2026-09-21. The unchanged corpus contains 4,474 distinct GPT-2
text token IDs; adding EOS gives 4,475. Native and independent Hugging Face
tokenization produce the same sorted compact map, with original EOS 50,256
mapped to 4,474. Prompt-only IDs are included. All audit TSVs use original
GPT-2 IDs; compact IDs appear in device batches and embedding indices only.

The eight-block, width-16, one-head, FF-64 model stores exactly 4,475 embedding
rows (71,600 FP32 values), not 4,480. Its tied head adds no weights. The full
model has **114,256 parameters / 457,024 FP32 bytes**, across 100 unique weight
files, down from 847,008 parameters. Logits have 4,480 columns for kernel
compatibility; the five padding columns are masked and add no parameters.

## Tests

All **66 native test targets** passed uncached, including the actual GPT-2
tokenizer and sample parquet integration inputs. GPU tests ran serially.
All **151 Python experiment tests** passed, including converter validation,
raw row preservation, failed/competing publication, refusal to overwrite,
and explicit legacy-vocabulary flags in both search drivers.

New numerical tests compare CPU and GPU lookup/tied-head forward/backward for
FP16 and BF16, including the last active row of exact 4,475-by-16 and 17-by-3
tables, repeated gradient accumulation, and nonzero padded upstream lanes.
The complete eight-block compact BF16 recipe runs through cross-entropy and
backward, checking exact parameter count, tying, logit masks, finite loss and
gradients, and participation by every block. Eleven tokenizer-wrapper tests
cover deterministic remapping, EOS, per-line parsing, invalid IDs, shared
storage, and mapping persistence/validation.

Memory checking passed with zero errors:

```sh
compute-sanitizer --tool memcheck --report-api-errors explicit \
  --error-exitcode=99 bazel-bin/src/llm/embedding_reference_test \
  --gtest_filter=LayerReferenceTest.LookupAndTiedHeadMatchAcrossShapesAndTypes
```

The initial default extended-API run reported the already documented
`HOST/HOST_NUMA pools are always read-write accessible on the HOST` diagnostic
in the unchanged executor setup. The explicit-API mode retains actual CUDA
API failures and kernel memory checking; see the earlier explanation in
`../compact_width_validation_0/README.md`. No executor code was changed.

## Fresh training smoke

`fresh_training/layers_8/` records two updates from scratch with batch 16,
seed 1337, peak LR 0.0006, warmup 100, and evaluation/checkpointing every step.
Context remains 1,024 and the five-token-prompt suffix-plus-EOS objective is
unchanged except for softmax's reduced vocabulary. Command:

```sh
bazel-bin/src/llm/experiments/memorize_general_facts/memorize_general_facts \
  --tokenizer=/home/ubuntu/datasets/tokenizer/gpt2 \
  --corpus=testdata/general_facts_dataset.txt \
  --checkpoint_dir=/home/ubuntu/checkpoints/memorize_general_facts/compact_vocabulary_validation_0/fresh_training \
  --output_dir=src/llm/experiments/memorize_general_facts/runs/compact_vocabulary_validation_0/fresh_training \
  --layers=8 --model_width=16 --attention_heads=1 --feed_forward_width=64 \
  --batch_size=16 --steps=2 --eval_every=1 --checkpoint_every=1 --seed=1337
```

Fresh native reload in `fresh_verification/` matches the training TSV
byte-for-byte. The independent Python verifier checks all 10,002 targets.
Its nonperfect result is expected after two updates; this is a mechanical
test, not a compact-model memorization failure under a meaningful budget.
Steps 0 and 2 each have 100 correctly sized finite FP32 arrays. After the two
updates, embeddings, positions, and every block's QKV and input-MLP matrices
have changed.

## Converted successful checkpoint

`conversion.json` records the source, destination, counts, and mapping and
embedding hashes. The source is the verified full-vocabulary eight-block
width-16 model at step 25,472. The converter selects active embedding rows
without changing any FP32 bits; the other 99 unique weights are unchanged.
It adds the canonical mapping beside the new weight files and leaves the
source intact. No training updates are performed on the converted checkpoint.

Compact checkpoint:
`/home/ubuntu/checkpoints/memorize_general_facts/compact_width_16_layers_8/step_25472`.

Fresh native inference in `converted_verification/` and independent original
tokenization confirm **0 / 10,002 errors, all 1,024 sentences exact**, with mean
loss **0.008918320218642271 nats**. All original prediction IDs are preserved;
the losses change because the softmax denominator excludes inactive classes.
Repeating native inference with batch size 17, including a partial final
batch, gives an identical TSV in `converted_partial_batch/`.

The unchanged full-vocabulary checkpoint also still yields the historical
prediction TSV byte-for-byte with `--compact_vocabulary=false`; see
`legacy_verification/`. This checks the default recipe/kernel compatibility,
not merely successful inference from the compact copy.

## Identities

- Optimized executable:
  `8a0e8ef78d04a8a86e1dd76a29864aff8dd1ef909a6d1c7bbef4c200424672ae`.
- Corpus:
  `814c062e7d7592fe4a4e5b158a37bd37da51700f817c19eb981c1e93d33f245c`.
- Original tokenizer:
  `1fe93b6152957cf9cfd6d89002467f789ce8b3f3e000b3a2edf27c808ddd0b9e`.
- Canonical mapping:
  `ff91c01867df309f841508fbfbcad0af95e20bfdcc5b63e6ae74e803835a9fb6`.
- Fresh smoke prediction TSV:
  `113c08d54c6dc7b1d441a044f3c2b7917616998d0b9a28774f241b62e2e0d549`.
- Fresh smoke step-2 weights, concatenated in numeric file order:
  `c8185f1e3ec0583c750036eb4f26068dde3f6d6925eb5e9eb331ce0510710a7f`.
- Converted successful prediction TSV:
  `99b68ade00802912776a2da001ff13f4c89c290acbd7b09f9392c787ee4a3aec`.
- Converted successful weights, concatenated in numeric file order:
  `d6de325a209e04e7ad61cea1ea6e7a729e46b79afe30d4aa763c19af33d4ab4d`.
- Unchanged source weights, concatenated in numeric file order:
  `cd0ad387e4d383e23e3465d82d28d5f78040252e8a31ea6b74a36774738f1aac`.

These compact results use a changed output vocabulary. They are deliberately
kept separate from the historical full-vocabulary width/depth frontier.
