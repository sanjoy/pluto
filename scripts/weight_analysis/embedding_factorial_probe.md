# Factor the tied embedding patch into input and output effects

`embedding_factorial_probe` compares a recipient checkpoint **A** with an
embedding-row-patched checkpoint **J**. The probe verifies that all 99 other
weight files, including final LayerNorm, are byte-identical. It measures which
logical embedding rows actually differ instead of trusting patch labels, and
rejects changes to padded vocabulary rows.

The four cells are complete teacher-forced models with independently selected
input and output embeddings:

| Cell | Input embedding | Output dictionary |
| --- | --- | --- |
| AA | Recipient | Recipient |
| AJ | Recipient | Patched |
| JA | Patched | Recipient |
| JJ | Patched | Patched |

For each supplied input, native full forwards produce the A/J final residuals.
Native final-LayerNorm/head kernels apply both dictionaries to both residuals,
with recipient normalization fixed. At every **selected row**, both diagonal
results must match their respective full-model logits **byte-for-byte** across
the full padded vocabulary. Nonselected rows are not compared. Mixed
cells are intentionally untied experimental models, not intermediate-layer
logit-lens interpretations. No model weights are modified after loading, and
no backward/optimizer operations run.

This answers a narrower question than locating a word's complete mechanism:
how much of this embedding patch's functional effect comes from consuming
changed input vectors, from classifying with changed output vectors, or their
interaction? It does not isolate changed transformer weights or establish a
unique storage location for the word.

## Inputs and invocation

```sh
bazel build -c opt //scripts/weight_analysis:embedding_factorial_probe
bazel-bin/scripts/weight_analysis/embedding_factorial_probe \
  --recipient /path/to/recipient/checkpoints/step_N \
  --patched /path/to/recipient_with_donor_rows/step_N \
  --batch /path/to/word_cases/packed_cases.bin \
  --rows /path/to/word_cases/scored_rows.i32.bin \
  --rows_per_case 3 \
  --batch_sequences 1 \
  --output_dir /path/to/new_factorial_evidence
```

The packed batch is the existing native probe format: headerless little-endian
int32 `[all inputs][all targets]`, each half `[case_count, 1024]`. Inputs and
targets must agree with next-token prediction within each case. Neither the
native probe nor its readout retokenizes any text.

The row file is headerless little-endian int32 `[case_count, rows_per_case]`.
Its rows are loss/output indices, **not target-input indices**: output row r
predicts `targets[r]`, normally `inputs[r+1]`. For existing `paired_word_cases`
documents, use each case's exact `scored_rows` array in unchanged case order.
Three consecutive rows score the three-token spelling; `--rows_per_case=4`
can additionally score an explicitly supplied following delimiter. Arbitrary
strictly increasing rows are also supported, but a nonconsecutive NLL sum is
not the log probability of one contiguous continuation. The probe does not
infer which rows are meaningful from padding.

Always keep both words' separately teacher-forced candidate cases. The second
and third token predictions condition on their candidate's preceding tokens;
they are not three alternatives predicted from an unchanged original prefix.

Run the reverse direction with the replacement-trained recipient and its
original-row patch. A checkpoint paired with itself is an identity control.
Use new output directories: existing paths are refused and source checkpoints
are never overwritten. A failed run may leave partial new output; only the
last-written `metadata.json` certifies success.

## Outputs and integrity

Each of `AA.logits.f32.bin`, `AJ.logits.f32.bin`, `JA.logits.f32.bin`, and
`JJ.logits.f32.bin` has shape `[case_count, rows_per_case, 50257]` in little-endian
FP32. All logical vocabulary entries are retained; the 15 padded lanes are
excluded. The 160-case, three-row suite writes about 368 MiB of logit evidence.

`metadata.json` records exact row/target IDs, target ranks, argmax IDs, per-token
NLLs, and selected-row NLL sums for every cell and case. Its
`checks.native_diagonal_verification_scope` is
`selected_rows_full_padded_vocabulary`: the accompanying equality flag does
not certify any nonselected rows. NLL is computed with
a stable CPU FP64 log-sum-exp over the full native FP32 vocabulary at
temperature 1. It need not be bit-identical to the native cross-entropy
kernel's different reduction order. Rank/argmax ties prefer smaller token IDs.
Exponentiating the negative sum of consecutive rows gives the probability of
that exact token sequence, not a sum over all tokenizations of its spelling.

The probe compares all loaded model weight bytes against their disk files
before inference, then compares all device weights and disk files against
those snapshots afterward. Frozen input files and optional patch metadata are
also byte-checked. All CUDA transfers use page-locked host storage and the
explicit executor. No cryptographic hashes are computed internally: the
external evidence runner should hash source checkpoints, tokenizer/case
manifest, batch, rows, binary, and output files before accepting the result.

For interpretation, retain absolute NLLs for **both** words, not merely the
fraction of log-odds preference transferred. A patch can shift the preference
while damaging both spellings. At prefixes containing no changed embedding
IDs, causality requires AA=JA and AJ=JJ; this supplies a strong input/readout
labeling control. A preceding teacher-forced word piece can remove that
invariance at later predictions.

## Tests

```sh
# Does not initialize CUDA; safe while controlled training occupies the GPU.
bazel test -c opt //scripts/weight_analysis:embedding_factorial_probe_test

# Full GPT-2 native kernels: run only when the controlled training GPU is free.
bazel test -c opt //scripts/weight_analysis:embedding_factorial_probe_gpu_test
```

CPU tests cover row-file validation, boundaries, full-vocabulary normalization,
ties, extreme finite logits, nonfinite rejection, signed-zero changes, and
padding rejection. The GPU test checks native diagonals, identity controls,
input-exposure causality, output-row specificity, and rejection of incorrect
diagonal evidence. It also compares a two-sequence forward's globally indexed
selected rows with the corresponding one-sequence results, including sequence
boundary rows and a patch exposed only in the second sequence. This exercises
the native helper at both batch sizes; it does not test the CLI's outer
microbatch/output-serialization loop. The GPU test is separate so ordinary CPU
validation cannot interrupt the paired training experiment.
