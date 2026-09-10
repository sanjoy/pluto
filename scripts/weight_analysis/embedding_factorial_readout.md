# Validating and reading embedding-factorial evidence

This CPU-only reader independently recomputes every selected token's NLL,
rank, and argmax from all **50,257** stored native FP32 logits. It does not use
the native JSON summaries as its likelihood authority and does not run CUDA.

```sh
/home/ubuntu/.venv/bin/python -m scripts.weight_analysis.embedding_factorial_readout \
  --cases /experiment/word_cases/cases.json \
  --scores /experiment/analysis/factorial_original \
  --patch /experiment/patches/original_with_replacement_rows/step_N/patch.json \
  --expected-row 45 --expected-row 68 --expected-row 303 --expected-row 400 \
  --expected-row 1475 --expected-row 2797 --expected-row 3109 \
  --expected-row 21733 --expected-row 45177 \
  --execution-record /experiment/analysis/factorial_original_execution.json \
  --output /experiment/analysis/factorial_original_readout.json
```

The row selection is explicit and must exactly equal the patch's predeclared
selection. Copied rows that happen to be byte-identical are allowed; the
probe's reported **actually changed** rows must match the checkpoint bytes.
All selected rows must equal the donor, and all other recipient weights and
embedding padding must remain unchanged. Current A/J/donor checkpoint hashes
must match `patch.json` and the patched files must not share source inodes.

For a supplemental suite containing four-token word cases and three-token
shared-piece controls, use the original full supplemental `cases.json` and
either `--case-kind word_next_native` or `--case-kind shared_piece`. The native
probe's packed subset must exactly match those selected frozen cases in their
original order. Its selected-row file must match their exact `scored_rows`.

## Scores and contrasts

Cells are **AA** recipient, **AJ** output dictionary only, **JA** input embedding
only, and **JJ** joint patch. All four keep recipient final LayerNorm fixed.
For log probabilities L, the reported contrasts are:

```text
input       = L(JA) - L(AA)
output      = L(AJ) - L(AA)
interaction = L(JJ) - L(JA) - L(AJ) + L(AA)
joint       = L(JJ) - L(AA) = input + output + interaction
```

A positive value supports the scored event. The same contrasts are computed
for the log odds `log P(Nuveth triple) / P(Exeunt triple)`. These are
log-probability/log-odds effects, not fractions of donor preference recovered.
Absolute NLLs, geometric/arithmetic mean probabilities, token ranks, and
teacher-forced greedy success remain visible: a preference shift can result
from damaging both words.

`groups` provide combined deduplicated and separate overlapping prefix-domain
strata. Sequences are deduplicated by actual causal base prefix and candidate
token sequence, not occurrence labels or domain names. Individual token
predictions are deduplicated by their own causal prefix and target ID. All
original aliases are retained. `word_three` independently deduplicates the
word triple, so cases with the same word/prefix but different following tokens
do not overcount the word. Counts describe evaluation events, not independent
statistical samples or corpus-frequency weighting.

For four-row cases, three distinct quantities are retained:

- Probability of the three-token word sequence.
- Conditional probability of the **one exact supplied following native token**.
- Joint probability of the word and that exact following token.

The following-token score is **not** a sum over all delimiters or all ways for
the word to end. None of these probabilities sums alternative tokenizations.

## Structural and provenance checks

Identical causal prefixes must have identical full-vocabulary logits, even if
future teacher-forced targets or prefix-domain labels differ. Where the causal
prefix contains no actually changed embedding row, the reader verifies
`AA == JA` and `AJ == JJ` byte-for-byte across every logical logit. This is
checked at all selected positions, including the first word token, with
deduplicated exposure counts in `exposure_groups`. The output-only switch must
also leave every **unselected raw-logit column** unchanged; their normalized
probabilities may change through the softmax denominator.

Native metadata must certify diagonal equality specifically over
`selected_rows_full_padded_vocabulary`. This is not a claim that unselected
rows were compared. The CPU reader verifies all metadata assertions and
source identities it can, but does not independently reexecute the native
diagonal checks.

The optional execution record has format `pluto-paired-probe-execution-v1`
and fields `command`, `pid`, `returncode`, `started_utc`, `finished_utc`,
`inputs_before`, `inputs_after`, and `outputs`. Each file record contains
absolute `path`, `bytes`, and SHA-256. Before/after input lists must be identical
and match current files. Required coverage includes all A/J weight files,
J's `patch.json`, frozen original cases JSON, actual packed subset, row file,
and native binary. Outputs must cover native metadata and all four logit files.
Command paths, row count, microbatch, and output directory are checked too.

Without an execution record, the report explicitly declines to establish
execution-time input/binary identity. Even a validated record is recorded
provenance, not independent attestation that a named executable ran. Every
registered file is rehashed before the exclusively created report is written;
outputs inside source checkpoints and existing output paths are rejected.

Run the CPU tests with:

```sh
/home/ubuntu/.venv/bin/python -m unittest scripts.weight_analysis.embedding_factorial_readout_test
```
