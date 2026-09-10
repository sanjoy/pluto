# Native source-value attention probe

This diagnostic tests one **source → query/head value pathway**. It is not a
whole-head ablation, a token deletion, or a mask that renormalizes attention.
It does not change checkpoint or model parameter bytes.

For a captured production GPT-2 forward, the probe:

1. Copies the selected block's native packed BF16 QKV tensor.
2. Scales only one source/head's 64 V entries by 1, 0.5, or 0. Q and K, all
   other V entries, and the source position remain unchanged.
3. Runs the real `AttentionLayer` on the modified QKV tensor.
4. Copies **only the selected query/head's** 64 output entries into a copy of
   the original context. It does not pass the whole modified attention tensor
   downstream: doing that would intervene on every later query at once.
5. Runs the production output projection, residual addition, target block MLP,
   remaining blocks, final LayerNorm, and tied language-model head.

The tail is assembled from existing production layers, with independently
allocated copies of the original weights. A constant-output diagnostic branch
feeds the real `ResidualLayer` so its BF16 addition is not approximated on CPU.
The original full batch/context shape is retained; no selected-row truncation
changes the kernel geometry.

## Validity gates

`SourceValueProbe::Create` must reproduce the complete original attention
context and **all 50,272 physical logits at all rows** byte for byte. Each dose
1 also genuinely copies, reruns attention, splices, and replays the tail; it
cannot return saved logits as a shortcut. A future-source value intervention
is allowed by the library and must leave all downstream logits byte-identical.

Every application verifies that modified QKV differs only in the requested V
slice, that the slice equals the requested dose, and that spliced context
differs only in the requested query/head slice. It checks all captured clean
activation inputs against snapshots before and after. Values and logits must
be finite. Half-dose BF16 rounding uses integer round-to-nearest/even including
subnormal ties; dose 1 preserves signed zero and dose 0 writes positive zero.

The executable additionally snapshots every original weight, compares device
and disk bytes before/after, rechecks checkpoint layout and optional metadata,
checks original input bytes, and repeats the entire original model forward.
An optional `--expected_logits` file makes agreement with an earlier trace an
additional prerequisite. External experiment runners must also hash inputs,
source code, executable, and outputs before/after execution; the C++ metadata
is not itself a cryptographic provenance certificate.

## Usage

Preparation status (2026-09-10): the optimized executable and GPU test build;
all six CPU tests pass, including exhaustive finite BF16 half-rounding and
rejection of all NaN/infinity encodings. Nine CLI rejection-path checks also
pass before CUDA initialization. The full Python analysis suite passes 939
tests (`/tmp/pluto-exeunt-source-value-cpu-suite.log`). The first discovery
invocation omitted the package root and failed imports; the successful run
uses `unittest discover -s scripts/weight_analysis -t . -p '*_test.py'`.
**The GPU test and real source interventions have not run.**

The executable accepts one exact token-ID prefix of 1..1024 tokens. Positions
are zero based within that prefix. The default query is its final position.
The suffix is padded with the final token to the production context length.
Source positions must be actual prefix positions; no padding-token source is
silently substituted. The library also supports multiple packed sequences.

After the timed paired training and its already queued analyses finish:

```sh
bazel build -c opt //scripts/weight_analysis:source_value_probe \
  //scripts/weight_analysis:source_value_probe_gpu_test
bazel test -c opt //scripts/weight_analysis:source_value_probe_test
# Run the GPU test, inspect its results, and require no skipped tests before
# collecting evidence. Do NOT execute during either four-hour training arm.
bazel test -c opt //scripts/weight_analysis:source_value_probe_gpu_test
```

For each listed source the CLI runs doses **1, 0.5, 0, 1 again** independently
from pristine captured activations. Output is a new, exclusive directory with:

- Original/padded token IDs, original full QKV/context, and baseline logits.
- Each arm's full modified QKV, full rerun attention, full spliced context,
  and selected query's padded-vocabulary FP32 logits.
- `metadata.json` written only after all integrity checks succeed, including
  full-vocabulary temperature-1 target probability/NLL/rank, argmax, and a
  fixed-rival margin. The rival is the highest-scoring clean non-target token,
  selected once before any interventions.

Raw logits permit independent recomputation. Probability is normalized over
the 50,257 logical token IDs, not the physical padding. The three doses are
not an assumption of linear logit/probability response: downstream operations
and BF16 rounding can produce nonlinear or zero visible changes.

## Predeclared first historical assay

This follows `EXEUNT_ATTENTION_ROUTE_HISTORICAL.md`. It is **not** a result from
the new deterministic paired runs, and no source intervention has run yet.

- Checkpoint: `/home/ubuntu/checkpoints/shakespeare/step_13030`.
- Prefix: `/tmp/pluto-three-words.qPVE2k/run/prefix_step_1342.i32`.
- Block 1, head 2, query 1023 (`e`, ID 68), target `unt` (ID 2797).
- Exact prior-logit gate:
  `/tmp/pluto-three-words.qPVE2k/run/trace_step_1342/logits.f32`.

Source choices use clean attention or fixed position geometry, not new
intervention outcomes:

| Role | Source position | Token ID / text | Clean reconstructed attention |
| --- | ---: | --- | ---: |
| Hypothesized spelling route | 1022 | 1475 / ` Ex` | 0.917294402140 |
| Strongest alternative source | 889 | 508 / ` who` | 0.007131891735 |
| Self/query control | 1023 | 68 / `e` | 0.000119081720 |
| Adjacent nonword source | 1021 | 220 / space | 0.000859188646 |
| Fixed midpoint/distant control | 512 | 40802 / ` Speak` | 0.000000069457 |

These probabilities are FP64 reconstructions from saved BF16 QKV, not recorded
native FlashAttention softmax statistics. All five V vectors are nonzero.
Controls are not attention-magnitude matched, and weak effects may disappear
under BF16 rounding. There is no other occurrence of the same ` Ex` or `e`
token ID in this prefix. Absolute source positions add 324 to local positions.

Exact source hashes, checked against the historical evidence manifest:

| File | SHA-256 |
| --- | --- |
| `prefix_step_1342.i32` | `c9753f5e8c52fc0cba0b00ea77a63f796608d7791abfc55deb227ebfa8d890ce` |
| `trace_step_1342/metadata.json` | `9a10092a4717d5856d5ef059b8133a34a88ccd8da47998462876e0aacee9c937` |
| `trace_step_1342/blocks.1.qkv.bf16` | `5cbf534d4837d7b4f300ed017e4356b2bd1fe1354ce0651aacd742dbbe679ce2` |
| `trace_step_1342/logits.f32` | `e1009035307297e9e41b39addadbafc72ef18601c88148c9f57d631be2e574b1` |
| `analysis_step_1342/analysis.json` | `e610a9b01263013d52d69bd7d07a834c150c280a028815cf11c19627c37018c1` |

The command is predeclared; execute it only after the existing GPU queue:

```sh
bazel-bin/scripts/weight_analysis/source_value_probe \
  --checkpoint=/home/ubuntu/checkpoints/shakespeare/step_13030 \
  --tokens_file=/tmp/pluto-three-words.qPVE2k/run/prefix_step_1342.i32 \
  --block=1 --head=2 --query=1023 --target_id=2797 \
  --sources=1022,889,1023,1021,512 \
  --expected_logits=/tmp/pluto-three-words.qPVE2k/run/trace_step_1342/logits.f32 \
  --output_dir=/path/to/a/new/exclusive/evidence-directory
```

An effect supports a causal contribution of this value pathway to **one `unt`
prediction**. It does not establish full-word encoding, lexical exclusivity,
necessity across contexts, or a unique parameter location. This intervention
acts on a contextual value activation, itself produced by earlier computation
and QKV weights, rather than directly transferring a word-specific weight.
The paired weight-transfer experiments and broader contexts remain necessary.

## Serialized observer and independent readout

The observer/readout addition passes **56 new CPU tests** (32 observer and 24
readout), and the full analysis suite passes **995 tests in 32.185 seconds**.
Log: `/tmp/pluto-exeunt-source-value-queue-cpu-tests.log`. Native CPU validation
targets also pass; the optimized probe and its GPU test have been rebuilt.
No GPU validation or source measurement is implied by these CPU results.

`source_value_followup.py` can queue this assay behind one exact
`paired_branch_followup` process. It records that process's PID, kernel start
ticks, and command; a completion JSON alone cannot unlock GPU work while the
publisher is still alive. Transient observation failures are retried without
restarting or replacing the upstream process.

Before a child may start, the observer revalidates completed paired training,
the determinism gate, the immutable matched-step intervention plan, and all
branch-stage completion artifacts. It also checks that the same single GPU
is free of other compute processes. Then it runs the new native GPU test,
requires every test to complete without skips, and only then executes this
source assay. Failed gates preserve diagnostic logs and do not resume or
alter training.

The historical checkpoint's 100 files and native trace inputs are checked
against the archived manifest and hashed plan before the wait. The observer
copies the executables and implementation sources into its own new directory,
and verifies original and copied input hashes after waiting and before/after
execution. The historical executable is not assumed to equal today's binary.

`source_value_readout.py` independently validates the actual recorded command,
all 20 source/dose arms and nested output files, exact BF16 V scaling, exact
single-query/head splices, identity controls, and full-vocabulary scores.
It requires baseline logits, QKV, and context to reproduce the archived native
trace. It does not claim to independently recompute the counterfactual native
attention or the all-row GPU parity check from exported selected-row logits.
The readout and observer explicitly mark results as historical, not measurements
from the new paired models, and never claim completion of the overall goal.

Source snapshots cover the selected production/probe sources and imported
Python analysis modules, not a full transitive C++/CUDA build-environment
archive. Copied executable hashes identify the actual native programs; runtime
identity/parity gates remain required.
