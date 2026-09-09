# Phrase-forward evidence and reproduction

This is the explicitly requested **model-forward investigation**, separate from
the older weight-only checkpoint-delta extraction experiments. No training
corpus was used as an analysis input. Production layers were not modified.

Read [the short report](PHRASE_FORWARD_TRACE.md) and
[all intermediate readouts](PHRASE_FORWARD_READOUTS.md).

## Evidence

The retained [evidence archive](phrase_forward_evidence.tar.gz) contains:

- Original native run plan/result: exact command, current Git HEAD, hashes of
  all 100 checkpoint files, tokenizer files, binary, native probe sources,
  BUILD file, and production C++ sources; input hashes checked again afterward.
- Both native manifests: exact dtypes/shapes, file roles, checkpoint identity,
  tokenization, byte checks, and all intervention coordinates.
- Both complete analysis reports and numerical arrays: 17 native readouts,
  per-position predictions/ranks/probabilities, all 64 head and 16,384 neuron
  accounting contributions, all reconstructed attention/source contributions,
  CPU comparisons, and all 94 intervention results.
- The prospectively saved **neuron-selection plan**: for each position, the
  most positive/negative B7 neuron under the fixed final-margin accounting,
  plus the corresponding B0 extremes at position 4. Deduplication leaves 14
  neurons. These were selected from this prompt, not held-out discoveries.
  There was no neuron exactly inactive at all seven positions, so no inactive
  control was claimed. Clean replay supplies the no-change control.
- Two independent audits, with their executable audit sources. The native
  audit checks raw dumps without using the analysis implementation; the second
  independently checks the calculations and selected-neuron results.

The archive contains 31 verified payload files, occupies 2,425,277 bytes, and
has SHA-256 `303542fbb470e64665e8ab0478235969036635e0a951f897f0c965c7c3994a46`.
Its embedded manifest identifies every payload; all archived bytes were
checked against the originals after creation.

Raw native tensors/logits remain in
[/tmp/pluto-phrase-trace.z028c66l/native](/tmp/pluto-phrase-trace.z028c66l/native)
and [native_neurons](/tmp/pluto-phrase-trace.z028c66l/native_neurons).
These large raw files are **not** in the archive; their hashes are. Temporary
files may eventually be removed, so the commands below regenerate them.
The live full reports are
[analysis.json](/tmp/pluto-phrase-trace.z028c66l/analysis_v2/analysis.json) and
[the neuron analysis](/tmp/pluto-phrase-trace.z028c66l/analysis_neurons_v2/analysis.json).
These final reports consistently subtract saved FP32 logits in FP64. The
superseded reports rounded intervention differences to FP32 first (maximum
extra error `2.39e-7`); correcting this changed no reported conclusion.

The independent native audit checks 217 predeclared inputs and 185 native
outputs before and after inspection. Stage export comprises 84 tensors plus
the token-ID dump: 67 tensors directly retained from the original forward,
and 17 production-kernel replays (embedding lookup and 16 output projections).
Replay inputs/weights are exact, and every reconstructed residual addition is
byte-checked against the original forward. Final-logit parity includes all
50,272 physical columns; probabilities exclude the 15 padded columns.

Attention probabilities are recomputed in FP64 from captured native Q/K.
The predicted context differs from native BF16 context by 0.138–0.160% relative
L2 across blocks. Largest source-sum versus native-head margin difference is
0.001525. These numerical differences remain visible in the results.

## Reproduce

Run from `/home/ubuntu/code/pluto`. Output directories must not already exist.
The native binary reads a checkpoint; it never trains or writes checkpoint
files. It restores every intervention in device memory and verifies bytes.

```sh
bazel build -c opt //scripts/weight_analysis:phrase_probe
phrase_run_dir=$(mktemp -d /tmp/pluto-phrase-reproduction.XXXXXX)

bazel-bin/scripts/weight_analysis/phrase_probe \
  --checkpoint=/home/ubuntu/checkpoints/shakespeare/step_13030 \
  --tokenizer=/home/ubuntu/datasets/tokenizer/gpt2 \
  --output_dir="$phrase_run_dir/native" \
  --prompt='to be or not to be,' \
  --interventions=true \
  --ablate_neuron=7:1504,7:1889,7:147,7:1870,7:464,7:712,7:1170,7:1573,7:377,7:323,7:254,7:1301,0:371,0:580

OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
  /home/ubuntu/.venv/bin/python -m scripts.weight_analysis.phrase_trace_analysis \
  --native-directory "$phrase_run_dir/native" \
  --checkpoint-directory /home/ubuntu/checkpoints/shakespeare/step_13030 \
  --tokenizer-directory /home/ubuntu/datasets/tokenizer/gpt2 \
  --output-directory "$phrase_run_dir/analysis"
```

The recorded experiment ran the 80 standard arms and 14 neuron arms separately,
selecting neurons after inspecting baseline accounting. The reproduction command
combines those now-fixed arms. Both original baselines match exactly.
For just the original trace and parity checks, omit both intervention flags.

## Tests

Final verification: **511 Python analysis tests passed**, including the
independent scalar CPU oracle, per-term accounting, schema corruption, and a
small end-to-end analysis. Both optimized native test targets freshly passed:

```sh
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
  /home/ubuntu/.venv/bin/python -m unittest discover \
  -s scripts/weight_analysis -p '*_test.py' -t .

bazel test -c opt \
  //scripts/weight_analysis:phrase_probe_test \
  //scripts/weight_analysis:causal_probe_test \
  --test_output=errors --nocache_test_results
```

All 119 cells in the written 17-by-7 readout table were also checked against
the native analysis and independently against raw logits.
