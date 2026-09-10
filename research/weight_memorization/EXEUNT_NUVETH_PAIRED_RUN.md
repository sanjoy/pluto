# Exeunt / Nuveth paired run — September 10, 2026

Status: **running; not yet an analysis result**. The supervisor started at
2026-09-10 02:10:40 UTC. It first creates the shared initialization and runs
three short controls, then trains the original and replacement arms for four
hours each, sequentially on the GH200. Check authoritative `state.json` and
live processes for current status rather than interpreting this note as a
heartbeat.

## Preserved artifacts

Root: `/home/ubuntu/checkpoints/exeunt_nuveth_20260910`

- `manifest.json`: frozen inputs, binary hashes, settings, native alignment.
- `state.json` / `runner.log`: process transitions and validated terminal results.
- `initial/checkpoints/step_0`: shared random initialization.
- `control_a`, `control_b`: identical original-corpus two-update runs.
- `control_replacement`: replacement-corpus two-update run.
- `original/checkpoints/step_N`: original-corpus four-hour arm.
- `replacement/checkpoints/step_N`: replacement-corpus four-hour arm.

Every child retains its initial and final checkpoints, with periodic
checkpoints every 100 updates in the long arms. No checkpoints are deleted.
All runs load the shared step_0 with fresh optimizer state and seed 17.

## Code and validation before launch

- `6a33946`: completed-GPU-update wall-clock training budget, initial/final saves.
- `5d9211f`: paired-run harness, native alignment proof, read-only weight diff,
  protocol, and sampling-replay build target.
- Optimized build of the trainer, native tokenizer exporter, and sampler passed.
- Five optimized regression targets passed: trainer, checkpoint, dataset,
  GPT-2 recipe, and CLI validation. Dataset tests require the tokenizer path;
  the first invocation omitted it, then all five passed with it supplied.
- 40 CPU experiment/comparison/sampler tests passed.
- Real full-model, batch-10 timed smoke test stopped after one completed update
  (41.175 seconds), saved step_0 and step_1, and verified all weights finite.
  Smoke train/test loss changed from 10.7595/10.735 to 9.32366/9.28387 using
  one evaluation batch. This is only a smoke test, not the paired result.

Full configuration and interpretation rules are in
`EXEUNT_NUVETH_PAIRED_PROTOCOL.md`.

## Exact native corpus checks

| Split | Tokens in each arm | Exact replacements | Changed token IDs |
| --- | ---: | ---: | ---: |
| Full | 1,835,163 | 1,028 | 3,084 |
| Training | 1,650,781 | 936 | 2,808 |
| Test | 184,382 | 92 | 276 |

Every replacement occupies the same three token slots in both arms. All other
IDs and byte boundaries are identical; only internal subword boundaries may
differ. Counts are case-sensitive `Exeunt`, not a case-insensitive word count.

## Immutable identifiers

Source commit recorded in the prepared manifest:
`5d9211f54a4365ecbff11be6fcfdf95e61729221`.

SHA-256 values:

```text
manifest.json
e2164761044698b3832e819649c78e6ad33748a2ea3cc40c5376d42e0d255277

frozen optimized trainer
07799aced9622c19c1d91ba00bdd5ac43fc2dfb31bbad7693be159643086e82a

original corpus
4249913cc5998b89bb845fffe1c200dfe6094d4dbe5c0b77b3e806fc680c2f3f

replacement corpus
40f4d3326f1a8abf737adc0dbac8e168044875b166d8d7ab6083fbf25da5d795
```

## Early two-update controls (not the four-hour result)

All controls completed two updates from byte-identical step_0 files. Original
control A ended at training/test loss 8.76844/8.6767; the comparison below uses
the complete FP32 checkpoint tensors, not rounded logged losses.

Across all weights, the original-vs-original repeat difference has L2 norm
0.03123350 (0.01962% of weight norm). Original-vs-replacement has L2 0.04023048,
only 1.288 times larger. Thus global early differences are substantially
confounded by numerical run-to-run variation.

Nevertheless, the **seven largest embedding-row differences** in the
replacement comparison are all constituent tokens of Exeunt/Nuveth. Several
stand well above the corresponding unchanged-data repeat variation:

| Token piece | ID | Replacement delta L2 | Repeat delta L2 | Ratio |
| --- | ---: | ---: | ---: | ---: |
| `unt` | 2797 | 0.00924474 | 0.0000232964 | 396.8 |
| `ve` | 303 | 0.00921043 | 0.00000147955 | 6,225.1 |
| ` Nu` | 21733 | 0.00909185 | 0.00000137529 | 6,610.8 |
| `Ex` | 3109 | 0.00882267 | 0.0000602149 | 146.5 |
| ` Ex` | 1475 | 0.00600423 | 0.0000243522 | 246.6 |
| `th` | 400 | 0.00530857 | 0.0000301525 | 176.1 |
| `e` | 68 | 0.00195537 | 0.000780016 | 2.5 |

These are descriptive ratios from one pair of repeat controls, not statistical
significance tests. The embeddings also serve as tied output weights, so a
changed training target directly affects them; this is not evidence that a
whole word or its contextual mechanism lives in a single row. Short-run repeat
variation cannot bound long-run numerical divergence.

Full evidence is retained in the experiment's `analysis/` directory:
`control_repeat_step2.json`, `control_replacement_step2.json`, and
`control_token_rows_step2.json`, with input checkpoint SHA-256 hashes.

The original four-hour arm started its training budget at **02:17:30 UTC**.
Final steps, elapsed times, losses, long-run weight comparisons, and behavioral
results remain to be measured. Raw weight deltas alone will not be described
as a localized word memory.
