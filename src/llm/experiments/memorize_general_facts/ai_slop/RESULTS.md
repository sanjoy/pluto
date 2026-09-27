# General-facts memorization results

Paths under `runs/` below are relative to the parent experiment directory and
refer to local, generated artifacts. They are ignored by Git and are not included
in a fresh clone; this report retains their provenance.

This is a depth search within the fixed-width GPT-2 family described in
[README.md](README.md), not a claim about the smallest possible language model.
The approved task supplies the first five GPT-2 tokens of each sentence and
requires every remaining top-1 token prediction, including EOS, to be correct.
All 1,024 sentences are independently right-padded to 1,024 positions. Prompt
targets and padding are excluded from training loss and evaluation.
All 1,024 five-token prefixes also preserve their exact token IDs when decoded
to text and retokenized, so these prompts can be supplied as ordinary text.

**One transformer block suffices.** Every tested depth from eight through one
achieved exact memorization and passed independent checkpoint verification.
One block is the minimum depth in this fixed-width family: a zero-block model
has an unavoidable ambiguity bound, described below. Width and vocabulary size
were not reduced, so this is not a globally minimum-parameter model.

## Verified results

| Transformer blocks | Physical parameters | Updates | Corpus epochs | Incorrect targets | Exact sentences | Mean loss (nats) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 51,483,648 | 3,840 | 60 | 0 / 10,002 | 1,024 / 1,024 | 0.000187040678 |
| 7 | 48,331,264 | 3,584 | 56 | 0 / 10,002 | 1,024 / 1,024 | 0.000569335296 |
| 6 | 45,178,880 | 3,712 | 58 | 0 / 10,002 | 1,024 / 1,024 | 0.000166443523 |
| 5 | 42,026,496 | 3,200 | 50 | 0 / 10,002 | 1,024 / 1,024 | 0.000290182373 |
| 4 | 38,874,112 | 3,456 | 54 | 0 / 10,002 | 1,024 / 1,024 | 0.000289197068 |
| 3 | 35,721,728 | 3,712 | 58 | 0 / 10,002 | 1,024 / 1,024 | 0.000043708074 |
| 2 | 32,569,344 | 3,200 | 50 | 0 / 10,002 | 1,024 / 1,024 | 0.000072873985 |
| 1 | 29,416,960 | 2,944 | 46 | 0 / 10,002 | 1,024 / 1,024 | 0.000131258831 |

Parameter counts include allocated vocabulary padding and count the tied
embedding/LM-head weights once. A nonzero loss is consistent with perfect
top-1 predictions; the criterion is zero errors, not a rounded loss of zero.
Update counts are the first observed perfect full-corpus evaluations, checked
every 128 updates and at the budget boundary, not a proof that an earlier
unevaluated step could not have been perfect.
The eight-block run took about 53 minutes, including its evaluations. All
models start from fresh initialization with seed 1337 and the same training
schedule; a shallower model does not inherit a deeper model's learned weights.
Comparing their step-zero checkpoint files confirms that all 88 weight tensors
shared by the seven- and eight-block models are byte-identical, including the
final LayerNorm tensors after accounting for their different file indices.
The same comparison passes for all 76 shared tensors in the six-block model.
It also passes for all 64 tensors in the five-block model's initial checkpoint.
All 52 tensors in the four-block initialization pass the same comparison.
All 40 tensors in the three-block initialization also pass.
The same comparison passes for all 28 tensors in the two-block initialization.
It passes for all 16 tensors in the one-block initialization as well.

Eight-block checkpoint:
`/home/ubuntu/checkpoints/memorize_general_facts/trial_0/layers_8/step_3840`.
Configuration, complete loss history, snapshots, and final per-token results
are in `runs/trial_0/layers_8/`.

The saved weights were reloaded first by the trainer, then by a fresh native
process. Both produced zero errors and byte-identical prediction TSVs. A
separate Python implementation retokenized the saved corpus and checked all
10,002 target identities, positions, and predictions, including EOS and full
coverage of all 1,024 sentences. Its input TSV is
`runs/verify_final_layers_8/final_predictions.tsv`. Causal attention and row-local
remaining layers make exact teacher-forced top-1 predictions imply exact
greedy suffix generation by induction; this is not a separate free-running
generation measurement.
Earlier fresh-process checks reproduced 5,203 errors at step 512 and 17 errors
at step 2,560; their complete prediction audits are preserved in
`runs/verify_step_512/` and `runs/verify_step_2560/`.

As an additional smoke check, the existing `gpt2_shakespeare_llm` inference
binary greedily completed corpus lines 1, 2, and 985 from only their first five
tokens, using the saved eight-block weights. All three matched the entire
sentence plus EOS exactly. Line 985 is the longest sentence (26 tokens):
`In standard Sudoku,` generated
` each row, column, and three-by-three box contains the digits one through nine once each.<|endoftext|>`.
These are three actual free-running checks; the full-corpus evidence remains
the exact causal per-token audit above.

Seven-block checkpoint:
`/home/ubuntu/checkpoints/memorize_general_facts/trial_0_remaining/layers_7/step_3584`.
Training and checkpoint reload took about 45 minutes. Its fresh-process native
verification and independent Python target audit both passed; their artifacts
are in `runs/trial_0_remaining/layers_7/independent_verification/`.
Repeating the entire seven-block evaluation with batch size 17, including a
partial final batch, also gave zero errors and a byte-identical per-token TSV;
that check is in `runs/verify_layers_7_batch_17/`.

Six-block checkpoint:
`/home/ubuntu/checkpoints/memorize_general_facts/trial_0_remaining/layers_6/step_3712`.
Training and checkpoint reload took about 42 minutes. Both independent checks
passed, and their per-token TSV is byte-identical to the trainer's final TSV.
The evidence is in `runs/trial_0_remaining/layers_6/independent_verification/`.

Five-block checkpoint:
`/home/ubuntu/checkpoints/memorize_general_facts/trial_0_remaining/layers_5/step_3200`.
Training and checkpoint reload took about 33 minutes. Both independent checks
passed, with a byte-identical final TSV; the evidence is in
`runs/trial_0_remaining/layers_5/independent_verification/`.

Four-block checkpoint:
`/home/ubuntu/checkpoints/memorize_general_facts/trial_0_remaining/layers_4/step_3456`.
Training and checkpoint reload took about 31 minutes. Both independent checks
passed, with a byte-identical final TSV; the evidence is in
`runs/trial_0_remaining/layers_4/independent_verification/`.

Three-block checkpoint:
`/home/ubuntu/checkpoints/memorize_general_facts/trial_0_remaining/layers_3/step_3712`.
Training and checkpoint reload took about 30 minutes. Both independent checks
passed, with a byte-identical final TSV; the evidence is in
`runs/trial_0_remaining/layers_3/independent_verification/`.

Two-block checkpoint:
`/home/ubuntu/checkpoints/memorize_general_facts/trial_0_remaining/layers_2/step_3200`.
Training and checkpoint reload took about 22 minutes. Both independent checks
passed, with a byte-identical final TSV; the evidence is in
`runs/trial_0_remaining/layers_2/independent_verification/`.

One-block checkpoint:
`/home/ubuntu/checkpoints/memorize_general_facts/trial_0_remaining/layers_1/step_2944`.
Training and checkpoint reload took about 17 minutes. Both independent checks
passed, with a byte-identical final TSV; the evidence is in
`runs/trial_0_remaining/layers_1/independent_verification/`.
An additional fresh-process evaluation with batch size one, processing each
sentence individually, also gave zero errors and a byte-identical per-token
TSV, independently retokenized and audited. Its artifacts are in
`runs/verify_layers_1_batch_1/`.

Evidence SHA-256 hashes:

- Corpus: `814c062e7d7592fe4a4e5b158a37bd37da51700f817c19eb981c1e93d33f245c`.
- Tokenizer: `1fe93b6152957cf9cfd6d89002467f789ce8b3f3e000b3a2edf27c808ddd0b9e`.
- Final eight-block prediction TSV: `a2e645bd61caf9b132e8c1e62718393fd9bcb26b0231d01477a1d4a7f7d0fe89`.
- Final seven-block independent prediction TSV: `b4782113e59c9d3ed8c3609ae6217dca27805426b247f2869869f52946cbd802`.
- Final six-block independent prediction TSV: `ee26c5f5457d99a46c9fc86899a8eb3cbc6d42a7628161cb212412237c8f696a`.
- Final five-block independent prediction TSV: `8cf6cfcafb9f83417f5fae2f8d82bd1101049193eb921d0c6a313872b71755da`.
- Final four-block independent prediction TSV: `04dbb7b048c83b563c6b435e312e15306fc055696d3d52fe1db440d4c7c1e870`.
- Final three-block independent prediction TSV: `c30c1fa5cff363c58142a3e45e7f4183225fb82d286d3d7b4ed4f551afb09eff`.
- Final two-block independent prediction TSV: `ad890fab231c8d5dc6a7171a3f8c4c4eba8f75aee9dc0e105ad62840ca45094c`.
- Final one-block independent prediction TSV: `08fe662b4151c030ac67e230e10f6f7aff755d25156328801262418d1410329a`.

## Completed search and interpretation

One-block training started on 2026-09-20 at 23:40:44 UTC, after the two-block
checkpoint passed both independent checks. Its final independent verification
finished at 23:58:00 UTC. The driver completed successfully with
`smallest_verified_layers=1`; its complete command/result history is in
`runs/trial_0_remaining/depth_search_summary.json`. The eight-block baseline was
run and independently verified separately before starting this driver.
No depth required retries, extra training budget, or hyperparameter changes.

Zero blocks are ruled out independently: without attention, the model can use
only the current token and its absolute position. The corpus contains 809 such
contexts with contradictory targets, forcing at least 2,923 errors out of
10,002, even with ideal optimization (see the prefix audit in README.md).
The one-block construction plus this zero-block lower bound establish minimum
depth for the tested family, not for arbitrary architectures. No held-out
generalization claim is made; prompt-token prediction is not scored.

All 65 optimized native test targets pass, including padding, CPU/GPU loss
agreement, determinism, gradient clipping, dataset batching, and checkpoint
validation. All 47 Python audit/verifier/driver tests pass. The final full native
suite check reused passing cached results; Python tests were rerun.

To reevaluate the smallest checkpoint on each sentence individually (use a
fresh output directory):

```sh
bazel build -c opt //src/llm/experiments/memorize_general_facts:memorize_general_facts
bazel-bin/src/llm/experiments/memorize_general_facts/memorize_general_facts \
  --layers=1 --batch_size=1 \
  --verify_checkpoint=/home/ubuntu/checkpoints/memorize_general_facts/trial_0_remaining/layers_1/step_2944 \
  --compact_vocabulary=false \
  --corpus=testdata/general_facts_dataset.txt \
  --tokenizer=/home/ubuntu/datasets/tokenizer/gpt2 \
  --output_dir=src/llm/experiments/memorize_general_facts/runs/new_verification
```

This is full-corpus checkpoint evaluation, not an interactive generation CLI.
