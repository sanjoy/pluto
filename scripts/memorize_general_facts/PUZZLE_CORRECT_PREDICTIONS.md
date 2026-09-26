# Which predictions does the A3 puzzle readout get right?

This is an analysis of a **saved** readout, with no retraining. It lives on
`codex/puzzle-correct-predictions`; the `main` puzzle implementation is unchanged.

Source: `dataset_weights_canonical_order_0/baseline/checkpoints/layers_4/step_120000`
under `/home/ubuntu/checkpoints/memorize_general_facts/`. Configuration: four
blocks, width 10, source MLP width 20, context 27, compact vocabulary 4,475.
Readout: the best checkpoint from `puzzle_main_train_20260926/best_mlp`, saved
at step 290,000 of the 300,000-step reproduction. It uses a 10/150/10 residual
MLP with trainable input and final LayerNorm. All other parameters are frozen.

The fresh native probe reproduced exactly **3,341/10,002 correct** predictions.
The source model gets all 10,002 right; the untrained replacement gets 217 right.
All following counts are teacher-forced scored suffix/EOS rows, not supplied
prompt or padding rows. Correct next tokens do not imply complete fact recall.

## The successful predictions are strongly skewed

| Target category | Correct / total | Accuracy |
|---|---:|---:|
| EOS | 1,024 / 1,024 | 100.00% |
| Punctuation | 970 / 1,223 | 79.31% |
| Whole function-word tokens (explicit heuristic list) | 792 / 2,507 | 31.59% |
| Other word/subword tokens | 552 / 5,226 | 10.56% |
| Number pieces | 3 / 22 | 13.64% |

EOS plus punctuation account for **59.68% of all successes**. All EOS examples
follow a true final period. Excluding those two categories gives
**1,347/7,755 = 17.37% accuracy**. Function-word labeling checks whole-word
boundaries in the original fact, so a subword such as `is` inside a name is not
mistaken for a grammatical word.

The ten most successful IDs account for **2,641/3,341 = 79.05%** of successes:
EOS, `.`, ` the`, ` of`, ` in`, ` a`, ` and`, `,`, ` from`, ` to`.
Of 2,900 distinct scored target IDs, only **485 are ever correctly predicted**.
However, successes are not exclusively frequent words: **232/1,844 singleton
target occurrences are correct**. Examples include ` Cambodia` after
“Angkor Wat is a temple complex in present-day”, and ` grams` after
“A kilogram contains one thousand”.

From the actual five-token prompts, only **169/1,024 = 16.50%** first predictions
are correct. Leading correct-token counts per fact are 0:855, 1:148, 2:17, 3:3,
4:1. No fact is entirely correct. Later teacher-forced successes can occur
after an earlier failure because the next input supplies the true prefix.

## Geometry is associated with success, beyond endings

Compute exact leave-one-out nearest neighbors among the 7,755 rows remaining
after EOS and punctuation are removed from **both queries and candidate pool**.
For the normalized comparison, center each A3 vector and divide by its RMS
with epsilon 1e-5; this is before any learned affine LayerNorm scaling.

- If the nearest normalized vector has the same target, the fitted readout is
  correct on **307/533 = 57.60%** of rows.
- Otherwise it is correct on **1,040/7,222 = 14.40%**.
- The fraction of matching targets among 25 neighbors has Spearman correlation
  **0.392** with correctness. Demeaning both variables within each target ID
  leaves Pearson correlation **0.333**, controlling target identity/frequency.

This is association, not causation or a sufficient condition for fitting.
Most correct lexical rows (77.21%) do not have a same-target nearest neighbor.
Correct lexical vectors are not simply in denser regions: their average
nearest arbitrary neighbor is actually farther away. The useful feature is
local agreement about the required output, not density alone.

## Preselected named-answer checks

Answer spans were declared before inspecting their predictions. Spans inside
the supplied prompt are excluded. This is a small descriptive cohort, not an
independent held-out test or exhaustive semantic annotation.

| Cohort | Eligible spans | Correct first piece | Whole span correct | Correct pieces |
|---|---:|---:|---:|---:|
| Named answers after definition phrases | 13 | 1/13 | 0/13 | 6/29 |
| Capital city/country names | 30 | 0/30 | 0/30 | 1/40 |

The only successful capital-name piece is `us` inside Cusco after the true prefix
already supplies ` C`; that is not successful recall of Cusco. France's capital
prompt predicts ` organisms` instead of ` Paris`; Greece predicts ` made`
instead of ` Athens`. Conversely, isolated whole lexical answers do succeed
(Cambodia, grams, Gibraltar), so it would be wrong to say the readout learns
only punctuation.

## Reproduce

Build the GPU inference probe with
`bazel build -c opt //scripts/memorize_general_facts:puzzle_prediction_probe`.
Run it with `--source_checkpoint`, `--readout_checkpoint`, `--tokenizer`,
`--corpus`, and a fresh `--output=/tmp/predictions.tsv`. Defaults match the
configuration above. No optimizer is created, and no checkpoint is written.

Run the CPU analysis and tests:

```sh
python3 -B scripts/memorize_general_facts/analyze_puzzle_predictions.py \
  /tmp/predictions.tsv --output_dir=/tmp/puzzle_correct_predictions
python3 -B scripts/memorize_general_facts/analyze_puzzle_named_answers.py \
  /tmp/predictions.tsv
python3 -B -m unittest discover -s scripts/memorize_general_facts \
  -p 'analyze_puzzle*_test.py'
```

The first command creates a standalone `report.html` containing every scored
prediction and a machine-readable `summary.json`. The geometry companion
accepts `--html` (the puzzle capture HTML) and `--predictions`; it requires NumPy
and SciPy, joins on exact fact/position IDs, and prints JSON. Generated files
and checkpoints are local artifacts, not tracked source.
