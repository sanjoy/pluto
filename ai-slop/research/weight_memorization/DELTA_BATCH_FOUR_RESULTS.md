# Four-sequence replay: strong early alignment, no later recovery

The exploratory batch-size-four hypothesis sharply localizes the early
ordinary-after token bag: **128/128** selected IDs occur in the prescribed
draws 81–120, versus alternative-seed mean **6.711** and maximum **32**.
The same bag has just one ID in the previous forty draws and none in the next
forty or reset-phase forty draws.

However, it **does not rescue the later-checkpoint experiment**. Both later
shared bags have zero IDs in their predicted draws 401–440. The adjusted bag
also has zero in both adjacent forty-draw phases. This is a mixed result, not
evidence that the historical batch size is now authenticated or that arbitrary
later updates can be decoded.

## Why four was tested

The [preceding corpus-assisted localization](DELTA_TEXT_EXTENSION_RESULTS.md)
showed the ordinary-after bag's high-ranking windows clustered near replay
draws 82–100. Four sequences per training step would place its checkpoint
interval N+20→N+30 at draws 81–120. The inspected training code consumes one
dataset batch per optimizer step and samples one start per sequence.

This motivated a [separate fixed protocol](DELTA_BATCH_FOUR_PROTOCOL.md),
written before the new counts. No candidate IDs, weight scores, checkpoint
intervals, seed, or corpus were changed. This is an explicitly exploratory,
outcome-informed phase test, not clean held-out validation. The earlier
batch-one and batch-ten results remain intact.

## All primary and control outcomes

Every entry is an overlap out of 128 DISTINCT candidate token IDs. Each
forty-window union includes input and target positions, 1,025 tokens per
window; repeated tokens count once. `Alt mean / max` refers to the predicted
phase across exactly seeds 10000–10999. Previous, next, and reset columns
use seed 17. Every alternative-phase count is retained in the full evidence.

| Bag | Labels | Predicted | Previous | Next | Reset | Predicted alt mean / max |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Ordinary-after, adjusted | Real | **128** | 1 | 0 | 0 | 6.711 / 32 |
| Ordinary-after, adjusted | Shuffled | 15 | 9 | 6 | 9 | 9.908 / 17 |
| Ordinary-after, raw | Real | 80 | 23 | 14 | 60 | 29.095 / 52 |
| Ordinary-after, raw | Shuffled | 13 | 7 | 7 | 10 | 11.522 / 20 |
| Later follow-through, adjusted | Real | **0** | 0 | 0 | 12 | 11.377 / 34 |
| Later follow-through, adjusted | Shuffled | 20 | 11 | 22 | 16 | 16.188 / 25 |
| Later follow-through, raw | Real | **0** | 7 | 0 | 21 | 18.923 / 51 |
| Later follow-through, raw | Shuffled | 10 | 8 | 11 | 14 | 10.279 / 18 |

The zero-based half-open draw slices were fixed as follows:

| Bag family | Predicted | Previous | Next | Reset |
| --- | --- | --- | --- | --- |
| Ordinary-after | [80,120) | [40,80) | [120,160) | [0,40) |
| Later follow-through | [400,440) | [360,400) | [440,480) | [0,40) |

The raw early bag has appreciable reset-phase overlap too, unlike the adjusted
bag's much narrower signal. Shuffled-label controls are not frequency-matched
null distributions, and alternative-seed ranks are descriptive rather than
calibrated p-values.

## Interpretation

The early deltas contain rare-token information associated with a specific
later part of the conditional replay, not merely generic Shakespeare
vocabulary. The four-sequence phase mapping is consistent with that early
signal. It is not established as the actual invocation setting: historical
corpus/split identity, sampler implementation, restarts, and flags still lack
authenticated provenance.

The later zero overlaps rule out **this fixed batch-four alignment as a repair
for the tested later bags**. They do not prove that no later update contains
recoverable information. Optimizer memory, the transient score, historical
sampling differences, and other mechanisms remain unproved explanations.
No further batch-size or phase search was performed in this experiment.

This result concerns unordered token membership. The positive passage lookup
still gets sentence order from the supplied corpus, and the separately tested
weight-only ordering heuristic still fails. There is no ordered-text decoder.

## Evidence and reproduction

The complete run is `/tmp/pluto-delta-batch-four.VoLjtg/`:

- `batch_four_replay.py`: standalone counter, synthetic checks, freeze and count
  commands. It imports the previously tested scalar sampler oracle.
- `plan.json`: exclusive plan before new token-payload reads and overlap counts.
- `starts.npz`: all 1,001 seeds × 480 starts, independently regenerated and
  compared exactly with the previously authenticated 1,200-draw export.
- `membership.npz`: all 4,100,096 seed/phase/list/ID membership bits.
- `results.json`: all 32 overlap vectors, summaries, and unchanged input/source
  identities; SHA256
  `8594e730eda3676f03aae0fb055e5236a582316b41a6eba150ebddb7cdd5cfb8`.

The count run took 7.26 seconds. Synthetic endpoint, duplicate-window,
out-of-range, tie, phase-arithmetic, and sampler checks passed before it ran.
No new checkpoint weights, model execution, GPU work, or source-corpus search
were needed; the fixed native export supplies membership verification only.

An independent native C++ sampler reproduced all **480,480 draws** exactly.
A separate NumPy histogram implementation reproduced all 32 overlap vectors
(32,032 counts), all 4,100,096 membership bits, and all 7,007 phase/seed distinct
token counts. Summaries, ties, histograms, and unchanged source/input identities
also passed. This independently verifies the mixed result, not historical
settings. The audit program, start record, result, and independent sampler
output are retained beside the counter's outputs.

The [evidence archive](delta_batch_four_evidence.tar.gz) preserves these ten
files, including the fixed protocol. Every member was compared byte-for-byte
against its original and gzip integrity passed. Original corpus, checkpoint,
and prior-experiment dependencies are identified but not duplicated.

- Archive size: 3,778,316 bytes; member payload: 4,575,874 bytes.
- SHA256: `d47070a94ec403f257216219d4e3841f9255fca1e20d012c78463dab989cc88e`.

The script exposes `self-test`, `freeze`, and `count` subcommands. Its run paths
are intentionally fixed for provenance. To reproduce in a new directory,
copy the script there, inspect its declared input paths, then run `self-test`,
`freeze`, and `count` in order. It refuses to overwrite a plan or result.
