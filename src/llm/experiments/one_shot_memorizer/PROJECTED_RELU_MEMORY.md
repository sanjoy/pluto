# A smaller dataset-only neural memory

## Prospective construction

The existing explicit ReLU memory obtains all 1,024 exact suffix-plus-EOS
completions from text and tokenizer alone. Its corpus-dependent biases and
16-dimensional output-code columns take 1,360,136 bytes, before metadata.
This follow-up changes the constructed architecture to reduce that storage;
it is **not** a claim about the trained GPT-2's internal code.

The query `q` contains the final nine original GPT-2 token IDs, left padded
with -1. Nine is fixed from the earlier complete context-conflict audit; it
is not a newly selected window. Merge identical corpus query/target pairs
and reject a query associated with different targets.

Choose nine positive integer projection coefficients in `[1, 2^20]` with
SplitMix64 seed zero. Compute `h(q) = sum_j a_j*q_j`. Require distinct
compiled queries to have distinct hashes, even if their target labels match.
If not, draw another nine coefficients from the same fixed generator; stop
with an error after 256 attempts. This is a deterministic collision test,
not gradient descent, a fitted classifier, or an accuracy-based search.

For every compiled hash `h_i`, build actual ReLU units:

```
positive_i = ReLU(h - h_i)
negative_i = ReLU(h_i - h)
indicator_i = ReLU(1 - positive_i - negative_i)
token = sum_i indicator_i * target_token_i
```

An integer hash is either exactly equal to `h_i`, yielding indicator one, or
has distance at least one, yielding zero. Inference evaluates **every unit's
ReLU arithmetic**; it must not use a dictionary lookup, nearest-neighbor
search, or a supplied target. The output is a scalar vocabulary ID, not the
old model's 16-dimensional binary code. No vocabulary-sized learned decoder
is required.

Store explicit opposite biases `-h_i,+h_i` and scalar output weights as FP64,
plus the nine FP64 projection coefficients and metadata. Each hash uses at
most `9*2^20*65535 < 2^40` in magnitude under the allowed vocabulary bound.
Differences and intermediate sums remain integers safely within FP64's exact
integer range. Check these bounds in construction and deserialization; do not
rely on silently rounded hashes or allow nonfinite/partial numeric records.
Report all fixed sparse weights and biases separately from stored values.

## What this does and does not guarantee

There is no trained checkpoint, teacher activation, optimizer, or gradient
in the construction. Its core construction, serialization, and neural
execution are CPU-only. The repository's tokenizer API still needs a CUDA
Executor for its pinned host allocations; that dependency does not imply
GPU-based model computation.

**Unknown contexts can collide with a stored hash.** This deliberately weaker
out-of-corpus behavior differs from retaining all nine key coordinates. An
unknown hash yields NotFound, but an unseen key with a known hash can receive
that key's answer. Include a test demonstrating this limitation. Do not claim
exact membership testing, arbitrary-prompt equivalence, or that every unseen
history is rejected. As with the old suffix memory, earlier history beyond
nine tokens is ignored.

The comparison trades width and FP64 precision against learned low-dimensional
features. Count bytes, not just scalar parameters, and keep tokenizer,
architecture constants, and other shared information explicit. A compact
stored artifact would not show that gradient descent finds this representation.

## Verification gate

* Unit tests: deterministic construction, duplicate keys, conflicting labels,
  exact neuron outputs, collisions/retries, bounds, malformed artifacts,
  serialization round trip, no-match errors, and the unseen-alias example.
* Preserve existing balanced-token-code assignments when sharing their
  deterministic random generator.
* Compile from the complete corpus and original tokenizer; serialize, reload,
  and independently generate every suffix plus EOS from five-token prompts.
* Require 1,024/1,024 full completions and report all 10,002 target decisions,
  the number of distinct keys, the selected coefficients/attempt, actual file
  bytes, and construction versus verification times.
* Keep generated weights local; commit tested C++ code and measured results.

Budget: approximately one hour. Do not add a larger projection, retain hidden
key tables, or change to token-dependent coefficients to conceal a failed
fixed construction.

## Result: exact corpus recall with a smaller numeric artifact

The fixed construction succeeded on 2026-09-23 without changing its window,
seed, coefficient bounds, collision rule, or inference arithmetic. The first
coefficient draw was collision-free. After writing the numeric weights to
disk, the tool mapped that file back in, deserialized it, checked an identical
serialization round trip, and generated each sentence's suffix from only its
first five tokens. It fed back its own predictions until EOS, with the same
1,024-token generation cap for every case. The expected suffix was consulted
only after generation to score the result.

| Quantity | Result |
| --- | ---: |
| Exact suffix-plus-EOS completions | 1,024 / 1,024 |
| Generated targets, all matching | 10,002 / 10,002 |
| Text input tokens / active tokens including EOS | 14,098 / 4,475 |
| Distinct nine-token contexts / memory units | 10,001 / 10,001 |
| Projection attempts, including the successful draw | 1 |
| First / second ReLU layer units | 20,002 / 10,001 |
| Explicit stored FP64 scalars | 30,012 |
| Numeric weight bytes / metadata bytes | 240,096 / 60 |
| Complete saved artifact bytes | 240,156 |
| Generated fixed sparse weights / threshold biases | 40,004 / 10,001 |
| Largest absolute stored hash | 132,212,058,765 |
| Smallest distance between stored hashes | 26 |
| Construction / verification time | 0.00271 s / 0.12675 s |

The timing separates construction from verification, excludes tokenization,
and is one local measurement rather than an isolated performance benchmark.
An independent sum of all 1,024 per-sentence verification rows reproduced
every aggregate. A second complete compile/verify run produced byte-identical
weights, projection coefficients, and per-sentence results; only timings
changed. A separate inference invocation, with no corpus input, returned:

```
The capital of France is Paris, a city on the Seine.
```

The selected coefficients, from oldest to newest suffix coordinate, are:

```
904624, 615925, 607568, 819693, 554140,
959211, 340706, 437053, 625348
```

### Storage comparison, with the tradeoffs kept explicit

The former full-key ReLU construction stores 340,034 corpus-dependent FP32
values: **1,360,136 numeric bytes**, or 1,360,182 bytes with its metadata. This
construction stores three FP64 weights per unit plus nine projection weights:
`(3*10001+9)*8 = 240096` numeric bytes. The explicit opposite biases are both
counted; they are not silently deduplicated.

The learned GPT-2 has 114,256 trainable scalars, whose FP32 master values take
**457,024 bytes** before checkpoint metadata. Thus the new numeric artifact
is smaller even after charging eight bytes rather than four per stored value.
This is **not** a like-for-like neural parameter-efficiency or compression
result. It uses 30,003 ReLU units, a nine-token scalar-coded input, exact FP64
integer arithmetic, and a scalar token-ID output. GPT-2 uses narrow BF16
hidden states, a learned embedding, attention, GELU, and a vocabulary softmax.
The 40,004 fixed sparse weights and 10,001 threshold biases above are generated
by the architecture and not stored. Materializing them as FP64 would add
400,040 bytes. Neither comparison includes tokenizer storage, executable
code, or runtime allocation overhead.

It is also not text compression: the 14,098 original token IDs fit in 28,196
bytes at two bytes each, before sentence boundaries. This model deliberately
spends more storage to expose an explicit arithmetic next-token function.

### The unseen-context limitation is demonstrated, not merely hypothetical

Under the selected first projection, these distinct five-token inputs have
the same hash after left padding:

```
[61,157,0,0,1]
[0,0,14,411,1]
```

A unit test constructs a model from only the first input and verifies that
the unseen second input receives its answer. A separate test includes both
in the corpus with the same EOS label and verifies that construction rejects
that projection and retries, as specified. The corpus guarantee therefore
holds, but arbitrary unseen inputs can still collide; there is no claim of
exact membership rejection or equivalence to the learned model outside the
corpus. Earlier history beyond nine tokens is ignored as well.

Ten focused CPU tests also cover repeated tokens, token ID zero, EOS, prefix
padding/truncation, conflicting labels, ordering/duplicate invariance, maximum
vocabulary arithmetic, invalid numeric weights, and malformed/truncated/
trailing artifacts. The loader validates the named seed-0 projection draw;
without original keys it cannot prove earlier draws collided. Existing
balanced-token-code tests pass after extracting their unchanged SplitMix64
implementation into a shared helper. No trained checkpoint is read anywhere
in the new constructor or inference path.

## Reproduction

```bash
bazel build -c opt \
  //src/llm/experiments/one_shot_memorizer:projected_relu_memorizer
bazel test -c opt \
  //src/llm/experiments/one_shot_memorizer:projected_relu_memory_test \
  //src/llm/experiments/one_shot_memorizer:token_codes_test

tokenizer=/home/ubuntu/checkpoints/memorize_general_facts/compact_batch_32_no_clip_0/inputs/tokenizer
tool=bazel-bin/src/llm/experiments/one_shot_memorizer/projected_relu_memorizer

"$tool" --mode=compile --tokenizer="$tokenizer" \
  --corpus=testdata/general_facts_dataset.txt \
  --output_dir=/tmp/projected_relu_memory_new

"$tool" --mode=infer --tokenizer="$tokenizer" \
  --model_file=/tmp/projected_relu_memory_new/projected_relu.weights \
  --prompt='The capital of France is'
```

The output directory must be fresh. The tokenizer uses its original 50,257
IDs, not the trained checkpoint's compact vocabulary. Although its location
is under a previous run's directory, only tokenizer files are read, not model
weights. The neural library has no CUDA dependency; the CLI still requires a
CUDA-capable environment because the existing tokenizer allocates pinned
host arrays through an Executor. No GPU model computation runs.

Evidence is local in `/tmp/projected_relu_memory_0/` and the independent
repeat `/tmp/projected_relu_memory_repeat_0/`. Each contains the numeric
weights, every sentence's verification result, the projection, and a summary.
Generated artifacts stay outside Git. This closes the narrower constructive
gap of smaller dataset-only exact recall; it still does not explain how
gradient descent discovers the learned GPT-2 representation.
