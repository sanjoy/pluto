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
