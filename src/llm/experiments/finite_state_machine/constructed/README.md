# Constructed-weight FSM interpreter

This experiment sets neural weights by construction instead of training them.
The weights implement an interpreter: a different machine description in the
prompt changes the computation without changing any weight. The training and
test files are used only for evaluation, never to construct the weights.

It is a CPU-only reference network with sparse affine maps, ordinary causal
softmax attention, ReLU, and an untied vocabulary projection. It is **not** a
checkpoint for the GPT-2 model being trained in the parent directory. In
particular, it uses FP64 arithmetic, no LayerNorm, explicitly allocated register
channels, and ReLU rather than GELU. Its success does not establish that SGD can
discover the same algorithm in our GPT-2 architecture.

## Run

From the repository root:

```sh
bazel build -c opt //src/llm/experiments/finite_state_machine/constructed:constructed_fsm

binary=bazel-bin/src/llm/experiments/finite_state_machine/constructed/constructed_fsm
"$binary" --prompt='000X093;093A044;XA>' --trace
# 000 093 044

"$binary" --prompt='000X093;093A044;XB>'
# 000 093 ERR

# Free-running exact evaluation of both full-trace corpus splits, with a
# human-readable dump of every nonzero matrix coefficient and bias.
"$binary" --weights=/tmp/constructed_fsm_weights.tsv

bazel test -c opt //src/llm/experiments/finite_state_machine/constructed:all
```

The prompt ends at `>`; no output state is supplied. ASCII spaces are ignored.
Output starts at `000`, lists one state per successfully consumed letter, and
ends in `ERR` on the first missing transition. Empty input produces just `000`.
An internal END output stops generation; it is not part of the dataset format.
The full prompt plus emitted trace must fit `--max_context` (default and maximum
1024). Duplicate `(state, letter)` definitions are rejected as ambiguous.

## What the weights do

Each token has 10 state bits, 5 letter bits, and five token-kind flags. Absolute
position embeddings add 10 position bits, position `p`, and `p²`. Along with a
constant channel, this is a 33-channel embedding. The code contains explicit
matrices for every projection and MLP; temporary register widths differ between
stages. This is a multi-stage attention network, not a stack of standard
equal-width GPT-2 blocks.

1. **Assemble table rows.** Two positional attention heads retrieve the previous
   one and two tokens. At the destination of `031X994`, their values are `X` and
   `031`. A ReLU gate detects the pattern `(state, letter, state)`. That row's
   key encodes `(031, X)` and its value encodes `994`. Other rows cannot masquerade
   as transitions. This runs over the original token stream, not a host-built
   transition dictionary.
2. **Locate the input.** A head finds the last semicolon before `>`. Another
   retrieves that boundary and the position of `>`. With current position `p`,
   semicolon position `b`, and `>` position `g`, an affine layer computes the
   next input position `t = b + p - g`.
3. **Read the next letter.** A positional attention head retrieves the token at
   `t`. Its score for candidate position `j` is `16(2tj-j²)`, equivalent to
   `-16(j-t)²` up to a query-dependent constant. No host array indexing by `t`
   occurs.
4. **Look up the transition.** The current output state and retrieved letter
   produce a query of 15 bipolar bits. A matching row scores 15, any mismatching
   row at most 13. The `>` row scores 14 and supplies ERR, so it wins exactly
   when no transition exists. All other invalid rows score -32. Query weights
   multiply these scores by 16 before ordinary softmax.
5. **Emit and repeat.** Fixed control MLPs emit initial `000`, detect exhaustion
   of input, and stop after ERR. A fixed linear head decodes the destination
   bits into vocabulary logits. The host chooses the top token, appends it, and
   calls the same network again, as in ordinary autoregressive generation.

No inference code executes a state transition or consults the expected answer.
The host performs lexical/syntax validation, embedding lookup, generic numeric
operations, attention caching, final vocabulary argmax, and the generation loop.
The separate integer-table simulator in `model_test.cc` is only a test oracle.
Attention's winning indices are recorded for diagnostics, but never select its
returned values: those are actual softmax-weighted sums over all causal rows.

## Why finite softmax is sufficient

Attention does not become a hard lookup by taking an infinite temperature.
Instead, each retrieved binary register passes through this fixed ReLU MLP:

```text
clean(x) = ReLU(2x - 0.5) - ReLU(2x - 1.5)
```

It returns exactly zero below 0.25 and exactly one above 0.75. For at most 1024
rows with a winning score gap of at least 16, the winning probability is at least

```text
1 / (1 + 1023 * exp(-16)) > 0.9998848.
```

Thus irrelevant rows contribute too little to change any cleaned bit. The same
bound applies to the integer positional lookups. Retrieved positions are encoded
as bits and cleaned **before** reconstructing their integer value, so attention
errors do not accumulate into the counter. Position squaring is part of the
fixed absolute-position embedding matrix, not a new runtime nonlinearity.

These margins give a construction argument for syntactically valid deterministic
machines and traces within the context bound, using real arithmetic. Tests also
exercise the finite FP64 implementation. They are not a proof of behavior in
BF16 or for unbounded sequence lengths.

## Verification

Initial free-running evaluation (2026-10-01):

| Split | Exact complete traces | Accuracy |
| --- | ---: | ---: |
| Training | 4096 / 4096 | 100% |
| Test | 128 / 128 | 100% |

Both splits together took 8.21 seconds on this machine with an optimized CPU
build. With the default 1024-token context, there are 25,809 nonzero matrix
coefficients and biases; all are constructed constants, not fitted parameters.
The text export is approximately 276 KiB. Generated exports are not checked in:
the weight-construction code fully reproduces them.

Unit tests additionally compare 250 newly generated machines with an independent
interpreter, test table shuffling, every state label and input letter, missing
transitions, self-loops, invalid prompts, 250-transition tables near the context
limit, and a 509-step walk whose END is predicted at position 1023. Numerical tests
verify that attention computes soft mixtures rather than hard argmax selection.

The code and tests are AI-generated and have not been human-reviewed.

## Relation to earlier constructions

The broad approach is to compile an algorithm into attention and MLP weights,
as in [Tracr](https://arxiv.org/abs/2301.05062) and
[ALTA](https://arxiv.org/abs/2410.18077). This is an independent, task-specific
implementation, not a use or reproduction of either compiler. The explicit
binary-code lookup, separator fallback, and finite-softmax cleanup argument above
describe the construction actually implemented here.
