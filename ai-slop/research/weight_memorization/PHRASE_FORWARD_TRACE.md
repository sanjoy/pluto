# How `to be or not to be,` becomes next-token predictions

## Result

Intermediate final-head readouts change at different depths and sometimes
reverse. Five of seven positions only persistently match the final winner after
the last MLP. On this exact input, top-1 predictions do **not** follow the quotation.

I ran the exact lowercase phrase, including its comma, through the unmodified
native BF16 GPT-2 at `/home/ubuntu/checkpoints/shakespeare/step_13030`.
There was no BOS, newline insertion, sampling, or autoregressive continuation.
Each row below predicts the token **after** that input position, using only its
prefix. Spaces inside quoted tokens are significant. Blocks are zero-based.

| Position | Input token | Final top token | Probability | First / persistent final-token readout |
|---|---|---|---:|---|
| 0 | `to` | `\n` | 5.63% | B7 MLP / B7 MLP |
| 1 | ` be` | ` the` | 9.38% | B7 MLP / B7 MLP |
| 2 | ` or` | ` no` | 7.47% | B7 MLP / B7 MLP |
| 3 | ` not` | ` in` | 12.11% | B6 MLP / B6 MLP |
| 4 | ` to` | ` be` | 28.22% | B0 MLP / B0 MLP |
| 5 | ` be` | ` the` | 7.20% | B7 MLP / B7 MLP |
| 6 | `,` | `\n` | 42.10% | B2 attention / B7 MLP |

These seven outputs are **not** a generated sentence. Only one of the six
supplied next tokens is the model's top prediction. The native token IDs are
`[1462, 307, 393, 407, 284, 307, 11]`.

## The transformation

1. **Tokens become vectors.** Each ID selects a 512-dimensional embedding;
   its learned position embedding is added. Reading these vectors through the
   final LayerNorm and tied vocabulary head favors the input token itself at
   all seven positions. This is an intermediate readout, not an actual copying
   operation or a discrete token inside the network.
2. **Attention mixes prefix information.** Each of eight pre-LayerNorm blocks
   forms Q/K/V, applies eight causal heads, projects their combined output, and
   adds it to the residual vector. Future tokens cannot contribute.
3. **MLPs transform those features.** Each normalizes, expands 512 channels to
   2,048, applies tanh-approximate GELU, projects back to 512, and adds a residual
   update. B0 attention still reads as input tokens; B0 MLP changes all seven
   readouts to possible continuations. Position 4's ` be` remains top thereafter.
4. **Later blocks revise the predictions.** At the final comma the readout
   moves from comma, to ` and` after B0 MLP, to ` bump` and back to ` and`
   in B1, to newline at B2 attention. Newline remains top until B5 MLP favors
   ` but`; the final MLP restores newline. At position 2, the final MLP changes
   ` I` to ` no`; at position 5 it changes newline to ` the`.
5. **The final head makes vocabulary scores.** Final LayerNorm followed by a
   dot product against the tied token embeddings gives 50,257 usable logits.
   The probabilities above are their temperature-one softmax. Stored
   activations and matrix operands use BF16; reductions and logits use FP32.

The [complete 17-stage readout table](PHRASE_FORWARD_READOUTS.md) gives every
attention/MLP transition, not just the examples above. Applying the final head
early is a *logit-lens diagnostic*: its first match is not an irreversible
decision time, and early-layer vectors need not be calibrated for that head.

## What actually affects these outputs?

I separately reran the same supplied prompt while removing each of 16 residual
branches, each of 64 heads, and 14 selected MLP neurons. Each removal was
independent, and the original device weights were restored and byte-checked.
These diagnostic forwards never feed predicted tokens back into the model.

- **Early processing matters despite late readout changes.** Removing B0's MLP
  changes all seven winners. Removing its attention branch, or just head 5,
  changes six. The final MLP therefore is not doing the whole computation alone.
- **The comma prediction has distributed support.** Removing B2 attention
  changes its winner to ` but`: newline falls from 42.10% to 22.32%, versus
  28.52% for ` but`. Removing B7 MLP also gives ` but` (40.46%, versus 33.94%
  newline). The latter output is byte-identical to the pre-final-MLP readout.
- **Individual late features can tip close contests.** Removing B7 neuron 464
  changes position 2 from ` no` to ` I`; the original margin was only 0.06968.
  Removing B7 neuron 147 changes position 5 from ` the` to newline, reducing
  their margin by 0.71238. These features also affect other positions; they are
  not uniquely identifiable “no” or “the” neurons.

An additive ledger further decomposes each fixed final winner-versus-runner
margin into embeddings, every head, every MLP neuron, biases, and explicit
numerical remainders. For newline versus ` but` at the comma, B2's head writes
contribute +2.863, B4's head writes −2.545, and B7's MLP neuron writes +1.267,
before all other terms produce the final +1.089 margin. All seven ledgers close
within `1.34e-15`; this is an accounting identity, **not** a deletion effect.

Reconstructed Q/K routing also explains why attention weight alone is not
importance: B2 head 2 attends most to position 0 (43.1%), but that source's
projected contribution opposes newline versus ` but` (−0.291). Position 5 gets
only 9.3% attention yet contributes +0.776. These source values are already
contextualized; this does not assign a universal meaning to the original words.
The probabilities are reconstructed diagnostics, not exported GPU softmax
tensors. Their summed contributions closely match the captured native context.

## Confidence and limits

The trace exports 84 stage tensors: original-forward values plus 17 native
replays of otherwise-discarded outputs. Final readout, alternate padding, clean
replay, and all 16 reconstructed residual additions pass exact byte checks.
Checkpoint bytes never changed. An independent CPU BF16 transformer agrees on
all seven winners (logit RMS difference 0.0350).
Full-FP32 diagnostic arithmetic flips position 5: ` the` at 7.20% is barely ahead
of newline at 7.04%. That choice should not be interpreted as robust.

The ledger, 94 interventions, and two independent audits characterize this
specific computation. They do not establish a general semantic circuit, locate
the quotation in the weights, or recover training data. A pairwise margin can
also miss a third token becoming dominant: removing B0 head 5 increases the
comma's newline-versus-` but` margin, yet its actual winner becomes ` nor`.

Full numerical results and reproduction instructions are in
[the evidence index](PHRASE_FORWARD_EVIDENCE.md).
