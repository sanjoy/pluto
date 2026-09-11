# Fixed protocol: first-layer geometry and a closed token graph

Date: 2026-09-09. This protocol is committed before the new real-checkpoint
candidate runs or corpus verification. It is an exploratory follow-up to the
negative experiments in REPORT.md, not an independent confirmation study.
Previous results motivated its design; the new vocabulary, candidates, and
ranking must not be tuned after inspecting their corpus matches.

## Question

The previous QK-by-OV probe used raw embedding coordinates even though the
model projects LayerNorm outputs. Its independently selected previous/current
tokens also formed a disconnected graph. Do correcting the coordinates and
removing that selection obstacle yield weight-derived text beyond ordinary
short transition patterns? Producing connected paths alone is not success.

## Frozen inputs and vocabulary

- Evaluated trained checkpoint: `/home/ubuntu/checkpoints/shakespeare/step_13030`.
- Early control: the already validated `step_10` extraction; original archive
  SHA256 `e14d4a939e055775a59be73cce8039922644b8095add1c873eba1cf65dc23eb0`.
- Token labels: `/home/ubuntu/datasets/tokenizer/gpt2/tokenizer.json`.
- Logical vocabulary: 50,257 tokens; physical padding is excluded.
- No corpus, token frequencies, prompts, or model-derived contextual states
  may be supplied to candidate extraction.

Using the **trained checkpoint as vocabulary selector in every arm**, form
first-layer normalized dictionary rows

```text
X_p[t] = gamma * (E[t] + P[p] - mean(E[t] + P[p]))
                  / sqrt(var(E[t] + P[p]) + 1e-5) + beta
```

Here mean and population variance reduce over residual channels; gamma and
beta are block-zero pre-attention LayerNorm weights. Position zero supplies
the previous-token dictionary and position one the current-token dictionary.
The epsilon is explicit in the inspected GPT-2 recipe. Use FP64 analysis, not
claimed BF16/FP32 bitwise replay.

Select `S`, the 128 token IDs with greatest joint query norm
`||concat_h(X_1[t] WQ_h + bQ_h)||`, breaking ties by ascending ID. Record both
rank order and canonical sorted IDs. The final-weight-derived vocabulary is
an explicit learned prior also given to the early control; that control is
not an entirely independent random-vocabulary experiment.

## Four arms, with identical S and search limits

| Arm | Evaluated weights | Input dictionaries | Routing/content pairing |
| --- | --- | --- | --- |
| normalized | step 13,030 | X_0 and X_1 above | intact |
| raw | step 13,030 | X_0 = X_1 = E | intact |
| broken | step 13,030 | X_0 and X_1 above | QK_h with intact OV_(h+1 mod 8) |
| early | step 10 | its own X_0 and X_1 | intact |

For each ordered pair `(b,a)` in `S x S`, compute

```text
r_h = (X_1[a] WQ_h + bQ_h) · ((X_0[b] - X_1[a]) WK_h) / sqrt(head_dim)
write(b,a) = sum_h r_h * ((X_0[b] - X_1[a]) WV_h WO_h) / 4
score(b,a,c) = cosine(write(b,a), E[c])
```

In the broken arm, use the specified intact OV pair in each term. Retain the
top four destinations **inside S**, searching all 128. This restricted rank
is not a full-vocabulary rank or a prediction accuracy. Zero write/target
directions are undefined and must not generate arbitrary ties. Retain finite
negative scores if they are among the top four; report them rather than
silently switching selection policy.

Emit all resulting triples, including valid repeated-token triples. Build
paths only by exact `(b,a,c) -> (a,c,d)` overlap. Reuse the existing bounded
path search: 256 weight-ranked start pairs, beam width four, maximum length
12, no immediate repetitions, and at most two occurrences of each token.
These are maxima; preserve actual length histograms and zero/short outcomes.

This is a static first derivative of two-position attention at uniform
routing, not its full trained-scale output. Position zero/one are reused in
every graph edge; a path does not execute the position-dependent model over
its entire length. Zeroth-order attention, the residual path, MLPs, later
blocks, final normalization, and contextual nonlinear effects are absent.
No softmax or transformer forward call belongs in the extractor.

## Verification and interpretation

Freeze candidate JSONL and metadata hashes for **all four arms** before
opening the verification corpus. Use the same native C++ corpus export as
the earlier experiments; its token SHA256 is
`17680667f56fe95ba5903339d867d0ecc940e789b65f5d56e484ff8b33a73b21`.
This is full-source-corpus matching, not authenticated historical training
membership; the original checkpoint invocation/split remains unknown.

Report exact complete-candidate and internal-substring matches separately,
longest match, actual candidate lengths, distinct matching strings, and
physical weight provenance. No long candidate may be called recovered when
only a short part matches. Single tokens and generic formatting are weak
evidence. Display examples selected after matching must be labeled as such.

Run both existing bigram-preserving controls and new **trigram-preserving**
controls with fixed seeds 17, 29, and 43. The latter randomizes an Euler trail
on token-pair nodes, preserving every directed triple and its multiplicity,
unigrams, bigrams, and endpoint pairs. Candidates shorter than four tokens
cannot demonstrate ordering beyond triples and are omitted from that control.
Record those denominators explicitly. These shuffles are not uniform and
produce descriptive diagnostics, not calibrated p-values.

Even successful extraction from this two-token-state graph may reflect only
learned trigrams. A passage-specific claim needs evidence beyond those
preserved transitions, stable replay from addressed weights, and separate
causal validation. This protocol does not redefine the original goal around
making a graph connected, increasing a short-match count, or passing tests.
No training or checkpoint writes are authorized by this experiment.
