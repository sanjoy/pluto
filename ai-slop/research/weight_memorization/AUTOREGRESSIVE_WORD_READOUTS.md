# `shagemper`: complete intermediate readouts and neuron removals

This supplement retains all 17 recorded residual-stream readouts for each of
the four generated pieces and all 16 selected neuron-removal experiments.
The original prompt was `to be or not to be` (no comma). The sampled pieces
were `' sh'` (427), `'ag'` (363), `'em'` (368), and `'per'` (525), at zero-based
generation steps 51–54. The leading space belongs to token 427. Each column
below is a **different forward pass**, conditioned on all preceding generated
token IDs; it is not four predictions made from the same input.

Checkpoint: `/home/ubuntu/checkpoints/shakespeare/step_13030`. Blocks B0–B7
and neuron IDs are zero-based. Activations and matrix operands use native BF16;
saved logits are FP32. Probabilities use all 50,257 logical vocabulary entries
and the production temperature **0.8**. The 15 padded entries are excluded.

## All intermediate readouts

Each cell is **target probability / target rank**. Probabilities are percentages,
rounded to six significant digits; ranks are exact, descending by raw logit,
with lower token IDs breaking ties. At each stage, the diagnostic applies the
trained final LayerNorm and tied output head to that stage's residual stream.
These are saved native diagnostic logits, not a separately fitted readout.
They show when a token becomes readable by this particular lens—not when the
network makes an irreversible decision or which component is causally necessary.
The last row is byte-identical to the actual final logits.

| Residual stage | Step 51: `' sh'` | Step 52: `'ag'` | Step 53: `'em'` | Step 54: `'per'` |
| --- | ---: | ---: | ---: | ---: |
| Input + position | 1.60111e-29% / 8969 | 9.30824e-36% / 41123 | 7.89032e-23% / 54 | 7.46749e-28% / 243 |
| B0 attention | 2.99634e-28% / 46199 | 4.65335e-34% / 45738 | 5.95184e-21% / 34 | 2.35864e-27% / 36978 |
| B0 MLP | 0.00123437% / 240 | 1.96613% / 8 | 0.285794% / 5 | 3.22304% / 2 |
| B1 attention | 0.0011566% / 621 | 0.413998% / 20 | 0.0516174% / 18 | 4.20642% / 3 |
| B1 MLP | 0.000755714% / 501 | 0.941888% / 6 | 0.0268001% / 23 | 2.62319% / 3 |
| B2 attention | 0.00048459% / 522 | 0.451053% / 7 | 0.118936% / 15 | 6.69093% / 4 |
| B2 MLP | 6.96624e-05% / 726 | 0.867772% / 7 | 0.195289% / 14 | 7.22385% / 2 |
| B3 attention | 1.38192e-05% / 1090 | 1.9461% / 4 | 0.0842834% / 7 | 18.39% / 2 |
| B3 MLP | 5.35545e-05% / 670 | 0.31248% / 23 | 0.581908% / 5 | 28.8364% / 2 |
| B4 attention | 1.42235e-05% / 771 | 1.75248% / 9 | 18.3155% / 2 | 8.26166% / 2 |
| B4 MLP | 5.29002e-05% / 253 | 1.46327% / 14 | 67.7584% / 1 | 88.8247% / 1 |
| B5 attention | 0.000298543% / 122 | 1.09058% / 17 | 74.8927% / 1 | 87.8289% / 1 |
| B5 MLP | 0.0248269% / 16 | 0.127579% / 79 | 90.9144% / 1 | 98.1686% / 1 |
| B6 attention | 0.0252918% / 19 | 0.146069% / 65 | 96.0356% / 1 | 98.6729% / 1 |
| B6 MLP | 8.32005% / 2 | 0.52% / 19 | 93.6401% / 1 | 98.6078% / 1 |
| B7 attention | 13.5441% / 3 | 1.2757% / 9 | 97.8025% / 1 | 97.5957% / 1 |
| B7 MLP | 47.9521% / 1 | 2.35557% / 5 | 91.2067% / 1 | 81.8388% / 1 |

`' sh'` becomes rank one only at the final recorded stage. `'ag'` is never
rank one at these stages; its actual emission was a non-greedy sample.
`'em'` and `'per'` first become rank one after B4's MLP, but this observation
alone does not establish that B4's MLP is necessary for either output.

## All selected neuron removals

The selection was frozen before these interventions: per step, the two most
positive and one most negative direct neuron terms across all blocks, plus
the two most positive B4 terms for steps 53 and 54. All 16 selections are
retained below, including opposing or small measured effects.

Each native intervention zeros the neuron's **entire output-weight row W2**,
removing its output at **every prefix position**, and reruns the complete
forward pass. The row is restored and checked afterward. This is not an
intervention confined to the last query position, and it does not zero the
neuron's input weights or the MLP output bias.

`Direct` is the original, selected-position contribution to the target-minus-
fixed-competitor logit margin, with the final LayerNorm scale held fixed for
bookkeeping. `Δmargin` is the actually measured post-removal margin minus
baseline; the original competitor remains fixed. `Δlog P` is
`log(P_after / P_before)`, in nats, using temperature 0.8. These are different
quantities: a direct term is not an intervention effect.

| Step; target vs fixed competitor | Baseline logit margin | Original sampling uniform |
| --- | ---: | ---: |
| 51; `' sh'` (427) vs `' years'` (812) | +0.222759 | 0.2181119136065706 |
| 52; `'ag'` (363) vs `'ire'` (557) | -2.053700 | 0.035620306436279704 |
| 53; `'em'` (368) vs `"'d"` (1549) | +2.813996 | 0.3642201397881389 |
| 54; `'per'` (525) vs `'inate'` (4559) | +2.297270 | 0.05544122820452272 |

The final column reapplies that step's original uniform to the changed full-
vocabulary CDF. It is **not** the greedy output. This coupled sample depends
on vocabulary ordering; increasing a target's probability does not guarantee
that a previously successful fixed uniform still selects it.

| Step | Removed block / neuron | Direct | Δmargin | Target probability before → after | Δlog P | Same-uniform piece (ID) |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 51 | B7 / 407 | +0.722610 | -0.742914 | 47.9521% → 29.9628% | -0.470247 | `' sh'` (427) |
| 51 | B6 / 837 | +0.698525 | -0.798936 | 47.9521% → 27.9797% | -0.538722 | `' sh'` (427) |
| 51 | B5 / 368 | -0.703962 | +0.324338 | 47.9521% → 55.2637% | +0.141914 | `' sh'` (427) |
| 52 | B2 / 1963 | +0.857624 | -1.289653 | 2.3556% → 0.8921% | -0.970906 | `'ain'` (391) |
| 52 | B6 / 958 | +0.474506 | -0.590604 | 2.3556% → 1.5136% | -0.442290 | `'ag'` (363) |
| 52 | B4 / 1211 | -0.594638 | +0.732804 | 2.3556% → 3.8046% | +0.479435 | `'if'` (361) |
| 53 | B6 / 1831 | +0.756720 | -0.703137 | 91.2067% → 80.0445% | -0.130546 | `'em'` (368) |
| 53 | B6 / 1415 | +0.638013 | -0.674615 | 91.2067% → 85.5366% | -0.064184 | `'em'` (368) |
| 53 | B0 / 418 | -0.822422 | -0.520237 | 91.2067% → 84.8577% | -0.072153 | `'em'` (368) |
| 53 | B4 / 1855 | +0.422644 | -0.496401 | 91.2067% → 88.9479% | -0.025078 | `'em'` (368) |
| 53 | B4 / 526 | +0.334183 | -0.051868 | 91.2067% → 91.6627% | +0.004987 | `'em'` (368) |
| 54 | B6 / 1642 | +0.610169 | -0.561320 | 81.8388% → 70.8479% | -0.144217 | `'per'` (525) |
| 54 | B3 / 695 | +0.419079 | -0.228342 | 81.8388% → 75.5064% | -0.080535 | `'per'` (525) |
| 54 | B7 / 659 | -0.634440 | +0.643575 | 81.8388% → 89.1542% | +0.085615 | `'per'` (525) |
| 54 | B4 / 1717 | +0.299081 | +0.037973 | 81.8388% → 84.4235% | +0.031094 | `'per'` (525) |
| 54 | B4 / 588 | +0.267188 | -0.227872 | 81.8388% → 70.6074% | -0.147617 | `'per'` (525) |

The direct sign predicts the opposite sign for a removal's margin change in
14/16 cases and probability change in 13/16. The exceptions are informative:

- B0/418 has a negative direct term for `'em'`, yet removing it lowers both
  the target margin and probability.
- B4/1717 has a positive direct term for `'per'`, yet removing it raises both.
- Removing B4/526 slightly lowers `'em'`'s margin against the fixed competitor
  while increasing its probability against the entire vocabulary.

These observations are consistent with the distinction between linear
bookkeeping and a full-network intervention; they do not identify which
downstream path causes each discrepancy.

## Worked gate: B2 neuron 1963 while predicting `'ag'`

At generation step 52, the last input token is `' sh'`, at context row 57.
This selected neuron supplies a positive direct margin term for `'ag'` over
`'ire'`, and its removal substantially reduces the probability of `'ag'`.
The checkpoint stores little-endian FP32 master weights; matrix operands
are rounded to BF16 for the forward computation.

| Parameter | File within checkpoint | Exact byte address |
| --- | --- | --- |
| W1 column 1963, 512 elements | `weight_34.bin` | Element i begins at `7852 + 8192*i`, i = 0…511 |
| Input bias 1963 | `weight_35.bin` | `[7852, 7856)` |
| W2 row 1963, 512 elements | `weight_36.bin` | `[4020224, 4022272)` |

The saved normalized input dotted with the BF16 W1 column gives
`2.994059106335044`; adding bias `-0.02942458912730217` gives the analytical
pre-GELU value `2.9646345172077417`. The recorded native pre-GELU and post-GELU
activations are both `2.96875`. The analytical tanh-GELU at the recorded
preactivation is `2.9647347333402023`, which rounds to that BF16 output.
The analytical input contraction is not claimed to reproduce native reduction
rounding bit-for-bit.

The weight-affinity calculation gives `'ag'` score `0.19349509387906513` and
`'ire'` score `-0.07864385246742436`. Their difference is
`0.2721389463464895`. Dividing the native activation by the diagnostic final
residual standard deviation `0.9420356922272133` gives `3.151419871343851`, so

```text
direct margin term = 3.151419871343851 * 0.2721389463464895
                   ≈ +0.8576240832829051
actual removal Δmargin = -1.2896528244018555
```

The affinity uses the centered BF16 decoder row and trained final-LayerNorm
gamma; it is not a next-token probability. The direct term is a contextual
gate times a weight/readout alignment, not a stored word. The larger actual
removal effect includes the all-position/full-forward intervention described
above.

## Evidence and checks

Original run directory: `/tmp/pluto-autoreg-word.iu9n_w5k/`.
The four source analyses are
[step 51](/tmp/pluto-autoreg-word.iu9n_w5k/token_analysis_step_51/analysis.json),
[step 52](/tmp/pluto-autoreg-word.iu9n_w5k/token_analysis_step_52/analysis.json),
[step 53](/tmp/pluto-autoreg-word.iu9n_w5k/token_analysis_step_53/analysis.json), and
[step 54](/tmp/pluto-autoreg-word.iu9n_w5k/token_analysis_step_54/analysis.json).
These retain all native-lens logits' identities, probabilities, ranks, direct
terms, gate products, and the original generation events.

The [frozen feature selection](/tmp/pluto-autoreg-word.iu9n_w5k/weight_affinity_selection_plan.json),
[weight affinities](/tmp/pluto-autoreg-word.iu9n_w5k/weight_affinities.json),
[neuron-run plan](/tmp/pluto-autoreg-word.iu9n_w5k/neuron_causal_plan.json), and
[native run results](/tmp/pluto-autoreg-word.iu9n_w5k/neuron_causal_result.json)
preserve selection and byte identities.

The [independent neuron audit](/tmp/pluto-autoreg-word.iu9n_w5k/neuron_causal_independent_audit.json)
verified all four baseline rows against original generation bytes, all 340
native arrays against the original focused traces, the exact 16-arm selection,
20 independently calculated baseline/intervention sampler rows, and direct
terms recomputed from raw checkpoint rows and saved stages. Maximum direct-term
discrepancy was `3.89e-16`. Its JSON is 425,716 bytes, SHA256
`f79bdae05368ea2099a7908b5f2051135257f0260d757c4d0792f1295f115c9d`.
The earlier [full-path math audit](/tmp/pluto-autoreg-word.iu9n_w5k/token_path_math_independent_audit.json)
also covers the 17-stage lens readouts and selected gate operations.

Displayed tables were checked programmatically against these JSON artifacts.
An initial audit serialization failure left a file named
`neuron_causal_independent_audit.incomplete.json`; it is not a valid completed
audit and is not a source for this report. No new model runs were needed to
produce this supplement.
