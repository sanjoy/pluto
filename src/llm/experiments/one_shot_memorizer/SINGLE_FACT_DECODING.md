# Recovering single-fact text from checkpoint history

This experiment uses models trained on **one** fact for 512 updates, not the
full-corpus checkpoint. The same procedure reconstructs the entire suffix and
EOS for four such models, without reading their corpus text or five-token
prompts. It needs the initial checkpoint, trained checkpoint, tokenizer and
compact vocabulary mapping, and the known task prompt length of five.

This is a limited, empirical decoder of the learned representation. It is not
a reconstruction of all 1,024 facts from the full-corpus model, a new training
algorithm, or an explanation of every internal weight. It also assumes each
selected non-EOS token occurs **exactly once**; repeated tokens are unsupported.
The later sections add prompt-set recovery and recover its order using the
saved first training update, reconstructing all four complete facts under
these additional assumptions. The final checkpoint alone is not sufficient
for the demonstrated procedure.

## Frozen token-set readout

For each embedding row, compute its FP32 change from initialization. Select
rows whose dot product with the mean change over all vocabulary rows is
negative:

```text
delta[t] = trained_embedding[t] - initial_embedding[t]
common = mean_t(delta[t])
selected = {t : dot(delta[t], common) < 0}
```

The zero-threshold rule was discovered on Durian and committed before training
the other three models. All four 512-update endpoints select exactly their
distinct suffix/EOS token IDs, with no extra IDs. This cross-sentence check
uses one shared initialization, not four independent seeds. This rule does
not identify the five-token prompt or token multiplicities.

## Ordering the selected tokens

Let `S` be the selected set excluding EOS, with `n = |S|`. For every source
token `s` and suffix position `j`, run the trained model on constant EOS tokens
followed by `s` at absolute input position `5 + j`. All earlier positions are
EOS, not a recovered prefix or the original prompt. Record full-vocabulary
log-softmax scores for each selected destination and EOS.

An exact subset dynamic program finds the two highest-scoring paths through
all `n` distinct tokens, followed by EOS. Starting tokens all have score zero;
there is no known first token or hand-selected initial word. A path is scored
by the position-specific source-to-next-token scores, including its last
token's transition to EOS. Scores for unselected vocabulary tokens still
participate in each softmax denominator.

This objective is a **surrogate**, not the autoregressive probability of the
whole recovered sequence: each edge was scored with an artificial EOS history.
The DP is exact for that objective, not a proof that its winning text must be
the training text. Its cost is `n^2` model queries and `O(2^n n^2)` CPU search;
the implementation caps `n` at 16. No language model outside the checkpoint
or linguistic dictionary is used.

## Measured result

The ordering procedure was fixed before running it on these four endpoints.
Text was compared afterward; no per-fact search settings were changed.
All outputs below additionally end with EOS. Leading spaces matter.

| Single-fact model | Recovered suffix | Best score | Gap to runner-up |
| --- | --- | ---: | ---: |
| Durian, line 631 | ` for its strong smell and creamy edible flesh.` | -7.7605 | 2.7159 |
| France, line 80 | ` Paris, a city on the Seine.` | -15.7763 | 3.0967 |
| Mammals, line 1 | ` nourish their young.` | -4.1920 | 2.9455 |
| Prasad, line 258 | `ad became the first president of India in 1950.` | -9.6218 | 3.9248 |

Gaps are differences in the surrogate log-score, **not** calibrated confidence
in recovering the training data. The apparent `ad` fragment is correct: the
five-token prefix ends inside the name Prasad.

The simpler preceding attempt did not generally work: taking each selected
token alone and greedily following its highest-probability successor recovered
only the mammals suffix. Durian repeated `smell`; France and Prasad fragmented.
Using five EOS-prefix positions repaired Durian but not France or Prasad.
Thus the successful result must not be reported as universal greedy bigram
recovery. Global ordering and position handling are additional assumptions.

### Post-hoc controls: global ordering, not varying positions, is sufficient

Reusing **only the absolute-position-5 scores** at every path step also recovers
all four suffixes. Averaging each edge's log-scores over positions likewise
recovers all four. Thus these examples do not require position-dependent
queries once the global use-every-selected-token-once constraint is imposed.
The fixed-position variant needs only `n` model queries. This does not show the
actual network ignores position embeddings or behaves as a bigram model on
arbitrary prompts.

For either position-independent score mode, every source is used exactly
once, so the sum of source-specific log-softmax normalizers is constant over
candidate paths. Raw logits (or mean logits) therefore select the same
ordering in those two controls. This cancellation generally does not hold
when a source's score changes with its assigned position.

An independent C++ exhaustive enumeration of all permutations, up to `10!`,
checks both highest scores for all 12 unshuffled cases against the subset DP.
The saved original paths and scores also reproduce. Reference orders are
read only after decoding.

Independently permuting the edge scores' source and destination token labels
(EOS fixed), with 100 deterministic trials per fact, recovers one exact suffix
out of 400 trials: the shortest mammals case. Mean positional agreement is
10.5--19.2%, close to the respective `1/n` rates. These are **post-hoc sanity
controls**, not another prospective data set or a calibrated significance
test. Evidence: `/tmp/suffix_path_controls_0/`.

Local evidence: `/tmp/single_fact_position_path_{631,80,1,258}_0/` contains
selected tokens, every edge score, two decoded paths, individual edge
probabilities, and a manifest. The earlier greedy attempt is recorded in
`/tmp/single_fact_successors_{631,80,1,258}_0/`. Model paths and training
provenance are documented in the main experiment README.

The committed C++ implementation reproduces all four outputs under all three
score modes. CPU tests independently enumerate small permutations, including
ties, and check terminal handling, shape checks, nonfinite values, arithmetic
overflow, and score preparation. To run the primary version:

```sh
bazel build -c opt //src/llm/experiments/one_shot_memorizer:checkpoint_single_fact_position_path
bazel-bin/src/llm/experiments/one_shot_memorizer/checkpoint_single_fact_position_path \
  --initial_checkpoint=/home/ubuntu/checkpoints/memorize_general_facts/compact_batch_32_no_clip_0/layers_8/step_0 \
  --checkpoint=/tmp/one_shot_sentence_pilot_0/only_line_631/step_512 \
  --tokenizer=/home/ubuntu/checkpoints/memorize_general_facts/compact_batch_32_no_clip_0/inputs/tokenizer \
  --score_mode=position_aware --output_dir=/tmp/single_fact_path_new
```

The post-hoc alternatives are `--score_mode=fixed_position` and
`--score_mode=mean_positions`. Each run requires a fresh output directory.
Production reruns are under
`/tmp/single_fact_path_bazel_{631,80,1,258}_{MODE}_0/`.

## Exploratory prompt-token set readout

A separate weight-only diagnostic excludes the already sign-selected suffix
rows, then sorts the remaining rows by `||delta[t] - common||^2`, descending.
Taking five rows uses the known prompt length, not the sentence text. This
recovers **all five prompt-token IDs in each of the four models**, with no
extras. The ranking code and both tested rules were frozen before reading the
saved prompt labels; all four score files were written before that comparison.
This is nevertheless exploratory, not prospective: these models and related
facts had already been studied.

| Model | Correct prompt IDs in top five | Fifth/sixth residual-norm ratio |
| --- | ---: | ---: |
| Mammals | 5/5 | 19.41 |
| France | 5/5 | 23.34 |
| Prasad | 5/5 | 19.17 |
| Durian | 5/5 | 20.39 |

The ratios describe observed separation, not thresholds used for selection.
A secondary ranking by the component perpendicular to `common` also recovers
all four sets. These prompts each contain five distinct IDs, none in the
selected suffix set. A repeated prompt token or a token shared with the suffix
would invalidate this naive five-distinct-remaining-rows assumption.

An interpretation consistent with the gradient formula is that prompt tokens
receive input-lookup gradients in addition to the dense tied-head gradient;
subtracting the common direction exposes their distinct updates. This result
alone does not establish that causal decomposition over the whole trajectory.
It also does not recover order. The largest row happens to be the fifth input
token in all four cases, but that is a post-hoc observation, not a validated
general ordering rule. Local scores and provenance:
`/tmp/single_fact_prompt_rank_{1,80,258,631}_0/`.

The production suffix-decoder command now also emits `prompt_candidates.tsv`:
20 rows ranked by the primary residual-norm rule, with the top five marked as
an unordered candidate set. It never feeds those candidates into suffix
decoding. All four production reruns reproduce the candidate sets and the
previous suffixes; see `/tmp/single_fact_prompt_bazel_{1,80,258,631}_0/`.
Six additional CPU tests check exclusion, ordering/ties, unchanged weights,
zero mean, malformed/nonfinite inputs, and large finite inputs.

## Recovering prompt order: forward probability fails, update matching works

The forward-only control enumerates all 120 permutations of the recovered five
prompt IDs. It scores the first recovered suffix token under each five-token
input, using full-vocabulary log probability. Neither corpus text nor a known
prompt order enters the search. **None** of its four winners is the actual
training order:

| Model | Original order's rank | Orders predicting the correct next token |
| --- | ---: | ---: |
| Mammals | 11 | 120/120 |
| France | 6 | 72/120 |
| Prasad | 6 | 120/120 |
| Durian | 10 | 120/120 |

For example, France's highest-scoring prompt is `The of capital France is`,
not the original `The capital of France is`. These models were trained on
one fact, not pretrained English. A higher completion probability is therefore
not a reliable criterion for recovering the original input order. This
control scores the first suffix token only, not an entire autoregressive
completion. Every run checks repeated logits bit-for-bit. Evidence:
`/tmp/single_fact_prompt_order_{1,80,258,631}_0/`.

The successful second method uses **additional checkpoint information**:
`step_0` and the first update, alongside the 512-update endpoint used for the
token-set and suffix readouts. For each candidate prompt order, append the
already recovered suffix, then compute the first gradient at the known
initialization. Predict its first Adam embedding update using zero moments,
zero weight decay, the recorded rate `6e-6`, epsilon `1e-8`, and the original
mean suffix/EOS loss:

```text
normalized_gradient = FP32(g / (abs(g) + epsilon))
predicted_E = FP32(initial_E - learning_rate * normalized_gradient)
score = ||predicted_E - observed_step_1_E||_2
```

No candidate is trained iteratively: it receives one forward/backward at
initialization, and all candidate runs leave the initial weights unchanged.
The primary criterion uses all 71,600 embedding coordinates. The score is
not tuned using the true prompt. Actual text/order is checked only afterward.
All four unique minima recover the exact original prompt order:

| Model | Best L2 error | Runner-up L2 error | Update-sign mismatches at winner |
| --- | ---: | ---: | ---: |
| Mammals | 6.37e-9 | 3.71e-4 | 0/71,600 |
| France | 2.80e-8 | 2.01e-4 | 0/71,600 |
| Prasad | 6.03e-9 | 2.06e-4 | 0/71,600 |
| Durian | 2.20e-7 | 1.36e-4 | 0/71,600 |

Zero sign mismatches also uniquely identify the winner in each case, a
secondary check rather than a replacement for the frozen L2 rule. Looking
only at prompt-row L2 leaves three tied minima for Durian: the dense head
updates outside those five rows supply additional ordering evidence. The
scalar CPU/BF16 backward and simplified Adam expression do not exactly match
all GPU arithmetic, explaining the nonzero winning errors. Each 120-candidate
search took roughly 1.4 CPU seconds; no CUDA runtime was linked.

The committed `single_fact_first_update_probe --permute_prompt` mode reproduces
every ranking and numerical score from all 480 local candidates, not just the
four winners. Its original diagnostic mode still reproduces its previous
coordinate/summary files byte-for-byte. Eleven new CPU tests cover the update
score, epsilon, signs, signed zeros, FP32 rounding, shape/finite checks and
overflow. Search mode rejects implicit use of the diagnostic's labeled default:
`--token_ids` must be supplied explicitly.

This example reads the **unordered candidate set and recovered suffix** from
the earlier weight-only output, removing its terminal EOS before passing IDs
to the backward probe. It does not read the corpus or a true prompt:

```sh
bazel build -c opt //src/llm/experiments/one_shot_memorizer:single_fact_first_update_probe
decoded=/tmp/single_fact_prompt_bazel_631_0
recovered_tokens=$(awk -F '\t' '
  FNR==NR { if (FNR>1 && $6==1) { ids=ids sep $2; sep="," } ; next }
  FNR==2 { n=split($4,a,","); for (i=1;i<n;++i) ids=ids "," a[i] }
  END { print ids }
' "$decoded/prompt_candidates.tsv" "$decoded/paths.tsv")
bazel-bin/src/llm/experiments/one_shot_memorizer/single_fact_first_update_probe \
  --permute_prompt --token_ids="$recovered_tokens" \
  --initial_checkpoint=/home/ubuntu/checkpoints/memorize_general_facts/compact_batch_32_no_clip_0/layers_8/step_0 \
  --first_update_checkpoint=/tmp/one_shot_sentence_pilot_0/only_line_631/step_1 \
  --output_dir=/tmp/recovered_prompt_order_new
```

Production search reports are under
`/tmp/single_fact_prompt_update_bazel_{1,80,258,631}_0/`.

Local evidence: `/tmp/single_fact_prompt_update_search_{1,80,258,631}_0/`.
This is an inversion over a small, already recovered candidate set, not a
general inversion of arbitrary minibatches or late checkpoints. It assumes
known initialization, first-step optimizer settings, loss definition, and a
saved update attributable to this one fact. Together with the earlier stages,
it recovers all four complete training sentences, but does not decode the
1,024-fact model or construct its learned nonlinear features without training.

The reconstructed text, checked against the corpus only after decoding, is:

- `Female mammals produce milk to nourish their young.`
- `The capital of France is Paris, a city on the Seine.`
- `Rajendra Prasad became the first president of India in 1950.`
- `Durian fruit is known for its strong smell and creamy edible flesh.`

Importantly, **recoverability does not imply that the original order is
required for the correct next-token choice**. The early update distinguishes
the original order even though many reordered prompts produce the same later
argmax. Probabilities still differ, so this does not prove the network ignores
order. It distinguishes recovering exact training history from identifying
what is necessary for the observed next-token decision.

## Relation to existing work

FILM similarly separates word-set recovery from ordering using model scores,
beam search and reordering. It starts from communicated gradients; our probe
uses initialization and a checkpoint after 512 Adam updates. FILM's nonzero
embedding-gradient criterion cannot be copied directly here because the tied
output head generally gives every row a gradient. This is not a reproduction
of FILM or a claim that the two-stage idea is new.
[Gupta et al., FILM](https://arxiv.org/abs/2205.08514).

iDLG exploits label-dependent signs/relations in single-example cross-entropy
gradients. That helps motivate looking for a contrast between target and
non-target updates, but does not prove our rule for multiple scored positions,
tied embeddings, and accumulated Adam updates.
[Zhao et al., iDLG](https://arxiv.org/abs/2001.02610).

The first-update replay in the main README supports a narrower mechanism:
many non-target rows receive a shared dense softmax contribution, and Adam
makes their coordinate changes similar. This explains why a common direction
can be informative. It does not establish that the separation persists for
arbitrary facts, seeds, repeated-token sentences, or multi-fact training.
