# Recovering a single-fact suffix from learned weights

This experiment uses models trained on **one** fact for 512 updates, not the
full-corpus checkpoint. The same procedure reconstructs the entire suffix and
EOS for four such models, without reading their corpus text or five-token
prompts. It needs the initial checkpoint, trained checkpoint, tokenizer and
compact vocabulary mapping, and the known task prompt length of five.

This is a limited, empirical decoder of the learned representation. It is not
a reconstruction of all 1,024 facts from the full-corpus model, a new training
algorithm, or an explanation of every internal weight. It also assumes each
selected non-EOS token occurs **exactly once**; repeated tokens are unsupported.

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
