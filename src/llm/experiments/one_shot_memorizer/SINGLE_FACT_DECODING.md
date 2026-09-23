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

## Prospective stress tests (protocol fixed before these runs)

The four successful examples have distinct suffix IDs and no overlap between
their prompt and suffix token sets. Before training further cases, fix the
following tests of those assumptions. Use the original initialization, frozen
compact vocabulary, seed, Adam settings, and learning-rate schedule; do not
retune the decoder from the outcomes.

1. **Line 32, repeated suffix:** the first corpus sentence whose suffix repeats
   a non-punctuation token: `At a smooth reflecting surface, the angle of
   reflection equals the angle of incidence.` Train alone for 512 updates.
   Its suffix/EOS has 12 occurrences but nine distinct IDs. In particular,
   ` the`, ` angle`, and ` of` each occur twice.
2. **Line 2, prompt/suffix overlap:** the first corpus sentence sharing a token
   between its first five tokens and suffix: `In English, the plural of mouse
   is usually mice when referring to the animal.` Train alone for 512 updates.
   The shared token is ` the`. This tests the prompt-readout exclusion rule.
3. **France/Greece mixture, lines 80 and 406:** train the two facts together
   with batch size one for 1,024 updates, saving every 128 updates. Evaluate
   the frozen sign rule against the *union* of suffix/EOS IDs at 512 updates
   and at the predeclared 1,024-update extension. These sets share comma,
   period, ` city`, and EOS. At the endpoint each fact has 512 exposures, but
   the global Adam clock differs from the single-fact runs. Record that
   distinction rather than treating these as optimizer-matched conditions.

The existing one-use-per-token path decoder is deliberately unchanged for
the first assessment. A wrong order, wrong length, or explicit rejection is
a negative result, not permission to supply missing multiplicities. Record
prompt-candidate failures too. The trainer's accompanying full-corpus
baseline is retained for provenance, not called an exposure-matched control.
Training text is joined to the weight readout only for scoring afterward.

### Stress-test results

Both new single-fact checkpoints learned their entire suffix and EOS exactly
(12/12 teacher-forced targets and exact greedy completion). The unchanged
readout gives:

| Case | Distinct suffix/EOS IDs recovered | Prompt set | Ordered suffix |
| --- | --- | --- | --- |
| Repeated tokens, line 32 | 9/9, no extra IDs | 5/5 | Incorrect: repetitions absent |
| Prompt/suffix overlap, line 2 | 12/12, no extra IDs | 4/5 | Exact |

The repeated-token result is `, the angle of reflection equals incidence.`:
three occurrences are missing. Selecting nine distinct IDs cannot specify
twelve occurrences. The overlap case recovers ` of mouse is usually mice
when referring to the animal.` exactly, but excludes shared ID 61 (` the`)
from its prompt candidates and replaces it with background ID 3092. The
four eligible true prompt IDs are ranks 1..4. These are decoder limitations,
not failures of those two trained networks.

For the jointly trained France/Greece pair, the frozen sign rule identifies
**all 14 distinct suffix/EOS IDs with no extras** at both 512 and 1,024 updates.
At 512 neither sentence completes exactly (France 9/10 targets, Greece 6/8);
at 1,024 both complete exactly (10/10 and 8/8). Thus the vocabulary-set signal
can be present before correct sequencing and prompt-dependent selection.
It does not partition the set into two sentences. Each fact received exactly
512 scheduled updates at the endpoint. The first update used France and is
byte-identical to the earlier France-only first update across all 100 tensors.
New and old full-corpus baselines also match bytewise at steps 1, 128 and 512.
All initial checkpoints and the complete sample/exposure logs were checked.

The small pair also supplies a useful execution-trace contrast to the full
corpus. Using the final LayerNorm/head as a diagnostic lens, Paris first wins
after block 3 attention; Athens wins after block 0 MLP. Their final target
probabilities are 83.79% and 84.32%. Zeroing **any one** of the sixteen
attention/MLP residual branches still preserves both next-token answers,
although confidence changes. These are 32 independently restored ablations,
not removal of all branches together and not full-suffix preservation tests.
By contrast, all sixteen individual branch removals break France's answer in
the fully trained 1,024-fact checkpoint. A block's apparent necessity is a
property of a particular trained computation, not a fixed semantic address.
The 512-update pair trace was rejected by the trace tool because Greece's
sixth token was wrong; it is not included as a successful answer-preservation
trace.

An exploratory CPU-only extension allowed repeated vertices in the
fixed-position graph: shortest path over `(visited-set, last-token)` with
nonnegative negative-log-probability costs, permitting EOS only after full
set coverage. It did **not** help. All six winners used no revisits; the four
earlier examples remained exact, but repetition was not recovered. For line
32, the real repeated walk has score -17.4883, below the shorter incorrect
winner's -13.7990. The issue is the surrogate scores, not just a prohibition
on repeats. This unsuccessful extension was kept local, not added to the
production decoder. Fixed-position scoring also gives the wrong order for
line 2, while the original position-aware scoring succeeds.

Evidence (local, not committed model artifacts):

- `/tmp/one_shot_single_stress_line_{32,2}_0/`: training and checkpoints.
- `/tmp/single_fact_stress_path_{32,2}_0/`: unchanged decoder results.
- `/tmp/single_fact_stress_audit_{32,2}_0/`: independent CPU token-set audit.
- `/tmp/one_shot_pair_stress_80_406_0/`: pair training and sample provenance.
- `/tmp/one_shot_pair_sign_{512,1024}_0.{tsv,summary}`: frozen union readouts.
- `/tmp/one_shot_pair_trace_1024_0/`: layer-by-layer trace and HTML report.
- `/tmp/one_shot_pair_branch_trace_1024_0/`: branch removals and controls.
- `/tmp/single_fact_covering_walk_1/`: negative repeated-walk control.

To reproduce the pair without changing vocabulary or initialization:

```sh
bazel build -c opt //src/llm/experiments/one_shot_memorizer:checkpoint_sentence_ablation
facts=/home/ubuntu/checkpoints/memorize_general_facts/compact_batch_32_no_clip_0
bazel-bin/src/llm/experiments/one_shot_memorizer/checkpoint_sentence_ablation \
  --initial_checkpoint="$facts/layers_8/step_0" \
  --tokenizer="$facts/inputs/tokenizer" \
  --output_dir=/tmp/two_fact_readout_new \
  --steps=1024 --batch_size=1 --omitted_lines= --single_fact_line=0 \
  --isolated_lines=80,406 --norepeat_baseline \
  --checkpoint_every=128 --evaluate_every=512
```

`isolated_mixture/isolated_schedule.tsv` identifies the source corpus line at
every step, including its supervised-target count (10 versus 8). The loss is
the mean over that sentence's suffix and EOS; this is one update per sentence
occurrence, not a pooled token-weighted multi-sentence objective.
`isolated_exposures.tsv` records counts at each saved checkpoint. The original
full-corpus baseline still runs; it is not mislabeled as a mixture control.

## Coordinate-convention check

The earlier Durian GPU control subtracts the common embedding update `c` from
every trained embedding row and adds `c` to every trained position row. It
still gets 10/10 targets and exact greedy completion. Reconstructing that
same FP32 intervention on CPU, with all 16 recorded shift coordinates matched
bit-for-bit, changes the frozen sign readout from ten correct IDs to 3,274
selected IDs with **none** of the ten correct ones. Prompt-set recovery goes
from 5/5 to 0/5.

This is not a new meaningful alternative vocabulary. Removing the mean makes
the decoder's reference direction zero in ideal arithmetic; its residual
norm is only 6.39e-10, versus 0.909 before intervention. The selected signs
then reflect numerical residue. In exact arithmetic the compensated shift
preserves input sums and changes all tied-head logits by a common scalar;
FP32/BF16 rounding prevents assuming equality on arbitrary prompts. Only the
reported Durian behavior has been verified.

The original initialization remains fixed in this intervention. Transforming
both initialization and endpoint by the same fixed shift would preserve their
delta in exact arithmetic. Consequently this experiment demonstrates a
dependence on the training-coordinate convention and initialization reference,
not erasure of the fact or impossibility of all weight-based decoders. It
reinforces the distinction between recovering training-history information
and finding a functionally invariant semantic representation.

Evidence: `/tmp/single_fact_gauge_readout_631_0/`, with the existing behavioral
control in `/tmp/one_shot_single_fact_probe_0/conditions.tsv`.
