# Passage-level causal validation: verified partial results

Date: 2026-09-09. Checkpoint: /home/ubuntu/checkpoints/shakespeare/step_13030.
This is **model-based validation, not analytical extraction**. Known corpus
tokens are inputs. No text here was analytically decoded from weights. No
optimizer, backward pass, training update, or checkpoint write was performed.

## Findings and baseline

The model is strongly familiar with the current training prefix, but does not
exactly continue any of these 16 sampled passages for the full 512-token test.
The interventions produce a reproducible functional-dependence map, not a
private storage address for each passage:

- All 32 interventions increase mean loss on every sampled prefix passage:
  512 positive passage-by-intervention effects. Removing a branch hurts more
  than halving it in all 256 prefix passage-by-group comparisons.
- The first block's MLP is the largest intervention effect on every prefix
  passage at both doses. It also damages the suffix substantially. This broad
  effect does not distinguish stored text from shared computation or routing.
- Halving either of the last two MLPs improves all 16 suffix passages while
  hurting all 16 prefix passages. This is consistent with split-dependent,
  overfitting-related behavior, not proof of historical membership or unique
  storage. Suffix calibration changes are another possible factor.

The [protocol](/home/ubuntu/code/pluto/research/weight_memorization/CAUSAL_VALIDATION_PROTOCOL.md)
was committed as e7c397d before measurements. Tools were committed as d1125b7,
followed by pre-run JSON provenance correction 886f617. The fresh plan was
authenticated before the first forward; its passages and packed token bytes
are identical to the first, never-executed plan.

We sampled 16 nonoverlapping windows per side of the current 90/10 byte split,
at fixed stratified midpoints, without selecting by text or loss. Native IDs
and every original byte interval were validated. The split is at byte 4,892,836
/ native token 1,650,781; the suffix has 184,382 tokens. Historical training
membership, corpus version, and split settings are not authenticated.

Each window has 1,025 tokens. We score its final 512 targets after 513 source
tokens, using BF16 activations and FP32 logits/loss, in microbatches of four
sequences. NLL is in nats per target; perplexity is exp(mean NLL).

| Clean baseline | Current training prefix | Current suffix |
| --- | ---: | ---: |
| Windows / scored targets | 16 / 8,192 | 16 / 8,192 |
| Mean NLL | 0.416707 | 5.979143 |
| Perplexity | 1.51696 | 395.10179 |
| Teacher-forced top-1 accuracy | 90.9302% | 36.2305% |
| Exact matched-prefix length, min–max | 2–39 tokens | 0–3 tokens |
| Complete 512-token continuations | 0/16 | 0/16 |

The prefix mean exact match is 14.625 tokens. Count only consecutive correct
predictions before the first mismatch; later teacher-forced matches are not
recitation. The greedy-prefix interpretation follows causal induction under
this fixed forward computation and smallest-ID tie rule. Different precision,
batching implementations, or sampling policies were not tested. Secondary
all-1,024-target NLLs are 0.536803 and 5.901140, respectively. These sampled
means are not whole-corpus memorization guarantees.

## Weight-group dependence map

Blocks are **zero-based**. Scale both the output matrix and output bias from
pristine weights; zero removes the branch but retains its residual skip.
Restore and byte-check each group before changing another. Entries are paired
changes from the clean mean: positive means worse prediction.

| Block / branch | Half: prefix | Half: suffix | Zero: prefix | Zero: suffix |
| --- | ---: | ---: | ---: | ---: |
| 0 attention | +1.047735 | +0.120167 | +4.170221 | +0.948061 |
| 0 MLP | +3.785091 | +1.333279 | +11.663191 | +6.940600 |
| 1 attention | +0.420607 | +0.234174 | +2.863660 | +1.145467 |
| 1 MLP | +0.039839 | +0.038515 | +0.214433 | +0.134046 |
| 2 attention | +0.246805 | +0.185032 | +1.502499 | +0.762965 |
| 2 MLP | +0.042550 | +0.030622 | +0.204895 | +0.084874 |
| 3 attention | +0.140364 | +0.063367 | +0.712706 | +0.202885 |
| 3 MLP | +0.102648 | +0.062285 | +0.603366 | +0.256981 |
| 4 attention | +0.237281 | +0.063309 | +1.159373 | +0.247809 |
| 4 MLP | +0.148944 | +0.021389 | +0.734323 | +0.077474 |
| 5 attention | +0.069374 | −0.001140 | +0.307705 | +0.020345 |
| 5 MLP | +0.156476 | −0.018741 | +0.806242 | +0.041190 |
| 6 attention | +0.050839 | +0.011640 | +0.214458 | +0.053373 |
| 6 MLP | +0.161490 | −0.135753 | +0.861537 | −0.057221 |
| 7 attention | +0.039334 | +0.051183 | +0.160839 | +0.123110 |
| 7 MLP | +0.199848 | −0.136756 | +1.029242 | +0.174052 |

Attention output has a 512×512 matrix plus 512-element bias; MLP output has
a 2,048×512 matrix plus 512-element bias. Compare blocks within a family;
neither raw effects nor effects divided by parameter count measure bits stored.
Do not add single-branch effects across this nonlinear model.

Attention rank is 0, 1, 2, 4, 3, 5, 6, 7 at both doses. Apart from block 0,
MLP damage generally increases toward the output: zero-dose rank is
0, 7, 6, 5, 4, 3, 1, 2; half-dose swaps the final 1 and 2.

Exact checkpoint addresses for block b, little-endian FP32 [input, output]:

- Attention: weight_(6+12*b).bin, bytes [0, 1,048,576), plus
  weight_(7+12*b).bin, bytes [0, 2,048).
- MLP: weight_(12+12*b).bin, bytes [0, 4,194,304), plus
  weight_(13+12*b).bin, bytes [0, 2,048).

The first MLP is files 12/13; the last two MLPs are files 84/85 and 96/97.
The 101-handle traversal deduplicates the tied embedding to 100 actual files.
Mutations affect underlying device bytes, not copied buffer handles.

The suffix has 12/256 reversed-dose comparisons: zero is less damaging than
half. Halving block 7 MLP improves every suffix window, but removing it improves
only one. Removing block 6 MLP improves 11/16, with mean delta −0.057221.
Weaker branches are not uniformly better. No architecture or training change
is proposed on the basis of this validation.

## Passage-specific examples, not confirmed storage sites

The prespecified selectivity score subtracts the mean delta of the other three
passages in the same split and clean-loss quartile. All background IDs and the
full matrix remain in the results JSON. These examples are selected after
measurement for illustration, not independent confirmations:

- T04, scored bytes [1,387,218, 1,388,942): removing block 1 attention raises
  NLL by 3.873783, or 0.964432 above its matched background. Halving gives
  +0.485784 raw / +0.069628 matched. Physical files: 18/19.
- T11, scored bytes [3,530,814, 3,532,265), contains the known corpus line
  “In this same interlude it doth befall”. Removing block 7 MLP raises NLL by
  1.972797, or 0.851633 above its background; halving gives +0.430348 raw /
  +0.191901 matched. Files 96/97 support this passage, but also every other
  prefix passage. The quoted line was read from the corpus, not weights.

These tiny descriptive comparisons amid many effects are not p-values,
causal editing results, or proof of independent text storage. Broad first-block
effects dominate both examples too.

## Integrity, artifacts, and reproduction

The run took 208.888 seconds on the GH200, with no competing GPU process.
Both clean_repeat and clean_after have **byte-identical loss and argmax files**
to clean_before: maximum and mean replay differences are zero. All 32
interventions passed exact device-byte restoration. All 17 frozen input/source/
binary hashes and all 100 on-disk weight hashes are unchanged. All losses are
finite/nonnegative; all argmax IDs are in the logical vocabulary.

An independent reporter produces identical JSON. Independent raw calculations
also match every primary/secondary per-passage NLL, delta, accuracy, prefix
length, split summary, and selectivity vector. Before real measurements,
231 Python tests and five selected GPU targets passed, including eight
synthetic intervention tests and five argmax tests.

The initial Python discovery command omitted the package root and failed
relative imports; the corrected command below passes. The first provenance
preflight failed closed on tuple-versus-JSON-list metadata before any model
forward. Commit 886f617 fixes that representation bug with a real-checkpoint
regression. No hashes of weights, selected passages, or batch bytes changed.

Full results: [causal_validation_results.json](/home/ubuntu/code/pluto/research/weight_memorization/causal_validation_results.json).
Raw files, manifests, launch record and log remain in
/tmp/pluto-causal-validation.kUA8yE/. The first unexecuted plan is retained;
the actual run uses plan_v2. There are 70 raw files of 131,072 bytes each.

| Artifact | SHA-256 |
| --- | --- |
| plan_v2/manifest.json | 45773585ebda41b8fffed5e188569e537cbd4ca1d129d92642641109d1110fba |
| plan_v2/batch_tokens.bin | 54cc2720fb0831a7b77af974eda9bff880cff47c8e51c7c16cd8b4d38fab26b5 |
| Optimized probe binary | ebd47f5b789467a60c0435c561e246b7ed3521719079b23afe103f4ae320fa7b |
| run/metadata.json | 761ebdd7da8d8210ce0177b218023343fedfd907fcad8aa866bfb83097d8d35a |
| Report / committed results JSON | 13a9d88ddbea0142c9716a3fd7fef2171a8493bcff5ba2a8730b98b2590977b9 |

Reproduce from the repository root with fresh output names. Regenerate the
native export with the documented tokenize_corpus binary if its temporary files
are unavailable. The example output paths must not already exist.

    bazel build -c opt //scripts/weight_analysis:causal_probe
    OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 /home/ubuntu/.venv/bin/python -m unittest discover -s scripts/weight_analysis -t . -p '*_test.py'
    bazel test -c opt //scripts/weight_analysis:token_argmax_test //scripts/weight_analysis:causal_probe_test //src/llm:attention_reference_test //src/llm:cross_entropy_loss_reference_test //src/llm/recipes:gpt2_test --test_output=errors --local_test_jobs=1

    /home/ubuntu/.venv/bin/python -m scripts.weight_analysis.causal_validation plan --corpus testdata/shakespeare.txt --corpus-token-ids /tmp/pluto-native-corpus.KMY7AZ/shakespeare.tokens.bin --corpus-byte-offsets /tmp/pluto-native-corpus.KMY7AZ/shakespeare.tokens.bin.offsets.bin --tokenizer-dir /home/ubuntu/datasets/tokenizer/gpt2 --protocol research/weight_memorization/CAUSAL_VALIDATION_PROTOCOL.md --checkpoint /home/ubuntu/checkpoints/shakespeare/step_13030 --binary bazel-bin/scripts/weight_analysis/causal_probe --output-dir /tmp/causal-new-plan
    bazel-bin/scripts/weight_analysis/causal_probe --checkpoint /home/ubuntu/checkpoints/shakespeare/step_13030 --batch_tokens /tmp/causal-new-plan/batch_tokens.bin --output_dir /tmp/causal-new-run --batch_sequences 4
    /home/ubuntu/.venv/bin/python -m scripts.weight_analysis.causal_validation report --plan /tmp/causal-new-plan/manifest.json --run /tmp/causal-new-run/metadata.json --output /tmp/causal-new-report.json

Changing hashed sources or the binary after planning invalidates authentication;
create a fresh plan instead of bypassing the check. Stashed work is untouched.

## Passage address index

T00..T15 denote training_prefix_00..15; S00..S15 denote current_suffix_00..15
in JSON matrix order. Intervals are zero-based, half-open bytes in the current
testdata/shakespeare.txt. Full windows contain 1,025 tokens; scored intervals
contain the last 512. Boundaries can split a word or UTF-8 character. These are
known-input addresses, not analytically extracted-output addresses.

| ID | Full window bytes | Scored target bytes | Clean NLL | Exact prefix tokens |
| --- | --- | --- | ---: | ---: |
| T00 | [165854, 168708) | [167384, 168708) | 0.244012 | 39 |
| T01 | [463089, 465644) | [464405, 465644) | 0.270803 | 25 |
| T02 | [771640, 774313) | [773063, 774313) | 0.115565 | 26 |
| T03 | [1075363, 1078396) | [1076931, 1078396) | 0.798846 | 6 |
| T04 | [1385557, 1388942) | [1387218, 1388942) | 0.451123 | 11 |
| T05 | [1697695, 1700928) | [1699389, 1700928) | 0.323635 | 6 |
| T06 | [2005744, 2008565) | [2007207, 2008565) | 0.192484 | 18 |
| T07 | [2316245, 2319400) | [2317854, 2319400) | 0.274670 | 15 |
| T08 | [2625436, 2628256) | [2626901, 2628256) | 0.317966 | 10 |
| T09 | [2923268, 2926316) | [2924821, 2926316) | 0.734240 | 5 |
| T10 | [3228763, 3231581) | [3230347, 3231581) | 0.509168 | 7 |
| T11 | [3529383, 3532265) | [3530814, 3532265) | 1.042519 | 2 |
| T12 | [3827816, 3830397) | [3829116, 3830397) | 0.595275 | 15 |
| T13 | [4135224, 4138258) | [4136682, 4138258) | 0.287534 | 10 |
| T14 | [4437236, 4439970) | [4438670, 4439970) | 0.122701 | 35 |
| T15 | [4735800, 4738787) | [4737268, 4738787) | 0.386774 | 4 |
| S00 | [4910132, 4912854) | [4911553, 4912854) | 5.030201 | 1 |
| S01 | [4943524, 4946789) | [4945164, 4946789) | 6.448136 | 1 |
| S02 | [4976676, 4979559) | [4978210, 4979559) | 4.682028 | 0 |
| S03 | [5009331, 5012148) | [5010686, 5012148) | 5.741711 | 0 |
| S04 | [5041482, 5044546) | [5043069, 5044546) | 5.846731 | 1 |
| S05 | [5074238, 5077327) | [5075787, 5077327) | 7.792340 | 0 |
| S06 | [5107510, 5110641) | [5109070, 5110641) | 6.580863 | 0 |
| S07 | [5140614, 5143713) | [5142079, 5143713) | 6.191099 | 0 |
| S08 | [5174475, 5177529) | [5176013, 5177529) | 5.550346 | 1 |
| S09 | [5208762, 5211948) | [5210348, 5211948) | 5.936238 | 0 |
| S10 | [5243166, 5246284) | [5244705, 5246284) | 5.314905 | 1 |
| S11 | [5277025, 5279996) | [5278447, 5279996) | 6.282250 | 0 |
| S12 | [5310890, 5313978) | [5312367, 5313978) | 6.382100 | 3 |
| S13 | [5345197, 5348043) | [5346777, 5348043) | 5.057378 | 3 |
| S14 | [5378297, 5381444) | [5379927, 5381444) | 6.383583 | 0 |
| S15 | [5413756, 5416819) | [5415260, 5416819) | 6.446385 | 0 |

## What remains

We have a coarse, byte-addressed **reliance map**, not an analytical passage
decoder. Finer localization needs controls for shared computation and collateral
damage before assigning passage-specific meaning to neurons or directions.
The last two MLPs are now a concrete, explicitly data-assisted candidate for
that investigation. Any later weight-only extraction must be frozen and
verified separately, and disclose how this validation guided its design.

The earlier static attention-polynomial probe recovered no four-token corpus
substring. These forward measurements do not reverse that result. A separate
[checkpoint-difference audit](/home/ubuntu/code/pluto/research/weight_memorization/CHECKPOINT_DELTA_FEASIBILITY.md)
explains why the existing ten-step, weight-only AdamW snapshots cannot simply
be treated as raw gradients. The desired decompressor remains unresolved.
