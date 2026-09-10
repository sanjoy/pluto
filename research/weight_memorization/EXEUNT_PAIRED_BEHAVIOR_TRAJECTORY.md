# Paired behavior: preference appears early; spelling completion develops later

The nine native checkpoint reports show a clear spelling-dependent behavioral
change, **not evidence that the sampled passages are memorized**. By the first
saved training checkpoint, step 100, every sampled original-prefix context
favors its model's trained spelling over the alternative. Most of the later
increase in leading-title word probability comes from completing the suffix,
not from making the first piece likely in its context. No word candidate has
all three teacher-forced argmax predictions correct at any measured checkpoint.

## Scope and verification

ROOT is `/home/ubuntu/checkpoints/exeunt_nuveth_deterministic_20260910_1137`;
AMENDMENT is `ROOT/lowercase_amendment`; evidence is under
`AMENDMENT/analysis_trajectory`. The completed analysis summary is timestamped
2026-09-10 19:55:49 UTC. Actual endpoints are **331 versus 331 steps**. The
summary's generic warning about unequal endpoint steps does not apply here.

This readout uses only the **original-prefix domain**, with no duplicate
replacement-prefix observations: 16 title-case training contexts (14 leading,
2 bare), 16 title-case test contexts (15 leading, 1 bare), and all 7 lowercase
training contexts (all leading). There are no lowercase test contexts.
Controls are 16 separate, shared-prefix next-three-token cases in each split.

The frozen selection records `model_outputs_used=false`: title-case contexts
were sampled with native-token-variant coverage, not chosen for model success.
Nevertheless, this is a variant-stratified diagnostic sample, not a
corpus-frequency-weighted sample or arbitrary-context evaluation. Word cases
are selected at known word occurrences; all seven lowercase occurrences are
included. Every tested prefix contains **128 tokens**, with positions starting
at zero, rather than the training stream's 1,024-token windows. Its available
history and local position IDs need not match an occurrence's training window.
Similar training/test curves alone therefore establish neither generalization
nor the absence of memorization.

All nine reports were regenerated using the frozen scoring source in temporary
directories and compared exactly with their saved JSON values. An independent
FP64 calculation read the native FP32 loss and integer argmax dumps directly.
The check rehashed 116 distinct frozen/source/case/score/report/terminal records
and separately checked nine process logs and exact probe arguments. No GPU
work or training restart was performed for this analysis. Checkpoint execution
and training provenance remain recorded in the completed upstream audit.
The common step-0 model was scored once and explicitly aliased, not counted as
two independent measurements.

Here `A` means the original-trained model evaluated on `Exeunt`/`exeunt`; `J`
means the amended model evaluated on `Nuveth`/`nuveth`. At step 0 the weights
are shared, but the candidate token sequences differ. Probabilities below are
unit fractions, **not percentages**. Unless marked arithmetic, they are
geometric means across the named contexts. `suffix` means the probability of
the remaining two IDs conditional on the prefix **and the supplied first ID**.
Thus geometric full-word probability equals geometric first-piece probability
times geometric suffix probability. These are three-ID, temperature-1,
teacher-forced events, not delimiter-terminated strings or sums over alternate
tokenizations.

No autoregressive rollout was performed. Later teacher-forced argmax
predictions receive the correct preceding candidate IDs, not the model's own
earlier predictions. Under the same inference and tie-breaking semantics,
zero all-three matches means greedy decoding from these tested prefixes would
not produce the exact three-ID candidate: it must depart from that forced path
at or before the first mismatching argmax. This is not a measured free-generation
run and does **not** mean zero sampling probability.

## Leading title case: selection improves first, completion follows

Training contexts, n=14; each entry is `A / J`:

| Step | First-piece P | Conditional suffix P | Full-word P |
| ---: | ---: | ---: | ---: |
| 0 | 1.48565e-5 / 9.80646e-6 | 5.68269e-10 / 1.87236e-10 | 8.44248e-15 / 1.83612e-15 |
| 100 | .00180336 / .00177874 | 2.72527e-6 / 2.46611e-6 | 4.91465e-9 / 4.38657e-9 |
| 200 | .00214881 / .00175318 | .00892463 / .00481085 | 1.91773e-5 / 8.43428e-6 |
| 300 | .00212107 / .00201696 | .780215 / .824491 | .00165489 / .00166296 |
| 331 | .00214230 / .00214361 | .742372 / .876177 | .00159038 / .00187818 |

Test contexts, n=15, kept separate; each entry is `A / J`:

| Step | First-piece P | Conditional suffix P | Full-word P |
| ---: | ---: | ---: | ---: |
| 0 | 1.54031e-5 / 9.89424e-6 | 4.97193e-10 / 1.89397e-10 | 7.65829e-15 / 1.87394e-15 |
| 100 | .00185397 / .00182443 | 2.72251e-6 / 2.37094e-6 | 5.04746e-9 / 4.32561e-9 |
| 200 | .00214220 / .00174397 | .00867706 / .00471810 | 1.85880e-5 / 8.22820e-6 |
| 300 | .00211843 / .00202874 | .783540 / .824984 | .00165987 / .00167368 |
| 331 | .00214403 / .00213765 | .745008 / .883236 | .00159732 / .00188805 |

First-piece probability is already close to its endpoint scale by step 100;
suffix probability rises by roughly five orders of magnitude afterward. The
original model's suffix probability peaks at step 300, rather than improving
monotonically. The similar training/test curves concern this selected sample,
not equal performance over the complete corpora.

## Bare title case is a different tokenization and learning trajectory

`Exeunt` is `[3109,68,2797]`; bare `Nuveth` is `[45,45177,400]`.
Their leading forms instead start with 1475 and 21733 and use `ve` (303),
not `uve` (45177), for the replacement's second piece.

| Step | Bare A train, n=2 | Bare A test, n=1 | Bare J train, n=2 | Bare J test, n=1 |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 1.15083e-14 | 1.13992e-14 | 3.73068e-15 | 4.25887e-15 |
| 100 | 1.63363e-11 | 3.12208e-11 | 1.54321e-11 | 2.47583e-11 |
| 200 | 1.76508e-8 | 2.77655e-8 | 3.58599e-10 | 4.94193e-10 |
| 300 | .00130363 | .00410706 | 2.58267e-8 | 4.29520e-8 |
| 331 | .00465248 | .00618246 | 1.04055e-6 | 9.33268e-7 |

These are full-word geometric probabilities. At the endpoint, J's bare
first-piece probabilities are .00774526 (train) and .00662840 (test), but its
suffix probabilities are only .000134347 and .000140798. In particular,
`N -> uve` has mean NLL 7.63454/7.61330, versus .03536/.03300 for leading
` Nu -> ve`. Pooling bare and leading forms conceals this failure.

There is also an early selection/completion dissociation in A: at steps 100
and 200, all three bare contexts favor the alternative's first ID `N` over
`Ex`, while the complete candidate still favors `Exeunt`. By step 300 the
first-piece preference favors `Ex` too. These are only two training and one
test contexts; no broad bare-form generalization claim follows.

## Lowercase: full-word preference can disagree with first-piece preference

Training only, n=7; each entry is `A / J`:

| Step | First-piece P | Conditional suffix P | Full-word P |
| ---: | ---: | ---: | ---: |
| 0 | 1.67291e-5 / 1.76123e-5 | 6.43027e-10 / 1.88850e-10 | 1.07573e-14 / 3.32609e-15 |
| 100 | 5.38292e-5 / 3.96564e-5 | 1.87155e-7 / 8.47145e-8 | 1.00744e-11 / 3.35947e-12 |
| 200 | 3.07943e-5 / 3.62252e-5 | 3.22546e-6 / 7.44755e-7 | 9.93258e-11 / 2.69789e-11 |
| 300 | 5.78062e-5 / 2.03346e-5 | 6.30624e-5 / 2.93407e-5 | 3.64540e-9 / 5.96632e-10 |
| 331 | 5.43733e-5 / 1.79378e-5 | .000174113 / .000454641 | 9.46710e-9 / 8.15524e-9 |

J favors the complete `nuveth` candidate in all seven contexts from step 100
onward. However, its first-piece choice is nonmonotonic: 4/7 contexts favor
` ex` over ` nu` at step 100, 0/7 at step 200, and 7/7 at steps 300 and 331.
At the endpoint, mean log P(`exeunt`)/P(`nuveth`) is **-9.04130**, composed of
**+0.85837** from the first piece and **-9.89967** from the suffix. Both complete
words nevertheless remain extremely improbable.

The endpoint between-model NLL change J-minus-A for `exeunt` is
`[+.25059,-1.58696,+10.52683]`; for `nuveth`, it is
`[-.88295,+1.15999,-8.72326]`. The final piece is especially important to this
observed contrast. Shared suffix IDs and very sparse lowercase exposure are
plausible explanations to test, not an established weight-level mechanism.

## Preference is not memorization; aggregation matters

At initialization the shared model already weakly favors the original complete
spelling in every sampled context. The trained contrast is that, by step 100,
the original model favors it in 39/39 contexts while J favors the replacement
in 39/39. The observation is bounded by checkpoint spacing: the onset occurred
**by** step 100, not necessarily at step 100.

At step 331, title-case mean log P(`Exeunt`)/P(`Nuveth`) is +29.8241 in A and
-21.0335 in J on training contexts, and +30.0422/-21.1522 on test contexts.
All individual context signs agree. But the trained leading word has only
about .0016--.0019 full probability, and no three-token word argmax match was
observed across the entire measured trajectory. Strong relative preferences
can coexist with poor prediction of the actual next word.

Endpoint arithmetic means of own-trained full-word P are:

| Stratum | A arithmetic / geometric | J arithmetic / geometric |
| --- | ---: | ---: |
| Leading title train | .00161236 / .00159038 | .00188308 / .00187818 |
| Leading title test | .00160520 / .00159732 | .00188978 / .00188805 |
| Bare title train | .00465734 / .00465248 | 1.04175e-6 / 1.04055e-6 |
| Bare title test | .00618246 / .00618246 | 9.33268e-7 / 9.33268e-7 |
| Lowercase train | 1.48434e-8 / 9.46710e-9 | 1.22950e-8 / 8.15524e-9 |

Unrelated controls have 1/16 training and 5/16 test three-token argmax matches
in each trained model at every measured step from 100 onward (0 at initial).
Endpoint mean three-token NLL changes J-minus-A are -.0209731 training and
+.00597519 test. The control arithmetic probabilities change .0377868 to
.0372389 training and .1560513 to .1572551 test; geometric probabilities are
much smaller: 4.55113e-7 to 4.64758e-7 and 8.32079e-5 to 8.27122e-5.
This small control panel does not certify absence of unrelated damage.
Controls use disjoint sampled windows rather than known word-occurrence
positions. Compare the two models' changes within each panel, not the absolute
difficulty of word and control panels.

The next causal tests should distinguish input-embedding effects, output-head
effects, and downstream transformations separately for first-piece selection
and suffix completion. Lowercase's opposite-sign first/suffix preferences and
bare versus leading replacement tokenization provide particularly useful
controls. Native factorial/branch interventions, not this behavioral curve,
must establish which changed weights cause these effects.

## Evidence and audit housekeeping

The completed summary is [analysis_trajectory/summary.json](/home/ubuntu/checkpoints/exeunt_nuveth_deterministic_20260910_1137/lowercase_amendment/analysis_trajectory/summary.json),
638,413 bytes, SHA-256
`e625a92b82b58f33a74041c32d4b5fd74dded4ef634d3a0a86213d4371235bf7`.
It binds the nine score reports, their native process records, frozen cases
and sources, terminal training records, and checkpoint inventory. Report
indices are A331=0, J331=1, shared0=2, A100=3, J100=4, A200=5, J200=6,
A300=7, J300=8. Each `checkpoint_N_behavior.json` binds its actual loss,
argmax, metadata, and case bytes. The endpoint report hashes are
`4eee7bfe94edfecd7afd6070e2b3c1b422bdd6f0c74baea2113ceefa63816073`
and `8dc99d8553d4d8abc797a586c18aa0fcbe83b9883c66c27769c5801350dddeff`.

An earlier independent Python import unintentionally created a bytecode cache
at `AMENDMENT/analysis_trajectory/sources/__pycache__/paired_lowercase_scores.cpython-312.pyc`.
Only that generated file was moved, unchanged, to
`/tmp/pluto-endpoint-readout-cache.q57FUa/paired_lowercase_scores.cpython-312.pyc`;
its SHA-256 before and after was
`abc22479e656263623e174529e9dc572fec8808db7f66795bce4b71d9f0384ec`.
No frozen source or evidence bytes were changed; the empty cache directory
was left alone. Subsequent checks used `PYTHONDONTWRITEBYTECODE=1`.
