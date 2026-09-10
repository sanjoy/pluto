# Embedding rows transfer early spelling preference, not endpoint completion

The first verified intervention changes **11 embedding rows**, transferring
their values from the replacement-trained model into the original-trained
model at the same training step. At step 100, changing the output dictionary
alone reverses `Exeunt` versus `Nuveth` preference in all 39 tested contexts,
including lowercase. Changing only their input embeddings does not. Changing
both gives a much larger improvement in replacement-word probability.

**That early result does not generalize to the endpoint.** The just-completed
step-331 main assay shows no output-only preference reversal in any of those
39 contexts. The joint edit reverses only the three bare-title cases, not any
of the 29 leading-title or seven lowercase cases. The initial-token change
still transfers, but the developed suffix-completion behavior does not.
Shared-piece controls also show large collateral effects: this is not an edit
isolated to a whole word.

The endpoint is asymmetric: a subsequent reverse-direction main assay does
flip all 32 title-case preferences under the joint edit, but still fails to
restore the donor's high absolute completion probability. The robust late
finding is failure to recover donor-like word completion, **not universal
failure to change preference**.

## Intervention and scope

`ROOT=/home/ubuntu/checkpoints/exeunt_nuveth_deterministic_20260910_1137`;
`AMENDMENT=ROOT/lowercase_amendment`; results are in
`AMENDMENT/causal_stage_recovery_20260910_2020/`.
`step_100_replacement_to_original_word_rows` supplies the first completed
three-suite stage. `step_331_replacement_to_original_word_rows` supplies the
additional endpoint **main-suite** comparison below. This report does not
claim completion of other interventions or the full causal stage. A later
main-only comparison from `step_331_original_to_replacement_word_rows` is
explicitly separated below; its supplemental suites are outside this report.

The selected token IDs are
`[45,68,303,400,409,1475,2797,3109,14364,21733,45177]`, covering the native
leading/bare title and lowercase tokenizations. All other dictionary rows,
position embeddings, transformer blocks, and final LayerNorm are held at the
original-trained recipient's values. No training or optimizer update occurs.

| Cell | Input embedding rows | Output dictionary rows |
| --- | --- | --- |
| AA | Original | Original |
| AJ | Original | Replacement |
| JA | Replacement | Original |
| JJ | Replacement | Replacement |

The off-diagonal cells deliberately untie input and output use. **JJ is not the
complete replacement model**: it is the original model with these 11 rows
changed in both roles.

Only original-prefix-domain observations are aggregated here: title training
14 leading / 2 bare, title test 15 leading / 1 bare, lowercase training 7
leading. There is no lowercase test group. Replacement-prefix aliases are not
extra observations. Cases use 128-token prefixes at local positions starting
at zero, not the training stream's 1,024-token history. Title cases were
variant-stratified at known occurrences without using model outputs; results
are diagnostic, not corpus-frequency-weighted or arbitrary-context estimates.

Probabilities are unit fractions at temperature 1. Unless marked arithmetic,
they are geometric means across contexts. A word event is the exact three
native IDs; its suffix probability conditions on the supplied first ID.
Later targets are teacher-forced. No free-generation run is implied, nor are
alternative tokenizations or all possible word delimiters summed.

## Step 100: preference reversal includes an absolute probability increase

Full-word geometric probabilities:

| Group | Candidate | AA | AJ: output | JA: input | JJ: joint |
| --- | --- | ---: | ---: | ---: | ---: |
| Title train, leading, n=14 | Exeunt | 4.91465e-9 | 6.99931e-14 | 1.59310e-9 | 1.26189e-13 |
| | Nuveth | 4.42571e-15 | 6.80580e-11 | 2.98131e-16 | 3.08324e-9 |
| Title test, leading, n=15 | Exeunt | 5.04746e-9 | 7.35182e-14 | 1.64539e-9 | 1.32326e-13 |
| | Nuveth | 4.59650e-15 | 6.97115e-11 | 3.06340e-16 | 3.14090e-9 |
| Title train, bare, n=2 | Exeunt | 1.63363e-11 | 1.27070e-15 | 1.15467e-11 | 1.53284e-15 |
| | Nuveth | 1.65855e-14 | 1.13276e-11 | 1.19172e-14 | 1.94464e-11 |
| Title test, bare, n=1 | Exeunt | 3.12208e-11 | 2.11910e-15 | 2.30506e-11 | 2.61242e-15 |
| | Nuveth | 2.26564e-14 | 1.79674e-11 | 1.69661e-14 | 2.98483e-11 |
| Lowercase train, n=7 | exeunt | 1.00744e-11 | 9.39563e-14 | 9.69172e-12 | 1.04236e-13 |
| | nuveth | 5.85910e-13 | 2.27855e-12 | 2.61072e-13 | 4.13270e-12 |

AJ and JJ favor the replacement in **every** context within every group;
AA and JA favor the original in every context. Mean log preference, defined
here as `log P(replacement)/P(original)`, is:

| Group | AA | AJ | JA | JJ |
| --- | ---: | ---: | ---: | ---: |
| Title train, leading | -13.9203 | +6.87972 | -15.4914 | +10.1037 |
| Title test, leading | -13.9091 | +6.85459 | -15.4965 | +10.0748 |
| Title train, bare | -6.89261 | +9.09543 | -6.87617 | +9.44830 |
| Title test, bare | -7.22840 | +9.04532 | -7.21423 | +9.34360 |
| Lowercase train | -2.84459 | +3.18846 | -3.61423 | +3.68003 |

For all 16 title test contexts, arithmetic full-word means in AA/AJ/JA/JJ
order are `4.74031e-9 / 7.01339e-14 / 1.54815e-9 / 1.24624e-13` for Exeunt
and `5.75193e-15 / 6.66862e-11 / 1.38961e-15 / 2.99742e-9` for Nuveth.
Both aggregation methods therefore show increased replacement probability,
not just preference reversal by damaging Exeunt. Absolute word probabilities
are nevertheless tiny at this early checkpoint.

## Initial selection and suffix completion behave differently

For the replacement candidate, each entry below is geometric
`first-piece P / conditional two-piece suffix P`. Their product gives its
full-word probability above.

| Group | AA | AJ | JA | JJ |
| --- | ---: | ---: | ---: | ---: |
| Title train, leading | 2.07158e-7 / 2.13640e-8 | .00148346 / 4.58778e-8 | 2.07158e-7 / 1.43915e-9 | .00148346 / 2.07841e-6 |
| Title test, leading | 2.13701e-7 / 2.15090e-8 | .00153291 / 4.54765e-8 | 2.13577e-7 / 1.43433e-9 | .00153213 / 2.05002e-6 |
| Title train, bare | .000160018 / 1.03648e-10 | .000278186 / 4.07196e-8 | .000160018 / 7.44744e-11 | .000278186 / 6.99045e-8 |
| Title test, bare | .000217228 / 1.04298e-10 | .000457270 / 3.92926e-8 | .000217228 / 7.81024e-11 | .000457270 / 6.52750e-8 |
| Lowercase train | 2.99721e-5 / 1.95485e-8 | 4.37856e-5 / 5.20388e-8 | 2.99301e-5 / 8.72273e-9 | 4.37331e-5 / 9.44983e-8 |

For comparison, leading-title test Exeunt has AA first/suffix
`.00185397 / 2.72251e-6`; AJ `4.34397e-5 / 1.69242e-9`; JA
`.00185332 / 8.87807e-7`; JJ `4.34171e-5 / 3.04779e-9`.
Input-only changes primarily hurt suffix probabilities here, while joint
input/output changes make the replacement suffix much stronger than
output-only changes. Small nonzero first-piece input effects in some groups
are possible because edited IDs can already occur in their 128-token prefix;
absence of such exposure was checked by exact full-logit equality.

The independently authenticated fixed-rival decomposition gives a more
specific account for all 16 title test Nuveth cases. Summed over three targets,
the input-only change in log probability is **-2.55716**: target-minus-fixed-
rival margin improves by **+6.06358**, but relative log normalization rises
by **+8.62073**. Output-only gives **+9.44239**, and joint gives **+13.0440**.
The factorial interaction is **+6.15879 = +6.16117 margin - .00238226
normalizer**. Consequently, the joint gain is not merely nonlinear softmax
normalization of otherwise additive target-relative logits. The rival is
the highest AA non-target logit, fixed separately for each prediction across
all four cells. These contrasts are not mediation fractions or proof of a
unique storage location.

## Controls rule out a whole-word-selective interpretation

The 16 generic next-three-token controls per split change little. Mean
**summed three-token NLL** changes AJ/JA/JJ relative to AA are
`+.00022008 / +.00019153 / +.00041264` in training and
`+.00034201 / 0 / +.00034201` in test. This panel alone would miss important
collateral effects on the selected tokens in other contexts.

For shared-piece controls, index 0 is the named piece itself, not its next
token. Selected examples of mean NLL change (positive is worse):

| Piece ID / split | n | AJ: piece at index 0 | JJ: piece at index 0 | JA: following token at index 1 |
| --- | ---: | ---: | ---: | ---: |
| 1475, leading Ex / train | 8 | +3.33073 | +3.33073 | +.552267 |
| 1475, leading Ex / test | 3 | +3.26168 | +3.26209 | +.386223 |
| 2797, unt / train | 8 | +3.96015 | +3.95767 | -1.93137 |
| 68, e / train | 8 | +.983247 | +.983249 | -.133380 |
| 68, e / test | 8 | +.191754 | +.191437 | -.0625040 |

There are 109 shared-piece cases in the complete suite; the table is not a
claim of exhaustive vocabulary coverage. Other pieces can improve instead:
AJ changes train `ve` (303) by -.776735 and `eth` (400) by -.854605 nats.
The intervention acts on reusable token directions, not only the three-ID
word events.

The supplemental word suite scores one **exact native following token** after
the supplied three-ID candidate. It is not a test of any delimiter or of
word termination in general. Mean NLL changes relative to AA:

| Candidate / split | AJ | JA | JJ |
| --- | ---: | ---: | ---: |
| Exeunt / train | -.000444 | +.198092 | +.197285 |
| Nuveth / train | -.000779 | -.081861 | -.082455 |
| Exeunt / test | -.000463 | +.516530 | +.515721 |
| Nuveth / test | -.000787 | -.244949 | -.245561 |
| exeunt / train | -.000397 | -.364936 | -.365759 |
| nuveth / train | -.000787 | +.119996 | +.119417 |

## Endpoint main comparison: initial-token transfer is not enough

The step-331 main probe finished at **20:36:11 UTC on 2026-09-10**. It uses
the same intervention definition and cases, but the two step-331 checkpoints.
For all 16 title test contexts:

| Cell | Exeunt full-word P | Nuveth full-word P | Mean log P(Nuveth)/P(Exeunt) |
| --- | ---: | ---: | ---: |
| AA | .00173831 | 1.55950e-16 | -30.0422 |
| AJ | 2.61259e-7 | 2.13688e-12 | -11.7139 |
| JA | 5.24452e-5 | 9.75621e-17 | -27.0103 |
| JJ | 2.15919e-8 | 7.15573e-9 | -1.10441 |

AJ flips 0/16 title training, 0/16 title test, and 0/7 lowercase contexts;
JJ flips 2/16, 1/16, and 0/7, respectively. Its three flips are exactly the
bare-title contexts. Endpoint lowercase JJ still favors exeunt: geometric
P(exeunt)=1.81242e-10 versus P(nuveth)=4.33840e-12.

For the 15 **leading-title test** contexts, replacement first-piece
probability changes from AA `1.05195e-8` to AJ `.00139949`; JJ is `.00139855`.
But its suffix probability is only `1.26086e-9` under AJ and `5.54311e-6`
under JJ. JJ full-word probability is `7.75234e-9`. The actual complete
replacement model's corresponding leading-title full-word probability is
about `.00188805`; its conditional suffix probability is `.883236`.
Thus swapping the selected rows does not recover that model's endpoint
spelling completion, even though the output-side initial-token switch works.

This motivates testing additional dictionary rows and transformer pathways
or their coadaptation. It does not yet tell us which unchanged component is
responsible. The stronger endpoint original-word baseline also means early
and late preference reversal are different thresholds, not interchangeable
notions of recovered behavior. Component interventions are needed before
localizing the missing completion behavior.

### Reverse endpoint main: asymmetric preference, same completion limitation

The reverse main assay finished at **20:40:05 UTC** and also independently
regenerated exactly. Here AA is the **replacement-trained recipient**, and J
uses the original-trained donor's selected rows; do not interpret these
cells as the same four models in the preceding tables.

For title training/test, AJ favors donor Exeunt in 11/16 and 10/16 contexts;
JA in 0/16 and 0/16; JJ in 16/16 and 16/16. Lowercase behaves differently:
AJ and JJ favor donor exeunt in 0/7, while JA does in 7/7. This rules out a
single direction-independent account based only on which path was patched.

Across all 16 title test contexts, geometric Exeunt probability is AA
`7.64047e-13`, AJ `3.74312e-11`, JA `3.17136e-12`, JJ `5.13779e-9`;
the actual original-trained donor is about `.00173831`. Recipient Nuveth
probability falls from AA `.00117324` to JJ `1.67928e-14`.
JJ therefore increases donor spelling probability, but its strong relative
preference is accompanied by collapse of the recipient spelling and remains
far from the donor's absolute completion behavior. For the 15 leading-title
test contexts, Exeunt JJ has first-piece P `.00141051` and suffix P
`3.28870e-6`, versus the donor suffix P `.745008`. Initial selection again
transfers much more successfully than conditional suffix completion.

## Independent verification and evidence identities

The step-100 main (188 cases), following-token (156), and shared-piece (109)
readouts were regenerated in temporary directories from every saved logical-
vocabulary FP32 logit row and compared exactly with their saved JSON values.
All three have verified recorded execution provenance. The audit checked
560 distinct frozen/input/output records before and after computation, and
the completion marker's exact 125-file inventory. All 1,872 shared full-
vocabulary rows (156 cases x 3 targets x 4 cells) are byte-identical between
main and following-token suites. The complete native step-100 executions
ended at 20:28:22, 20:29:42, and 20:30:40 UTC. No discrepancies were found.

The step-331 main readout separately regenerated exactly, authenticating its
356 recorded files and execution (PID 1902763, return code 0). The reverse
endpoint main likewise regenerated exactly with its own 356-file ledger
(PID 1903824, return code 0). The fixed-rival mechanism report's source ledger
was rehashed and its aggregate effects
independently calculated. These are CPU-only checks; they do not rerun native
GPU kernels or independently attest to a process's execution. Native
full-padded-row diagonal equality and device no-mutation remain checked
assertions from the recorded probes. Readout checks independently verify
on-disk weight selection, finite logical logits, probability calculations,
identical-prefix invariance, unexposed-input invariance, and unchanged
unselected output columns.

SHA-256 identities, relative to the step-100 stage unless otherwise specified:

| Artifact | SHA-256 |
| --- | --- |
| complete.json | f1acdf2932a74c338c78bceb869fa03bff373e05ff49fbc33ed1b16cbae893ce |
| factorial_main_readout.json | 074389ff41a585ab77f13aa5c54e373c85c5a7882e2c6de05b5f6a5a07d32a2a |
| factorial_word_next_native_readout.json | c9417f69a7c4006b318a35893be51c891f27e3b211c9b064c0edfb6c1c246b99 |
| factorial_shared_piece_readout.json | 97a2f8d8a022b623db34fcb1337ad4a4440795a8812b6682bc25b49edbbb77a9 |
| step_100/patch.json | d95a56d0ed0e4b0a522b3978e268d2e94ce55590fabc6e052a2ae5291cd2098f |
| AMENDMENT/step100_replacement_to_original_main_embedding_mechanism.json | 2c2dbeaaf56582fe4d6b2c4ba8140edf970a075315caba85d89f26c294bb9a39 |
| Step-331 stage/factorial_main_readout.json | ba5dad1e353079d03c458636630fba5cb0e6449c75a17f7b7795768ee36b43b3 |
| Reverse step-331 stage/factorial_main_readout.json | d8c844260ac33a8d22a4f53b11ef6507fbd749ab1b2018dd8ff88484976f3685 |

The endpoint complete-model comparison uses the separately verified
`AMENDMENT/analysis_trajectory/summary.json` (SHA-256
`e625a92b82b58f33a74041c32d4b5fd74dded4ef634d3a0a86213d4371235bf7`),
described in `EXEUNT_PAIRED_BEHAVIOR_TRAJECTORY.md`. Its stored FP32-loss
rounding differs slightly from factorial scores recomputed in FP64 from
native logits; rounded values are compared here, not asserted byte-identical
across the two scoring methods. No source, checkpoint, or live evidence file
was modified for this report.
