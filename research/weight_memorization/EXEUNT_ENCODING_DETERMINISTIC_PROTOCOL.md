# How the weights encode Exeunt: deterministic paired experiment

Protocol recorded before inspecting the new experiment's learned weights.
This is a new study; the preserved `exeunt_nuveth_20260910` experiment used
nondeterministic backward kernels and is exploratory evidence only.

## Objective and evidentiary standard

Explain how GPT-2's parameters and computations cause the model to predict
`Exeunt` in context and complete its multi-token spelling. Training two models
and ranking their weight differences are means to that end, not success criteria.
An embedding row is both an input representation and, in this tied model, an
output classifier. A large row difference therefore does not establish a unique
storage location, an input-side mechanism, or a functioning word generator.

Separate three questions throughout:

1. What makes the initial subtoken likely in a relevant context?
2. What makes the remaining subtokens likely given the preceding spelling?
3. Which parameter changes actually cause these behaviors, rather than merely
   correlate with them?

## Controlled training

- Eight-block production GPT-2 architecture, deterministic implementation at
  commit `08edf21940f06344b2b649d631b8716b6fb8e0a2`, BF16 training, seed 17,
  batch size 10 sequences of 1,024 tokens.
- Original Shakespeare versus every exact case-sensitive `Exeunt` replaced by
  `Nuveth`. No other corpus edits. Native tokenizer exports must demonstrate
  equal token counts, three token slots per replacement, identical token IDs
  outside those slots, and unchanged outer byte boundaries and split location.
- Identical saved random initialization, architecture, tokenizer, optimizer,
  train/test split, and seeded random-window sampling. Both arms start with
  fresh AdamW state; neither resumes an already-trained optimizer trajectory.
- AdamW: learning rate 0.0003, beta1 0.9, beta2 0.95, epsilon 1e-8, weight decay
  0.1. Test fraction 0.1; four evaluation batches; evaluation every 100 steps.
- Two unchanged-corpus controls must match every weight file byte-for-byte
  (verified with SHA-256) at steps 0, 1, and 2. Missing or mismatched checkpoints
  abort before either long arm. An additional two-step replacement control is
  retained. This is a tested same-stack guarantee, not a cross-hardware claim.
- Four training hours per arm, sequentially on the same GPU, original then
  replacement. Preserve step 0, every 100th step, and the final checkpoint in a
  new directory. Never overwrite the historical experiment or restart an arm
  automatically after failure. Weight-only checkpoints do not contain optimizer
  or sampler state.
- Freeze binaries, corpora, tokenizer files, native sampling evidence, and
  launcher source provenance. Record wall-clock endpoints and completed steps.
  Different endpoint step counts must not be treated as matched-step evidence.
- Do not run competing GPU diagnostics during either timed arm. CPU analysis
  and compilation may proceed; their possible wall-clock effects are a reason
  to emphasize matched optimizer steps rather than endpoints alone.

## Frozen behavior cases and trajectories

Before examining new learned weights, prepare the existing native-token case
suite with seed 17, 16 word occurrences per split, 16 unrelated continuations
per split, and up to 128 preceding tokens. Include both candidate words and both
original/replacement prefix domains. Preserve exact token IDs, packed batches,
selection rules, and hashes. Deduplicate identical prefixes when aggregating;
the two domains do not constitute independent observations when IDs match.

The training split is for discovery and the held-out corpus split for
confirmation. Report both tokenization variants separately as well as their
aggregate. Variant-stratified cases are not corpus-frequency-weighted.

At initialization, every common periodic checkpoint, and both endpoints, report:

- Per-subtoken conditional NLL/probability and complete three-token sequence
  log probability for each word; retain absolute probabilities, not just odds.
- First-subtoken selection versus the second/third-subtoken spelling behavior.
- Greedy successes and, where complete logits are collected, vocabulary ranks.
- Unrelated-continuation loss and collateral effects on shared subwords outside
  `Exeunt`; generic unrelated windows alone cannot rule out subword damage.
- A separately identified following-boundary check. Three-token probability
  alone does not show that the model stops the word or emits a valid delimiter.

All candidate suffixes are teacher-forced with that candidate's preceding IDs.
Short prefixes are placed at position zero in the native context; these tests
are not necessarily the same position/context as a sampled training window.
Use native full-vocabulary softmax and temperature 1 for likelihoods. Legacy
sampling-reconstruction scripts must not be assumed equivalent to the current
production sampler without an explicit compatibility check.

## Preselected causal tests

### 1. Disentangle the tied input embedding and output dictionary

The native tokenizer identifies the union of nine word-piece rows:
`45, 68, 303, 400, 1475, 2797, 3109, 21733, 45177`. Verify this set against the
frozen case suite rather than selecting rows from weight-delta rankings.

For recipient A and J (A with only those rows copied from donor B), run a 2x2
assay: final residual from A or J crossed with output dictionary from A or J.
Keep A's final LayerNorm parameters fixed. The A/A and J/J diagonal cells must
reproduce their full native forward logits byte-for-byte. The off-diagonal
cells separate input-side effects, output-side effects, and their interaction.
Repeat in both donor directions, with clean replay and unchanged-weight checks.

Report each word's absolute likelihood and each subtoken's effect. A large
fraction of transferred log odds can result from destroying both words; do not
call that successful transfer of the donor's spelling behavior. Near-zero
baseline differences make transfer fractions unstable and should be reported as
absolute changes instead.

### 2. Identify the upstream computations that matter

Use matched checkpoints to transfer complete attention or MLP branches and
measure their effects on word cases and unrelated controls. Distinguish complete
branch transfers from output-write-only interventions: a feature detector and
the direction it writes into the residual stream are different mechanisms.
Use zero/half/full doses, exact restoration, and clean replays. Confirm any
discovery on held-out contexts and more than one matched checkpoint.

Only then narrow reproducible effects to selected heads or neuron groups.
Include fixed-seed size/norm-matched random groups and shared-subword controls.
Attention-head dependence does not, by itself, identify which source token was
routed; source-specific routing interventions are a separate required test if
that is the proposed explanation.

### 3. Explain causal effects with the actual computation

Capture native residual states, attention inputs/outputs, and MLP gates for each
predicted subtoken. Trace selected contributions into signed target-versus-rival
logit margins, explicitly retaining final-LayerNorm, BF16-rounding, and reduction
residuals. Accounting identities and intermediate logit lenses are diagnostics,
not substitutes for the intervention results above.

## Reporting and completion

Publish partial findings, including null results and failures, as they become
available. Preserve raw evidence, source/binary hashes, case hashes, checkpoint
hashes, actual commands, and verified process status. Historical exploratory
results must not be relabeled as deterministic paired results.

Completion requires an evidence-backed computational and causal explanation of
how the weights support `Exeunt`, with context/subtoken specificity, collateral
controls, and clearly bounded conclusions. Do not claim a unique storage site
or universal mechanism from a few successful interventions. Unresolved parts
remain explicit work rather than being hidden by a successful training run.
