# Finite-state-machine language-model experiment

By default, this trains the same transformer as Shakespeare from scratch:
8 pre-LayerNorm transformer blocks, width 512, 8 attention heads, MLP width
2,048, GELU, learned absolute positions, context 1,024, no dropout, and a tied
1,029-token FSM embedding/head. There are exactly 1,029 stored embedding rows;
only intermediate logits are tile-padded. The model has 26,271,232 parameters.
Activations use BF16; master weights, AdamW state, reductions, and losses use FP32.
`--layers=N` changes only the number of transformer blocks; the default is 8.
`--attention_heads=N` changes how the fixed width 512 is partitioned; the default
is 8 heads of width 64. N must be positive and divide 512. For example,
`--attention_heads=2` gives two 256-dimensional heads. Q/K/V and output
projection sizes, parameter count, and seeded initialization stay unchanged.

Each dataset line is `transition[;transition...];input>output`. Every transition
is `000X093`: source state, letter, destination state. Execution starts at 000;
the answer is either the final three-digit state/`ERR`, or a full execution
trace. The loader independently executes every sentence to verify the entire
output. All ASCII spaces are ignored, including spaces inside transitions;
newlines still separate examples, and tabs are not accepted.

For example, `000X999;X>999` succeeds, while `000X999;Y>ERR` fails.

The separate `finite_state_machine_full_training_data.txt` and
`finite_state_machine_full_test_data.txt` contain the same 4,096/128 prompts in
the same order, but output every visited state, starting with `000`:

```text
000X093;093A044;XA>000 093 044
000X093;093A044;XB>000 093 ERR
```

Repeated states are retained. The trace stops immediately at the first missing
transition; `ERR` is not followed by states for the remaining input. Regenerate
these copies with
`python3 scripts/generate_finite_state_machine_data.py --full_trace`.
This leaves the original final-answer datasets unchanged.

The tokenizer is built in; no downloaded tokenizer is needed. State `NNN` has
token ID NNN (0 through 999); A through Z have IDs 1000 through 1025; `;`, `>`,
and `ERR` have IDs 1026, 1027, and 1028 respectively. `ERR` is atomic in the
answer field, but the letters E/R/R in an input are three separate input tokens.
Every state or `ERR` occupies one token. Spaces are not part of the vocabulary;
decoding returns the canonical text with spaces removed. Unsupported characters
and incomplete numeric triples are errors rather than unknown-token substitutions.

The default objective is **answer-only next-token cross entropy**: the model
predicts all output tokens after `>` (one token for final-only data, multiple
for traces). During training/evaluation, each trace token is predicted using
the preceding ground-truth trace tokens, not the model's generated tokens
(teacher forcing). There is no EOS token or EOS prediction: trace targets end
at the final state or first `ERR`. The entire description and input are visible
causal context,
but their targets and all right padding are masked. State 000 doubles as a
padding input; padding never contributes a target or influences preceding
causal positions. Lines are never concatenated or truncated.
`--answer_only=false` instead scores every available next token in the sample,
still without adding EOS.

## Train

Run from the repository root:

```sh
bazel build -c opt //src/llm/experiments/finite_state_machine:finite_state_machine

bazel-bin/src/llm/experiments/finite_state_machine/finite_state_machine \
  --training_data=testdata/finite_state_machine_training_data.txt \
  --test_data=testdata/finite_state_machine_test_data.txt \
  --checkpoint_dir=/home/ubuntu/checkpoints/finite_state_machine/simple_tokens_0/checkpoints \
  --batch_size=4 --steps=10000 --seed=17 \
  --learning_rate=0.0003 --checkpoint_every=100 --eval_every=100 \
  --eval_samples=128 --answer_only=true
```

Checkpoints are weights-only `step_N` directories, written atomically at step 0,
every 100 updates, and the final update. AdamW moments are not checkpointed.
A nonempty checkpoint directory is rejected to avoid overwriting another run.
The binary intentionally has no resume or inference mode yet.

To try the larger 16-block model, use the same invocation with `--layers=16`
and a fresh `--checkpoint_dir`, for example
`/home/ubuntu/checkpoints/finite_state_machine/simple_tokens_16_layers_0/checkpoints`.
This has 51,490,304 parameters. Width, heads, MLP width, context, tokenizer,
AdamW settings, batch size, data order, and evaluation coverage stay unchanged.
The shared GPT-2 recipe keeps its existing residual initialization scaling;
this experiment changes depth, not the initialization recipe or learning rate.

To train that same 16-block model on the full traces instead, use:

```sh
bazel-bin/src/llm/experiments/finite_state_machine/finite_state_machine \
  --layers=16 \
  --training_data=testdata/finite_state_machine_full_training_data.txt \
  --test_data=testdata/finite_state_machine_full_test_data.txt \
  --checkpoint_dir=/home/ubuntu/checkpoints/finite_state_machine/full_trace_16_layers_0/checkpoints \
  --batch_size=4 --steps=10000 --seed=17 \
  --learning_rate=0.0003 --checkpoint_every=100 --eval_every=100 \
  --eval_samples=128 --answer_only=true
```

For the two-head full-trace comparison, add `--attention_heads=2` to the command
above and use the fresh checkpoint directory
`/home/ubuntu/checkpoints/finite_state_machine/full_trace_16_layers_2_heads_0/checkpoints`.
Keep the remaining flags, including `--layers=16`, unchanged.

Each evaluation reports mean cross entropy in **nats per supervised output
token**, not answer accuracy. For full traces, this includes the initial `000`
and any terminal `ERR`, and longer traces contribute more tokens to the mean.
Low teacher-forced loss does not guarantee correct autoregressive execution.
Final-answer and full-trace losses are not directly comparable, nor are they to
the older GPT-2-tokenizer run, which also scored EOS and sometimes multiple
answer tokens. By default training loss uses a fixed first 128
training examples; test loss uses all 128 held-out examples. `--eval_samples=0`
evaluates all training examples. Positive sample limits round up to a full
batch, capped at corpus size, and the exact coverage is logged. Evaluation uses
separate unshuffled iterators and never changes the training sample schedule.
Training shuffles all 4,096 examples without replacement each epoch.

Logs include UTC timestamps to the second and go to stdout and
`checkpoint_dir/train.log` (or `--log_file`). AdamW uses beta1=0.9, beta2=0.95,
epsilon=1e-8, weight decay=0.1, and constant learning rate, matching Shakespeare
defaults. No gradient clipping is used. The default 10,000-step budget is an
initial training run, not a claim that the model will learn the FSM algorithm.

## Tests

```sh
bazel test -c opt //src/llm/experiments/finite_state_machine:all \
  --test_output=errors
```

Tests cover depth/head selection and invalid CLI arguments, every vocabulary entry,
exact tokenization, decoding, spaces, ERR ambiguity, valid and malformed traces,
syntax/execution validation, prompt/padding masks, exact loss counts, partial
batches, deterministic shuffling, buffer reuse, real-corpus tokenization without
external dependencies, and a small GPT-2 training and
checkpoint round trip for both final answers and variable-length traces.
The small test model does not change the binary's fixed
default Shakespeare-size configuration.
