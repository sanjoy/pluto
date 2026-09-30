# Finite-state-machine language-model experiment

This trains the same GPT-2 recipe as the Shakespeare experiment from scratch:
8 pre-LayerNorm transformer blocks, width 512, 8 attention heads, MLP width
2,048, GELU, learned absolute positions, context 1,024, no dropout, and a tied
50,257-token GPT-2 embedding/head (50,272 physical embedding rows). Activations
use BF16; master weights, AdamW state, reductions, and losses use FP32.

Each dataset line is `transition[;transition...];input;output`. Every transition
is `000X093`: source state, letter, destination state. Execution starts at 000;
the answer is the final three-digit state or `ERR` for a missing transition.
The loader independently executes every sentence to verify its answer.

The default objective is **answer-only next-token cross entropy**, including
EOS after the answer. The entire description and input are visible causal
context, but their targets and all right padding are masked. Lines are never
concatenated or truncated. `--answer_only=false` instead scores all next tokens
and EOS, as an ordinary language-model objective. No vocabulary reduction is
performed; all predictions use the original GPT-2 vocabulary.

## Train

Run from the repository root with an installed GPT-2 tokenizer:

```sh
bazel build -c opt //src/llm/experiments/finite_state_machine:finite_state_machine

bazel-bin/src/llm/experiments/finite_state_machine/finite_state_machine \
  --tokenizer_dir=/home/ubuntu/datasets/tokenizer/gpt2 \
  --training_data=testdata/finite_state_machine_training_data.txt \
  --test_data=testdata/finite_state_machine_test_data.txt \
  --checkpoint_dir=/home/ubuntu/checkpoints/finite_state_machine/gpt2_0 \
  --batch_size=4 --steps=10000 --seed=17 \
  --learning_rate=0.0003 --checkpoint_every=100 --eval_every=100 \
  --eval_samples=128 --answer_only=true
```

Checkpoints are weights-only `step_N` directories, written atomically at step 0,
every 100 updates, and the final update. AdamW moments are not checkpointed.
A nonempty checkpoint directory is rejected to avoid overwriting another run.
The binary intentionally has no resume or inference mode yet.

Each evaluation reports mean cross entropy in **nats per scored answer/EOS
token**, not answer accuracy. By default training loss uses a fixed first 128
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
  --test_env=PLUTO_GPT2_TOKENIZER_DIR=/home/ubuntu/datasets/tokenizer/gpt2 \
  --test_output=errors
```

Tests cover syntax/execution validation, variable-length answers, token-boundary
checks, prompt/padding masks, exact loss counts, partial batches, deterministic
shuffling, buffer reuse, real-corpus tokenization, and a small GPT-2 training and
checkpoint round trip. The small test model does not change the binary's fixed
Shakespeare-size configuration.
