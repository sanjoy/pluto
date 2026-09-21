# Shakespeare model examples

Run these commands from the repository root. Set the tokenizer path to a GPT-2
tokenizer directory containing `tokenizer.json`. The checked-in configuration
uses CUDA at `/usr/local/cuda` and targets Hopper (`sm_90`); the build and runs
require a compatible CUDA/cuTile toolchain and GPU.

Build the Shakespeare experiment:

```sh
bazel build -c opt //src/llm/experiments/shakespeare:gpt2_shakespeare_llm
export PLUTO_GPT2_TOKENIZER_DIR=/path/to/gpt2
```

Train for 100 updates, saving checkpoints every 50 updates and at completion:

```sh
bazel-bin/src/llm/experiments/shakespeare/gpt2_shakespeare_llm \
  --mode=train_model \
  --corpus=testdata/shakespeare_dataset.txt \
  --batch_size=1 \
  --steps=100 \
  --checkpoint_dir=/path/to/checkpoints/shakespeare \
  --checkpoint_every=50
```

Resume from the latest valid `step_N` child of the checkpoint parent directory
for 100 additional updates:

```sh
bazel-bin/src/llm/experiments/shakespeare/gpt2_shakespeare_llm \
  --mode=train_model \
  --corpus=testdata/shakespeare_dataset.txt \
  --batch_size=1 \
  --resume_from=/path/to/checkpoints/shakespeare \
  --steps=100 \
  --checkpoint_every=50
```

New checkpoints default to the resume directory. Checkpoints restore model
weights and logical step numbering; optimizer and data-iterator state restart,
so resuming is not equivalent to uninterrupted training.

Generate one completion from an exact checkpoint directory:

```sh
bazel-bin/src/llm/experiments/shakespeare/gpt2_shakespeare_llm \
  --mode=infer_model \
  --inference_from=/path/to/checkpoints/shakespeare/step_200 \
  --prompt='To be, or not to be' \
  --generation_tokens=100 \
  --temperature=0
```

Omit `--prompt` for an interactive prompt loop. `--temperature=0` uses greedy
decoding; a positive temperature enables sampling.

Inspect the prompt's activations at every layer without generating a completion:

```sh
bazel-bin/src/llm/experiments/shakespeare/gpt2_shakespeare_llm \
  --mode=infer_model \
  --inference_from=/path/to/checkpoints/shakespeare/step_200 \
  --prompt='To be, or not to be' \
  --generation_tokens=0 \
  --inspect_activations='neighboring_vocab(min_prob=0.01)'
```

Each heading includes the zero-based position and original input token, such as
`Position 6 (" be"):` (including the token's leading space). Under each heading,
this prints up to three tokens per layer
under `softmax(activation * token_embedding^T)`; the LM head uses its existing
logits. Probabilities are normalized over the whole vocabulary, independently
of sampling temperature. Intermediate readouts are diagnostics, not the model's
final next-token predictions. Tokens with probability below `min_prob` are
omitted, without renormalizing the survivors. The default is `0.01` (1%);
bare `neighboring_vocab` uses that default, and `min_prob=0` shows all three.
Quote the argument when including parentheses. Incompatible outputs (such as
the wider MLP expansion), layers with no surviving tokens, and empty position
sections are omitted. Hierarchical layer names distinguish repeated blocks.
The model is named `gpt2`, its blocks are `transformer_block_0` through
`transformer_block_7`, and each block's composed branches are `attention` and
`mlp`.
Only the prompt is inspected, once; prompts longer than the context limit use
their last 1,024 tokens, retaining the original position numbers.
