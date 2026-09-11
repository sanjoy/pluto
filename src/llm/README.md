# Shakespeare model examples

Run these commands from the repository root. Set the tokenizer path to a GPT-2
tokenizer directory containing `tokenizer.json`. The checked-in configuration
uses CUDA at `/usr/local/cuda` and targets Hopper (`sm_90`); the build and runs
require a compatible CUDA/cuTile toolchain and GPU.

Build the recipe:

```sh
bazel build -c opt //src/llm/recipes:gpt2_shakespeare_llm
export PLUTO_GPT2_TOKENIZER_DIR=/path/to/gpt2
```

Train for 100 updates, saving checkpoints every 50 updates and at completion:

```sh
bazel-bin/src/llm/recipes/gpt2_shakespeare_llm \
  --mode=train_model \
  --corpus=testdata/shakespeare.txt \
  --batch_size=1 \
  --steps=100 \
  --checkpoint_dir=/path/to/checkpoints/shakespeare \
  --checkpoint_every=50
```

Resume from the latest valid `step_N` child of the checkpoint parent directory
for 100 additional updates:

```sh
bazel-bin/src/llm/recipes/gpt2_shakespeare_llm \
  --mode=train_model \
  --corpus=testdata/shakespeare.txt \
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
bazel-bin/src/llm/recipes/gpt2_shakespeare_llm \
  --mode=infer_model \
  --inference_from=/path/to/checkpoints/shakespeare/step_200 \
  --prompt='To be, or not to be' \
  --generation_tokens=100 \
  --temperature=0
```

Omit `--prompt` for an interactive prompt loop. `--temperature=0` uses greedy
decoding; a positive temperature enables sampling.
