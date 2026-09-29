# Memory-bounded Qwen training

`qwen_llm --mode=train_model` implements short-sequence, full-parameter text
fine-tuning using block-coordinate Adam (BAdam). It is not LoRA: every
text-decoder parameter can be updated over a complete cycle, but only one
parameter group is active at a time. This implementation and its tests are
AI-generated and not human reviewed.

## Quick smoke run

Use the checkpoint downloaded in [README.md](README.md), then run:

```sh
bazel build -c opt //src/llm/qwen:qwen_llm
bazel-bin/src/llm/qwen/qwen_llm \
  --mode=train_model \
  --checkpoint=/home/ubuntu/checkpoints/models/Qwen3.8-27B-FP8 \
  --batch_size=1 --sequence_length=8 \
  --text='The capital of France is Paris. The capital of Greece is Athens.' \
  --steps=3 --switch_every=1 --start_block=64 --learning_rate=1e-5
```

This repeats one next-token training batch: the first eight tokens are inputs,
and tokens 2..9 are targets. The loss is ordinary cross-entropy averaged over
the eight target positions. Text is tokenized without a chat template. The
driver is a smoke-training entry point, not a corpus/dataloader pipeline or a
validated fine-tuning recipe.

For the 64-block checkpoint, optimizer groups are:

- `0`: token embedding.
- `1..64`: decoder blocks `0..63`, each including attention, MLP and norms.
- `65`: final RMSNorm and untied language-model projection.

Groups cycle downward and wrap. `--start_block=-1` starts with the head;
`--switch_every` defaults to 50 updates per group. The example uses one update
per group to exercise both full attention and DeltaNet quickly. Small-block
cycling saves memory, but these smoke-run hyperparameters are not evidence of
fine-tuning quality.

## Implementation and memory

`TrainingModel` is an ordinary Pluto `Layer`. Each decoder block is assembled
from composed, parallel and residual layers, with full-sequence cuTile forward
and backward implementations for gated full attention and GatedDeltaNet.
DeltaNet differentiates through its entire recurrent history within the sample;
full attention differentiates through all causal query/key/value contributions.
Later frozen blocks still propagate input gradients. Saved state is discarded
only for the prefix preceding the active block.

Training and cached inference use the same `QwenAttentionLayer`. Training
passes a null `KeyValueCache*`, so forward saves the full-sequence state needed
by backward. Supplying a cache instead selects one-token inference and makes
backward return an error; the caller owns and resets that cache.

`BlockParameter` keeps a stable resident tensor shared by its layers and
checkpoint views. Imported FP8 matrices are dequantized once to BF16; norm and
recurrence parameters stay FP32. Training does not use inference's dynamic FP8
activation quantization, so training and FP8 inference are not bitwise equal.
Layer boundaries generally use BF16; reductions, logits, loss, backward
gradients and optimizer calculations use FP32.

`BAdamOptimizer` allocates FP32 master weights, gradients and two Adam moments
only for the active group: **16 bytes per active parameter**, in addition to
resident weights and activations. Publishing an update changes resident data
in place. At a switch, the old working tensors are freed; moments and local
bias-correction step reset for the new group, including on revisits. This
follows Algorithm 1 of the [BAdam paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/2c570b0f9938c7a58a612e5b00af9cc0-Paper-Conference.pdf).

`--max_active_gib` optionally rejects a configuration if *any* group would
exceed that working-state budget. This is not a total-memory cap: allow for
resident weights, activations, temporary buffers and allocator reservations.
Weight decay is zero in the smoke driver; the optimizer API supports optional
decoupled decay via `AdamWConfig`.

The explicit training loop calls `ZeroGrad()` before **every** forward, then
`SetBackwardStart(active_block())`. Switching happens in `ZeroGrad`, never
underneath a saved backward state. Do not pass this optimizer to a loop that
only calls `ZeroGrad` once at startup, including Pluto's current generic
`Train` helper. Multiple microbatches can accumulate gradients before a single
`ApplyStep`; their scaling is the caller's responsibility.

## Scope and checkpointing

- One sample per forward; sequence length 1..128. Longer sequences and batched
  training fail explicitly rather than silently truncating history.
- Each sequence starts with zero attention/recurrent history; there is no
  cross-sequence recurrent state or multimodal training.
- `--save_weights=/new/directory` writes all resident weights in Pluto's
  checkpoint format. Expect roughly 54 GB for this model, even after a small
  update. It never overwrites the original HF checkpoint.
- `--resume_weights=/directory` loads those resident weights after loading the
  original checkpoint architecture. Adam state, schedule position and step
  count are **not** restored; this starts a fresh optimizer cycle.
- Saved weights are not an HF FP8 export and cannot be passed directly to the
  FP8 loader in `qwen_llm --mode=infer_model`. Requantization/export is not
  implemented; `--resume_weights` is valid only in training mode.

## Verification

```sh
bazel test -c opt \
  //src/llm:block_training_test \
  //src/llm:qwen_attention_test \
  //src/llm:sequence_delta_net_test \
  //src/llm:badam_optimizer_test \
  //src/llm/qwen:training_model_test
```

Tests compare operator gradients to independent CPU calculations/finite
differences, including causal history, grouped heads and BF16 straight-through
conversion. A tiny hybrid checkpoint cycles through every parameter group,
checks that all inactive weights remain bitwise unchanged, verifies active
updates and a loss decrease, and compares forward results with cached
inference. Optimizer tests check moment resets, local bias correction, working
memory release, BF16 master accumulation, duplicate parameters and budgets.

### Real checkpoint smoke results

On GH200 (95.6 GiB device memory), using the pinned HF revision, eight-token
sequences, batch size one, three updates, learning rate `1e-5`, and switching
every update:

| Start group / groups updated | Initial mean CE | Final mean CE | Largest recorded device use |
| --- | ---: | ---: | ---: |
| `64`: decoder 63 (full attention), 62 and 61 (DeltaNet) | 2.23560 | 1.79949 | 58.75 GiB |
| `1`: decoder 0 (DeltaNet), embedding, LM head | 2.23560 | 2.17911 | 72.03 GiB |

Each row starts from the original checkpoint independently. The second run
propagates gradients through the complete 64-block suffix when updating the
embedding. Resident weights occupy 53,797,287,936 bytes (50.10 GiB). Active
working state is about 5.55–5.71 GiB for decoder blocks and 18.95 GiB for the
embedding/head. Device use is sampled after synchronized updates and includes
allocator reservations and other device use; it is not a measured transient
peak. Both runs completed without OOM or non-finite loss. The unified binary's
89-target test suite also passes; its `train_model` smoke run reproduces the
first row, and `infer_model` still answers `4` to the chat prompt `What is 2 + 2?
Reply with just the number.`
