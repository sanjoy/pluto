#pragma once

#include <memory>

#include "absl/status/statusor.h"
#include "src/llm/gpt2.h"
#include "src/llm/layer.h"
#include "src/llm/layers/combinators.h"
#include "src/llm/layers/embedding.h"

namespace pluto::llm::fit_attention_readout {

// A replacement suffix with a trainable MLP and optional trainable final LN.
// The embedding must outlive the head in model, hence the declaration order.
struct Readout {
  std::unique_ptr<EmbeddingLookupLayer> embedding;  // Frozen tied head weights.
  std::unique_ptr<ComposedLayer> model;             // MLP -> final LN -> head.
  Layer* branch;  // Owned by model; the six-tensor residual MLP alone.
  // Owned by model; branch alone, or branch followed by final LN (eight
  // tensors). Pass this layer to the optimizer and save/restore its complete
  // checkpoint.
  Layer* trainable;
};

// Controls fresh initialization without changing the readout architecture.
// Defaults reproduce the original seeded experiments exactly. Checkpoint
// restoration, when requested by the caller, overrides these initial values.
struct ReadoutInitializationOptions {
  // Standard deviation of the first MLP matrix; must be finite and positive.
  float input_standard_deviation = 0.2f;
  // Standard deviation of the second matrix; zero starts with no correction.
  float output_standard_deviation = 0.1f;
  // Multiply the output deviation by sqrt(source MLP width / readout width).
  // This controls the increased initial residual variance in wider readouts.
  bool scale_output_by_width = false;
  // Initialize final LN's affine parameters to identity instead of copying
  // the source. Requires fresh_branch and train_final_norm.
  bool fresh_final_norm = false;
};

// Copies one original block's pre-LN/residual MLP and the original final LN
// and embedding. All copies are independent: source is never modified. The
// input is that block's post-attention residual vector, not its normalized
// vector. A nonnegative random_seed reinitializes only the two MLP matrices;
// their biases and the pre-LN parameters retain checkpoint initialization.
// -1 warm-starts all branch parameters from the checkpoint. With fresh_branch
// true (requires random_seed >= 0), none of the six branch tensors is copied:
// LayerNorm starts with gamma=1/beta=0 and both affine biases start at zero.
// replacement_width overrides the MLP's hidden width only; zero retains the
// checkpoint width. A different width requires fresh_branch and a seed, since
// the checkpoint's dense tensors no longer have compatible shapes. The source,
// residual width, final LayerNorm, and vocabulary head remain unchanged.
// train_final_norm includes final LN's gamma/beta in the trainable suffix,
// initialized from the same checkpoint unless initialization.fresh_final_norm
// is set. The optimizer/checkpoint
// then covers eight tensors, not six; branch still exposes only the six MLP
// tensors for restoring an existing MLP-only fit. The source and tied
// vocabulary head are always frozen. model keeps the order: six MLP tensors,
// two final LN tensors, then the embedding.
absl::StatusOr<Readout> CreateReadout(
    cuda::Executor& executor, const Layer& source, const Gpt2Config& config,
    int block, int random_seed = -1, bool fresh_branch = false,
    int replacement_width = 0, bool train_final_norm = false,
    const ReadoutInitializationOptions& initialization = {});

}  // namespace pluto::llm::fit_attention_readout
