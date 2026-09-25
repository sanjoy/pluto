#pragma once

#include <memory>

#include "absl/status/statusor.h"
#include "src/llm/gpt2.h"
#include "src/llm/layer.h"
#include "src/llm/layers/combinators.h"
#include "src/llm/layers/embedding.h"

namespace pluto::llm::fit_attention_readout {

// A replacement suffix, with only its residual MLP branch trainable. The
// embedding must outlive the head in model, hence the declaration order.
struct Readout {
  std::unique_ptr<EmbeddingLookupLayer> embedding;  // Frozen tied head weights.
  std::unique_ptr<ComposedLayer> model;             // MLP -> final LN -> head.
  Layer* trainable;  // Owned by model; pass only this layer to the optimizer.
};

// Copies one original block's pre-LN/residual MLP and the original final LN
// and embedding. All copies are independent: source is never modified. The
// input is that block's post-attention residual vector, not its normalized
// vector. A nonnegative random_seed reinitializes only the two MLP matrices;
// their biases and the pre-LN parameters retain checkpoint initialization.
// -1 warm-starts all branch parameters from the checkpoint. With fresh_branch
// true (requires random_seed >= 0), none of the six branch tensors is copied:
// LayerNorm starts with gamma=1/beta=0 and both affine biases start at zero.
absl::StatusOr<Readout> CreateReadout(cuda::Executor& executor,
                                      const Layer& source,
                                      const Gpt2Config& config, int block,
                                      int random_seed = -1,
                                      bool fresh_branch = false);

}  // namespace pluto::llm::fit_attention_readout
