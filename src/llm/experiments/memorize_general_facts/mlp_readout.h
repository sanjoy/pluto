#pragma once

#include <cstdint>
#include <memory>

#include "absl/status/statusor.h"
#include "src/llm/gpt2.h"
#include "src/llm/layer.h"
#include "src/llm/layers/combinators.h"
#include "src/llm/layers/embedding.h"

namespace pluto::llm::memorize_general_facts {

// Compare all replacement affine layers against the entire original
// suffix after A3. The shared frozen token embedding/head is excluded on both
// sides; the source count includes every suffix bias and LayerNorm parameter.
struct MlpReadoutParameterBudget {
  int minimum_mlp_width;  // Smallest positive width meeting the source budget.
  int mlp_width;          // Requested width, optionally widened to the minimum.
  int64_t source_tail_parameters;  // MLP3, later blocks, and final LayerNorm.
  int64_t mlp_parameters;        // All FC1/FC2 layers, including their biases.
  int64_t trainable_parameters;  // MLPs plus input and final LayerNorms.
};

// Pure shape calculation, with overflow/backend-limit validation. A request
// below the minimum is widened when match_parameter_budget is true. Otherwise
// the requested width is used exactly, even below the source-suffix budget.
absl::StatusOr<MlpReadoutParameterBudget> ResolveMlpReadoutParameterBudget(
    const Gpt2Config& config, int requested_min_mlp_width, int mlp_depth = 1,
    bool match_parameter_budget = true);

// The head is tied to an independent, frozen copy of the source embedding.
// Its owner precedes model so the embedding outlives the head. Source need not
// outlive this object; none of its buffers is changed or shared with training.
struct MlpReadout {
  std::unique_ptr<EmbeddingLookupLayer> embedding;
  std::unique_ptr<ComposedLayer> model;
  // Owned by model. Only this suffix may enter the optimizer/checkpoint.
  // CreateMlpReadout gives 6*depth+2 tensors; the experimental transformer
  // builder also includes an attention sublayer and its input LayerNorm.
  Layer* trainable;
  MlpReadoutParameterBudget parameter_budget;  // Actual allocated shape.
};

// Builds mlp_depth residual blocks x += FC2(GELU(FC1(input_LN(x)))), then
// head(final_LN(x)). Input LNs start at identity, biases at zero, FC1/FC2 at
// normal stddev .2/.1 using seed+2*i/seed+2*i+1 for block i. Initialization is
// not depth-scaled. Final LN is an independent, trainable copy of the source
// final LN. The frozen head follows the 6*depth+2 trainable tensors. At model
// width 10, trainable parameters total depth*(21*mlp_width+30)+20.
// parameter_budget reports any widening needed when matching the sum of all
// affine MLPs to the original suffix; disabled matching preserves exact width.
absl::StatusOr<MlpReadout> CreateMlpReadout(
    cuda::Executor& executor, const Layer& source, const Gpt2Config& config,
    int mlp_width, int seed, int mlp_depth = 1,
    bool match_parameter_budget = true);

struct PuzzleCapture {
  Buffer hidden;  // BF16 post-attention residual in transformer_block_2.
  Buffer logits;  // FP32 full-source logits, for baseline verification only.
};

// Runs the source with integer token sequences, retaining the FIRST residual
// output of block 2, before its MLP. Later blocks run only to provide logits;
// their activations never enter hidden. Saved backward state is discarded.
absl::StatusOr<PuzzleCapture> CaptureThirdAttention(cuda::Executor& executor,
                                                    const Layer& source,
                                                    const Buffer& tokens);

}  // namespace pluto::llm::memorize_general_facts
