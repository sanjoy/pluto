#pragma once

#include "absl/status/statusor.h"
#include "src/llm/experiments/memorize_general_facts/mlp_readout.h"

namespace pluto::llm::memorize_general_facts {

// The exact four-block GPT-2 suffix after A3: MLP3, attention4, MLP4,
// and final LN. Both width fields equal config.feed_forward_width; no budget
// matching or widening applies. mlp_parameters counts the two bare MLPs,
// while trainable_parameters includes all three pre-LNs and attention.
absl::StatusOr<MlpReadoutParameterBudget>
ResolveMlpTransformerReadoutParameterBudget(const Gpt2Config& config);

// Builds residual pre-LN MLP -> residual pre-LN causal attention -> residual
// pre-LN MLP -> final LN -> frozen tied head. The 20 trainable tensors have
// exactly the source suffix's order (source tensors 32..51). At width 10 and
// feed-forward width 20, the two MLPs have 860 parameters and the complete
// trainable suffix has 1380 parameters.
//
// Fresh MLPs use .2/.1 FC1/FC2 normal initialization with seeds seed..seed+3.
// Attention QKV/output use .02/.005 and seeds seed+4/seed+5. All pre-LNs
// start at identity; biases start at zero. Final LN and the frozen embedding
// are independent source copies. copy_source_tail additionally copies every
// trainable source-suffix tensor, providing an exact architecture control.
// Only readout.trainable may be optimized; no source buffers are shared.
absl::StatusOr<MlpReadout> CreateMlpTransformerReadout(
    cuda::Executor& executor, const Layer& source, const Gpt2Config& config,
    int seed, bool copy_source_tail = false);

}  // namespace pluto::llm::memorize_general_facts
