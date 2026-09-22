#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "src/cuda/executor.h"
#include "src/llm/layer.h"

namespace pluto::llm::memorize_general_facts {

struct ActivationTraceBoundary {
  std::string name;
  // Raw, unnormalized residual-stream values in [token, model_width] order.
  std::vector<float> values;
};

struct ActivationTrace {
  std::string prompt;
  std::string continuation;
  size_t prompt_token_count = 0;
  int model_width = 0;
  // IDs/text for all actual prompt and generated tokens, without EOS/padding.
  // IDs are in the original tokenizer vocabulary, even for a compact model.
  std::vector<int> token_ids;
  std::vector<std::string> token_texts;
  std::vector<ActivationTraceBoundary> boundaries;
};

// Replays one causal forward pass over the complete prompt and continuation,
// including the last generated token (which ordinary generation has not yet
// fed back into the model). Records the summed token/position embedding and
// each transformer block's residual output, before the final LayerNorm. Prior
// positions are unaffected by the extra future tokens because attention is
// causal. Only actual token rows are downloaded; EOS-filled padding is not.
//
// complete_tokens and eos_token use MODEL IDs (compact IDs when applicable).
// vocabulary_size must be the model tokenizer's actual vocabulary size,
// excluding logit padding; all IDs are checked against it before GPU work.
// The recipe must expose the usual named, top-level GPT-2 boundaries and a
// single INT32 [-2, context] input. Exactly width 16 is supported by this
// visualization; FP32 and BF16 physical activations are converted to float.
absl::StatusOr<std::vector<ActivationTraceBoundary>>
CaptureActivationBoundaries(cuda::Executor& executor, const Layer& model,
                            absl::Span<const int> complete_tokens,
                            int vocabulary_size, int eos_token,
                            int transformer_blocks, int model_width);

}  // namespace pluto::llm::memorize_general_facts
