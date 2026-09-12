#pragma once

#include <filesystem>
#include <memory>

#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "experiments/mlp_automaton/top_transitions.h"
#include "src/cuda/executor.h"
#include "src/cuda/page_locked_host_array.h"
#include "src/llm/layer.h"
#include "src/llm/recipes/gpt2.h"

namespace pluto::llm::mlp_automaton {

// Defaults describe the GPT-2 recipe checkpoint. Smaller dimensions allow
// numerical tests without depending on an external trained checkpoint.
struct Dimensions {
  int vocab_size = kGpt2VocabularySize;
  int model_width = kGpt2ModelWidth;
  int feed_forward_width = kGpt2FeedForwardWidth;
};

// A synthetic readout, not a complete language model:
//   x = E[token]
//   h = x + FC2(GELU(FC1(LN2_B0(x))))
//   logits = LN_final(h) E^T
// There are no positions, attention, or subsequent transformer blocks. Uses
// the production BF16 layers with FP32 master weights, reductions and logits.
// The tied embedding remains owned by the returned composition.
absl::StatusOr<std::unique_ptr<Layer>> CreateReadout(
    cuda::Executor& executor, const Dimensions& dimensions = {});

// Reads only the nine tensors needed by CreateReadout from the GPT-2 recipe's
// checkpoint layout: 0, 8..13, 98, 99. Checks all sizes and finite FP32 values
// before changing any device weight. Unneeded checkpoint files are ignored.
// The caller must pass a layer constructed by CreateReadout. All H2D staging
// is page-locked and remains alive until the copies finish.
absl::Status LoadB0Weights(cuda::Executor& executor, Layer& readout,
                           const std::filesystem::path& directory);

// Evaluates each logical token exactly once, in independent tiled batches.
// Returns only the best token/probability per row, not the O(vocab^2) logits.
// batch_size must be a positive multiple of 16; the final batch is padded
// with token zero and those extra rows are discarded. Padded output vocabulary
// columns never enter the softmax denominator. Calls progress(completed)
// after each batch has completed. The callback must not throw.
absl::StatusOr<cuda::PageLockedHostArray<TopTransition>> ScanVocabulary(
    cuda::Executor& executor, const Layer& readout, int vocab_size,
    int batch_size, absl::FunctionRef<void(int)> progress);

}  // namespace pluto::llm::mlp_automaton
