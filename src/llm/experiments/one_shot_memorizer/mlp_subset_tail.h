#pragma once

#include <cstdint>
#include <memory>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "src/cuda/executor.h"
#include "src/cuda/page_locked_host_array.h"
#include "src/llm/experiments/one_shot_memorizer/feature_subset.h"
#include "src/llm/gpt2.h"
#include "src/llm/layer.h"

namespace pluto::llm::one_shot_memorizer {

// One cached query and intervention. Views contain raw BF16 bits, not float
// conversions, and are borrowed only for the synchronous Evaluate() call.
struct FinalMlpSubsetInput {
  absl::Span<const uint16_t> gelu;      // Exactly 64 final-MLP GELU channels.
  absl::Span<const uint16_t> residual;  // Exactly 16 pre-MLP residual channels.
  FeatureSubset keep_mask = 0;          // Unretained GELU channels become +0.
};

// Replays only the final contraction, residual addition, LayerNorm and tied
// head, using the existing production BF16/FP32 kernels. Earlier activations
// are supplied by the caller; neither targets nor a search algorithm belong
// here. A zero mask retains the contraction bias and residual skip.
//
// Forward-only and not thread-safe. The executor must outlive this object and
// returned pinned arrays. Create copies its five master tensors: the source
// model is never changed and need not outlive this object. Cached inputs must
// come from the same model snapshot; that provenance cannot be inferred from
// their bytes. Callers should check full-mask equivalence to the source model
// at their actual batch geometries before interpreting interventions.
class FinalMlpSubsetTail final {
 public:
  // Currently accepts unpadded GPT-2 embedding tables, BF16 compute, residual
  // width 16, expansion width 64, and at least one transformer block. Checks
  // model signatures, the entire master inventory, and its final tied alias
  // before allocating/copying tail weights on this executor.
  static absl::StatusOr<std::unique_ptr<FinalMlpSubsetTail>> Create(
      cuda::Executor& executor, const Layer& model, const Gpt2Config& config);

  ~FinalMlpSubsetTail();

  int vocab_size() const;
  int logit_stride() const;  // Vocabulary rounded up to the head's 16 lanes.

  // Returns CPU-ready FP32 logits, with input order preserved. Storage is
  // [round_up(inputs.size(),16), logit_stride()]; extra rows are padding, not
  // samples. Only columns [0,vocab_size()) are real classes. Requires nonempty
  // input and the same executor as Create. All shape/range/executor checks
  // precede GPU allocations or transfers. The call synchronizes its executor.
  absl::StatusOr<cuda::PageLockedHostArray<float>> Evaluate(
      cuda::Executor& executor, absl::Span<const FinalMlpSubsetInput> inputs);

 private:
  struct Impl;
  explicit FinalMlpSubsetTail(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace pluto::llm::one_shot_memorizer
