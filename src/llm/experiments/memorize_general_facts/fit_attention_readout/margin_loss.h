#pragma once

#include "absl/status/statusor.h"
#include "src/cuda/buffer.h"
#include "src/cuda/executor.h"

namespace pluto::llm::fit_attention_readout {

// All results stay on the input executor; rows retain their original order.
struct MarginLossResult {
  cuda::Buffer losses;     // FP32[rows], unnormalized squared violations.
  cuda::Buffer gradients;  // FP32[rows, stride], d(sum(loss)/normalizer)/dz.
  cuda::Buffer predicted_ids;  // int32[rows], lowest-ID top-1; -1 for ignored.
  cuda::Buffer margins;        // FP32[rows], target minus strongest competitor.
};

// Computes max(0, required_margin + max_{v != target} z_v - z_target)^2.
// logits contains FP32 rows, optionally with vocabulary padding; targets is
// int32[rows]. At least two vocabulary entries are required. The positive
// normalizer is supplied by the caller (normally the number of scored rows);
// it affects gradients, not reported losses. Equal competitors choose the
// smallest ID, giving a deterministic subgradient at ties.
//
// Targets equal to -1 are ignored WITHOUT reading their logits: loss, margin,
// and gradients are zero, and prediction is -1. Vocabulary-padding gradients
// are always zero, even when padding logits are NaN or infinite. Other invalid
// target IDs or nonfinite logical logits produce prediction -2 and NaN loss,
// margin, and logical gradients. Device values are checked asynchronously; the
// caller must reject those invalid results before applying an optimizer step.
absl::StatusOr<MarginLossResult> SquaredMarginLoss(cuda::Executor& executor,
                                                   const cuda::Buffer& logits,
                                                   const cuda::Buffer& targets,
                                                   int vocabulary_size,
                                                   float required_margin,
                                                   int normalizer);

}  // namespace pluto::llm::fit_attention_readout
