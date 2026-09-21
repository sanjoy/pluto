#pragma once

#include <cstddef>
#include <type_traits>

#include "absl/status/statusor.h"
#include "src/cuda/buffer.h"
#include "src/cuda/executor.h"
#include "src/llm/layer.h"

namespace pluto::llm {

// The three highest-probability distinct vocabulary IDs for one activation row.
// This POD is copied directly from device memory by callers that need text.
struct TopThreeTokens {
  int tokens[3];
  float probabilities[3];
};
static_assert(std::is_trivially_copyable_v<TopThreeTokens>);
static_assert(std::is_standard_layout_v<TopThreeTokens>);
// The cuTile writer uses separate strided views into these packed fields.
static_assert(sizeof(int) == sizeof(float));
static_assert(offsetof(TopThreeTokens, tokens) == 0);
static_assert(offsetof(TopThreeTokens, probabilities) == 3 * sizeof(int));
static_assert(sizeof(TopThreeTokens) == 24);

// Stable temperature-one softmax over columns [0, logical_vocab), returning
// device-resident TopThreeTokens[rows]. Ties prefer the smaller token ID.
// row_stride includes any physical vocabulary padding, which is never read.
// logits must contain whole FP32 rows, at least the requested leading rows.
// Any nonfinite logical logit invalidates its row: all IDs -1, probabilities
// NaN. Padding and rows beyond the requested prefix do not affect the result.
//
// Requires rows > 0, logical_vocab >= 3, row_stride >= logical_vocab, and all
// buffers on executor. Enqueues work without host copies or synchronization.
absl::StatusOr<cuda::Buffer> ReadTopThreeTokens(cuda::Executor& executor,
                                                const cuda::Buffer& fp32_logits,
                                                int rows, int logical_vocab,
                                                int row_stride);

// Reads activations in the token embedding basis: logits = activations * E^T,
// followed by the same full-vocabulary softmax/top-three reduction above.
// This is a diagnostic readout, not a cosine similarity or model prediction:
// no normalization, bias, learned final LayerNorm, or scaling is added.
//
// storage is the PHYSICAL activation dtype: FP32 or BF16 (not the legacy FP16
// compute policy). Activations contain at least rows complete width-element
// rows; E is row-major FP32 with at least logical_vocab complete rows. Padding
// in E and activation rows beyond the prefix are ignored. Dot products use a
// fixed-order FP32 tile reduction. Nonfinite projected logits use the same
// invalid-row sentinel as ReadTopThreeTokens.
//
// Processes at most 16 rows at a time, reusing one bounded logits allocation;
// the result alone grows with rows. Returns device TopThreeTokens[rows] with
// no host copies or synchronization. Executor must outlive all buffers.
absl::StatusOr<cuda::Buffer> ReadEmbeddingNeighbors(
    cuda::Executor& executor, const cuda::Buffer& activations, DataType storage,
    const cuda::Buffer& fp32_embedding, int rows, int logical_vocab, int width);

}  // namespace pluto::llm
