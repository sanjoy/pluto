#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "src/cuda/executor.h"
#include "src/llm/layer.h"

namespace pluto::llm::completion_trace {

// An owning snapshot of a layer output restricted to the active causal prefix.
// Shape includes a resolved batch dimension of one. Attention snapshots have
// shape [1, heads, prefix, prefix]; other snapshots retain their channel axes.
struct TensorSnapshot {
  std::string scope;        // Slash-separated enclosing combinator names.
  std::string name;         // Layer diagnostic name.
  size_t occurrence = 0;    // Disambiguates repeated names within this scope.
  size_t output_index = 0;  // Ordered layer output buffer number.
  DataType data_type = DataType::FP32;
  std::vector<int64_t> dimensions;
  std::vector<float> values;       // Row-major decoded numeric values.
  std::vector<uint8_t> raw_bytes;  // Original bits, with the same prefix slice.
};

// One full forward evaluation, with no future generated or gold tokens as
// input. Unused physical context slots contain EOS and are omitted from
// snapshots.
struct ForwardTrace {
  std::vector<int> prefix;
  std::vector<TensorSnapshot> activations;  // Actual hook execution order.
  std::vector<TensorSnapshot> attention;    // Causal probabilities, head-major.
  std::vector<float> next_logits;  // Last prefix row, logical vocabulary only.
};

// Batch-one model capture using read-only LayerHooks and pinned host transfers.
// Requires INT32 [batch, context] input and FP32 [batch, context, vocabulary]
// output. Captures all layer outputs at every active prefix position, including
// complete physical logit rows; future padding is never interpreted as text.
// BF16, FP16, FP32, and INT32 tensors retain their original bits. FP8 is
// rejected because the backend has no defined FP8 format/scaling policy to
// decode.
absl::StatusOr<ForwardTrace> TraceForward(cuda::Executor& executor,
                                          const Layer& model,
                                          absl::Span<const int> prefix,
                                          int eos_token, int vocabulary_size);

// The same forward protocol without installing hooks. Used to verify that
// instrumentation preserves every logical next-token logit bit-for-bit.
absl::StatusOr<std::vector<float>> PredictNextLogits(
    cuda::Executor& executor, const Layer& model, absl::Span<const int> prefix,
    int eos_token, int vocabulary_size);

}  // namespace pluto::llm::completion_trace
