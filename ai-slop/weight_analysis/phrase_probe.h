#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "src/cuda/page_locked_host_array.h"
#include "src/llm/layer.h"
#include "src/llm/layers/embedding.h"
#include "src/llm/layers/norm.h"

namespace pluto::weight_analysis {

// A named native tensor, not a converted or recomputed CPU activation. Buffers
// retain the full 1024-row forward allocation; only prompt rows are exported.
struct TraceFrame {
  std::string name;
  cuda::Buffer buffer;
  int width;
  bool fp32 = false;
  bool native_replay = false;
};

// Rejects stale assumptions about the production recipe's private tape tree.
// The diagnostic never changes the recipe or installs forward hooks in it.
absl::Status ValidateGpt2Tape(const llm::Tape& tape);

// Copies matching tensors D2D on the explicit executor. Used only for separate
// replay/readout layers, never to replace model handles or checkpoint files.
absl::Status CopyDeviceWeights(cuda::Executor& executor,
                               absl::Span<const cuda::Buffer> source,
                               absl::Span<cuda::Buffer> destination);

// Replays a production dense kernel on an EXACT saved input. This recovers the
// two branch outputs the residual tape does not retain. It is not subtraction
// of BF16 residual snapshots (which would include rounding error).
absl::StatusOr<cuda::Buffer> ReplayProjection(cuda::Executor& executor,
                                              const cuda::Buffer& input,
                                              const cuda::Buffer& matrix,
                                              const cuda::Buffer& bias,
                                              int input_width,
                                              int output_width);

absl::StatusOr<std::vector<TraceFrame>> CollectGpt2Trace(
    cuda::Executor& executor, const llm::Tape& tape, const cuda::Buffer& logits,
    absl::Span<const cuda::Buffer> weights);

// Read only a contiguous prefix into page-locked memory; every successful
// return has synchronized, so the returned bytes are immediately usable.
absl::StatusOr<cuda::PageLockedHostArray<uint8_t>> ReadPrefix(
    cuda::Executor& executor, const cuda::Buffer& buffer, int rows, int width,
    size_t element_bytes);

// FP32 addition followed by BF16 round-to-nearest-even, matching ResidualLayer.
// Inputs must be finite. This tests replayed branch outputs against the actual
// saved post-add residual, including BF16 rounding rather than a tolerance.
uint16_t AddBf16Bits(uint16_t left, uint16_t right);
absl::Status VerifyResidualReplay(cuda::Executor& executor,
                                  const cuda::Buffer& before,
                                  const cuda::Buffer& branch,
                                  const cuda::Buffer& after, int rows,
                                  int width);

// A diagnostic logit lens: applies the CHECKPOINT'S final LayerNorm and tied
// head to an intermediate residual, using the unmodified production kernels.
// These are diagnostic readouts, not logits emitted by intermediate blocks.
// The embedding owner must outlive its tied head; declaration order ensures it.
class NativeLogitLens final {
 public:
  static absl::StatusOr<std::unique_ptr<NativeLogitLens>> Create(
      cuda::Executor& executor, absl::Span<const cuda::Buffer> weights);
  absl::StatusOr<cuda::Buffer> Embed(cuda::Executor& executor,
                                     const cuda::Buffer& tokens) const;
  absl::StatusOr<cuda::Buffer> Apply(cuda::Executor& executor,
                                     const cuda::Buffer& residual) const;

 private:
  NativeLogitLens(std::unique_ptr<llm::EmbeddingLookupLayer> embedding,
                  std::unique_ptr<llm::LayerNormLayer> norm,
                  std::unique_ptr<llm::LanguageModelingHeadLayer> head);
  std::unique_ptr<llm::EmbeddingLookupLayer> embedding_;
  std::unique_ptr<llm::LayerNormLayer> norm_;
  std::unique_ptr<llm::LanguageModelingHeadLayer> head_;
};

// Strict, duplicate-free comma-separated zero-based block:feature IDs. Empty
// input requests no neuron interventions. Parsing performs no GPU operations.
absl::StatusOr<std::vector<std::pair<int, int>>> ParseNeuronInterventions(
    absl::string_view text);

}  // namespace pluto::weight_analysis
