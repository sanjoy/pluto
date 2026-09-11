#pragma once

#include <cstdint>
#include <memory>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "src/llm/layer.h"

namespace pluto::weight_analysis {

// A single value edge in the production GPT-2 attention computation. Query
// and source are positions within the same 1024-token sequence, not flattened
// batch offsets. Sources after the query are allowed as causal null controls.
struct SourceValueSelection {
  int block;
  int head;
  int sequence;
  int query_position;
  int source_position;
  float scale;
};

// CPU-only validation. Only the predeclared doses 0, 0.5, and 1 are accepted.
// The full row count must fit native kernels' signed-int element counters.
absl::Status ValidateSourceValueSelection(const SourceValueSelection& selection,
                                          int total_rows);

// Exact finite BF16 scaling for the three supported doses. Zero produces
// positive zero, half rounds to nearest/even (including subnormal ties), and
// one preserves the input bits, including negative zero. No CUDA is used.
absl::StatusOr<uint16_t> ScaleSourceBf16(uint16_t value, float scale);

struct SourceValueResult {
  cuda::Buffer modified_qkv;
  // Attention recomputed with the changed V vector at all queries. This is
  // exported for checking, but it is NOT passed wholesale to the model tail.
  cuda::Buffer replayed_attention;
  // Clean context except for precisely the selected query/head's 64 values.
  cuda::Buffer spliced_context;
  cuda::Buffer logits;
};

// Analysis-only native suffix replay. It owns separately allocated copies of
// every tail weight, uses the real AttentionLayer, dense, residual-add, MLP,
// LayerNorm, and tied-head kernels, and never changes the production model.
// Create fails unless clean attention and ALL padded logits at ALL rows match
// the actual CreateGpt2 forward byte for byte. Apply retains that full shape.
//
// The intervention changes only one source V vector, leaving Q/K and hence
// native softmax statistics unchanged. Only the selected query/head output is
// spliced back: this is neither a whole-head ablation nor key masking with
// renormalization. Later computations are rerun normally.
//
// Keep the executor alive and do not concurrently mutate any supplied model
// weights or tape buffers. The caller must independently snapshot/verify the
// model weights and checkpoint files; this helper checks all saved activation
// inputs against its own original byte snapshots on every successful Apply.
class SourceValueProbe final {
 public:
  static absl::StatusOr<std::unique_ptr<SourceValueProbe>> Create(
      cuda::Executor& executor, const llm::Tape& production_tape,
      const cuda::Buffer& clean_logits,
      absl::Span<const cuda::Buffer> original_weights, int block);
  ~SourceValueProbe();

  absl::StatusOr<SourceValueResult> Apply(
      cuda::Executor& executor, const SourceValueSelection& selection) const;

  const cuda::Buffer& original_qkv() const;
  const cuda::Buffer& original_context() const;
  const cuda::Buffer& clean_logits() const;

 private:
  struct Impl;
  explicit SourceValueProbe(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace pluto::weight_analysis
