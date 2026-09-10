#pragma once

#include <memory>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "scripts/weight_analysis/head_context/selection.h"
#include "src/llm/layer.h"

namespace pluto::weight_analysis::head_context {

struct Result {
  cuda::Buffer context;
  // Every row and all 50,272 physical columns, not just the selected query.
  cuda::Buffer logits;
};

// An analysis-only replay of the production GPT-2 tail after attention.
// Context scaling is performed exactly in BF16 by the CPU helper, uploaded
// from page-locked storage, then evaluated by the unchanged native projection,
// residual addition, MLPs/attention blocks, final norm and tied output head.
// No approximation of residual arithmetic, LayerNorm or softmax is used.
//
// Create checks the complete production Tape layout and 101-handle/100-unique
// tied-weight traversal. It independently allocates every replay weight and
// refuses to return unless native attention reconstruction and tail replay
// reproduce clean full context and ALL padded logits at ALL original rows.
//
// The probe retains the original Buffer handles and byte snapshots for tokens,
// selected pre-attention residual/QKV/context, clean logits and all 100
// weights. Apply verifies those originals before and after every successful
// call. A caller must also authenticate checkpoint FILE bytes externally: this
// library receives device buffers, not paths, and cannot certify disk
// provenance. Keep the Executor alive and never mutate original buffers
// concurrently.
class Probe final {
 public:
  static absl::StatusOr<std::unique_ptr<Probe>> Create(
      cuda::Executor& executor, const llm::Layer& production_model,
      const cuda::Buffer& tokens, const llm::Tape& production_tape,
      const cuda::Buffer& clean_logits, int block);
  ~Probe();

  // The selection must name this probe's block. Every dose, INCLUDING 1,
  // genuinely uploads the declared context and executes the complete tail.
  // It also checks other sequences and causally unaffected rows byte-for-byte.
  absl::StatusOr<Result> Apply(cuda::Executor& executor,
                               const Selection& selection) const;

  absl::Status VerifyOriginals(cuda::Executor& executor) const;
  const cuda::Buffer& original_context() const;
  const cuda::Buffer& clean_logits() const;

 private:
  struct Impl;
  explicit Probe(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace pluto::weight_analysis::head_context
