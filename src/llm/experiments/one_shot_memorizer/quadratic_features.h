#pragma once

#include <cstddef>
#include <memory>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "src/llm/layer.h"

namespace pluto::llm::one_shot_memorizer {

// Checks the fixed BF16 [batch, sequence_length, 16] input shape and computes
// the byte size of [batch, sequence_length, 152]. This allocation-free check is
// also usable before allocating inputs. Rejects empty/partial rows, fractional
// batches, and dimensions exceeding the kernel's signed-int indexing range.
absl::StatusOr<size_t> QuadraticFeaturesOutputBytes(size_t input_bytes,
                                                    int sequence_length);

// An experiment-local, parameter-free polynomial basis, not a learned MLP.
// For each BF16 row x, emits x[0..15], then x[i]*x[j] in lexicographic (i,j)
// order for 0 <= i <= j < 16. There are 16 + 136 = 152 columns; the last is
// x[15]^2. Products use decoded BF16 operands, one FP32 multiplication, and
// one final BF16 conversion. Linear columns preserve their original BF16 bits.
// No centering, rescaling, interaction between rows, or constant column is
// added. A downstream affine projection supplies its own bias.
//
// Forward-only: bwd returns Unimplemented, and no backward intermediates or
// trainable parameters are retained. Finite values may overflow in products;
// IEEE nonfinite results propagate and must be rejected by the experiment's
// existing finite-feature checks. Executor must outlive the layer/results.
class QuadraticFeaturesLayer final : public Layer {
 public:
  static constexpr int kInputWidth = 16;
  static constexpr int kOutputWidth = 152;

  // sequence_length is tokens per sample, not the number of samples.
  static absl::StatusOr<std::unique_ptr<QuadraticFeaturesLayer>> Create(
      cuda::Executor& executor, int sequence_length = 1);

  absl::string_view name() const override { return "QuadraticFeaturesLayer"; }
  DataType output_type() const override { return DataType::BF16; }
  absl::Span<Buffer> weights() override { return {}; }
  absl::Span<const ActivationType> input_types() const override {
    return absl::MakeConstSpan(&input_type_, 1);
  }
  absl::Span<const ActivationType> output_types() const override {
    return absl::MakeConstSpan(&output_type_signature_, 1);
  }

 private:
  QuadraticFeaturesLayer(cuda::Executor& executor, int sequence_length)
      : executor_(executor), sequence_length_(sequence_length) {}

  absl::StatusOr<FwdResult> fwd_impl(cuda::Executor& executor,
                                     absl::Span<const Buffer> inputs,
                                     LayerHooks*) const override;
  absl::StatusOr<BufferVec> bwd_impl(cuda::Executor& executor,
                                     absl::Span<const Buffer> gradients,
                                     BackwardState state, LayerHooks*) override;

  cuda::Executor& executor_;
  int sequence_length_;
  const ActivationType input_type_{
      DataType::BF16,
      {ActivationType::kBatchDimension, sequence_length_, kInputWidth}};
  const ActivationType output_type_signature_{
      DataType::BF16,
      {ActivationType::kBatchDimension, sequence_length_, kOutputWidth}};
};

}  // namespace pluto::llm::one_shot_memorizer
