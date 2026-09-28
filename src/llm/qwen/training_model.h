#pragma once

#include <filesystem>
#include <functional>
#include <memory>
#include <vector>

#include "src/llm/badam_optimizer.h"
#include "src/llm/layer.h"
#include "src/llm/qwen/checkpoint.h"

namespace pluto::llm::qwen {

// Fixed short training sequences are independent, with zero initial attention
// history. One sample per forward; callers may accumulate several microbatches.
struct TrainingModelOptions {
  int sequence_length = 8;                      // Supported range: 1..128.
  std::function<void(int, int)> load_progress;  // Loaded decoder blocks/total.
};

// Full-sequence Qwen text decoder using ordinary Pluto layers/combinators.
// Imported FP8 matrices are dequantized once to resident BF16; scalar weights
// stay FP32. There is no dynamic FP8 activation quantization in training.
// BAdam allocates optimizer state only for one disjoint parameter group:
// embedding, each decoder block, and final norm + untied head, in that order.
class TrainingModel final : public Layer {
 public:
  static absl::StatusOr<std::unique_ptr<TrainingModel>> Load(
      cuda::Executor& executor, const std::filesystem::path& checkpoint,
      const TrainingModelOptions& options = {});
  absl::string_view name() const override { return "QwenTrainingModel"; }
  DataType output_type() const override { return DataType::FP32; }
  absl::Span<const ActivationType> input_types() const override;
  absl::Span<const ActivationType> output_types() const override;
  absl::Span<Buffer> weights() override { return absl::MakeSpan(weights_); }
  const Config& config() const { return config_; }
  const std::vector<BAdamBlock>& parameter_blocks() const {
    return parameters_;
  }
  size_t resident_weight_bytes() const;

  // Set this after BAdam::ZeroGrad selects its active block, before forward.
  // Earlier blocks still execute forward, but their backward state is dropped.
  // Frozen later blocks must propagate input gradients: freezing is not detach.
  absl::Status SetBackwardStart(int block);

 private:
  TrainingModel(cuda::Executor& executor, Config config,
                std::vector<std::unique_ptr<Layer>> blocks,
                std::vector<BAdamBlock> parameters);
  absl::StatusOr<FwdResult> fwd_impl(cuda::Executor& executor,
                                     absl::Span<const Buffer> inputs,
                                     LayerHooks* hooks) const override;
  absl::StatusOr<BufferVec> bwd_impl(cuda::Executor& executor,
                                     absl::Span<const Buffer> gradients,
                                     BackwardState state,
                                     LayerHooks* hooks) override;
  cuda::Executor& executor_;
  Config config_;
  std::vector<std::unique_ptr<Layer>> blocks_;
  std::vector<BAdamBlock> parameters_;
  BufferVec weights_;  // Stable resident handles for Pluto checkpoints.
  int backward_start_ = 0;
};

}  // namespace pluto::llm::qwen
