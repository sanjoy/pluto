#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "src/llm/adamw_optimizer.h"
#include "src/llm/block_parameter.h"

namespace pluto::llm {

// A disjoint group of parameters optimized together; order is model order.
struct BAdamBlock {
  std::string name;  // Diagnostic name, e.g. QwenBlock63.
  std::vector<std::shared_ptr<BlockParameter>> parameters;
};

// BAdam is block-coordinate Adam: keep moments only for the active block,
// discard them at a switch, and restart local bias correction from step one.
struct BAdamConfig {
  AdamWConfig adam;  // Adam hyperparameters, including optional weight decay.
  int switch_every = 50;   // Updates before moving to the next block.
  int start_block = -1;    // -1 starts at the last block (the output head).
  bool descending = true;  // Cycle toward the input, wrapping at either end.
  // Optional cap for active FP32 master+gradient+two moments (not activations
  // or resident weights). Zero leaves allocation limited by device capacity.
  size_t max_active_bytes = 0;
};

// Inactive resident weights remain available for forward and input gradients;
// only the active block allocates FP32 masters, parameter gradients and Adam
// moments. Publish writes each update back to resident storage. ZeroGrad may
// switch blocks, so call it BEFORE forward/backward and inspect active_block()
// afterward. ApplyStep never switches underneath saved backward state.
class BAdamOptimizer final : public Optimizer {
 public:
  static absl::StatusOr<std::unique_ptr<BAdamOptimizer>> Create(
      cuda::Executor& executor, std::vector<BAdamBlock> blocks,
      const BAdamConfig& config = {});
  ~BAdamOptimizer() override;
  absl::Status ZeroGrad() override;
  absl::Status ApplyStep() override;
  int step() const override { return step_; }
  // Number of tensors with optimizer state now, not all resident tensors.
  size_t parameter_tensor_count() const override;
  int active_block() const { return active_block_; }
  int block_step() const { return block_step_; }
  size_t active_state_bytes() const;
  const std::vector<BAdamBlock>& blocks() const { return blocks_; }

 private:
  BAdamOptimizer(cuda::Executor& executor, std::vector<BAdamBlock> blocks,
                 BAdamConfig config);
  absl::Status Activate(int index);
  void ReleaseActive();
  cuda::Executor& executor_;
  std::vector<BAdamBlock> blocks_;
  BAdamConfig config_;
  std::unique_ptr<AdamWOptimizer> adam_;
  int active_block_ = -1;
  int block_step_ = 0;
  int step_ = 0;
  bool prepared_ = false;  // ApplyStep must follow a successful ZeroGrad.
};

}  // namespace pluto::llm
