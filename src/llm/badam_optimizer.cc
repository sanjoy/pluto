#include "src/llm/badam_optimizer.h"

#include <cmath>
#include <limits>
#include <utility>

#include "absl/container/flat_hash_set.h"
#include "absl/memory/memory.h"
#include "src/util/status_macros.h"

namespace pluto::llm {
namespace {

// Adam's existing Layer interface only needs a view of masters and gradients.
// This view owns shared handles, never runs a forward, and outlives Create
// only.
class ActiveParameters final : public Layer {
 public:
  explicit ActiveParameters(const BAdamBlock& block) {
    for (const auto& parameter : block.parameters) {
      weights_.push_back(parameter->master());
      gradients_.push_back(parameter->gradient());
    }
  }
  absl::string_view name() const override { return "BAdamActiveParameters"; }
  absl::Span<const ActivationType> input_types() const override { return {}; }
  absl::Span<const ActivationType> output_types() const override { return {}; }
  absl::Span<Buffer> weights() override { return absl::MakeSpan(weights_); }
  absl::Span<Buffer> gradients() override { return absl::MakeSpan(gradients_); }
  DataType output_type() const override { return DataType::FP32; }

 private:
  absl::StatusOr<FwdResult> fwd_impl(cuda::Executor&, absl::Span<const Buffer>,
                                     LayerHooks*) const override {
    return absl::UnimplementedError("optimizer parameter view has no forward");
  }
  absl::StatusOr<BufferVec> bwd_impl(cuda::Executor&, absl::Span<const Buffer>,
                                     BackwardState, LayerHooks*) override {
    return absl::UnimplementedError("optimizer parameter view has no backward");
  }
  std::vector<Buffer> weights_;
  std::vector<Buffer> gradients_;
};

}  // namespace

BAdamOptimizer::BAdamOptimizer(cuda::Executor& executor,
                               std::vector<BAdamBlock> blocks,
                               BAdamConfig config)
    : executor_(executor), blocks_(std::move(blocks)), config_(config) {}

// Validate all blocks before allocating the first active block. This avoids
// discovering an oversized/aliased block only after other weights have changed.
absl::StatusOr<std::unique_ptr<BAdamOptimizer>> BAdamOptimizer::Create(
    cuda::Executor& executor, std::vector<BAdamBlock> blocks,
    const BAdamConfig& config) {
  const auto& a = config.adam;
  if (blocks.empty() || blocks.size() > std::numeric_limits<int>::max() ||
      config.switch_every <= 0 || config.start_block < -1 ||
      config.start_block >= static_cast<int>(blocks.size()) ||
      !(a.learning_rate > 0) || !std::isfinite(a.learning_rate) ||
      !std::isfinite(a.beta1) || a.beta1 < 0 || a.beta1 >= 1 ||
      !std::isfinite(a.beta2) || a.beta2 < 0 || a.beta2 >= 1 ||
      !(a.epsilon > 0) || !std::isfinite(a.epsilon) || a.weight_decay < 0 ||
      !std::isfinite(a.weight_decay))
    return absl::InvalidArgumentError("invalid BAdam configuration");
  absl::flat_hash_set<void*> seen;
  for (const auto& block : blocks) {
    if (block.parameters.empty())
      return absl::InvalidArgumentError("BAdam blocks must not be empty");
    size_t elements = 0;
    for (const auto& p : block.parameters) {
      if (!p || p->active() || &p->value().executor() != &executor ||
          !seen.insert(p->value().data()).second)
        return absl::InvalidArgumentError(
            "BAdam requires disjoint inactive parameters on its executor");
      if (p->elements() > std::numeric_limits<size_t>::max() / 16 - elements)
        return absl::InvalidArgumentError("BAdam block size overflows");
      elements += p->elements();
    }
    if (config.max_active_bytes && elements * 16 > config.max_active_bytes)
      return absl::ResourceExhaustedError(
          "a BAdam block exceeds the active optimizer memory budget");
  }
  auto result =
      absl::WrapUnique(new BAdamOptimizer(executor, std::move(blocks), config));
  const int start = config.start_block < 0
                        ? static_cast<int>(result->blocks_.size()) - 1
                        : config.start_block;
  RETURN_IF_ERROR(result->Activate(start));
  return result;
}

BAdamOptimizer::~BAdamOptimizer() { ReleaseActive(); }

// Release every reference to the old moments/masters before activating a new
// block. Stream-ordered frees keep queued work safe without a host wait.
void BAdamOptimizer::ReleaseActive() {
  adam_.reset();
  if (active_block_ >= 0)
    for (const auto& p : blocks_[active_block_].parameters)
      (void)p->Deactivate();
  active_block_ = -1;
  prepared_ = false;
}

// Switching deliberately discards moments; BAdam does not offload and restore
// stale moments on later visits. Partial allocation failures release the block.
absl::Status BAdamOptimizer::Activate(int index) {
  ReleaseActive();
  active_block_ = index;
  block_step_ = 0;
  for (const auto& p : blocks_[index].parameters) {
    auto status = p->Activate();
    if (!status.ok()) {
      ReleaseActive();
      return status;
    }
  }
  ActiveParameters view(blocks_[index]);
  auto adam = AdamWOptimizer::Create(executor_, view, config_.adam);
  if (!adam.ok()) {
    ReleaseActive();
    return adam.status();
  }
  adam_ = std::move(*adam);
  return absl::OkStatus();
}

// Establish which block receives this batch before its forward is recorded.
// Repeated ZeroGrad without ApplyStep simply clears the same block again.
absl::Status BAdamOptimizer::ZeroGrad() {
  if (!adam_)
    return absl::FailedPreconditionError("BAdam has no active block");
  if (block_step_ == config_.switch_every) {
    const int count = static_cast<int>(blocks_.size());
    const int next =
        (active_block_ + (config_.descending ? count - 1 : 1)) % count;
    RETURN_IF_ERROR(Activate(next));
  }
  RETURN_IF_ERROR(adam_->ZeroGrad());
  prepared_ = true;
  return absl::OkStatus();
}

// Adam updates only the active FP32 master tensors. Publishing preserves small
// accumulated changes within the block visit while keeping resident BF16 data.
absl::Status BAdamOptimizer::ApplyStep() {
  if (!adam_ || !prepared_)
    return absl::FailedPreconditionError("BAdam ApplyStep requires ZeroGrad");
  if (step_ == std::numeric_limits<int>::max())
    return absl::OutOfRangeError("BAdam step count overflow");
  prepared_ = false;
  RETURN_IF_ERROR(adam_->ApplyStep());
  for (const auto& p : blocks_[active_block_].parameters)
    RETURN_IF_ERROR(p->Publish());
  ++step_;
  ++block_step_;
  return absl::OkStatus();
}

size_t BAdamOptimizer::parameter_tensor_count() const {
  return adam_ ? adam_->parameter_tensor_count() : 0;
}

size_t BAdamOptimizer::active_state_bytes() const {
  size_t bytes = 0;
  if (active_block_ >= 0)
    for (const auto& p : blocks_[active_block_].parameters)
      bytes += p->elements() * 16;
  return bytes;
}

}  // namespace pluto::llm
