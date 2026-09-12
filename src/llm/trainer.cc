#include "src/llm/trainer.h"

#include <cuda_runtime.h>
#include <cuda_tile.h>

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "src/cuda/buffer.h"
#include "src/cuda/executor.h"
#include "src/cuda/page_locked_host_array.h"
#include "src/dataset/dataset.h"
#include "src/llm/layer.h"
#include "src/llm/optimizer.h"
#include "src/util/status_macros.h"

namespace pluto::llm {
namespace {

// Adds all FP32 loss values to accumulator. On the final evaluation batch,
// output_scale converts the accumulated sum into a mean. A single tile program
// intentionally owns the scalar so batches remain deterministic without
// atomics; loss vectors are tiny relative to the model kernels that produce
// them.
__tile_global__ void AddLossKernel(const float* __restrict__ losses,
                                   int loss_count, float output_scale,
                                   float* __restrict__ accumulator) {
  namespace ct = ::cuda::tiles;
  using namespace ct::literals;

  auto loss_view = ct::partition_view{
      ct::tensor_span{losses, ct::extents{loss_count}}, ct::shape{1_ic}};
  auto accumulator_view = ct::partition_view{
      ct::tensor_span{accumulator, ct::extents{1}}, ct::shape{1_ic}};
  auto sum = accumulator_view.load(0);
  for (int index = 0; index < loss_count; ++index)
    sum = sum + loss_view.load(index);
  accumulator_view.store(sum * output_scale, 0);
}

// Private, per-call state only: layer-specific routing is expressed by the
// forward output vector, not by a model-specific training adapter.
struct ForwardPass {
  Buffer loss;
  int row_count;
  size_t model_output_count;
  BackwardState model_state;
  BackwardState loss_state;
};

absl::StatusOr<ForwardPass> Forward(cuda::Executor& executor,
                                    const Layer& model, const Layer& loss_layer,
                                    const DataBatch& batch) {
  ASSIGN_OR_RETURN(const int rows, batch.token_count());
  if (&batch.inputs.executor() != &executor ||
      &batch.targets.executor() != &executor)
    return absl::InvalidArgumentError(
        "dataset inputs and targets must belong to the supplied CUDA Executor");
  // Preserve sample boundaries before layers infer flattened shapes from bytes.
  RETURN_IF_ERROR(model.ValidateSequenceLength(batch.sequence_length));
  RETURN_IF_ERROR(loss_layer.ValidateSequenceLength(batch.sequence_length));

  ASSIGN_OR_RETURN(auto model_fwd, model.fwd(executor, {batch.inputs}));
  if (model_fwd.outputs.empty())
    return absl::InvalidArgumentError("model must return at least one output");
  for (const Buffer& output : model_fwd.outputs)
    if (&output.executor() != &executor)
      return absl::InvalidArgumentError(
          "model output belongs to a different CUDA Executor");
  const size_t model_output_count = model_fwd.outputs.size();
  BufferVec loss_inputs = std::move(model_fwd.outputs);
  loss_inputs.push_back(batch.targets);
  ASSIGN_OR_RETURN(auto loss_fwd, loss_layer.fwd(executor, loss_inputs));
  if (loss_fwd.outputs.size() != 1)
    return absl::InvalidArgumentError("loss must return exactly one buffer");
  Buffer loss = std::move(loss_fwd.outputs[0]);
  if (&loss.executor() != &executor)
    return absl::InvalidArgumentError(
        "loss belongs to a different CUDA Executor");
  if (loss.size_bytes() != static_cast<size_t>(rows) * sizeof(float))
    return absl::InvalidArgumentError(
        "loss must return one FP32 value per batch token/activation row");
  return ForwardPass{std::move(loss), rows, model_output_count,
                     std::move(model_fwd.state), std::move(loss_fwd.state)};
}

absl::Status Backward(cuda::Executor& executor, Layer& model, Layer& loss_layer,
                      ForwardPass pass) {
  ASSIGN_OR_RETURN(auto gradients,
                   loss_layer.bwd(executor, {}, std::move(pass.loss_state)));
  if (gradients.size() != pass.model_output_count)
    return absl::InvalidArgumentError(
        "loss must return exactly one gradient per model output");
  for (const Buffer& gradient : gradients)
    if (&gradient.executor() != &executor)
      return absl::InvalidArgumentError(
          "loss gradient belongs to a different CUDA Executor");
  ASSIGN_OR_RETURN(auto input_gradients,
                   model.bwd(executor, gradients, std::move(pass.model_state)));
  (void)input_gradients;
  return absl::OkStatus();
}

absl::StatusOr<double> ReadDeviceLoss(cuda::Executor& executor,
                                      const Buffer& loss) {
  if (loss.size_bytes() != sizeof(float)) {
    return absl::InvalidArgumentError(
        "evaluation result must contain one FP32 scalar");
  }
  if (&loss.executor() != &executor) {
    return absl::InvalidArgumentError(
        "evaluation result belongs to a different CUDA Executor");
  }
  ASSIGN_OR_RETURN(auto host_loss,
                   cuda::PageLockedHostArray<float>::Allocate(executor, 1));
  RETURN_IF_ERROR(cuda::CudaStatus(
      cudaMemcpyAsync(host_loss.data(), loss.data(), loss.size_bytes(),
                      cudaMemcpyDeviceToHost, executor.stream()),
      "cudaMemcpyAsync(evaluation result)"));
  RETURN_IF_ERROR(executor.Synchronize());
  return host_loss[0];
}

absl::Status ValidateTrainingOptions(const TrainingOptions& options) {
  if (options.max_steps < kUnlimitedTrainingSteps) {
    return absl::InvalidArgumentError(
        "max_steps must be non-negative or kUnlimitedTrainingSteps");
  }
  if (options.initial_step < 0)
    return absl::InvalidArgumentError("initial_step must be non-negative");
  if (options.training_seconds.has_value() &&
      (!std::isfinite(*options.training_seconds) ||
       *options.training_seconds <= 0.0)) {
    return absl::InvalidArgumentError(
        "training_seconds must be finite and positive");
  }
  if (options.max_steps != kUnlimitedTrainingSteps &&
      options.max_steps >
          std::numeric_limits<int>::max() - options.initial_step) {
    return absl::InvalidArgumentError("training step count would overflow");
  }
  if (options.evaluation_interval <= 0 || options.evaluation_batches <= 0) {
    return absl::InvalidArgumentError(
        "training evaluation counts must be positive");
  }
  if (!std::isfinite(options.stop_loss))
    return absl::InvalidArgumentError("stop_loss must be finite");
  if (options.initial_loss.has_value() &&
      (!std::isfinite(*options.initial_loss) || *options.initial_loss < 0.0)) {
    return absl::InvalidArgumentError(
        "initial_loss must be finite and non-negative");
  }
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<Buffer> Evaluate(cuda::Executor& executor, const Layer& model,
                                const Layer& loss_layer,
                                DataSetIterator& eval_data,
                                const EvaluationOptions& options) {
  if (options.batches <= 0)
    return absl::InvalidArgumentError("evaluation batches must be positive");
  RETURN_IF_ERROR(eval_data.Reset());
  ASSIGN_OR_RETURN(auto mean_loss, Buffer::Allocate(executor, sizeof(float)));
  RETURN_IF_ERROR(cuda::CudaStatus(
      cudaMemsetAsync(mean_loss.data(), 0, mean_loss.size_bytes(),
                      executor.stream()),
      "cudaMemsetAsync(evaluation result)"));

  int64_t normalization_count = 0;
  for (int index = 0; index < options.batches; ++index) {
    ASSIGN_OR_RETURN(DataBatch batch, eval_data.Next());
    ASSIGN_OR_RETURN(auto pass, Forward(executor, model, loss_layer, batch));
    if (normalization_count >
        std::numeric_limits<int64_t>::max() - pass.row_count) {
      return absl::OutOfRangeError("evaluation normalization count overflowed");
    }
    normalization_count += pass.row_count;
    const bool final_batch = index + 1 == options.batches;
    const float output_scale =
        final_batch ? 1.0f / static_cast<float>(normalization_count) : 1.0f;
    AddLossKernel<<<1, 1, 0, executor.stream()>>>(
        static_cast<const float*>(pass.loss.data()),
        pass.row_count, output_scale,
        static_cast<float*>(mean_loss.data()));
    RETURN_IF_ERROR(
        cuda::CudaStatus(cudaGetLastError(), "AddLossKernel launch"));
  }
  return mean_loss;
}

absl::StatusOr<TrainingResult> Train(cuda::Executor& executor, Layer& model,
                                     Layer& loss_layer, Optimizer& optimizer,
                                     DataSetIterator& training_data,
                                     const TrainingOptions& options) {
  RETURN_IF_ERROR(ValidateTrainingOptions(options));
  DataSetIterator& evaluation_data = options.evaluation_data == nullptr
                                         ? training_data
                                         : *options.evaluation_data;

  if (options.stop_loss >= 0.0) {
    double initial_loss;
    if (options.initial_loss.has_value()) {
      initial_loss = *options.initial_loss;
    } else {
      ASSIGN_OR_RETURN(
          auto device_initial_loss,
          Evaluate(executor, model, loss_layer, evaluation_data,
                   EvaluationOptions{.batches = options.evaluation_batches}));
      ASSIGN_OR_RETURN(initial_loss,
                       ReadDeviceLoss(executor, device_initial_loss));
      if (options.evaluation_callback)
        options.evaluation_callback(options.initial_step, initial_loss);
    }
    if (initial_loss <= options.stop_loss) {
      return TrainingResult{.steps_completed = options.initial_step,
                            .reached_stop_loss = true};
    }
  }

  RETURN_IF_ERROR(training_data.Reset());
  RETURN_IF_ERROR(optimizer.ZeroGrad());
  if (options.training_seconds.has_value()) {
    // Do not charge queued initialization work to the training budget.
    RETURN_IF_ERROR(executor.Synchronize());
  }
  const auto training_start = std::chrono::steady_clock::now();
  const auto elapsed_seconds = [&training_start]() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                         training_start)
        .count();
  };
  const bool has_step_limit = options.max_steps != kUnlimitedTrainingSteps;
  int updates_completed = 0;
  int steps_completed = options.initial_step;
  bool reached_time_limit = false;
  double elapsed_training_seconds = 0.0;
  while (!has_step_limit || updates_completed < options.max_steps) {
    if (steps_completed == std::numeric_limits<int>::max())
      return absl::OutOfRangeError("training step number overflowed");
    ASSIGN_OR_RETURN(DataBatch batch, training_data.Next());
    ASSIGN_OR_RETURN(auto pass, Forward(executor, model, loss_layer, batch));
    RETURN_IF_ERROR(Backward(executor, model, loss_layer, std::move(pass)));
    RETURN_IF_ERROR(optimizer.ApplyStep());
    if (options.training_seconds.has_value()) {
      // CUDA launches are asynchronous. The clock must measure completed
      // optimizer work, not how quickly the host fills the stream's queue.
      RETURN_IF_ERROR(executor.Synchronize());
    }
    ++updates_completed;
    ++steps_completed;
    if (options.step_callback)
      RETURN_IF_ERROR(options.step_callback(steps_completed));
    elapsed_training_seconds = elapsed_seconds();
    reached_time_limit = options.training_seconds.has_value() &&
                         elapsed_training_seconds >= *options.training_seconds;

    const bool evaluation_enabled =
        options.stop_loss >= 0.0 || options.evaluation_callback;
    const bool should_evaluate =
        evaluation_enabled &&
        (steps_completed % options.evaluation_interval == 0 ||
         (has_step_limit && updates_completed == options.max_steps) ||
         reached_time_limit);
    if (should_evaluate) {
      ASSIGN_OR_RETURN(
          auto device_training_loss,
          Evaluate(executor, model, loss_layer, evaluation_data,
                   EvaluationOptions{.batches = options.evaluation_batches}));
      ASSIGN_OR_RETURN(double training_loss,
                       ReadDeviceLoss(executor, device_training_loss));
      if (options.evaluation_callback)
        options.evaluation_callback(steps_completed, training_loss);
      if (!reached_time_limit) {
        // A periodic evaluation can itself cross the deadline. Its weights
        // already correspond to the final completed update in that case.
        elapsed_training_seconds = elapsed_seconds();
        reached_time_limit =
            options.training_seconds.has_value() &&
            elapsed_training_seconds >= *options.training_seconds;
      }
      if (options.stop_loss >= 0.0 && training_loss <= options.stop_loss) {
        return TrainingResult{
            .steps_completed = steps_completed,
            .reached_stop_loss = true,
            .reached_time_limit = reached_time_limit,
            .elapsed_training_seconds = elapsed_training_seconds};
      }
    }
    if (reached_time_limit)
      break;
  }
  RETURN_IF_ERROR(executor.Synchronize());
  if (!reached_time_limit)
    elapsed_training_seconds = elapsed_seconds();
  return TrainingResult{.steps_completed = steps_completed,
                        .reached_stop_loss = false,
                        .reached_time_limit = reached_time_limit,
                        .elapsed_training_seconds = elapsed_training_seconds};
}

}  // namespace pluto::llm
