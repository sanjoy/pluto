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
#include "src/llm/layers/sparse_autoencoder.h"
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

struct LanguageModelingBatch {
  Buffer tokens;
  Buffer targets;
  int32_t batch_size;
};

// InMemoryDataSetIterator keeps the generic DataBatch surface small by packing
// its language-modeling inputs and targets into one allocation. The language-
// modeling objective unpacks those two contiguous halves here.
absl::StatusOr<LanguageModelingBatch> PrepareLanguageModelingBatch(
    cuda::Executor& executor, const DataBatch& batch) {
  if (batch.batch_size <= 0)
    return absl::InvalidArgumentError("dataset returned an empty batch");
  const size_t token_bytes =
      static_cast<size_t>(batch.batch_size) * sizeof(int);
  if (batch.data.size_bytes() != 2 * token_bytes) {
    return absl::InvalidArgumentError(
        "language-modeling dataset data must contain batch_size input tokens "
        "followed by batch_size target tokens");
  }
  if (&batch.data.executor() != &executor) {
    return absl::InvalidArgumentError(
        "dataset data must belong to the supplied CUDA Executor");
  }

  ASSIGN_OR_RETURN(auto tokens, Buffer::Allocate(executor, token_bytes));
  ASSIGN_OR_RETURN(auto targets, Buffer::Allocate(executor, token_bytes));
  const auto* data = static_cast<const char*>(batch.data.data());
  RETURN_IF_ERROR(cuda::CudaStatus(
      cudaMemcpyAsync(tokens.data(), data, token_bytes,
                      cudaMemcpyDeviceToDevice, executor.stream()),
      "cudaMemcpyAsync(language-modeling inputs)"));
  RETURN_IF_ERROR(cuda::CudaStatus(
      cudaMemcpyAsync(targets.data(), data + token_bytes, token_bytes,
                      cudaMemcpyDeviceToDevice, executor.stream()),
      "cudaMemcpyAsync(language-modeling targets)"));
  return LanguageModelingBatch{
      .tokens = std::move(tokens),
      .targets = std::move(targets),
      .batch_size = batch.batch_size,
  };
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
                   cuda::PageLockedHostArray<float>::Allocate(1));
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

absl::Status ValidateSparseAutoEncoderBatch(cuda::Executor& executor,
                                            const SparseAutoEncoderLayer& model,
                                            const DataBatch& batch) {
  if (batch.batch_size <= 0) {
    return absl::InvalidArgumentError(
        "sparse-autoencoder dataset returned an empty batch");
  }
  if (&batch.data.executor() != &executor) {
    return absl::InvalidArgumentError(
        "sparse-autoencoder dataset belongs to a different executor");
  }
  const size_t element_bytes =
      model.output_type() == DataType::BF16 ? sizeof(uint16_t) : sizeof(float);
  const size_t expected_bytes =
      static_cast<size_t>(batch.batch_size) * model.input_dim() * element_bytes;
  if (batch.data.size_bytes() != expected_bytes) {
    return absl::InvalidArgumentError(
        "sparse-autoencoder dataset data does not match batch_size and "
        "input_dim");
  }
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<ObjectiveForwardPass> LanguageModelingObjective::Forward(
    cuda::Executor& executor, const DataBatch& data_batch) const {
  ASSIGN_OR_RETURN(auto batch,
                   PrepareLanguageModelingBatch(executor, data_batch));
  Tape model_tape;
  BufferVec model_inputs = {batch.tokens};
  ASSIGN_OR_RETURN(auto output, model_.fwd(executor, model_inputs, &model_tape));
  Tape loss_tape;
  BufferVec loss_inputs = {output, batch.targets};
  ASSIGN_OR_RETURN(auto losses,
                   loss_layer_.fwd(executor, loss_inputs, &loss_tape));
  if (losses.size_bytes() !=
      static_cast<size_t>(batch.batch_size) * sizeof(float)) {
    return absl::InvalidArgumentError(
        "language-modeling loss must return one FP32 value per batch token");
  }
  return ObjectiveForwardPass{
      .loss = std::move(losses),
      .normalization_count = batch.batch_size,
      .model_tape = std::move(model_tape),
      .loss_tape = std::move(loss_tape),
  };
}

absl::Status LanguageModelingObjective::Backward(cuda::Executor& executor,
                                                 ObjectiveForwardPass pass) {
  ASSIGN_OR_RETURN(auto output_gradient,
                   loss_layer_.bwd(executor, {}, std::move(pass.loss_tape)));
  ASSIGN_OR_RETURN(auto input_gradient, model_.bwd(executor, output_gradient,
                                                   std::move(pass.model_tape)));
  (void)input_gradient;
  return absl::OkStatus();
}

absl::StatusOr<ObjectiveForwardPass> SparseAutoEncoderObjective::Forward(
    cuda::Executor& executor, const DataBatch& batch) const {
  RETURN_IF_ERROR(ValidateSparseAutoEncoderBatch(executor, model_, batch));
  Tape model_tape;
  BufferVec model_inputs = {batch.data};
  ASSIGN_OR_RETURN(auto reconstruction,
                   model_.fwd(executor, model_inputs, &model_tape));
  ASSIGN_OR_RETURN(auto latents, model_.latent_activations(model_tape));
  Tape loss_tape;
  BufferVec loss_inputs = {batch.data, reconstruction, latents,
                           model_.decoder()};
  ASSIGN_OR_RETURN(auto loss,
                   loss_layer_.fwd(executor, loss_inputs, &loss_tape));
  if (loss.size_bytes() != sizeof(float)) {
    return absl::InvalidArgumentError(
        "sparse-autoencoder loss must return one FP32 scalar");
  }
  return ObjectiveForwardPass{
      .loss = std::move(loss),
      .normalization_count = batch.batch_size,
      .model_tape = std::move(model_tape),
      .loss_tape = std::move(loss_tape),
  };
}

absl::Status SparseAutoEncoderObjective::Backward(cuda::Executor& executor,
                                                  ObjectiveForwardPass pass) {
  ASSIGN_OR_RETURN(auto loss_gradients,
                   loss_layer_.bwd(executor, {}, std::move(pass.loss_tape)));
  if (loss_gradients.size() != 4) {
    return absl::InternalError(
        "sparse-autoencoder loss must return gradients for x, x1, z, and D");
  }
  // The activation generator is frozen, so dL/dx is intentionally dropped.
  // The remaining gradients match SparseAutoEncoderLayer's documented
  // auxiliary backward inputs.
  BufferVec model_gradients = {loss_gradients[1], loss_gradients[2],
                               loss_gradients[3]};
  ASSIGN_OR_RETURN(auto input_gradient, model_.bwd(executor, model_gradients,
                                                   std::move(pass.model_tape)));
  (void)input_gradient;
  return absl::OkStatus();
}

absl::StatusOr<Buffer> Evaluate(cuda::Executor& executor,
                                const TrainingObjective& objective,
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
    ASSIGN_OR_RETURN(auto pass, objective.Forward(executor, batch));
    if (pass.normalization_count <= 0) {
      return absl::InvalidArgumentError(
          "objective normalization count must be positive");
    }
    if (pass.loss.size_bytes() == 0 ||
        pass.loss.size_bytes() % sizeof(float) != 0 ||
        pass.loss.size_bytes() / sizeof(float) >
            static_cast<size_t>(std::numeric_limits<int>::max())) {
      return absl::InvalidArgumentError(
          "objective loss must contain a supported number of FP32 values");
    }
    if (&pass.loss.executor() != &executor) {
      return absl::InvalidArgumentError(
          "objective loss belongs to a different CUDA Executor");
    }
    if (normalization_count >
        std::numeric_limits<int64_t>::max() - pass.normalization_count) {
      return absl::OutOfRangeError("evaluation normalization count overflowed");
    }
    normalization_count += pass.normalization_count;
    const bool final_batch = index + 1 == options.batches;
    const float output_scale =
        final_batch ? 1.0f / static_cast<float>(normalization_count) : 1.0f;
    AddLossKernel<<<1, 1, 0, executor.stream()>>>(
        static_cast<const float*>(pass.loss.data()),
        static_cast<int>(pass.loss.size_bytes() / sizeof(float)), output_scale,
        static_cast<float*>(mean_loss.data()));
    RETURN_IF_ERROR(
        cuda::CudaStatus(cudaGetLastError(), "AddLossKernel launch"));
  }
  return mean_loss;
}

absl::StatusOr<TrainingResult> Train(cuda::Executor& executor,
                                     TrainingObjective& objective,
                                     Optimizer& optimizer,
                                     DataSetIterator& training_data,
                                     const TrainingOptions& options) {
  RETURN_IF_ERROR(ValidateTrainingOptions(options));
  DataSetIterator& evaluation_data = options.evaluation_tokens == nullptr
                                         ? training_data
                                         : *options.evaluation_tokens;

  if (options.stop_loss >= 0.0) {
    double initial_loss;
    if (options.initial_loss.has_value()) {
      initial_loss = *options.initial_loss;
    } else {
      ASSIGN_OR_RETURN(
          auto device_initial_loss,
          Evaluate(executor, objective, evaluation_data,
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
    ASSIGN_OR_RETURN(auto pass, objective.Forward(executor, batch));
    RETURN_IF_ERROR(objective.Backward(executor, std::move(pass)));
    RETURN_IF_ERROR(optimizer.Step());
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
          Evaluate(executor, objective, evaluation_data,
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
