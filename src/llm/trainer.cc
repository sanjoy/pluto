#include "src/llm/trainer.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "src/cuda/buffer.h"
#include "src/cuda/executor.h"
#include "src/dataset/dataset.h"
#include "src/llm/layer.h"
#include "src/llm/layers/sparse_autoencoder.h"
#include "src/llm/optimizer.h"
#include "src/util/status_macros.h"

namespace pluto::llm {
namespace {

struct LanguageModelingBatch {
  Buffer tokens;
  Buffer targets;
  int32_t batch_size;
};

// InMemoryDataSetIterator keeps the generic DataBatch surface small by packing
// its language-modeling inputs and targets into one allocation. Training
// unpacks those two contiguous halves into ordinary layer inputs here.
absl::StatusOr<LanguageModelingBatch> PrepareLanguageModelingBatch(
    cuda::Executor& executor, const DataBatch& batch) {
  if (batch.batch_size <= 0) {
    return absl::InvalidArgumentError("dataset returned an empty batch");
  }
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

absl::StatusOr<double> CopyLossSum(cuda::Executor& executor,
                                   const Buffer& losses, int expected_count) {
  if (expected_count <= 0 ||
      losses.size_bytes() !=
          static_cast<size_t>(expected_count) * sizeof(float)) {
    return absl::InvalidArgumentError(
        "loss layer must return one FP32 value per batch token");
  }
  if (&losses.executor() != &executor) {
    return absl::InvalidArgumentError(
        "loss buffer belongs to a different CUDA Executor");
  }
  std::vector<float> host_losses(expected_count);
  RETURN_IF_ERROR(cuda::CudaStatus(
      cudaMemcpyAsync(host_losses.data(), losses.data(), losses.size_bytes(),
                      cudaMemcpyDeviceToHost, executor.stream()),
      "cudaMemcpyAsync(evaluation losses)"));
  RETURN_IF_ERROR(executor.Synchronize());
  double total = 0.0;
  for (float loss : host_losses) total += loss;
  return total;
}

absl::Status ValidateTrainingOptions(const TrainingOptions& options) {
  if (options.max_steps < kUnlimitedTrainingSteps) {
    return absl::InvalidArgumentError(
        "max_steps must be non-negative or kUnlimitedTrainingSteps");
  }
  if (options.initial_step < 0) {
    return absl::InvalidArgumentError("initial_step must be non-negative");
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
  if (!std::isfinite(options.stop_loss)) {
    return absl::InvalidArgumentError("stop_loss must be finite");
  }
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

absl::StatusOr<Buffer> ForwardSparseAutoEncoderLoss(
    cuda::Executor& executor, const SparseAutoEncoderLayer& model,
    const SparseAutoEncoderLossLayer& loss_layer, const DataBatch& batch,
    Tape* model_tape, Tape* loss_tape) {
  RETURN_IF_ERROR(ValidateSparseAutoEncoderBatch(executor, model, batch));
  BufferVec model_inputs = {batch.data};
  ASSIGN_OR_RETURN(auto reconstruction,
                   model.fwd(executor, model_inputs, model_tape));
  ASSIGN_OR_RETURN(auto latents, model.latent_activations(*model_tape));
  BufferVec loss_inputs = {batch.data, reconstruction, latents,
                           model.decoder()};
  return loss_layer.fwd(executor, loss_inputs, loss_tape);
}

}  // namespace

absl::StatusOr<double> Evaluate(cuda::Executor& executor, const Layer& model,
                                const Layer& loss_layer,
                                DataSetIterator& eval_tokens,
                                const EvaluationOptions& options) {
  if (options.batches <= 0) {
    return absl::InvalidArgumentError("evaluation batches must be positive");
  }
  RETURN_IF_ERROR(eval_tokens.Reset());

  double loss_sum = 0.0;
  size_t token_count = 0;
  for (int index = 0; index < options.batches; ++index) {
    ASSIGN_OR_RETURN(DataBatch data_batch, eval_tokens.Next());
    ASSIGN_OR_RETURN(auto batch,
                     PrepareLanguageModelingBatch(executor, data_batch));
    Tape model_tape;
    BufferVec model_inputs = {batch.tokens};
    ASSIGN_OR_RETURN(auto output,
                     model.fwd(executor, model_inputs, &model_tape));
    Tape loss_tape;
    BufferVec loss_inputs = {output, batch.targets};
    ASSIGN_OR_RETURN(auto losses,
                     loss_layer.fwd(executor, loss_inputs, &loss_tape));
    ASSIGN_OR_RETURN(double batch_sum,
                     CopyLossSum(executor, losses, batch.batch_size));
    loss_sum += batch_sum;
    token_count += batch.batch_size;
  }
  return loss_sum / token_count;
}

absl::StatusOr<TrainingResult> Train(cuda::Executor& executor, Layer& model,
                                     Layer& loss_layer, Optimizer& optimizer,
                                     DataSetIterator& training_tokens,
                                     const TrainingOptions& options) {
  RETURN_IF_ERROR(ValidateTrainingOptions(options));
  DataSetIterator& evaluation_tokens = options.evaluation_tokens == nullptr
                                           ? training_tokens
                                           : *options.evaluation_tokens;

  if (options.stop_loss >= 0.0) {
    double initial_loss;
    if (options.initial_loss.has_value()) {
      initial_loss = *options.initial_loss;
    } else {
      ASSIGN_OR_RETURN(
          initial_loss,
          Evaluate(executor, model, loss_layer, evaluation_tokens,
                   EvaluationOptions{.batches = options.evaluation_batches}));
      if (options.evaluation_callback) {
        options.evaluation_callback(options.initial_step, initial_loss);
      }
    }
    if (initial_loss <= options.stop_loss) {
      return TrainingResult{.steps_completed = options.initial_step,
                            .reached_stop_loss = true};
    }
  }

  RETURN_IF_ERROR(training_tokens.Reset());
  RETURN_IF_ERROR(optimizer.ZeroGrad());
  const bool has_step_limit = options.max_steps != kUnlimitedTrainingSteps;
  int updates_completed = 0;
  int steps_completed = options.initial_step;
  while (!has_step_limit || updates_completed < options.max_steps) {
    if (steps_completed == std::numeric_limits<int>::max()) {
      return absl::OutOfRangeError("training step number overflowed");
    }
    ASSIGN_OR_RETURN(DataBatch data_batch, training_tokens.Next());
    ASSIGN_OR_RETURN(auto batch,
                     PrepareLanguageModelingBatch(executor, data_batch));

    Tape model_tape;
    BufferVec model_inputs = {batch.tokens};
    ASSIGN_OR_RETURN(auto output,
                     model.fwd(executor, model_inputs, &model_tape));
    Tape loss_tape;
    BufferVec loss_inputs = {output, batch.targets};
    ASSIGN_OR_RETURN(auto losses,
                     loss_layer.fwd(executor, loss_inputs, &loss_tape));
    if (losses.size_bytes() !=
        static_cast<size_t>(batch.batch_size) * sizeof(float)) {
      return absl::InvalidArgumentError(
          "loss layer must return one FP32 value per batch token");
    }
    ASSIGN_OR_RETURN(auto output_gradient,
                     loss_layer.bwd(executor, {}, std::move(loss_tape)));
    ASSIGN_OR_RETURN(auto input_gradient, model.bwd(executor, output_gradient,
                                                    std::move(model_tape)));
    (void)input_gradient;
    RETURN_IF_ERROR(optimizer.Step());
    ++updates_completed;
    ++steps_completed;
    if (options.step_callback) {
      RETURN_IF_ERROR(options.step_callback(steps_completed));
    }

    const bool evaluation_enabled =
        options.stop_loss >= 0.0 || options.evaluation_callback;
    const bool should_evaluate =
        evaluation_enabled &&
        (steps_completed % options.evaluation_interval == 0 ||
         (has_step_limit && updates_completed == options.max_steps));
    if (should_evaluate) {
      ASSIGN_OR_RETURN(
          double training_loss,
          Evaluate(executor, model, loss_layer, evaluation_tokens,
                   EvaluationOptions{.batches = options.evaluation_batches}));
      if (options.evaluation_callback) {
        options.evaluation_callback(steps_completed, training_loss);
      }
      if (options.stop_loss >= 0.0 && training_loss <= options.stop_loss) {
        return TrainingResult{.steps_completed = steps_completed,
                              .reached_stop_loss = true};
      }
    }
  }
  RETURN_IF_ERROR(executor.Synchronize());
  return TrainingResult{.steps_completed = steps_completed,
                        .reached_stop_loss = false};
}

absl::StatusOr<double> Evaluate(cuda::Executor& executor,
                                const SparseAutoEncoderLayer& model,
                                const SparseAutoEncoderLossLayer& loss_layer,
                                DataSetIterator& eval_activations,
                                const EvaluationOptions& options) {
  if (options.batches <= 0) {
    return absl::InvalidArgumentError("evaluation batches must be positive");
  }
  RETURN_IF_ERROR(eval_activations.Reset());

  double loss_sum = 0.0;
  size_t activation_count = 0;
  for (int index = 0; index < options.batches; ++index) {
    ASSIGN_OR_RETURN(DataBatch batch, eval_activations.Next());
    Tape model_tape;
    Tape loss_tape;
    ASSIGN_OR_RETURN(auto loss, ForwardSparseAutoEncoderLoss(
                                    executor, model, loss_layer, batch,
                                    &model_tape, &loss_tape));
    ASSIGN_OR_RETURN(double batch_loss, CopyLossSum(executor, loss, 1));
    loss_sum += batch_loss;
    activation_count += batch.batch_size;
  }
  return loss_sum / activation_count;
}

absl::StatusOr<TrainingResult> Train(cuda::Executor& executor,
                                     SparseAutoEncoderLayer& model,
                                     SparseAutoEncoderLossLayer& loss_layer,
                                     Optimizer& optimizer,
                                     DataSetIterator& training_activations,
                                     const TrainingOptions& options) {
  RETURN_IF_ERROR(ValidateTrainingOptions(options));
  DataSetIterator& evaluation_activations = options.evaluation_tokens == nullptr
                                                ? training_activations
                                                : *options.evaluation_tokens;

  if (options.stop_loss >= 0.0) {
    double initial_loss;
    if (options.initial_loss.has_value()) {
      initial_loss = *options.initial_loss;
    } else {
      ASSIGN_OR_RETURN(
          initial_loss,
          Evaluate(executor, model, loss_layer, evaluation_activations,
                   EvaluationOptions{.batches = options.evaluation_batches}));
      if (options.evaluation_callback) {
        options.evaluation_callback(options.initial_step, initial_loss);
      }
    }
    if (initial_loss <= options.stop_loss) {
      return TrainingResult{.steps_completed = options.initial_step,
                            .reached_stop_loss = true};
    }
  }

  RETURN_IF_ERROR(training_activations.Reset());
  RETURN_IF_ERROR(optimizer.ZeroGrad());
  const bool has_step_limit = options.max_steps != kUnlimitedTrainingSteps;
  int updates_completed = 0;
  int steps_completed = options.initial_step;
  while (!has_step_limit || updates_completed < options.max_steps) {
    if (steps_completed == std::numeric_limits<int>::max()) {
      return absl::OutOfRangeError("training step number overflowed");
    }
    ASSIGN_OR_RETURN(DataBatch batch, training_activations.Next());
    Tape model_tape;
    Tape loss_tape;
    ASSIGN_OR_RETURN(auto loss, ForwardSparseAutoEncoderLoss(
                                    executor, model, loss_layer, batch,
                                    &model_tape, &loss_tape));
    if (loss.size_bytes() != sizeof(float)) {
      return absl::InvalidArgumentError(
          "sparse-autoencoder loss must return one FP32 scalar");
    }
    ASSIGN_OR_RETURN(auto loss_gradients,
                     loss_layer.bwd(executor, {}, std::move(loss_tape)));
    if (loss_gradients.size() != 4) {
      return absl::InternalError(
          "sparse-autoencoder loss must return gradients for x, x1, z, and D");
    }
    // The activation generator is frozen, so dL/dx is intentionally dropped.
    // The remaining gradients match SparseAutoEncoderLayer's documented
    // auxiliary backward inputs.
    BufferVec model_gradients = {loss_gradients[1], loss_gradients[2],
                                 loss_gradients[3]};
    ASSIGN_OR_RETURN(auto input_gradient, model.bwd(executor, model_gradients,
                                                    std::move(model_tape)));
    (void)input_gradient;
    RETURN_IF_ERROR(optimizer.Step());
    ++updates_completed;
    ++steps_completed;
    if (options.step_callback) {
      RETURN_IF_ERROR(options.step_callback(steps_completed));
    }

    const bool evaluation_enabled =
        options.stop_loss >= 0.0 || options.evaluation_callback;
    const bool should_evaluate =
        evaluation_enabled &&
        (steps_completed % options.evaluation_interval == 0 ||
         (has_step_limit && updates_completed == options.max_steps));
    if (should_evaluate) {
      ASSIGN_OR_RETURN(
          double training_loss,
          Evaluate(executor, model, loss_layer, evaluation_activations,
                   EvaluationOptions{.batches = options.evaluation_batches}));
      if (options.evaluation_callback) {
        options.evaluation_callback(steps_completed, training_loss);
      }
      if (options.stop_loss >= 0.0 && training_loss <= options.stop_loss) {
        return TrainingResult{.steps_completed = steps_completed,
                              .reached_stop_loss = true};
      }
    }
  }
  RETURN_IF_ERROR(executor.Synchronize());
  return TrainingResult{.steps_completed = steps_completed,
                        .reached_stop_loss = false};
}

}  // namespace pluto::llm
