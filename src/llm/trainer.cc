#include "src/llm/trainer.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <optional>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "src/common/status_macros.h"
#include "src/cuda/buffer.h"
#include "src/dataset/dataset.h"
#include "src/llm/layer.h"
#include "src/llm/optimizer.h"

namespace pluto::llm {
namespace {

absl::Status CudaStatus(cudaError_t error, const char* operation) {
  if (error == cudaSuccess) return absl::OkStatus();
  return absl::InternalError(absl::StrCat(operation,
                                          " failed: ", cudaGetErrorName(error),
                                          ": ", cudaGetErrorString(error)));
}

absl::Status ValidateBatch(const TokenBatch& batch) {
  if (batch.batch_size <= 0) {
    return absl::InvalidArgumentError("dataset returned an empty batch");
  }
  const size_t expected_bytes =
      static_cast<size_t>(batch.batch_size) * sizeof(int);
  if (batch.tokens.size_bytes() != expected_bytes ||
      batch.targets.size_bytes() != expected_bytes) {
    return absl::InvalidArgumentError(
        "dataset token buffers do not match batch_size");
  }
  if (batch.tokens.stream() != batch.targets.stream()) {
    return absl::InvalidArgumentError(
        "dataset token buffers must use the same CUDA stream");
  }
  return absl::OkStatus();
}

absl::StatusOr<double> CopyLossSum(const Buffer& losses, int expected_count) {
  if (expected_count <= 0 ||
      losses.size_bytes() !=
          static_cast<size_t>(expected_count) * sizeof(float)) {
    return absl::InvalidArgumentError(
        "loss layer must return one FP32 value per batch token");
  }
  std::vector<float> host_losses(expected_count);
  RETURN_IF_ERROR(CudaStatus(
      cudaMemcpyAsync(host_losses.data(), losses.data(), losses.size_bytes(),
                      cudaMemcpyDeviceToHost, losses.stream()),
      "cudaMemcpyAsync(evaluation losses)"));
  RETURN_IF_ERROR(CudaStatus(cudaStreamSynchronize(losses.stream()),
                             "cudaStreamSynchronize(evaluation)"));
  double total = 0.0;
  for (float loss : host_losses) total += loss;
  return total;
}

absl::Status ValidateTrainingOptions(const TrainingOptions& options) {
  if (options.max_steps < 0) {
    return absl::InvalidArgumentError("max_steps must be non-negative");
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

}  // namespace

absl::StatusOr<double> Evaluate(const Layer& model, const Layer& loss_layer,
                                DataSetIterator& eval_tokens,
                                const EvaluationOptions& options) {
  if (options.batches <= 0) {
    return absl::InvalidArgumentError("evaluation batches must be positive");
  }
  RETURN_IF_ERROR(eval_tokens.Reset());

  double loss_sum = 0.0;
  size_t token_count = 0;
  for (int index = 0; index < options.batches; ++index) {
    ASSIGN_OR_RETURN(TokenBatch batch, eval_tokens.Next());
    RETURN_IF_ERROR(ValidateBatch(batch));
    Tape model_tape;
    BufferVec model_inputs = {batch.tokens};
    ASSIGN_OR_RETURN(auto output, model.fwd(model_inputs, &model_tape));
    Tape loss_tape;
    BufferVec loss_inputs = {output, batch.targets};
    ASSIGN_OR_RETURN(auto losses, loss_layer.fwd(loss_inputs, &loss_tape));
    ASSIGN_OR_RETURN(double batch_sum, CopyLossSum(losses, batch.batch_size));
    loss_sum += batch_sum;
    token_count += batch.batch_size;
  }
  return loss_sum / token_count;
}

absl::StatusOr<TrainingResult> Train(Layer& model, Layer& loss_layer,
                                     Optimizer& optimizer,
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
          Evaluate(model, loss_layer, evaluation_tokens,
                   EvaluationOptions{.batches = options.evaluation_batches}));
    }
    if (initial_loss <= options.stop_loss) {
      return TrainingResult{.steps_completed = 0, .reached_stop_loss = true};
    }
  }

  RETURN_IF_ERROR(training_tokens.Reset());
  RETURN_IF_ERROR(optimizer.ZeroGrad());
  cudaStream_t training_stream = nullptr;
  for (int step = 0; step < options.max_steps; ++step) {
    ASSIGN_OR_RETURN(TokenBatch batch, training_tokens.Next());
    RETURN_IF_ERROR(ValidateBatch(batch));
    training_stream = batch.tokens.stream();

    Tape model_tape;
    BufferVec model_inputs = {batch.tokens};
    ASSIGN_OR_RETURN(auto output, model.fwd(model_inputs, &model_tape));
    Tape loss_tape;
    BufferVec loss_inputs = {output, batch.targets};
    ASSIGN_OR_RETURN(auto losses, loss_layer.fwd(loss_inputs, &loss_tape));
    if (losses.size_bytes() !=
        static_cast<size_t>(batch.batch_size) * sizeof(float)) {
      return absl::InvalidArgumentError(
          "loss layer must return one FP32 value per batch token");
    }
    ASSIGN_OR_RETURN(auto output_gradient,
                     loss_layer.bwd({}, std::move(loss_tape)));
    ASSIGN_OR_RETURN(auto input_gradient,
                     model.bwd(output_gradient, std::move(model_tape)));
    (void)input_gradient;
    RETURN_IF_ERROR(optimizer.Step());

    const bool should_evaluate =
        options.stop_loss >= 0.0 &&
        ((step + 1) % options.evaluation_interval == 0 ||
         step + 1 == options.max_steps);
    if (should_evaluate) {
      ASSIGN_OR_RETURN(
          double training_loss,
          Evaluate(model, loss_layer, evaluation_tokens,
                   EvaluationOptions{.batches = options.evaluation_batches}));
      if (training_loss <= options.stop_loss) {
        return TrainingResult{.steps_completed = step + 1,
                              .reached_stop_loss = true};
      }
    }
  }
  if (training_stream != nullptr) {
    RETURN_IF_ERROR(CudaStatus(cudaStreamSynchronize(training_stream),
                               "cudaStreamSynchronize(after training)"));
  }
  return TrainingResult{.steps_completed = options.max_steps,
                        .reached_stop_loss = false};
}

}  // namespace pluto::llm
