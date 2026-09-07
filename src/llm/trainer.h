#pragma once

#include <cstdint>
#include <functional>
#include <optional>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "src/cuda/executor.h"
#include "src/dataset/dataset.h"
#include "src/llm/layer.h"
#include "src/llm/optimizer.h"

namespace pluto::llm {

class SparseAutoEncoderLayer;
class SparseAutoEncoderLossLayer;

inline constexpr int kUnlimitedTrainingSteps = -1;

struct EvaluationOptions {
  // Evaluate exactly this many batches after resetting the iterator.
  int batches = 1;
};

struct TrainingOptions {
  // Hard cap on optimizer updates. kUnlimitedTrainingSteps runs until an
  // explicit stop-loss condition or callback error; zero is evaluation-only.
  int max_steps = 0;

  // Logical step represented by the input model weights. The first optimizer
  // update completes initial_step + 1. Callbacks and TrainingResult use this
  // absolute numbering so resumed runs continue checkpoint step numbers.
  int initial_step = 0;

  // When stop_loss or evaluation_callback is enabled, run an evaluation at
  // this update interval and after the final update.
  int evaluation_interval = 100;
  int evaluation_batches = 1;

  // A negative value disables loss-based stopping. Zero requests exact zero
  // according to the loss layer's FP32 output.
  double stop_loss = -1.0;

  // Prefer a distinct sequential iterator here. Evaluate() resets it before
  // every pass, keeping the sample stable without perturbing random training.
  // If null, training_tokens is used and its position is reset by evaluation.
  DataSetIterator* evaluation_tokens = nullptr;

  // A caller that already evaluated the model can avoid a duplicate initial
  // pass. When absent and stop_loss is enabled, Train() evaluates once before
  // applying the first update so an already-satisfied target takes zero steps.
  std::optional<double> initial_loss;

  // Called synchronously after every evaluation performed by Train(). The
  // arguments are the number of completed optimizer steps and the mean loss.
  // Supplying a callback enables periodic evaluation even when stop_loss is
  // disabled. A supplied initial_loss does not trigger the callback because
  // Train() did not perform that evaluation.
  std::function<void(int, double)> evaluation_callback;

  // Called synchronously after every successful optimizer update. The argument
  // is the number of completed updates. Returning an error stops training and
  // propagates that error to the caller. This is suitable for periodic work
  // such as checkpointing that must not depend on the evaluation cadence.
  std::function<absl::Status(int)> step_callback;
};

struct TrainingResult {
  // Absolute logical step, including TrainingOptions::initial_step.
  int steps_completed;
  bool reached_stop_loss;
};

// Everything retained from an objective's forward pass until evaluation
// consumes its loss or training runs its backward pass. normalization_count
// states how many examples the loss represents; it may differ from the number
// of FP32 values in loss. For example, the SAE loss is one sum for an entire
// activation batch.
struct ObjectiveForwardPass {
  Buffer loss;
  int64_t normalization_count;
  Tape model_tape;
  Tape loss_tape;
};

// Adapts a model, loss, and DataBatch schema to the common training loop.
//
// The objective owns no layers. Its referenced model and loss must outlive it.
// Forward() validates and connects a dataset batch to those layers. Backward()
// wires the saved loss gradients back through the model. This boundary keeps
// Train() and Evaluate() independent of model-specific auxiliary outputs.
class TrainingObjective {
 public:
  virtual ~TrainingObjective() = default;

  virtual absl::StatusOr<ObjectiveForwardPass> Forward(
      cuda::Executor& executor, const DataBatch& batch) const = 0;
  virtual absl::Status Backward(cuda::Executor& executor,
                                ObjectiveForwardPass pass) = 0;
};

// Language-modeling wiring for a conventional model and terminal loss layer.
// DataBatch::data contains batch_size int32 input tokens followed by batch_size
// int32 next-token targets. The loss must return one FP32 value per token.
class LanguageModelingObjective final : public TrainingObjective {
 public:
  LanguageModelingObjective(Layer& model, Layer& loss_layer)
      : model_(model), loss_layer_(loss_layer) {}

  absl::StatusOr<ObjectiveForwardPass> Forward(
      cuda::Executor& executor, const DataBatch& batch) const override;
  absl::Status Backward(cuda::Executor& executor,
                        ObjectiveForwardPass pass) override;

 private:
  Layer& model_;
  Layer& loss_layer_;
};

// Sparse-autoencoder wiring for activation batches. It exposes the SAE's
// latent activations and decoder to SparseAutoEncoderLossLayer, then routes the
// loss's auxiliary gradients back to the SAE. The activation generator remains
// outside this objective and is therefore frozen.
class SparseAutoEncoderObjective final : public TrainingObjective {
 public:
  SparseAutoEncoderObjective(SparseAutoEncoderLayer& model,
                             SparseAutoEncoderLossLayer& loss_layer)
      : model_(model), loss_layer_(loss_layer) {}

  absl::StatusOr<ObjectiveForwardPass> Forward(
      cuda::Executor& executor, const DataBatch& batch) const override;
  absl::Status Backward(cuda::Executor& executor,
                        ObjectiveForwardPass pass) override;

 private:
  SparseAutoEncoderLayer& model_;
  SparseAutoEncoderLossLayer& loss_layer_;
};

// Computes the mean FP32 loss produced by objective over the requested data.
// Evaluate() never runs backward or mutates weights. It resets eval_data so
// repeated calls measure the same batches.
absl::StatusOr<double> Evaluate(cuda::Executor& executor,
                                const TrainingObjective& objective,
                                DataSetIterator& eval_tokens,
                                const EvaluationOptions& options);

// Runs the common forward/loss/backward/update loop for any TrainingObjective.
// Train() clears gradients before the first backward; Optimizer::Step() applies
// an update and clears them after each step.
absl::StatusOr<TrainingResult> Train(cuda::Executor& executor,
                                     TrainingObjective& objective,
                                     Optimizer& optimizer,
                                     DataSetIterator& training_data,
                                     const TrainingOptions& options);

}  // namespace pluto::llm
