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

inline constexpr int kUnlimitedTrainingSteps = -1;

struct EvaluationOptions {
  // Required, non-owning dependencies. These must outlive Evaluate(); copying
  // options preserves the bindings. The loss is read-only, but evaluation
  // resets and advances the referenced iterator even with const options.
  const Layer& loss_layer;
  DataSetIterator& eval_data;

  // Evaluate exactly this many batches after resetting the iterator.
  int batches = 1;

  // Optional, non-owning hooks for the model and loss forward passes. The
  // instance must outlive Evaluate(); nullptr leaves calls uninstrumented.
  // These hooks are not passed to dataset transforms.
  LayerHooks* layer_hooks = nullptr;
};

struct TrainingOptions {
  // Required, non-owning dependencies. These must outlive Train(); copying
  // options keeps references to the same objects. A const options object still
  // permits training to mutate the referenced layers, optimizer, and iterator.
  Layer& loss_layer;
  Optimizer& optimizer;
  DataSetIterator& training_data;

  // Hard cap on optimizer updates. kUnlimitedTrainingSteps runs until an
  // explicit loss/time limit or callback error; zero is evaluation-only.
  int max_steps = 0;

  // Logical step represented by the input model weights. The first optimizer
  // update completes initial_step + 1. Callbacks and TrainingResult use this
  // absolute numbering so resumed runs continue checkpoint step numbers.
  int initial_step = 0;

  // Optional positive finite wall-clock budget, in seconds. Timing starts
  // after initial evaluation, dataset reset, and gradient initialization.
  // Periodic evaluation and step callbacks count toward the budget. With a
  // budget, each optimizer update is synchronized on executor's stream before
  // checking the deadline, so a timeout always leaves completed weights.
  // Final evaluation after detecting a timeout is outside the budget. An
  // absent budget preserves the usual asynchronous update scheduling.
  // Wall-clock stopping cannot guarantee the same final step on repeated
  // runs. For byte-identical final weights, use max_steps without a time
  // budget; fixed-seed trajectories remain reproducible at matching steps.
  std::optional<double> training_seconds;

  // When stop_loss or evaluation_callback is enabled, run an evaluation at
  // this update interval and after the final update.
  int evaluation_interval = 100;
  int evaluation_batches = 1;

  // A negative value disables loss-based stopping. Zero requests exact zero
  // according to the loss layer's FP32 output.
  double stop_loss = -1.0;

  // Prefer a distinct sequential iterator here. Evaluate() resets it before
  // every pass, keeping the sample stable without perturbing random training.
  // If null, training_data is used and its position is reset by evaluation.
  DataSetIterator* evaluation_data = nullptr;

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

  // Optional, non-owning hooks for model/loss forward and backward, including
  // evaluations performed by Train(). Must outlive Train(). These hooks are
  // not passed to dataset transforms or attached to the Executor.
  LayerHooks* layer_hooks = nullptr;
};

struct TrainingResult {
  // Absolute logical step, including TrainingOptions::initial_step.
  int steps_completed;
  bool reached_stop_loss;
  bool reached_time_limit = false;
  // Elapsed time in the update loop, excluding initial setup and any final
  // evaluation triggered by an already-expired time limit.
  double elapsed_training_seconds = 0.0;
};

// Computes mean loss over supervised token/activation rows in the requested
// batches. DataBatch::supervised_row_count excludes padding or prompt rows
// from the denominator; the loss must emit zero for those excluded rows. A
// zero-supervision batch is allowed, but the full evaluation must have at
// least one supervised row. Without explicit counts, every row contributes.
// The model consumes {batch.inputs}; the loss consumes model outputs followed
// by batch.targets and must return one buffer of per-row FP32 loss values.
//
// The returned Buffer contains one device-resident float ordered on executor's
// stream. No device-to-host copy or synchronization is performed. Callers that
// need a CPU value must explicitly read it back. Evaluation never runs backward
// or changes weights, and resets options.eval_data for comparable passes.
absl::StatusOr<Buffer> Evaluate(cuda::Executor& executor, const Layer& model,
                                const EvaluationOptions& options);

// Runs model forward, loss forward/backward, model backward, and an optimizer
// update, with the same wiring for language models and sparse autoencoders.
// Loss inputs are model outputs in order, followed by the dataset target.
// Loss backward must return exactly one gradient per model output, in the same
// order. Targets are constants: no target gradient is returned, even when a
// target aliases an input. Dataset transforms (such as an activation generator)
// are not trained.
//
// Training batches must contain at least one supervised row. Both losses
// expose per-row values for evaluation. Backward scaling belongs
// to the loss: cross-entropy currently differentiates the mean, while SAE
// differentiates the sum. Train does not rescale either gradient.
//
// Train clears gradients before the first backward; ApplyStep applies an update
// and clears gradients afterwards. Referenced layers, optimizer, and iterators
// must remain alive for the call.
absl::StatusOr<TrainingResult> Train(cuda::Executor& executor, Layer& model,
                                     const TrainingOptions& options);

}  // namespace pluto::llm
