#pragma once

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

// Computes the mean of the loss layer's per-token FP32 outputs.
//
// Each dataset batch must use InMemoryDataSetIterator's language-modeling
// schema: batch_size int32 input tokens followed by batch_size int32 targets.
// Inputs are passed to the model; the model output and targets are then passed
// to the loss layer. Evaluate() never runs backward or mutates weights. It
// resets eval_tokens so repeated calls measure the same batches.
absl::StatusOr<double> Evaluate(cuda::Executor& executor, const Layer& model,
                                const Layer& loss_layer,
                                DataSetIterator& eval_tokens,
                                const EvaluationOptions& options);

// Runs a conventional forward/loss/backward/update training loop.
//
// training_tokens owns batch creation and host-to-device staging and must use
// the packed language-modeling schema described by Evaluate(). The loss layer
// must emit one FP32 scalar per input token and seed its own backward pass when
// called with no upstream gradients, as CrossEntropyLossLayer does. Train()
// clears gradients before the first backward; Optimizer::Step() is responsible
// for applying an update and clearing them after every step.
absl::StatusOr<TrainingResult> Train(cuda::Executor& executor, Layer& model,
                                     Layer& loss_layer, Optimizer& optimizer,
                                     DataSetIterator& training_tokens,
                                     const TrainingOptions& options);

}  // namespace pluto::llm
