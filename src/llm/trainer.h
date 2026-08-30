#pragma once

#include <optional>

#include "absl/status/statusor.h"
#include "src/dataset.h"
#include "src/llm/layer.h"
#include "src/llm/optimizer.h"

namespace pluto::llm {

struct EvaluationOptions {
  // Evaluate exactly this many batches after resetting the iterator.
  int batches = 1;
};

struct TrainingOptions {
  // Hard cap on optimizer updates. Zero is useful for evaluation-only runs.
  int max_steps = 0;

  // When stop_loss is enabled, run an evaluation at this update interval and
  // after the final update.
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
};

struct TrainingResult {
  int steps_completed;
  bool reached_stop_loss;
};

// Computes the mean of the loss layer's per-token FP32 outputs.
//
// Each dataset batch is passed to the model as {tokens}; the model output and
// batch targets are then passed to the loss layer as {output, targets}.
// Evaluate() never runs backward or mutates weights. It resets eval_tokens so
// repeated calls measure the same batches.
absl::StatusOr<double> Evaluate(Layer* model, Layer* loss_layer,
                                DataSetIterator* eval_tokens,
                                const EvaluationOptions& options);

// Runs a conventional forward/loss/backward/update training loop.
//
// training_tokens owns batch creation and host-to-device staging. The loss
// layer must emit one FP32 scalar per input token and seed its own backward
// pass when called with no upstream gradients, as CrossEntropyLossLayer does.
// Train() clears gradients before the first backward; Optimizer::Step() is
// responsible for applying an update and clearing them after every step.
absl::StatusOr<TrainingResult> Train(Layer* model, Layer* loss_layer,
                                     Optimizer* optimizer,
                                     DataSetIterator* training_tokens,
                                     const TrainingOptions& options);

}  // namespace pluto::llm
