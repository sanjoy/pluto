#pragma once

#include <cstddef>
#include <functional>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "src/llm/experiments/ntk/empirical_ntk.h"

namespace pluto::llm::path_kernel {

// An example contributes one scalar loss, whose callback returns the FULL
// derivative with respect to every model output. Query-coordinate selection
// never restricts this derivative (especially the softmax denominator).
struct TrainingExample {
  BufferVec inputs;
  ntk::ScalarFunction loss;
};

struct StepResult {
  int step;  // One-based count of completed, actual parameter updates.
  std::vector<double> values_before;
  std::vector<double> values_after;
  std::vector<double> training_losses;  // Before the update, one per example.
  ntk::Matrix tangent_kernel;  // Query-output Gram J_q J_q^T at theta_s.
  ntk::Matrix contributions;   // [query coordinate, training example].
  std::vector<double> predicted_delta;
  std::vector<double> residual;  // Observed delta minus predicted_delta.
};

struct Options {
  int steps = 3;
  double learning_rate = 1e-5;
  // Explicit query AND loss Jacobian storage only; additional weights, saved
  // state, backup weights/gradients, and Gram scratch need GPU memory too.
  size_t max_jacobian_bytes = size_t{4} * 1024 * 1024 * 1024;
  // Runs synchronously after each completed update. It must not modify model
  // or input buffers. Returning an error cancels and rolls back ALL updates.
  std::function<absl::Status(const StepResult&)> progress;
};

struct Result {
  std::vector<double> initial_values;
  std::vector<double> final_values;
  std::vector<double> reconstructed_values;
  std::vector<double> residual;
  std::vector<double> final_training_losses;
  ntk::Matrix path_kernel;    // Sum_s eta * J_q(theta_s) J_q(theta_s)^T.
  ntk::Matrix contributions;  // Sum_s -eta/N * J_q(theta_s) grad(loss_i).
  std::vector<StepResult> steps;
  std::vector<ntk::ParameterBlock> parameters;
  size_t parameter_count = 0;
};

// Domingos (2020), loss-weighted path representation (Remark 2), discretized
// along actual full-batch plain gradient descent on the MEAN example loss:
//   theta_{s+1} = theta_s - eta/N * sum_i grad(loss_i(theta_s))
//   C[q,i] += -eta/N * <grad(f_q(theta_s)), grad(loss_i(theta_s))>.
// Unlike a frozen NTK, derivatives and losses are recomputed at every step.
// All derivatives for a step use exactly the same pre-update parameters.
// This is neither AdamW nor a regression fit to an averaged kernel. The
// signed contributions are NOT themselves a symmetric positive-semidefinite
// kernel; path_kernel separately integrates the unweighted query Gram.
//
// Returns f_initial + sum_i C and the separately measured final model output.
// Their difference includes finite-step Taylor error, reduced-precision
// backward conventions, and rounding. Never assume the difference is zero.
// Query ordering is sample-major then coordinate-major, as in the NTK API.
// Every unique FP32 master parameter participates; tied weights update once.
// All transfers use pinned memory and reductions/updates have fixed order.
//
// On success the model retains its final weights; preexisting gradient
// accumulators remain unchanged. On failure original weights are restored
// (unless CUDA prevents recovery). Callbacks must not mutate model state,
// and the model/inputs must not be used concurrently during this operation.
absl::StatusOr<Result> Run(cuda::Executor& executor, Layer& model,
                           absl::Span<const TrainingExample> training,
                           absl::Span<const ntk::Sample> queries,
                           const Options& options = {});

// Full-vocabulary next-token cross entropy on a contiguous logit row starting
// at first_logit. target_class is relative to that row, not a flattened index.
// Only vocabulary_size logical logits enter softmax: physical padding and
// other token positions have zero seed. Stable FP64 CPU log-sum-exp computes
// the scalar loss and derivative, which is cast to FP32 for model backward.
// This avoids materializing one Jacobian row for each vocabulary entry while
// retaining ALL their contributions in the loss vector-Jacobian product.
ntk::ScalarFunction CrossEntropy(ntk::OutputCoordinate first_logit,
                                 size_t vocabulary_size, size_t target_class);

}  // namespace pluto::llm::path_kernel
