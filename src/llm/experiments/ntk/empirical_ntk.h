#pragma once

#include <cstddef>
#include <functional>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "src/cuda/executor.h"
#include "src/llm/experiments/ntk/kernel_regression.h"
#include "src/llm/layer.h"

namespace pluto::llm::ntk {

// One scalar component of a layer's forward output, not a loss gradient.
// element is a flattened FP32 element offset (including batch/token axes).
// Selecting different components on different examples computes the matching
// submatrix of the vector-valued NTK, including its cross-output entries.
struct OutputCoordinate {
  size_t output_index = 0;
  size_t element = 0;
};

struct Sample {
  // Buffers must belong to the supplied executor and remain unchanged during
  // the call. Copy reusable dataset batches before advancing their iterator.
  BufferVec inputs;
  std::vector<OutputCoordinate> coordinates;
};

// The concatenated parameter space uses first-occurrence weights() order.
// Tied weights occupy one block and their accumulated derivative is counted
// once. Offsets/elements count floats, not bytes; weight_index is the original
// index in Layer::weights(), not a checkpoint filename index.
struct ParameterBlock {
  size_t weight_index;
  size_t elements;
  size_t offset;
};

struct KernelResult {
  // Rows follow sample order, then each sample's coordinate order. This is
  // raw J J^T: no division by parameter count, diagonal normalization, output
  // trace, loss weighting, parameter subsampling, or random projection.
  Matrix gram;
  std::vector<double> initial_values;
  std::vector<ParameterBlock> parameters;
  size_t parameter_count = 0;
};

struct KernelOptions {
  // Limit the explicit device Jacobian only. The model, saved forward state,
  // one gradient backup, seeds, and reduction scratch need additional memory.
  // Reject oversized experiments instead of silently approximating the NTK.
  size_t max_jacobian_bytes = size_t{4} * 1024 * 1024 * 1024;
  // Called after a completed Jacobian row (and stream synchronization).
  // Returning an error cancels, still restoring the original gradients.
  std::function<absl::Status(size_t completed, size_t total)> progress;
};

// A scalar function of the model's forward outputs and its derivatives with
// respect to those outputs. For example, selecting one logit supplies a
// one-hot seed, whereas a scalar loss supplies dLoss/dLogits. Each gradient
// must be an FP32 buffer matching the corresponding output's size/executor.
struct ScalarOutput {
  double value;
  BufferVec gradients;
};

// Callbacks must not mutate forward outputs, model weights, or parameter
// gradients. Queue all device work on the supplied executor. Return a finite
// host scalar and exactly one derivative buffer per forward output.
using ScalarFunction = std::function<absl::StatusOr<ScalarOutput>(
    cuda::Executor&, absl::Span<const Buffer>)>;

struct DifferentiationSample {
  // The same input ownership/immutability rules as Sample apply here.
  BufferVec inputs;
  std::vector<ScalarFunction> outputs;
};

struct JacobianResult {
  // Row-major FP32 device matrix: values.size() rows, parameter_count columns.
  // The executor must outlive this buffer. Parameter aliases are deduplicated
  // only after their backward contributions have accumulated.
  Buffer derivatives;
  std::vector<double> values;
  std::vector<ParameterBlock> parameters;
  size_t parameter_count = 0;
};

// Constructs a scalar callback selecting one flattened forward output. It
// reads the value through pinned memory and creates its one-hot derivative.
ScalarFunction MakeOutputCoordinate(OutputCoordinate coordinate);

// Computes parameter derivatives of arbitrary scalar functions of the model
// outputs. Rows follow sample order, then scalar-function order. Each row gets
// a fresh forward/backward pass; existing parameter gradients are restored
// on both success and failure. Parameters and physical output signatures must
// be FP32 (mixed-precision internal operations remain supported). The explicit
// memory budget and progress callback are shared with ComputeEmpiricalKernel.
absl::StatusOr<JacobianResult> ComputeJacobian(
    cuda::Executor& executor, Layer& model,
    absl::Span<const DifferentiationSample> samples,
    const KernelOptions& options = {});

// Measures the finite network's empirical NTK at its CURRENT weights:
//   K[(sample, output), (other, output')] =
//       sum over unique parameters p of df_output/dp * df_other_output'/dp.
// This is not the infinite-width limiting kernel, nor a claim that this kernel
// stays constant under training. Master parameters/gradients must be FP32 and
// all forward output buffers must have physical FP32 signatures. BF16 models
// with FP32 logits work, but their derivatives are the implemented
// mixed-precision backward rule, not derivatives of a smooth full-precision
// function. The legacy FP16 policy also rounds matrix operands despite FP32
// output storage.
//
// Each scalar uses a fresh forward/backward pass: backward states need not be
// reusable or immutable. No optimizer is invoked. Weights are never written,
// and preexisting parameter gradients are backed up and restored on success
// or failure (unless CUDA itself prevents restoration). The model must not be
// used concurrently. The result is CPU-owned; all device transfers use pinned
// storage and reductions use a fixed, atomic-free FP64 accumulation order.
absl::StatusOr<KernelResult> ComputeEmpiricalKernel(
    cuda::Executor& executor, Layer& model, absl::Span<const Sample> samples,
    const KernelOptions& options = {});

}  // namespace pluto::llm::ntk
