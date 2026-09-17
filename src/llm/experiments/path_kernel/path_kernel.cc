#include "src/llm/experiments/path_kernel/path_kernel.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "src/cuda/page_locked_host_array.h"
#include "src/llm/experiments/ntk/gram_kernel.h"
#include "src/llm/experiments/path_kernel/update_kernel.h"
#include "src/util/status_macros.h"

namespace pluto::llm::path_kernel {
namespace {

absl::StatusOr<ntk::ScalarOutput> EvaluateCrossEntropy(
    cuda::Executor& executor, absl::Span<const Buffer> outputs,
    ntk::OutputCoordinate first, size_t vocabulary_size, size_t target) {
  if (first.output_index >= outputs.size() || vocabulary_size == 0 ||
      target >= vocabulary_size)
    return absl::InvalidArgumentError("invalid path-kernel cross-entropy row");
  for (const Buffer& output : outputs)
    if (&output.executor() != &executor ||
        output.size_bytes() % sizeof(float) != 0)
      return absl::InvalidArgumentError(
          "cross entropy needs FP32 outputs on executor");
  const Buffer& logits = outputs[first.output_index];
  const size_t elements = logits.size_bytes() / sizeof(float);
  if (first.element > elements || vocabulary_size > elements - first.element)
    return absl::InvalidArgumentError("cross-entropy row extends past output");
  ASSIGN_OR_RETURN(auto host, cuda::PageLockedHostArray<float>::Allocate(
                                  executor, vocabulary_size));
  RETURN_IF_ERROR(cuda::CudaStatus(
      cudaMemcpyAsync(
          host.data(), static_cast<const float*>(logits.data()) + first.element,
          host.size_bytes(), cudaMemcpyDeviceToHost, executor.stream()),
      "read path-kernel cross-entropy logits"));
  RETURN_IF_ERROR(executor.Synchronize());
  double maximum = -std::numeric_limits<double>::infinity();
  for (float value : host.span()) {
    if (std::isnan(value) || value == std::numeric_limits<float>::infinity())
      return absl::InvalidArgumentError("nonfinite cross-entropy logit");
    maximum = std::max(maximum, static_cast<double>(value));
  }
  if (!std::isfinite(maximum) || !std::isfinite(host[target]))
    return absl::InvalidArgumentError(
        "cross-entropy target/loss is not finite");
  double denominator = 0;
  for (float value : host.span())
    denominator += std::exp(static_cast<double>(value) - maximum);
  const double loss =
      maximum - static_cast<double>(host[target]) + std::log(denominator);
  // Transform the pinned readback in place into d(loss)/d(logit). Sum all
  // logical classes, not only the query coordinates being inspected.
  for (size_t i = 0; i < vocabulary_size; ++i)
    host[i] = static_cast<float>(
        std::exp(static_cast<double>(host[i]) - maximum) / denominator -
        static_cast<double>(i == target));
  BufferVec gradients;
  for (const Buffer& output : outputs) {
    ASSIGN_OR_RETURN(auto seed, Buffer::Allocate(executor, output.size_bytes()));
    RETURN_IF_ERROR(cuda::CudaStatus(
        cudaMemsetAsync(seed.data(), 0, seed.size_bytes(), executor.stream()),
        "zero path-kernel loss seed"));
    gradients.push_back(std::move(seed));
  }
  RETURN_IF_ERROR(cuda::CudaStatus(
      cudaMemcpyAsync(
          static_cast<float*>(gradients[first.output_index].data()) +
              first.element,
          host.data(), host.size_bytes(), cudaMemcpyHostToDevice,
          executor.stream()),
      "seed full-vocabulary path-kernel loss"));
  return ntk::ScalarOutput{loss, std::move(gradients)};
}

// Read selected query values after an update without doing another backward
// pass or allocating a full-output seed for every coordinate.
absl::StatusOr<std::vector<double>> QueryValues(
    cuda::Executor& executor, Layer& model,
    absl::Span<const ntk::Sample> queries, size_t count) {
  ASSIGN_OR_RETURN(auto host,
                   cuda::PageLockedHostArray<float>::Allocate(executor, count));
  size_t row = 0;
  for (const auto& query : queries) {
    ASSIGN_OR_RETURN(auto forward, model.fwd(executor, query.inputs));
    if (forward.outputs.size() != model.output_types().size())
      return absl::InvalidArgumentError(
          "path-kernel forward output count changed");
    for (const auto& coordinate : query.coordinates) {
      if (coordinate.output_index >= forward.outputs.size())
        return absl::InvalidArgumentError(
            "path-kernel query output is out of bounds");
      const Buffer& output = forward.outputs[coordinate.output_index];
      if (&output.executor() != &executor ||
          output.size_bytes() % sizeof(float) != 0 ||
          coordinate.element >= output.size_bytes() / sizeof(float))
        return absl::InvalidArgumentError(
            "invalid path-kernel query output buffer");
      RETURN_IF_ERROR(cuda::CudaStatus(
          cudaMemcpyAsync(
              host.data() + row,
              static_cast<const float*>(output.data()) + coordinate.element,
              sizeof(float), cudaMemcpyDeviceToHost, executor.stream()),
          "read path-kernel query output"));
      ++row;
    }
  }
  RETURN_IF_ERROR(executor.Synchronize());
  std::vector<double> values(host.begin(), host.end());
  for (double value : values)
    if (!std::isfinite(value))
      return absl::OutOfRangeError("path-kernel query became nonfinite");
  return values;
}

absl::Status CopyWeights(cuda::Executor& executor, Layer& model,
                         absl::Span<const ntk::ParameterBlock> parameters,
                         const Buffer& packed, bool restore) {
  absl::Status status;
  for (const auto& block : parameters) {
    void* weight = model.weights()[block.weight_index].data();
    void* backup = static_cast<float*>(packed.data()) + block.offset;
    status.Update(cuda::CudaStatus(
        cudaMemcpyAsync(restore ? weight : backup, restore ? backup : weight,
                        block.elements * sizeof(float),
                        cudaMemcpyDeviceToDevice, executor.stream()),
        restore ? "restore path-kernel weights" : "save path-kernel weights"));
  }
  return status;
}

absl::StatusOr<Result> Advance(
    cuda::Executor& executor, Layer& model,
    absl::Span<const TrainingExample> training,
    absl::Span<const ntk::Sample> queries,
    absl::Span<const ntk::DifferentiationSample> samples,
    const Options& options, ntk::JacobianResult initial, size_t query_count) {
  const size_t n = training.size();
  const size_t rows = query_count + n;
  Result result;
  result.parameters = initial.parameters;
  result.parameter_count = initial.parameter_count;
  result.initial_values.assign(initial.values.begin(),
                               initial.values.begin() + query_count);
  result.final_values = result.initial_values;
  result.path_kernel = {query_count, query_count,
                        std::vector<double>(query_count * query_count)};
  result.contributions = {query_count, n, std::vector<double>(query_count * n)};
  result.steps.reserve(options.steps);
  ntk::KernelOptions jacobian_options;
  jacobian_options.max_jacobian_bytes = options.max_jacobian_bytes;
  for (int step = 0; step < options.steps; ++step) {
    // The Jacobian lives only for this iteration. Replacing an outer-scope
    // buffer would allocate the next matrix BEFORE releasing the previous
    // one, temporarily doubling the explicitly budgeted device storage.
    ASSIGN_OR_RETURN(
        auto measured,
        step == 0
            ? absl::StatusOr<ntk::JacobianResult>(std::move(initial))
            : ntk::ComputeJacobian(executor, model, samples, jacobian_options));
    if (measured.parameter_count != result.parameter_count ||
        measured.parameters.size() != result.parameters.size())
      return absl::FailedPreconditionError(
          "path-kernel parameter layout changed");
    for (size_t i = 0; i < result.parameters.size(); ++i)
      if (measured.parameters[i].weight_index !=
              result.parameters[i].weight_index ||
          measured.parameters[i].elements != result.parameters[i].elements ||
          measured.parameters[i].offset != result.parameters[i].offset)
        return absl::FailedPreconditionError(
            "path-kernel parameter layout changed");
    // Query output rows and full-loss rows share one feature space and one
    // immutable pre-update parameter setting. The QxN crossblock contracts
    // the entire vector-valued tangent kernel with each example's loss seed.
    ASSIGN_OR_RETURN(auto gram,
                     ntk::internal::ComputeGram(executor, measured.derivatives,
                                                rows, measured.parameter_count));
    StepResult record;
    record.step = step + 1;
    record.values_before.assign(measured.values.begin(),
                                measured.values.begin() + query_count);
    record.training_losses.assign(measured.values.begin() + query_count,
                                  measured.values.end());
    record.tangent_kernel = {query_count, query_count,
                             std::vector<double>(query_count * query_count)};
    record.contributions = {query_count, n,
                            std::vector<double>(query_count * n)};
    record.predicted_delta.resize(query_count);
    for (size_t q = 0; q < query_count; ++q) {
      for (size_t other = 0; other < query_count; ++other) {
        record.tangent_kernel(q, other) = gram(q, other);
        result.path_kernel(q, other) += options.learning_rate * gram(q, other);
      }
      for (size_t i = 0; i < n; ++i) {
        const double contribution =
            -(options.learning_rate / static_cast<double>(n)) *
            gram(q, query_count + i);
        record.contributions(q, i) = contribution;
        result.contributions(q, i) += contribution;
        record.predicted_delta[q] += contribution;
      }
    }
    RETURN_IF_ERROR(ntk::ValidateMatrix(result.path_kernel));
    RETURN_IF_ERROR(ntk::ValidateMatrix(result.contributions));
    // Update once AFTER all example/query derivatives have been measured.
    // The metadata deduplicates tied embedding/head handles; the packed loss
    // derivative already sums the contribution of every use of that tensor.
    for (const auto& block : measured.parameters)
      RETURN_IF_ERROR(internal::ApplyGradientDescent(
          executor, model.weights()[block.weight_index], measured.derivatives,
          measured.parameter_count, block.offset, query_count, n,
          options.learning_rate));
    ASSIGN_OR_RETURN(record.values_after,
                     QueryValues(executor, model, queries, query_count));
    record.residual.resize(query_count);
    for (size_t q = 0; q < query_count; ++q) {
      record.residual[q] = record.values_after[q] - record.values_before[q] -
                           record.predicted_delta[q];
      if (!std::isfinite(record.residual[q]))
        return absl::OutOfRangeError("path-kernel step residual is nonfinite");
    }
    result.final_values = record.values_after;
    result.steps.push_back(std::move(record));
    if (options.progress)
      RETURN_IF_ERROR(options.progress(result.steps.back()));
  }
  result.reconstructed_values = result.initial_values;
  result.residual.resize(query_count);
  for (size_t q = 0; q < query_count; ++q) {
    for (size_t i = 0; i < n; ++i)
      result.reconstructed_values[q] += result.contributions(q, i);
    result.residual[q] =
        result.final_values[q] - result.reconstructed_values[q];
    if (!std::isfinite(result.residual[q]))
      return absl::OutOfRangeError("path-kernel total residual is nonfinite");
  }
  for (const auto& example : training) {
    ASSIGN_OR_RETURN(auto forward, model.fwd(executor, example.inputs));
    ASSIGN_OR_RETURN(auto loss, example.loss(executor, forward.outputs));
    if (!std::isfinite(loss.value))
      return absl::OutOfRangeError("final path-kernel loss is nonfinite");
    result.final_training_losses.push_back(loss.value);
  }
  RETURN_IF_ERROR(executor.Synchronize());
  return result;
}

}  // namespace

ntk::ScalarFunction CrossEntropy(ntk::OutputCoordinate first_logit,
                                 size_t vocabulary_size, size_t target_class) {
  return [=](cuda::Executor& executor, absl::Span<const Buffer> outputs) {
    return EvaluateCrossEntropy(executor, outputs, first_logit, vocabulary_size,
                                target_class);
  };
}

absl::StatusOr<Result> Run(cuda::Executor& executor, Layer& model,
                           absl::Span<const TrainingExample> training,
                           absl::Span<const ntk::Sample> queries,
                           const Options& options) {
  if (training.empty() || queries.empty() || options.steps < 0 ||
      !std::isfinite(options.learning_rate) || options.learning_rate <= 0)
    return absl::InvalidArgumentError("invalid path-kernel data or GD options");
  size_t query_count = 0;
  std::vector<ntk::DifferentiationSample> samples;
  for (const auto& query : queries) {
    if (query.coordinates.empty() ||
        query.coordinates.size() > 65535 - query_count)
      return absl::InvalidArgumentError(
          "path-kernel needs 1..65535 query rows");
    query_count += query.coordinates.size();
    ntk::DifferentiationSample sample{query.inputs, {}};
    for (const auto& coordinate : query.coordinates)
      sample.outputs.push_back(ntk::MakeOutputCoordinate(coordinate));
    samples.push_back(std::move(sample));
  }
  if (training.size() > 65535 - query_count)
    return absl::ResourceExhaustedError(
        "path-kernel total Jacobian rows exceed 65535");
  for (const auto& example : training) {
    if (!example.loss)
      return absl::InvalidArgumentError(
          "path-kernel example has no loss callback");
    samples.push_back({example.inputs, {example.loss}});
  }
  ntk::KernelOptions jacobian_options;
  jacobian_options.max_jacobian_bytes = options.max_jacobian_bytes;
  ASSIGN_OR_RETURN(auto measured, ntk::ComputeJacobian(executor, model, samples,
                                                       jacobian_options));
  ASSIGN_OR_RETURN(
      auto backup,
      Buffer::Allocate(executor, measured.parameter_count * sizeof(float)));
  const auto parameters = measured.parameters;
  RETURN_IF_ERROR(CopyWeights(executor, model, parameters, backup, false));
  auto result = Advance(executor, model, training, queries, samples, options,
                        std::move(measured), query_count);
  if (result.ok())
    return result;
  auto restore = CopyWeights(executor, model, parameters, backup, true);
  restore.Update(executor.Synchronize());
  if (restore.ok())
    return result;
  return absl::Status(
      result.status().code(),
      absl::StrCat(
          result.status().message(),
          "; restoring path-kernel weights also failed: ", restore.message()));
}

}  // namespace pluto::llm::path_kernel
