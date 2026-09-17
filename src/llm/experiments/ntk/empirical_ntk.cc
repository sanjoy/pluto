#include "src/llm/experiments/ntk/empirical_ntk.h"

#include <cuda_runtime_api.h>

#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "src/cuda/page_locked_host_array.h"
#include "src/llm/experiments/ntk/gram_kernel.h"
#include "src/util/status_macros.h"

namespace pluto::llm::ntk {
namespace {

absl::StatusOr<size_t> Product(size_t a, size_t b) {
  if (b != 0 && a > std::numeric_limits<size_t>::max() / b)
    return absl::InvalidArgumentError("NTK tensor size overflows size_t");
  return a * b;
}

absl::StatusOr<std::vector<ParameterBlock>> Parameters(cuda::Executor& executor,
                                                       Layer& model) {
  const auto weights = model.weights();
  const auto gradients = model.gradients();
  if (weights.size() != gradients.size())
    return absl::InvalidArgumentError(
        "NTK needs one FP32 gradient accumulator per weight");
  absl::flat_hash_map<void*, void*> seen_weights;
  absl::flat_hash_set<void*> seen_gradients;
  std::vector<ParameterBlock> blocks;
  size_t offset = 0;
  for (size_t i = 0; i < weights.size(); ++i) {
    const Buffer& weight = weights[i];
    const Buffer& gradient = gradients[i];
    if (&weight.executor() != &executor || &gradient.executor() != &executor ||
        weight.size_bytes() == 0 || weight.size_bytes() % sizeof(float) != 0 ||
        weight.size_bytes() != gradient.size_bytes())
      return absl::InvalidArgumentError(
          "NTK weights/gradients must be matching nonempty FP32 buffers on "
          "the supplied executor");
    const auto [entry, inserted] =
        seen_weights.emplace(weight.data(), gradient.data());
    if (!inserted) {
      if (entry->second != gradient.data())
        return absl::InvalidArgumentError(
            "tied NTK weights must share their gradient accumulator");
      continue;
    }
    if (!seen_gradients.insert(gradient.data()).second)
      return absl::InvalidArgumentError(
          "distinct NTK weights cannot share a gradient accumulator");
    const size_t elements = weight.size_bytes() / sizeof(float);
    if (elements > std::numeric_limits<size_t>::max() - offset)
      return absl::InvalidArgumentError("NTK parameter count overflows");
    blocks.push_back({i, elements, offset});
    offset += elements;
  }
  for (const auto& [weight, gradient] : seen_weights)
    if (seen_weights.contains(gradient))
      return absl::InvalidArgumentError(
          "NTK gradient storage cannot alias parameter storage");
  if (blocks.empty())
    return absl::InvalidArgumentError(
        "NTK model must have trainable parameters");
  return blocks;
}

absl::Status CopyGradients(cuda::Executor& executor, Layer& model,
                           absl::Span<const ParameterBlock> parameters,
                           const Buffer& packed, bool restore) {
  absl::Status status;
  auto gradients = model.gradients();
  for (const auto& block : parameters) {
    void* packed_data = static_cast<float*>(packed.data()) + block.offset;
    void* gradient_data = gradients[block.weight_index].data();
    // Attempt every restore even if one CUDA operation reports an error.
    status.Update(cuda::CudaStatus(
        cudaMemcpyAsync(restore ? gradient_data : packed_data,
                        restore ? packed_data : gradient_data,
                        block.elements * sizeof(float),
                        cudaMemcpyDeviceToDevice, executor.stream()),
        restore ? "restore NTK parameter gradients" : "save NTK gradients"));
  }
  return status;
}

absl::StatusOr<JacobianResult> Compute(
    cuda::Executor& executor, Layer& model,
    absl::Span<const DifferentiationSample> samples,
    const KernelOptions& options, std::vector<ParameterBlock> parameters,
    size_t row_count, size_t parameter_count, size_t jacobian_bytes) {
  ASSIGN_OR_RETURN(auto jacobian, Buffer::Allocate(executor, jacobian_bytes));
  std::vector<double> values;
  values.reserve(row_count);
  size_t row = 0;
  for (const DifferentiationSample& sample : samples) {
    for (const ScalarFunction& function : sample.outputs) {
      for (const auto& block : parameters) {
        const Buffer& gradient = model.gradients()[block.weight_index];
        RETURN_IF_ERROR(cuda::CudaStatus(
            cudaMemsetAsync(gradient.data(), 0, gradient.size_bytes(),
                            executor.stream()),
            "clear NTK parameter gradient"));
      }
      ASSIGN_OR_RETURN(auto forward, model.fwd(executor, sample.inputs));
      if (forward.outputs.size() != model.output_types().size())
        return absl::InvalidArgumentError("NTK forward output count mismatch");
      for (const Buffer& output : forward.outputs)
        if (&output.executor() != &executor ||
            output.size_bytes() % sizeof(float) != 0)
          return absl::InvalidArgumentError("NTK needs FP32 forward outputs");
      ASSIGN_OR_RETURN(auto scalar, function(executor, forward.outputs));
      if (!std::isfinite(scalar.value))
        return absl::FailedPreconditionError(
            "Jacobian scalar callback returned a nonfinite value");
      if (scalar.gradients.size() != forward.outputs.size())
        return absl::InvalidArgumentError(
            "Jacobian callback must return one gradient per model output");
      for (size_t i = 0; i < scalar.gradients.size(); ++i)
        if (&scalar.gradients[i].executor() != &executor ||
            scalar.gradients[i].size_bytes() != forward.outputs[i].size_bytes())
          return absl::InvalidArgumentError(
              "Jacobian callback gradients must match FP32 output sizes and "
              "belong to the supplied executor");
      ASSIGN_OR_RETURN(
          auto input_gradients,
          model.bwd(executor, scalar.gradients, std::move(forward.state)));
      (void)input_gradients;
      for (const auto& block : parameters)
        RETURN_IF_ERROR(cuda::CudaStatus(
            cudaMemcpyAsync(static_cast<float*>(jacobian.data()) +
                                row * parameter_count + block.offset,
                            model.gradients()[block.weight_index].data(),
                            block.elements * sizeof(float),
                            cudaMemcpyDeviceToDevice, executor.stream()),
            "copy NTK Jacobian row"));
      values.push_back(scalar.value);
      ++row;
      if (options.progress) {
        RETURN_IF_ERROR(executor.Synchronize());
        RETURN_IF_ERROR(options.progress(row, row_count));
      }
    }
  }
  return JacobianResult{std::move(jacobian), std::move(values),
                        std::move(parameters), parameter_count};
}

}  // namespace

ScalarFunction MakeOutputCoordinate(OutputCoordinate coordinate) {
  return [coordinate](
             cuda::Executor& executor,
             absl::Span<const Buffer> outputs) -> absl::StatusOr<ScalarOutput> {
    if (coordinate.output_index >= outputs.size())
      return absl::InvalidArgumentError("NTK output index out of bounds");
    const Buffer& selected = outputs[coordinate.output_index];
    if (coordinate.element >= selected.size_bytes() / sizeof(float))
      return absl::InvalidArgumentError("NTK output coordinate out of bounds");
    BufferVec gradients;
    for (const Buffer& output : outputs) {
      if (&output.executor() != &executor ||
          output.size_bytes() % sizeof(float) != 0)
        return absl::InvalidArgumentError("NTK needs FP32 forward outputs");
    ASSIGN_OR_RETURN(auto seed, Buffer::Allocate(executor, output.size_bytes()));
      RETURN_IF_ERROR(cuda::CudaStatus(
          cudaMemsetAsync(seed.data(), 0, seed.size_bytes(), executor.stream()),
          "zero NTK output seed"));
      gradients.push_back(std::move(seed));
    }
    ASSIGN_OR_RETURN(auto host,
                     cuda::PageLockedHostArray<float>::Allocate(executor, 2));
    host[0] = 1;
    RETURN_IF_ERROR(cuda::CudaStatus(
        cudaMemcpyAsync(
            static_cast<float*>(gradients[coordinate.output_index].data()) +
                coordinate.element,
            host.data(), sizeof(float), cudaMemcpyHostToDevice,
            executor.stream()),
        "seed NTK scalar output"));
    RETURN_IF_ERROR(cuda::CudaStatus(
        cudaMemcpyAsync(
            host.data() + 1,
            static_cast<const float*>(selected.data()) + coordinate.element,
            sizeof(float), cudaMemcpyDeviceToHost, executor.stream()),
        "read NTK initial output"));
    RETURN_IF_ERROR(executor.Synchronize());
    return ScalarOutput{host[1], std::move(gradients)};
  };
}

absl::StatusOr<JacobianResult> ComputeJacobian(
    cuda::Executor& executor, Layer& model,
    absl::Span<const DifferentiationSample> samples,
    const KernelOptions& options) {
  if (samples.empty())
    return absl::InvalidArgumentError("NTK requires at least one sample");
  for (const ActivationType& output : model.output_types())
    if (output.data_type() != DataType::FP32)
      return absl::InvalidArgumentError(
          "NTK requires physical FP32 output signatures");
  size_t row_count = 0;
  for (const DifferentiationSample& sample : samples) {
    if (sample.inputs.size() != model.input_types().size() ||
        sample.outputs.empty())
      return absl::InvalidArgumentError(
          "NTK sample needs matching inputs and nonempty output coordinates");
    for (const Buffer& input : sample.inputs)
      if (&input.executor() != &executor)
        return absl::InvalidArgumentError(
            "NTK input belongs to another executor");
    for (const ScalarFunction& function : sample.outputs)
      if (!function)
        return absl::InvalidArgumentError("Jacobian scalar callback is empty");
    if (sample.outputs.size() > 65535 - row_count)
      return absl::ResourceExhaustedError(
          "NTK row count exceeds CUDA grid limit");
    row_count += sample.outputs.size();
  }
  ASSIGN_OR_RETURN(auto parameters, Parameters(executor, model));
  const size_t count = parameters.back().offset + parameters.back().elements;
  ASSIGN_OR_RETURN(size_t gradient_bytes, Product(count, sizeof(float)));
  ASSIGN_OR_RETURN(size_t jacobian_bytes, Product(row_count, gradient_bytes));
  if (jacobian_bytes > options.max_jacobian_bytes)
    return absl::ResourceExhaustedError(
        absl::StrCat("NTK Jacobian needs ", jacobian_bytes, " bytes for ",
                     row_count, " rows and ", count, " parameters; limit is ",
                     options.max_jacobian_bytes,
                     ". Select fewer coordinates/examples or "
                     "increase the explicit memory budget."));

  ASSIGN_OR_RETURN(auto backup, Buffer::Allocate(executor, gradient_bytes));
  RETURN_IF_ERROR(CopyGradients(executor, model, parameters, backup, false));
  auto result = Compute(executor, model, samples, options, parameters,
                        row_count, count, jacobian_bytes);
  auto restore = CopyGradients(executor, model, parameters, backup, true);
  restore.Update(executor.Synchronize());
  if (restore.ok())
    return result;
  if (result.ok())
    return restore;
  return absl::Status(result.status().code(),
                      absl::StrCat(result.status().message(),
                                   "; restoring NTK gradients also failed: ",
                                   restore.message()));
}

absl::StatusOr<KernelResult> ComputeEmpiricalKernel(
    cuda::Executor& executor, Layer& model, absl::Span<const Sample> samples,
    const KernelOptions& options) {
  std::vector<DifferentiationSample> differentiation_samples;
  differentiation_samples.reserve(samples.size());
  for (const Sample& sample : samples) {
    DifferentiationSample converted{.inputs = sample.inputs};
    for (const OutputCoordinate coordinate : sample.coordinates) {
      if (coordinate.output_index >= model.output_types().size())
        return absl::InvalidArgumentError("NTK output index out of bounds");
      converted.outputs.push_back(MakeOutputCoordinate(coordinate));
    }
    differentiation_samples.push_back(std::move(converted));
  }
  ASSIGN_OR_RETURN(
      auto jacobian,
      ComputeJacobian(executor, model, differentiation_samples, options));
  ASSIGN_OR_RETURN(
      auto gram,
      internal::ComputeGram(executor, jacobian.derivatives,
                            jacobian.values.size(), jacobian.parameter_count));
  RETURN_IF_ERROR(ValidateMatrix(gram));
  return KernelResult{std::move(gram), std::move(jacobian.values),
                      std::move(jacobian.parameters), jacobian.parameter_count};
}

}  // namespace pluto::llm::ntk
