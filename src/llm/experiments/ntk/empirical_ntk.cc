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

absl::StatusOr<KernelResult> Compute(cuda::Executor& executor, Layer& model,
                                     absl::Span<const Sample> samples,
                                     const KernelOptions& options,
                                     std::vector<ParameterBlock> parameters,
                                     size_t row_count, size_t parameter_count,
                                     size_t jacobian_bytes) {
  ASSIGN_OR_RETURN(auto jacobian, Buffer::Allocate(executor, jacobian_bytes));
  ASSIGN_OR_RETURN(auto values, cuda::PageLockedHostArray<float>::Allocate(
                                    executor, row_count));
  ASSIGN_OR_RETURN(auto one,
                   cuda::PageLockedHostArray<float>::Allocate(executor, 1));
  one[0] = 1;
  size_t row = 0;
  for (const Sample& sample : samples) {
    for (const OutputCoordinate& coordinate : sample.coordinates) {
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
      BufferVec seeds;
      for (size_t i = 0; i < forward.outputs.size(); ++i) {
        const Buffer& output = forward.outputs[i];
        if (&output.executor() != &executor ||
            output.size_bytes() % sizeof(float) != 0)
          return absl::InvalidArgumentError("NTK needs FP32 forward outputs");
        ASSIGN_OR_RETURN(auto seed,
                         Buffer::Allocate(executor, output.size_bytes()));
        RETURN_IF_ERROR(
            cuda::CudaStatus(cudaMemsetAsync(seed.data(), 0, seed.size_bytes(),
                                             executor.stream()),
                             "zero NTK output seed"));
        seeds.push_back(std::move(seed));
      }
      const Buffer& output = forward.outputs[coordinate.output_index];
      if (coordinate.element >= output.size_bytes() / sizeof(float))
        return absl::InvalidArgumentError(
            "NTK output coordinate out of bounds");
      RETURN_IF_ERROR(cuda::CudaStatus(
          cudaMemcpyAsync(
              values.data() + row,
              static_cast<const float*>(output.data()) + coordinate.element,
              sizeof(float), cudaMemcpyDeviceToHost, executor.stream()),
          "read NTK initial output"));
      RETURN_IF_ERROR(cuda::CudaStatus(
          cudaMemcpyAsync(
              static_cast<float*>(seeds[coordinate.output_index].data()) +
                  coordinate.element,
              one.data(), sizeof(float), cudaMemcpyHostToDevice,
              executor.stream()),
          "seed NTK scalar output"));
      // Seeding df/df = 1 measures a row of the OUTPUT Jacobian.
      // Backpropagating cross-entropy instead would produce a different,
      // loss-weighted kernel.
      ASSIGN_OR_RETURN(auto input_gradients,
                       model.bwd(executor, seeds, std::move(forward.state)));
      (void)input_gradients;
      for (const auto& block : parameters)
        RETURN_IF_ERROR(cuda::CudaStatus(
            cudaMemcpyAsync(static_cast<float*>(jacobian.data()) +
                                row * parameter_count + block.offset,
                            model.gradients()[block.weight_index].data(),
                            block.elements * sizeof(float),
                            cudaMemcpyDeviceToDevice, executor.stream()),
            "copy NTK Jacobian row"));
      ++row;
      if (options.progress) {
        RETURN_IF_ERROR(executor.Synchronize());
        RETURN_IF_ERROR(options.progress(row, row_count));
      }
    }
  }
  ASSIGN_OR_RETURN(auto gram, internal::ComputeGram(executor, jacobian,
                                                    row_count, parameter_count));
  // ComputeGram synchronizes this executor, including the earlier output
  // transfers into values. No additional synchronization is needed here.
  KernelResult result;
  result.gram = std::move(gram);
  result.initial_values.assign(values.begin(), values.end());
  result.parameters = std::move(parameters);
  result.parameter_count = parameter_count;
  RETURN_IF_ERROR(ValidateMatrix(result.gram));
  for (double value : result.initial_values)
    if (!std::isfinite(value))
      return absl::FailedPreconditionError(
          "NTK output contains nonfinite values");
  return result;
}

}  // namespace

absl::StatusOr<KernelResult> ComputeEmpiricalKernel(
    cuda::Executor& executor, Layer& model, absl::Span<const Sample> samples,
    const KernelOptions& options) {
  if (samples.empty())
    return absl::InvalidArgumentError("NTK requires at least one sample");
  for (const ActivationType& output : model.output_types())
    if (output.data_type() != DataType::FP32)
      return absl::InvalidArgumentError(
          "NTK requires physical FP32 output signatures");
  size_t row_count = 0;
  for (const Sample& sample : samples) {
    if (sample.inputs.size() != model.input_types().size() ||
        sample.coordinates.empty())
      return absl::InvalidArgumentError(
          "NTK sample needs matching inputs and nonempty output coordinates");
    for (const Buffer& input : sample.inputs)
      if (&input.executor() != &executor)
        return absl::InvalidArgumentError(
            "NTK input belongs to another executor");
    for (const auto& coordinate : sample.coordinates)
      if (coordinate.output_index >= model.output_types().size())
        return absl::InvalidArgumentError("NTK output index out of bounds");
    if (sample.coordinates.size() > 65535 - row_count)
      return absl::ResourceExhaustedError(
          "NTK row count exceeds CUDA grid limit");
    row_count += sample.coordinates.size();
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

}  // namespace pluto::llm::ntk
