#include "src/llm/layers/util.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda_tile.h>

#include "src/util/status_macros.h"

namespace pluto::llm::internal {
namespace {

// Preserve the physical BF16 rounding at public layer boundaries while using
// FP32 scratch for reductions in the imported-checkpoint kernels.
template <class In, class Out>
__tile_global__ void CastKernel(const In* input, int n, Out* output) {
  namespace ct = ::cuda::tiles;
  using namespace ct::literals;
  auto x = ct::partition_view{ct::tensor_span{input, ct::extents{n}},
                              ct::shape{256_ic}};
  auto y = ct::partition_view{ct::tensor_span{output, ct::extents{n}},
                              ct::shape{256_ic}};
  y.store_masked(ct::element_cast<Out>(x.load_masked(ct::bid().x)),
                 ct::bid().x);
}

template <class In, class Out>
absl::StatusOr<Buffer> Cast(cuda::Executor& executor, const Buffer& input,
                            int elements) {
  RETURN_IF_ERROR(ValidatePositiveExtent(elements, "conversion elements"));
  RETURN_IF_ERROR(ValidateBuffer(executor, input, size_t(elements) * sizeof(In),
                                 "conversion input"));
  ASSIGN_OR_RETURN(Buffer output,
                   Buffer::Allocate(executor, size_t(elements) * sizeof(Out)));
  CastKernel<<<1 + (elements - 1) / 256, 1, 0, executor.stream()>>>(
      static_cast<const In*>(input.data()), elements,
      static_cast<Out*>(output.data()));
  RETURN_IF_ERROR(
      cuda::CudaStatus(cudaGetLastError(), "activation CastKernel"));
  return output;
}

}  // namespace

absl::StatusOr<Buffer> AllocateFloatVector(cuda::Executor& executor,
                                           int elements) {
  RETURN_IF_ERROR(ValidatePositiveExtent(elements, "float vector elements"));
  return Buffer::Allocate(executor, size_t(elements) * sizeof(float));
}

absl::StatusOr<Buffer> ToFloat(cuda::Executor& executor, const Buffer& input,
                               int elements) {
  return Cast<__nv_bfloat16, float>(executor, input, elements);
}

absl::StatusOr<Buffer> ToBFloat16(cuda::Executor& executor, const Buffer& input,
                                  int elements) {
  return Cast<float, __nv_bfloat16>(executor, input, elements);
}

absl::Status ValidateBFloat16Inputs(cuda::Executor& executor,
                                    absl::Span<const Buffer> inputs,
                                    absl::Span<const int> element_counts) {
  if (inputs.size() != element_counts.size())
    return absl::InvalidArgumentError("layer received incorrect input count");
  for (size_t i = 0; i < inputs.size(); ++i) {
    RETURN_IF_ERROR(ValidatePositiveExtent(element_counts[i], "BF16 elements"));
    RETURN_IF_ERROR(ValidateBuffer(
        executor, inputs[i], size_t(element_counts[i]) * sizeof(__nv_bfloat16),
        "BF16 input"));
  }
  return absl::OkStatus();
}

}  // namespace pluto::llm::internal
