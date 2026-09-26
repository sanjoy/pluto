#include "src/llm/experiments/memorize_general_facts/fit_attention_readout/activation_graft.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <vector>

#include "absl/status/status.h"
#include "src/util/status_macros.h"

namespace pluto::llm::fit_attention_readout {

absl::StatusOr<cuda::Buffer> GraftBf16Dimensions(
    cuda::Executor& executor, const cuda::Buffer& base,
    const cuda::Buffer& donor, int model_width,
    absl::Span<const int> zero_based_dimensions) {
  if (model_width <= 0 || zero_based_dimensions.empty())
    return absl::InvalidArgumentError(
        "BF16 graft requires positive model_width and nonempty dimensions");
  std::vector<int> dimensions(zero_based_dimensions.begin(),
                              zero_based_dimensions.end());
  std::sort(dimensions.begin(), dimensions.end());
  if (dimensions.front() < 0 || dimensions.back() >= model_width)
    return absl::InvalidArgumentError(
        "BF16 graft dimensions must be in [0, model_width)");
  if (std::adjacent_find(dimensions.begin(), dimensions.end()) !=
      dimensions.end())
    return absl::InvalidArgumentError("BF16 graft dimensions must be unique");
  if (&base.executor() != &executor || &donor.executor() != &executor)
    return absl::InvalidArgumentError("BF16 graft input executor mismatch");
  const size_t row_bytes = static_cast<size_t>(model_width) * sizeof(uint16_t);
  if (base.size_bytes() == 0 || base.size_bytes() != donor.size_bytes() ||
      base.size_bytes() % row_bytes != 0)
    return absl::InvalidArgumentError(
        "BF16 graft inputs must have equal, nonempty whole-row shapes");

  ASSIGN_OR_RETURN(auto output,
                   cuda::Buffer::Allocate(executor, base.size_bytes()));
  RETURN_IF_ERROR(cuda::CudaStatus(
      cudaMemcpyAsync(output.data(), base.data(), base.size_bytes(),
                      cudaMemcpyDeviceToDevice, executor.stream()),
      "copy BF16 graft base"));
  for (size_t first = 0; first < dimensions.size();) {
    size_t end = first + 1;
    while (end < dimensions.size() &&
           dimensions[end] == dimensions[end - 1] + 1)
      ++end;
    const size_t offset_bytes =
        static_cast<size_t>(dimensions[first]) * sizeof(uint16_t);
    RETURN_IF_ERROR(cuda::CudaStatus(
        cudaMemcpy2DAsync(
            static_cast<uint8_t*>(output.data()) + offset_bytes, row_bytes,
            static_cast<const uint8_t*>(donor.data()) + offset_bytes, row_bytes,
            (end - first) * sizeof(uint16_t), base.size_bytes() / row_bytes,
            cudaMemcpyDeviceToDevice, executor.stream()),
        "copy selected BF16 graft dimensions"));
    first = end;
  }
  return output;
}

absl::StatusOr<cuda::Buffer> GraftBf16LeadingDimensions(
    cuda::Executor& executor, const cuda::Buffer& base,
    const cuda::Buffer& donor, int model_width, int dimensions) {
  if (model_width <= 0 || dimensions <= 0 || dimensions > model_width)
    return absl::InvalidArgumentError(
        "BF16 graft requires positive model_width and dimensions in "
        "[1, model_width]");
  std::vector<int> zero_based_dimensions(dimensions);
  std::iota(zero_based_dimensions.begin(), zero_based_dimensions.end(), 0);
  return GraftBf16Dimensions(executor, base, donor, model_width,
                             zero_based_dimensions);
}

}  // namespace pluto::llm::fit_attention_readout
