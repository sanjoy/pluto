#include "src/llm/experiments/one_shot_memorizer/quantization.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include "absl/status/status.h"

namespace pluto::llm::one_shot_memorizer {

absl::StatusOr<QuantizedTensor> QuantizeTensorSymmetric(
    absl::Span<const float> tensor, int bits) {
  if (bits < 2 || bits > 16)
    return absl::InvalidArgumentError(
        "quantization bits must be between 2 and 16");
  if (tensor.empty())
    return absl::InvalidArgumentError("quantization tensor must not be empty");
  if (tensor.size() > std::vector<float>().max_size())
    return absl::OutOfRangeError(
        "quantized tensor exceeds supported storage range");
  double maximum = 0;
  for (float value : tensor) {
    if (!std::isfinite(value))
      return absl::InvalidArgumentError("quantization values must be finite");
    maximum = std::max(maximum, std::abs(static_cast<double>(value)));
  }
  const int maximum_code = (1 << (bits - 1)) - 1;
  QuantizedTensor result{.values = std::vector<float>(tensor.size(), 0),
                         .scale = maximum / maximum_code,
                         .bits = bits,
                         .maximum_code = maximum_code};
  if (maximum == 0)
    return result;
  if (!std::isfinite(result.scale) || result.scale <= 0)
    return absl::OutOfRangeError(
        "quantization scale is not finite and positive");
  for (size_t index = 0; index < tensor.size(); ++index) {
    const double scaled = static_cast<double>(tensor[index]) / result.scale;
    if (!std::isfinite(scaled))
      return absl::OutOfRangeError("quantization code overflowed");
    // Clamping only protects endpoint roundoff: the exact ratio is already
    // between +/-maximum_code by construction. std::round specifies ties
    // independently of the current floating-point rounding direction.
    const double code =
        std::clamp(std::round(scaled), -static_cast<double>(maximum_code),
                   static_cast<double>(maximum_code));
    // The maximum input must remain finite even if code*scale rounds a tiny
    // amount above FP32 max. Exact real arithmetic never exceeds absmax.
    const double restored = std::clamp(code * result.scale, -maximum, maximum);
    if (!std::isfinite(restored) ||
        std::abs(restored) > std::numeric_limits<float>::max())
      return absl::OutOfRangeError(
          "dequantized value exceeds finite FP32 range");
    const float stored = static_cast<float>(restored);
    if (!std::isfinite(stored))
      return absl::OutOfRangeError("dequantized value overflowed FP32");
    result.values[index] = stored == 0 ? 0.0f : stored;
  }
  return result;
}

}  // namespace pluto::llm::one_shot_memorizer
