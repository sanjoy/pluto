#pragma once

#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"

namespace pluto::llm::one_shot_memorizer {

// A destructive per-tensor quantization intervention, expanded back to FP32
// for existing GPU layers. Values are NOT a packed bitstream. A successful
// behavior test measures robustness to this intervention, not entropy or an
// exact minimum number of bits needed to encode the model's knowledge.
struct QuantizedTensor {
  std::vector<float> values;  // Dequantized values in original element order.
  // absmax / maximum_code, or zero for an all-zero tensor. Double precision
  // prevents underflow when quantizing even the smallest FP32 subnormals.
  double scale = 0;
  int bits = 0;          // Requested signed-code width, between 2 and 16.
  int maximum_code = 0;  // (1 << (bits - 1)) - 1; one signed code is unused.
};

// Use symmetric integer levels [-maximum_code, maximum_code], with absmax
// measured over this tensor alone. Codes are round(value/scale), with exact
// half-integer ties rounded AWAY from zero. Zero outputs are canonical +0.
// Empty/nonfinite input or unsupported bits fail; input storage is unchanged.
// No GPU resources, calibration examples, learned scales or random seeds are
// used. The caller must independently evaluate the changed model's behavior.
absl::StatusOr<QuantizedTensor> QuantizeTensorSymmetric(
    absl::Span<const float> tensor, int bits);

}  // namespace pluto::llm::one_shot_memorizer
