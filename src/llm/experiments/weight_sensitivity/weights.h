#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "src/llm/gpt2.h"
#include "src/llm/layer.h"

namespace pluto::llm::weight_sensitivity {

// One independently corrupted logical weight. Most targets occupy a whole
// checkpoint tensor; Q, K, and V occupy separate column slices of packed QKV.
struct WeightTarget {
  std::string name;            // Human-readable architectural role.
  int checkpoint_index = 0;    // Index among unique checkpoint tensors.
  int block = -1;              // Zero-based transformer, or -1 outside it.
  std::vector<int64_t> shape;  // Logical shape, excluding packed neighbors.
  size_t tensor_elements = 0;  // Full physical tensor's FP32 element count.
  size_t offset = 0;           // First target element in the physical tensor.
  size_t rows = 0;             // Number of independently strided rows.
  size_t columns = 0;          // Contiguous target elements within each row.
  size_t row_stride = 0;       // Physical elements between consecutive rows.
};

// Uses the GPT-2 recipe's checkpoint order. The tied embedding/LM head is one
// target. Every stored parameter, including optional embedding padding, belongs
// to exactly one target. Each Q/K/V matrix and bias is a separate target.
absl::StatusOr<std::vector<WeightTarget>> DescribeWeights(
    const Gpt2Config& config);

// Rejects the wrong checkpoint tensor count, size, or repeated allocation.
// Caller supplies unique weights in first-occurrence model traversal order.
absl::Status ValidateWeights(const Gpt2Config& config,
                             absl::Span<const Buffer> unique_weights);

// Copies the original full tensor, replacing only this target with independent
// zero-mean Gaussian noise. The standard deviation is noise_scale times the
// target's original RMS (or zero_rms_stddev when its RMS is zero); returned
// value is the standard deviation used. This makes initially zero biases
// meaningful interventions instead of accidentally leaving them unchanged.
//
// Both spans must have exactly tensor_elements elements. Their storage must be
// disjoint or identical. The source, scale, and fallback must be finite; scale
// and fallback must be positive. The same seed and inputs reproduce the same
// noise in the same build, independently of other targets or evaluation order.
// No CUDA memory is modified by this host-only operation.
absl::StatusOr<double> ReplaceWithNoise(const WeightTarget& target,
                                        absl::Span<const float> original,
                                        absl::Span<float> destination,
                                        uint64_t seed, double noise_scale = 1.0,
                                        double zero_rms_stddev = 0.02);

}  // namespace pluto::llm::weight_sensitivity
