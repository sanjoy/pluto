#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"

namespace pluto::llm::one_shot_memorizer {

// Explicit logical dimensions, not values inferred from checkpoint byte sizes.
// The tokenizer mapping and these dimensions must be identical in both runs.
struct Gpt2ParameterDimensions {
  int vocabulary_size = 0;
  int model_width = 0;
  int feed_forward_width = 0;
  int context_length = 0;
  int transformer_block_count = 0;
};

// One UNIQUE FP32 checkpoint tensor. The tied language-modeling head aliases
// the token embedding and therefore does not get another checkpoint index.
struct TensorSpec {
  size_t checkpoint_index = 0;  // Index in weight_<index>.bin.
  std::string name;             // Model path and parameter role.
  std::vector<size_t> shape;    // A vector, or a row-major [input, output] map.
  size_t flat_offset = 0;  // First scalar in the concatenated unique tensors.
  size_t element_count = 0;
  bool token_embedding = false;  // Matrix rows identify compact vocabulary IDs.
  // Nonzero only for combined Q/K/V weights or biases. Their final axis is
  // [query channels, key channels, value channels], each this many elements.
  size_t qkv_width = 0;
};

// Matches CreateGpt2's unique weight order, including every bias and both norms
// per block. No model weights, tokenizer, files, or GPU resources are read.
absl::StatusOr<std::vector<TensorSpec>> BuildGpt2ParameterLayout(
    const Gpt2ParameterDimensions& dimensions);

enum class QkvComponent { kNone, kQuery, kKey, kValue };
const char* QkvComponentName(QkvComponent component);

// Both physical checkpoint indexing and optional semantic indexing of a scalar.
struct ParameterCoordinate {
  size_t checkpoint_index = 0;
  size_t element_index = 0;  // Scalar index within this tensor.
  size_t flat_index = 0;     // Scalar index across all unique tensors.
  size_t row = 0;            // Vector index for rank one; matrix row otherwise.
  std::optional<size_t> column;  // Present only for a matrix.
  std::optional<int>
      compact_token_id;  // Present only for token-embedding rows.
  QkvComponent qkv_component = QkvComponent::kNone;
  std::optional<size_t> qkv_channel;  // Channel WITHIN Q, K or V, not combined.
};

absl::StatusOr<ParameterCoordinate> LocateParameter(const TensorSpec& tensor,
                                                    size_t element_index);

// Validates layout order/offsets and the exact tensor cardinality and lengths.
// All supplied FP32 values must be finite; signed zero is accepted.
absl::Status ValidateParameterValues(
    absl::Span<const TensorSpec> layout,
    absl::Span<const absl::Span<const float>> tensors);

struct DeltaSummary {
  size_t element_count = 0;
  size_t bitwise_changed_count = 0;  // Includes +0 versus -0.
  size_t numerically_changed_count = 0;
  double baseline_l2 = 0;
  double ablated_l2 = 0;
  double delta_l1 = 0;
  double delta_l2 = 0;
  double maximum_absolute_delta = 0;
  // delta_l2 / baseline_l2; absent when baseline_l2 is zero, even for 0/0.
  std::optional<double> relative_l2;
};

struct CoordinateDelta {
  ParameterCoordinate coordinate;
  float baseline = 0;
  float ablated = 0;
  double delta = 0;  // Baseline MINUS ablated, evaluated in double precision.
};

struct TensorDelta {
  TensorSpec tensor;
  DeltaSummary summary;
  // Every scalar is retained in physical row-major order, not only top entries.
  std::vector<double> deltas;
  std::vector<bool> bitwise_changed;
  // Only bitwise-changed scalars, descending abs(delta), then ascending index.
  // A signed-zero change can appear here with delta zero.
  std::vector<CoordinateDelta> top_coordinates;
};

struct ParameterDeltaReport {
  DeltaSummary total;
  std::vector<TensorDelta> tensors;
};

// Compares aligned checkpoint tensors; does not attribute exclusive ownership
// of any fact. Callers must ensure matched initialization, optimizer step and
// training schedule. Arrays in the result own their data. top_count applies
// independently to each tensor; zero suppresses only the top-coordinate lists.
absl::StatusOr<ParameterDeltaReport> CompareParameterValues(
    absl::Span<const TensorSpec> layout,
    absl::Span<const absl::Span<const float>> baseline,
    absl::Span<const absl::Span<const float>> ablated, size_t top_count = 20);

}  // namespace pluto::llm::one_shot_memorizer
