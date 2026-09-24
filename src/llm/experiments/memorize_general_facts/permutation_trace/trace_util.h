#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"

namespace pluto::llm::permutation_trace {

// Tensor serialization uses native little-endian scalar bytes. Shapes are
// concrete and contiguous; vocabulary padding is preserved in the dump.
struct TensorDescription {
  std::string stage;
  std::string dtype;  // bf16, fp32, or int32.
  std::vector<int64_t> shape;
  // none, token_ids, vocab_rows, or vocab_columns (old -> new permutation).
  std::string alignment = "none";
};

struct Difference {
  size_t elements = 0;
  size_t mismatches = 0;  // Bitwise unequal scalars after vocabulary alignment.
  double max_abs = 0;
  double l2 = 0;
};

// An empty permutation requests a literal comparison, used for the duplicate
// baseline. A supplied permutation aligns renamed values to original IDs.
absl::StatusOr<Difference> Compare(const TensorDescription& description,
                                   absl::Span<const uint8_t> baseline,
                                   absl::Span<const uint8_t> candidate,
                                   absl::Span<const int> permutation);

// Parses and validates a complete bijection, including the fixed EOS entry.
absl::StatusOr<std::vector<int>> ParsePermutation(absl::string_view text,
                                                  int vocabulary_size, int eos);

std::string ShapeText(absl::Span<const int64_t> shape);

}  // namespace pluto::llm::permutation_trace
