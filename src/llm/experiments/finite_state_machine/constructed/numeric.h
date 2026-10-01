#pragma once

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "absl/types/span.h"

namespace pluto::llm::fsm::constructed {

// The constructed network uses FP64 throughout. Sparse storage changes only
// how its fixed affine weights and activations are stored, not their math.
using Vector = std::vector<double>;

// One nonzero coefficient in a matrix whose rows are output coordinates.
struct Coefficient {
  int row;
  int column;
  double value;
};

// A named, inspectable affine map y = Wx + b. Shapes and finite coefficients
// are trusted construction-time invariants, checked with assertions.
class Affine {
 public:
  Affine(std::string name, int input_size, int output_size);

  // Adds a coefficient, omitting zeros. Repeated (row, column) entries add to
  // each other during Apply; they are retained separately for inspection.
  void Add(int row, int column, double value);

  // Sets, rather than increments, the bias for one output coordinate.
  void Bias(int row, double value);
  Vector Apply(absl::Span<const double> input) const;

  const std::string& name() const { return name_; }
  int input_size() const { return input_size_; }
  int output_size() const { return output_size_; }
  absl::Span<const Coefficient> coefficients() const { return coefficients_; }
  absl::Span<const double> bias() const { return bias_; }

 private:
  std::string name_;
  int input_size_;
  int output_size_;
  std::vector<Coefficient> coefficients_;
  Vector bias_;
};

// Ordinary coordinate-wise ReLU, with no thresholding or symbol decoding.
Vector Relu(Vector input);
Vector AddVectors(absl::Span<const double> left,
                  absl::Span<const double> right);

// Coordinates are strictly increasing, unique, and nonzero. These are numeric
// vectors, not token IDs or symbolic state-transition records.
using SparseVector = std::vector<std::pair<int, double>>;
SparseVector Sparsify(absl::Span<const double> input);

// One previously projected key/value pair in an attention head's memory.
struct MemoryEntry {
  SparseVector key;
  SparseVector value;
};

// The full softmax-weighted value sum and diagnostics about its largest weight.
// The winning index is diagnostic only: it never selects the returned value.
struct AttentionResult {
  Vector output;
  size_t winning_index;
  double winning_probability;
};

// Computes softmax(query * keys^T) * values using a stable FP64 softmax. The
// caller supplies nonempty, already-causal memory and incorporates any desired
// attention scale into its query/key projections. All values participate;
// there is no hard argmax, token lookup, or special-case transition handling.
AttentionResult Attend(absl::Span<const double> query,
                       absl::Span<const MemoryEntry> memory, int value_size);

}  // namespace pluto::llm::fsm::constructed
