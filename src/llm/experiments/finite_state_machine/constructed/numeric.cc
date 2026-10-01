#include "src/llm/experiments/finite_state_machine/constructed/numeric.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <utility>

namespace pluto::llm::fsm::constructed {
namespace {

// Sparse vectors passed into attention are produced by fixed projections. This
// catches malformed construction code without introducing a different math
// path for any token or FSM state.
void CheckSparse(const SparseVector& vector, size_t dimension) {
  [[maybe_unused]] int previous = -1;
  for (const auto& [index, value] : vector) {
    assert(index > previous);
    assert(static_cast<size_t>(index) < dimension);
    assert(value != 0 && std::isfinite(value));
    previous = index;
  }
}

// Both inputs are sorted, so coordinates missing from either vector contribute
// exactly zero without requiring a dense dot product.
double Dot(const SparseVector& left, const SparseVector& right) {
  size_t left_index = 0;
  size_t right_index = 0;
  double result = 0;
  while (left_index < left.size() && right_index < right.size()) {
    if (left[left_index].first < right[right_index].first) {
      ++left_index;
    } else if (right[right_index].first < left[left_index].first) {
      ++right_index;
    } else {
      result += left[left_index].second * right[right_index].second;
      ++left_index;
      ++right_index;
    }
  }
  assert(std::isfinite(result));
  return result;
}

}  // namespace

Affine::Affine(std::string name, int input_size, int output_size)
    : name_(std::move(name)),
      input_size_(input_size),
      output_size_(output_size) {
  assert(input_size >= 0);
  assert(output_size >= 0);
  bias_.resize(output_size);
}

void Affine::Add(int row, int column, double value) {
  assert(row >= 0 && row < output_size_);
  assert(column >= 0 && column < input_size_);
  assert(std::isfinite(value));
  if (value != 0)
    coefficients_.push_back({row, column, value});
}

void Affine::Bias(int row, double value) {
  assert(row >= 0 && row < output_size_);
  assert(std::isfinite(value));
  bias_[row] = value;
}

Vector Affine::Apply(absl::Span<const double> input) const {
  assert(input.size() == static_cast<size_t>(input_size_));
  Vector output = bias_;
  for (const Coefficient& coefficient : coefficients_)
    output[coefficient.row] += coefficient.value * input[coefficient.column];
  return output;
}

Vector Relu(Vector input) {
  for (double& value : input)
    value = std::max(0.0, value);
  return input;
}

Vector AddVectors(absl::Span<const double> left,
                  absl::Span<const double> right) {
  assert(left.size() == right.size());
  Vector output(left.begin(), left.end());
  for (size_t index = 0; index < output.size(); ++index)
    output[index] += right[index];
  return output;
}

SparseVector Sparsify(absl::Span<const double> input) {
  assert(input.size() <= static_cast<size_t>(std::numeric_limits<int>::max()));
  SparseVector output;
  for (size_t index = 0; index < input.size(); ++index) {
    assert(std::isfinite(input[index]));
    if (input[index] != 0)
      output.emplace_back(static_cast<int>(index), input[index]);
  }
  return output;
}

AttentionResult Attend(absl::Span<const double> query,
                       absl::Span<const MemoryEntry> memory, int value_size) {
  assert(!memory.empty());
  assert(value_size >= 0);
  const SparseVector sparse_query = Sparsify(query);
  Vector scores(memory.size());
  size_t winning_index = 0;
  double largest_score = -std::numeric_limits<double>::infinity();
  for (size_t index = 0; index < memory.size(); ++index) {
    CheckSparse(memory[index].key, query.size());
    CheckSparse(memory[index].value, value_size);
    scores[index] = Dot(sparse_query, memory[index].key);
    if (scores[index] > largest_score) {
      largest_score = scores[index];
      winning_index = index;
    }
  }

  // Subtracting the maximum leaves the probabilities unchanged and ensures
  // every exponential lies in [0, 1], even for very large positional scores.
  double denominator = 0;
  for (double& score : scores) {
    score = std::exp(score - largest_score);
    denominator += score;
  }
  assert(std::isfinite(denominator) && denominator > 0);
  Vector output(value_size);
  for (size_t index = 0; index < memory.size(); ++index) {
    const double probability = scores[index] / denominator;
    for (const auto& [coordinate, value] : memory[index].value)
      output[coordinate] += probability * value;
  }
  return {std::move(output), winning_index, 1.0 / denominator};
}

}  // namespace pluto::llm::fsm::constructed
