#pragma once

#include <cstddef>
#include <memory>
#include <ostream>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

namespace pluto::llm::fsm::constructed {

// Diagnostics for one autoregressive output, including the internal stop token.
// All positions index the space-free token stream, starting at zero.
struct StepTrace {
  int position;               // Query position in the growing context.
  int input_position;         // Position selected by the input-reading head.
  int transition_position;    // Destination row selected by the lookup head.
  double lookup_probability;  // Actual softmax mass on that row.
  int output_token;           // 0..999: state, 1028: ERR, 1029: internal END.
};

// A free-running execution, not a teacher-forced evaluation.
struct Generation {
  std::vector<int> tokens;       // Visited states, possibly followed by ERR.
  std::vector<StepTrace> steps;  // Includes the final END prediction.
};

// A fixed-weight, CPU-only causal neural FSM interpreter. It uses sparse affine
// maps, real softmax attention, and ReLU cleanup. No training data or
// transition table enters the constructor; the machine is supplied only in the
// prompt. This is deliberately not the GPT-2 architecture: it has no LayerNorm,
// uses ReLU rather than GELU, and has an untied output projection and FP64
// arithmetic.
class Model {
 public:
  // The positional code supports at most 1024 tokens, including generated ones.
  static absl::StatusOr<std::unique_ptr<Model>> Create(int max_context = 1024);
  ~Model();

  // Validates/tokenizes a prompt ending in '>', then feeds back predictions
  // until the network emits END. Validation never supplies transitions to
  // inference.
  absl::StatusOr<Generation> Generate(absl::string_view prompt) const;

  // Writes every nonzero coefficient and bias; omitted entries are exactly
  // zero. Matrix shapes and names make the constructed weights independently
  // auditable.
  absl::Status WriteWeights(std::ostream& output) const;
  size_t nonzero_weight_count() const;
  int max_context() const;

 private:
  struct Impl;
  explicit Model(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

// Canonical human-readable rendering of the model's output state/ERR tokens.
std::string Render(const Generation& generation);

}  // namespace pluto::llm::fsm::constructed
