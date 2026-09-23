#pragma once

#include <string>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "src/llm/experiments/one_shot_memorizer/automaton.h"

namespace pluto::llm::one_shot_memorizer {

// Portable little-endian weights, without a copy of the original sentences.
// Stores sparse transition coordinates (all nonzero matrix entries are one),
// terminal weights, suffix masses, and the initial-state coordinate. The
// tokenizer vocabulary must be the same at construction and inference.
absl::StatusOr<std::string> SerializeModel(const Model& model);

// Rejects malformed, truncated, trailing, or inconsistent weights. Dimensions
// are bounded by the supplied byte string before allocating vectors.
absl::StatusOr<Model> DeserializeModel(absl::string_view bytes);

}  // namespace pluto::llm::one_shot_memorizer
