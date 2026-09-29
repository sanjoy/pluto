#pragma once

#include <string>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"

namespace pluto::llm::qwen {

enum class Mode { kInferModel, kTrainModel, kEmbeddingAlgebra };

// An owning snapshot of the flags used for mode selection and validation.
// The caller supplies flag defaults; these initializers are only empty values.
struct CommandLineOptions {
  std::string mode;
  std::string checkpoint;
  std::string prompt;
  int max_new_tokens = 0;
  int context_length = 0;
  bool raw_prompt = false;
  bool thinking = false;
  std::string text;
  int sequence_length = 0;
  int batch_size = 0;
  int steps = 0;
  int switch_every = 0;
  int start_block = 0;
  double learning_rate = 0;
  double max_active_gib = 0;
  std::string resume_weights;
  std::string save_weights;
  std::string expression;
};

// Requires an exact infer_model/train_model/embedding_algebra mode and a
// nonempty checkpoint. Embedding algebra is interactive when no expression is
// supplied; explicitly supplying an empty expression is an error.
// Validates only the selected mode's values, but rejects every explicitly set
// flag belonging to another mode, even when its value is empty or default.
// Names in explicitly_set_flags have no leading "--"; an unknown name is an
// internal error indicating a missing policy rule. Performs no I/O or CUDA
// work, so call this before opening checkpoints or allocating GPU resources.
absl::StatusOr<Mode> ParseAndValidateRunMode(
    const CommandLineOptions& options,
    absl::Span<const absl::string_view> explicitly_set_flags);

}  // namespace pluto::llm::qwen
