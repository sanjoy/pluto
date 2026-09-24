#pragma once

#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"

namespace pluto::llm::memorize_general_facts {

enum class Mode { kTrainModel, kInferModel };

// Snapshot of the flag values needed for command-line validation. The caller
// supplies the actual values, including CLI defaults; these initializers are
// only safe empty values, not a second set of flag defaults. Owning the strings
// keeps snapshots returned by absl::GetFlag alive throughout validation.
struct CommandLineOptions {
  std::string mode;
  std::string tokenizer;
  std::string checkpoint_dir;
  std::string infer_checkpoint;
  std::string verify_checkpoint;
  std::string prompt;
  std::string corpus;
  std::string token_corpus;
  std::string output_dir;
  bool compact_vocabulary = false;
  int generation_tokens = 0;
  int context_length = 0;
  int batch_size = 0;
  int steps = 0;
  int eval_every = 0;
  int checkpoint_every = 0;
  int warmup_steps = 0;
  double learning_rate = 0;
  double training_seconds = 0;
};

// Validates the mode, explicit flag usage, and the selected path's values.
// Unused values are ignored, but explicitly supplying an unused flag is an
// error even if it equals its default. This performs no I/O or CUDA work, so
// call it before creating an executor or opening any input/output files.
absl::StatusOr<Mode> ParseAndValidateRunMode(
    const CommandLineOptions& options,
    absl::Span<const absl::string_view> explicitly_set_flags);

// Accepts exactly train_model or infer_model; an omitted/empty mode is invalid.
absl::StatusOr<Mode> ParseMode(absl::string_view mode);

absl::string_view ModeName(Mode mode);

// Rejects explicitly supplied flags that the selected execution path does not
// consume, including flags supplied with empty or default values. Names have
// no leading "--". The caller must include every explicitly supplied local
// flag; an unknown name is an internal error indicating a missing policy rule.
//
// Both modes require a tokenizer, and training requires a checkpoint directory.
// Inference requires exactly one nonempty checkpoint selector: infer_checkpoint
// enables prompt generation, while verify_checkpoint enables a corpus audit.
// Supplying both selectors explicitly is invalid even when one value is empty.
// This helper performs no I/O, CUDA initialization, or numeric validation.
absl::Status ValidateModeFlags(
    Mode mode, absl::Span<const absl::string_view> explicitly_set_flags,
    absl::string_view tokenizer, absl::string_view checkpoint_dir,
    absl::string_view infer_checkpoint, absl::string_view verify_checkpoint);

}  // namespace pluto::llm::memorize_general_facts
