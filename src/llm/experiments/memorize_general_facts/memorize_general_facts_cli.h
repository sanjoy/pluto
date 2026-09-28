#pragma once

#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"

namespace pluto::llm::memorize_general_facts {

enum class Mode { kTrainModel, kInferModel, kPuzzle };

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
  std::string puzzle_checkpoint;
  std::string prompt;
  std::string corpus;
  std::string output_dir;
  int generation_tokens = 0;
  int layers = 0;
  int model_width = 0;
  int context_length = 0;
  int batch_size = 0;
  int steps = 0;
  int eval_every = 0;
  int checkpoint_every = 0;
  int warmup_steps = 0;
  int seed = 0;
  int mlp_width = 0;
  bool a3_mlp_stack = false;
  bool search = false;
  bool train_mlp = false;
  bool train_stacked_mlp = false;
  bool train_mlp_transformer = false;
  double learning_rate = 0;
  double training_seconds = 0;
};

// The shared optimizer flags retain their historical train/infer defaults.
// Puzzle mode uses its own defaults only when the user omitted those flags.
// Resolve before validation so explicitly supplied invalid values are checked.
CommandLineOptions ResolveModeDefaults(
    CommandLineOptions options,
    absl::Span<const absl::string_view> explicitly_set_flags);

// Validates the mode, explicit flag usage, and the selected path's values.
// Unused values are ignored, but explicitly supplying an unused flag is an
// error even if it equals its default. This performs no I/O or CUDA work, so
// call it before creating an executor or opening any input/output files.
absl::StatusOr<Mode> ParseAndValidateRunMode(
    const CommandLineOptions& options,
    absl::Span<const absl::string_view> explicitly_set_flags);

// Accepts exactly train_model, infer_model, or puzzle; an empty mode is
// invalid.
absl::StatusOr<Mode> ParseMode(absl::string_view mode);

absl::string_view ModeName(Mode mode);

// Rejects explicitly supplied flags that the selected execution path does not
// consume, including flags supplied with empty or default values. Names have
// no leading "--". The caller must include every explicitly supplied local
// flag; an unknown name is an internal error indicating a missing policy rule.
//
// Every mode requires a tokenizer; training requires a checkpoint directory.
// Inference requires exactly one nonempty checkpoint selector: infer_checkpoint
// enables prompt generation, while verify_checkpoint enables a corpus audit.
// Supplying both selectors explicitly is invalid even when one value is empty.
// Puzzle requires puzzle_checkpoint and consumes optimizer settings only when
// train_mlp, train_stacked_mlp, or train_mlp_transformer is enabled. These
// training selectors are mutually exclusive when enabled; mlp_width applies
// only to train_mlp.
// This helper performs no I/O, CUDA initialization, or numeric validation.
absl::Status ValidateModeFlags(
    Mode mode, absl::Span<const absl::string_view> explicitly_set_flags,
    absl::string_view tokenizer, absl::string_view checkpoint_dir,
    absl::string_view infer_checkpoint, absl::string_view verify_checkpoint,
    absl::string_view puzzle_checkpoint = "", bool train_mlp = false,
    bool train_stacked_mlp = false, bool train_mlp_transformer = false);

}  // namespace pluto::llm::memorize_general_facts
