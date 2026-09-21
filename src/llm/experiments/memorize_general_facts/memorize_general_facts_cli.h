#pragma once

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"

namespace pluto::llm::memorize_general_facts {

enum class Mode { kTrainModel, kInferModel };

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
