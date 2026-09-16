#pragma once

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"

namespace pluto::llm {

enum class Gpt2ShakespeareMode {
  kTrainModel,
  kInferModel,
  kTrainSparseAutoEncoder,
  kInferSparseAutoEncoder,
};

// Optional diagnostic readout while running model inference.
enum class ActivationInspectionMode {
  kDisabled,
  kNeighboringVocab,
};

struct ActivationInspectionOptions {
  ActivationInspectionMode mode = ActivationInspectionMode::kDisabled;
  // Inclusive probability threshold, not a percentage: 0.01 means 1%.
  double min_prob = 0.01;
};

// Empty means disabled; otherwise accepts one neighboring_vocab mode,
// optionally followed by "(min_prob=0.05)". Bare mode names and empty
// parentheses retain defaults. ASCII whitespace around the mode, settings,
// keys, and values is ignored. Probabilities must be finite and in [0, 1];
// unknown or repeated keys, malformed parentheses, and lists of modes are
// errors. Explicit presence is checked separately by
// ValidateGpt2ShakespeareModeFlags, even when the supplied value is empty.
absl::StatusOr<ActivationInspectionOptions> ParseActivationInspectionMode(
    absl::string_view mode);

absl::StatusOr<Gpt2ShakespeareMode> ParseGpt2ShakespeareMode(
    absl::string_view mode);

absl::string_view Gpt2ShakespeareModeName(Gpt2ShakespeareMode mode);

// Validates an explicitly supplied --training_seconds. The omitted flag leaves
// time-based stopping disabled; explicit zero is rejected to catch mistakes.
absl::Status ValidateGpt2ShakespeareTrainingSeconds(double training_seconds);

// Rejects explicitly supplied flags that the selected mode does not consume.
// inference_from and sparse_autoencoder_from are passed separately because
// their modes require non-empty values, not merely explicit flag presence.
absl::Status ValidateGpt2ShakespeareModeFlags(
    Gpt2ShakespeareMode mode,
    absl::Span<const absl::string_view> explicitly_set_flags,
    absl::string_view inference_from,
    absl::string_view sparse_autoencoder_from);

}  // namespace pluto::llm
