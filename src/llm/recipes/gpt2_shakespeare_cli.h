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
};

absl::StatusOr<Gpt2ShakespeareMode> ParseGpt2ShakespeareMode(
    absl::string_view mode);

absl::string_view Gpt2ShakespeareModeName(Gpt2ShakespeareMode mode);

// Rejects explicitly supplied flags that the selected mode does not consume.
// inference_from and sparse_autoencoder_from are passed separately because
// their modes require non-empty values, not merely explicit flag presence.
absl::Status ValidateGpt2ShakespeareModeFlags(
    Gpt2ShakespeareMode mode,
    absl::Span<const absl::string_view> explicitly_set_flags,
    absl::string_view inference_from,
    absl::string_view sparse_autoencoder_from);

}  // namespace pluto::llm
