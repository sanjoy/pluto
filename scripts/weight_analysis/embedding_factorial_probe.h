#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "scripts/weight_analysis/phrase_probe.h"
#include "src/cuda/buffer.h"
#include "src/cuda/executor.h"
#include "src/cuda/page_locked_host_array.h"

namespace pluto::weight_analysis {

// A denotes a recipient model; J differs from A only in token-embedding rows.
// The first letter selects the input embedding (and hence the resulting final
// residual), and the second selects the output dictionary. The two mixed cells
// are deliberately untied diagnostic models, not intermediate logit lenses.
inline constexpr std::array<const char*, 4> kEmbeddingFactorialCells{
    "AA", "AJ", "JA", "JJ"};

// CPU-only validation. Each case has exactly rows_per_case strictly increasing
// loss-row indices in [0, context_length). Row r predicts targets[r], which is
// input[r+1] when present. Three consecutive rows score a three-token spelling.
absl::Status ValidateFactorialRows(absl::Span<const int32_t> rows,
                                   int case_count, int rows_per_case,
                                   int context_length);
absl::StatusOr<std::vector<int32_t>> LoadFactorialRows(
    const std::filesystem::path& path, int case_count, int rows_per_case,
    int context_length);

struct FactorialTokenScore {
  double nll;
  int argmax;
  int target_rank;
};

// CPU FP64 log-sum-exp of ALL supplied logical-vocabulary native FP32 logits.
// This is temperature-1 teacher-forced NLL, not the native CE kernel's
// reduction order, sampling frequency, or probability over alternative
// tokenizations. Ties in argmax/rank prefer lower token IDs. Nonfinite logits
// are rejected.
absl::StatusOr<FactorialTokenScore> ScoreFactorialToken(
    absl::Span<const float> logits, int target);

// Return exactly the changed logical rows, comparing FP32 BYTES (including
// signed zero), and reject modifications to physical padding. Empty changes
// are allowed for an identity control. This helper never initializes CUDA.
absl::StatusOr<std::vector<int>> ChangedEmbeddingRows(
    absl::Span<const float> original, absl::Span<const float> patched,
    int vocabulary, int width);

struct FactorialMeasurements {
  // Each cell is [selected rows, logical vocabulary], in supplied row order.
  // D2H storage is page locked; physical padding is never written or scored.
  std::array<cuda::PageLockedHostArray<float>, 4> logits;
  std::array<std::vector<FactorialTokenScore>, 4> scores;
};

// Apply both native final-LN/head readouts to both native final residuals.
// Caller must establish that all nonembedding weights (including final LN)
// agree and build the two lenses from those weights. The diagonal readouts are
// checked byte-for-byte against the original complete-model selected rows,
// including padded lanes. A failed diagonal check returns no measurements.
// This does not mutate either model or run any backward/optimizer operation.
absl::StatusOr<FactorialMeasurements> EvaluateEmbeddingFactorial(
    cuda::Executor& executor,
    const std::array<const NativeLogitLens*, 2>& lenses,
    const std::array<cuda::Buffer, 2>& residuals,
    const std::array<cuda::Buffer, 2>& native_logits,
    absl::Span<const int32_t> selected_rows, absl::Span<const int32_t> targets);

}  // namespace pluto::weight_analysis
