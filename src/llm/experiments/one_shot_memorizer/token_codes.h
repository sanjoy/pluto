#pragma once

#include <cstdint>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"

namespace pluto::llm::one_shot_memorizer {

// Output-only token codes for label-supervised construction experiments. A
// fixed codebook is not a learned embedding and must not replace a retained,
// tied input embedding. Any target amplitude is chosen separately by callers.
struct TokenCodes {
  int width = 0;
  int vocab_size = 0;
  std::vector<float> values;  // Row-major [vocab_size, width].
};

// Selects distinct balanced sign vectors in a reproducible, shuffled order.
// Every row has width/2 entries of each sign (+1/-1): coordinate mean zero,
// squared Euclidean norm width, and exact FP32/BF16 representation. The width
// must be even and between 2 and 20; vocab_size must be positive and at most
// C(width,width/2). Uses SplitMix64 and unbiased bounded draws for
// Fisher-Yates, without implementation-dependent standard-library
// distributions/shuffles.
absl::StatusOr<TokenCodes> MakeBalancedTokenCodes(int vocab_size,
                                                  int width = 16,
                                                  uint64_t seed = 0);

// Centers each token row across coordinates, then divides by its coordinate
// RMS. Computation uses double precision and stores float results, so the
// resulting zero means and squared norms of width hold up to float rounding.
// Rejects empty/incomplete matrices, nonpositive width, nonfinite entries, and
// constant rows. This operation does not center across vocabulary entries or
// guarantee distinct rows. Learned input rows remain learned information.
absl::StatusOr<TokenCodes> NormalizeTokenCodes(absl::Span<const float> input,
                                               int width);

// Permutes whole rows without changing their values or their geometry. This
// breaks the original token-to-code assignment while preserving the codebook.
// Validates positive dimensions, matching storage, and finite values; does not
// require balanced, normalized, or distinct input rows. The source is
// unchanged.
absl::StatusOr<TokenCodes> PermuteTokenCodes(const TokenCodes& codes,
                                             uint64_t seed = 0);

}  // namespace pluto::llm::one_shot_memorizer
