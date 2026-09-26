#pragma once

#include <cstdint>

#include "absl/status/statusor.h"
#include "src/cuda/buffer.h"
#include "src/cuda/executor.h"

namespace pluto::llm::fit_attention_readout {

// Fixed, width-preserving features applied independently to each token vector.
// No method uses corpus statistics, labels, positions, or neighboring tokens.
enum class FixedPreprocessingKind {
  kIdentity,
  kDct,
  kRealDft,
  kSin,
  kCos,
  kSignedSqrt,
  kRandomFourier,
};

struct FixedPreprocessingOptions {
  FixedPreprocessingKind kind = FixedPreprocessingKind::kIdentity;
  // Multiplies the argument of sin/cos, including random Fourier projections.
  // Other transforms ignore it; it must always be finite and positive.
  float scale = 1.0f;
  // Reproducible fixed projection seed; only random Fourier uses this field.
  uint32_t seed = 0;
};

// Transforms a contiguous, nonempty BF16 matrix without modifying its input.
// Widths 1..128 are supported; real DFT and random Fourier require even width.
// Identity returns a reference-counted alias; other kinds return new storage.
// Work and temporary lifetimes are ordered on executor, without synchronizing.
//
// DCT is orthonormal DCT-II. Real DFT returns DC, cos(1), -sin(1), ..., Nyquist
// coordinates, with orthonormal scaling; both are invertible before BF16
// rounding. Signed square root is sign(x)*sqrt(abs(x)). Sin/cos and Fourier
// features are noninvertible and can discard information. Random Fourier uses
// width/2 fixed Gaussian projections with variance 1/width, returning first
// their cosines, then their sines. It preserves width, not necessarily norm.
absl::StatusOr<cuda::Buffer> PreprocessHiddenStates(
    cuda::Executor& executor, const cuda::Buffer& input, int width,
    const FixedPreprocessingOptions& options);

}  // namespace pluto::llm::fit_attention_readout
