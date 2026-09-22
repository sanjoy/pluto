#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "src/cuda/executor.h"
#include "src/llm/layer.h"

namespace pluto::llm::memorize_general_facts::discretized_model {

inline constexpr int kCaptureWidth = 16;
inline constexpr int kCaptureContext = 1024;
using CapturedRow = std::array<uint16_t, kCaptureWidth>;

struct CaptureOptions {
  int layers = 8;
  int vocab_size = 0;
  int eos_token = -1;
  int prompt_tokens = 5;
};

struct CapturedSample {
  // Only actual text positions, never EOS padding. Predictions include prompt
  // rows; the final real row predicts EOS in an exactly memorized sample.
  std::vector<int> tokens;
  std::vector<int> predictions;
  // Exact native BF16 bits, indexed [stage][real position][channel]. Stage 0
  // is token + position embedding; stages 2*l+1 and 2*l+2 are the residual
  // sums after attention and MLP, respectively, in transformer block l.
  std::vector<std::vector<CapturedRow>> boundaries;
};

// Runs a single batch-one forward pass with a fixed 1024-position context.
// Observes named/nested GPT-2 boundaries, validates their native BF16 storage,
// and transfers only real rows using executor-owned pinned staging. Prefixes
// shorter than prompt_tokens are permitted here for independent diagnostics.
absl::StatusOr<CapturedSample> CaptureSample(cuda::Executor& executor,
                                             const Layer& model,
                                             absl::Span<const int> tokens,
                                             const CaptureOptions& options);

// Requires every continuation target, including final EOS, to be exact.
absl::Status ValidateCapturedPredictions(const CapturedSample& sample,
                                         const CaptureOptions& options);

// Autonomously completes the prompt using each newly captured last-row top-1.
// Requires an explicit EOS prediction, the exact expected continuation, and
// bitwise equality of all preceding boundary rows with the full-sample pass.
absl::Status VerifyGreedyCapture(cuda::Executor& executor, const Layer& model,
                                 const CapturedSample& expected,
                                 const CaptureOptions& options);

}  // namespace pluto::llm::memorize_general_facts::discretized_model
