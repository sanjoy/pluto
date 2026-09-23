#pragma once

#include <cstdint>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "src/cuda/executor.h"
#include "src/llm/layer.h"

namespace pluto::llm::one_shot_memorizer {

enum class PrefixReplacement { kEos, kOtherSentence };

struct ContextProbeResult {
  int window = -1;
  PrefixReplacement replacement = PrefixReplacement::kEos;
  int64_t targets = 0;
  int64_t correct = 0;
  // Scored rows with any nonfinite logical-vocabulary logit. These rows count
  // in targets and cannot count in correct. Unscored rows are never inspected.
  int64_t nonfinite = 0;
};

// Intervenes on independent teacher-forced prefixes of a causal model. Every
// supervised next token, including one terminal EOS per sentence, is evaluated
// at its original absolute position. For window k >= 0, retain the final k
// prefix tokens and replace all earlier tokens. Window -1 retains the complete
// prefix as an unmodified control. Positions after the prefix contain EOS.
//
// kOtherSentence takes replacement tokens from the next sentence in corpus
// order, wrapping around the corpus and cycling that sentence's token sequence
// if necessary. It requires at least two sentences. Different sentence indices
// need not contain different text, so some replacement tokens can be unchanged.
//
// Sentences are unpadded text tokens in the model's vocabulary, without EOS.
// Their lengths must be between prompt_tokens and the model context length.
// The model must expose INT32 [batch, context] -> FP32 [batch, context, stride]
// with stride >= vocabulary_size and belong to executor. One fixed-capacity
// batch allocation is reused; unused sample slots have no scored rows.
//
// Results preserve window order, including repeated windows. This is neither
// autoregressive accuracy nor a proof of a learned circuit: prefix replacement
// can introduce out-of-distribution inputs and interacts with absolute
// position.
absl::StatusOr<std::vector<ContextProbeResult>> ProbeContextWindows(
    cuda::Executor& executor, const Layer& model,
    absl::Span<const std::vector<int>> sentences, int vocabulary_size,
    int eos_token, int prompt_tokens, int batch_size,
    absl::Span<const int> windows,
    PrefixReplacement replacement = PrefixReplacement::kEos);

}  // namespace pluto::llm::one_shot_memorizer
