#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "src/cuda/executor.h"
#include "src/dataset/padded_line_dataset.h"
#include "src/llm/experiments/one_shot_memorizer/token_trace.h"
#include "src/llm/layer.h"

namespace pluto::llm::one_shot_memorizer {

struct MlpSamples {
  int width = 0;
  int feature_width = 0;
  // Corpus/sample/position order; include every real text token, including
  // prompt tokens, and exclude right padding. Values expand physical BF16.
  std::vector<float> normalized_inputs;  // [real_token_rows, width]
  std::vector<float> features;           // [real_token_rows, feature_width]
  std::vector<float> outputs;  // [real_token_rows, width], before skip add
  // Row-major W2 [feature_width, width], rounded to its effective BF16 values.
  // Bias remains FP32: the dense kernel adds it without BF16 operand rounding.
  std::vector<float> effective_output_weights;
  std::vector<float> output_bias;
};

struct MlpCorpusCapture {
  std::vector<MlpSamples> blocks;
  int64_t target_count = 0;
  int64_t correct_targets = 0;
  int64_t exact_sentences = 0;
  int64_t sentences = 0;
};

// Resets an unshuffled padded-line iterator and performs one unmodified pass.
// Captures the MLP's own LayerNorm output, GELU output, and branch output for
// each transformer_block_N. The provided widths/block count must match the
// BF16 GPT-2 model. Only continuation/EOS targets enter accuracy; capture also
// includes the prompt rows needed when replacing a branch at all positions.
// The iterator's batches are consumed before its reused buffers advance.
absl::StatusOr<MlpCorpusCapture> CaptureGpt2Mlps(
    cuda::Executor& executor, const Layer& model,
    PaddedLineDataSetIterator& dataset, int width, int feature_width,
    int block_count, int vocabulary_size);

enum class MlpSource { kLayerNorm, kGelu };

struct MlpReplacement {
  int block = -1;
  MlpSource source = MlpSource::kLayerNorm;
  // Borrowed, immutable projection; caller retains ownership. Requires BF16
  // [batch, context, source_width] -> BF16 [batch, context, model_width].
  const Layer* projection = nullptr;
};

struct MlpEvaluation {
  int64_t targets = 0;
  int64_t correct_targets = 0;
  int64_t sentences = 0;
  int64_t exact_sentences = 0;
  // Original unshuffled corpus order, for reporting externally selected
  // fitting/held-out sentence groups from the same fresh evaluation pass.
  std::vector<int64_t> targets_per_sentence;
  std::vector<int64_t> correct_per_sentence;
};

// Resets the unshuffled iterator and evaluates a fresh forward with the listed
// branches replaced, simultaneously when several blocks are supplied. Empty
// replacements give an unmodified baseline. Each replacement uses this SAME
// forward's fresh LayerNorm/GELU activation, including changes from earlier
// replaced blocks; it never replays stored corpus activations.
//
// At the named 'mlp' output hook, run projection.fwd(..., nullptr) and replace
// the complete branch output handle, preserving the recipient residual skip.
// Every padded-context position is projected. Original branches still execute
// before their outputs are replaced, so this measures behavior, not speed.
// No original parameter or activation bytes are modified; no backward runs.
// Unknown/duplicate blocks, source/shape/executor mismatches, or nonfinite
// scored logits fail explicitly. This does not establish out-of-corpus
// fidelity.
absl::StatusOr<MlpEvaluation> EvaluateMlpReplacements(
    cuda::Executor& executor, const Layer& model,
    PaddedLineDataSetIterator& dataset,
    absl::Span<const MlpReplacement> replacements, int vocabulary_size);

// One fresh prefix-only forward with the same live branch substitutions as
// EvaluateMlpReplacements, while retaining TokenTraceResult captures/patches.
// No label or gold continuation is accepted. Source-site patches are applied
// BEFORE retaining the fresh LayerNorm/GELU source; at the MLP branch output,
// substitution runs BEFORE trace patches and capture. Thus a branch-output
// patch deliberately overrides the replacement, and later blocks consume the
// actual intervened states. Replacement internals add no trace scope/events.
//
// Empty replacements delegate exactly to TraceNextToken. Nonempty replacement
// lists reserve options.compose_hooks for this adapter and reject a supplied
// composer. All original/replacement weights and original activation bytes
// remain unchanged; projection ownership stays with the caller.
absl::StatusOr<TokenTraceResult> TraceMlpReplacements(
    cuda::Executor& executor, const Layer& model, absl::Span<const int> prefix,
    absl::Span<const MlpReplacement> replacements,
    const TokenTraceOptions& options);

// The first failed prediction, after all earlier generated tokens matched.
// Recording this scoring metadata does not change prediction selection or
// feed any gold suffix token into the generated input prefix.
struct MlpGreedyMismatch {
  // Absolute zero-based token position: first suffix=prompt_tokens; EOS=text
  // size.
  int target_position;
  // Actual full-vocabulary argmax returned by the model, including premature
  // EOS.
  int predicted_token;
  // Gold token at target_position, or EOS at the original sentence end.
  int expected_token;

  bool operator==(const MlpGreedyMismatch&) const = default;
};

struct MlpGreedyEvaluation {
  int64_t sentences = 0;
  int64_t exact_sentences = 0;
  // Actual greedy top-1 predictions, including terminal EOS or the first
  // mismatch. Cases stop at their first mismatch, so this is not the full
  // teacher-forced target denominator when any case fails early.
  int64_t generated_targets = 0;
  // Original corpus order, independent of when each active case finishes.
  // True only for a complete matching continuation and correctly timed EOS.
  std::vector<bool> exact_per_sentence;
  // Same corpus order as exact_per_sentence. nullopt if and only if the
  // sentence completed exactly, including EOS; failed cases contain one event.
  std::vector<std::optional<MlpGreedyMismatch>> first_mismatch_per_sentence;
};

// Independently verifies exact greedy continuations from each sentence's
// supplied prompt. Only predicted tokens are fed back; gold suffix tokens are
// consulted after argmax solely to accept/reject the prediction. EOS must occur
// exactly at the gold sentence end, including at the context's final row.
// Finished or failed cases are masked; unused batch slots contain EOS. Every
// forward creates fresh MLP hooks, and listed substitutions apply at all rows.
// Reads the unshuffled dataset's original token storage without advancing its
// iterator. Reuses one fixed-capacity input/score-mask/prediction allocation.
// Records only the first mismatch; it never feeds a gold suffix token back to
// continue a failed case. Its absolute target position is one past the query
// row.
absl::StatusOr<MlpGreedyEvaluation> VerifyMlpGreedyCompletions(
    cuda::Executor& executor, const Layer& model,
    PaddedLineDataSetIterator& dataset,
    absl::Span<const MlpReplacement> replacements, int vocabulary_size);

}  // namespace pluto::llm::one_shot_memorizer
