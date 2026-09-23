#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "src/cuda/executor.h"
#include "src/llm/layer.h"

namespace pluto::llm::one_shot_memorizer {

// Exact enclosing combinator names, outermost first, followed by an output
// hook's name. Occurrence counts callbacks with this same complete identity,
// not all hooks with a matching leaf name. -1 requires exactly one occurrence;
// captures always populate an explicit zero-based occurrence.
struct TokenTraceSite {
  std::vector<std::string> scope;
  std::string layer_name;
  int occurrence = -1;
  size_t output_index = 0;
};

// Owned CPU snapshot of contiguous rows. Raw physical bytes are authoritative
// for donor/identity patches; values are their decoded FP32 presentation.
// Normally rows [0,prefix_length) are retained. The LM head and outer model
// retain only the final query row, never logits for future positions.
struct TokenTraceActivation {
  TokenTraceSite site;
  DataType data_type = DataType::FP32;
  size_t first_row = 0;
  size_t row_count = 0;
  size_t channels = 0;
  std::vector<uint8_t> bytes;
  std::vector<float> values;
};

// Read-only attention weights, [head,query,key], restricted to prefix rows and
// columns. The full prefix square is retained; its upper triangle is zero.
struct TokenTraceAttention {
  TokenTraceSite site;
  size_t heads = 0;
  size_t prefix_length = 0;
  std::vector<float> probabilities;
};

enum class TokenTraceRows { kQuery, kOne, kAllPrefix };
enum class TokenTraceReplacement { kZero, kDonor };

struct TokenTracePatch {
  TokenTraceSite site;
  TokenTraceRows rows = TokenTraceRows::kQuery;
  size_t row = 0;  // Used only by kOne; absolute zero-based prefix position.
  size_t first_channel = 0;
  // Zero means all remaining channels. Otherwise this is an exact count.
  size_t channel_count = 0;
  TokenTraceReplacement replacement = TokenTraceReplacement::kZero;
  // Required only for kDonor. Borrowed for the call, never modified. Dtype and
  // channel width must match; selected absolute rows must exist in the donor.
  const TokenTraceActivation* donor = nullptr;
};

struct TokenTraceOptions {
  int vocabulary_size = 0;  // Real token count, excluding padded logit columns.
  int padding_token = 0;  // Fixed valid ID for every position after the prefix.
  bool capture_activations = false;
  bool capture_attention = false;
  // Independent callers make separate calls for separate interventions.
  // Multiple entries in this span intentionally compose within ONE forward.
  absl::Span<const TokenTracePatch> patches;
};

struct TokenTraceResult {
  size_t query_row = 0;     // prefix.size()-1; predicts the first unseen token.
  int predicted_token = 0;  // Stable top-1 over real vocabulary only.
  std::vector<float> logits;  // FP32 query row, excluding padded columns.
  std::vector<TokenTraceActivation> activations;
  std::vector<TokenTraceAttention> attention;
};

// Exactly one next-token forward for a supplied prefix; no target or suffix is
// accepted. The model must take INT32 [batch,context] and return FP32
// [batch,context,padded_vocabulary]. Runs batch one, pads all other positions,
// and returns only the final prefix row's real-vocabulary logits. Captures are
// optional and describe post-intervention outputs. Patches clone the complete
// original buffer before any writes; original producers and donors stay intact.
// All transfers use pinned memory. No backward or training occurs.
absl::StatusOr<TokenTraceResult> TraceNextToken(
    cuda::Executor& executor, const Layer& model, absl::Span<const int> prefix,
    const TokenTraceOptions& options);

}  // namespace pluto::llm::one_shot_memorizer
