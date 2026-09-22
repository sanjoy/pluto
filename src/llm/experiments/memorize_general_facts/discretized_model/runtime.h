#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"

namespace pluto::llm::discretized {

using TokenId = int32_t;
using StateId = uint32_t;

struct VocabularyRow {
  int32_t original_id;
  absl::string_view bytes;
};
struct EntryRow {
  TokenId token;
  uint32_t position;
  StateId state;
};
struct StateRow {
  StateId input;
  StateId output;
};
struct AttentionRow {
  uint32_t offset;
  uint32_t length;
  StateId output;
};

// A compiled transition must report unsupported inputs explicitly. Zero is a
// valid vocabulary ID, so it cannot double as a failure sentinel. Functions
// implementing these transitions are pure and cannot carry history across
// calls.
struct TransitionResult {
  StateId output = 0;
  bool supported = false;
};

struct AttentionTable {
  absl::Span<const StateId> keys;
  // Sorted lexicographically by the complete sequence in keys.
  absl::Span<const AttentionRow> rows;
  // Alternative to stored rows: generated exact-domain control flow. Never
  // populate both forms. The complete ordered causal prefix is still supplied.
  TransitionResult (*function)(absl::Span<const StateId>) = nullptr;
};
struct StateTable {
  // Sorted by input, with no duplicate inputs.
  absl::Span<const StateRow> rows;
  TransitionResult (*function)(StateId) = nullptr;
};

// A finite integer network. Tables or pure functions implement each boundary;
// no sample identity, corpus text, expected suffix, or floating-point weights
// are available to this object. All spans must outlive its use.
struct Model {
  uint32_t context_length;
  uint32_t prompt_tokens;
  TokenId eos_token;
  absl::Span<const VocabularyRow> vocabulary;
  // Sorted by (token, position).
  absl::Span<const EntryRow> entry;
  absl::Span<const AttentionTable> attention;
  absl::Span<const StateTable> mlp;
  StateTable snap;
  // Alternative to entry rows, with the same token/position domain.
  TransitionResult (*entry_function)(TokenId, uint32_t) = nullptr;
};

// Validate a newly loaded/generated model once before inference. Runtime
// tables are immutable; this checks key order, ranges, and storage bounds.
// Compiled functions are verified against the source transitions by generation
// tests; here we reject ambiguous table/function representations.
absl::Status ValidateModel(const Model& model);

// Executes entry -> (causal attention -> pointwise MLP)* -> final snap.
// Recomputes all real positions. Unknown keys fail explicitly; there is no
// nearest-state fallback and no corpus-line or continuation lookup.
absl::StatusOr<TokenId> PredictNext(const Model& model,
                                    absl::Span<const TokenId> tokens);

// Returns newly generated tokens, INCLUDING EOS when reached. Zero budget
// returns an empty vector after prompt validation. Stops at context_length.
absl::StatusOr<std::vector<TokenId>> Generate(const Model& model,
                                              absl::Span<const TokenId> prompt,
                                              size_t max_new_tokens);
absl::StatusOr<std::string> Decode(const Model& model,
                                   absl::Span<const TokenId> tokens);

// Implemented by the generated production tables; they do not link fixtures.
const Model& GeneratedModel();

}  // namespace pluto::llm::discretized
