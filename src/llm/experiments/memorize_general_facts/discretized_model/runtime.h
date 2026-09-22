#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"

namespace pluto::llm::discretized {

using TokenId = int32_t;
using StateId = uint32_t;

// One vocabulary token, indexed by its compact TokenId in Model::vocabulary.
struct VocabularyRow {
  int32_t original_id;      // Token ID before compact-vocabulary remapping.
  absl::string_view bytes;  // Exact decoded bytes, not necessarily valid UTF-8.
};

// Token/absolute-position lookup implementing the combined embedding boundary.
struct EntryRow {
  TokenId token;      // Input compact vocabulary ID.
  uint32_t position;  // Zero-based position within the causal sequence.
  StateId state;      // Residual-stream symbol after token/position embeddings.
};

// One pointwise transition in an MLP residual boundary or vocabulary snap.
struct StateRow {
  StateId input;   // Residual symbol in this boundary's input alphabet.
  StateId output;  // Output residual symbol, or compact token ID for the snap.
};

// An attention transition keyed by a complete causal prefix in shared storage.
struct AttentionRow {
  uint32_t offset;  // First StateId of this prefix in AttentionTable::keys.
  uint32_t length;  // Number of prefix symbols, including the current position.
  StateId output;   // Current position's symbol after the attention residual.
};

// A compiled transition's output, or nullopt for an unsupported input. An
// engaged zero is a valid vocabulary ID, not a failure sentinel. Transition
// functions must be pure: they cannot retain history across calls.
struct TransitionResult {
  std::optional<StateId> output;  // Result symbol, or nullopt if unsupported.
};

// One attention residual boundary, represented by rows or a pure function.
// Both forms consume the entire ordered causal prefix at a position.
struct AttentionTable {
  // Flattened prefix storage; each row selects a contiguous slice.
  absl::Span<const StateId> keys;
  // Unique transitions, sorted lexicographically by their prefix in keys.
  absl::Span<const AttentionRow> rows;
  // Optional compiled lookup; when non-null, both keys and rows must be empty.
  TransitionResult (*function)(absl::Span<const StateId>) = nullptr;
};

// A pointwise MLP or snap boundary, represented by rows or a pure function.
struct StateTable {
  // Transitions sorted by input symbol, with no duplicate inputs.
  absl::Span<const StateRow> rows;
  // Optional compiled lookup; when non-null, rows must be empty.
  TransitionResult (*function)(StateId) = nullptr;
};

// A finite integer network. Tables or pure functions implement each boundary;
// no sample identity, corpus text, expected suffix, or floating-point weights
// are available to this object. Referenced storage must remain immutable and
// outlive the model's use.
struct Model {
  uint32_t context_length;  // Maximum number of tokens in a causal sequence.
  uint32_t prompt_tokens;   // Prefix length for corpus verification.
  TokenId eos_token;        // Compact ID that terminates generation.
  // Indexed by compact TokenId; its bytes are also used for decoding.
  absl::Span<const VocabularyRow> vocabulary;
  // Embedding rows sorted by (token, position); empty with entry_function.
  absl::Span<const EntryRow> entry;
  // Attention residual boundaries in transformer-block order.
  absl::Span<const AttentionTable> attention;
  // MLP residual boundaries, one per attention block in the same order.
  absl::Span<const StateTable> mlp;
  // Final LayerNorm/top-1 readout: residual symbols -> compact tokens.
  StateTable snap;
  // Optional pure embedding lookup; when non-null, entry must be empty.
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
