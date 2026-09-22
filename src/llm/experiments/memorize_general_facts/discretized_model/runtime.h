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

// A compiled transition's output, or nullopt for an unsupported input. An
// engaged zero is a valid vocabulary ID, not a failure sentinel. Transition
// functions must be pure: they cannot retain history across calls.
struct TransitionResult {
  std::optional<StateId> output;  // Result symbol, or nullopt if unsupported.
};

// Maps the complete ordered prefix of residual symbols, including the current
// position, to that position's post-attention residual symbol.
struct AttentionTable {
  // Required pure function; unsupported inputs produce an empty output.
  TransitionResult (*function)(absl::Span<const StateId>) = nullptr;
};

// Maps one residual symbol to a post-MLP residual symbol, or to a compact
// vocabulary token ID for the final snap.
struct StateTable {
  // Required pure function; unsupported inputs produce an empty output.
  TransitionResult (*function)(StateId) = nullptr;
};

// A finite integer network. Pure compiled functions implement each boundary;
// no sample identity, corpus text, expected suffix, or floating-point weights
// are available to this object. Referenced storage must remain immutable and
// outlive the model's use.
struct Model {
  uint32_t context_length;  // Maximum number of tokens in a causal sequence.
  uint32_t prompt_tokens;   // Prefix length for corpus verification.
  TokenId eos_token;        // Compact ID that terminates generation.
  // Indexed by compact TokenId; its bytes are also used for decoding.
  absl::Span<const VocabularyRow> vocabulary;
  // Attention residual boundaries in transformer-block order.
  absl::Span<const AttentionTable> attention;
  // MLP residual boundaries, one per attention block in the same order.
  absl::Span<const StateTable> mlp;
  // Final LayerNorm/top-1 readout: residual symbols -> compact tokens.
  StateTable snap;
  // Required pure lookup: (compact token, absolute position) -> state.
  TransitionResult (*entry_function)(TokenId, uint32_t) = nullptr;
};

// Checks model dimensions and required lookup functions. Generation validates
// any private table storage; independent tests verify the compiled transitions.
// Inference also calls this inexpensive structural check before dispatch.
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

// Implemented by the generated production code; it does not link fixtures.
const Model& GeneratedModel();

}  // namespace pluto::llm::discretized
