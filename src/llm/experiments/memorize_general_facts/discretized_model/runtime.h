#pragma once

#include <compare>
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

struct DiscreteHiddenState;

// A compact vocabulary token; explicit state conversion preserves its ID.
struct DiscreteToken {
  int value = 0;  // Index in Model::vocabulary; negative values are invalid.

  constexpr explicit operator DiscreteHiddenState() const;
  constexpr auto operator<=>(const DiscreteToken&) const = default;
};

// A symbolic residual activation. Conversion to a token preserves its label;
// the caller must ensure that label is a valid vocabulary index.
struct DiscreteHiddenState {
  int value = 0;  // Symbol label; negative values are invalid.

  constexpr explicit operator DiscreteToken() const {
    return DiscreteToken{value};
  }
  constexpr auto operator<=>(const DiscreteHiddenState&) const = default;
};

constexpr DiscreteToken::operator DiscreteHiddenState() const {
  return DiscreteHiddenState{value};
}

// One vocabulary entry, indexed by DiscreteToken::value in Model::vocabulary.
struct VocabularyRow {
  int32_t original_id;      // Token ID before compact-vocabulary remapping.
  absl::string_view bytes;  // Exact decoded bytes, not necessarily valid UTF-8.
};

// A compiled transition's output, or nullopt for an unsupported input. An
// engaged zero is a valid vocabulary ID, not a failure sentinel. Transition
// functions must be pure: they cannot retain history across calls.
struct TransitionResult {
  // Result symbol, or nullopt if unsupported.
  std::optional<DiscreteHiddenState> output;
};

// Maps the complete ordered prefix of residual symbols, including the current
// position, to that position's post-attention residual symbol.
struct AttentionTable {
  // Required pure function; unsupported inputs produce an empty output.
  TransitionResult (*function)(absl::Span<const DiscreteHiddenState>) = nullptr;
};

// Maps one residual symbol to a post-MLP residual symbol, or to a compact
// vocabulary token ID for the final snap.
struct StateTable {
  // Required pure function; unsupported inputs produce an empty output.
  TransitionResult (*function)(DiscreteHiddenState) = nullptr;
};

// A finite integer network. Pure compiled functions implement each boundary;
// no sample identity, corpus text, expected suffix, or floating-point weights
// are available to this object. Referenced storage must remain immutable and
// outlive the model's use.
struct Model {
  uint32_t context_length;  // Maximum number of tokens in a causal sequence.
  uint32_t prompt_tokens;   // Prefix length for corpus verification.
  DiscreteToken eos_token;  // Compact ID that terminates generation.
  // Indexed by compact DiscreteToken; its bytes are also used for decoding.
  absl::Span<const VocabularyRow> vocabulary;
  // Attention residual boundaries in transformer-block order.
  absl::Span<const AttentionTable> attention;
  // MLP residual boundaries, one per attention block in the same order.
  absl::Span<const StateTable> mlp;
  // Final LayerNorm/top-1 readout: residual symbols -> compact tokens.
  StateTable snap;
  // Required pure lookup: (compact token, absolute position) -> state.
  TransitionResult (*entry_function)(DiscreteToken, uint32_t) = nullptr;
};

// Checks model dimensions and required lookup functions. Generation validates
// any private table storage; independent tests verify the compiled transitions.
// Inference also calls this inexpensive structural check before dispatch.
absl::Status ValidateModel(const Model& model);

// Executes entry -> (causal attention -> pointwise MLP)* -> final snap.
// Recomputes all real positions. Unknown keys fail explicitly; there is no
// nearest-state fallback and no corpus-line or continuation lookup.
absl::StatusOr<DiscreteToken> PredictNext(
    const Model& model, absl::Span<const DiscreteToken> tokens);

// Returns newly generated tokens, INCLUDING EOS when reached. Zero budget
// returns an empty vector after prompt validation. Stops at context_length.
absl::StatusOr<std::vector<DiscreteToken>> Generate(
    const Model& model, absl::Span<const DiscreteToken> prompt,
    size_t max_new_tokens);
absl::StatusOr<std::string> Decode(const Model& model,
                                   absl::Span<const DiscreteToken> tokens);

// Implemented by the generated production code; it does not link fixtures.
const Model& GeneratedModel();

}  // namespace pluto::llm::discretized
