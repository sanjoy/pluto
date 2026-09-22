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
  // Index in DiscreteModel::vocabulary; negative values are invalid.
  int value = 0;

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

// One vocabulary entry, indexed by DiscreteToken::value in
// DiscreteModel::vocabulary.
struct VocabularyRow {
  int32_t original_id;      // Token ID before compact-vocabulary remapping.
  absl::string_view bytes;  // Exact decoded bytes, not necessarily valid UTF-8.
};

// Maps the complete ordered prefix of residual symbols, including the current
// position, to that position's post-attention residual symbol. Implementations
// must be pure: they cannot retain sequence history across calls.
class CausalAttention {
 public:
  virtual ~CausalAttention() = default;

  // Returns nullopt when the complete causal prefix is unsupported.
  virtual std::optional<DiscreteHiddenState> operator()(
      absl::Span<const DiscreteHiddenState> prefix) = 0;
};

// Maps one residual symbol to a post-MLP residual symbol, or to a compact
// vocabulary token ID for the language modeling head. Implementations are pure.
class Map {
 public:
  virtual ~Map() = default;

  // Returns nullopt for an unsupported symbol. An engaged zero is a valid
  // vocabulary ID, not an error sentinel.
  virtual std::optional<DiscreteHiddenState> operator()(
      DiscreteHiddenState state) = 0;
};

// One transformer block; both residual boundaries are evaluated independently.
// The referenced operations must outlive this block and its model.
struct Transformer {
  CausalAttention& attention;  // Complete causal prefix -> attention residual.
  Map& mlp;                    // Attention residual -> MLP residual.
};

// Encodes a vocabulary token and its absolute position as a residual symbol.
// Implementations are pure and must not retain sequence history across calls.
class PositionEmbedding {
 public:
  virtual ~PositionEmbedding() = default;

  // Returns nullopt for an unsupported token or position, including negatives.
  virtual std::optional<DiscreteHiddenState> operator()(DiscreteToken token,
                                                        int32_t position) = 0;
};

// A finite integer network. Pure operations implement each boundary;
// no sample identity, corpus text, expected suffix, or floating-point weights
// are available to this object. The model borrows its operations and
// vocabulary; all referenced objects must outlive its use.
struct DiscreteModel {
  uint32_t context_length;  // Maximum number of tokens in a causal sequence.
  uint32_t prompt_token_count;  // Prefix length for corpus verification.
  DiscreteToken eos_token;      // Compact ID that terminates generation.
  // Indexed by compact DiscreteToken; its bytes are also used for decoding.
  absl::Span<const VocabularyRow> vocabulary;
  // Ordered transformer blocks; each pairs its attention and MLP operations.
  absl::Span<const Transformer> transformers;
  // Final LayerNorm/top-1 readout: residual symbols -> compact tokens.
  Map& language_modeling_head;
  // (Compact token, absolute position) -> token-plus-position residual symbol.
  PositionEmbedding& position_embedding;
};

// Checks dimensions, including that positions fit in int32_t. Generation and
// independent tests verify the individual operations' transition behavior.
// Inference also calls this inexpensive structural check before dispatch.
absl::Status ValidateModel(const DiscreteModel& model);

// Executes entry -> (causal attention -> pointwise MLP)* -> language modeling
// head. Recomputes all real positions. Unknown keys fail explicitly; there is
// no nearest-state fallback and no corpus-line or continuation lookup.
absl::StatusOr<DiscreteToken> PredictNext(
    const DiscreteModel& model, absl::Span<const DiscreteToken> tokens);

// Returns newly generated tokens, INCLUDING EOS when reached. Zero budget
// returns an empty vector after prompt validation. Stops at context_length.
absl::StatusOr<std::vector<DiscreteToken>> Generate(
    const DiscreteModel& model, absl::Span<const DiscreteToken> prompt,
    size_t max_new_tokens);
absl::StatusOr<std::string> Decode(const DiscreteModel& model,
                                   absl::Span<const DiscreteToken> tokens);

}  // namespace pluto::llm::discretized
