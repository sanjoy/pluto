#pragma once

#include <compare>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "src/llm/experiments/memorize_general_facts/discretized_model/discrete_model_generator/model_statistics.h"

namespace pluto::llm::discretized::generator {

// Original tokenizer identity and exact bytes for one compact vocabulary ID.
// Its index in ModelMetadata::vocabulary is the compact ID.
struct VocabularyToken {
  int original_id = 0;
  std::string bytes;
  auto operator<=>(const VocabularyToken&) const = default;
};

struct ModelMetadata {
  int width = 0;   // Number of BF16 channels in each captured residual vector.
  int layers = 0;  // Number of attention/MLP transformer blocks.
  int vocab_size = 0;
  int eos_token = -1;
  int prompt_tokens = 0;  // Initial tokens supplied to autonomous verification.
  std::vector<VocabularyToken> vocabulary;
  int context_length = 1024;  // Maximum number of tokens in a causal sequence.
  auto operator<=>(const ModelMetadata&) const = default;
};

struct Sample {
  std::vector<int> tokens;  // Complete fact, excluding the final predicted EOS.
  auto operator<=>(const Sample&) const = default;
};

struct CapturedState {
  int id = 0;        // Globally unique hidden-state ID, above vocabulary IDs.
  int boundary = 0;  // Entry=0; block l attention=2*l+1, MLP=2*l+2.
  // Original IDs in this equivalence class; absent when no map is available.
  std::optional<std::vector<int>> members;
  auto operator<=>(const CapturedState&) const = default;
};

struct EntryTransition {
  int token = 0;
  int position = 0;
  int output = 0;
  auto operator<=>(const EntryTransition&) const = default;
};

struct AttentionTransition {
  std::vector<int> prefix;  // Entire ordered causal input-state history.
  int output = 0;
  auto operator<=>(const AttentionTransition&) const = default;
};

// Pointwise state mapping. For the language modeling head, output is a compact
// vocabulary ID; for an MLP, it is a hidden-state ID at the next boundary.
struct StateTransition {
  int input = 0;
  int output = 0;
  auto operator<=>(const StateTransition&) const = default;
};

// Observed token/position pairs and their embedding residual-state outputs.
struct CapturedPositionEmbedding {
  std::vector<EntryTransition> transitions;
  auto operator<=>(const CapturedPositionEmbedding&) const = default;
};

// Observed ordered causal prefixes and their attention residual-state outputs.
struct CapturedCausalAttention {
  std::vector<AttentionTransition> transitions;
  auto operator<=>(const CapturedCausalAttention&) const = default;
};

// Observed pointwise input/output states for an MLP or language modeling head.
struct CapturedMap {
  std::vector<StateTransition> transitions;
  auto operator<=>(const CapturedMap&) const = default;
};

// The same attention-then-MLP boundary structure used by DiscreteModel.
struct CapturedTransformer {
  CapturedCausalAttention attention;
  CapturedMap mlp;
  auto operator<=>(const CapturedTransformer&) const = default;
};

struct StateRelabeling {
  int old_id = 0;
  int new_id = 0;
  int boundary = 0;
  auto operator<=>(const StateRelabeling&) const = default;
};

// Finite transition system consumed by compaction and code generation. The
// expected corpus is verification-only and never consulted by a transition.
struct CapturedModel {
  ModelMetadata metadata;
  std::vector<CapturedState> states;
  std::vector<Sample> samples;
  CapturedPositionEmbedding position_embedding;
  std::vector<CapturedTransformer> transformers;
  // Required suffix/EOS readouts; prompt-position predictions are not targets.
  CapturedMap language_modeling_head;
  std::vector<StateRelabeling> state_relabeling;
  ModelStatistics stats;
  auto operator<=>(const CapturedModel&) const = default;
};

}  // namespace pluto::llm::discretized::generator
