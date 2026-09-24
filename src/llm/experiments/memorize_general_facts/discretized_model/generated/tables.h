// Private generated boundary declarations.
#pragma once
#include "src/llm/experiments/memorize_general_facts/discretized_model/runtime.h"
namespace pluto::llm::discretized::gen::internal {
PositionEmbedding& GeneratedPositionEmbedding();
absl::Span<const VocabularyRow> GeneratedVocabulary();
Map& GeneratedLanguageModelingHead();
CausalAttention& GeneratedAttention0();
Map& GeneratedMlp0();
CausalAttention& GeneratedAttention1();
Map& GeneratedMlp1();
CausalAttention& GeneratedAttention2();
Map& GeneratedMlp2();
CausalAttention& GeneratedAttention3();
Map& GeneratedMlp3();
}  // namespace pluto::llm::discretized::gen::internal
