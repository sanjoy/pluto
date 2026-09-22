// Generated declarations.
#pragma once
#include "src/llm/experiments/memorize_general_facts/discretized_model/runtime.h"
namespace pluto::llm::discretized {
absl::Span<const EntryRow> GeneratedEntry();
absl::Span<const VocabularyRow> GeneratedVocabulary();
StateTable GeneratedSnap();
AttentionTable GeneratedAttention0();
StateTable GeneratedMlp0();
AttentionTable GeneratedAttention1();
StateTable GeneratedMlp1();
AttentionTable GeneratedAttention2();
StateTable GeneratedMlp2();
AttentionTable GeneratedAttention3();
StateTable GeneratedMlp3();
AttentionTable GeneratedAttention4();
StateTable GeneratedMlp4();
AttentionTable GeneratedAttention5();
StateTable GeneratedMlp5();
AttentionTable GeneratedAttention6();
StateTable GeneratedMlp6();
AttentionTable GeneratedAttention7();
StateTable GeneratedMlp7();
TransitionResult GeneratedEntryFunction(TokenId, uint32_t);
}  // namespace pluto::llm::discretized
