#pragma once

#include <filesystem>
#include <functional>

#include "absl/status/statusor.h"
#include "src/llm/experiments/memorize_general_facts/discretized_model/discretize_core.h"
#include "src/llm/experiments/memorize_general_facts/discretized_model/generator_model.h"

namespace pluto::llm::discretized::generator {

// Conversion loads native weights and captures the corpus on the GPU. All
// intermediate traces and compacted models stay in memory; only C++ and
// readable inspection reports are written. Generated inference needs neither
// CUDA nor these input files.
struct GeneratorOptions {
  std::filesystem::path output;
  std::filesystem::path checkpoint;
  std::filesystem::path tokenizer;
  std::filesystem::path corpus;
  std::filesystem::path clang_format_config;
  int layers = 8;
  int attention_heads = 1;
  int feed_forward_width = 64;
  int prompt_tokens = 5;
  int expected_samples = 1024;
  bool verify_greedy = true;
  bool state_index = false;
  bool compaction = false;  // Identify compatible states within each boundary.
  // Only changes transition-code representation, not the state partition.
  bool compact_transitions = false;
  CompactionOptions compaction_options;
  std::function<void(const ProgressEvent&)> progress;
};

// One-step GPU checkpoint -> native trace -> verified CPU-only generated C++.
// Output is published atomically after capture, compaction, and verification.
absl::StatusOr<SymbolicModel> Generate(const GeneratorOptions& options);

}  // namespace pluto::llm::discretized::generator
