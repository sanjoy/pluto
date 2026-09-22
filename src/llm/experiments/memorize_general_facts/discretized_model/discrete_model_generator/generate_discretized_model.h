#pragma once

#include <filesystem>
#include <functional>

#include "absl/status/statusor.h"
#include "src/llm/experiments/memorize_general_facts/discretized_model/discrete_model_generator/captured_model.h"
#include "src/llm/experiments/memorize_general_facts/discretized_model/discrete_model_generator/model_recorder.h"
#include "src/llm/experiments/memorize_general_facts/discretized_model/discrete_model_generator/state_compactor.h"

namespace pluto::llm::discretized::generator {

// Conversion loads native weights and captures the corpus on the GPU. All
// intermediate traces and compacted models stay in memory; only C++ and
// readable inspection reports are written. Generated inference needs neither
// CUDA nor these input files.
struct GeneratorOptions {
  ModelRecorderOptions recorder;
  std::filesystem::path output;
  std::filesystem::path clang_format_config;
  bool state_index = false;
  bool compaction = false;  // Identify compatible states within each boundary.
  // Only changes transition-code representation, not the state partition.
  bool compact_transitions = false;
  CompactionOptions compaction_options;
  std::function<void(const ProgressEvent&)> progress;
};

// One-step GPU checkpoint -> native trace -> verified CPU-only generated C++.
// Output is published atomically after capture, compaction, and verification.
absl::StatusOr<CapturedModel> Generate(const GeneratorOptions& options);

}  // namespace pluto::llm::discretized::generator
