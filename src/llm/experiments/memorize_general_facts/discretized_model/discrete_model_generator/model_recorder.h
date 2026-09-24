#pragma once

#include <filesystem>
#include <functional>

#include "absl/status/statusor.h"
#include "src/llm/experiments/memorize_general_facts/discretized_model/discrete_model_generator/captured_model.h"
#include "src/llm/experiments/memorize_general_facts/discretized_model/discrete_model_generator/model_statistics.h"
#include "src/llm/experiments/memorize_general_facts/discretized_model/discrete_model_generator/state_vector_hints.h"

namespace pluto::llm::discretized::generator {

// Everything needed to reproduce a trained GPT-2 model's observed
// execution. Recording is independent of code generation and output paths.
struct ModelRecorderOptions {
  std::filesystem::path checkpoint;  // Weights and compact vocabulary mapping.
  std::filesystem::path tokenizer;   // Matching original GPT-2 tokenizer.
  std::filesystem::path corpus;      // One fact per line, with no EOS padding.
  int layers = 8;
  int model_width = 16;  // Residual width; must match the checkpoint.
  int context_length =
      1024;  // Learned position count and padded sequence size.
  int attention_heads = 1;
  int feed_forward_width = 64;
  int prompt_tokens = 5;  // Prefix supplied for autonomous verification.
  int expected_samples = 1024;
  bool verify_greedy = true;  // Also verify exact prefix states during rollout.
  std::function<void(const CaptureProgress&)> progress;
};

class ModelRecorder {
 public:
  // Records real corpus positions, interns exact BF16 residual states, and
  // checks every expected suffix and EOS. No files are written. Vector hints
  // are optional search guidance for compaction, not part of CapturedModel;
  // when requested, replaces *vector_hints only after successful recording.
  static absl::StatusOr<CapturedModel> Record(
      const ModelRecorderOptions& options,
      StateVectorHints* vector_hints = nullptr);
};

}  // namespace pluto::llm::discretized::generator
