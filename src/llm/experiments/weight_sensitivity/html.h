#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "src/llm/experiments/weight_sensitivity/evaluation.h"
#include "src/llm/experiments/weight_sensitivity/weights.h"
#include "src/llm/gpt2.h"

namespace pluto::llm::weight_sensitivity {

// One independent corruption; every other weight retains its checkpoint value.
struct AblationResult {
  WeightTarget target;      // Logical tensor or slice replaced with noise.
  int trial = 0;            // Zero-based trial number for this target.
  uint64_t seed = 0;        // Seed used to generate this corruption.
  double noise_stddev = 0;  // Actual Gaussian standard deviation.
  double seconds = 0;       // Elapsed evaluation time, excluding other targets.
  CompletionScores
      scores;  // Per-sentence exactness and aggregate token errors.
};

// Report metadata plus any completed trials. Partial reports remain useful if a
// long experiment is interrupted; they never imply unmeasured targets passed.
struct SensitivityReport {
  std::string checkpoint;       // Original, unmodified checkpoint directory.
  std::string corpus;           // Corpus used to measure exact completions.
  std::string target_filter;    // Optional substring selecting logical targets.
  Gpt2Config config;            // Architecture of the checkpoint.
  int expected_samples = 1024;  // Corpus-size assertion supplied to the tool.
  int prompt_tokens = 5;   // Initial ground-truth tokens supplied per line.
  int batch_size = 32;     // Number of corpus sentences evaluated together.
  uint64_t seed = 0;       // Root experiment seed; rows record actual seeds.
  double noise_scale = 1;  // Multiplier applied to each target's RMS.
  double zero_rms_stddev =
      .02;                     // Base standard deviation if target RMS is zero.
  int trials = 1;              // Planned independent corruptions per target.
  CompletionScores baseline;   // Scores without any corruption.
  size_t planned_results = 0;  // Total target/trial combinations planned.
  bool complete = false;  // True only when all planned trials have finished.
  std::vector<AblationResult> results;
};

// Produces a standalone, sortable HTML report, with highest impact first.
// Validates counters and escapes every supplied string before serialization.
absl::StatusOr<std::string> RenderHtml(const SensitivityReport& report);

}  // namespace pluto::llm::weight_sensitivity
