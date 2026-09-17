#pragma once

#include <cstddef>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "src/llm/experiments/ntk/kernel_regression.h"

namespace pluto::llm::ntk {

// Token offsets refer to one encoding of the complete corpus, not encodings of
// independent substrings (which can have different BPE boundaries).
struct WindowOptions {
  size_t train_examples = 2;
  size_t eval_examples = 1;
  size_t context_tokens = 16;
  size_t stride = 1025;
  size_t offset = 0;
};

struct Example {
  // "train", "eval", or "prompt". Evaluation windows follow training windows.
  std::string split;
  std::optional<size_t> corpus_token_offset;
  std::vector<int> tokens;
  std::optional<int> next_token;
  // Decoded bytes, not necessarily valid UTF-8 at a token boundary. Reports use
  // explicit byte escapes rather than pretending every token is Unicode text.
  std::string text;
};

// Selects non-overlapping context-plus-target windows. It rejects insufficient
// data and arithmetic overflow rather than wrapping or reusing training data.
absl::StatusOr<std::vector<Example>> SelectCorpusExamples(
    absl::Span<const int> corpus, const WindowOptions& options);

// Empty requested IDs selects the sorted union of training next-token labels.
// Explicit IDs retain their order, must be distinct, and must cover every
// training label. Query/evaluation labels need not be in this subset.
absl::StatusOr<std::vector<int>> ResolveOutputTokens(
    absl::Span<const Example> examples, absl::Span<const int> requested,
    int vocabulary_size);
absl::StatusOr<std::vector<int>> ParseOutputTokenIds(absl::string_view text);

struct ParameterSummary {
  size_t weight_index;
  size_t elements;
  size_t offset;
};

struct ModelDimensions {
  int vocabulary_size;
  int padded_vocabulary_size;
  int context_length;
  int layers;
  int width;
  int attention_heads;
  int head_dimension;
  int feed_forward_width;
};

// All scalar rows use sample-major, output-token-major ordering. These are
// original model logits and fixed-kernel predictions, never updated weights.
struct ExperimentReport {
  std::string checkpoint;
  std::string tokenizer_directory;
  std::string corpus_path;
  int seed = 0;
  std::string compute_type = "fp16";
  WindowOptions windows;
  ModelDimensions dimensions{};
  int padding_token = 0;
  size_t max_jacobian_bytes = 0;
  std::vector<Example> examples;
  std::vector<int> output_tokens;
  std::vector<std::string> output_token_text;
  std::vector<ParameterSummary> parameters;
  size_t parameter_count = 0;
  double ridge = 1e-3;
  int kernel_steps = 0;
  double learning_rate = 1e-5;
  Matrix kernel;
  std::vector<double> initial_values;
  std::vector<double> ridge_predictions;
  // Empty when --kernel_steps=0 (the optional GD experiment is not requested).
  std::vector<double> gradient_descent_predictions;
};

struct RenderedReport {
  std::string metadata_json;
  std::string kernel_csv;
  std::string predictions_csv;
};

// CPU-only formatting/validation. JSON contains every matrix/prediction as well
// as provenance and a complete row map; the CSVs are convenience duplicates.
// Softmax values are explicitly conditional on selected output token IDs and
// must not be interpreted as probabilities over the full GPT-2 vocabulary.
absl::StatusOr<RenderedReport> RenderReport(const ExperimentReport& report);

// Check early, then exclusively create at write time to close the race. The
// parent must exist; neither helper overwrites any existing file or directory,
// including dangling symlinks. A write failure can leave a partial NEW report.
absl::Status CheckOutputDirectory(const std::filesystem::path& directory);
absl::Status WriteReportDirectory(const std::filesystem::path& directory,
                                  const RenderedReport& report);

}  // namespace pluto::llm::ntk
