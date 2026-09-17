#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "src/llm/experiments/ntk/experiment.h"
#include "src/llm/experiments/path_kernel/path_kernel.h"

namespace pluto::llm::path_kernel {

// Empty requested IDs defaults to sorted training-target IDs. Explicit IDs
// retain their order and may omit every training target: these select query
// logits only, independently of the full-vocabulary training loss.
absl::StatusOr<std::vector<int>> ResolveQueryTokens(
    absl::Span<const ntk::Example> examples, absl::Span<const int> requested,
    int vocabulary_size);

// Provenance of a new, controlled trajectory. A checkpoint is an initial
// condition, never a claim to reconstruct the optimizer history that made it.
// Examples are train-first, then held-out/prompt queries. Every example is also
// a query, with scalar coordinates ordered sample-major then
// output-token-major.
struct ExperimentReport {
  std::string checkpoint;
  std::string tokenizer_directory;
  std::string corpus_path;
  int seed = 123;
  std::string compute_type = "fp16";
  ntk::WindowOptions windows;
  ntk::ModelDimensions dimensions{};
  int padding_token = 0;
  size_t max_jacobian_bytes = 0;
  int steps = 3;
  double learning_rate = 1e-5;
  std::vector<ntk::Example> examples;
  std::vector<int> output_tokens;
  std::vector<std::string> output_token_text;
  Result result;
};

struct RenderedReport {
  std::string metadata_json;
  std::string path_kernel_csv;
  std::string contributions_csv;
  std::string predictions_csv;
};

// JSON is self-contained: provenance, scalar row map, full trajectory, signed
// contributions, and supporting/opposing examples sorted separately per query.
// CSVs duplicate the final matrices/predictions for convenient offline
// analysis. Arbitrary text/path bytes use explicitly named *_escaped_bytes
// fields.
absl::StatusOr<RenderedReport> RenderReport(const ExperimentReport& report);

// Exclusively creates a NEW directory below an existing parent. Existing paths,
// including dangling symlinks, are never overwritten. An I/O failure may leave
// a partial new report directory so that already-written evidence is retained.
absl::Status WriteReportDirectory(const std::filesystem::path& directory,
                                  const RenderedReport& report);

}  // namespace pluto::llm::path_kernel
