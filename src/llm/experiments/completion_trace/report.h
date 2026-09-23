#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/types/span.h"
#include "src/llm/experiments/completion_trace/trace.h"

namespace pluto::llm::completion_trace {

// One genuine autoregressive decision. Expected IDs are scoring metadata only;
// they must never be copied into the next forward's prefix.
struct CompletionStep {
  ForwardTrace forward;
  int predicted = -1;  // Argmax of the captured logical-vocabulary logits.
  int expected = -1;   // Corpus next token, or -1 when unavailable.
};

struct CompletionExample {
  int corpus_line = 0;  // One-based line in the source dataset.
  std::string sentence;
  std::vector<int> prompt;
  std::vector<CompletionStep> steps;
};

// Provenance and the checkpoint's exact compact-to-original token mapping.
struct ReportMetadata {
  std::string checkpoint;
  std::string tokenizer;
  std::string corpus;
  std::string source_revision;
  int model_width = 0;
  int layers = 0;
  int heads = 0;
  int feed_forward_width = 0;
  int eos_token = -1;
  std::vector<int> original_ids;
  std::vector<std::string> token_text;  // Raw decoded bytes, escaped on output.
};

// Undefined for a zero-norm vector, invalid lengths, or nonfinite values.
// This geometric similarity is not a causal effect or semantic label.
std::optional<double> CosineSimilarity(absl::Span<const float> a,
                                       absl::Span<const float> b);

// Human-readable complete numeric dumps (TXT and expandable, self-contained
// HTML), a vocabulary index, and descriptive query-state/attention statistics.
// query_walkthrough.txt is a smaller companion with the complete final-position
// feature vectors, labeled final attention rows, and top next-token candidates.
// No numeric arrays are truncated. Only causal attention entries are printed;
// the validated upper triangle is zero. Floating values use round-trip FP32
// precision. Existing output files are rejected; directory must already exist.
// Each snapshot site must retain its dtype and non-prefix dimensions across
// all forwards. Correlation groups refer to the predicted next-token IDs.
absl::Status WriteReports(const std::filesystem::path& directory,
                          const ReportMetadata& metadata,
                          absl::Span<const CompletionExample> examples);

}  // namespace pluto::llm::completion_trace
