#pragma once

#include <filesystem>

#include "absl/status/status.h"
#include "src/llm/experiments/one_shot_memorizer/sentence_ablation.h"

namespace pluto::llm::one_shot_memorizer {

// Writes per_tensor.tsv, coordinates.tsv and report.html in an EXISTING output
// directory, replacing only those files. The HTML is self-contained and shows
// summary statistics plus at most 20 supplied top coordinates per tensor.
// coordinates.tsv includes EVERY bitwise-changed scalar in physical order;
// its delta is baseline minus ablated. Baseline/ablated values are available
// only for top coordinates, so the exhaustive TSV deliberately records delta
// alone, plus physical and semantic coordinates and a bitwise-change marker.
// Signed-zero changes are retained even when their numerical delta is zero.
// These are matched-run parameter differences, not exclusive fact ownership.
// Malformed reports are rejected before opening any output file.
absl::Status WriteSentenceAblationReport(
    const ParameterDeltaReport& report,
    const std::filesystem::path& output_directory);

}  // namespace pluto::llm::one_shot_memorizer
