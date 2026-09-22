#pragma once

#include <filesystem>
#include <ostream>

#include "absl/status/status.h"
#include "absl/types/span.h"
#include "src/llm/experiments/memorize_general_facts/activation_trace.h"

namespace pluto::llm::memorize_general_facts {

// Writes one self-contained HTML document, with a section for each prompt.
// Each width-16 activation is shown as eight consecutive dimension-pair plots.
// One symmetric scale per prompt applies to every token and boundary; vectors
// are not normalized. Exact values remain available in expandable text.
// Invalid traces are rejected before any output is written. Text, including
// partial UTF-8 tokens, is escaped and never interpreted as HTML or JavaScript.
absl::Status WriteActivationTraceHtml(absl::Span<const ActivationTrace> traces,
                                      std::ostream& output);

// Atomically replaces output_file after rendering and writing the whole report
// successfully. The parent directory must exist. A failed render/write leaves
// an existing report untouched; only this operation's temporary file is
// removed.
absl::Status WriteActivationTraceHtmlFile(
    absl::Span<const ActivationTrace> traces,
    const std::filesystem::path& output_file);

}  // namespace pluto::llm::memorize_general_facts
