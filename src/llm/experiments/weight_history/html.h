#pragma once

#include <iosfwd>

#include "absl/status/status.h"
#include "src/llm/experiments/weight_history/history.h"

namespace pluto::llm::weight_history {

// Writes an offline report: inline SVG graphs, full-precision measurements, and
// optional JavaScript controls. Every tensor has a graph even without scripts.
// Checks the history's ordering, metric validity, and sample alignment before
// writing. The caller owns the stream and is responsible for closing its file.
absl::Status WriteHtml(std::ostream& output, const History& history);

}  // namespace pluto::llm::weight_history
