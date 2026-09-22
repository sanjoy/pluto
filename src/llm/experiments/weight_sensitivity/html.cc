#include "src/llm/experiments/weight_sensitivity/html.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <locale>
#include <numeric>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "absl/status/status.h"
#include "src/util/status_macros.h"

namespace pluto::llm::weight_sensitivity {
namespace {

// User-supplied values are emitted only into HTML text, never into JavaScript.
std::string Escape(std::string_view text) {
  std::string result;
  for (char c : text) {
    switch (c) {
      case '&':
        result += "&amp;";
        break;
      case '<':
        result += "&lt;";
        break;
      case '>':
        result += "&gt;";
        break;
      case '\"':
        result += "&quot;";
        break;
      case '\'':
        result += "&#39;";
        break;
      default:
        result += c;
        break;
    }
  }
  return result;
}

bool NonnegativeFinite(double value) {
  return std::isfinite(value) && value >= 0;
}

absl::Status ValidateScores(const CompletionScores& scores, size_t samples) {
  if (scores.exact.size() != samples || samples == 0)
    return absl::InvalidArgumentError(
        "score vectors must match a nonempty corpus");
  if (scores.scored_tokens <= 0 || scores.token_errors < 0 ||
      scores.token_errors > scores.scored_tokens || scores.nonfinite_rows < 0 ||
      scores.nonfinite_rows > scores.scored_tokens)
    return absl::InvalidArgumentError("invalid completion score counters");
  for (uint8_t exact : scores.exact)
    if (exact > 1)
      return absl::InvalidArgumentError("exactness values must be zero or one");
  return absl::OkStatus();
}

absl::Status Validate(const SensitivityReport& report) {
  const size_t samples = report.baseline.exact.size();
  RETURN_IF_ERROR(ValidateScores(report.baseline, samples));
  if (report.expected_samples < 1 || report.prompt_tokens < 1 ||
      report.batch_size < 1 || report.trials < 1 ||
      !NonnegativeFinite(report.noise_scale) ||
      !NonnegativeFinite(report.zero_rms_stddev) ||
      report.planned_results == 0 ||
      report.results.size() > report.planned_results ||
      (report.complete && report.results.size() != report.planned_results))
    return absl::InvalidArgumentError(
        "invalid sensitivity report options or progress");
  for (const AblationResult& result : report.results) {
    RETURN_IF_ERROR(ValidateScores(result.scores, samples));
    if (result.scores.scored_tokens != report.baseline.scored_tokens ||
        !NonnegativeFinite(result.noise_stddev) ||
        !NonnegativeFinite(result.seconds) || result.trial < 0 ||
        result.trial >= report.trials)
      return absl::InvalidArgumentError(
          "invalid ablation metrics or trial number");
    const WeightTarget& target = result.target;
    if (target.name.empty() || target.checkpoint_index < 0 ||
        target.rows == 0 || target.columns == 0 ||
        target.row_stride < target.columns ||
        target.offset >= target.tensor_elements ||
        target.columns > target.tensor_elements - target.offset ||
        target.rows - 1 >
            (target.tensor_elements - target.offset - target.columns) /
                target.row_stride ||
        target.shape.empty())
      return absl::InvalidArgumentError("invalid weight target metadata");
    for (int64_t dimension : target.shape)
      if (dimension <= 0)
        return absl::InvalidArgumentError(
            "weight shapes must have positive dimensions");
  }
  return absl::OkStatus();
}

size_t ExactCount(const CompletionScores& scores) {
  return std::count(scores.exact.begin(), scores.exact.end(), uint8_t{1});
}

size_t NewlyWrong(const CompletionScores& baseline,
                  const CompletionScores& trial) {
  size_t result = 0;
  for (size_t i = 0; i < baseline.exact.size(); ++i)
    result += baseline.exact[i] && !trial.exact[i];
  return result;
}

std::string Number(double value, int precision = 6) {
  std::ostringstream output;
  output.imbue(std::locale::classic());
  output << std::setprecision(precision) << value;
  return output.str();
}

std::string Shape(const WeightTarget& target) {
  std::ostringstream output;
  output.imbue(std::locale::classic());
  for (size_t i = 0; i < target.shape.size(); ++i) {
    if (i != 0)
      output << " &times; ";
    output << target.shape[i];
  }
  return output.str();
}

void CountCell(std::ostream& output, size_t count, size_t total) {
  output << "<td data-sort=\"" << count << "\">" << count << " / " << total
         << " (" << Number(100.0 * count / total, 4) << "%)</td>";
}

}  // namespace

absl::StatusOr<std::string> RenderHtml(const SensitivityReport& report) {
  RETURN_IF_ERROR(Validate(report));
  const size_t samples = report.baseline.exact.size();
  std::vector<size_t> order(report.results.size());
  std::iota(order.begin(), order.end(), 0);
  // Sort stably so ties retain target enumeration and trial order.
  std::stable_sort(order.begin(), order.end(), [&](size_t left, size_t right) {
    const CompletionScores& a = report.results[left].scores;
    const CompletionScores& b = report.results[right].scores;
    const size_t a_new = NewlyWrong(report.baseline, a);
    const size_t b_new = NewlyWrong(report.baseline, b);
    if (a_new != b_new)
      return a_new > b_new;
    return ExactCount(a) < ExactCount(b);
  });

  std::ostringstream out;
  out.imbue(std::locale::classic());
  out << R"html(<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Weight sensitivity: exact completion ablations</title>
<style>
:root { color-scheme: light dark; font: 15px/1.5 system-ui, sans-serif; }
body { margin: 2rem auto; padding: 0 1.5rem; max-width: 1500px; }
h1 { font-size: 1.8rem; } h2 { font-size: 1.2rem; }
code { overflow-wrap: anywhere; } .muted { opacity: .75; }
.summary { border: 1px solid #8993a4; border-radius: 8px; padding: 1rem; }
dl { display: grid; grid-template-columns: max-content minmax(0,1fr); gap: .3rem 1rem; }
dd { margin: 0; } .table-container { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }
th, td { border-bottom: 1px solid #8993a4; padding: .55rem; text-align: left; }
th { white-space: nowrap; } td:first-child { min-width: 12rem; }
th button { font: inherit; color: inherit; border: 0; background: transparent;
            text-align: left; padding: 0; cursor: pointer; }
tbody tr:hover { background: #8993a422; } .slice { font-size: .85rem; }
</style></head><body>
<h1>Weight sensitivity</h1>
<p>One tensor is replaced with noise at a time; every other weight is restored
to its original checkpoint value. Each row measures exact corpus completions,
not just the next token.</p>
<section class="summary">
)html";
  out << "<h2>" << (report.complete ? "Complete" : "Partial results") << ": "
      << report.results.size() << " / " << report.planned_results
      << " target/trial evaluations</h2>\n";
  if (!report.complete)
    out << "<p>Unmeasured targets are not shown; no conclusion can be drawn "
           "about them.</p>\n";
  out << "<p><strong>Unmodified baseline:</strong> "
      << ExactCount(report.baseline) << " / " << samples
      << " exact completions; " << report.baseline.token_errors << " / "
      << report.baseline.scored_tokens << " token errors; "
      << report.baseline.nonfinite_rows << " nonfinite logit rows.</p>\n"
      << "<dl><dt>Checkpoint</dt><dd><code>" << Escape(report.checkpoint)
      << "</code></dd><dt>Corpus</dt><dd><code>" << Escape(report.corpus)
      << "</code></dd><dt>Architecture</dt><dd>"
      << report.config.transformer_block_count << " blocks; width "
      << report.config.model_width << "; " << report.config.attention_heads
      << " heads; MLP width " << report.config.feed_forward_width
      << "; vocabulary " << report.config.vocabulary_size
      << "</dd><dt>Evaluation</dt><dd>" << samples << " actual / "
      << report.expected_samples << " expected sentences; "
      << report.prompt_tokens << " prompt tokens; batch size "
      << report.batch_size << "; root seed " << report.seed << "; "
      << report.trials
      << " trial(s) per target</dd><dt>Noise</dt><dd>Zero-centered Gaussian; "
         "scale "
      << Number(report.noise_scale) << "; zero-RMS base standard deviation "
      << Number(report.zero_rms_stddev) << "</dd>";
  if (!report.target_filter.empty())
    out << "<dt>Target filter</dt><dd><code>" << Escape(report.target_filter)
        << "</code></dd>";
  out << "</dl></section>\n";
  out << R"html(<h2>Completion failures by target</h2>
<p><strong>Newly wrong</strong> counts completions correct at baseline but wrong
after corruption. <strong>All wrong</strong> also includes baseline failures.
Both percentages use the full corpus size. Highest newly-wrong count appears
first; click a column heading to sort.</p>
<noscript>The table is already sorted by impact; interactive sorting requires JavaScript.</noscript>
<div class="table-container"><table id="results">
<thead><tr>
<th><button>Tensor</button></th><th><button>Logical shape</button></th>
<th aria-sort="descending"><button>Newly wrong</button></th>
<th><button>All wrong</button></th><th><button>Token errors</button></th>
<th><button>Noise stddev</button></th><th><button>Checkpoint tensor / slice</button></th>
<th><button>Trial / seed</button></th><th><button>Seconds</button></th>
<th><button>Nonfinite rows</button></th>
</tr></thead><tbody>
)html";
  for (size_t index : order) {
    const AblationResult& result = report.results[index];
    const WeightTarget& target = result.target;
    out << "<tr><td>" << Escape(target.name) << "</td><td>" << Shape(target)
        << "</td>";
    CountCell(out, NewlyWrong(report.baseline, result.scores), samples);
    CountCell(out, samples - ExactCount(result.scores), samples);
    out << "<td data-sort=\"" << result.scores.token_errors << "\">"
        << result.scores.token_errors << " / " << result.scores.scored_tokens
        << "</td><td data-sort=\"" << Number(result.noise_stddev, 17) << "\">"
        << Number(result.noise_stddev) << "</td><td class=\"slice\">weight_"
        << target.checkpoint_index << ".bin<br>offset " << target.offset
        << "; rows " << target.rows << "; columns " << target.columns
        << "; row stride " << target.row_stride << "; tensor elements "
        << target.tensor_elements << "</td><td>" << result.trial + 1 << " / "
        << result.seed << "</td><td data-sort=\"" << Number(result.seconds, 17)
        << "\">" << Number(result.seconds) << "</td><td data-sort=\""
        << result.scores.nonfinite_rows << "\">" << result.scores.nonfinite_rows
        << "</td></tr>\n";
  }
  out << R"html(</tbody></table></div>
<h2>Method and interpretation</h2>
<ul>
<li>Noise replaces weights rather than adding to them. Its standard deviation
is the target's original root-mean-square value multiplied by the noise scale.
For a zero-RMS target, the configured fallback standard deviation is used
before applying the same scale. Each table row records the actual standard deviation.</li>
<li>Q, K and V matrices are independently corrupted slices of a combined checkpoint tensor.
Embedding and language-modeling head weights are tied: corrupting the token
embedding changes both uses. Physical offsets and strides are in float elements.</li>
<li>Evaluation supplies each sentence's first prompt tokens and checks the entire
remaining suffix plus EOS. A single causal teacher-forced pass determines exact
greedy-completion success: if every next-token prediction is correct, generated
and ground-truth histories coincide by induction; otherwise the first mismatch
already makes the completion wrong. Token-error counts after that first mismatch
are teacher-forced diagnostics, not errors along a generated continuation.</li>
<li>These are controlled corruption sensitivities, not unique semantic attributions.
A target may support many mechanisms, and redundant mechanisms may hide its role.
Magnitude and sampled noise affect the result; one trial does not establish
robustness across noise seeds or corruption scales. No claim is made for unseen prompts.</li>
</ul>
<script>
(() => {
  const table = document.getElementById('results');
  const headers = Array.from(table.tHead.rows[0].cells);
  let selected = 2, descending = true;
  headers.forEach((header, column) => header.querySelector('button').addEventListener('click', () => {
    descending = selected === column ? !descending : column !== 0 && column !== 1;
    selected = column;
    const rows = Array.from(table.tBodies[0].rows);
    rows.sort((left, right) => {
      const a = left.cells[column], b = right.cells[column];
      const comparison = a.dataset.sort !== undefined && b.dataset.sort !== undefined
        ? Number(a.dataset.sort) - Number(b.dataset.sort)
        : a.textContent.localeCompare(b.textContent, undefined, {numeric: true});
      return descending ? -comparison : comparison;
    });
    rows.forEach(row => table.tBodies[0].appendChild(row));
    headers.forEach(header => header.removeAttribute('aria-sort'));
    headers[column].setAttribute('aria-sort', descending ? 'descending' : 'ascending');
  }));
})();
</script></body></html>
)html";
  return out.str();
}

}  // namespace pluto::llm::weight_sensitivity
