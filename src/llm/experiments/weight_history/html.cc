#include "src/llm/experiments/weight_history/html.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <limits>
#include <locale>
#include <optional>
#include <ostream>
#include <sstream>
#include <string>
#include <string_view>

#include "absl/status/status.h"

namespace pluto::llm::weight_history {
namespace {

// The fixed viewBox scales to the browser width. Normalize integer step values
// in C++, not JavaScript: JS numbers cannot represent every int64 step exactly.
constexpr double kLeft = 70;
constexpr double kRight = 880;
constexpr double kTop = 20;
constexpr double kBottom = 210;

bool ValidMetric(double value) { return std::isfinite(value) && value >= 0; }

bool ValidMetric(const std::optional<double>& value) {
  return !value.has_value() || ValidMetric(*value);
}

absl::Status Validate(const History& history) {
  for (size_t i = 0; i < history.steps.size(); ++i)
    if (history.steps[i] < 0 ||
        (i > 0 && history.steps[i] <= history.steps[i - 1]))
      return absl::InvalidArgumentError(
          "HTML history steps must be increasing");
  for (size_t tensor_index = 0; tensor_index < history.tensors.size();
       ++tensor_index) {
    const TensorHistory& tensor = history.tensors[tensor_index];
    if (tensor.weight_id < 0 ||
        (tensor_index > 0 &&
         tensor.weight_id <= history.tensors[tensor_index - 1].weight_id))
      return absl::InvalidArgumentError("HTML weight IDs must be increasing");
    if (tensor.element_count == 0 || tensor.samples.empty() ||
        tensor.samples.size() != history.steps.size())
      return absl::InvalidArgumentError(
          "HTML tensor samples do not match steps");
    for (size_t i = 0; i < tensor.samples.size(); ++i) {
      const Sample& sample = tensor.samples[i];
      if (sample.step != history.steps[i] || !ValidMetric(sample.rms) ||
          !ValidMetric(sample.l2) || !ValidMetric(sample.from_first_rms) ||
          !ValidMetric(sample.delta_rms) || !ValidMetric(sample.delta_l2) ||
          !ValidMetric(sample.relative_l2) ||
          !ValidMetric(sample.max_abs_delta) ||
          !ValidMetric(sample.changed_fraction) ||
          (sample.changed_fraction && *sample.changed_fraction > 1))
        return absl::InvalidArgumentError(
            "HTML sample has invalid measurements");
      if (i == 0) {
        if (sample.delta_rms || sample.delta_l2 || sample.relative_l2 ||
            sample.max_abs_delta || sample.changed_fraction ||
            sample.from_first_rms != 0)
          return absl::InvalidArgumentError(
              "First HTML sample has no predecessor");
      } else if (!sample.delta_rms || !sample.delta_l2 ||
                 !sample.max_abs_delta || !sample.changed_fraction) {
        return absl::InvalidArgumentError(
            "HTML sample is missing delta metrics");
      } else if (sample.relative_l2.has_value() !=
                 (tensor.samples[i - 1].l2 > 0)) {
        return absl::InvalidArgumentError(
            "HTML relative change does not match predecessor norm");
      }
    }
  }
  return absl::OkStatus();
}

std::string EscapeHtml(std::string_view text) {
  std::string escaped;
  for (char c : text) {
    switch (c) {
      case '&':
        escaped += "&amp;";
        break;
      case '<':
        escaped += "&lt;";
        break;
      case '>':
        escaped += "&gt;";
        break;
      case '"':
        escaped += "&quot;";
        break;
      case '\'':
        escaped += "&#39;";
        break;
      default:
        escaped += c;
    }
  }
  return escaped;
}

double X(const History& history, int64_t step) {
  if (history.steps.front() == history.steps.back())
    return (kLeft + kRight) / 2;
  const long double first = history.steps.front();
  const long double last = history.steps.back();
  return kLeft +
         (kRight - kLeft) *
             static_cast<double>((static_cast<long double>(step) - first) /
                                 (last - first));
}

void NumberOrNull(std::ostream& output, const std::optional<double>& value) {
  if (value)
    output << *value;
  else
    output << "null";
}

void Cell(std::ostream& output, const std::optional<double>& value) {
  output << "<td>";
  if (value)
    output << *value;
  else
    output << "N/A";
  output << "</td>";
}

// Only validated numbers and decimal integer strings enter the embedded JSON.
// In particular, filesystem paths never enter script elements or JS source.
void Data(std::ostream& output, const History& history,
          const TensorHistory& tensor) {
  output << "<script type=\"application/json\" class=\"series\">[";
  bool first = true;
  for (const Sample& sample : tensor.samples) {
    if (!first)
      output << ',';
    first = false;
    output << "{\"step\":\"" << sample.step
           << "\",\"x\":" << X(history, sample.step)
           << ",\"rms\":" << sample.rms << ",\"l2\":" << sample.l2
           << ",\"from_first_rms\":" << sample.from_first_rms
           << ",\"delta_rms\":";
    NumberOrNull(output, sample.delta_rms);
    output << ",\"delta_l2\":";
    NumberOrNull(output, sample.delta_l2);
    output << ",\"relative_l2\":";
    NumberOrNull(output, sample.relative_l2);
    output << ",\"max_abs_delta\":";
    NumberOrNull(output, sample.max_abs_delta);
    output << ",\"changed_fraction\":";
    NumberOrNull(output, sample.changed_fraction);
    output << '}';
  }
  output << "]</script>\n";
}

void Svg(std::ostream& output, const History& history,
         const TensorHistory& tensor) {
  double maximum = 0;
  size_t points = 0;
  for (const Sample& sample : tensor.samples)
    if (sample.delta_rms) {
      maximum = std::max(maximum, *sample.delta_rms);
      ++points;
    }
  // A zero series must still produce a visible horizontal line and finite SVG
  // coordinates. Nonzero series use their actual maximum, without clipping.
  const double scale = maximum > 0 ? maximum : 1;
  output
      << "<svg class=\"chart\" viewBox=\"0 0 900 250\" role=\"img\" "
         "aria-label=\"RMS change from previous checkpoint for weight_"
      << tensor.weight_id << ".bin\">\n"
      << "<path class=\"axes\" d=\"M70 20V210H880\"/>"
      << "<text x=\"62\" y=\"214\" text-anchor=\"end\">0</text>"
      << "<text class=\"y-max\" x=\"62\" y=\"24\" text-anchor=\"end\">"
      << std::setprecision(4) << scale
      << std::setprecision(std::numeric_limits<double>::max_digits10)
      << "</text>" << "<text x=\"70\" y=\"238\">" << history.steps.front()
      << "</text>" << "<text x=\"880\" y=\"238\" text-anchor=\"end\">"
      << history.steps.back() << "</text>"
      << "<text x=\"475\" y=\"238\" text-anchor=\"middle\">training step</text>"
      << "<g class=\"plot\">";
  if (points == 0)
    output
        << "<text class=\"empty\" x=\"475\" y=\"120\" "
           "text-anchor=\"middle\">No predecessor checkpoint to compare</text>";
  else {
    output << "<polyline class=\"line\" points=\"";
    for (const Sample& sample : tensor.samples)
      if (sample.delta_rms)
        output << X(history, sample.step) << ','
               << kBottom - (*sample.delta_rms / scale) * (kBottom - kTop)
               << ' ';
    output << "\"/>";
    for (const Sample& sample : tensor.samples)
      if (sample.delta_rms)
        output << "<circle cx=\"" << X(history, sample.step) << "\" cy=\""
               << kBottom - (*sample.delta_rms / scale) * (kBottom - kTop)
               << "\" r=\"3\"><title>Step " << sample.step << ": "
               << *sample.delta_rms << "</title></circle>";
  }
  output << "</g></svg>\n";
}

constexpr char kHeader[] = R"HTML(<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Weight history</title>
<style>
:root{color-scheme:light;--ink:#182331;--muted:#576574;--accent:#176b73;--line:#dce4e9}
*{box-sizing:border-box}body{margin:0;background:#f2f5f7;color:var(--ink);font:15px/1.5 system-ui,sans-serif}
main{max-width:1180px;margin:auto;padding:38px 24px}h1{font-size:34px;letter-spacing:-1px;margin:0}
h2{font-size:19px;margin:0}p{margin:8px 0}.subtitle,.meta,.note{color:var(--muted)}
code{overflow-wrap:anywhere}header{margin-bottom:24px}.controls,.tensor{background:white;border:1px solid var(--line);border-radius:10px;padding:20px;margin:18px 0}
.controls{display:flex;align-items:end;gap:22px;flex-wrap:wrap}.controls label{display:grid;gap:5px;font-weight:600}
input,select{font:inherit;color:inherit;border:1px solid #bdcbd3;background:white;border-radius:5px;padding:8px}
.description{flex-basis:100%;margin:0;color:var(--muted)}.tensor-head{display:flex;align-items:baseline;justify-content:space-between;gap:20px;flex-wrap:wrap}
.chart{display:block;width:100%;height:auto;min-height:150px;margin-top:12px;overflow:visible}
.chart text{font:11px system-ui,sans-serif;fill:var(--muted)}.chart .empty{font-size:15px}
.axes{fill:none;stroke:var(--line)}.line{fill:none;stroke:var(--accent);stroke-width:2}
circle{fill:var(--accent)}circle:hover{r:5;fill:#ba5321}details{margin-top:10px}summary{cursor:pointer;color:var(--accent)}
.table-scroll{overflow:auto}table{width:100%;border-collapse:collapse;font-size:12px;font-variant-numeric:tabular-nums}
th,td{text-align:right;padding:8px;white-space:nowrap;border-bottom:1px solid var(--line)}th:first-child,td:first-child{text-align:left}
.note{font-size:13px}.js-controls{display:none}[hidden]{display:none!important}
@media print{.controls{display:none}.tensor{break-inside:avoid}main{padding:0}}
</style></head><body><main><header><h1>Weight history</h1>
<p class="subtitle">Checkpoint-to-checkpoint changes, one graph per weight tensor.</p>
)HTML";

constexpr char kControls[] =
    R"HTML(<section class="controls js-controls" aria-label="Graph controls">
<label>Metric<select id="metric">
<option value="delta_rms">RMS change from previous checkpoint</option>
<option value="delta_l2">L2 change from previous checkpoint</option>
<option value="relative_l2">Relative L2 change</option>
<option value="from_first_rms">RMS distance from first checkpoint</option>
<option value="rms">Weight RMS magnitude</option>
<option value="l2">Weight L2 magnitude</option>
<option value="max_abs_delta">Largest absolute coordinate change</option>
<option value="changed_fraction">Fraction of coordinates changed</option>
</select></label>
<label>Filter tensors<input id="filter" type="search" placeholder="weight_12.bin"></label>
<span id="visible-count" class="meta"></span>
<p id="description" class="description">RMS change = ||W(current) - W(previous)||₂ / √(number of elements).</p>
</section>
<noscript><p class="note">JavaScript is disabled. Default RMS-change graphs and all measurement tables are still available.</p></noscript>
)HTML";

// The report's JavaScript never interprets user-controlled text as markup.
// SVG nodes are created explicitly; labels are assigned through textContent.
constexpr char kScript[] = R"HTML(<script>
(() => {
  'use strict';
  const descriptions = {
    delta_rms: 'RMS change = ||W(current) - W(previous)||₂ / √(number of elements).',
    delta_l2: 'L2 change = ||W(current) - W(previous)||₂. Larger tensors can have larger L2 changes for the same per-coordinate change.',
    relative_l2: 'Relative L2 change = ||W(current) - W(previous)||₂ / ||W(previous)||₂. N/A when the previous norm is zero.',
    from_first_rms: 'RMS distance from the first available checkpoint. This is net displacement, not accumulated travel distance.',
    rms: 'Weight magnitude = ||W(current)||₂ / √(number of elements).',
    l2: 'Weight magnitude = ||W(current)||₂.',
    max_abs_delta: 'Largest |W(current)[i] - W(previous)[i]| over all coordinates.',
    changed_fraction: 'Fraction of stored coordinates whose numerical value changed since the previous checkpoint (0 to 1).'
  };
  const cards = Array.from(document.querySelectorAll('.tensor'), card => ({
    card, samples: JSON.parse(card.querySelector('.series').textContent)
  }));
  const metric = document.getElementById('metric');
  const filter = document.getElementById('filter');
  function node(name, attributes, text) {
    const element = document.createElementNS('http://www.w3.org/2000/svg', name);
    for (const [key, value] of Object.entries(attributes)) element.setAttribute(key, value);
    if (text !== undefined) element.textContent = text;
    return element;
  }
  function draw({card, samples}) {
    const key = metric.value;
    const svg = card.querySelector('.chart');
    const plot = card.querySelector('.plot');
    const values = samples.filter(sample => sample[key] !== null);
    let maximum = 0;
    for (const sample of values) maximum = Math.max(maximum, sample[key]);
    const scale = maximum > 0 ? maximum : 1;
    svg.setAttribute('aria-label', metric.selectedOptions[0].textContent + ' for ' + card.dataset.name);
    card.querySelector('.y-max').textContent = scale.toPrecision(4);
    plot.replaceChildren();
    if (!values.length) {
      plot.appendChild(node('text', {class:'empty',x:475,y:120,'text-anchor':'middle'}, 'No defined values for this metric'));
      return;
    }
    // Null metrics break the polyline. Connecting across an undefined relative
    // change would visually imply a measurement where none exists.
    let segment = [];
    function flush() {
      if (segment.length) plot.appendChild(node('polyline', {class:'line',points:segment.join(' ')}));
      segment = [];
    }
    for (const sample of samples) {
      if (sample[key] === null) { flush(); continue; }
      segment.push(sample.x + ',' + (210 - sample[key] / scale * 190));
    }
    flush();
    for (const sample of values) {
      const point = node('circle', {cx:sample.x,cy:210 - sample[key] / scale * 190,r:3});
      point.appendChild(node('title', {}, 'Step ' + sample.step + ': ' + sample[key]));
      plot.appendChild(point);
    }
  }
  function applyFilter() {
    const query = filter.value.trim().toLowerCase();
    let visible = 0;
    for (const {card} of cards) {
      card.hidden = !card.dataset.name.includes(query);
      if (!card.hidden) ++visible;
    }
    document.getElementById('visible-count').textContent = visible + ' / ' + cards.length + ' tensors';
  }
  metric.addEventListener('change', () => {
    document.getElementById('description').textContent = descriptions[metric.value];
    for (const entry of cards) draw(entry);
  });
  filter.addEventListener('input', applyFilter);
  document.querySelector('.js-controls').style.display = 'flex';
  applyFilter();
})();
</script></main></body></html>
)HTML";

}  // namespace

absl::Status WriteHtml(std::ostream& output, const History& history) {
  const absl::Status valid = Validate(history);
  if (!valid.ok())
    return valid;

  // Isolate locale and formatting flags from the caller. JSON numbers always
  // use '.' and max_digits10, even when the caller has a localized stream.
  std::ostringstream html;
  html.imbue(std::locale::classic());
  html << std::setprecision(std::numeric_limits<double>::max_digits10);
  html << kHeader << "<p>Source: <code>" << EscapeHtml(history.directory)
       << "</code></p><p class=\"meta\">" << history.steps.size()
       << " checkpoints &middot; " << history.tensors.size() << " tensors";
  if (!history.steps.empty())
    html << " &middot; steps " << history.steps.front() << " to "
         << history.steps.back();
  html
      << "</p><p class=\"note\">Each point compares two successive available "
         "checkpoints, not one optimizer step. Missing checkpoints are not "
         "observed training intervals; lines only connect measured snapshots. "
         "The first checkpoint has no predecessor, so its change is N/A. "
         "Relative change is N/A when the previous norm is zero. All stored "
         "coordinates are included, including padding. Graph axes scale "
         "independently per tensor; hover over points for exact step and value."
         "</p></header>\n"
      << kControls;
  if (history.tensors.empty())
    html << "<p>No tensors to display.</p>\n";
  for (const TensorHistory& tensor : history.tensors) {
    html << "<section class=\"tensor\" data-name=\"weight_" << tensor.weight_id
         << ".bin\"><div class=\"tensor-head\"><h2>weight_" << tensor.weight_id
         << ".bin</h2><span class=\"meta\">" << tensor.element_count
         << " elements</span></div>\n";
    Svg(html, history, tensor);
    Data(html, history, tensor);
    html << "<details><summary>Full-precision measurements</summary>"
            "<div class=\"table-scroll\"><table><thead><tr><th>Step</th>"
            "<th>Δ RMS</th><th>Δ L2</th><th>Relative Δ L2</th>"
            "<th>RMS from first</th><th>Weight RMS</th><th>Weight L2</th>"
            "<th>Max |Δ|</th><th>Changed fraction</th></tr></thead><tbody>";
    for (const Sample& sample : tensor.samples) {
      html << "<tr><td>" << sample.step << "</td>";
      Cell(html, sample.delta_rms);
      Cell(html, sample.delta_l2);
      Cell(html, sample.relative_l2);
      Cell(html, sample.from_first_rms);
      Cell(html, sample.rms);
      Cell(html, sample.l2);
      Cell(html, sample.max_abs_delta);
      Cell(html, sample.changed_fraction);
      html << "</tr>";
    }
    html << "</tbody></table></div></details></section>\n";
  }
  html << kScript;
  output << html.str();
  if (!output.good())
    return absl::InternalError("Failed to write weight-history HTML");
  return absl::OkStatus();
}

}  // namespace pluto::llm::weight_history
