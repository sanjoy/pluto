#include "src/llm/experiments/memorize_general_facts/puzzle_report.h"

#include <cmath>
#include <cstddef>
#include <fstream>
#include <functional>
#include <iomanip>
#include <limits>
#include <locale>
#include <ostream>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_cat.h"

namespace pluto::llm::memorize_general_facts {

// Validate the common capture format before computing any geometric report.
// Cross-class collisions are allowed here so distance reports can count them.
absl::Status ValidatePuzzleReportData(const PuzzleReportData& data) {
  if (data.model_width == 0 || data.context_length == 0 || data.facts.empty())
    return absl::InvalidArgumentError(
        "puzzle report requires positive width, context length, and fact "
        "count");
  if (data.facts.size() >
          static_cast<size_t>(std::numeric_limits<int>::max()) ||
      data.facts.size() > std::numeric_limits<size_t>::max() /
                              static_cast<size_t>(data.context_length))
    return absl::InvalidArgumentError("puzzle report point count overflows");
  const size_t expected = data.facts.size() * data.context_length;
  if (data.points.size() != expected)
    return absl::InvalidArgumentError(
        "puzzle report requires exactly one point per fact and position");
  std::vector<bool> seen(expected, false);
  for (const PuzzlePoint& point : data.points) {
    if (point.fact_index < 0 ||
        static_cast<size_t>(point.fact_index) >= data.facts.size() ||
        point.position < 0 ||
        static_cast<unsigned int>(point.position) >= data.context_length)
      return absl::InvalidArgumentError(
          "puzzle report fact or position is out of bounds");
    const size_t index =
        static_cast<size_t>(point.fact_index) * data.context_length +
        point.position;
    if (seen[index])
      return absl::InvalidArgumentError(
          absl::StrCat("duplicate puzzle report point: fact ", point.fact_index,
                       ", position ", point.position));
    seen[index] = true;
    if (point.input_token < 0 || point.target_token < -1 ||
        (point.padding && point.target_token != -1))
      return absl::InvalidArgumentError(
          absl::StrCat("invalid puzzle token/padding metadata: fact ",
                       point.fact_index, ", position ", point.position));
    if (point.coordinates.size() != static_cast<size_t>(data.model_width))
      return absl::InvalidArgumentError(absl::StrCat(
          "puzzle coordinate width differs from model width: fact ",
          point.fact_index, ", position ", point.position));
    for (float coordinate : point.coordinates)
      if (!std::isfinite(coordinate))
        return absl::InvalidArgumentError(
            absl::StrCat("nonfinite puzzle coordinate: fact ", point.fact_index,
                         ", position ", point.position));
  }
  return absl::OkStatus();
}

namespace {

struct CoordinatesHash {
  size_t operator()(const std::vector<float>& coordinates) const {
    size_t hash = 0;
    for (float coordinate : coordinates) {
      // Canonicalize signed zero explicitly; vector equality is numerical.
      const size_t value = std::hash<float>{}(coordinate == 0 ? 0 : coordinate);
      hash ^=
          value + static_cast<size_t>(0x9e3779b9) + (hash << 6) + (hash >> 2);
    }
    return hash;
  }
};

// Escape both JSON syntax and HTML's script-closing syntax. Untrusted text is
// subsequently displayed only with textContent, never parsed as HTML.
void JsonString(std::ostream& output, absl::string_view text) {
  constexpr char kHex[] = "0123456789abcdef";
  output << '"';
  for (size_t i = 0; i < text.size(); ++i) {
    const unsigned char c = static_cast<unsigned char>(text[i]);
    if (c == '"' || c == '\\') {
      output << '\\' << static_cast<char>(c);
    } else if (c < 0x20 || c == '<' || c == '>' || c == '&') {
      output << "\\u00" << kHex[c >> 4] << kHex[c & 15];
    } else if (c == 0xe2 && i + 2 < text.size() &&
               static_cast<unsigned char>(text[i + 1]) == 0x80 &&
               (static_cast<unsigned char>(text[i + 2]) == 0xa8 ||
                static_cast<unsigned char>(text[i + 2]) == 0xa9)) {
      output << (static_cast<unsigned char>(text[i + 2]) == 0xa8 ? "\\u2028"
                                                                 : "\\u2029");
      i += 2;
    } else {
      output << static_cast<char>(c);
    }
  }
  output << '"';
}

void WriteData(std::ostream& output, const PuzzleReportData& data) {
  output << "{\"model_width\":" << data.model_width
         << ",\"context_length\":" << data.context_length << ",\"facts\":[";
  for (size_t i = 0; i < data.facts.size(); ++i) {
    if (i != 0)
      output << ',';
    JsonString(output, data.facts[i]);
  }
  output << "],\"points\":[";
  for (size_t i = 0; i < data.points.size(); ++i) {
    if (i != 0)
      output << ',';
    const PuzzlePoint& point = data.points[i];
    output << "{\"fact_index\":" << point.fact_index
           << ",\"position\":" << point.position
           << ",\"input_token\":" << point.input_token
           << ",\"target_token\":" << point.target_token
           << ",\"padding\":" << (point.padding ? "true" : "false")
           << ",\"coordinates\":[";
    for (size_t j = 0; j < point.coordinates.size(); ++j) {
      if (j != 0)
        output << ',';
      output << point.coordinates[j];
    }
    output << "]}";
  }
  output << "]}";
}

constexpr absl::string_view kPageStart = R"HTML(<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>A3 puzzle residuals</title>
<style>
:root{color-scheme:light;font:15px/1.5 system-ui,sans-serif;color:#162238;background:#f3f6fa}
body{margin:0;padding:32px clamp(16px,4vw,64px)}main{max-width:1800px;margin:auto}
h1{font-size:30px;margin:0 0 6px}h2{font-size:19px;margin:0}p{max-width:1100px}
.subtitle,.counts,.note{color:#56637a}.audit{padding:16px 20px;border:1px solid #b9dccc;
border-radius:12px;background:#eefaf3;margin:22px 0}.controls{display:flex;gap:22px;
flex-wrap:wrap;align-items:center;margin:18px 0}.legend{display:flex;gap:16px;flex-wrap:wrap}
.key:before{content:"";display:inline-block;width:10px;height:10px;border-radius:50%;
margin-right:6px;background:var(--color)}.position{background:white;border:1px solid #dce3ed;
border-radius:14px;margin:24px 0;padding:20px}.counts{margin:4px 0 16px}
.plots{display:grid;grid-template-columns:repeat(auto-fit,minmax(270px,1fr));gap:14px}
figure{margin:0;border:1px solid #e4e9f1;border-radius:9px;overflow:hidden}
figcaption{padding:9px 12px;background:#f8fafc;font-weight:600}canvas{width:100%;height:250px;
display:block}#tooltip{position:fixed;display:none;pointer-events:none;z-index:10;
white-space:pre-wrap;overflow-wrap:anywhere;max-width:min(600px,calc(100vw - 32px));
max-height:65vh;overflow:hidden;background:#17263af5;color:white;padding:12px 15px;
border-radius:8px;font:12px/1.5 ui-monospace,monospace;box-shadow:0 6px 25px #18253a33}
label{cursor:pointer}input{accent-color:#326ba5}noscript{display:block;color:#a12a2a}
</style></head><body><main>
<h1>A3 puzzle residuals</h1>
<p class="subtitle">Residual activations after the third attention layer (A3), before that block's MLP.
BF16 values are expanded exactly to float for inspection.</p>
<p>Each position has one point per fact in every chart, including padding by default.
Axes are raw coordinate pairs (1, 2), (3, 4), and so on, not PCA.
An odd final coordinate is plotted against zero. Coordinate ranges are shared across positions.
Overlapping two-dimensional points need not be equal in the full residual space.</p>
<div class="audit"><strong>Exact scored-row collision audit passed.</strong><br>
)HTML";

constexpr absl::string_view kPageScript = R"HTML(;
const root = document.getElementById('positions');
const tooltip = document.getElementById('tooltip');
const paddingToggle = document.getElementById('show-padding');
const byPosition = Array.from({length: data.context_length}, () => []);
const extents = Array.from({length: data.model_width}, () => [Infinity, -Infinity]);
for (const point of data.points) {
  byPosition[point.position].push(point);
  point.coordinates.forEach((value, dimension) => {
    extents[dimension][0] = Math.min(extents[dimension][0], value);
    extents[dimension][1] = Math.max(extents[dimension][1], value);
  });
}
for (const range of extents) {
  const margin = range[0] === range[1] ? Math.max(1, Math.abs(range[0]) * .1)
                                    : (range[1] - range[0]) * .08;
  range[0] -= margin;
  range[1] += margin;
}
const color = token => `hsl(${(token * 137.508) % 360} 65% 43%)`;
const number = value => value === 0 ? '0' : Number(value.toPrecision(4)).toString();
const charts = [];
function draw(chart) {
  const {canvas, points, xDimension, yDimension} = chart;
  const width = canvas.getBoundingClientRect().width || 300;
  const height = 250;
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(height * ratio);
  const ctx = canvas.getContext('2d');
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  const left = 62, right = width - 18, top = 20, bottom = height - 44;
  const xr = extents[xDimension], yr = yDimension < data.model_width
      ? extents[yDimension] : [-1, 1];
  const x = value => left + (value - xr[0]) / (xr[1] - xr[0]) * (right - left);
  const y = value => bottom - (value - yr[0]) / (yr[1] - yr[0]) * (bottom - top);
  ctx.font = '10px system-ui';
  ctx.lineWidth = 1;
  for (let tick = 0; tick <= 4; ++tick) {
    const xv = xr[0] + tick / 4 * (xr[1] - xr[0]);
    const yv = yr[0] + tick / 4 * (yr[1] - yr[0]);
    ctx.strokeStyle = '#e6ebf2';
    ctx.beginPath();ctx.moveTo(x(xv), top);ctx.lineTo(x(xv), bottom);ctx.stroke();
    ctx.beginPath();ctx.moveTo(left, y(yv));ctx.lineTo(right, y(yv));ctx.stroke();
    ctx.fillStyle = '#637086';ctx.textAlign = 'center';
    ctx.fillText(number(xv), x(xv), bottom + 18);
    ctx.textAlign = 'right';ctx.fillText(number(yv), left - 7, y(yv) + 3);
  }
  ctx.fillStyle = '#35445b';ctx.textAlign = 'center';
  ctx.fillText(`coordinate ${xDimension + 1}`, (left + right) / 2, height - 6);
  chart.visible = [];
  // Draw padding first so scored points remain visible when projections overlap.
  const ordered = [...points].sort((a, b) => Number(b.padding) - Number(a.padding)
      || Number(a.target_token >= 0) - Number(b.target_token >= 0));
  for (const point of ordered) {
    if (point.padding && !paddingToggle.checked) continue;
    const px = x(point.coordinates[xDimension]);
    const py = y(yDimension < data.model_width ? point.coordinates[yDimension] : 0);
    ctx.globalAlpha = point.padding ? .32 : .82;
    ctx.fillStyle = point.padding ? '#a0a8b5' : point.target_token < 0
        ? '#7b899e' : color(point.target_token);
    ctx.beginPath();ctx.arc(px, py, point.padding ? 3 : 4, 0, Math.PI * 2);ctx.fill();
    chart.visible.push({point, px, py});
  }
  ctx.globalAlpha = 1;
}
for (let position = 0; position < data.context_length; ++position) {
  const points = byPosition[position];
  const padding = points.filter(point => point.padding).length;
  const scored = points.filter(point => point.target_token >= 0).length;
  const section = document.createElement('section');section.className = 'position';
  const title = document.createElement('h2');title.textContent = `Position ${position}`;
  const counts = document.createElement('p');counts.className = 'counts';
  counts.textContent = `${points.length} total / ${points.length - padding} valid / `
      + `${scored} scored / ${padding} padding (IDs and positions are zero-based)`;
  const plots = document.createElement('div');plots.className = 'plots';
  section.append(title, counts, plots);root.append(section);
  for (let dimension = 0; dimension < data.model_width; dimension += 2) {
    const figure = document.createElement('figure');
    const caption = document.createElement('figcaption');
    caption.textContent = dimension + 1 < data.model_width
        ? `Coordinates ${dimension + 1} and ${dimension + 2}`
        : `Coordinate ${dimension + 1} against zero`;
    const canvas = document.createElement('canvas');
    canvas.setAttribute('role', 'img');
    canvas.setAttribute('aria-label', `Position ${position}, ${caption.textContent}; `
        + `${points.length} total points including ${padding} padding points`);
    figure.append(caption, canvas);plots.append(figure);
    const chart = {canvas, points, xDimension: dimension, yDimension: dimension + 1};
    charts.push(chart);
    canvas.addEventListener('pointermove', event => {
      const rect = canvas.getBoundingClientRect();
      const mx = event.clientX - rect.left, my = event.clientY - rect.top;
      const hits = chart.visible.filter(hit => Math.hypot(hit.px - mx, hit.py - my) <= 9);
      if (!hits.length) {tooltip.style.display = 'none';return;}
      const point = hits[hits.length - 1].point;
      tooltip.textContent = `Fact ${point.fact_index}, position ${point.position}\n`
          + data.facts[point.fact_index] + '\n'
          + `Input token ID: ${point.input_token}\nTarget token ID: ${point.target_token}\n`
          + `Row: ${point.padding ? 'padding' : point.target_token < 0 ? 'masked prompt' : 'scored'}\n`
          + `Coordinates: [${point.coordinates.join(', ')}]`
          + (hits.length > 1 ? `\n${hits.length} points overlap near this location.` : '');
      tooltip.style.display = 'block';
      tooltip.style.left = Math.max(8, Math.min(event.clientX + 14,
          window.innerWidth - tooltip.offsetWidth - 8)) + 'px';
      tooltip.style.top = Math.max(8, Math.min(event.clientY + 14,
          window.innerHeight - tooltip.offsetHeight - 8)) + 'px';
    });
    canvas.addEventListener('pointerleave', () => {tooltip.style.display = 'none';});
  }
}
// Appending plots changes the auto-fit grid widths. Draw only after the final
// layout exists so canvas pixels and hover coordinates use the same CSS width.
charts.forEach(draw);
paddingToggle.addEventListener('change', () => {
  tooltip.style.display = 'none';charts.forEach(draw);
});
window.addEventListener('resize', () => charts.forEach(draw));
</script></body></html>
)HTML";

}  // namespace

absl::StatusOr<PuzzleSeparation> VerifyPuzzleSeparation(
    const PuzzleReportData& data) {
  const absl::Status valid = ValidatePuzzleReportData(data);
  if (!valid.ok())
    return valid;
  absl::flat_hash_map<std::vector<float>, const PuzzlePoint*, CoordinatesHash>
      representatives;
  absl::flat_hash_set<int> targets;
  PuzzleSeparation result;
  for (const PuzzlePoint& point : data.points) {
    if (point.target_token < 0)
      continue;
    ++result.scored_points;
    targets.insert(point.target_token);
    const auto [found, inserted] =
        representatives.emplace(point.coordinates, &point);
    if (!inserted && found->second->target_token != point.target_token) {
      const PuzzlePoint& prior = *found->second;
      return absl::FailedPreconditionError(absl::StrCat(
          "puzzle residual collision: fact ", prior.fact_index, ", position ",
          prior.position, " (target ", prior.target_token, ") and fact ",
          point.fact_index, ", position ", point.position, " (target ",
          point.target_token, ") have the same full residual vector"));
    }
  }
  result.unique_vectors = static_cast<int64_t>(representatives.size());
  result.distinct_targets = static_cast<int64_t>(targets.size());
  return result;
}

absl::Status WritePuzzleHtml(const PuzzleReportData& data,
                             const PuzzleSeparation& separation,
                             absl::string_view path) {
  const auto verified = VerifyPuzzleSeparation(data);
  if (!verified.ok())
    return verified.status();
  if (verified->scored_points != separation.scored_points ||
      verified->unique_vectors != separation.unique_vectors ||
      verified->distinct_targets != separation.distinct_targets)
    return absl::InvalidArgumentError(
        "puzzle separation counts do not match the report data");
  if (path.empty() || path.find('\0') != absl::string_view::npos)
    return absl::InvalidArgumentError(
        "puzzle HTML path is empty or contains NUL");
  std::ofstream output(std::string(path), std::ios::binary | std::ios::trunc);
  if (!output)
    return absl::InternalError(absl::StrCat("cannot open puzzle HTML: ", path));
  output.imbue(std::locale::classic());
  output
      << std::setprecision(std::numeric_limits<float>::max_digits10)
      << kPageStart << separation.scored_points << " scored points / "
      << separation.unique_vectors << " distinct full residual vectors / "
      << separation.distinct_targets << " distinct target token IDs. "
      << "Equal vectors have equal target IDs across all positions.</div>\n"
      << "<p class=\"note\">The five-token prompt mask and padding are "
         "excluded "
         "from the audit: only rows with target token ID &ge; 0 are scored. "
         "This audit does not establish unambiguous next-token prediction "
         "for the full prefix, and does not test MLP learnability.</p>\n"
      << "<div class=\"controls\"><label><input id=\"show-padding\" "
         "type=\"checkbox\" checked> Show padding</label>"
      << "<div class=\"legend\"><span class=\"key\" style=\"--color:#3b83b8\">"
         "Scored: color by target ID</span><span class=\"key\" "
         "style=\"--color:#7b899e\">Masked prompt</span>"
      << "<span class=\"key\" style=\"--color:#a0a8b552\">Padding</span>"
         "</div></div><p class=\"note\">Hover over a point for fact text, "
         "token IDs, and all coordinates.</p>\n"
      << "<noscript>Enable JavaScript to display the coordinate plots. "
         "All data is embedded in this file.</noscript>"
      << "<div id=\"positions\"></div></main><div id=\"tooltip\"></div>\n"
      << "<script>\n'use strict';\nconst data = ";
  WriteData(output, data);
  output << kPageScript;
  output.close();
  if (!output)
    return absl::InternalError(
        absl::StrCat("cannot write puzzle HTML: ", path));
  return absl::OkStatus();
}

}  // namespace pluto::llm::memorize_general_facts
