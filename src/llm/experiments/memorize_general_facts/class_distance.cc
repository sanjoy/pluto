#include "src/llm/experiments/memorize_general_facts/class_distance.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <locale>
#include <numeric>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/escaping.h"
#include "absl/strings/str_cat.h"
#include "src/util/status_macros.h"

namespace pluto::llm::memorize_general_facts {
namespace {

// One observed target class and its scored state locations in capture order.
// The outer class list is sorted by ID to make pair order deterministic.
struct TargetClass {
  int token;  // Model-vocabulary target ID; prompt-only tokens are absent.
  std::vector<size_t> points;  // Indices into PuzzleReportData::points.
};

// Prevent cancellation near coincident large vectors by subtracting first.
// Captured float values are exact in double; even float-max differences fit.
double SquaredDistance(const PuzzlePoint& a, const PuzzlePoint& b) {
  double squared = 0;
  for (size_t i = 0; i < a.coordinates.size(); ++i) {
    const double difference =
        static_cast<double>(a.coordinates[i]) - b.coordinates[i];
    squared += difference * difference;
  }
  return squared;
}

// Escape arbitrary corpus/token text before embedding it as HTML text.
// Token labels have separately been C-escaped to show spaces and control bytes.
std::string Html(absl::string_view text) {
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
      case '"':
        result += "&quot;";
        break;
      case '\'':
        result += "&#39;";
        break;
      default:
        result += c;
    }
  }
  return result;
}

// Use linear interpolation between neighboring sorted observations.
// The caller only asks for quantiles when at least one class pair exists.
double Quantile(absl::Span<const double> sorted, double probability) {
  const double index = probability * (sorted.size() - 1);
  const size_t lower = static_cast<size_t>(index);
  const size_t upper = std::min(lower + 1, sorted.size() - 1);
  return sorted[lower] + (sorted[upper] - sorted[lower]) * (index - lower);
}

// Render fixed-width bins as standalone SVG; hover titles expose exact counts.
// Log-distance bins exclude zeros explicitly, which are reported separately.
void WriteHistogram(std::ostream& output, absl::Span<const double> sorted,
                    bool logarithmic) {
  constexpr size_t kBins = 60;
  const auto begin = logarithmic
                         ? std::upper_bound(sorted.begin(), sorted.end(), 0.0)
                         : sorted.begin();
  output << "<section><h2>"
         << (logarithmic ? "Log-distance histogram"
                         : "Linear-distance histogram")
         << "</h2>";
  if (begin == sorted.end()) {
    output << "<p>No " << (logarithmic ? "positive " : "")
           << "class-pair distances to plot.</p></section>";
    return;
  }
  const auto transform = [logarithmic](double value) {
    return logarithmic ? std::log10(value) : value;
  };
  double low = transform(*begin);
  double high = transform(sorted.back());
  if (low == high) {
    // A constant distribution needs a nonzero display range, not division by 0.
    const double padding = std::max(0.5, std::abs(low) * 0.05);
    low -= padding;
    high += padding;
    if (!logarithmic)
      low = std::max(0.0, low);
  }
  std::array<size_t, kBins> counts{};
  for (auto it = begin; it != sorted.end(); ++it) {
    const double fraction =
        std::clamp((transform(*it) - low) / (high - low), 0.0, 1.0);
    const size_t bin =
        std::min(kBins - 1, static_cast<size_t>(fraction * kBins));
    ++counts[bin];
  }
  const size_t peak = *std::max_element(counts.begin(), counts.end());
  output << "<p>" << (sorted.end() - begin)
         << " unordered class pairs; count axis is linear. "
         << (logarithmic ? "Zero distances are excluded from this panel." : "")
         << "</p><svg viewBox=\"0 0 1060 380\" role=\"img\" aria-label=\""
         << (logarithmic ? "Log" : "Linear")
         << " distance histogram\"><path d=\"M70 25V310H1030\" "
            "stroke=\"#58677b\" fill=\"none\"/>";
  for (size_t bin = 0; bin < kBins; ++bin) {
    const double height = 275.0 * counts[bin] / peak;
    const double left = low + (high - low) * bin / kBins;
    const double right = low + (high - low) * (bin + 1) / kBins;
    output << "<rect x=\"" << 70 + bin * 16 << "\" y=\"" << 310 - height
           << "\" width=\"15\" height=\"" << height
           << "\" fill=\"#397db3\"><title>["
           << (logarithmic ? std::pow(10.0, left) : left) << ", "
           << (logarithmic ? std::pow(10.0, right) : right)
           << (bin + 1 == kBins ? "]: " : "): ") << counts[bin]
           << " pairs</title></rect>";
  }
  for (int tick = 0; tick <= 5; ++tick) {
    const double value = low + (high - low) * tick / 5;
    output << "<text x=\"" << 70 + tick * 192
           << "\" y=\"333\" text-anchor=\"middle\">"
           << (logarithmic ? std::pow(10.0, value) : value) << "</text>";
  }
  output << "<text x=\"62\" y=\"35\" text-anchor=\"end\">" << peak
         << "</text><text x=\"62\" y=\"311\" text-anchor=\"end\">0</text>"
         << "<text x=\"550\" y=\"370\" text-anchor=\"middle\">"
            "Minimum Euclidean L2 distance between classes"
         << (logarithmic ? " (logarithmic spacing)" : "")
         << "</text></svg></section>";
}

// Print the target's model ID and optional decoded token without HTML markup.
// Explicit quotes distinguish leading/trailing whitespace from formatting.
void WriteToken(std::ostream& output, int token,
                absl::Span<const std::string> labels) {
  output << token;
  if (!labels.empty())
    output << " <code>&quot;" << Html(absl::CEscape(labels[token]))
           << "&quot;</code>";
}

// Show a concrete vector pair witnessing the minimum, rather than centroids.
// Positions are one-based for reading, with the exact next-token role stated.
void WriteWitness(std::ostream& output, const PuzzleReportData& data,
                  size_t point_index) {
  const PuzzlePoint& point = data.points[point_index];
  output << "fact " << point.fact_index + 1 << ", input position "
         << point.position + 1 << "<br>" << Html(data.facts[point.fact_index])
         << "<details><summary>Full A3 vector</summary><code>[";
  for (size_t i = 0; i < point.coordinates.size(); ++i) {
    if (i)
      output << ", ";
    output << std::setprecision(std::numeric_limits<float>::max_digits10)
           << point.coordinates[i];
  }
  output << "]</code></details>" << std::setprecision(8);
}

}  // namespace

// Group only scored rows, then exhaust every cross-class state pair once.
// Taking square roots after minimization avoids millions of unnecessary roots.
absl::StatusOr<ClassDistanceAnalysis> ComputeClassDistances(
    const PuzzleReportData& data) {
  RETURN_IF_ERROR(ValidatePuzzleReportData(data));
  absl::flat_hash_map<int, std::vector<size_t>> grouped;
  ClassDistanceAnalysis result;
  for (size_t i = 0; i < data.points.size(); ++i) {
    const PuzzlePoint& point = data.points[i];
    if (point.target_token < 0)
      continue;
    grouped[point.target_token].push_back(i);
    ++result.scored_points;
  }
  std::vector<TargetClass> classes;
  classes.reserve(grouped.size());
  for (auto& [token, points] : grouped)
    classes.push_back({token, std::move(points)});
  std::sort(classes.begin(), classes.end(),
            [](const auto& a, const auto& b) { return a.token < b.token; });
  result.target_classes = classes.size();
  const size_t count = classes.size();
  if (count > 1 && count > std::numeric_limits<size_t>::max() / (count - 1))
    return absl::InvalidArgumentError("class pair count overflows");
  result.pairs.reserve(count == 0 ? 0 : count * (count - 1) / 2);
  for (size_t first = 0; first < count; ++first) {
    for (size_t second = first + 1; second < count; ++second) {
      double minimum_squared = std::numeric_limits<double>::infinity();
      ClassDistancePair pair{.first_token = classes[first].token,
                             .second_token = classes[second].token,
                             .distance = 0,
                             .first_point = 0,
                             .second_point = 0};
      for (size_t a : classes[first].points) {
        for (size_t b : classes[second].points) {
          const double squared =
              SquaredDistance(data.points[a], data.points[b]);
          if (squared < minimum_squared) {
            minimum_squared = squared;
            pair.first_point = a;
            pair.second_point = b;
          }
        }
      }
      pair.distance = std::sqrt(minimum_squared);
      result.pairs.push_back(pair);
    }
  }
  return result;
}

// Compute the exact class-pair distribution and render a portable HTML report.
// The full pair list stays in memory; only bins, summaries and examples go out.
absl::Status WriteClassDistanceHtml(const PuzzleReportData& data,
                                    absl::Span<const std::string> token_labels,
                                    absl::string_view path) {
  if (path.empty() || path.find('\0') != absl::string_view::npos)
    return absl::InvalidArgumentError("class distance HTML path is invalid");
  ASSIGN_OR_RETURN(auto analysis, ComputeClassDistances(data));
  for (const auto& pair : analysis.pairs)
    if (!token_labels.empty() &&
        static_cast<size_t>(pair.second_token) >= token_labels.size())
      return absl::InvalidArgumentError(
          "class distance token label is missing");
  std::vector<double> distances;
  distances.reserve(analysis.pairs.size());
  for (const auto& pair : analysis.pairs)
    distances.push_back(pair.distance);
  std::sort(distances.begin(), distances.end());
  std::ofstream output(std::string(path), std::ios::binary | std::ios::trunc);
  if (!output)
    return absl::InternalError(
        absl::StrCat("cannot open class distance HTML: ", path));
  output.imbue(std::locale::classic());
  output << std::setprecision(8) << R"HTML(<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>A3 minimum distances between output-token classes</title><style>
body{font:15px/1.5 system-ui,sans-serif;color:#172b44;background:#f4f7fb;margin:0;padding:30px}
main{max-width:1200px;margin:auto}section{background:white;border:1px solid #d5dfeb;border-radius:10px;padding:20px;margin:20px 0}
h1{line-height:1.2}h2{font-size:21px}svg{width:100%;height:auto}svg text{font:12px system-ui}
table{border-collapse:collapse;width:100%}th,td{text-align:left;vertical-align:top;padding:8px;border-bottom:1px solid #d5dfeb}
code{white-space:pre-wrap;overflow-wrap:anywhere}details{max-width:400px}.summary{font-size:17px}.table-wrap{overflow:auto}
</style></head><body><main><h1>A3 minimum distances between output-token classes</h1>
<p>For each unordered pair of distinct expected next-token IDs (x, y), compute
<code>min ||a - b||₂</code> over every scored A3 state a targeting x and every scored A3 state b targeting y.
Each class pair contributes exactly one observation. These are not centroid distances, not every state-pair distance,
and not one nearest-other-class distance per token.</p>
<p>All scored suffix and terminal EOS positions are pooled across all facts. Masked prompt and padding rows are excluded.
Coordinates are the full post-attention-3 residual, before its MLP, with no normalization or projection.
BF16 capture values are expanded exactly; differences and squared sums are evaluated in FP64, followed by a square root.</p>
)HTML";
  output << "<p class=\"summary\">Scored states: " << analysis.scored_points
         << "; observed target classes: " << analysis.target_classes
         << "; unordered class pairs: " << analysis.pairs.size()
         << "; dimensions: " << data.model_width << ".</p>";
  if (!distances.empty()) {
    const auto zeros =
        std::upper_bound(distances.begin(), distances.end(), 0.0) -
        distances.begin();
    output << "<section><h2>Summary</h2><p>Zero-distance class pairs: " << zeros
           << ".</p><table><tr><th>Statistic</th><th>L2 distance</th></tr>";
    const std::array<std::pair<absl::string_view, double>, 9> quantiles{
        {{"Minimum", 0},
         {"1st percentile", .01},
         {"5th percentile", .05},
         {"25th percentile", .25},
         {"Median", .5},
         {"75th percentile", .75},
         {"95th percentile", .95},
         {"99th percentile", .99},
         {"Maximum", 1}}};
    for (const auto& [name, probability] : quantiles)
      output << "<tr><td>" << name << "</td><td>" << std::setprecision(17)
             << Quantile(distances, probability) << "</td></tr>";
    output << "</table><p>Percentiles use linear interpolation of sorted pair "
              "minima.</p></section>"
           << std::setprecision(8);
  }
  WriteHistogram(output, distances, false);
  WriteHistogram(output, distances, true);
  const size_t closest_count = std::min<size_t>(50, analysis.pairs.size());
  std::partial_sort(analysis.pairs.begin(),
                    analysis.pairs.begin() + closest_count,
                    analysis.pairs.end(), [](const auto& a, const auto& b) {
                      if (a.distance != b.distance)
                        return a.distance < b.distance;
                      if (a.first_token != b.first_token)
                        return a.first_token < b.first_token;
                      return a.second_token < b.second_token;
                    });
  output << "<section><h2>Closest " << closest_count
         << " class pairs</h2><p>IDs use the model's vocabulary. Fact and "
            "input-position numbers are one-based;"
            " each state's label is the next token. Ties use the first witness "
            "in capture order.</p>"
            "<div class=\"table-wrap\"><table><tr><th>First "
            "target</th><th>Second target</th>"
            "<th>Minimum L2</th><th>First witness</th><th>Second "
            "witness</th></tr>";
  for (size_t i = 0; i < closest_count; ++i) {
    const auto& pair = analysis.pairs[i];
    output << "<tr><td>";
    WriteToken(output, pair.first_token, token_labels);
    output << "</td><td>";
    WriteToken(output, pair.second_token, token_labels);
    output << "</td><td>" << std::setprecision(17) << pair.distance
           << std::setprecision(8) << "</td><td>";
    WriteWitness(output, data, pair.first_point);
    output << "</td><td>";
    WriteWitness(output, data, pair.second_point);
    output << "</td></tr>";
  }
  output << "</table></div></section><p>This is descriptive geometry on a "
            "finite corpus, not a proof of linear"
            " or small-MLP separability. Class frequency affects the number of "
            "candidate state pairs;"
            " common classes have more opportunities for a small minimum. No "
            "external assets or JavaScript are required."
            "</p></main></body></html>";
  output.close();
  if (!output)
    return absl::InternalError(
        absl::StrCat("cannot write class distance HTML: ", path));
  return absl::OkStatus();
}

}  // namespace pluto::llm::memorize_general_facts
