#include "src/llm/experiments/one_shot_memorizer/sentence_ablation_report.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <string>
#include <system_error>

#include "src/util/status_macros.h"

namespace pluto::llm::one_shot_memorizer {
namespace {

std::string EscapeHtml(const std::string& text) {
  std::string result;
  for (char ch : text)
    switch (ch) {
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
        result += ch;
    }
  return result;
}

// TSV names use visible backslash escapes so arbitrary layer names cannot add
// columns or records. Numeric fields remain directly parseable as numbers.
std::string EscapeTsv(const std::string& text) {
  std::string result;
  for (char ch : text)
    switch (ch) {
      case '\\':
        result += "\\\\";
        break;
      case '\t':
        result += "\\t";
        break;
      case '\n':
        result += "\\n";
        break;
      case '\r':
        result += "\\r";
        break;
      default:
        result += ch;
    }
  return result;
}

std::string Shape(const TensorSpec& tensor) {
  std::string result;
  for (size_t dim : tensor.shape) {
    if (!result.empty())
      result += "x";
    result += std::to_string(dim);
  }
  return result;
}

bool Near(double left, double right) {
  if (!std::isfinite(left) || !std::isfinite(right))
    return false;
  return std::abs(left - right) <=
         1e-10 * std::max({1.0, std::abs(left), std::abs(right)});
}

absl::Status ValidateSummary(const DeltaSummary& summary) {
  if (summary.bitwise_changed_count > summary.element_count ||
      summary.numerically_changed_count > summary.bitwise_changed_count)
    return absl::InvalidArgumentError("inconsistent ablation summary counts");
  for (double value :
       {summary.baseline_l2, summary.ablated_l2, summary.delta_l1,
        summary.delta_l2, summary.maximum_absolute_delta})
    if (!std::isfinite(value) || value < 0)
      return absl::InvalidArgumentError("invalid ablation summary norm");
  if (summary.baseline_l2 == 0) {
    if (summary.relative_l2)
      return absl::InvalidArgumentError(
          "relative L2 is undefined for zero baseline");
  } else if (!summary.relative_l2 || !std::isfinite(*summary.relative_l2) ||
             !Near(*summary.relative_l2,
                   summary.delta_l2 / summary.baseline_l2)) {
    return absl::InvalidArgumentError("inconsistent relative ablation norm");
  }
  return absl::OkStatus();
}

bool SameCoordinate(const ParameterCoordinate& left,
                    const ParameterCoordinate& right) {
  return left.checkpoint_index == right.checkpoint_index &&
         left.element_index == right.element_index &&
         left.flat_index == right.flat_index && left.row == right.row &&
         left.column == right.column &&
         left.compact_token_id == right.compact_token_id &&
         left.qkv_component == right.qkv_component &&
         left.qkv_channel == right.qkv_channel;
}

absl::Status ValidateReport(const ParameterDeltaReport& report) {
  RETURN_IF_ERROR(ValidateSummary(report.total));
  if (report.tensors.empty())
    return absl::InvalidArgumentError("ablation report has no tensors");
  size_t offset = 0, changed = 0, numeric = 0;
  double total_l1 = 0, total_squared = 0, total_max = 0;
  for (size_t index = 0; index < report.tensors.size(); ++index) {
    const auto& tensor = report.tensors[index];
    // LocateParameter also validates rank, shape, offsets and semantic axes.
    RETURN_IF_ERROR(LocateParameter(tensor.tensor, 0).status());
    RETURN_IF_ERROR(ValidateSummary(tensor.summary));
    if (tensor.tensor.checkpoint_index != index ||
        tensor.tensor.flat_offset != offset ||
        tensor.summary.element_count != tensor.tensor.element_count ||
        tensor.deltas.size() != tensor.tensor.element_count ||
        tensor.bitwise_changed.size() != tensor.tensor.element_count)
      return absl::InvalidArgumentError(
          "malformed ablation tensor arrays or order");
    size_t tensor_changed = 0, tensor_numeric = 0;
    double l1 = 0, squared = 0, maximum = 0;
    for (size_t element = 0; element < tensor.deltas.size(); ++element) {
      const double delta = tensor.deltas[element];
      if (!std::isfinite(delta) ||
          (delta != 0 && !tensor.bitwise_changed[element]))
        return absl::InvalidArgumentError("invalid ablation coordinate delta");
      tensor_changed += tensor.bitwise_changed[element];
      tensor_numeric += delta != 0;
      l1 += std::abs(delta);
      squared += delta * delta;
      maximum = std::max(maximum, std::abs(delta));
    }
    if (tensor_changed != tensor.summary.bitwise_changed_count ||
        tensor_numeric != tensor.summary.numerically_changed_count ||
        !Near(l1, tensor.summary.delta_l1) ||
        !Near(std::sqrt(squared), tensor.summary.delta_l2) ||
        !Near(maximum, tensor.summary.maximum_absolute_delta))
      return absl::InvalidArgumentError(
          "ablation tensor summary disagrees with deltas");
    double previous_absolute = std::numeric_limits<double>::infinity();
    size_t previous_element = 0;
    for (size_t top = 0; top < tensor.top_coordinates.size(); ++top) {
      const auto& entry = tensor.top_coordinates[top];
      ASSIGN_OR_RETURN(
          const auto coordinate,
          LocateParameter(tensor.tensor, entry.coordinate.element_index));
      const size_t element = coordinate.element_index;
      const double absolute = std::abs(entry.delta);
      if (!SameCoordinate(coordinate, entry.coordinate) ||
          !std::isfinite(entry.baseline) || !std::isfinite(entry.ablated) ||
          !tensor.bitwise_changed[element] ||
          entry.delta != tensor.deltas[element] ||
          entry.delta != static_cast<double>(entry.baseline) - entry.ablated ||
          absolute > previous_absolute ||
          (top != 0 && absolute == previous_absolute &&
           element <= previous_element))
        return absl::InvalidArgumentError("malformed ablation top coordinates");
      previous_absolute = absolute;
      previous_element = element;
    }
    offset += tensor.tensor.element_count;
    changed += tensor_changed;
    numeric += tensor_numeric;
    total_l1 += l1;
    total_squared += squared;
    total_max = std::max(total_max, maximum);
  }
  if (report.total.element_count != offset ||
      report.total.bitwise_changed_count != changed ||
      report.total.numerically_changed_count != numeric ||
      !Near(report.total.delta_l1, total_l1) ||
      !Near(report.total.delta_l2, std::sqrt(total_squared)) ||
      !Near(report.total.maximum_absolute_delta, total_max))
    return absl::InvalidArgumentError(
        "ablation total disagrees with tensor summaries");
  return absl::OkStatus();
}

void WriteSummaryTsv(std::ostream& out, const DeltaSummary& summary) {
  out << summary.element_count << '\t' << summary.bitwise_changed_count << '\t'
      << summary.numerically_changed_count << '\t' << summary.baseline_l2
      << '\t' << summary.ablated_l2 << '\t' << summary.delta_l1 << '\t'
      << summary.delta_l2 << '\t' << summary.maximum_absolute_delta << '\t';
  if (summary.relative_l2)
    out << *summary.relative_l2;
  out << '\n';
}

void WriteSummaryHtml(std::ostream& out, const DeltaSummary& summary) {
  out << "<td>" << summary.element_count << "</td><td>"
      << summary.bitwise_changed_count << "</td><td>"
      << summary.numerically_changed_count << "</td><td>" << summary.delta_l1
      << "</td><td>" << summary.delta_l2 << "</td><td>"
      << summary.maximum_absolute_delta << "</td><td>";
  if (summary.relative_l2)
    out << *summary.relative_l2;
  else
    out << "undefined";
  out << "</td>";
}

void WriteCoordinateTsv(std::ostream& out,
                        const ParameterCoordinate& coordinate) {
  out << coordinate.checkpoint_index << '\t' << coordinate.element_index << '\t'
      << coordinate.flat_index << '\t' << coordinate.row << '\t';
  if (coordinate.column)
    out << *coordinate.column;
  out << '\t';
  if (coordinate.compact_token_id)
    out << *coordinate.compact_token_id;
  out << '\t' << QkvComponentName(coordinate.qkv_component) << '\t';
  if (coordinate.qkv_channel)
    out << *coordinate.qkv_channel;
}

void WriteCoordinateHtml(std::ostream& out, const CoordinateDelta& entry) {
  const auto& coordinate = entry.coordinate;
  out << "<tr><td>" << coordinate.element_index << "</td><td>"
      << coordinate.flat_index << "</td><td>" << coordinate.row << "</td><td>";
  if (coordinate.column)
    out << *coordinate.column;
  out << "</td><td>";
  if (coordinate.compact_token_id)
    out << *coordinate.compact_token_id;
  out << "</td><td>" << QkvComponentName(coordinate.qkv_component)
      << "</td><td>";
  if (coordinate.qkv_channel)
    out << *coordinate.qkv_channel;
  out << "</td><td>" << entry.baseline << "</td><td>" << entry.ablated
      << "</td><td>" << entry.delta << "</td></tr>\n";
}

}  // namespace

absl::Status WriteSentenceAblationReport(
    const ParameterDeltaReport& report,
    const std::filesystem::path& output_directory) {
  RETURN_IF_ERROR(ValidateReport(report));
  std::error_code error;
  if (output_directory.empty() ||
      !std::filesystem::is_directory(output_directory, error))
    return absl::FailedPreconditionError(
        "ablation report output must be an existing directory");
  if (error)
    return absl::InternalError("cannot inspect ablation output directory: " +
                               error.message());
  // Never follow an existing output symlink or replace a directory. Validate
  // all fixed destinations before truncating any earlier report artifact.
  for (const char* name :
       {"per_tensor.tsv", "coordinates.tsv", "report.html"}) {
    const auto status =
        std::filesystem::symlink_status(output_directory / name, error);
    if (error && error != std::errc::no_such_file_or_directory)
      return absl::InternalError("cannot inspect ablation report output: " +
                                 error.message());
    error.clear();
    if (std::filesystem::exists(status) &&
        !std::filesystem::is_regular_file(status))
      return absl::FailedPreconditionError(
          "ablation report output is not a regular file: " + std::string(name));
  }
  std::ofstream table(output_directory / "per_tensor.tsv", std::ios::trunc);
  std::ofstream coordinates(output_directory / "coordinates.tsv",
                            std::ios::trunc);
  std::ofstream html(output_directory / "report.html", std::ios::trunc);
  if (!table || !coordinates || !html)
    return absl::InternalError("cannot open ablation report output files");
  for (std::ostream* stream : {&table, &coordinates, &html})
    *stream << std::setprecision(std::numeric_limits<double>::max_digits10);
  table << "# delta = baseline minus ablated; names use backslash escapes\n"
        << "checkpoint_index\ttensor\tshape\telements\tbitwise_"
           "changed\tnumerically_changed"
           "\tbaseline_l2\tablated_l2\tdelta_l1\tdelta_l2\tmax_abs_"
           "delta\trelative_l2\n";
  coordinates
      << "# Every bitwise-changed scalar; delta = baseline minus ablated.\n"
         "# Baseline/ablated values are not retained for all coordinates; "
         "signed-zero"
         " differences remain present with delta zero.\n"
         "tensor\tcheckpoint_index\telement_index\tflat_index\trow\tcolumn"
         "\tcompact_token_id\tqkv_component\tqkv_channel\tdelta\tbitwise_"
         "changed\n";
  html << "<!doctype html><html lang=\"en\"><meta charset=\"utf-8\">"
          "<meta name=\"viewport\" "
          "content=\"width=device-width,initial-scale=1\">"
          "<title>Sentence ablation: parameter differences</title><style>"
          "body{font:15px system-ui,sans-serif;max-width:1400px;margin:2rem "
          "auto;padding:0 1rem}"
          "table{border-collapse:collapse;margin:1rem "
          "0;font-variant-numeric:tabular-nums}"
          "th,td{border:1px solid "
          "#ccc;padding:.4rem;text-align:right}th:first-child,td:first-child"
          "{text-align:left}thead{background:#eee}details{margin:1rem "
          "0}code{white-space:pre-wrap}"
          "</style><body><h1>Sentence ablation: parameter differences</h1>"
          "<p>Every delta is <strong>baseline minus ablated</strong>. These "
          "are matched-run "
          "differences, not evidence of exclusive ownership of a fact. Shared "
          "computation, "
          "optimizer history and interactions with other sentences can spread "
          "each effect.</p>"
          "<p>Bitwise changes include +0 versus -0. Relative L2 is undefined "
          "when the baseline "
          "norm is zero. Coordinates use physical checkpoint layout; compact "
          "token IDs and "
          "Q/K/V channels are shown only where applicable.</p>"
          "<table><thead><tr><th>Tensor</th><th>Shape</th><th>Elements</"
          "th><th>Bitwise changed</th>"
          "<th>Numerically changed</th><th>Delta L1</th><th>Delta "
          "L2</th><th>Max |delta|</th>"
          "<th>Relative L2</th></tr></thead><tbody>";
  html << "<tr><td>All unique tensors</td><td></td>";
  WriteSummaryHtml(html, report.total);
  html << "</tr>\n";
  for (const auto& tensor : report.tensors) {
    table << tensor.tensor.checkpoint_index << '\t'
          << EscapeTsv(tensor.tensor.name) << '\t' << Shape(tensor.tensor)
          << '\t';
    WriteSummaryTsv(table, tensor.summary);
    html << "<tr><td>weight_" << tensor.tensor.checkpoint_index << ": "
         << EscapeHtml(tensor.tensor.name) << "</td><td>"
         << Shape(tensor.tensor) << "</td>";
    WriteSummaryHtml(html, tensor.summary);
    html << "</tr>\n";
    for (size_t element = 0; element < tensor.deltas.size(); ++element) {
      if (!tensor.bitwise_changed[element])
        continue;
      ASSIGN_OR_RETURN(const auto coordinate,
                       LocateParameter(tensor.tensor, element));
      coordinates << EscapeTsv(tensor.tensor.name) << '\t';
      WriteCoordinateTsv(coordinates, coordinate);
      coordinates << '\t' << tensor.deltas[element] << "\t1\n";
    }
  }
  html << "</tbody></table><h2>Largest supplied changes per tensor</h2>"
          "<p>Up to 20 supplied top coordinates are shown per tensor, ordered "
          "by absolute "
          "delta. The companion coordinates.tsv contains every bitwise-changed "
          "coordinate, "
          "including any omitted from these short lists.</p>";
  for (const auto& tensor : report.tensors) {
    html << "<details><summary>weight_" << tensor.tensor.checkpoint_index
         << ": " << EscapeHtml(tensor.tensor.name)
         << "</summary><table><thead><tr>"
            "<th>Element</th><th>Flat "
            "index</th><th>Row</th><th>Column</th><th>Compact token</th>"
            "<th>QKV "
            "part</th><th>Channel</th><th>Baseline</th><th>Ablated</"
            "th><th>Delta</th>"
            "</tr></thead><tbody>";
    for (size_t top = 0;
         top < std::min(size_t{20}, tensor.top_coordinates.size()); ++top)
      WriteCoordinateHtml(html, tensor.top_coordinates[top]);
    html << "</tbody></table></details>\n";
  }
  html << "</body></html>\n";
  for (std::ofstream* stream : {&table, &coordinates, &html}) {
    stream->flush();
    stream->close();
    if (!*stream)
      return absl::InternalError("failed while writing ablation report");
  }
  return absl::OkStatus();
}

}  // namespace pluto::llm::one_shot_memorizer
