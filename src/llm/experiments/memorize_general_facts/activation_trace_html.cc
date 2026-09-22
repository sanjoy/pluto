#include "src/llm/experiments/memorize_general_facts/activation_trace_html.h"

#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"

namespace pluto::llm::memorize_general_facts {
namespace {

constexpr int kWidth = 16;

// to_chars is locale-independent: decimal commas would corrupt SVG coordinates.
std::string Number(double value, int precision = 6) {
  char buffer[64];
  const auto result = std::to_chars(buffer, buffer + sizeof(buffer), value,
                                    std::chars_format::general, precision);
  return std::string(buffer, result.ptr);
}

std::string RawNumber(float value) {
  return Number(value, std::numeric_limits<float>::max_digits10);
}

void EscapeByte(unsigned char byte, std::string& output) {
  constexpr char kHex[] = "0123456789ABCDEF";
  output += "\\x";
  output += kHex[byte >> 4];
  output += kHex[byte & 15];
}

// Tokens are byte strings, not necessarily complete UTF-8 characters. Preserve
// valid Unicode but show each invalid byte explicitly, rather than allowing the
// browser to silently replace it. Text is then safe in both HTML and SVG text.
std::string EscapeText(absl::string_view text, bool visible_whitespace = true) {
  std::string output;
  for (size_t i = 0; i < text.size();) {
    const unsigned char byte = text[i];
    if (byte < 0x80) {
      ++i;
      switch (byte) {
        case '&':
          output += "&amp;";
          break;
        case '<':
          output += "&lt;";
          break;
        case '>':
          output += "&gt;";
          break;
        case '"':
          output += "&quot;";
          break;
        case '\'':
          output += "&#39;";
          break;
        case '\\':
          output += "\\\\";
          break;
        case '\n':
          output += visible_whitespace ? "\\n" : "\n";
          break;
        case '\r':
          output += visible_whitespace ? "\\r" : "\r";
          break;
        case '\t':
          output += visible_whitespace ? "\\t" : "\t";
          break;
        default:
          if (byte < 0x20 || byte == 0x7f)
            EscapeByte(byte, output);
          else
            output += static_cast<char>(byte);
      }
      continue;
    }
    size_t length = byte >= 0xc2 && byte <= 0xdf   ? 2
                    : byte >= 0xe0 && byte <= 0xef ? 3
                    : byte >= 0xf0 && byte <= 0xf4 ? 4
                                                   : 0;
    bool valid = length != 0 && length <= text.size() - i;
    if (valid) {
      for (size_t j = 1; j < length; ++j)
        valid &= (static_cast<unsigned char>(text[i + j]) & 0xc0) == 0x80;
      const unsigned char second = text[i + 1];
      // Reject overlong encodings, surrogates, and values beyond U+10FFFF.
      valid &= !(byte == 0xe0 && second < 0xa0);
      valid &= !(byte == 0xed && second >= 0xa0);
      valid &= !(byte == 0xf0 && second < 0x90);
      valid &= !(byte == 0xf4 && second >= 0x90);
    }
    if (!valid) {
      EscapeByte(byte, output);
      ++i;
    } else {
      output.append(text.data() + i, length);
      i += length;
    }
  }
  return output;
}

absl::Status ValidateTrace(const ActivationTrace& trace) {
  if (trace.model_width != kWidth)
    return absl::InvalidArgumentError(
        "activation HTML tracing requires model_width=16");
  if (trace.token_ids.empty())
    return absl::InvalidArgumentError("activation trace contains no tokens");
  if (trace.token_ids.size() != trace.token_texts.size())
    return absl::InvalidArgumentError("activation trace token arrays differ");
  if (trace.prompt_token_count > trace.token_ids.size())
    return absl::InvalidArgumentError(
        "activation trace prompt extends beyond its tokens");
  if (trace.token_ids.size() > std::numeric_limits<size_t>::max() / kWidth)
    return absl::InvalidArgumentError("activation trace size overflows");
  if (trace.boundaries.empty())
    return absl::InvalidArgumentError(
        "activation trace contains no boundaries");
  for (int token : trace.token_ids)
    if (token < 0)
      return absl::InvalidArgumentError(
          "activation trace has a negative token");
  for (const auto& boundary : trace.boundaries) {
    if (boundary.values.size() != trace.token_ids.size() * kWidth)
      return absl::InvalidArgumentError(
          "activation trace boundary has the wrong number of values");
    for (float value : boundary.values)
      if (!std::isfinite(value))
        return absl::InvalidArgumentError(
            "activation trace contains a non-finite value");
  }
  return absl::OkStatus();
}

// Eight local coordinate systems, not eight independently rescaled vectors.
// All share the same range, so arrow lengths retain their actual relationship.
void WriteVector(absl::Span<const float> values, double scale,
                 std::ostream& output) {
  output << "<div class=\"planes\">\n";
  for (int pair = 0; pair < kWidth / 2; ++pair) {
    const int x_dimension = pair * 2;
    const float x = values[x_dimension];
    const float y = values[x_dimension + 1];
    // SVG y increases downward, opposite to the mathematical y axis.
    const double end_x = 60 + 40 * (x / scale);
    const double end_y = 60 - 40 * (y / scale);
    output
        << "<svg class=\"plane\" viewBox=\"0 0 120 120\" role=\"img\">"
           "<title>Dimensions "
        << x_dimension << ", " << x_dimension + 1 << ": (" << RawNumber(x)
        << ", " << RawNumber(y) << ")</title>"
        << "<rect class=\"plane-bg\" x=\"1\" y=\"1\" width=\"118\" "
           "height=\"118\" rx=\"12\"/>"
           "<path class=\"axes\" d=\"M12 60H108M60 108V12\"/>"
           "<path class=\"axis-tip\" d=\"M104 57L108 60L104 63M57 16L60 12L63 "
           "16\"/>"
           "<line class=\"vector\" x1=\"60\" y1=\"60\" x2=\""
        << Number(end_x) << "\" y2=\"" << Number(end_y) << '"';
    if (x != 0 || y != 0)
      output << " marker-end=\"url(#arrow-tip)\"";
    output << "/>"
              "<circle class=\"origin\" cx=\"60\" cy=\"60\" r=\"2\"/>"
              "<text class=\"dimension\" x=\"99\" y=\"75\">d"
           << x_dimension
           << "</text><text class=\"dimension\" x=\"66\" y=\"17\">d"
           << x_dimension + 1 << "</text></svg>\n";
  }
  output << "</div><details><summary>16 raw values</summary>"
            "<code class=\"raw-values\">[";
  for (size_t i = 0; i < values.size(); ++i) {
    if (i != 0)
      output << ", ";
    output << RawNumber(values[i]);
  }
  output << "]</code></details>\n";
}

constexpr absl::string_view kDocumentStart = R"HTML(<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'">
<title>Activation trace · Pluto</title>
<style>
:root{color-scheme:light;--ink:#172b35;--muted:#536772;--border:#d4e0e3;--paper:#fff;--prompt:#336c96;--generated:#13745a}
*{box-sizing:border-box}body{margin:0;background:#f3f7f7;color:var(--ink);font:14px/1.5 system-ui,sans-serif}
main{max-width:1800px;margin:auto;padding:36px 24px 64px}h1{font-size:30px;letter-spacing:-.03em;margin:4px 0 8px}h2{font-size:20px;margin:0 0 16px}
.eyebrow{font-size:11px;font-weight:750;letter-spacing:.16em;text-transform:uppercase;color:var(--muted)}.intro{max-width:850px;color:var(--muted);margin:0 0 28px}
section{margin:28px 0;background:var(--paper);border:1px solid var(--border);border-radius:16px;padding:22px;overflow:hidden}
.transcript{display:grid;grid-template-columns:100px minmax(0,1fr);gap:8px 16px;margin:0 0 18px}.transcript dt{color:var(--muted);font-weight:650}.transcript dd{margin:0;white-space:pre-wrap;overflow-wrap:anywhere}
.scale-note{color:var(--muted);margin:14px 0}.scroll{overflow:auto;max-height:85vh;border:1px solid var(--border);border-radius:10px}
table{border-collapse:separate;border-spacing:0}th,td{border-right:1px solid var(--border);border-bottom:1px solid var(--border);padding:12px;vertical-align:top}thead th{position:sticky;top:0;z-index:3;background:#f5f9fa;text-align:left}
th:first-child{position:sticky;left:0;min-width:180px;max-width:240px;background:#f5f9fa;z-index:2;overflow-wrap:anywhere}thead th:first-child{z-index:4}tbody th{font-size:13px;padding-top:18px;text-align:left}
.token{display:block;font:600 14px/1.5 ui-monospace,monospace;white-space:pre-wrap;overflow-wrap:anywhere;max-width:235px}.token-meta{display:block;font-size:11px;color:var(--muted);margin-top:4px}
.kind{display:inline-block;font-size:10px;font-weight:750;text-transform:uppercase;letter-spacing:.06em;border-radius:12px;padding:2px 8px;margin:5px 0}.prompt .kind{color:var(--prompt);background:#e7f1fa}.generated .kind{color:var(--generated);background:#dff4e9}
.planes{display:grid;grid-template-columns:repeat(2,112px);gap:8px}.plane{width:112px;height:112px}.plane-bg{fill:#fafcfc;stroke:#c8d9dd;stroke-width:1.3}.axes,.axis-tip{fill:none;stroke:#b7c8cd;stroke-width:1}.vector{stroke:#18816b;stroke-width:2.5}.arrow-tip{fill:#18816b}.origin{fill:#536772}.dimension{font:9px ui-monospace,monospace;fill:#6f858e}
details{max-width:232px;margin-top:10px;font-size:11px;color:var(--muted)}summary{cursor:pointer}.raw-values{display:block;overflow-wrap:anywhere;white-space:normal;margin-top:8px;font-size:11px;color:var(--ink)}.defs{position:absolute;width:0;height:0;overflow:hidden}
@media(max-width:600px){main{padding:20px 10px}section{padding:12px}.transcript{grid-template-columns:1fr;gap:3px}.transcript dd{margin-bottom:10px}}
</style></head><body>
<svg class="defs" aria-hidden="true"><defs><marker id="arrow-tip" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path class="arrow-tip" d="M0 0L10 5L0 10Z"/></marker></defs></svg>
<main><div class="eyebrow">Pluto · inference inspection</div><h1>Activation trace</h1>
<p class="intro">Columns follow the token sequence; rows follow transformer boundaries. Each activation is a 16-dimensional residual-stream vector, shown as eight 2D planes: (d0, d1) through (d14, d15). Arrows start at the origin; right is positive x and up is positive y. All planes within a prompt use one shared scale, without normalizing individual vectors. Expand a cell for its exact values. Prompt and generated tokens are marked separately.</p>
)HTML";

}  // namespace

absl::Status WriteActivationTraceHtml(absl::Span<const ActivationTrace> traces,
                                      std::ostream& output) {
  if (traces.empty())
    return absl::InvalidArgumentError(
        "activation HTML requires at least one trace");
  // Validate the whole report before emitting even its header. In particular,
  // malformed later prompts must not leave a plausibly valid partial document.
  for (const auto& trace : traces) {
    const auto status = ValidateTrace(trace);
    if (!status.ok())
      return status;
  }
  output << kDocumentStart;
  for (size_t index = 0; index < traces.size(); ++index) {
    const auto& trace = traces[index];
    double maximum = 0;
    for (const auto& boundary : trace.boundaries)
      for (float value : boundary.values)
        maximum = std::max(maximum, std::abs(static_cast<double>(value)));
    // All-zero vectors stay at the origin on a useful [-1, 1] coordinate grid.
    const double scale = maximum == 0 ? 1 : maximum;
    output << "<section><h2>Prompt " << index + 1
           << "</h2><dl class=\"transcript\"><dt>Prompt</dt><dd>"
           << EscapeText(trace.prompt, false) << "</dd><dt>Generated</dt><dd>"
           << EscapeText(trace.continuation, false)
           << "</dd></dl><p class=\"scale-note\">" << trace.token_ids.size()
           << " tokens · " << trace.boundaries.size()
           << " boundaries · Shared coordinate range: [−" << Number(scale, 9)
           << ", +" << Number(scale, 9)
           << "]</p><div class=\"scroll\"><table><thead><tr><th "
              "scope=\"col\">Transformer boundary ↓<br>Token position →</th>";
    for (size_t token = 0; token < trace.token_ids.size(); ++token) {
      const bool prompt = token < trace.prompt_token_count;
      output << "<th scope=\"col\" class=\""
             << (prompt ? "prompt" : "generated")
             << "\"><span class=\"token\">&quot;"
             << EscapeText(trace.token_texts[token])
             << "&quot;</span><span class=\"token-meta\">Position " << token
             << " · Token ID " << trace.token_ids[token]
             << "</span><span class=\"kind\">"
             << (prompt ? "Prompt" : "Generated") << "</span></th>";
    }
    output << "</tr></thead><tbody>\n";
    for (const auto& boundary : trace.boundaries) {
      output << "<tr><th scope=\"row\">" << EscapeText(boundary.name)
             << "</th>";
      for (size_t token = 0; token < trace.token_ids.size(); ++token) {
        output << "<td>";
        WriteVector(absl::MakeConstSpan(boundary.values)
                        .subspan(token * kWidth, kWidth),
                    scale, output);
        output << "</td>";
      }
      output << "</tr>\n";
    }
    output << "</tbody></table></div></section>\n";
  }
  output << "</main></body></html>\n";
  if (!output.good())
    return absl::InternalError("failed to write activation trace HTML");
  return absl::OkStatus();
}

absl::Status WriteActivationTraceHtmlFile(
    absl::Span<const ActivationTrace> traces,
    const std::filesystem::path& output_file) {
  if (output_file.empty() || output_file.filename().empty())
    return absl::InvalidArgumentError(
        "activation trace output filename is empty");
  if (output_file.native().find('\0') != std::string::npos)
    return absl::InvalidArgumentError(
        "activation trace output filename contains a null byte");
  std::ostringstream rendered;
  const auto status = WriteActivationTraceHtml(traces, rendered);
  if (!status.ok())
    return status;
  const std::string contents = rendered.str();
  // Creating the temporary file beside its destination ensures rename stays
  // on one filesystem and atomically publishes only a complete HTML document.
  const auto parent = output_file.has_parent_path()
                          ? output_file.parent_path()
                          : std::filesystem::path(".");
  std::string temporary = (parent / ".pluto-activation-trace-XXXXXX").string();
  const int fd = mkstemp(temporary.data());
  if (fd < 0)
    return absl::ErrnoToStatus(
        errno, absl::StrCat("cannot create temporary activation trace in ",
                            parent.string()));
  size_t offset = 0;
  while (offset != contents.size()) {
    const size_t count =
        std::min(contents.size() - offset,
                 static_cast<size_t>(std::numeric_limits<ssize_t>::max()));
    const ssize_t written = write(fd, contents.data() + offset, count);
    if (written < 0 && errno == EINTR)
      continue;
    if (written <= 0) {
      const int error = written < 0 ? errno : EIO;
      close(fd);
      unlink(temporary.c_str());
      return absl::ErrnoToStatus(error, "cannot write activation trace file");
    }
    offset += static_cast<size_t>(written);
  }
  if (close(fd) != 0) {
    const int error = errno;
    unlink(temporary.c_str());
    return absl::ErrnoToStatus(error, "cannot close activation trace file");
  }
  if (rename(temporary.c_str(), output_file.c_str()) != 0) {
    const int error = errno;
    unlink(temporary.c_str());
    return absl::ErrnoToStatus(
        error, absl::StrCat("cannot publish activation trace at ",
                            output_file.string()));
  }
  return absl::OkStatus();
}

}  // namespace pluto::llm::memorize_general_facts
