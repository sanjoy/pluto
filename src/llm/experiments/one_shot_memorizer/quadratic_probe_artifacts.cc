#include "src/llm/experiments/one_shot_memorizer/quadratic_probe_artifacts.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "src/util/status_macros.h"

namespace pluto::llm::one_shot_memorizer {
namespace {

constexpr std::string_view kCoefficientHeader =
    "condition\tblock\ttensor\tflat_index\tfp32_master";
constexpr std::string_view kFailureHeader =
    "condition\tselection\tline_1based\tgroup\ttarget_position\t"
    "query_position\tpredicted_token\texpected_token\tgold_prefix_ids";

absl::Status Header(std::istream& input, std::string_view expected) {
  std::string line;
  if (!std::getline(input, line) || line != expected)
    return absl::InvalidArgumentError(
        "artifact header is missing or malformed");
  return absl::OkStatus();
}

bool KnownCondition(absl::string_view value) {
  return value == "original" || value == "original_post" ||
         value == "original_mlp_clone" || value == "learned_w1_refit" ||
         value == "initial_w1_raw" || value == "initial_w1_standardized" ||
         value == "quadratic_refit";
}

// Digits only, with full consumption: signs, spaces, overflow, and suffixes
// cannot silently alter the coordinate or alias a different parsed field.
absl::StatusOr<int> Decimal(absl::string_view field, int minimum, int maximum) {
  if (field.empty() || !std::all_of(field.begin(), field.end(), [](char c) {
        return c >= '0' && c <= '9';
      }))
    return absl::InvalidArgumentError(
        "artifact integer must contain only digits");
  int value;
  const auto parsed =
      std::from_chars(field.data(), field.data() + field.size(), value);
  if (parsed.ec != std::errc{} || parsed.ptr != field.data() + field.size() ||
      value < minimum || value > maximum)
    return absl::InvalidArgumentError("artifact integer is out of range");
  return value;
}

absl::StatusOr<float> FiniteFloat(absl::string_view field) {
  if (field.empty())
    return absl::InvalidArgumentError("artifact coefficient is empty");
  float value;
  const auto parsed = std::from_chars(field.data(), field.data() + field.size(),
                                      value, std::chars_format::general);
  if (parsed.ec != std::errc{} || parsed.ptr != field.data() + field.size() ||
      !std::isfinite(value))
    return absl::InvalidArgumentError(
        "artifact coefficient is not finite FP32");
  return value;
}

absl::Status EndOfInput(std::istream& input) {
  return input.bad() || !input.eof()
             ? absl::DataLossError("failed while reading artifact stream")
             : absl::OkStatus();
}

}  // namespace

absl::StatusOr<std::array<QuadraticCoefficients, 8>> ReadQuadraticCoefficients(
    std::istream& input) {
  RETURN_IF_ERROR(Header(input, kCoefficientHeader));
  std::array<QuadraticCoefficients, 8> result;
  std::array<std::array<bool, 2448>, 8> seen{};
  for (auto& block : result) {
    block.weights.resize(152 * 16);
    block.bias.resize(16);
  }
  std::string line;
  while (std::getline(input, line)) {
    const std::vector<absl::string_view> fields = absl::StrSplit(line, '\t');
    if (fields.size() != 5 || !KnownCondition(fields[0]))
      return absl::InvalidArgumentError("malformed coefficient artifact row");
    ASSIGN_OR_RETURN(const int block, Decimal(fields[1], 0, 7));
    const bool quadratic = fields[0] == "quadratic_refit";
    const int features = quadratic ? 152 : 64;
    int extent;
    if (fields[2] == "W2")
      extent = features * 16;
    else if (fields[2] == "b2")
      extent = 16;
    else if (!quadratic && fields[2] == "W1")
      extent = 16 * features;
    else if (!quadratic && fields[2] == "b1")
      extent = features;
    else
      return absl::InvalidArgumentError(
          "unexpected tensor in coefficient artifact");
    ASSIGN_OR_RETURN(const int index, Decimal(fields[3], 0, extent - 1));
    ASSIGN_OR_RETURN(const float value, FiniteFloat(fields[4]));
    if (!quadratic)
      continue;
    const bool matrix = fields[2] == "W2";
    const int flat = matrix ? index : 2432 + index;
    if (seen[block][flat])
      return absl::InvalidArgumentError(
          "duplicate quadratic coefficient coordinate");
    seen[block][flat] = true;
    (matrix ? result[block].weights : result[block].bias)[index] = value;
  }
  RETURN_IF_ERROR(EndOfInput(input));
  for (const auto& block : seen)
    if (!std::all_of(block.begin(), block.end(),
                     [](bool value) { return value; }))
      return absl::InvalidArgumentError(
          "incomplete quadratic coefficient tensors");
  return result;
}

absl::StatusOr<std::vector<QuadraticFailure>> ReadQuadraticFailures(
    std::istream& input) {
  RETURN_IF_ERROR(Header(input, kFailureHeader));
  std::vector<QuadraticFailure> result;
  std::array<bool, 1024> seen{};
  std::string line;
  while (std::getline(input, line)) {
    const std::vector<absl::string_view> fields = absl::StrSplit(line, '\t');
    if (fields.size() != 9 || !KnownCondition(fields[0]))
      return absl::InvalidArgumentError("malformed first-failure artifact row");
    if (fields[1] != "all") {
      ASSIGN_OR_RETURN(const int selection, Decimal(fields[1], 0, 7));
      (void)selection;
    }
    ASSIGN_OR_RETURN(const int corpus_line, Decimal(fields[2], 1, 1024));
    const absl::string_view group = (corpus_line - 1) % 5 == 0 ? "held" : "fit";
    if (fields[3] != group)
      return absl::InvalidArgumentError(
          "failure group disagrees with corpus split");
    ASSIGN_OR_RETURN(const int position, Decimal(fields[4], 5, 1024));
    ASSIGN_OR_RETURN(const int query, Decimal(fields[5], 4, 1023));
    if (query != position - 1)
      return absl::InvalidArgumentError(
          "failure query must precede target position");
    ASSIGN_OR_RETURN(const int predicted, Decimal(fields[6], 0, 4474));
    ASSIGN_OR_RETURN(const int expected, Decimal(fields[7], 0, 4474));
    if (predicted == expected)
      return absl::InvalidArgumentError("first failure has matching token IDs");
    std::vector<int> prefix;
    for (absl::string_view field : absl::StrSplit(fields[8], ',')) {
      ASSIGN_OR_RETURN(const int token, Decimal(field, 0, 4474));
      prefix.push_back(token);
    }
    if (prefix.size() != static_cast<size_t>(position))
      return absl::InvalidArgumentError(
          "failure prefix length differs from target position");
    if (fields[0] != "quadratic_refit" || fields[1] != "all")
      continue;
    if (seen[corpus_line - 1])
      return absl::InvalidArgumentError(
          "duplicate joint quadratic failure line");
    seen[corpus_line - 1] = true;
    result.push_back({static_cast<size_t>(corpus_line), position, predicted,
                      expected, std::move(prefix)});
  }
  RETURN_IF_ERROR(EndOfInput(input));
  if (result.empty())
    return absl::InvalidArgumentError(
        "artifact contains no joint quadratic failures");
  return result;
}

}  // namespace pluto::llm::one_shot_memorizer
