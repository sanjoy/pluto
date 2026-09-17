#include "src/llm/experiments/path_kernel/report.h"

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>
#include <system_error>
#include <utility>

#include "absl/strings/escaping.h"
#include "absl/strings/str_cat.h"
#include "src/util/status_macros.h"

namespace pluto::llm::path_kernel {
namespace {

// Callers C-escape arbitrary byte strings first. Keeping the enclosing JSON
// ASCII-only also makes non-UTF-8 token boundaries and Linux paths reversible.
void JsonString(std::ostream& out, absl::string_view value) {
  constexpr char digits[] = "0123456789abcdef";
  out << '"';
  for (unsigned char byte : value)
    if (byte == '"' || byte == '\\')
      out << '\\' << static_cast<char>(byte);
    else if (byte < 32 || byte >= 127)
      out << "\\u00" << digits[byte >> 4] << digits[byte & 15];
    else
      out << static_cast<char>(byte);
  out << '"';
}

template <class T>
void JsonArray(std::ostream& out, absl::Span<const T> values) {
  out << '[';
  for (size_t i = 0; i < values.size(); ++i) {
    if (i != 0)
      out << ',';
    out << values[i];
  }
  out << ']';
}

void JsonMatrix(std::ostream& out, const ntk::Matrix& matrix) {
  out << "{\"rows\":" << matrix.rows << ",\"columns\":" << matrix.columns
      << ",\"row_major_values\":";
  JsonArray<double>(out, matrix.values);
  out << '}';
}

void CsvString(std::ostream& out, absl::string_view value) {
  out << '"';
  for (char byte : value) {
    if (byte == '"')
      out << '"';
    out << byte;
  }
  out << '"';
}

absl::Status ValidateValues(absl::Span<const double> values, size_t count) {
  if (values.size() != count)
    return absl::InvalidArgumentError("report vector dimension mismatch");
  for (double value : values)
    if (!std::isfinite(value))
      return absl::InvalidArgumentError("report has a nonfinite value");
  return absl::OkStatus();
}

absl::Status ValidateMatrix(const ntk::Matrix& matrix, size_t rows,
                            size_t columns) {
  RETURN_IF_ERROR(ntk::ValidateMatrix(matrix));
  if (matrix.rows != rows || matrix.columns != columns)
    return absl::InvalidArgumentError("report matrix dimension mismatch");
  return absl::OkStatus();
}

absl::Status ValidateReport(const ExperimentReport& report) {
  if (report.examples.empty() || report.output_tokens.empty() ||
      report.examples.size() >
          std::numeric_limits<size_t>::max() / report.output_tokens.size() ||
      report.output_token_text.size() != report.output_tokens.size())
    return absl::InvalidArgumentError("invalid report query dimensions");
  if (report.steps <= 0 ||
      report.result.steps.size() != static_cast<size_t>(report.steps) ||
      !std::isfinite(report.learning_rate) || report.learning_rate <= 0 ||
      (report.compute_type != "fp16" && report.compute_type != "bf16"))
    return absl::InvalidArgumentError("invalid report trajectory settings");
  const auto& dimensions = report.dimensions;
  if (dimensions.vocabulary_size <= 0 ||
      dimensions.padded_vocabulary_size < dimensions.vocabulary_size ||
      dimensions.context_length <= 0 || dimensions.layers <= 0 ||
      dimensions.width <= 0 || dimensions.attention_heads <= 0 ||
      dimensions.head_dimension <= 0 || dimensions.feed_forward_width <= 0 ||
      report.padding_token < 0 ||
      report.padding_token >= dimensions.vocabulary_size)
    return absl::InvalidArgumentError("invalid report model dimensions");
  RETURN_IF_ERROR(ResolveQueryTokens(report.examples, report.output_tokens,
                                     dimensions.vocabulary_size)
                      .status());
  size_t train = 0;
  size_t eval = 0;
  bool queries_started = false;
  for (const auto& example : report.examples) {
    if (example.split == "train") {
      if (queries_started)
        return absl::InvalidArgumentError("training examples must come first");
      ++train;
    } else {
      queries_started = true;
      if (example.split == "eval")
        ++eval;
      else if (example.split != "prompt")
        return absl::InvalidArgumentError("unknown report split");
    }
    if (example.tokens.empty() ||
        example.tokens.size() > static_cast<size_t>(dimensions.context_length))
      return absl::InvalidArgumentError("invalid report context length");
    if (example.split != "prompt" && !example.next_token.has_value())
      return absl::InvalidArgumentError("training/eval target is missing");
    for (int token : example.tokens)
      if (token < 0 || token >= dimensions.vocabulary_size)
        return absl::InvalidArgumentError("invalid report input token");
    if (example.next_token.has_value() &&
        (*example.next_token < 0 ||
         *example.next_token >= dimensions.vocabulary_size))
      return absl::InvalidArgumentError("invalid report target token");
  }
  if (train == 0 || train != report.windows.train_examples ||
      eval != report.windows.eval_examples)
    return absl::InvalidArgumentError("report split counts disagree");
  const size_t rows = report.examples.size() * report.output_tokens.size();
  for (const auto* values :
       {&report.result.initial_values, &report.result.final_values,
        &report.result.reconstructed_values, &report.result.residual})
    RETURN_IF_ERROR(ValidateValues(*values, rows));
  RETURN_IF_ERROR(ValidateMatrix(report.result.path_kernel, rows, rows));
  RETURN_IF_ERROR(ValidateMatrix(report.result.contributions, rows, train));
  RETURN_IF_ERROR(ValidateValues(report.result.final_training_losses, train));
  size_t index = 0;
  for (const auto& step : report.result.steps) {
    if (step.step != static_cast<int>(++index))
      return absl::InvalidArgumentError("report step numbers must be 1-based");
    for (const auto* values : {&step.values_before, &step.values_after,
                               &step.predicted_delta, &step.residual})
      RETURN_IF_ERROR(ValidateValues(*values, rows));
    RETURN_IF_ERROR(ValidateValues(step.training_losses, train));
    RETURN_IF_ERROR(ValidateMatrix(step.tangent_kernel, rows, rows));
    RETURN_IF_ERROR(ValidateMatrix(step.contributions, rows, train));
  }
  size_t offset = 0;
  for (const auto& parameter : report.result.parameters) {
    if (parameter.offset != offset || parameter.elements == 0 ||
        parameter.elements > std::numeric_limits<size_t>::max() - offset)
      return absl::InvalidArgumentError("invalid report parameter offsets");
    offset += parameter.elements;
  }
  if (offset == 0 || offset != report.result.parameter_count)
    return absl::InvalidArgumentError("report parameter count disagrees");
  return absl::OkStatus();
}

// Ranking signs are relative to the selected logit, not overall correctness:
// raising a wrong token's logit is still a positive contribution to that logit.
void JsonRanking(std::ostream& json, const ExperimentReport& report, size_t row,
                 bool supporting) {
  std::vector<size_t> examples;
  for (size_t i = 0; i < report.windows.train_examples; ++i) {
    const double contribution = report.result.contributions(row, i);
    if (supporting ? contribution > 0 : contribution < 0)
      examples.push_back(i);
  }
  std::stable_sort(examples.begin(), examples.end(), [&](size_t a, size_t b) {
    return supporting ? report.result.contributions(row, a) >
                            report.result.contributions(row, b)
                      : report.result.contributions(row, a) <
                            report.result.contributions(row, b);
  });
  json << '[';
  for (size_t rank = 0; rank < examples.size(); ++rank) {
    if (rank != 0)
      json << ',';
    const size_t i = examples[rank];
    json << "{\"training_example\":" << i
         << ",\"contribution\":" << report.result.contributions(row, i)
         << ",\"text_escaped_bytes\":";
    JsonString(json, absl::CEscape(report.examples[i].text));
    json << '}';
  }
  json << ']';
}

absl::Status WriteNewFile(const std::filesystem::path& path,
                          absl::string_view contents) {
  const int fd =
      open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0666);
  if (fd < 0)
    return absl::ErrnoToStatus(errno,
                               absl::StrCat("cannot create ", path.string()));
  size_t offset = 0;
  while (offset < contents.size()) {
    const ssize_t count =
        write(fd, contents.data() + offset,
              std::min(contents.size() - offset, size_t{1} << 30));
    if (count < 0 && errno == EINTR)
      continue;
    if (count <= 0) {
      const int error = count < 0 ? errno : EIO;
      close(fd);
      return absl::ErrnoToStatus(error, "partial new path-kernel report");
    }
    offset += static_cast<size_t>(count);
  }
  if (close(fd) != 0)
    return absl::ErrnoToStatus(errno, "cannot close path-kernel report");
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<std::vector<int>> ResolveQueryTokens(
    absl::Span<const ntk::Example> examples, absl::Span<const int> requested,
    int vocabulary_size) {
  if (requested.empty())
    return ntk::ResolveOutputTokens(examples, requested, vocabulary_size);
  if (vocabulary_size <= 0)
    return absl::InvalidArgumentError("vocabulary size must be positive");
  for (int token : requested)
    if (token < 0 || token >= vocabulary_size)
      return absl::InvalidArgumentError(
          "query output token is out of vocabulary");
  std::vector<int> sorted(requested.begin(), requested.end());
  std::sort(sorted.begin(), sorted.end());
  if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end())
    return absl::InvalidArgumentError("query output token IDs must be unique");
  return std::vector<int>(requested.begin(), requested.end());
}

absl::StatusOr<RenderedReport> RenderReport(const ExperimentReport& report) {
  RETURN_IF_ERROR(ValidateReport(report));
  std::ostringstream json, kernel, contributions, predictions;
  for (auto* out : {&json, &kernel, &contributions, &predictions}) {
    out->imbue(std::locale::classic());
    *out << std::setprecision(std::numeric_limits<double>::max_digits10);
  }
  json << "{\n\"format\":\"pluto_path_kernel_v1\",\n"
          "\"paper\":\"https://arxiv.org/abs/2012.00152\",\n"
          "\"trajectory\":\"new full-batch plain gradient descent\",\n"
          "\"training_objective\":\"mean full-vocabulary last-position cross "
          "entropy\",\n"
          "\"quadrature\":\"left endpoint, dt=learning_rate\",\n"
          "\"path_kernel_definition\":\"sum_s learning_rate * J_query(s) "
          "J_query(s)^T\",\n"
          "\"contribution_definition\":\"-sum_s learning_rate/train_examples * "
          "dot(grad query_logit, grad example_loss)\",\n"
          "\"residual_definition\":\"actual_final - initial - "
          "sum_example_contributions\",\n"
          "\"limitations\":\"Finite-step and inherited mixed-precision "
          "approximation; not a reconstruction of historical AdamW training. "
          "Signed contributions are not a PSD kernel or causal leave-one-out "
          "effects.\",\n"
          "\"checkpoint_escaped_bytes\":";
  JsonString(json, absl::CEscape(report.checkpoint));
  json << ",\n\"tokenizer_directory_escaped_bytes\":";
  JsonString(json, absl::CEscape(report.tokenizer_directory));
  json << ",\n\"corpus_path_escaped_bytes\":";
  JsonString(json, absl::CEscape(report.corpus_path));
  json << ",\n\"seed\":" << report.seed << ",\n\"compute_type\":";
  JsonString(json, report.compute_type);
  json << ",\n\"steps\":" << report.steps
       << ",\n\"learning_rate\":" << report.learning_rate
       << ",\n\"max_jacobian_bytes\":" << report.max_jacobian_bytes
       << ",\n\"padding_token\":" << report.padding_token
       << ",\n\"windows\":{\"train_examples\":" << report.windows.train_examples
       << ",\"eval_examples\":" << report.windows.eval_examples
       << ",\"context_tokens\":" << report.windows.context_tokens
       << ",\"stride\":" << report.windows.stride
       << ",\"offset\":" << report.windows.offset << "},\n";
  const auto& d = report.dimensions;
  json << "\"model_dimensions\":{\"vocabulary_size\":" << d.vocabulary_size
       << ",\"padded_vocabulary_size\":" << d.padded_vocabulary_size
       << ",\"context_length\":" << d.context_length
       << ",\"layers\":" << d.layers << ",\"width\":" << d.width
       << ",\"attention_heads\":" << d.attention_heads
       << ",\"head_dimension\":" << d.head_dimension
       << ",\"feed_forward_width\":" << d.feed_forward_width << "},\n"
       << "\"output_token_ids\":";
  JsonArray<int>(json, report.output_tokens);
  json << ",\n\"output_token_text_escaped_bytes\":[";
  for (size_t i = 0; i < report.output_token_text.size(); ++i) {
    if (i != 0)
      json << ',';
    JsonString(json, absl::CEscape(report.output_token_text[i]));
  }
  json << "],\n\"examples\":[";
  for (size_t i = 0; i < report.examples.size(); ++i) {
    if (i != 0)
      json << ',';
    const auto& example = report.examples[i];
    json << "{\"example\":" << i << ",\"split\":";
    JsonString(json, example.split);
    json << ",\"corpus_token_offset\":";
    if (example.corpus_token_offset)
      json << *example.corpus_token_offset;
    else
      json << "null";
    json << ",\"tokens\":";
    JsonArray<int>(json, example.tokens);
    json << ",\"next_token\":";
    if (example.next_token)
      json << *example.next_token;
    else
      json << "null";
    json << ",\"text_escaped_bytes\":";
    JsonString(json, absl::CEscape(example.text));
    json << '}';
  }
  json << "],\n\"parameter_count\":" << report.result.parameter_count
       << ",\n\"parameter_blocks\":[";
  for (size_t i = 0; i < report.result.parameters.size(); ++i) {
    if (i != 0)
      json << ',';
    const auto& parameter = report.result.parameters[i];
    json << "{\"weight_index\":" << parameter.weight_index
         << ",\"elements\":" << parameter.elements
         << ",\"offset\":" << parameter.offset << '}';
  }
  json << "],\n\"query_rows\":[";
  const size_t rows = report.result.initial_values.size();
  predictions
      << "row,example,output_token_id,token_escaped_bytes,initial_logit,"
         "actual_final_logit,reconstructed_logit,residual\n";
  contributions
      << "query_row,training_example,contribution,text_escaped_bytes\n";
  for (size_t row = 0; row < rows; ++row) {
    if (row != 0)
      json << ',';
    const size_t sample = row / report.output_tokens.size();
    const size_t token_index = row % report.output_tokens.size();
    const int token = report.output_tokens[token_index];
    const size_t position = report.examples[sample].tokens.size() - 1;
    json << "{\"row\":" << row << ",\"example\":" << sample
         << ",\"output_index\":0,\"flat_output_element\":"
         << position * d.padded_vocabulary_size + token
         << ",\"output_token_id\":" << token << ",\"supporting\":";
    JsonRanking(json, report, row, true);
    json << ",\"opposing\":";
    JsonRanking(json, report, row, false);
    json << '}';
    predictions << row << ',' << sample << ',' << token << ',';
    CsvString(predictions,
              absl::CEscape(report.output_token_text[token_index]));
    predictions << ',' << report.result.initial_values[row] << ','
                << report.result.final_values[row] << ','
                << report.result.reconstructed_values[row] << ','
                << report.result.residual[row] << '\n';
    for (size_t i = 0; i < report.windows.train_examples; ++i) {
      contributions << row << ',' << i << ','
                    << report.result.contributions(row, i) << ',';
      CsvString(contributions, absl::CEscape(report.examples[i].text));
      contributions << '\n';
    }
  }
  json << "],\n\"path_kernel\":";
  JsonMatrix(json, report.result.path_kernel);
  json << ",\n\"contributions\":";
  JsonMatrix(json, report.result.contributions);
  json << ",\n\"initial_logits\":";
  JsonArray<double>(json, report.result.initial_values);
  json << ",\n\"actual_final_logits\":";
  JsonArray<double>(json, report.result.final_values);
  json << ",\n\"reconstructed_logits\":";
  JsonArray<double>(json, report.result.reconstructed_values);
  json << ",\n\"residual\":";
  JsonArray<double>(json, report.result.residual);
  json << ",\n\"final_training_losses\":";
  JsonArray<double>(json, report.result.final_training_losses);
  json << ",\n\"trajectory_steps\":[";
  for (size_t i = 0; i < report.result.steps.size(); ++i) {
    if (i != 0)
      json << ',';
    const auto& step = report.result.steps[i];
    json << "{\"step\":" << step.step << ",\"training_losses_before\":";
    JsonArray<double>(json, step.training_losses);
    json << ",\"logits_before\":";
    JsonArray<double>(json, step.values_before);
    json << ",\"logits_after\":";
    JsonArray<double>(json, step.values_after);
    json << ",\"tangent_kernel\":";
    JsonMatrix(json, step.tangent_kernel);
    json << ",\"contributions\":";
    JsonMatrix(json, step.contributions);
    json << ",\"predicted_delta\":";
    JsonArray<double>(json, step.predicted_delta);
    json << ",\"residual\":";
    JsonArray<double>(json, step.residual);
    json << '}';
  }
  json << "]\n}\n";
  kernel << "row";
  for (size_t column = 0; column < rows; ++column)
    kernel << ",row_" << column;
  kernel << '\n';
  for (size_t row = 0; row < rows; ++row) {
    kernel << row;
    for (size_t column = 0; column < rows; ++column)
      kernel << ',' << report.result.path_kernel(row, column);
    kernel << '\n';
  }
  return RenderedReport{std::move(json).str(), std::move(kernel).str(),
                        std::move(contributions).str(),
                        std::move(predictions).str()};
}

absl::Status WriteReportDirectory(const std::filesystem::path& directory,
                                  const RenderedReport& report) {
  RETURN_IF_ERROR(ntk::CheckOutputDirectory(directory));
  std::error_code error;
  if (!std::filesystem::create_directory(directory, error))
    return error ? absl::InternalError(
                       absl::StrCat("cannot create report: ", error.message()))
                 : absl::AlreadyExistsError(
                       "another writer created the report directory");
  RETURN_IF_ERROR(
      WriteNewFile(directory / "report.json", report.metadata_json));
  RETURN_IF_ERROR(
      WriteNewFile(directory / "path_kernel.csv", report.path_kernel_csv));
  RETURN_IF_ERROR(
      WriteNewFile(directory / "contributions.csv", report.contributions_csv));
  return WriteNewFile(directory / "predictions.csv", report.predictions_csv);
}

}  // namespace pluto::llm::path_kernel
