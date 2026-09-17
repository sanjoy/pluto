#include "src/llm/experiments/ntk/experiment.h"

#include <fcntl.h>
#include <sys/stat.h>
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

#include "absl/strings/ascii.h"
#include "absl/strings/escaping.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "src/util/status_macros.h"

namespace pluto::llm::ntk {
namespace {

// Emit ASCII-only JSON. Arbitrary byte strings (including filesystem paths)
// are first CEscaped by callers and stored in explicit *_escaped_bytes fields.
void JsonString(std::ostream& out, absl::string_view text) {
  static constexpr char digits[] = "0123456789abcdef";
  out << '"';
  for (unsigned char byte : text)
    if (byte == '"' || byte == '\\')
      out << '\\' << static_cast<char>(byte);
    else if (byte < 0x20 || byte >= 0x7f)
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

void CsvString(std::ostream& out, absl::string_view text) {
  out << '"';
  for (char byte : text) {
    if (byte == '"')
      out << '"';
    out << byte;
  }
  out << '"';
}

void ConfigureOutput(std::ostream& out) {
  // Report numbers must not depend on a process locale's decimal separator.
  out.imbue(std::locale::classic());
  out << std::setprecision(std::numeric_limits<double>::max_digits10);
}

bool TargetCovered(const Example& example, absl::Span<const int> ids) {
  return example.next_token.has_value() &&
         std::find(ids.begin(), ids.end(), *example.next_token) != ids.end();
}

absl::Status ValidateReport(const ExperimentReport& report) {
  RETURN_IF_ERROR(ValidateMatrix(report.kernel));
  if (report.examples.empty() || report.output_tokens.empty() ||
      report.examples.size() >
          std::numeric_limits<size_t>::max() / report.output_tokens.size())
    return absl::InvalidArgumentError(
        "report has empty/overflowing row dimensions");
  const size_t rows = report.examples.size() * report.output_tokens.size();
  if (report.kernel.rows != rows || report.kernel.columns != rows ||
      report.initial_values.size() != rows ||
      report.ridge_predictions.size() != rows ||
      report.output_token_text.size() != report.output_tokens.size() ||
      (report.kernel_steps > 0 &&
       report.gradient_descent_predictions.size() != rows) ||
      (report.kernel_steps == 0 &&
       !report.gradient_descent_predictions.empty()))
    return absl::InvalidArgumentError(
        "report matrix/prediction dimensions disagree");
  if (!std::isfinite(report.ridge) || report.ridge <= 0 ||
      !std::isfinite(report.learning_rate) || report.learning_rate <= 0 ||
      report.kernel_steps < 0)
    return absl::InvalidArgumentError("invalid report regression options");
  for (const auto* values : {&report.initial_values, &report.ridge_predictions,
                             &report.gradient_descent_predictions})
    for (double value : *values)
      if (!std::isfinite(value))
        return absl::InvalidArgumentError(
            "report contains a nonfinite prediction");
  if (report.dimensions.vocabulary_size <= 0 ||
      report.dimensions.padded_vocabulary_size <
          report.dimensions.vocabulary_size ||
      report.dimensions.context_length <= 0)
    return absl::InvalidArgumentError("invalid report model dimensions");
  RETURN_IF_ERROR(ResolveOutputTokens(report.examples, report.output_tokens,
                                      report.dimensions.vocabulary_size)
                      .status());
  size_t train_count = 0;
  size_t eval_count = 0;
  bool reached_queries = false;
  for (const Example& example : report.examples) {
    if (example.split == "train") {
      if (reached_queries)
        return absl::InvalidArgumentError(
            "training examples must precede queries");
      ++train_count;
    } else {
      reached_queries = true;
      if (example.split == "eval")
        ++eval_count;
      else if (example.split != "prompt")
        return absl::InvalidArgumentError("unknown example split");
    }
    if (example.tokens.empty() ||
        example.tokens.size() >
            static_cast<size_t>(report.dimensions.context_length))
      return absl::InvalidArgumentError(
          "example context does not fit the model");
    if (example.split != "prompt" && !example.next_token.has_value())
      return absl::InvalidArgumentError(
          "corpus examples require a next-token target");
    for (int token : example.tokens)
      if (token < 0 || token >= report.dimensions.vocabulary_size)
        return absl::InvalidArgumentError(
            "example contains an invalid token ID");
    if (example.next_token.has_value() &&
        (*example.next_token < 0 ||
         *example.next_token >= report.dimensions.vocabulary_size))
      return absl::InvalidArgumentError(
          "example contains an invalid target ID");
  }
  if (train_count != report.windows.train_examples ||
      eval_count != report.windows.eval_examples)
    return absl::InvalidArgumentError(
        "report split counts disagree with window options");
  size_t expected_offset = 0;
  for (const ParameterSummary& parameter : report.parameters) {
    if (parameter.offset != expected_offset || parameter.elements == 0 ||
        parameter.elements >
            std::numeric_limits<size_t>::max() - expected_offset)
      return absl::InvalidArgumentError("invalid parameter block offsets");
    expected_offset += parameter.elements;
  }
  if (expected_offset != report.parameter_count)
    return absl::InvalidArgumentError(
        "parameter count disagrees with parameter blocks");
  return absl::OkStatus();
}

std::vector<double> SelectedSoftmax(absl::Span<const double> values,
                                    size_t output_count) {
  std::vector<double> result(values.size());
  for (size_t start = 0; start < values.size(); start += output_count) {
    const double maximum = *std::max_element(
        values.begin() + start, values.begin() + start + output_count);
    double sum = 0;
    for (size_t i = 0; i < output_count; ++i) {
      result[start + i] = std::exp(values[start + i] - maximum);
      sum += result[start + i];
    }
    for (size_t i = 0; i < output_count; ++i)
      result[start + i] /= sum;
  }
  return result;
}

absl::Status WriteNewFile(const std::filesystem::path& path,
                          const std::string& contents) {
  const int fd =
      open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0666);
  if (fd < 0)
    return absl::ErrnoToStatus(errno,
                               absl::StrCat("cannot create ", path.string()));
  size_t offset = 0;
  while (offset < contents.size()) {
    const ssize_t written =
        write(fd, contents.data() + offset,
              std::min(contents.size() - offset, size_t{1} << 30));
    if (written < 0 && errno == EINTR)
      continue;
    if (written <= 0) {
      const int saved_errno = written < 0 ? errno : EIO;
      close(fd);
      return absl::ErrnoToStatus(
          saved_errno, absl::StrCat("partial new report: ", path.string()));
    }
    offset += static_cast<size_t>(written);
  }
  if (close(fd) != 0)
    return absl::ErrnoToStatus(errno,
                               absl::StrCat("cannot close ", path.string()));
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<std::vector<Example>> SelectCorpusExamples(
    absl::Span<const int> corpus, const WindowOptions& options) {
  const size_t maximum = std::numeric_limits<size_t>::max();
  if (options.train_examples == 0 || options.context_tokens == 0 ||
      options.context_tokens == maximum ||
      options.stride < options.context_tokens + 1)
    return absl::InvalidArgumentError(
        "training/context counts must be positive; stride must include context "
        "and target");
  if (options.eval_examples > maximum - options.train_examples)
    return absl::InvalidArgumentError("example count overflows size_t");
  const size_t count = options.train_examples + options.eval_examples;
  if (options.offset > corpus.size() ||
      options.context_tokens >= corpus.size() - options.offset ||
      count - 1 >
          (corpus.size() - options.offset - options.context_tokens - 1) /
              options.stride)
    return absl::OutOfRangeError(
        "not enough corpus tokens for the requested windows");
  std::vector<Example> result;
  result.reserve(count);
  for (size_t index = 0; index < count; ++index) {
    const size_t start = options.offset + index * options.stride;
    Example example;
    example.split = index < options.train_examples ? "train" : "eval";
    example.corpus_token_offset = start;
    example.tokens.assign(corpus.begin() + start,
                          corpus.begin() + start + options.context_tokens);
    example.next_token = corpus[start + options.context_tokens];
    result.push_back(std::move(example));
  }
  return result;
}

absl::StatusOr<std::vector<int>> ResolveOutputTokens(
    absl::Span<const Example> examples, absl::Span<const int> requested,
    int vocabulary_size) {
  if (vocabulary_size <= 0)
    return absl::InvalidArgumentError("vocabulary size must be positive");
  std::vector<int> training_labels;
  for (const Example& example : examples)
    if (example.split == "train") {
      if (!example.next_token.has_value() || *example.next_token < 0 ||
          *example.next_token >= vocabulary_size)
        return absl::InvalidArgumentError(
            "training example has no valid next token");
      training_labels.push_back(*example.next_token);
    }
  if (training_labels.empty())
    return absl::InvalidArgumentError(
        "at least one training example is required");
  std::sort(training_labels.begin(), training_labels.end());
  training_labels.erase(
      std::unique(training_labels.begin(), training_labels.end()),
      training_labels.end());
  if (requested.empty())
    return training_labels;
  std::vector<int> selected(requested.begin(), requested.end());
  std::vector<int> sorted = selected;
  std::sort(sorted.begin(), sorted.end());
  if (sorted.front() < 0 || sorted.back() >= vocabulary_size ||
      std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end())
    return absl::InvalidArgumentError(
        "output token IDs must be distinct and in vocabulary");
  for (int target : training_labels)
    if (!std::binary_search(sorted.begin(), sorted.end(), target))
      return absl::InvalidArgumentError(absl::StrCat(
          "output token IDs do not cover training target ", target));
  return selected;
}

absl::StatusOr<std::vector<int>> ParseOutputTokenIds(absl::string_view text) {
  std::vector<int> result;
  if (text.empty())
    return result;
  while (true) {
    const size_t comma = text.find(',');
    const absl::string_view entry =
        absl::StripAsciiWhitespace(text.substr(0, comma));
    int token;
    if (!absl::SimpleAtoi(entry, &token) || token < 0)
      return absl::InvalidArgumentError(
          "--output_token_ids must be comma-separated nonnegative integers");
    result.push_back(token);
    if (comma == absl::string_view::npos)
      return result;
    text.remove_prefix(comma + 1);
  }
}

absl::StatusOr<RenderedReport> RenderReport(const ExperimentReport& report) {
  RETURN_IF_ERROR(ValidateReport(report));
  const size_t outputs = report.output_tokens.size();
  const auto initial_softmax = SelectedSoftmax(report.initial_values, outputs);
  const auto ridge_softmax = SelectedSoftmax(report.ridge_predictions, outputs);
  const auto gd_softmax =
      SelectedSoftmax(report.gradient_descent_predictions, outputs);
  std::ostringstream json;
  std::ostringstream kernel_csv;
  std::ostringstream predictions_csv;
  ConfigureOutput(json);
  ConfigureOutput(kernel_csv);
  ConfigureOutput(predictions_csv);

  json << "{\n\"schema_version\":1,\n\"experiment\":\"finite_empirical_neural_"
          "tangent_kernel\",\n"
          "\"weights_updated\":false,\n\"kernel_normalization\":\"none: "
          "K=J*J^T\",\n"
          "\"row_order\":\"sample-major, selected-output-token-major\",\n"
          "\"supervision\":\"one-hot selected logits, squared loss; not "
          "full-vocabulary cross entropy\",\n"
          "\"probability_scope\":\"softmax over selected output tokens only, "
          "NOT full-vocabulary probabilities\",\n"
          "\"text_encoding\":\"*_escaped_bytes fields contain C-escaped "
          "original bytes\",\n"
          "\"reference\":\"Jacot, Gabriel, Hongler (2018), "
          "https://arxiv.org/abs/1806.07572\",\n";
  json << "\"checkpoint_escaped_bytes\":";
  JsonString(json, absl::CEscape(report.checkpoint));
  json << ",\n\"initialization\":";
  JsonString(json, report.checkpoint.empty() ? "seeded recipe initialization"
                                             : "checkpoint weights");
  json << ",\n\"tokenizer_directory_escaped_bytes\":";
  JsonString(json, absl::CEscape(report.tokenizer_directory));
  json << ",\n\"corpus_path_escaped_bytes\":";
  JsonString(json, absl::CEscape(report.corpus_path));
  json << ",\n\"seed\":" << report.seed << ",\n\"compute_type\":";
  JsonString(json, report.compute_type);
  json << ",\n\"precision_caveat\":\"Existing backward rules and "
          "finite-precision forward compute define J; casting/quantization "
          "uses their straight-through conventions. This is not an exact "
          "symbolic real-arithmetic derivative.\",\n"
       << "\"model_dimensions\":{\"vocabulary\":"
       << report.dimensions.vocabulary_size
       << ",\"padded_vocabulary\":" << report.dimensions.padded_vocabulary_size
       << ",\"context_length\":" << report.dimensions.context_length
       << ",\"layers\":" << report.dimensions.layers
       << ",\"width\":" << report.dimensions.width
       << ",\"attention_heads\":" << report.dimensions.attention_heads
       << ",\"head_dimension\":" << report.dimensions.head_dimension
       << ",\"feed_forward_width\":" << report.dimensions.feed_forward_width
       << "},\n"
       << "\"window_selection\":{\"train_examples\":"
       << report.windows.train_examples
       << ",\"eval_examples\":" << report.windows.eval_examples
       << ",\"context_tokens\":" << report.windows.context_tokens
       << ",\"stride\":" << report.windows.stride
       << ",\"offset\":" << report.windows.offset << "},\n"
       << "\"padding_token\":" << report.padding_token << ",\n"
       << "\"readout_position\":\"last real context token; all later positions "
          "are EOS-padded and causally invisible\",\n"
       << "\"max_jacobian_bytes\":" << report.max_jacobian_bytes << ",\n"
       << "\"parameter_count\":" << report.parameter_count << ",\n"
       << "\"ridge\":" << report.ridge << ",\n"
       << "\"ridge_objective\":\"0.5*sum(error^2) + "
          "0.5*ridge*||delta_theta||^2\",\n"
       << "\"kernel_steps\":" << report.kernel_steps << ",\n"
       << "\"learning_rate\":" << report.learning_rate << ",\n"
       << "\"gradient_descent_objective\":\"0.5*mean(error^2) over scalar "
          "training coordinates, no ridge\",\n"
       << "\"output_token_ids\":";
  JsonArray<int>(json, report.output_tokens);
  json << ",\n\"output_token_text_escaped_bytes\":[";
  for (size_t index = 0; index < outputs; ++index) {
    if (index != 0)
      json << ',';
    JsonString(json, absl::CEscape(report.output_token_text[index]));
  }
  json << "],\n\"parameter_blocks\":[";
  for (size_t index = 0; index < report.parameters.size(); ++index) {
    const auto& parameter = report.parameters[index];
    if (index != 0)
      json << ',';
    json << "{\"weight_index\":" << parameter.weight_index
         << ",\"elements\":" << parameter.elements
         << ",\"jacobian_offset\":" << parameter.offset << '}';
  }
  json << "],\n\"examples\":[";
  for (size_t index = 0; index < report.examples.size(); ++index) {
    const auto& example = report.examples[index];
    if (index != 0)
      json << ',';
    json << "{\"sample\":" << index << ",\"split\":";
    JsonString(json, example.split);
    json << ",\"corpus_token_offset\":";
    if (example.corpus_token_offset.has_value())
      json << *example.corpus_token_offset;
    else
      json << "null";
    json << ",\"token_ids\":";
    JsonArray<int>(json, example.tokens);
    json << ",\"text_escaped_bytes\":";
    JsonString(json, absl::CEscape(example.text));
    json << ",\"last_real_position\":" << example.tokens.size() - 1
         << ",\"next_token\":";
    if (example.next_token.has_value())
      json << *example.next_token;
    else
      json << "null";
    json << ",\"target_in_selected_outputs\":"
         << (TargetCovered(example, report.output_tokens) ? "true" : "false")
         << '}';
  }
  json << "],\n\"rows\":[";
  predictions_csv
      << "row,sample,split,last_real_position,output_token_id,token_text_"
         "escaped_bytes,target,initial_logit,ridge_logit,gd_logit,initial_"
         "selected_softmax,ridge_selected_softmax,gd_selected_softmax\n";
  for (size_t row = 0; row < report.kernel.rows; ++row) {
    const size_t sample = row / outputs;
    const size_t coordinate = row % outputs;
    const auto& example = report.examples[sample];
    const bool covered = TargetCovered(example, report.output_tokens);
    const int target =
        covered && *example.next_token == report.output_tokens[coordinate];
    if (row != 0)
      json << ',';
    json << "{\"row\":" << row << ",\"sample\":" << sample
         << ",\"output_index\":0,\"flat_output_element\":"
         << (example.tokens.size() - 1) *
                    report.dimensions.padded_vocabulary_size +
                report.output_tokens[coordinate]
         << ",\"output_token_id\":" << report.output_tokens[coordinate]
         << ",\"target\":";
    if (covered)
      json << target;
    else
      json << "null";
    json << '}';
    predictions_csv << row << ',' << sample << ',' << example.split << ','
                    << example.tokens.size() - 1 << ','
                    << report.output_tokens[coordinate] << ',';
    CsvString(predictions_csv,
              absl::CEscape(report.output_token_text[coordinate]));
    predictions_csv << ',';
    if (covered)
      predictions_csv << target;
    predictions_csv << ',' << report.initial_values[row] << ','
                    << report.ridge_predictions[row] << ',';
    if (!report.gradient_descent_predictions.empty())
      predictions_csv << report.gradient_descent_predictions[row];
    predictions_csv << ',' << initial_softmax[row] << ',' << ridge_softmax[row]
                    << ',';
    if (!gd_softmax.empty())
      predictions_csv << gd_softmax[row];
    predictions_csv << '\n';
  }
  json << "],\n\"kernel\":{\"rows\":" << report.kernel.rows
       << ",\"columns\":" << report.kernel.columns << ",\"row_major_values\":";
  JsonArray<double>(json, report.kernel.values);
  json << "},\n\"initial_logits\":";
  JsonArray<double>(json, report.initial_values);
  json << ",\n\"ridge_logits\":";
  JsonArray<double>(json, report.ridge_predictions);
  json << ",\n\"gradient_descent_logits\":";
  JsonArray<double>(json, report.gradient_descent_predictions);
  json << ",\n\"initial_selected_softmax\":";
  JsonArray<double>(json, initial_softmax);
  json << ",\n\"ridge_selected_softmax\":";
  JsonArray<double>(json, ridge_softmax);
  json << ",\n\"gradient_descent_selected_softmax\":";
  JsonArray<double>(json, gd_softmax);
  json << "\n}\n";

  kernel_csv << "row";
  for (size_t column = 0; column < report.kernel.columns; ++column)
    kernel_csv << ",row_" << column;
  kernel_csv << '\n';
  for (size_t row = 0; row < report.kernel.rows; ++row) {
    kernel_csv << row;
    for (size_t column = 0; column < report.kernel.columns; ++column)
      kernel_csv << ',' << report.kernel(row, column);
    kernel_csv << '\n';
  }
  return RenderedReport{std::move(json).str(), std::move(kernel_csv).str(),
                        std::move(predictions_csv).str()};
}

absl::Status CheckOutputDirectory(const std::filesystem::path& directory) {
  if (directory.empty())
    return absl::InvalidArgumentError("--output_dir is required");
  struct stat info;
  if (lstat(directory.c_str(), &info) == 0)
    return absl::AlreadyExistsError(
        absl::StrCat("output already exists: ", directory.string()));
  if (errno != ENOENT)
    return absl::ErrnoToStatus(errno, "cannot inspect report destination");
  auto parent = directory.parent_path();
  if (parent.empty())
    parent = ".";
  std::error_code error;
  const bool is_directory = std::filesystem::is_directory(parent, error);
  if (error || !is_directory)
    return absl::InvalidArgumentError(
        "output parent must be an existing directory");
  return absl::OkStatus();
}

absl::Status WriteReportDirectory(const std::filesystem::path& directory,
                                  const RenderedReport& report) {
  RETURN_IF_ERROR(CheckOutputDirectory(directory));
  std::error_code error;
  if (!std::filesystem::create_directory(directory, error))
    return error ? absl::InternalError(absl::StrCat(
                       "cannot create report directory: ", error.message()))
                 : absl::AlreadyExistsError(
                       "report directory was created by another writer");
  RETURN_IF_ERROR(
      WriteNewFile(directory / "report.json", report.metadata_json));
  RETURN_IF_ERROR(WriteNewFile(directory / "kernel.csv", report.kernel_csv));
  RETURN_IF_ERROR(
      WriteNewFile(directory / "predictions.csv", report.predictions_csv));
  return absl::OkStatus();
}

}  // namespace pluto::llm::ntk
