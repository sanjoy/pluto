#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "src/cuda/buffer.h"
#include "src/cuda/executor.h"
#include "src/cuda/page_locked_host_array.h"
#include "src/dataset/dataset.h"
#include "src/dataset/gpt2_detokenizer.h"
#include "src/dataset/gpt2_tokenizer.h"
#include "src/llm/checkpoint.h"
#include "src/llm/experiments/ntk/empirical_ntk.h"
#include "src/llm/experiments/ntk/experiment.h"
#include "src/llm/experiments/ntk/kernel_regression.h"
#include "src/llm/recipes/gpt2.h"
#include "src/util/status_macros.h"

ABSL_FLAG(
    std::string, checkpoint, "",
    "Optional checkpoint directory; empty measures the seeded initialization");
ABSL_FLAG(std::string, tokenizer_dir, "datasets/tokenizer/gpt2",
          "Directory containing the GPT-2 tokenizer.json");
ABSL_FLAG(std::string, corpus, "testdata/shakespeare.txt",
          "UTF-8 text corpus, mapped read-only and encoded once");
ABSL_FLAG(int, seed, 123,
          "Fixed model initialization seed (ignored for loaded weights)");
ABSL_FLAG(std::string, compute_type, "fp16",
          "fp16: FP32 activation storage with inherited FP16 MMA; bf16: BF16 "
          "activations");
ABSL_FLAG(int64_t, train_examples, 2,
          "Number of non-overlapping supervised contexts");
ABSL_FLAG(int64_t, eval_examples, 1, "Number of subsequent held-out contexts");
ABSL_FLAG(int64_t, context_tokens, 16,
          "Real corpus-context tokens, between 1 and 1024");
ABSL_FLAG(int64_t, stride, 1025,
          "Token offset between examples; must be at least context_tokens+1");
ABSL_FLAG(int64_t, offset, 0,
          "First context's offset in the full corpus tokenization");
ABSL_FLAG(std::string, prompt, "",
          "Optional additional unlabeled query, up to 1024 tokens");
ABSL_FLAG(std::string, output_token_ids, "",
          "Comma-separated shared logit coordinates; default: sorted training "
          "target IDs");
ABSL_FLAG(double, ridge, 1e-3,
          "Positive diagonal ridge added to the raw training Gram matrix");
ABSL_FLAG(int, kernel_steps, 0,
          "Optional frozen-kernel mean-squared GD steps; zero disables GD");
ABSL_FLAG(
    double, learning_rate, 1e-5,
    "Positive frozen-kernel GD rate, not a neural-network optimizer rate");
ABSL_FLAG(int64_t, max_jacobian_mib, 4096,
          "Maximum explicit Jacobian MiB; other model/gradient/state memory is "
          "additional");
ABSL_FLAG(std::string, output_dir, "",
          "Required NEW report directory; parent must exist and existing paths "
          "are refused");

namespace pluto::llm::ntk {
namespace {

absl::StatusOr<WindowOptions> ReadWindowOptions() {
  const int64_t train = absl::GetFlag(FLAGS_train_examples);
  const int64_t eval = absl::GetFlag(FLAGS_eval_examples);
  const int64_t context = absl::GetFlag(FLAGS_context_tokens);
  const int64_t stride = absl::GetFlag(FLAGS_stride);
  const int64_t offset = absl::GetFlag(FLAGS_offset);
  if (train <= 0 || eval < 0 || context <= 0 || context > kGpt2ContextLength ||
      stride < context + 1 || offset < 0)
    return absl::InvalidArgumentError(
        "require train_examples>0, eval_examples>=0, 1<=context_tokens<=1024, "
        "stride>=context_tokens+1, and offset>=0");
  for (int64_t value : {train, eval, context, stride, offset})
    if (static_cast<uint64_t>(value) > std::numeric_limits<size_t>::max())
      return absl::InvalidArgumentError("window option exceeds size_t range");
  return WindowOptions{static_cast<size_t>(train), static_cast<size_t>(eval),
                       static_cast<size_t>(context),
                       static_cast<size_t>(stride),
                       static_cast<size_t>(offset)};
}

absl::StatusOr<std::vector<Sample>> UploadSamples(
    cuda::Executor& executor, absl::Span<const Example> examples,
    absl::Span<const int> output_tokens, int eos_token) {
  static_assert(sizeof(int) == sizeof(int32_t),
                "GPT-2 input buffers contain int32 IDs");
  std::vector<Sample> samples;
  samples.reserve(examples.size());
  for (const Example& example : examples) {
    ASSIGN_OR_RETURN(auto host, cuda::PageLockedHostArray<int>::Allocate(
                                    executor, kGpt2ContextLength));
    std::fill(host.begin(), host.end(), eos_token);
    std::copy(example.tokens.begin(), example.tokens.end(), host.begin());
    ASSIGN_OR_RETURN(auto device, Buffer::Allocate(executor, host.size_bytes()));
    RETURN_IF_ERROR(cuda::CudaStatus(
        cudaMemcpyAsync(device.data(), host.data(), host.size_bytes(),
                        cudaMemcpyHostToDevice, executor.stream()),
        "upload NTK context"));
    Sample sample;
    sample.inputs.push_back(std::move(device));
    // All samples satisfy the recipe's fixed 1024-token input signature. A
    // causal logit at the final real token cannot depend on later EOS padding.
    const size_t last_position = example.tokens.size() - 1;
    for (int token : output_tokens)
      sample.coordinates.push_back(OutputCoordinate{
          0, last_position * kGpt2PaddedVocabularySize + token});
    samples.push_back(std::move(sample));
    // host's pool free is stream-ordered after the upload; no host sync or
    // pageable-memory staging is necessary to release this handle safely.
  }
  return samples;
}

void PrintMetrics(absl::string_view method,
                  absl::Span<const double> predictions,
                  const ExperimentReport& report) {
  const size_t outputs = report.output_tokens.size();
  for (absl::string_view split : {"train", "eval"}) {
    size_t covered = 0;
    size_t total = 0;
    size_t correct = 0;
    long double squared_error = 0;
    for (size_t sample = 0; sample < report.examples.size(); ++sample) {
      const Example& example = report.examples[sample];
      if (example.split != split)
        continue;
      ++total;
      if (!example.next_token.has_value() ||
          std::find(report.output_tokens.begin(), report.output_tokens.end(),
                    *example.next_token) == report.output_tokens.end())
        continue;
      ++covered;
      const auto begin = predictions.begin() + sample * outputs;
      const size_t winner = std::max_element(begin, begin + outputs) - begin;
      correct += report.output_tokens[winner] == *example.next_token;
      for (size_t coordinate = 0; coordinate < outputs; ++coordinate) {
        const double target =
            report.output_tokens[coordinate] == *example.next_token;
        const long double error =
            predictions[sample * outputs + coordinate] - target;
        squared_error += error * error;
      }
    }
    if (total == 0)
      continue;
    std::cout << method << ' ' << split << ": selected-target coverage "
              << covered << '/' << total;
    if (covered != 0)
      std::cout << ", mean squared error="
                << static_cast<double>(squared_error / (covered * outputs))
                << ", selected-class accuracy=" << correct << '/' << covered;
    std::cout << '\n';
  }
}

absl::Status Run() {
  const std::filesystem::path output_directory =
      absl::GetFlag(FLAGS_output_dir);
  RETURN_IF_ERROR(CheckOutputDirectory(output_directory));
  ASSIGN_OR_RETURN(auto windows, ReadWindowOptions());
  const std::string compute_name = absl::GetFlag(FLAGS_compute_type);
  if (compute_name != "fp16" && compute_name != "bf16")
    return absl::InvalidArgumentError("--compute_type must be fp16 or bf16");
  const DataType compute =
      compute_name == "bf16" ? DataType::BF16 : DataType::FP16;
  const double ridge = absl::GetFlag(FLAGS_ridge);
  const int steps = absl::GetFlag(FLAGS_kernel_steps);
  const double learning_rate = absl::GetFlag(FLAGS_learning_rate);
  if (!std::isfinite(ridge) || ridge <= 0 || steps < 0 ||
      !std::isfinite(learning_rate) || learning_rate <= 0)
    return absl::InvalidArgumentError(
        "ridge/learning_rate must be finite and positive; kernel_steps must be "
        "nonnegative");
  const int64_t mib = absl::GetFlag(FLAGS_max_jacobian_mib);
  constexpr size_t kMib = 1024 * 1024;
  if (mib <= 0 ||
      static_cast<uint64_t>(mib) > std::numeric_limits<size_t>::max() / kMib)
    return absl::InvalidArgumentError(
        "--max_jacobian_mib must be positive and fit size_t bytes");
  ASSIGN_OR_RETURN(auto requested,
                   ParseOutputTokenIds(absl::GetFlag(FLAGS_output_token_ids)));
  const std::string tokenizer_directory = absl::GetFlag(FLAGS_tokenizer_dir);
  ASSIGN_OR_RETURN(auto encoder,
                   tokenizer::Gpt2Tokenizer::Load(tokenizer_directory));
  ASSIGN_OR_RETURN(auto decoder,
                   tokenizer::Gpt2Detokenizer::Load(tokenizer_directory));
  if (encoder->vocab_size() != kGpt2VocabularySize ||
      decoder->vocab_size() != kGpt2VocabularySize)
    return absl::InvalidArgumentError(
        "the recipe requires a 50,257-token GPT-2 vocabulary");
  ASSIGN_OR_RETURN(auto executor, cuda::Executor::Create());
  const std::string corpus_path = absl::GetFlag(FLAGS_corpus);
  ASSIGN_OR_RETURN(auto corpus, LoadTextCorpus(corpus_path));
  ASSIGN_OR_RETURN(auto corpus_tokens,
                   encoder->Encode(*executor, corpus.text()));
  ASSIGN_OR_RETURN(auto examples,
                   SelectCorpusExamples(corpus_tokens.span(), windows));
  for (Example& example : examples) {
    ASSIGN_OR_RETURN(example.text, decoder->Decode(example.tokens));
  }
  const std::string prompt = absl::GetFlag(FLAGS_prompt);
  if (!prompt.empty()) {
    ASSIGN_OR_RETURN(auto tokens, encoder->Encode(*executor, prompt));
    if (tokens.empty() || tokens.size() > kGpt2ContextLength)
      return absl::InvalidArgumentError(
          "--prompt must encode to between 1 and 1024 tokens");
    examples.push_back(Example{"prompt", std::nullopt,
                               std::vector<int>(tokens.begin(), tokens.end()),
                               std::nullopt, prompt});
  }
  ASSIGN_OR_RETURN(auto output_tokens, ResolveOutputTokens(examples, requested,
                                                           kGpt2VocabularySize));
  if (examples.size() >
      std::numeric_limits<size_t>::max() / output_tokens.size())
    return absl::InvalidArgumentError("Jacobian row count overflows size_t");
  std::cout << "Fixed-weight empirical NTK: " << examples.size()
            << " contexts, " << output_tokens.size() << " output coordinates, "
            << examples.size() * output_tokens.size() << " backward passes.\n"
            << "Compute: " << compute_name
            << " (existing mixed-precision forward/backward rules).\n"
            << "Output tokens:";
  for (int token : output_tokens)
    std::cout << ' ' << token;
  std::cout << "\nSelected-class probabilities are NOT full-vocabulary "
               "probabilities.\n";
  if (output_tokens.size() == 1)
    std::cout << "Only one class selected: selected softmax is identically 1; "
                 "inspect logits/kernel instead.\n";
  ASSIGN_OR_RETURN(auto model,
                   CreateGpt2(*executor, compute, absl::GetFlag(FLAGS_seed)));
  const std::string checkpoint = absl::GetFlag(FLAGS_checkpoint);
  if (!checkpoint.empty())
    RETURN_IF_ERROR(ReadFromDirectory(*executor, *model, checkpoint));
  ASSIGN_OR_RETURN(auto samples,
                   UploadSamples(*executor, examples, output_tokens,
                                 encoder->eos_token_id()));
  KernelOptions kernel_options;
  kernel_options.max_jacobian_bytes = static_cast<size_t>(mib) * kMib;
  kernel_options.progress = [](size_t completed, size_t total) {
    std::cout << "Jacobian rows: " << completed << '/' << total << std::endl;
    return absl::OkStatus();
  };
  ASSIGN_OR_RETURN(
      auto measured,
      ComputeEmpiricalKernel(*executor, *model, samples, kernel_options));

  // The kernel is vector-valued: different selected tokens remain different
  // scalar observations, including every cross-token block. Fit only the
  // training prefix; held-out/prompt labels never influence coefficients.
  const size_t train_rows = windows.train_examples * output_tokens.size();
  Matrix train_kernel{train_rows, train_rows,
                      std::vector<double>(train_rows * train_rows)};
  Matrix cross_kernel{measured.gram.rows, train_rows,
                      std::vector<double>(measured.gram.rows * train_rows)};
  for (size_t row = 0; row < measured.gram.rows; ++row)
    for (size_t column = 0; column < train_rows; ++column) {
      cross_kernel(row, column) = measured.gram(row, column);
      if (row < train_rows)
        train_kernel(row, column) = measured.gram(row, column);
    }
  std::vector<double> targets(train_rows);
  for (size_t row = 0; row < train_rows; ++row)
    targets[row] = output_tokens[row % output_tokens.size()] ==
                   *examples[row / output_tokens.size()].next_token;
  const auto training_initial =
      absl::MakeConstSpan(measured.initial_values).subspan(0, train_rows);
  ASSIGN_OR_RETURN(auto coefficients,
                   FitRidge(train_kernel, training_initial, targets, ridge));
  ASSIGN_OR_RETURN(auto ridge_predictions,
                   Predict(cross_kernel, measured.initial_values, coefficients));
  std::vector<double> gd_predictions;
  if (steps != 0) {
    // A row-sum bound gives a conservative sufficient stable rate for PSD K:
    // eta < 2*N/lambda_max(K). It is guidance only; never rescale the raw K.
    double bound = 0;
    for (size_t row = 0; row < train_rows; ++row) {
      double row_sum = 0;
      for (size_t column = 0; column < train_rows; ++column)
        row_sum += std::abs(train_kernel(row, column));
      bound = std::max(bound, row_sum);
    }
    if (bound > 0)
      std::cout << "Conservative frozen-GD stability guidance: learning_rate < "
                << 2 * static_cast<double>(train_rows) / bound << '\n';
    ASSIGN_OR_RETURN(auto gd_coefficients,
                     FitGradientDescent(train_kernel, training_initial, targets,
                                        learning_rate, steps));
    ASSIGN_OR_RETURN(
        gd_predictions,
        Predict(cross_kernel, measured.initial_values, gd_coefficients));
  }
  ExperimentReport report;
  report.checkpoint = checkpoint;
  report.tokenizer_directory = tokenizer_directory;
  report.corpus_path = corpus_path;
  report.seed = absl::GetFlag(FLAGS_seed);
  report.compute_type = compute_name;
  report.windows = windows;
  report.dimensions = ModelDimensions{kGpt2VocabularySize,
                                      kGpt2PaddedVocabularySize,
                                      kGpt2ContextLength,
                                      kGpt2TransformerBlockCount,
                                      kGpt2ModelWidth,
                                      kGpt2AttentionHeads,
                                      kGpt2AttentionHeadDimension,
                                      kGpt2FeedForwardWidth};
  report.padding_token = encoder->eos_token_id();
  report.max_jacobian_bytes = kernel_options.max_jacobian_bytes;
  report.examples = std::move(examples);
  report.output_tokens = std::move(output_tokens);
  for (int token : report.output_tokens) {
    ASSIGN_OR_RETURN(auto text, decoder->Decode({&token, 1}));
    report.output_token_text.push_back(std::move(text));
  }
  for (const ParameterBlock& parameter : measured.parameters)
    report.parameters.push_back(
        {parameter.weight_index, parameter.elements, parameter.offset});
  report.parameter_count = measured.parameter_count;
  report.ridge = ridge;
  report.kernel_steps = steps;
  report.learning_rate = learning_rate;
  report.kernel = std::move(measured.gram);
  report.initial_values = std::move(measured.initial_values);
  report.ridge_predictions = std::move(ridge_predictions);
  report.gradient_descent_predictions = std::move(gd_predictions);
  PrintMetrics("Initial", report.initial_values, report);
  PrintMetrics("Ridge", report.ridge_predictions, report);
  if (steps != 0)
    PrintMetrics("Frozen GD", report.gradient_descent_predictions, report);
  ASSIGN_OR_RETURN(auto rendered, RenderReport(report));
  RETURN_IF_ERROR(WriteReportDirectory(output_directory, rendered));
  std::cout << "Wrote " << (output_directory / "report.json")
            << ", kernel.csv, predictions.csv\n"
            << "Model weights were held fixed throughout. Unique parameters: "
            << report.parameter_count << '\n';
  return absl::OkStatus();
}

}  // namespace
}  // namespace pluto::llm::ntk

int main(int argc, char** argv) {
  const auto positional = absl::ParseCommandLine(argc, argv);
  if (positional.size() != 1) {
    std::cerr << "Positional arguments are not accepted; use --prompt and "
                 "--output_dir.\n";
    return 1;
  }
  const auto status = pluto::llm::ntk::Run();
  if (!status.ok()) {
    std::cerr << status << '\n';
    return 1;
  }
  return 0;
}
