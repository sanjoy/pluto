#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/status/status.h"
#include "src/cuda/buffer.h"
#include "src/cuda/executor.h"
#include "src/cuda/page_locked_host_array.h"
#include "src/dataset/dataset.h"
#include "src/dataset/gpt2_detokenizer.h"
#include "src/dataset/gpt2_tokenizer.h"
#include "src/llm/checkpoint.h"
#include "src/llm/experiments/ntk/experiment.h"
#include "src/llm/experiments/path_kernel/path_kernel.h"
#include "src/llm/experiments/path_kernel/report.h"
#include "src/llm/recipes/gpt2.h"
#include "src/util/status_macros.h"

ABSL_FLAG(
    std::string, checkpoint, "",
    "Optional initial checkpoint; runs a NEW plain-GD trajectory from it, "
    "not a reconstruction of historical training");
ABSL_FLAG(std::string, tokenizer_dir, "datasets/tokenizer/gpt2",
          "Directory containing the GPT-2 tokenizer.json");
ABSL_FLAG(std::string, corpus, "testdata/shakespeare.txt",
          "Text corpus, mapped read-only and encoded once");
ABSL_FLAG(int, seed, 123,
          "Fixed initialization seed; ignored for loaded weights");
ABSL_FLAG(std::string, compute_type, "fp16",
          "fp16: FP32 activation storage with inherited FP16 MMA; bf16: BF16 "
          "activations");
ABSL_FLAG(int64_t, train_examples, 2,
          "Number of full-batch supervised contexts");
ABSL_FLAG(int64_t, eval_examples, 1, "Number of held-out query contexts");
ABSL_FLAG(int64_t, context_tokens, 16,
          "Real context tokens, between 1 and 1024");
ABSL_FLAG(int64_t, stride, 1025,
          "Token offset between non-overlapping context-plus-target windows");
ABSL_FLAG(int64_t, offset, 0,
          "First context offset in the full tokenized corpus");
ABSL_FLAG(std::string, prompt, "", "Optional additional unlabeled query");
ABSL_FLAG(
    std::string, output_token_ids, "",
    "Comma-separated query logit IDs; default sorted training-target IDs. "
    "This selection NEVER limits the training softmax vocabulary");
ABSL_FLAG(int, steps, 3,
          "Number of actual full-batch plain-GD parameter updates");
ABSL_FLAG(double, learning_rate, 1e-5,
          "Positive GD rate on the mean full-vocabulary example loss");
ABSL_FLAG(int64_t, max_jacobian_mib, 4096,
          "Memory budget for query AND training-loss Jacobians; model and "
          "temporary state require additional memory");
ABSL_FLAG(std::string, output_dir, "",
          "Required NEW report directory below an existing parent");

namespace pluto::llm::path_kernel {
namespace {

absl::StatusOr<ntk::WindowOptions> ReadWindows() {
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
  return ntk::WindowOptions{
      static_cast<size_t>(train), static_cast<size_t>(eval),
      static_cast<size_t>(context), static_cast<size_t>(stride),
      static_cast<size_t>(offset)};
}

absl::StatusOr<std::vector<ntk::Sample>> UploadQueries(
    cuda::Executor& executor, absl::Span<const ntk::Example> examples,
    absl::Span<const int> output_tokens, int eos_token) {
  static_assert(sizeof(int) == sizeof(int32_t),
                "GPT-2 requires int32 token IDs");
  std::vector<ntk::Sample> queries;
  queries.reserve(examples.size());
  for (const auto& example : examples) {
    ASSIGN_OR_RETURN(auto host, cuda::PageLockedHostArray<int>::Allocate(
                                    executor, kGpt2ContextLength));
    std::fill(host.begin(), host.end(), eos_token);
    std::copy(example.tokens.begin(), example.tokens.end(), host.begin());
    ASSIGN_OR_RETURN(auto device, Buffer::Allocate(executor, host.size_bytes()));
    RETURN_IF_ERROR(cuda::CudaStatus(
        cudaMemcpyAsync(device.data(), host.data(), host.size_bytes(),
                        cudaMemcpyHostToDevice, executor.stream()),
        "upload path-kernel context"));
    ntk::Sample sample;
    sample.inputs.push_back(std::move(device));
    // The fixed recipe still consumes 1024 positions. Causality makes logits
    // at the last real position independent of the later EOS padding.
    const size_t first_logit =
        (example.tokens.size() - 1) * kGpt2PaddedVocabularySize;
    for (int token : output_tokens)
      sample.coordinates.push_back({0, first_logit + token});
    queries.push_back(std::move(sample));
    // Releasing the host handle queues pool-free after its asynchronous copy.
  }
  return queries;
}

double MaximumMagnitude(absl::Span<const double> values) {
  double maximum = 0;
  for (double value : values)
    maximum = std::max(maximum, std::abs(value));
  return maximum;
}

double Mean(absl::Span<const double> values) {
  return std::accumulate(values.begin(), values.end(), 0.0) / values.size();
}

absl::Status RunExperiment() {
  const std::filesystem::path output_directory =
      absl::GetFlag(FLAGS_output_dir);
  RETURN_IF_ERROR(ntk::CheckOutputDirectory(output_directory));
  ASSIGN_OR_RETURN(auto windows, ReadWindows());
  const std::string compute_name = absl::GetFlag(FLAGS_compute_type);
  if (compute_name != "fp16" && compute_name != "bf16")
    return absl::InvalidArgumentError("--compute_type must be fp16 or bf16");
  const DataType compute =
      compute_name == "bf16" ? DataType::BF16 : DataType::FP16;
  const int steps = absl::GetFlag(FLAGS_steps);
  const double learning_rate = absl::GetFlag(FLAGS_learning_rate);
  if (steps <= 0 || !std::isfinite(learning_rate) || learning_rate <= 0)
    return absl::InvalidArgumentError(
        "steps must be positive; learning_rate must be finite and positive");
  const int64_t mib = absl::GetFlag(FLAGS_max_jacobian_mib);
  constexpr size_t kMib = 1024 * 1024;
  if (mib <= 0 ||
      static_cast<uint64_t>(mib) > std::numeric_limits<size_t>::max() / kMib)
    return absl::InvalidArgumentError(
        "max_jacobian_mib must fit positive size_t bytes");
  ASSIGN_OR_RETURN(auto requested, ntk::ParseOutputTokenIds(
                                       absl::GetFlag(FLAGS_output_token_ids)));
  const std::string tokenizer_directory = absl::GetFlag(FLAGS_tokenizer_dir);
  ASSIGN_OR_RETURN(auto encoder,
                   tokenizer::Gpt2Tokenizer::Load(tokenizer_directory));
  ASSIGN_OR_RETURN(auto decoder,
                   tokenizer::Gpt2Detokenizer::Load(tokenizer_directory));
  if (encoder->vocab_size() != kGpt2VocabularySize ||
      decoder->vocab_size() != kGpt2VocabularySize)
    return absl::InvalidArgumentError(
        "recipe requires a 50,257-token GPT-2 vocabulary");
  ASSIGN_OR_RETURN(auto executor, cuda::Executor::Create());
  const std::string corpus_path = absl::GetFlag(FLAGS_corpus);
  ASSIGN_OR_RETURN(auto corpus, LoadTextCorpus(corpus_path));
  ASSIGN_OR_RETURN(auto corpus_tokens,
                   encoder->Encode(*executor, corpus.text()));
  ASSIGN_OR_RETURN(auto examples,
                   ntk::SelectCorpusExamples(corpus_tokens.span(), windows));
  for (auto& example : examples) {
    ASSIGN_OR_RETURN(example.text, decoder->Decode(example.tokens));
  }
  const std::string prompt = absl::GetFlag(FLAGS_prompt);
  if (!prompt.empty()) {
    ASSIGN_OR_RETURN(auto tokens, encoder->Encode(*executor, prompt));
    if (tokens.empty() || tokens.size() > kGpt2ContextLength)
      return absl::InvalidArgumentError(
          "prompt must encode to 1 through 1024 tokens");
    examples.push_back({"prompt", std::nullopt,
                        std::vector<int>(tokens.begin(), tokens.end()),
                        std::nullopt, prompt});
  }
  ASSIGN_OR_RETURN(auto output_tokens,
                   ResolveQueryTokens(examples, requested, kGpt2VocabularySize));
  if (examples.size() >
      std::numeric_limits<size_t>::max() / output_tokens.size())
    return absl::InvalidArgumentError("query row count overflows size_t");
  std::cout << "New full-batch plain-GD path: " << windows.train_examples
            << " training contexts, " << examples.size() * output_tokens.size()
            << " query logit coordinates, " << steps << " updates.\n"
            << "Loss uses all " << kGpt2VocabularySize
            << " tokens; selected query outputs do not restrict training.\n"
            << "Compute: " << compute_name
            << "; finite-step and mixed-precision residuals are measured.\n"
            << "Output tokens:";
  for (int token : output_tokens)
    std::cout << ' ' << token;
  std::cout << std::endl;
  ASSIGN_OR_RETURN(auto model,
                   CreateGpt2(*executor, compute, absl::GetFlag(FLAGS_seed)));
  const std::string checkpoint = absl::GetFlag(FLAGS_checkpoint);
  if (!checkpoint.empty()) {
    RETURN_IF_ERROR(ReadFromDirectory(*executor, *model, checkpoint));
    std::cout << "Checkpoint is the initial condition; historical AdamW "
                 "updates are NOT reconstructed.\n";
  }
  ASSIGN_OR_RETURN(auto queries,
                   UploadQueries(*executor, examples, output_tokens,
                                 encoder->eos_token_id()));
  std::vector<TrainingExample> training;
  training.reserve(windows.train_examples);
  for (size_t i = 0; i < windows.train_examples; ++i) {
    const size_t first_logit =
        (examples[i].tokens.size() - 1) * kGpt2PaddedVocabularySize;
    training.push_back(
        {queries[i].inputs, CrossEntropy({0, first_logit}, kGpt2VocabularySize,
                                         *examples[i].next_token)});
  }
  Options options;
  options.steps = steps;
  options.learning_rate = learning_rate;
  options.max_jacobian_bytes = static_cast<size_t>(mib) * kMib;
  options.progress = [](const StepResult& step) {
    std::cout << "Step " << step.step
              << ": mean CE before=" << Mean(step.training_losses)
              << ", max predicted logit change="
              << MaximumMagnitude(step.predicted_delta)
              << ", max absolute step residual="
              << MaximumMagnitude(step.residual) << std::endl;
    return absl::OkStatus();
  };
  ASSIGN_OR_RETURN(auto measured,
                   Run(*executor, *model, training, queries, options));
  std::cout << "Final mean training CE=" << Mean(measured.final_training_losses)
            << "; max absolute cumulative reconstruction residual="
            << MaximumMagnitude(measured.residual) << '\n';
  ExperimentReport report;
  report.checkpoint = checkpoint;
  report.tokenizer_directory = tokenizer_directory;
  report.corpus_path = corpus_path;
  report.seed = absl::GetFlag(FLAGS_seed);
  report.compute_type = compute_name;
  report.windows = windows;
  report.dimensions = {kGpt2VocabularySize,
                       kGpt2PaddedVocabularySize,
                       kGpt2ContextLength,
                       kGpt2TransformerBlockCount,
                       kGpt2ModelWidth,
                       kGpt2AttentionHeads,
                       kGpt2AttentionHeadDimension,
                       kGpt2FeedForwardWidth};
  report.padding_token = encoder->eos_token_id();
  report.max_jacobian_bytes = options.max_jacobian_bytes;
  report.steps = steps;
  report.learning_rate = learning_rate;
  report.examples = std::move(examples);
  report.output_tokens = std::move(output_tokens);
  for (int token : report.output_tokens) {
    ASSIGN_OR_RETURN(auto text, decoder->Decode({&token, 1}));
    report.output_token_text.push_back(std::move(text));
  }
  report.result = std::move(measured);
  ASSIGN_OR_RETURN(auto rendered, RenderReport(report));
  RETURN_IF_ERROR(WriteReportDirectory(output_directory, rendered));
  std::cout
      << "Wrote " << output_directory
      << "/report.json, path_kernel.csv, contributions.csv, predictions.csv\n"
      << "Updated in-memory weights only; no checkpoint was overwritten. "
      << "Unique parameters: " << report.result.parameter_count << '\n';
  return absl::OkStatus();
}

}  // namespace
}  // namespace pluto::llm::path_kernel

int main(int argc, char** argv) {
  const auto positional = absl::ParseCommandLine(argc, argv);
  if (positional.size() != 1) {
    std::cerr << "Use --prompt and --output_dir; positional arguments are not "
                 "accepted.\n";
    return 1;
  }
  const auto status = pluto::llm::path_kernel::RunExperiment();
  if (!status.ok()) {
    std::cerr << status << '\n';
    return 1;
  }
  return 0;
}
