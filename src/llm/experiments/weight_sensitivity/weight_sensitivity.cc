// One-at-a-time randomized parameter interventions on a memorized GPT-2 model.
// Checkpoint files are read-only; mutations affect this process's GPU copy
// only.
#include <cuda_runtime.h>
#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/strings/str_cat.h"
#include "src/cuda/executor.h"
#include "src/cuda/page_locked_host_array.h"
#include "src/dataset/compact_vocabulary.h"
#include "src/dataset/dataset.h"
#include "src/dataset/gpt2_tokenizer.h"
#include "src/dataset/padded_line_dataset.h"
#include "src/llm/checkpoint.h"
#include "src/llm/experiments/weight_sensitivity/evaluation.h"
#include "src/llm/experiments/weight_sensitivity/html.h"
#include "src/llm/experiments/weight_sensitivity/runner.h"
#include "src/llm/experiments/weight_sensitivity/weights.h"
#include "src/llm/gpt2.h"
#include "src/util/status_macros.h"

ABSL_FLAG(std::string, checkpoint, "", "Checkpoint directory to analyze");
ABSL_FLAG(std::string, tokenizer, "", "Original GPT-2 tokenizer directory");
ABSL_FLAG(std::string, corpus, "testdata/general_facts_dataset.txt",
          "One complete fact per line");
ABSL_FLAG(std::string, output, "", "Fresh self-contained HTML output file");
ABSL_FLAG(int, layers, 8, "Transformer blocks in the checkpoint");
ABSL_FLAG(int, model_width, 16, "Residual/embedding width");
ABSL_FLAG(int, attention_heads, 1, "Attention heads per block");
ABSL_FLAG(int, feed_forward_width, 64, "Inner MLP width");
ABSL_FLAG(int, batch_size, 32, "Independent sentences evaluated per batch");
ABSL_FLAG(int, prompt_tokens, 5,
          "Initial corpus tokens supplied as the prompt");
ABSL_FLAG(int, expected_samples, 1024, "Expected number of corpus sentences");
ABSL_FLAG(uint64_t, seed, 1337, "Base seed for independent per-target noise");
ABSL_FLAG(int, trials, 1, "Independent noise draws per logical tensor");
ABSL_FLAG(double, noise_scale, 1.0, "Noise standard deviation / original RMS");
ABSL_FLAG(double, zero_rms_stddev, 0.02,
          "RMS fallback for all-zero targets, before multiplying noise_scale");
ABSL_FLAG(
    std::string, target_filter, "",
    "Optional case-sensitive tensor-name substring; default analyzes all");

namespace pluto::llm::weight_sensitivity {
namespace {
namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

// Stable mixing prevents a filtered run from changing a target's random draw.
uint64_t NoiseSeed(uint64_t base, size_t target, int trial) {
  uint64_t value =
      base ^ ((uint64_t{target} + 1) * 0x9e3779b97f4a7c15ULL) ^
      ((uint64_t{static_cast<uint32_t>(trial)} + 1) * 0xd1b54a32d192ed03ULL);
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31);
}

absl::Status IoError(absl::string_view operation) {
  return absl::UnknownError(
      absl::StrCat(operation, ": ", std::strerror(errno)));
}

// Claim a new path exclusively before any experiment. Later progress reports
// replace only this owned file; an existing user's report is never truncated.
absl::Status ClaimOutput(const fs::path& output) {
  std::error_code error;
  if (!output.parent_path().empty())
    fs::create_directories(output.parent_path(), error);
  if (error)
    return absl::UnknownError(error.message());
  const int descriptor =
      open(output.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0644);
  if (descriptor < 0) {
    if (errno == EEXIST)
      return absl::AlreadyExistsError("output must be a fresh HTML file");
    return IoError("create report");
  }
  if (close(descriptor) != 0)
    return IoError("close report");
  return absl::OkStatus();
}

// Publish after every completed intervention, so a long/aborted run still has
// a readable partial table. The rename never exposes half-written HTML.
absl::Status SaveReport(const fs::path& output,
                        const SensitivityReport& report) {
  ASSIGN_OR_RETURN(auto html, RenderHtml(report));
  std::string temporary = output.string() + ".XXXXXX";
  const int descriptor = mkstemp(temporary.data());
  if (descriptor < 0)
    return IoError("create report staging file");
  size_t offset = 0;
  absl::Status status;
  while (offset < html.size()) {
    const ssize_t count =
        write(descriptor, html.data() + offset, html.size() - offset);
    if (count < 0 && errno == EINTR)
      continue;
    if (count <= 0) {
      status = IoError("write report");
      break;
    }
    offset += count;
  }
  if (close(descriptor) != 0)
    status.Update(IoError("close report staging file"));
  if (status.ok() && rename(temporary.c_str(), output.c_str()) != 0)
    status = IoError("publish report");
  if (!status.ok())
    unlink(temporary.c_str());
  return status;
}

absl::Status Run() {
  SensitivityReport report;
  report.checkpoint = absl::GetFlag(FLAGS_checkpoint);
  report.corpus = absl::GetFlag(FLAGS_corpus);
  report.prompt_tokens = absl::GetFlag(FLAGS_prompt_tokens);
  report.batch_size = absl::GetFlag(FLAGS_batch_size);
  report.seed = absl::GetFlag(FLAGS_seed);
  report.noise_scale = absl::GetFlag(FLAGS_noise_scale);
  report.zero_rms_stddev = absl::GetFlag(FLAGS_zero_rms_stddev);
  report.trials = absl::GetFlag(FLAGS_trials);
  report.target_filter = absl::GetFlag(FLAGS_target_filter);
  const std::string tokenizer_path = absl::GetFlag(FLAGS_tokenizer);
  const fs::path output = absl::GetFlag(FLAGS_output);
  report.expected_samples = absl::GetFlag(FLAGS_expected_samples);
  const int expected_samples = report.expected_samples;
  if (report.checkpoint.empty() || report.corpus.empty() ||
      tokenizer_path.empty() || output.empty())
    return absl::InvalidArgumentError(
        "--checkpoint, --tokenizer, --corpus, and --output are required");
  if (report.batch_size <= 0 || report.prompt_tokens <= 0 ||
      report.prompt_tokens >= kGpt2ContextLength || expected_samples <= 0 ||
      report.trials <= 0 || !std::isfinite(report.noise_scale) ||
      report.noise_scale <= 0 || !std::isfinite(report.zero_rms_stddev) ||
      report.zero_rms_stddev <= 0)
    return absl::InvalidArgumentError(
        "invalid batch/prompt/trial/noise options");
  ASSIGN_OR_RETURN(auto original_tokenizer,
                   tokenizer::Gpt2Tokenizer::Load(tokenizer_path));
  ASSIGN_OR_RETURN(auto tokenizer,
                   tokenizer::CompactVocabularyTokenizer::LoadFromFile(
                       *original_tokenizer,
                       fs::path(report.checkpoint) / "compact_vocabulary.tsv"));
  if (tokenizer->original_eos_token_id() != original_tokenizer->eos_token_id())
    return absl::InvalidArgumentError("checkpoint tokenizer EOS mismatch");
  report.config = {
      .transformer_block_count = absl::GetFlag(FLAGS_layers),
      .model_width = absl::GetFlag(FLAGS_model_width),
      .attention_heads = absl::GetFlag(FLAGS_attention_heads),
      .feed_forward_width = absl::GetFlag(FLAGS_feed_forward_width),
      .vocabulary_size = tokenizer->vocab_size(),
      .pad_vocabulary = false};
  ASSIGN_OR_RETURN(auto targets, DescribeWeights(report.config));
  std::vector<size_t> selected;
  for (size_t index = 0; index < targets.size(); ++index)
    if (targets[index].name.find(report.target_filter) != std::string::npos)
      selected.push_back(index);
  if (selected.empty())
    return absl::InvalidArgumentError("target_filter matched no tensors");
  if (selected.size() > std::numeric_limits<size_t>::max() / report.trials)
    return absl::InvalidArgumentError("too many requested trials");
  report.planned_results = selected.size() * report.trials;
  ASSIGN_OR_RETURN(auto corpus, LoadTextCorpus(report.corpus));
  ASSIGN_OR_RETURN(auto executor, cuda::Executor::Create());
  ASSIGN_OR_RETURN(auto data, PaddedLineDataSetIterator::Create(
                                  *executor, corpus.text(), *tokenizer,
                                  {.batch_size = report.batch_size,
                                   .context_length = kGpt2ContextLength,
                                   .prompt_tokens = report.prompt_tokens,
                                   .eos_token = tokenizer->eos_token_id(),
                                   .shuffle = false}));
  if (data->sample_count() != static_cast<size_t>(expected_samples))
    return absl::InvalidArgumentError(
        absl::StrCat("expected ", expected_samples, " corpus sentences; got ",
                     data->sample_count()));
  ASSIGN_OR_RETURN(auto model,
                   CreateGpt2(*executor, DataType::BF16, 0, report.config));
  RETURN_IF_ERROR(ReadFromDirectory(*executor, *model, report.checkpoint,
                                    /*allow_prefix=*/false));
  std::vector<Buffer> weights;
  absl::flat_hash_set<const void*> seen;
  for (const auto& weight : model->weights())
    if (seen.insert(weight.data()).second)
      weights.push_back(weight);
  RETURN_IF_ERROR(ValidateWeights(report.config, weights));
  std::vector<cuda::PageLockedHostArray<float>> originals;
  for (const auto& weight : weights) {
    ASSIGN_OR_RETURN(auto host,
                     cuda::PageLockedHostArray<float>::Allocate(
                         *executor, weight.size_bytes() / sizeof(float)));
    RETURN_IF_ERROR(cuda::CudaStatus(
        cudaMemcpyAsync(host.data(), weight.data(), weight.size_bytes(),
                        cudaMemcpyDeviceToHost, executor->stream()),
        "snapshot original weight"));
    originals.push_back(std::move(host));
  }
  RETURN_IF_ERROR(executor->Synchronize());
  auto evaluate = [&] {
    return EvaluateCompletions(*executor, *model, *data,
                               tokenizer->vocab_size());
  };
  std::cout << "Checking uncorrupted baseline..." << std::endl;
  ASSIGN_OR_RETURN(report.baseline, evaluate());
  if (report.baseline.token_errors != 0)
    return absl::FailedPreconditionError(absl::StrCat(
        "checkpoint is not fully memorized under this protocol: ",
        report.baseline.token_errors, " / ", report.baseline.scored_tokens,
        " target predictions incorrect"));
  RETURN_IF_ERROR(ClaimOutput(output));
  RETURN_IF_ERROR(SaveReport(output, report));
  std::cout << "Baseline: " << report.baseline.exact.size() << " / "
            << report.baseline.exact.size() << " exact completions; "
            << report.planned_results << " interventions.\n"
            << std::flush;
  for (size_t index : selected) {
    const auto& target = targets[index];
    for (int trial = 0; trial < report.trials; ++trial) {
      const uint64_t seed = NoiseSeed(report.seed, index, trial);
      const auto start = Clock::now();
      ASSIGN_OR_RETURN(
          auto result,
          EvaluateCorruption(*executor, weights[target.checkpoint_index],
                             target, originals[target.checkpoint_index], seed,
                             report.noise_scale, report.zero_rms_stddev,
                             evaluate));
      const size_t wrong = std::count(result.scores.exact.begin(),
                                      result.scores.exact.end(), uint8_t{0});
      report.results.push_back(
          {.target = target,
           .trial = trial,
           .seed = seed,
           .noise_stddev = result.noise_stddev,
           .seconds =
               std::chrono::duration<double>(Clock::now() - start).count(),
           .scores = std::move(result.scores)});
      RETURN_IF_ERROR(SaveReport(output, report));
      std::cout << '[' << report.results.size() << '/' << report.planned_results
                << "] " << target.name << ": " << wrong << " / "
                << report.baseline.exact.size() << " completions wrong ("
                << report.results.back().seconds << " s)\n"
                << std::flush;
    }
  }
  ASSIGN_OR_RETURN(auto restored, evaluate());
  if (restored.exact != report.baseline.exact ||
      restored.token_errors != report.baseline.token_errors ||
      restored.scored_tokens != report.baseline.scored_tokens ||
      restored.nonfinite_rows != report.baseline.nonfinite_rows)
    return absl::DataLossError("restored model does not reproduce baseline");
  report.complete = true;
  RETURN_IF_ERROR(SaveReport(output, report));
  std::cout << "Restored baseline verified. Report: " << output << '\n';
  return absl::OkStatus();
}

}  // namespace
}  // namespace pluto::llm::weight_sensitivity

int main(int argc, char** argv) {
  const auto positional = absl::ParseCommandLine(argc, argv);
  if (positional.size() != 1) {
    std::cerr << "Unexpected positional arguments\n";
    return 1;
  }
  const auto status = pluto::llm::weight_sensitivity::Run();
  if (!status.ok()) {
    std::cerr << status << '\n';
    return 1;
  }
  return 0;
}
