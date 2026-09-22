// Extracts exact native-BF16 boundary states from a compact GPT-2 checkpoint.
#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <utility>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/escaping.h"
#include "absl/strings/str_cat.h"
#include "src/cuda/executor.h"
#include "src/dataset/compact_vocabulary.h"
#include "src/dataset/gpt2_detokenizer.h"
#include "src/dataset/gpt2_tokenizer.h"
#include "src/dataset/padded_line_dataset.h"
#include "src/llm/checkpoint.h"
#include "src/llm/experiments/memorize_general_facts/discretized_model/capture.h"
#include "src/llm/gpt2.h"
#include "src/util/status_macros.h"

ABSL_FLAG(std::string, checkpoint, "", "Compact GPT-2 checkpoint directory");
ABSL_FLAG(std::string, tokenizer, "",
          "Matching original GPT-2 tokenizer directory");
ABSL_FLAG(std::string, corpus, "testdata/general_facts_dataset.txt",
          "One fact per line, captured in file order");
ABSL_FLAG(std::string, output, "",
          "New JSONL snapshot file; never overwritten");
ABSL_FLAG(int, layers, 8, "Checkpoint transformer block count");
ABSL_FLAG(int, attention_heads, 1,
          "Checkpoint attention head count at width 16");
ABSL_FLAG(int, feed_forward_width, 64, "Checkpoint inner MLP width");
ABSL_FLAG(int, prompt_tokens, 5, "Number of initial prompt tokens");
ABSL_FLAG(int, max_samples, 0,
          "Capture only this many samples; zero captures all");
ABSL_FLAG(
    bool, verify_greedy, true,
    "Verify autonomous completion through EOS and exact prefix BF16 states");

namespace pluto::llm::memorize_general_facts::discretized_model {
namespace {

class ExclusiveOutput {
 public:
  static absl::StatusOr<std::unique_ptr<ExclusiveOutput>> Create(
      const std::string& path) {
    // C11 exclusive mode atomically refuses every existing file, including a
    // symlink, instead of relying on a racy exists-then-open check.
    FILE* file = std::fopen(path.c_str(), "wx");
    if (file == nullptr)
      return absl::FailedPreconditionError(absl::StrCat(
          "cannot create snapshot ", path, ": ", std::strerror(errno)));
    return absl::WrapUnique(new ExclusiveOutput(file));
  }

  ~ExclusiveOutput() {
    if (file_ != nullptr)
      std::fclose(file_);
  }

  absl::Status Write(const std::string& line) {
    if (std::fwrite(line.data(), 1, line.size(), file_) != line.size())
      return absl::DataLossError("writing snapshot failed");
    return absl::OkStatus();
  }

  absl::Status Finish() {
    const int flushed = std::fflush(file_);
    const int closed = std::fclose(std::exchange(file_, nullptr));
    if (flushed != 0 || closed != 0)
      return absl::DataLossError("flushing or closing snapshot failed");
    return absl::OkStatus();
  }

 private:
  explicit ExclusiveOutput(FILE* file) : file_(file) {}
  FILE* file_;
};

absl::StatusOr<std::string> ReadCorpus(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file)
    return absl::NotFoundError(absl::StrCat("cannot open corpus: ", path));
  std::ostringstream text;
  text << file.rdbuf();
  if (file.bad() || text.bad())
    return absl::DataLossError(absl::StrCat("cannot read corpus: ", path));
  return text.str();
}

absl::StatusOr<std::string> HeaderJson(
    const tokenizer::CompactVocabularyTokenizer& vocabulary,
    const tokenizer::Gpt2Detokenizer& detokenizer,
    const CaptureOptions& options) {
  std::string line = absl::StrCat(
      "{\"schema\":1,\"width\":", kCaptureWidth, ",\"layers\":", options.layers,
      ",\"vocab_size\":", options.vocab_size,
      ",\"eos_token\":", options.eos_token,
      ",\"prompt_tokens\":", options.prompt_tokens, ",\"vocabulary\":[");
  bool first = true;
  for (int original : vocabulary.original_token_ids()) {
    ASSIGN_OR_RETURN(auto bytes, detokenizer.Decode({&original, 1}));
    absl::StrAppend(&line, first ? "" : ",", "{\"original_id\":", original,
                    ",\"hex\":\"", absl::BytesToHexString(bytes), "\"}");
    first = false;
  }
  line += "]}\n";
  return line;
}

std::string SampleJson(const CapturedSample& sample) {
  std::string line = "{\"tokens\":[";
  for (size_t i = 0; i < sample.tokens.size(); ++i)
    absl::StrAppend(&line, i == 0 ? "" : ",", sample.tokens[i]);
  line += "],\"predictions\":[";
  for (size_t i = 0; i < sample.predictions.size(); ++i)
    absl::StrAppend(&line, i == 0 ? "" : ",", sample.predictions[i]);
  line += "],\"boundaries\":[";
  for (size_t stage = 0; stage < sample.boundaries.size(); ++stage) {
    line += stage == 0 ? "[" : ",[";
    for (size_t row = 0; row < sample.boundaries[stage].size(); ++row) {
      line += row == 0 ? "[" : ",[";
      for (size_t channel = 0; channel < kCaptureWidth; ++channel)
        absl::StrAppend(&line, channel == 0 ? "" : ",",
                        sample.boundaries[stage][row][channel]);
      line += "]";
    }
    line += "]";
  }
  line += "]}\n";
  return line;
}

absl::Status Run() {
  const std::string checkpoint = absl::GetFlag(FLAGS_checkpoint);
  const std::string tokenizer_path = absl::GetFlag(FLAGS_tokenizer);
  const std::string corpus_path = absl::GetFlag(FLAGS_corpus);
  const std::string output_path = absl::GetFlag(FLAGS_output);
  const int max_samples = absl::GetFlag(FLAGS_max_samples);
  const int prompt_tokens = absl::GetFlag(FLAGS_prompt_tokens);
  if (checkpoint.empty() || tokenizer_path.empty() || corpus_path.empty() ||
      output_path.empty() || max_samples < 0 || prompt_tokens <= 0 ||
      prompt_tokens > kCaptureContext)
    return absl::InvalidArgumentError(
        "checkpoint, tokenizer, corpus, and output are required; max_samples "
        "must be nonnegative and prompt_tokens must be in [1,1024]");
  ASSIGN_OR_RETURN(auto original,
                   tokenizer::Gpt2Tokenizer::Load(tokenizer_path));
  ASSIGN_OR_RETURN(auto detokenizer,
                   tokenizer::Gpt2Detokenizer::Load(tokenizer_path));
  ASSIGN_OR_RETURN(auto vocabulary,
                   tokenizer::CompactVocabularyTokenizer::LoadFromFile(
                       *original, std::filesystem::path(checkpoint) /
                                      "compact_vocabulary.tsv"));
  if (vocabulary->original_eos_token_id() != original->eos_token_id() ||
      original->vocab_size() != detokenizer->vocab_size() ||
      original->eos_token_id() != detokenizer->eos_token_id())
    return absl::InvalidArgumentError("checkpoint tokenizer identity mismatch");
  const CaptureOptions options{.layers = absl::GetFlag(FLAGS_layers),
                               .vocab_size = vocabulary->vocab_size(),
                               .eos_token = vocabulary->eos_token_id(),
                               .prompt_tokens = prompt_tokens};
  const Gpt2Config config{
      .transformer_block_count = options.layers,
      .model_width = kCaptureWidth,
      .attention_heads = absl::GetFlag(FLAGS_attention_heads),
      .feed_forward_width = absl::GetFlag(FLAGS_feed_forward_width),
      .vocabulary_size = options.vocab_size,
      .pad_vocabulary = false};
  RETURN_IF_ERROR(config.Validate());
  ASSIGN_OR_RETURN(auto corpus, ReadCorpus(corpus_path));
  ASSIGN_OR_RETURN(auto executor, cuda::Executor::Create());
  ASSIGN_OR_RETURN(auto data, PaddedLineDataSetIterator::Create(
                                  *executor, corpus, *vocabulary,
                                  {.batch_size = 1,
                                   .context_length = kCaptureContext,
                                   .prompt_tokens = options.prompt_tokens,
                                   .eos_token = options.eos_token,
                                   .shuffle = false}));
  ASSIGN_OR_RETURN(auto model, CreateGpt2(*executor, DataType::BF16, 0, config));
  RETURN_IF_ERROR(ReadFromDirectory(*executor, *model, checkpoint,
                                    /*allow_prefix=*/false));
  ASSIGN_OR_RETURN(auto header, HeaderJson(*vocabulary, *detokenizer, options));
  ASSIGN_OR_RETURN(auto output, ExclusiveOutput::Create(output_path));
  RETURN_IF_ERROR(output->Write(header));
  const size_t sample_count =
      max_samples == 0
          ? data->sample_count()
          : std::min(data->sample_count(), static_cast<size_t>(max_samples));
  const bool verify_greedy = absl::GetFlag(FLAGS_verify_greedy);
  size_t rows = 0;
  for (size_t sample = 0; sample < sample_count; ++sample) {
    ASSIGN_OR_RETURN(
        auto captured,
        CaptureSample(*executor, *model, data->sample_tokens(sample), options));
    const auto validated = ValidateCapturedPredictions(captured, options);
    if (!validated.ok())
      return absl::Status(
          validated.code(),
          absl::StrCat("sample ", sample, ": ", validated.message()));
    if (verify_greedy) {
      const auto verified =
          VerifyGreedyCapture(*executor, *model, captured, options);
      if (!verified.ok())
        return absl::Status(
            verified.code(),
            absl::StrCat("sample ", sample, ": ", verified.message()));
    }
    RETURN_IF_ERROR(output->Write(SampleJson(captured)));
    rows += captured.tokens.size();
    if ((sample + 1) % 32 == 0 || sample + 1 == sample_count)
      std::cerr << "Captured " << sample + 1 << '/' << sample_count
                << " samples, " << rows << " real rows"
                << (verify_greedy ? "; greedy/EOS/BF16 prefixes verified\n"
                                  : "; greedy verification disabled\n");
  }
  RETURN_IF_ERROR(output->Finish());
  return executor->Synchronize();
}

}  // namespace
}  // namespace pluto::llm::memorize_general_facts::discretized_model

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  const auto status =
      pluto::llm::memorize_general_facts::discretized_model::Run();
  if (!status.ok()) {
    std::cerr << status << '\n';
    return 1;
  }
  return 0;
}
