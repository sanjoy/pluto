// Measures whether a trained checkpoint behaves like a suffix-only mechanism.
// Preserved absolute positions avoid conflating context replacement with
// shifted position embeddings. Corrupted histories are still out of
// distribution.
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "src/cuda/executor.h"
#include "src/dataset/compact_vocabulary.h"
#include "src/dataset/dataset.h"
#include "src/dataset/gpt2_tokenizer.h"
#include "src/llm/checkpoint.h"
#include "src/llm/experiments/one_shot_memorizer/checkpoint_probe.h"
#include "src/llm/gpt2.h"
#include "src/util/status_macros.h"
#include "src/util/tee_stream.h"

ABSL_FLAG(std::string, checkpoint, "", "Exact memorized checkpoint directory");
ABSL_FLAG(std::string, tokenizer, "", "Original GPT-2 tokenizer directory");
ABSL_FLAG(std::string, corpus, "testdata/general_facts_dataset.txt",
          "Corpus used for evaluation");
ABSL_FLAG(std::string, output_dir, "", "Fresh report directory");
ABSL_FLAG(int, batch_size, 32, "Independent prefix interventions per batch");
ABSL_FLAG(int, prompt_tokens, 5,
          "Start scoring after this many supplied tokens");
ABSL_FLAG(int, layers, 8, "Checkpoint transformer blocks");
ABSL_FLAG(int, model_width, 16, "Checkpoint residual width");
ABSL_FLAG(int, attention_heads, 1, "Checkpoint attention heads");
ABSL_FLAG(int, feed_forward_width, 64, "Checkpoint inner MLP width");
ABSL_FLAG(std::vector<std::string>, windows,
          (std::vector<std::string>{"0", "1", "3", "5", "9", "12", "16", "24"}),
          "Retained suffix sizes, comma-separated; unmodified control always "
          "runs first");
ABSL_FLAG(std::string, replacement, "both",
          "Prefix replacement: eos, other_sentence, or both; eos also "
          "supports a corpus containing just one sentence");

namespace pluto::llm::one_shot_memorizer {
namespace {
absl::Status RunProbe() {
  namespace fs = std::filesystem;
  const fs::path checkpoint = absl::GetFlag(FLAGS_checkpoint);
  const fs::path output_dir = absl::GetFlag(FLAGS_output_dir);
  const auto tokenizer_path = absl::GetFlag(FLAGS_tokenizer);
  const int prompt_tokens = absl::GetFlag(FLAGS_prompt_tokens);
  const auto replacement_name = absl::GetFlag(FLAGS_replacement);
  std::vector<PrefixReplacement> replacements;
  if (replacement_name == "eos" || replacement_name == "both")
    replacements.push_back(PrefixReplacement::kEos);
  if (replacement_name == "other_sentence" || replacement_name == "both")
    replacements.push_back(PrefixReplacement::kOtherSentence);
  if (replacements.empty())
    return absl::InvalidArgumentError(
        "replacement must be eos, other_sentence, or both");
  if (checkpoint.empty() || output_dir.empty() || tokenizer_path.empty() ||
      prompt_tokens <= 0)
    return absl::InvalidArgumentError(
        "checkpoint, output_dir, tokenizer and positive prompt_tokens "
        "required");
  std::vector<int> windows;
  for (const auto& text : absl::GetFlag(FLAGS_windows)) {
    int window = 0;
    if (!absl::SimpleAtoi(text, &window) || window < 0)
      return absl::InvalidArgumentError("windows must be nonnegative integers");
    windows.push_back(window);
  }
  ASSIGN_OR_RETURN(auto original,
                   tokenizer::Gpt2Tokenizer::Load(tokenizer_path));
  ASSIGN_OR_RETURN(auto compact,
                   tokenizer::CompactVocabularyTokenizer::LoadFromFile(
                       *original, checkpoint / "compact_vocabulary.tsv"));
  if (compact->original_eos_token_id() != original->eos_token_id())
    return absl::InvalidArgumentError("tokenizer EOS differs from checkpoint");
  ASSIGN_OR_RETURN(auto executor, cuda::Executor::Create());
  ASSIGN_OR_RETURN(auto corpus, LoadTextCorpus(absl::GetFlag(FLAGS_corpus)));
  std::vector<std::vector<int>> sentences;
  auto text = corpus.text();
  if (!text.empty() && text.back() == '\n')
    text.remove_suffix(1);
  for (absl::string_view line : absl::StrSplit(text, '\n')) {
    if (!line.empty() && line.back() == '\r')
      line.remove_suffix(1);
    ASSIGN_OR_RETURN(auto ids, compact->Encode(*executor, line));
    sentences.emplace_back(ids.begin(), ids.end());
  }
  const Gpt2Config config{
      .transformer_block_count = absl::GetFlag(FLAGS_layers),
      .model_width = absl::GetFlag(FLAGS_model_width),
      .attention_heads = absl::GetFlag(FLAGS_attention_heads),
      .feed_forward_width = absl::GetFlag(FLAGS_feed_forward_width),
      .vocabulary_size = compact->vocab_size(),
      .pad_vocabulary = false};
  ASSIGN_OR_RETURN(auto model, CreateGpt2(*executor, DataType::BF16, 0, config));
  RETURN_IF_ERROR(
      ReadFromDirectory(*executor, *model, checkpoint.string(), false));
  std::error_code error;
  if (!fs::create_directory(output_dir, error))
    return absl::AlreadyExistsError(
        absl::StrCat("output_dir must be fresh: ", error.message()));
  std::ofstream file(output_dir / "context_probe.tsv");
  if (!file)
    return absl::UnknownError("cannot create probe report");
  util::TeeStream report(std::cout, file);
  report << "# checkpoint=" << checkpoint.string()
         << "\n# corpus=" << absl::GetFlag(FLAGS_corpus)
         << "\n# tokenizer=" << tokenizer_path
         << "\n# batch_size=" << absl::GetFlag(FLAGS_batch_size)
         << "\n# prompt_tokens=" << prompt_tokens
         << "\n# replacement=" << replacement_name
         << "\n# Independent next-token interventions, NOT autoregressive "
            "completion scores.\n"
         << "replacement\twindow\ttargets\tmodified_"
            "targets\twrong\tnonfinite\tseconds\n";
  auto run = [&](int window, PrefixReplacement replacement) -> absl::Status {
    const auto start = std::chrono::steady_clock::now();
    ASSIGN_OR_RETURN(
        auto results,
        ProbeContextWindows(*executor, *model, sentences, compact->vocab_size(),
                            compact->eos_token_id(), prompt_tokens,
                            absl::GetFlag(FLAGS_batch_size), {&window, 1},
                            replacement));
    const auto& result = results.front();
    size_t modified = 0;
    if (window >= 0)
      for (const auto& sentence : sentences) {
        const size_t retained = std::max(window, prompt_tokens - 1);
        if (sentence.size() > retained)
          modified += sentence.size() - retained;
      }
    report << (window == -1                             ? "control"
               : replacement == PrefixReplacement::kEos ? "eos"
                                                        : "other_sentence")
           << '\t' << window << '\t' << result.targets << '\t' << modified
           << '\t' << result.targets - result.correct << '\t'
           << result.nonfinite << '\t'
           << std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                            start)
                  .count()
           << '\n';
    if (!file)
      return absl::UnknownError("cannot write probe report");
    if (window == -1 && result.correct != result.targets)
      return absl::FailedPreconditionError(
          "unaltered checkpoint is not fully memorized under this protocol");
    return absl::OkStatus();
  };
  RETURN_IF_ERROR(run(-1, PrefixReplacement::kEos));
  for (auto replacement : replacements)
    for (int window : windows)
      RETURN_IF_ERROR(run(window, replacement));
  report << "# Complete\n";
  return absl::OkStatus();
}
}  // namespace
}  // namespace pluto::llm::one_shot_memorizer

int main(int argc, char** argv) {
  if (absl::ParseCommandLine(argc, argv).size() != 1) {
    std::cerr << "unexpected positional arguments\n";
    return 1;
  }
  const auto status = pluto::llm::one_shot_memorizer::RunProbe();
  if (!status.ok()) {
    std::cerr << status << '\n';
    return 1;
  }
  return 0;
}
