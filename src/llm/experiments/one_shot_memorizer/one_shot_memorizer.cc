// Dataset-to-model construction. Neither backend consults a trained checkpoint.
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <string>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "src/cuda/executor.h"
#include "src/dataset/dataset.h"
#include "src/dataset/gpt2_detokenizer.h"
#include "src/dataset/gpt2_tokenizer.h"
#include "src/llm/experiments/one_shot_memorizer/automaton.h"
#include "src/llm/experiments/one_shot_memorizer/context_analysis.h"
#include "src/llm/experiments/one_shot_memorizer/model_io.h"
#include "src/util/status_macros.h"

ABSL_FLAG(std::string, mode, "", "compile or infer");
ABSL_FLAG(std::string, tokenizer, "", "Original GPT-2 tokenizer directory");
ABSL_FLAG(std::string, corpus, "testdata/general_facts_dataset.txt",
          "One fact per line; compile mode only");
ABSL_FLAG(std::string, output_dir, "",
          "Fresh artifact directory; compile only");
ABSL_FLAG(std::string, model_file, "", "Compiled model weights; infer only");
ABSL_FLAG(std::string, prompt, "", "Prompt text; infer only");
ABSL_FLAG(int, prompt_tokens, 5,
          "Supplied prefix length for corpus verification");
ABSL_FLAG(int, max_new_tokens, 1024,
          "Inference generation limit, including EOS");

namespace pluto::llm::one_shot_memorizer {
namespace {
namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

absl::Status Write(const fs::path& path, absl::string_view contents) {
  std::ofstream output(path, std::ios::binary);
  output.write(contents.data(), contents.size());
  output.close();
  if (!output)
    return absl::UnknownError(absl::StrCat("cannot write ", path.string()));
  return absl::OkStatus();
}

absl::StatusOr<std::vector<std::vector<int>>> TokenizeLines(
    cuda::Executor& executor, const tokenizer::Gpt2Tokenizer& tokenizer,
    absl::string_view text, int prompt_tokens) {
  if (!text.empty() && text.back() == '\n')
    text.remove_suffix(1);
  std::vector<std::vector<int>> sentences;
  for (absl::string_view line : absl::StrSplit(text, '\n')) {
    if (!line.empty() && line.back() == '\r')
      line.remove_suffix(1);
    ASSIGN_OR_RETURN(auto tokens, tokenizer.Encode(executor, line));
    if (tokens.size() < static_cast<size_t>(prompt_tokens))
      return absl::InvalidArgumentError(
          "corpus line is shorter than the prompt");
    sentences.emplace_back(tokens.begin(), tokens.end());
  }
  return sentences;
}

absl::Status Compile(cuda::Executor& executor,
                     const tokenizer::Gpt2Tokenizer& tokenizer) {
  const int prompt_tokens = absl::GetFlag(FLAGS_prompt_tokens);
  const fs::path directory = absl::GetFlag(FLAGS_output_dir);
  if (directory.empty() || prompt_tokens <= 0 ||
      !absl::GetFlag(FLAGS_model_file).empty() ||
      !absl::GetFlag(FLAGS_prompt).empty())
    return absl::InvalidArgumentError(
        "compile needs output_dir and positive prompt_tokens; no "
        "model_file/prompt");
  ASSIGN_OR_RETURN(auto corpus, LoadTextCorpus(absl::GetFlag(FLAGS_corpus)));
  ASSIGN_OR_RETURN(auto sentences, TokenizeLines(executor, tokenizer,
                                                 corpus.text(), prompt_tokens));
  const auto start = Clock::now();
  ASSIGN_OR_RETURN(auto model, BuildModel(sentences, tokenizer.vocab_size(),
                                          tokenizer.eos_token_id()));
  const double compile_seconds =
      std::chrono::duration<double>(Clock::now() - start).count();
  ASSIGN_OR_RETURN(auto bytes, SerializeModel(model));
  // Verify the artifact representation, not only the builder's live object.
  ASSIGN_OR_RETURN(auto restored, DeserializeModel(bytes));
  size_t exact = 0, target_count = 0, input_tokens = 0;
  std::set<int> active_tokens{tokenizer.eos_token_id()};
  for (const auto& sentence : sentences) {
    input_tokens += sentence.size();
    active_tokens.insert(sentence.begin(), sentence.end());
    const auto prefix = absl::MakeConstSpan(sentence).first(prompt_tokens);
    std::vector<int> expected(sentence.begin() + prompt_tokens, sentence.end());
    expected.push_back(tokenizer.eos_token_id());
    target_count += expected.size();
    ASSIGN_OR_RETURN(auto actual,
                     GreedyContinuation(restored, prefix, expected.size() + 1));
    exact += actual == expected;
  }
  ASSIGN_OR_RETURN(
      auto context,
      AnalyzeContexts(sentences, {.prompt_tokens = prompt_tokens,
                                  .eos_token = tokenizer.eos_token_id()}));
  size_t edges = 0;
  for (const auto& state : model.states)
    edges += state.transitions.size();
  std::string summary = absl::StrCat(
      "metric\tvalue\ncorpus\t", absl::GetFlag(FLAGS_corpus), "\ntokenizer\t",
      absl::GetFlag(FLAGS_tokenizer), "\nsentences\t", sentences.size(),
      "\nprompt_tokens\t", prompt_tokens, "\ninput_tokens\t", input_tokens,
      "\nactive_tokens_including_eos\t", active_tokens.size(),
      "\nscored_targets_including_eos\t", target_count,
      "\nexact_autoregressive_completions\t", exact, "\ntrie_states\t",
      model.trie_state_count, "\ncompacted_states\t", model.states.size(),
      "\ntransition_nonzeros\t", edges, "\nserialized_bytes\t", bytes.size(),
      "\nconstruction_seconds\t", compile_seconds,
      "\nshortest_exact_suffix_window\t", context.shortest_exact_window, "\n");
  std::string windows =
      "window\tcontexts\tconflicting_contexts\tirreducible_errors\ttargets\n";
  for (const auto& window : context.windows)
    absl::StrAppend(&windows, window.window, "\t", window.contexts, "\t",
                    window.conflicting_contexts, "\t",
                    window.irreducible_top1_errors, "\t", window.targets, "\n");
  std::string targets =
      "sentence\ttarget_position\ttoken\tsufficient_window\tunique_window\n";
  for (const auto& target : context.targets)
    absl::StrAppend(&targets, target.sentence_index, "\t", target.target_index,
                    "\t", target.target_token, "\t",
                    target.shortest_sufficient_window, "\t",
                    target.shortest_unique_window, "\n");
  std::error_code error;
  if (!fs::create_directory(directory, error))
    return absl::AlreadyExistsError(absl::StrCat(
        "output_dir must be a fresh directory with an existing parent: ",
        error.message()));
  RETURN_IF_ERROR(Write(directory / "automaton.weights", bytes));
  RETURN_IF_ERROR(Write(directory / "summary.tsv", summary));
  RETURN_IF_ERROR(Write(directory / "context_windows.tsv", windows));
  RETURN_IF_ERROR(Write(directory / "target_contexts.tsv", targets));
  std::cout << summary << "Artifacts: " << directory << '\n';
  if (exact != sentences.size())
    return absl::FailedPreconditionError(
        "not all corpus completions are deterministic from the supplied "
        "prompt");
  return absl::OkStatus();
}

absl::Status Infer(cuda::Executor& executor,
                   const tokenizer::Gpt2Tokenizer& tokenizer) {
  const std::string prompt = absl::GetFlag(FLAGS_prompt);
  const int max_new = absl::GetFlag(FLAGS_max_new_tokens);
  if (prompt.empty() || max_new < 0 ||
      absl::GetFlag(FLAGS_model_file).empty() ||
      !absl::GetFlag(FLAGS_output_dir).empty())
    return absl::InvalidArgumentError(
        "infer needs model_file/prompt and nonnegative max_new_tokens, no "
        "output_dir");
  ASSIGN_OR_RETURN(auto file, LoadTextCorpus(absl::GetFlag(FLAGS_model_file)));
  ASSIGN_OR_RETURN(auto model, DeserializeModel(file.text()));
  if (model.vocabulary_size != tokenizer.vocab_size() ||
      model.eos_token_id != tokenizer.eos_token_id())
    return absl::InvalidArgumentError("tokenizer vocabulary/EOS mismatch");
  ASSIGN_OR_RETURN(auto prefix, tokenizer.Encode(executor, prompt));
  ASSIGN_OR_RETURN(auto suffix,
                   GreedyContinuation(model, prefix.span(), max_new));
  if (!suffix.empty() && suffix.back() == model.eos_token_id)
    suffix.pop_back();
  ASSIGN_OR_RETURN(auto decoder, tokenizer::Gpt2Detokenizer::Load(
                                     absl::GetFlag(FLAGS_tokenizer)));
  ASSIGN_OR_RETURN(auto text, decoder->Decode(suffix));
  std::cout << prompt << text << '\n';
  return absl::OkStatus();
}

absl::Status Run() {
  const auto mode = absl::GetFlag(FLAGS_mode);
  if (mode != "compile" && mode != "infer")
    return absl::InvalidArgumentError("--mode must be compile or infer");
  if (absl::GetFlag(FLAGS_tokenizer).empty())
    return absl::InvalidArgumentError("--tokenizer is required");
  ASSIGN_OR_RETURN(auto tokenizer, tokenizer::Gpt2Tokenizer::Load(
                                       absl::GetFlag(FLAGS_tokenizer)));
  ASSIGN_OR_RETURN(auto executor, cuda::Executor::Create());
  return mode == "compile" ? Compile(*executor, *tokenizer)
                           : Infer(*executor, *tokenizer);
}
}  // namespace
}  // namespace pluto::llm::one_shot_memorizer

int main(int argc, char** argv) {
  if (absl::ParseCommandLine(argc, argv).size() != 1) {
    std::cerr << "unexpected positional arguments\n";
    return 1;
  }
  const auto status = pluto::llm::one_shot_memorizer::Run();
  if (!status.ok()) {
    std::cerr << status << '\n';
    return 1;
  }
  return 0;
}
