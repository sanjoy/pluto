#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/memory/memory.h"
#include "rules_cc/cc/runfiles/runfiles.h"
#include "src/llm/experiments/memorize_general_facts/discretized_model/discrete_model_generator/generate_discretized_model.h"
#include "src/llm/experiments/memorize_general_facts/discretized_model/discrete_model_generator/generator_report.h"

ABSL_FLAG(std::string, checkpoint, "",
          "Source compact GPT-2 checkpoint directory");
ABSL_FLAG(std::string, tokenizer, "",
          "Matching original GPT-2 tokenizer directory");
ABSL_FLAG(std::string, corpus, "testdata/general_facts_dataset.txt",
          "Corpus, one fact per line");
ABSL_FLAG(std::string, output, "",
          "Fresh directory for generated CPU-only C++");
ABSL_FLAG(int, layers, 8, "Checkpoint transformer block count");
ABSL_FLAG(int, model_width, 16, "Checkpoint residual-stream width");
ABSL_FLAG(int, context_length, 1024,
          "Checkpoint learned position count and padded sequence length");
ABSL_FLAG(int, attention_heads, 1, "Checkpoint attention head count");
ABSL_FLAG(int, feed_forward_width, 64, "Checkpoint inner MLP width");
ABSL_FLAG(int, prompt_tokens, 5, "Number of input prompt tokens per sentence");
ABSL_FLAG(int, expected_samples, 1024, "Required number of corpus samples");
ABSL_FLAG(bool, verify_greedy, true,
          "Check native autonomous completion and exact prefix BF16 states");
ABSL_FLAG(bool, state_index, false,
          "Write inspection-only state membership and examples");
ABSL_FLAG(bool, compaction, false,
          "Compact compatible states within each residual boundary");
ABSL_FLAG(bool, compact_transitions, false,
          "Emit compact transition programs instead of private tables");
ABSL_FLAG(int, compaction_neighbors, 8,
          "Near-neighbor candidates per state in non-exhaustive search");
ABSL_FLAG(int, compaction_max_passes, 100, "Maximum state compaction passes");
ABSL_FLAG(int64_t, compaction_max_attempts, -1,
          "Maximum compaction attempts; -1 means unlimited");
ABSL_FLAG(int64_t, compaction_exhaustive_pair_limit, 1000000,
          "Pair count threshold for exhaustive search");

int main(int argc, char** argv) {
  namespace generator = pluto::llm::discretized::generator;
  // Resolve declared data before changing to the caller's directory. Neither
  // generation nor formatting depends on the repository's working directory.
  std::string error;
  auto runfiles = absl::WrapUnique(rules_cc::cc::runfiles::Runfiles::Create(
      argv[0], BAZEL_CURRENT_REPOSITORY, &error));
  if (!runfiles) {
    std::cerr << "cannot locate generator runfiles: " << error << '\n';
    return 1;
  }
  std::filesystem::path style = runfiles->Rlocation("_main/.clang-format");
  std::error_code style_error;
  style = std::filesystem::absolute(style, style_error);
  if (style_error) {
    std::cerr << "cannot resolve formatting configuration: "
              << style_error.message() << '\n';
    return 1;
  }
  if (const char* directory = std::getenv("BUILD_WORKING_DIRECTORY")) {
    std::error_code code;
    std::filesystem::current_path(directory, code);
    if (code) {
      std::cerr << "cannot enter caller directory: " << code.message() << '\n';
      return 1;
    }
  }
  auto arguments = absl::ParseCommandLine(argc, argv);
  if (arguments.size() != 1) {
    std::cerr << "unexpected positional argument\n";
    return 1;
  }
  generator::GeneratorOptions options;
  options.recorder.checkpoint = absl::GetFlag(FLAGS_checkpoint);
  options.recorder.tokenizer = absl::GetFlag(FLAGS_tokenizer);
  options.recorder.corpus = absl::GetFlag(FLAGS_corpus);
  options.output = absl::GetFlag(FLAGS_output);
  options.clang_format_config = style;
  options.recorder.layers = absl::GetFlag(FLAGS_layers);
  options.recorder.model_width = absl::GetFlag(FLAGS_model_width);
  options.recorder.context_length = absl::GetFlag(FLAGS_context_length);
  options.recorder.attention_heads = absl::GetFlag(FLAGS_attention_heads);
  options.recorder.feed_forward_width = absl::GetFlag(FLAGS_feed_forward_width);
  options.recorder.prompt_tokens = absl::GetFlag(FLAGS_prompt_tokens);
  options.recorder.expected_samples = absl::GetFlag(FLAGS_expected_samples);
  options.recorder.verify_greedy = absl::GetFlag(FLAGS_verify_greedy);
  options.state_index = absl::GetFlag(FLAGS_state_index);
  options.compaction = absl::GetFlag(FLAGS_compaction);
  options.compact_transitions = absl::GetFlag(FLAGS_compact_transitions);
  options.compaction_options.neighbors =
      absl::GetFlag(FLAGS_compaction_neighbors);
  options.compaction_options.max_passes =
      absl::GetFlag(FLAGS_compaction_max_passes);
  options.compaction_options.exhaustive_pair_limit =
      absl::GetFlag(FLAGS_compaction_exhaustive_pair_limit);
  int64_t attempts = absl::GetFlag(FLAGS_compaction_max_attempts);
  if (attempts < -1) {
    std::cerr << "compaction_max_attempts must be nonnegative or -1\n";
    return 1;
  }
  if (attempts >= 0)
    options.compaction_options.max_attempts = attempts;
  options.progress = [](const generator::ProgressEvent& progress) {
    std::cout << generator::FormatProgress(progress) << std::endl;
  };
  auto result = generator::Generate(options);
  if (!result.ok()) {
    std::cerr << "generation failed: " << result.status() << '\n';
    return 1;
  }
  return 0;
}
