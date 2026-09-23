// Measures query-position residual mediation under earlier-prefix corruption.
// Every site is patched independently in both directions; successful late
// patches can transfer an already-computed answer and do not locate storage.
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "src/cuda/executor.h"
#include "src/dataset/compact_vocabulary.h"
#include "src/dataset/dataset.h"
#include "src/dataset/gpt2_tokenizer.h"
#include "src/llm/checkpoint.h"
#include "src/llm/experiments/one_shot_memorizer/causal_trace.h"
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
ABSL_FLAG(int, retained_tokens, 9,
          "Keep this many final prefix tokens; replace earlier tokens using "
          "the next corpus sentence");
ABSL_FLAG(int, layers, 8, "Checkpoint transformer blocks");
ABSL_FLAG(int, model_width, 16, "Checkpoint residual width");
ABSL_FLAG(int, attention_heads, 1, "Checkpoint attention heads");
ABSL_FLAG(int, feed_forward_width, 64, "Checkpoint inner MLP width");

namespace pluto::llm::one_shot_memorizer {
namespace {

absl::Status RunCausalTrace() {
  namespace fs = std::filesystem;
  const fs::path checkpoint = absl::GetFlag(FLAGS_checkpoint);
  const fs::path output_dir = absl::GetFlag(FLAGS_output_dir);
  const auto tokenizer_path = absl::GetFlag(FLAGS_tokenizer);
  const int prompt_tokens = absl::GetFlag(FLAGS_prompt_tokens);
  const int batch_size = absl::GetFlag(FLAGS_batch_size);
  const int retained_tokens = absl::GetFlag(FLAGS_retained_tokens);
  const int layers = absl::GetFlag(FLAGS_layers);
  if (checkpoint.empty() || output_dir.empty() || tokenizer_path.empty() ||
      prompt_tokens <= 0 || batch_size <= 0 || retained_tokens < 0 ||
      layers <= 0)
    return absl::InvalidArgumentError(
        "checkpoint, output_dir, tokenizer, positive prompt_tokens, "
        "batch_size and layers, and nonnegative retained_tokens required");

  std::cout << "Loading checkpoint and tokenizing the corpus..." << std::endl;
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
  if (sentences.size() < 2)
    return absl::InvalidArgumentError(
        "causal tracing requires at least two corpus sentences");
  const Gpt2Config config{
      .transformer_block_count = layers,
      .model_width = absl::GetFlag(FLAGS_model_width),
      .attention_heads = absl::GetFlag(FLAGS_attention_heads),
      .feed_forward_width = absl::GetFlag(FLAGS_feed_forward_width),
      .vocabulary_size = compact->vocab_size(),
      .pad_vocabulary = false};
  ASSIGN_OR_RETURN(auto model, CreateGpt2(*executor, DataType::BF16, 0, config));
  RETURN_IF_ERROR(
      ReadFromDirectory(*executor, *model, checkpoint.string(), false));

  std::error_code error;
  const bool created = fs::create_directory(output_dir, error);
  if (error)
    return absl::UnknownError(
        absl::StrCat("cannot create output_dir: ", error.message()));
  if (!created)
    return absl::AlreadyExistsError("output_dir must be fresh");
  std::ofstream file(output_dir / "causal_trace.tsv");
  if (!file)
    return absl::UnknownError("cannot create causal trace report");
  util::TeeStream report(std::cout, file);
  report << "# checkpoint=" << checkpoint.string()
         << "\n# corpus=" << absl::GetFlag(FLAGS_corpus)
         << "\n# tokenizer=" << tokenizer_path
         << "\n# sentences=" << sentences.size()
         << "\n# vocabulary_size=" << compact->vocab_size()
         << "\n# eos_token=" << compact->eos_token_id()
         << "\n# batch_size=" << batch_size
         << "\n# prompt_tokens=" << prompt_tokens
         << "\n# retained_tokens=" << retained_tokens << "\n# layers=" << layers
         << "\n# model_width=" << config.model_width
         << "\n# attention_heads=" << config.attention_heads
         << "\n# feed_forward_width=" << config.feed_forward_width
         << "\n# Corruption cycles tokens from the next corpus sentence "
            "through the replaced earlier prefix; positions stay fixed."
         << "\n# Patch only the final query-position row of the post-addition "
            "residual; each site/direction starts from its own baseline."
         << "\n# Rescue: clean row into corrupt pass. Damage: corrupt row "
            "into clean pass."
         << "\n# Independent teacher-forced targets including EOS; these "
            "are not autoregressive completion scores."
         << "\n# Mediation under this corruption does not identify a unique "
            "storage location; a late patch can transfer a computed answer."
         << std::endl;

  std::vector<CausalTraceSite> sites;
  sites.reserve(static_cast<size_t>(layers) * 2);
  for (int block = 0; block < layers; ++block) {
    for (int occurrence = 0; occurrence < 2; ++occurrence) {
      sites.push_back(
          {.layer_name = "ResidualLayer",
           .enclosing_scope = absl::StrCat("transformer_block_", block),
           .occurrence = occurrence});
    }
  }
  report << "# Tracing " << sites.size()
         << " independent sites with shared clean/corrupt baseline passes..."
         << std::endl;
  if (!file)
    return absl::UnknownError("cannot write causal trace report");
  const auto start = std::chrono::steady_clock::now();
  ASSIGN_OR_RETURN(
      auto results,
      TracePrefixMediation(*executor, *model, sentences, compact->vocab_size(),
                           compact->eos_token_id(), prompt_tokens, batch_size,
                           sites, retained_tokens));
  if (results.size() != sites.size())
    return absl::InternalError("trace returned an unexpected result count");
  report << "# trace_seconds="
         << std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                          start)
                .count()
         << std::endl;
  report << "block\tcomponent\tlayer_name\tenclosing_scope\toccurrence"
            "\ttargets\tchanged_prefixes\tclean_correct\tcorrupt_correct"
            "\trescue_correct\tdamage_correct\tbaseline_wrong\trescued"
            "\tnewly_broken_by_rescue\tdamaged\tclean_nonfinite"
            "\tcorrupt_nonfinite\trescue_nonfinite\tdamage_nonfinite"
         << std::endl;
  bool clean_baseline_valid = true;
  for (size_t index = 0; index < results.size(); ++index) {
    const auto& result = results[index];
    const char* component = index % 2 == 0 ? "attention" : "mlp";
    report << index / 2 << '\t' << component << '\t' << result.site.layer_name
           << '\t' << result.site.enclosing_scope << '\t'
           << result.site.occurrence << '\t' << result.targets << '\t'
           << result.changed_prefixes << '\t' << result.clean_correct << '\t'
           << result.corrupt_correct << '\t' << result.rescue_correct << '\t'
           << result.damage_correct << '\t' << result.baseline_wrong << '\t'
           << result.rescued << '\t' << result.newly_broken_by_rescue << '\t'
           << result.damaged << '\t' << result.clean_nonfinite << '\t'
           << result.corrupt_nonfinite << '\t' << result.rescue_nonfinite
           << '\t' << result.damage_nonfinite << std::endl;
    if (!file)
      return absl::UnknownError("cannot write causal trace result");
    clean_baseline_valid =
        clean_baseline_valid && result.targets == results.front().targets &&
        result.clean_correct == result.targets && result.clean_nonfinite == 0;
  }
  if (!clean_baseline_valid)
    return absl::FailedPreconditionError(
        "unaltered checkpoint is not fully memorized under this protocol");
  report << "# Complete" << std::endl;
  if (!file)
    return absl::UnknownError("cannot finish causal trace report");
  return absl::OkStatus();
}

}  // namespace
}  // namespace pluto::llm::one_shot_memorizer

int main(int argc, char** argv) {
  if (absl::ParseCommandLine(argc, argv).size() != 1) {
    std::cerr << "unexpected positional arguments\n";
    return 1;
  }
  const auto status = pluto::llm::one_shot_memorizer::RunCausalTrace();
  if (!status.ok()) {
    std::cerr << status << '\n';
    return 1;
  }
  return 0;
}
