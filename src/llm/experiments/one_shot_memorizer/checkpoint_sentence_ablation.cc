// Controlled, short training counterfactuals from a shared checkpoint. Removing
// a sentence means erasing its loss gradient, NOT reshuffling the other data.
#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "src/cuda/page_locked_host_array.h"
#include "src/dataset/compact_vocabulary.h"
#include "src/dataset/gpt2_tokenizer.h"
#include "src/dataset/padded_line_dataset.h"
#include "src/llm/adamw_optimizer.h"
#include "src/llm/batch_validation.h"
#include "src/llm/checkpoint.h"
#include "src/llm/experiments/one_shot_memorizer/mlp_probe.h"
#include "src/llm/experiments/one_shot_memorizer/sentence_ablation.h"
#include "src/llm/experiments/one_shot_memorizer/sentence_ablation_report.h"
#include "src/llm/experiments/one_shot_memorizer/sentence_ablation_training.h"
#include "src/llm/gpt2.h"
#include "src/llm/layers/cross_entropy_loss.h"
#include "src/util/status_macros.h"

ABSL_FLAG(std::string, initial_checkpoint, "",
          "Shared step_0 weights and vocabulary mapping");
ABSL_FLAG(std::string, tokenizer, "", "Full GPT-2 tokenizer directory");
ABSL_FLAG(std::string, corpus, "testdata/general_facts_dataset.txt",
          "Full corpus, never rebuilt after omission");
ABSL_FLAG(std::string, output_dir, "",
          "Fresh directory for local experiment artifacts");
ABSL_FLAG(
    int, steps, 512,
    "Fixed update budget, independent of learning-rate horizon (max 4096)");
ABSL_FLAG(int, schedule_horizon, 40000, "Original cosine schedule horizon");
ABSL_FLAG(int, warmup_steps, 100, "Linear warmup updates");
ABSL_FLAG(double, learning_rate, 0.0006, "Peak AdamW learning rate");
ABSL_FLAG(int, batch_size, 32, "Samples per matched-schedule batch");
ABSL_FLAG(int, seed, 1337,
          "Fixed shuffle seed; checkpoint fixes all initial weights");
ABSL_FLAG(std::vector<std::string>, omitted_lines,
          (std::vector<std::string>{"1", "258", "631"}),
          "One-based corpus lines to omit, one separate run each");
ABSL_FLAG(int, single_fact_line, 631,
          "One-based line for batch-one single-fact training; zero disables");
ABSL_FLAG(int, layers, 8, "Transformer blocks");
ABSL_FLAG(int, model_width, 16, "Residual width");
ABSL_FLAG(int, attention_heads, 1, "Attention heads");
ABSL_FLAG(int, feed_forward_width, 64, "MLP width");

namespace pluto::llm::one_shot_memorizer {
namespace {
namespace fs = std::filesystem;

using Snapshot = std::vector<float>;

std::vector<absl::Span<const float>> TensorViews(
    absl::Span<const TensorSpec> layout, const Snapshot& values) {
  std::vector<absl::Span<const float>> views;
  for (const auto& tensor : layout)
    views.push_back(absl::MakeConstSpan(values).subspan(tensor.flat_offset,
                                                        tensor.element_count));
  return views;
}

// Reuse a single pinned staging allocation. Synchronizing once after all D2H
// copies makes the returned ordinary CPU vectors independent of GPU lifetimes.
absl::StatusOr<Snapshot> CopyWeights(
    cuda::Executor& executor, absl::Span<const Buffer> weights,
    absl::Span<const TensorSpec> layout,
    cuda::PageLockedHostArray<float>& staging) {
  for (size_t i = 0; i < weights.size(); ++i)
    RETURN_IF_ERROR(cuda::CudaStatus(
        cudaMemcpyAsync(staging.data() + layout[i].flat_offset,
                        weights[i].data(), weights[i].size_bytes(),
                        cudaMemcpyDeviceToHost, executor.stream()),
        "snapshot ablation parameters"));
  RETURN_IF_ERROR(executor.Synchronize());
  return Snapshot(staging.begin(), staging.end());
}

bool Identical(const Snapshot& a, const Snapshot& b) {
  return a.size() == b.size() &&
         std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

absl::Status MakeDirectory(const fs::path& path) {
  std::error_code error;
  if (!fs::create_directory(path, error))
    return absl::FailedPreconditionError(absl::StrCat(
        "expected fresh directory ", path.string(), ": ", error.message()));
  return absl::OkStatus();
}

absl::Status Save(cuda::Executor& executor, const Layer& model,
                  const tokenizer::CompactVocabularyTokenizer& vocabulary,
                  const fs::path& directory) {
  RETURN_IF_ERROR(WriteToDirectory(executor, model, directory));
  return vocabulary.SaveToFile(directory / "compact_vocabulary.tsv");
}

// The exactness audit includes every original sentence, even the omitted one.
// Teacher forcing identifies all affected target positions; a separate greedy
// audit accepts a sentence only when its complete suffix and EOS match.
absl::Status EvaluateRun(cuda::Executor& executor, const Layer& model,
                         PaddedLineDataSetIterator& evaluation, int vocabulary,
                         const fs::path& directory, std::ostream& summary,
                         const std::string& name, int step, int exposures,
                         double seconds) {
  ASSIGN_OR_RETURN(
      auto scored,
      EvaluateMlpReplacements(executor, model, evaluation, {}, vocabulary));
  ASSIGN_OR_RETURN(
      auto greedy,
      VerifyMlpGreedyCompletions(executor, model, evaluation, {}, vocabulary));
  if (scored.targets_per_sentence.size() != greedy.exact_per_sentence.size() ||
      scored.correct_per_sentence.size() != greedy.exact_per_sentence.size())
    return absl::InternalError("evaluation sample counts disagree");
  for (size_t i = 0; i < greedy.exact_per_sentence.size(); ++i)
    if ((scored.correct_per_sentence[i] == scored.targets_per_sentence[i]) !=
        greedy.exact_per_sentence[i])
      return absl::InternalError(
          "teacher-forced and greedy exactness disagree");
  std::ofstream details(directory / absl::StrCat("predictions_", step, ".tsv"));
  details << "line_1based\tcorrect_targets\ttargets\tgreedy_exact\n";
  for (size_t i = 0; i < scored.targets_per_sentence.size(); ++i)
    details << i + 1 << '\t' << scored.correct_per_sentence[i] << '\t'
            << scored.targets_per_sentence[i] << '\t'
            << greedy.exact_per_sentence[i] << '\n';
  details.close();
  if (!details)
    return absl::InternalError("writing prediction audit failed");
  summary << name << '\t' << step << '\t' << exposures << '\t' << seconds
          << '\t' << scored.correct_targets << '\t' << scored.targets << '\t'
          << greedy.exact_sentences << '\t' << greedy.sentences << std::endl;
  std::cout << name << " step=" << step
            << " correct_targets=" << scored.correct_targets << '/'
            << scored.targets << " greedy_exact=" << greedy.exact_sentences
            << '/' << greedy.sentences
            << " selected_fact_scheduled_occurrences=" << exposures
            << std::endl;
  return absl::OkStatus();
}

struct Condition {
  std::string name;
  std::optional<size_t> omitted;
  std::optional<size_t> single;
  bool repeat_baseline = false;
};

absl::Status RunCondition(
    cuda::Executor& executor, const Condition& condition,
    const tokenizer::CompactVocabularyTokenizer& vocabulary,
    const Gpt2Config& config, absl::string_view corpus,
    absl::Span<const std::string> lines, absl::Span<const TensorSpec> layout,
    PaddedLineDataSetIterator& evaluation, std::vector<Snapshot>& baseline,
    std::ostream& summary, size_t selected_fact) {
  const fs::path directory =
      fs::path(absl::GetFlag(FLAGS_output_dir)) / condition.name;
  RETURN_IF_ERROR(MakeDirectory(directory));
  PaddedLineDataSetOptions data_options{
      .batch_size = condition.single ? 1 : absl::GetFlag(FLAGS_batch_size),
      .context_length = kGpt2ContextLength,
      .prompt_tokens = 5,
      .eos_token = vocabulary.eos_token_id(),
      .shuffle = true,
      .seed = static_cast<uint64_t>(absl::GetFlag(FLAGS_seed))};
  const absl::string_view training_text =
      condition.single ? absl::string_view(lines[*condition.single]) : corpus;
  ASSIGN_OR_RETURN(auto training,
                   PaddedLineDataSetIterator::Create(executor, training_text,
                                                     vocabulary, data_options));
  ASSIGN_OR_RETURN(auto model, CreateGpt2(executor, DataType::BF16,
                                          absl::GetFlag(FLAGS_seed), config));
  RETURN_IF_ERROR(ReadFromDirectory(
      executor, *model, absl::GetFlag(FLAGS_initial_checkpoint), false));
  std::vector<Buffer> weights;
  absl::flat_hash_set<const void*> seen;
  for (const Buffer& weight : model->weights())
    if (seen.insert(weight.data()).second)
      weights.push_back(weight);
  if (weights.size() != layout.size())
    return absl::FailedPreconditionError(
        "model unique tensor count differs from declared layout");
  for (size_t i = 0; i < weights.size(); ++i)
    if (weights[i].size_bytes() != layout[i].element_count * sizeof(float))
      return absl::FailedPreconditionError(
          absl::StrCat("model tensor shape mismatch: ", layout[i].name));
  const size_t count = layout.back().flat_offset + layout.back().element_count;
  ASSIGN_OR_RETURN(auto staging,
                   cuda::PageLockedHostArray<float>::Allocate(executor, count));
  ASSIGN_OR_RETURN(auto initial,
                   CopyWeights(executor, weights, layout, staging));
  RETURN_IF_ERROR(
      ValidateParameterValues(layout, TensorViews(layout, initial)));
  const bool record = baseline.empty();
  if (record)
    baseline.push_back(initial);
  else if (!Identical(initial, baseline[0]))
    return absl::FailedPreconditionError("initial checkpoints differ bytewise");
  ASSIGN_OR_RETURN(auto loss, CrossEntropyLossLayer::Create(
                                  executor, vocabulary.vocab_size(),
                                  DataType::BF16, kGpt2ContextLength));
  ASSIGN_OR_RETURN(const float first_rate,
                   AblationLearningRate(1, absl::GetFlag(FLAGS_learning_rate),
                                        absl::GetFlag(FLAGS_warmup_steps),
                                        absl::GetFlag(FLAGS_schedule_horizon)));
  ASSIGN_OR_RETURN(auto optimizer,
                   AdamWOptimizer::Create(executor, *model,
                                          {.learning_rate = first_rate,
                                           .beta1 = 0.9f,
                                           .beta2 = 0.99f,
                                           .epsilon = 1e-8f,
                                           .weight_decay = 0}));
  RETURN_IF_ERROR(optimizer->ZeroGrad());
  std::ofstream trace(directory / "trajectory.tsv");
  trace << "step\tselected_fact_scheduled_occurrences\tomitted_"
           "exposures\tfirst_omission_"
           "step\tbitwise_changed_vs_baseline\tdelta_l2\telapsed_seconds\n";
  const auto start = std::chrono::steady_clock::now();
  const auto elapsed = [&] {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                         start)
        .count();
  };
  int fact_exposures = 0, omitted_exposures = 0, first_omission_step = 0;
  Snapshot final;
  // Always do the requested number of updates; memorization is an observation,
  // not an early-stop condition that could confound matched-step comparisons.
  for (int step = 1; step <= absl::GetFlag(FLAGS_steps); ++step) {
    ASSIGN_OR_RETURN(auto batch, training->Next());
    const auto sample_indices = training->last_batch_sample_indices();
    for (size_t id : sample_indices)
      fact_exposures += condition.single ? *condition.single == selected_fact
                                         : id == selected_fact;
    RETURN_IF_ERROR(ValidateTrainingBatch(executor, *model, *loss, batch));
    ASSIGN_OR_RETURN(auto forward, model->fwd(executor, {batch.inputs}));
    ASSIGN_OR_RETURN(auto loss_forward,
                     loss->fwd(executor, {forward.outputs[0], batch.targets}));
    ASSIGN_OR_RETURN(auto gradients,
                     loss->bwd(executor, {}, std::move(loss_forward.state)));
    if (condition.omitted) {
      if (gradients.size() != 1)
        return absl::InternalError(
            "cross entropy must return one logits gradient");
      ASSIGN_OR_RETURN(
          int masked,
          ZeroSentenceContribution(executor, gradients[0], sample_indices,
                                   *condition.omitted, kGpt2ContextLength,
                                   (vocabulary.vocab_size() + 15) / 16 * 16));
      omitted_exposures += masked;
      if (masked != 0 && first_omission_step == 0)
        first_omission_step = step;
    }
    ASSIGN_OR_RETURN(auto unused,
                     model->bwd(executor, gradients, std::move(forward.state)));
    (void)unused;
    ASSIGN_OR_RETURN(float rate, AblationLearningRate(
                                     step, absl::GetFlag(FLAGS_learning_rate),
                                     absl::GetFlag(FLAGS_warmup_steps),
                                     absl::GetFlag(FLAGS_schedule_horizon)));
    RETURN_IF_ERROR(optimizer->SetLearningRate(rate));
    RETURN_IF_ERROR(optimizer->ApplyStep());
    ASSIGN_OR_RETURN(auto current,
                     CopyWeights(executor, weights, layout, staging));
    if (record)
      baseline.push_back(current);
    const bool same = Identical(current, baseline[step]);
    if ((condition.repeat_baseline ||
         (condition.omitted && first_omission_step == 0)) &&
        !same)
      return absl::FailedPreconditionError(
          absl::StrCat("reproducibility control failed at step ", step, " in ",
                       condition.name));
    ASSIGN_OR_RETURN(
        auto delta,
        CompareParameterValues(layout, TensorViews(layout, baseline[step]),
                               TensorViews(layout, current), 0));
    trace << step << '\t' << fact_exposures << '\t' << omitted_exposures << '\t'
          << first_omission_step << '\t' << delta.total.bitwise_changed_count
          << '\t' << std::setprecision(17) << delta.total.delta_l2 << '\t'
          << elapsed() << std::endl;
    if (step % 128 == 0 || step == 1 || step == first_omission_step ||
        step == absl::GetFlag(FLAGS_steps)) {
      RETURN_IF_ERROR(Save(executor, *model, vocabulary,
                           directory / absl::StrCat("step_", step)));
      std::cout << condition.name << " step=" << step
                << " elapsed_seconds=" << elapsed()
                << " changed_vs_baseline=" << delta.total.bitwise_changed_count
                << '/' << count << " delta_l2=" << delta.total.delta_l2
                << std::endl;
    }
    final = std::move(current);
  }
  RETURN_IF_ERROR(EvaluateRun(
      executor, *model, evaluation, vocabulary.vocab_size(), directory, summary,
      condition.name, absl::GetFlag(FLAGS_steps), fact_exposures, elapsed()));
  ASSIGN_OR_RETURN(auto delta, CompareParameterValues(
                                   layout, TensorViews(layout, baseline.back()),
                                   TensorViews(layout, final)));
  RETURN_IF_ERROR(WriteSentenceAblationReport(delta, directory));
  std::ofstream comparison(directory / "comparison.txt");
  comparison << "delta = full-corpus baseline at matched step MINUS "
             << condition.name << " at matched step\n"
             << "selected_fact_scheduled_occurrences includes omitted slots; "
                "those slots contribute ZERO gradient in the omission run.\n";
  comparison.close();
  if (!comparison)
    return absl::InternalError("writing comparison metadata failed");
  // Every condition also gets an initialization-relative map, crucial for the
  // single-fact question: what did training on ONLY this sentence change?
  const fs::path from_initial = directory / "from_initial";
  RETURN_IF_ERROR(MakeDirectory(from_initial));
  ASSIGN_OR_RETURN(auto learned,
                   CompareParameterValues(layout, TensorViews(layout, initial),
                                          TensorViews(layout, final)));
  RETURN_IF_ERROR(WriteSentenceAblationReport(learned, from_initial));
  std::ofstream initial_comparison(from_initial / "comparison.txt");
  initial_comparison << "delta = INITIAL weights MINUS TRAINED weights\n"
                        "This is the negative of the usual learned update.\n";
  initial_comparison.close();
  if (!initial_comparison)
    return absl::InternalError("writing initialization comparison failed");
  trace.close();
  if (!trace || !summary)
    return absl::InternalError("writing ablation metrics failed");
  return absl::OkStatus();
}

absl::Status Run() {
  const int steps = absl::GetFlag(FLAGS_steps);
  if (steps <= 0 || steps > 4096 || absl::GetFlag(FLAGS_batch_size) <= 0 ||
      absl::GetFlag(FLAGS_initial_checkpoint).empty() ||
      absl::GetFlag(FLAGS_tokenizer).empty() ||
      absl::GetFlag(FLAGS_output_dir).empty() ||
      absl::GetFlag(FLAGS_single_fact_line) < 0)
    return absl::InvalidArgumentError("invalid ablation pilot arguments");
  ASSIGN_OR_RETURN(auto ignored_rate,
                   AblationLearningRate(1, absl::GetFlag(FLAGS_learning_rate),
                                        absl::GetFlag(FLAGS_warmup_steps),
                                        absl::GetFlag(FLAGS_schedule_horizon)));
  (void)ignored_rate;
  ASSIGN_OR_RETURN(
      auto checkpoint,
      InspectCheckpointDirectory(absl::GetFlag(FLAGS_initial_checkpoint)));
  if (checkpoint.step != 0)
    return absl::InvalidArgumentError(
        "initial_checkpoint must be step_0: optimizer state is initialized "
        "fresh");
  ASSIGN_OR_RETURN(auto executor, cuda::Executor::Create());
  ASSIGN_OR_RETURN(
      auto base, tokenizer::Gpt2Tokenizer::Load(absl::GetFlag(FLAGS_tokenizer)));
  ASSIGN_OR_RETURN(auto vocabulary,
                   tokenizer::CompactVocabularyTokenizer::LoadFromFile(
                       *base, checkpoint.directory / "compact_vocabulary.tsv"));
  if (vocabulary->original_eos_token_id() != base->eos_token_id())
    return absl::InvalidArgumentError(
        "tokenizer EOS differs from frozen mapping");
  ASSIGN_OR_RETURN(auto corpus, LoadTextCorpus(absl::GetFlag(FLAGS_corpus)));
  // Remove one optional final newline only. The dataset factory validates all
  // other empty lines; silently filtering them would change sentence indices.
  absl::string_view text = corpus.text();
  if (!text.empty() && text.back() == '\n')
    text.remove_suffix(1);
  std::vector<std::string> lines = absl::StrSplit(text, '\n');
  std::vector<Condition> conditions{
      {.name = "baseline"},
      {.name = "baseline_repeat", .repeat_baseline = true}};
  absl::flat_hash_set<int> seen;
  for (const auto& value : absl::GetFlag(FLAGS_omitted_lines)) {
    int line;
    if (!absl::SimpleAtoi(value, &line) || line <= 0 ||
        static_cast<size_t>(line) > lines.size() || !seen.insert(line).second)
      return absl::InvalidArgumentError(
          "omitted_lines must be distinct in-range one-based line numbers");
    conditions.push_back({.name = absl::StrCat("omit_line_", line),
                          .omitted = static_cast<size_t>(line - 1)});
  }
  const int single_line = absl::GetFlag(FLAGS_single_fact_line);
  if (static_cast<size_t>(single_line) > lines.size())
    return absl::InvalidArgumentError("single_fact_line is outside corpus");
  if (single_line > 0)
    conditions.push_back({.name = absl::StrCat("only_line_", single_line),
                          .single = static_cast<size_t>(single_line - 1)});
  Gpt2Config config{
      .transformer_block_count = absl::GetFlag(FLAGS_layers),
      .model_width = absl::GetFlag(FLAGS_model_width),
      .attention_heads = absl::GetFlag(FLAGS_attention_heads),
      .feed_forward_width = absl::GetFlag(FLAGS_feed_forward_width),
      .vocabulary_size = vocabulary->vocab_size(),
      .pad_vocabulary = false};
  RETURN_IF_ERROR(config.Validate());
  ASSIGN_OR_RETURN(
      auto layout,
      BuildGpt2ParameterLayout({vocabulary->vocab_size(), config.model_width,
                                config.feed_forward_width, kGpt2ContextLength,
                                config.transformer_block_count}));
  ASSIGN_OR_RETURN(auto evaluation,
                   PaddedLineDataSetIterator::Create(
                       *executor, text, *vocabulary,
                       {.batch_size = absl::GetFlag(FLAGS_batch_size),
                        .context_length = kGpt2ContextLength,
                        .prompt_tokens = 5,
                        .eos_token = vocabulary->eos_token_id()}));
  const fs::path output = absl::GetFlag(FLAGS_output_dir);
  RETURN_IF_ERROR(MakeDirectory(output));
  std::ofstream manifest(output / "manifest.txt");
  manifest
      << "initial_checkpoint=" << checkpoint.directory.string()
      << "\ncorpus=" << absl::GetFlag(FLAGS_corpus)
      << "\ntokenizer=" << absl::GetFlag(FLAGS_tokenizer) << "\nsteps=" << steps
      << "\nschedule_horizon=" << absl::GetFlag(FLAGS_schedule_horizon)
      << "\nwarmup_steps=" << absl::GetFlag(FLAGS_warmup_steps)
      << "\npeak_learning_rate=" << absl::GetFlag(FLAGS_learning_rate)
      << "\nseed=" << absl::GetFlag(FLAGS_seed)
      << "\nbatch_size=" << absl::GetFlag(FLAGS_batch_size)
      << "\nlayers=" << config.transformer_block_count
      << "\nmodel_width=" << config.model_width
      << "\nfeed_forward_width=" << config.feed_forward_width
      << "\nattention_heads=" << config.attention_heads
      << "\nvocabulary=" << vocabulary->vocab_size()
      << "\ncontext_length=1024\nprompt_tokens=5\nbeta1=0.9\nbeta2=0."
         "99\nepsilon=1e-8\nweight_decay=0\ngradient_clipping=none\n"
      << "intervention=fixed-schedule loss-gradient deletion with original "
         "normalization\n"
      << "single_fact=batch 1 repeated; not exposure-matched to leave-one-out\n"
      << "determinism=every baseline-repeat update byte-compared; omissions "
         "byte-compared before first exposure\n";
  for (const auto& condition : conditions)
    if (condition.omitted || condition.single) {
      const size_t index =
          condition.omitted ? *condition.omitted : *condition.single;
      manifest << condition.name << "=" << lines[index] << '\n';
    }
  manifest.close();
  if (!manifest)
    return absl::InternalError("writing manifest failed");
  std::ofstream summary(output / "summary.tsv");
  summary << "condition\tstep\tselected_fact_scheduled_"
             "occurrences\tseconds\tcorrect_"
             "targets\ttargets\tgreedy_exact\tsentences\n";
  std::vector<Snapshot> baseline;
  baseline.reserve(steps + 1);
  for (const auto& condition : conditions)
    RETURN_IF_ERROR(RunCondition(
        *executor, condition, *vocabulary, config, text, lines, layout,
        *evaluation, baseline, summary, single_line > 0 ? single_line - 1 : 0));
  summary.close();
  if (!summary)
    return absl::InternalError("writing summary failed");
  std::cout << "All conditions complete. Reports: " << output.string()
            << std::endl;
  return absl::OkStatus();
}
}  // namespace
}  // namespace pluto::llm::one_shot_memorizer

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  const auto status = pluto::llm::one_shot_memorizer::Run();
  if (!status.ok()) {
    std::cerr << status << std::endl;
    return 1;
  }
  return 0;
}
