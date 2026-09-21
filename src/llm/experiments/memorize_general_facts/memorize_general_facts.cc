// Controlled width/depth search for exact in-sample next-token memorization.
// Each run uses a fresh initialization; smaller models never inherit a larger
// model's weights. Success is an integer zero-error test, not a loss threshold.
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <utility>

#include "absl/container/flat_hash_set.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "src/cuda/executor.h"
#include "src/cuda/page_locked_host_array.h"
#include "src/dataset/compact_vocabulary.h"
#include "src/dataset/dataset.h"
#include "src/dataset/gpt2_tokenizer.h"
#include "src/dataset/tokenizer.h"
#include "src/llm/adamw_optimizer.h"
#include "src/llm/batch_validation.h"
#include "src/llm/checkpoint.h"
#include "src/llm/experiments/memorize_general_facts/dataset.h"
#include "src/llm/experiments/memorize_general_facts/gradient_clipper.h"
#include "src/llm/experiments/memorize_general_facts/predictions.h"
#include "src/llm/layers/cross_entropy_loss.h"
#include "src/llm/recipes/gpt2.h"
#include "src/util/status_macros.h"
#include "src/util/tee_stream.h"

ABSL_FLAG(std::string, corpus, "testdata/general_facts_dataset.txt",
          "One fact per line");
ABSL_FLAG(std::string, tokenizer, "",
          "Local GPT-2 tokenizer directory (required)");
ABSL_FLAG(std::string, checkpoint_dir, "", "Checkpoint parent (required)");
ABSL_FLAG(std::string, verify_checkpoint, "",
          "Load one checkpoint and independently evaluate it without training");
ABSL_FLAG(std::string, output_dir,
          "src/llm/experiments/memorize_general_facts/runs/baseline",
          "Experiment artifacts");
ABSL_FLAG(int, layers, 8, "Initial transformer depth (nonnegative; default 8)");
ABSL_FLAG(int, model_width, 512, "Residual-stream and embedding width");
ABSL_FLAG(int, attention_heads, 8, "Number of attention heads per block");
ABSL_FLAG(int, feed_forward_width, 2048, "Inner GELU MLP width");
ABSL_FLAG(bool, compact_vocabulary, true,
          "Remap corpus tokens plus EOS to a compact vocabulary; disable for "
          "historical full-vocabulary checkpoints and searches");
ABSL_FLAG(bool, search, false,
          "After success, train successively shallower models from scratch");
ABSL_FLAG(int, batch_size, 16, "Independent padded sentences per batch");
ABSL_FLAG(int, steps, 5000, "Maximum optimizer steps per depth");
ABSL_FLAG(int, eval_every, 128, "Full-corpus exact evaluation interval");
ABSL_FLAG(int, checkpoint_every, 512, "Periodic checkpoint interval");
ABSL_FLAG(int, seed, 1337, "Initialization and shuffle seed");
ABSL_FLAG(double, learning_rate, 6e-4, "Peak AdamW learning rate");
ABSL_FLAG(float, gradient_clip_norm, 0.0f,
          "Maximum global gradient L2 norm; zero disables clipping");
ABSL_FLAG(int, warmup_steps, 100,
          "Linear learning-rate warmup, then cosine decay");
ABSL_FLAG(double, training_seconds, 10800,
          "Wall-clock budget per depth; zero disables");

namespace pluto::llm::memorize_general_facts {
namespace {

using tokenizer::CompactVocabularyTokenizer;

// Record and reconstruct the full shape explicitly. A checkpoint's raw tensor
// files cannot identify its head count: changing the partition into heads does
// not change the Q/K/V matrix shapes, but does change the model's computation.
Gpt2Config ModelConfiguration(int layers, int vocabulary_size) {
  return {.transformer_block_count = layers,
          .model_width = absl::GetFlag(FLAGS_model_width),
          .attention_heads = absl::GetFlag(FLAGS_attention_heads),
          .feed_forward_width = absl::GetFlag(FLAGS_feed_forward_width),
          .vocabulary_size = vocabulary_size,
          .pad_vocabulary = !absl::GetFlag(FLAGS_compact_vocabulary)};
}

// The mapping is part of a compact checkpoint's meaning, not merely a training
// diagnostic. Save it beside every checkpoint so a different same-sized corpus
// cannot silently reinterpret the embedding rows on reload.
absl::Status SaveCheckpoint(cuda::Executor& executor, const Layer& model,
                            const std::filesystem::path& directory,
                            const CompactVocabularyTokenizer* vocabulary) {
  RETURN_IF_ERROR(WriteToDirectory(executor, model, directory));
  if (vocabulary != nullptr)
    RETURN_IF_ERROR(
        vocabulary->SaveToFile(directory / "compact_vocabulary.tsv"));
  return absl::OkStatus();
}

struct Metrics {
  int64_t targets = 0;
  int64_t errors = 0;
  int sentences = 0;
  int exact_sentences = 0;
  double loss_sum = 0;
};

// Reading small diagnostic arrays is intentional. The vocabulary-sized
// logits stay on-device; all transfers use executor-owned pinned memory.
template <class T>
absl::StatusOr<cuda::PageLockedHostArray<T>> Download(cuda::Executor& executor,
                                                      const Buffer& source) {
  ASSIGN_OR_RETURN(auto host, cuda::PageLockedHostArray<T>::Allocate(
                                  executor, source.size_bytes() / sizeof(T)));
  RETURN_IF_ERROR(cuda::CudaStatus(
      cudaMemcpyAsync(host.data(), source.data(), source.size_bytes(),
                      cudaMemcpyDeviceToHost, executor.stream()),
      "download metrics"));
  return host;
}

absl::StatusOr<Metrics> EvaluateExact(
    cuda::Executor& executor, const Layer& model, const Layer& loss,
    PaddedLineDataSetIterator& data, int vocabulary_size,
    const CompactVocabularyTokenizer* vocabulary,
    std::ostream* details = nullptr) {
  RETURN_IF_ERROR(data.Reset());
  Metrics metrics;
  for (size_t batch_index = 0; batch_index < data.batches_per_epoch();
       ++batch_index) {
    ASSIGN_OR_RETURN(auto batch, data.Next());
    RETURN_IF_ERROR(ValidateTrainingBatch(executor, model, loss, batch));
    ASSIGN_OR_RETURN(auto forward, model.fwd(executor, {batch.inputs}));
    ASSIGN_OR_RETURN(auto predictions,
                     PredictMaskedTokens(executor, forward.outputs[0],
                                         batch.targets, vocabulary_size));
    ASSIGN_OR_RETURN(auto loss_forward,
                     loss.fwd(executor, {forward.outputs[0], batch.targets}));
    ASSIGN_OR_RETURN(auto ids, Download<int>(executor, predictions));
    ASSIGN_OR_RETURN(auto targets, Download<int>(executor, batch.targets));
    ASSIGN_OR_RETURN(auto losses,
                     Download<float>(executor, loss_forward.outputs[0]));
    RETURN_IF_ERROR(executor.Synchronize());
    int counted = 0;
    for (int sample = 0; sample < batch.batch_size; ++sample) {
      bool exact = true;
      for (int position = 0; position < batch.sequence_length; ++position) {
        const int row = sample * batch.sequence_length + position;
        if (targets[row] == -1) continue;
        if (ids[row] < 0 || !std::isfinite(losses[row]))
          return absl::DataLossError(
              "nonfinite logits/loss in full-corpus evaluation");
        ++counted;
        ++metrics.targets;
        const bool correct = ids[row] == targets[row];
        metrics.errors += !correct;
        metrics.loss_sum += losses[row];
        exact &= correct;
        if (details != nullptr) {
          // Audit files retain original GPT-2 IDs, allowing the existing
          // independent tokenizer verifier to check text without trusting the
          // compact-ID implementation. Loss/accuracy use compact IDs above.
          int original_target = targets[row];
          int original_prediction = ids[row];
          if (vocabulary != nullptr) {
            ASSIGN_OR_RETURN(original_target,
                             vocabulary->OriginalId(targets[row]));
            ASSIGN_OR_RETURN(original_prediction,
                             vocabulary->OriginalId(ids[row]));
          }
          *details << metrics.sentences + 1 << '\t' << position + 1 << '\t'
                   << original_target << '\t' << original_prediction << '\t'
                   << losses[row] << '\n';
        }
      }
      ++metrics.sentences;
      metrics.exact_sentences += exact;
    }
    if (counted != batch.supervised_row_count)
      return absl::InternalError(
          "dataset supervised count disagrees with target mask");
  }
  if (static_cast<size_t>(metrics.sentences) != data.sample_count() ||
      metrics.targets != data.supervised_row_count())
    return absl::InternalError(
        "evaluation did not visit every corpus target exactly once");
  return metrics;
}

int64_t ParameterCount(Layer& model) {
  absl::flat_hash_set<void*> seen;
  int64_t count = 0;
  for (const auto& weight : model.weights())
    if (seen.insert(weight.data()).second)
      count += weight.size_bytes() / sizeof(float);
  return count;
}

double Elapsed(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
      .count();
}

std::string Timestamp() {
  return absl::FormatTime("%Y-%m-%d %H:%M:%S UTC", absl::Now(),
                          absl::UTCTimeZone());
}

float LearningRate(int step) {
  const double peak = absl::GetFlag(FLAGS_learning_rate);
  const int warmup = absl::GetFlag(FLAGS_warmup_steps);
  if (step <= warmup) return peak * step / warmup;
  const double progress =
      std::clamp(static_cast<double>(step - warmup) /
                     std::max(1, absl::GetFlag(FLAGS_steps) - warmup),
                 0.0, 1.0);
  return peak *
         (0.1 + 0.9 * (1 + std::cos(3.14159265358979323846 * progress)) / 2);
}

// Unlike Train's scalar loss threshold, this experiment stops on exact
// top-1 accuracy over the entire finite corpus. The update itself uses the
// same native forward/loss/backward/AdamW wiring as Train.
absl::StatusOr<bool> TrainDepth(cuda::Executor& executor,
                                const tokenizer::Tokenizer& tokenizer,
                                int eos_token, const TextCorpus& corpus,
                                int layers,
                                const CompactVocabularyTokenizer* vocabulary) {
  const auto output = std::filesystem::path(absl::GetFlag(FLAGS_output_dir)) /
                      absl::StrCat("layers_", layers);
  const auto checkpoints =
      std::filesystem::path(absl::GetFlag(FLAGS_checkpoint_dir)) /
      absl::StrCat("layers_", layers);
  std::error_code error;
  const bool output_exists = std::filesystem::exists(output, error);
  if (error) return absl::InternalError(error.message());
  if (output_exists)
    return absl::AlreadyExistsError(
        "refusing to overwrite an existing experiment run");
  std::filesystem::create_directories(output, error);
  if (error) return absl::InternalError(error.message());
  const bool checkpoints_exist = std::filesystem::exists(checkpoints, error);
  if (error) return absl::InternalError(error.message());
  if (checkpoints_exist)
    return absl::AlreadyExistsError(
        "use a fresh checkpoint directory for each trial");
  // Snapshot the exact text and tokenizer beside the metrics, so paths that
  // later change cannot make the experiment's inputs ambiguous.
  std::ofstream corpus_snapshot(output / "corpus.txt", std::ios::binary);
  corpus_snapshot.write(corpus.text().data(), corpus.text().size());
  corpus_snapshot.close();
  if (!corpus_snapshot)
    return absl::InternalError("cannot preserve the input corpus");
  std::filesystem::copy_file(
      std::filesystem::path(absl::GetFlag(FLAGS_tokenizer)) / "tokenizer.json",
      output / "tokenizer.json", error);
  if (error) return absl::InternalError(error.message());
  if (vocabulary != nullptr)
    RETURN_IF_ERROR(vocabulary->SaveToFile(output / "compact_vocabulary.tsv"));
  std::ofstream log_file(output / "train.log");
  std::ofstream table(output / "metrics.tsv");
  std::ofstream manifest(output / "config.txt");
  if (!log_file || !table || !manifest)
    return absl::InternalError("cannot open experiment artifacts");
  util::TeeStream log(std::cout, log_file);
  table << "step\tseconds\tmean_loss\terrors\ttargets\texact_"
           "sentences\tsentences\n";

  PaddedLineDataSetOptions options{
      .batch_size = absl::GetFlag(FLAGS_batch_size),
      .context_length = kGpt2ContextLength,
      .prompt_tokens = 5,
      .eos_token = eos_token,
      .shuffle = true,
      .seed = static_cast<uint64_t>(absl::GetFlag(FLAGS_seed))};
  ASSIGN_OR_RETURN(auto training,
                   PaddedLineDataSetIterator::Create(executor, corpus.text(),
                                                     tokenizer, options));
  options.shuffle = false;
  ASSIGN_OR_RETURN(auto evaluation,
                   PaddedLineDataSetIterator::Create(executor, corpus.text(),
                                                     tokenizer, options));
  const auto model_config = ModelConfiguration(layers, tokenizer.vocab_size());
  ASSIGN_OR_RETURN(auto model,
                   CreateGpt2(executor, DataType::BF16,
                              absl::GetFlag(FLAGS_seed), model_config));
  ASSIGN_OR_RETURN(auto loss, CrossEntropyLossLayer::Create(
                                  executor, tokenizer.vocab_size(),
                                  DataType::BF16, kGpt2ContextLength));
  const AdamWConfig config{.learning_rate = LearningRate(1),
                           .beta1 = 0.9f,
                           .beta2 = 0.99f,
                           .epsilon = 1e-8f,
                           .weight_decay = 0.0f};
  ASSIGN_OR_RETURN(auto optimizer,
                   AdamWOptimizer::Create(executor, *model, config));
  // Disabled clipping should add neither scratch allocations nor GPU work.
  std::unique_ptr<GradientClipper> clipper;
  const float gradient_clip_norm = absl::GetFlag(FLAGS_gradient_clip_norm);
  if (gradient_clip_norm > 0) {
    ASSIGN_OR_RETURN(
        clipper, GradientClipper::Create(executor, *model, gradient_clip_norm));
  }
  const int64_t parameters = ParameterCount(*model);
  manifest << "corpus=" << absl::GetFlag(FLAGS_corpus)
           << "\ntokenizer=" << absl::GetFlag(FLAGS_tokenizer)
           << "\nlayers=" << layers << "\nwidth=" << model_config.model_width
           << "\nheads=" << model_config.attention_heads << "\nhead_dimension="
           << model_config.model_width / model_config.attention_heads
           << "\nfeed_forward_width=" << model_config.feed_forward_width
           << "\ncontext_length=1024"
           << "\nprompt_tokens=5\nvocabulary=" << tokenizer.vocab_size()
           << "\ncompact_vocabulary=" << absl::GetFlag(FLAGS_compact_vocabulary)
           << "\ncompute=BF16\nmaster_"
              "weights=FP32"
           << "\ngradient_clip_norm=" << gradient_clip_norm
           << "\nparameters=" << parameters
           << "\nseed=" << absl::GetFlag(FLAGS_seed)
           << "\nbatch_size=" << options.batch_size
           << "\nmax_steps=" << absl::GetFlag(FLAGS_steps)
           << "\npeak_learning_rate=" << absl::GetFlag(FLAGS_learning_rate)
           << "\nwarmup_steps=" << absl::GetFlag(FLAGS_warmup_steps)
           << "\neval_every=" << absl::GetFlag(FLAGS_eval_every)
           << "\ncheckpoint_every=" << absl::GetFlag(FLAGS_checkpoint_every)
           << "\nmin_learning_rate_fraction=0.1\nbeta1=0.9\nbeta2=0.99\nweight_"
              "decay=0"
           << "\ntraining_seconds=" << absl::GetFlag(FLAGS_training_seconds)
           << "\nsamples=" << training->sample_count()
           << "\nscored_targets=" << training->supervised_row_count() << '\n';
  manifest.flush();
  log << "layers=" << layers << " width=" << model_config.model_width
      << " heads=" << model_config.attention_heads
      << " vocabulary=" << tokenizer.vocab_size()
      << " feed_forward_width=" << model_config.feed_forward_width
      << " parameters=" << parameters << " samples=" << training->sample_count()
      << " scored_targets=" << training->supervised_row_count() << std::endl;
  RETURN_IF_ERROR(
      SaveCheckpoint(executor, *model, checkpoints / "step_0", vocabulary));
  RETURN_IF_ERROR(executor.Synchronize());
  const auto start = std::chrono::steady_clock::now();
  auto report = [&](int step, const Metrics& metrics) {
    const double seconds = Elapsed(start);
    log << Timestamp() << " step=" << step << " elapsed_seconds=" << seconds
        << " mean_loss=" << metrics.loss_sum / metrics.targets
        << " errors=" << metrics.errors << '/' << metrics.targets
        << " exact_sentences=" << metrics.exact_sentences << '/'
        << metrics.sentences << std::endl;
    table << step << '\t' << seconds << '\t'
          << metrics.loss_sum / metrics.targets << '\t' << metrics.errors
          << '\t' << metrics.targets << '\t' << metrics.exact_sentences << '\t'
          << metrics.sentences << std::endl;
  };
  ASSIGN_OR_RETURN(auto metrics,
                   EvaluateExact(executor, *model, *loss, *evaluation,
                                 tokenizer.vocab_size(), vocabulary));
  report(0, metrics);
  int completed = 0;
  int64_t samples_seen = 0;
  bool reached_time_limit = false;
  for (int step = 1; step <= absl::GetFlag(FLAGS_steps) && metrics.errors != 0;
       ++step) {
    ASSIGN_OR_RETURN(auto batch, training->Next());
    RETURN_IF_ERROR(ValidateTrainingBatch(executor, *model, *loss, batch));
    ASSIGN_OR_RETURN(auto forward, model->fwd(executor, {batch.inputs}));
    ASSIGN_OR_RETURN(auto loss_forward,
                     loss->fwd(executor, {forward.outputs[0], batch.targets}));
    ASSIGN_OR_RETURN(auto gradients,
                     loss->bwd(executor, {}, std::move(loss_forward.state)));
    ASSIGN_OR_RETURN(auto unused,
                     model->bwd(executor, gradients, std::move(forward.state)));
    (void)unused;
    if (clipper != nullptr)
      RETURN_IF_ERROR(clipper->Clip());
    RETURN_IF_ERROR(optimizer->SetLearningRate(LearningRate(step)));
    RETURN_IF_ERROR(optimizer->ApplyStep());
    // Bound completed GPU work, not just host enqueue time. This also keeps
    // stream-ordered allocations from accumulating thousands of queued steps.
    RETURN_IF_ERROR(executor.Synchronize());
    completed = step;
    samples_seen += batch.batch_size;
    if (step % 16 == 0)
      log << Timestamp() << " completed_step=" << step
          << " elapsed_seconds=" << Elapsed(start)
          << " learning_rate=" << LearningRate(step) << std::endl;
    const bool timeout =
        absl::GetFlag(FLAGS_training_seconds) > 0 &&
        Elapsed(start) >= absl::GetFlag(FLAGS_training_seconds);
    if (step % absl::GetFlag(FLAGS_eval_every) == 0 ||
        step == absl::GetFlag(FLAGS_steps) || timeout) {
      ASSIGN_OR_RETURN(metrics,
                       EvaluateExact(executor, *model, *loss, *evaluation,
                                     tokenizer.vocab_size(), vocabulary));
      report(step, metrics);
    }
    if (step % absl::GetFlag(FLAGS_checkpoint_every) == 0)
      RETURN_IF_ERROR(SaveCheckpoint(executor, *model,
                                     checkpoints / absl::StrCat("step_", step),
                                     vocabulary));
    if (timeout) {
      reached_time_limit = true;
      break;
    }
  }
  const auto final_checkpoint = checkpoints / absl::StrCat("step_", completed);
  RETURN_IF_ERROR(
      SaveCheckpoint(executor, *model, final_checkpoint, vocabulary));
  // Reload the on-disk weights and repeat the full audit. This verifies that
  // success belongs to a usable checkpoint, not just an in-memory model.
  RETURN_IF_ERROR(ReadFromDirectory(executor, *model, final_checkpoint));
  std::ofstream details(output / "final_predictions.tsv");
  if (!details)
    return absl::InternalError("cannot write final prediction audit");
  details << "line_1based\ttarget_token_index_0based\ttarget_id\tpredicted_"
             "id\tloss\n";
  ASSIGN_OR_RETURN(metrics,
                   EvaluateExact(executor, *model, *loss, *evaluation,
                                 tokenizer.vocab_size(), vocabulary, &details));
  report(completed, metrics);
  std::ofstream result(output / "result.txt");
  result << "success=" << (metrics.errors == 0) << "\nlayers=" << layers
         << "\nwidth=" << model_config.model_width
         << "\nheads=" << model_config.attention_heads
         << "\nfeed_forward_width=" << model_config.feed_forward_width
         << "\nvocabulary=" << tokenizer.vocab_size()
         << "\nparameters=" << parameters << "\nstep=" << completed
         << "\nerrors=" << metrics.errors << "\ntargets=" << metrics.targets
         << "\nsamples_seen=" << samples_seen << "\nepochs="
         << static_cast<double>(samples_seen) / training->sample_count()
         << "\nreached_time_limit=" << reached_time_limit
         << "\ncheckpoint=" << final_checkpoint.string() << '\n';
  details.close();
  table.close();
  manifest.close();
  result.close();
  if (!details || !table || !manifest || !result)
    return absl::InternalError("writing experiment artifacts failed");
  log << (metrics.errors == 0 ? "MEMORIZED" : "BUDGET_EXHAUSTED")
      << " checkpoint=" << final_checkpoint.string() << std::endl;
  return metrics.errors == 0;
}

// Verification starts with a fresh model in a new process: no in-memory
// weights, optimizer state, or training iterator can explain a successful
// result. Use the training run's snapshots for --corpus and --tokenizer.
absl::StatusOr<bool> VerifyCheckpoint(
    cuda::Executor& executor, const tokenizer::Tokenizer& tokenizer,
    int eos_token, const TextCorpus& corpus,
    const CompactVocabularyTokenizer* vocabulary) {
  if (vocabulary != nullptr)
    RETURN_IF_ERROR(vocabulary->ValidateFile(
        std::filesystem::path(absl::GetFlag(FLAGS_verify_checkpoint)) /
        "compact_vocabulary.tsv"));
  const std::filesystem::path output(absl::GetFlag(FLAGS_output_dir));
  std::error_code error;
  const bool exists = std::filesystem::exists(output, error);
  if (error) return absl::InternalError(error.message());
  if (exists)
    return absl::AlreadyExistsError(
        "verification output directory must be fresh");
  std::filesystem::create_directories(output, error);
  if (error) return absl::InternalError(error.message());
  const PaddedLineDataSetOptions options{
      .batch_size = absl::GetFlag(FLAGS_batch_size),
      .context_length = kGpt2ContextLength,
      .prompt_tokens = 5,
      .eos_token = eos_token,
      .shuffle = false};
  ASSIGN_OR_RETURN(auto data, PaddedLineDataSetIterator::Create(
                                  executor, corpus.text(), tokenizer, options));
  const auto model_config =
      ModelConfiguration(absl::GetFlag(FLAGS_layers), tokenizer.vocab_size());
  ASSIGN_OR_RETURN(auto model,
                   CreateGpt2(executor, DataType::BF16,
                              absl::GetFlag(FLAGS_seed), model_config));
  ASSIGN_OR_RETURN(auto loss, CrossEntropyLossLayer::Create(
                                  executor, tokenizer.vocab_size(),
                                  DataType::BF16, kGpt2ContextLength));
  // Full-model verification must not use the generic reader's prefix-loading
  // allowance: a smaller depth can otherwise mistake the next block's input
  // norm for its final norm. The reader counts unique allocations itself,
  // including the tied embedding/head only once.
  RETURN_IF_ERROR(ReadFromDirectory(executor, *model,
                                    absl::GetFlag(FLAGS_verify_checkpoint),
                                    /*allow_prefix=*/false));
  std::ofstream details(output / "final_predictions.tsv");
  if (!details)
    return absl::InternalError("cannot write independent predictions");
  details << "line_1based\ttarget_token_index_0based\ttarget_id\tpredicted_"
             "id\tloss\n";
  ASSIGN_OR_RETURN(auto metrics,
                   EvaluateExact(executor, *model, *loss, *data,
                                 tokenizer.vocab_size(), vocabulary, &details));
  details.close();
  if (!details)
    return absl::InternalError("writing independent predictions failed");
  std::ofstream result(output / "result.txt");
  result << "checkpoint=" << absl::GetFlag(FLAGS_verify_checkpoint)
         << "\ncorpus=" << absl::GetFlag(FLAGS_corpus)
         << "\ntokenizer=" << absl::GetFlag(FLAGS_tokenizer)
         << "\nlayers=" << absl::GetFlag(FLAGS_layers)
         << "\nwidth=" << model_config.model_width
         << "\nheads=" << model_config.attention_heads
         << "\nfeed_forward_width=" << model_config.feed_forward_width
         << "\nvocabulary=" << tokenizer.vocab_size()
         << "\nparameters=" << ParameterCount(*model)
         << "\nerrors=" << metrics.errors << "\ntargets=" << metrics.targets
         << "\nmean_loss=" << metrics.loss_sum / metrics.targets
         << "\nexact_sentences=" << metrics.exact_sentences
         << "\nsentences=" << metrics.sentences << '\n';
  result.close();
  if (!result) return absl::InternalError("writing verification result failed");
  std::cout << "checkpoint_errors=" << metrics.errors << '/' << metrics.targets
            << " exact_sentences=" << metrics.exact_sentences << '/'
            << metrics.sentences
            << " mean_loss=" << metrics.loss_sum / metrics.targets << std::endl;
  return metrics.errors == 0;
}

absl::StatusOr<bool> Run() {
  if (absl::GetFlag(FLAGS_tokenizer).empty() ||
      (absl::GetFlag(FLAGS_checkpoint_dir).empty() &&
       absl::GetFlag(FLAGS_verify_checkpoint).empty()) ||
      (!absl::GetFlag(FLAGS_verify_checkpoint).empty() &&
       absl::GetFlag(FLAGS_search)) ||
      absl::GetFlag(FLAGS_batch_size) <= 0 || absl::GetFlag(FLAGS_steps) < 0 ||
      absl::GetFlag(FLAGS_eval_every) <= 0 ||
      absl::GetFlag(FLAGS_checkpoint_every) <= 0 ||
      absl::GetFlag(FLAGS_warmup_steps) < 0 ||
      !std::isfinite(absl::GetFlag(FLAGS_learning_rate)) ||
      absl::GetFlag(FLAGS_learning_rate) <= 0 ||
      !std::isfinite(absl::GetFlag(FLAGS_gradient_clip_norm)) ||
      absl::GetFlag(FLAGS_gradient_clip_norm) < 0 ||
      !std::isfinite(absl::GetFlag(FLAGS_training_seconds)) ||
      absl::GetFlag(FLAGS_training_seconds) < 0)
    return absl::InvalidArgumentError(
        "invalid experiment flags; tokenizer and checkpoint_dir are required");
  ASSIGN_OR_RETURN(auto executor, cuda::Executor::Create());
  ASSIGN_OR_RETURN(auto tokenizer, tokenizer::Gpt2Tokenizer::Load(
                                       absl::GetFlag(FLAGS_tokenizer)));
  if (tokenizer->vocab_size() != kGpt2VocabularySize)
    return absl::InvalidArgumentError(
        "the experiment requires the full GPT-2 vocabulary");
  ASSIGN_OR_RETURN(auto corpus, LoadTextCorpus(absl::GetFlag(FLAGS_corpus)));
  std::unique_ptr<CompactVocabularyTokenizer> vocabulary;
  const tokenizer::Tokenizer* model_tokenizer = tokenizer.get();
  int eos_token = tokenizer->eos_token_id();
  if (absl::GetFlag(FLAGS_compact_vocabulary)) {
    ASSIGN_OR_RETURN(auto mapping, tokenizer::BuildCompactVocabularyMapping(
                                       *executor, *tokenizer, corpus.text(),
                                       eos_token));
    ASSIGN_OR_RETURN(vocabulary,
                     CompactVocabularyTokenizer::Create(*tokenizer,
                                                        std::move(mapping)));
    model_tokenizer = vocabulary.get();
    eos_token = vocabulary->eos_token_id();
  }
  RETURN_IF_ERROR(ModelConfiguration(absl::GetFlag(FLAGS_layers),
                                     model_tokenizer->vocab_size())
                      .Validate());
  if (!absl::GetFlag(FLAGS_verify_checkpoint).empty())
    return VerifyCheckpoint(*executor, *model_tokenizer, eos_token, corpus,
                            vocabulary.get());
  for (int layers = absl::GetFlag(FLAGS_layers); layers >= 0; --layers) {
    ASSIGN_OR_RETURN(bool success,
                     TrainDepth(*executor, *model_tokenizer, eos_token, corpus,
                                layers, vocabulary.get()));
    if (!success || !absl::GetFlag(FLAGS_search)) return success;
  }
  return true;
}

}  // namespace
}  // namespace pluto::llm::memorize_general_facts

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  auto result = pluto::llm::memorize_general_facts::Run();
  if (!result.ok()) {
    std::cerr << result.status() << std::endl;
    return 1;
  }
  return *result ? 0 : 2;
}
