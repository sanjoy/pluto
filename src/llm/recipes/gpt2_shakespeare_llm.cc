#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "src/cuda/buffer.h"
#include "src/cuda/executor.h"
#include "src/cuda/page_locked_host_array.h"
#include "src/dataset/dataset.h"
#include "src/dataset/detokenizer.h"
#include "src/dataset/gpt2_detokenizer.h"
#include "src/dataset/gpt2_tokenizer.h"
#include "src/dataset/tokenizer.h"
#include "src/llm/adamw_optimizer.h"
#include "src/llm/checkpoint.h"
#include "src/llm/layer.h"
#include "src/llm/layers/cross_entropy_loss.h"
#include "src/llm/layers/sparse_autoencoder.h"
#include "src/llm/recipes/gpt2.h"
#include "src/llm/recipes/gpt2_shakespeare_cli.h"
#include "src/llm/recipes/sparse_autoencoder_dataset.h"
#include "src/llm/sampling.h"
#include "src/llm/trainer.h"
#include "src/util/status_macros.h"
#include "src/util/tee_stream.h"

ABSL_FLAG(
    std::string, mode, "",
    "Required run mode: train_model, infer_model, train_sae, or infer_SAE "
    "(infer_sae is also accepted)");
ABSL_FLAG(std::string, corpus, "",
          "Shakespeare corpus path; defaults to the Bazel testdata runfile");
ABSL_FLAG(std::string, tokenizer_dir, "",
          "GPT-2 tokenizer directory; defaults to "
          "PLUTO_GPT2_TOKENIZER_DIR");
ABSL_FLAG(std::string, resume_from, "",
          "Parent directory whose latest valid step_N resumes model or SAE "
          "training");
ABSL_FLAG(std::string, inference_from, "",
          "Exact step_N checkpoint: GPT-2 for infer_model, SAE for infer_SAE");
ABSL_FLAG(
    std::string, sparse_autoencoder_from, "",
    "Exact GPT-2 step_N checkpoint required by train_sae/infer_SAE to generate "
    "fourth-block activations");
ABSL_FLAG(
    std::string, checkpoint_dir, "",
    "Root directory for periodic and final step_N checkpoint directories");
ABSL_FLAG(int, checkpoint_every, 0,
          "Write a checkpoint every N optimizer steps; zero disables periodic "
          "writes (checkpoint_dir still enables a final checkpoint)");
ABSL_FLAG(bool, checkpoint_initial, false,
          "Also save the initial weights at step_N before training; requires "
          "checkpoint_dir or resume_from");
ABSL_FLAG(int, steps, -1,
          "Maximum AdamW updates in this invocation; omitted has no step cap");
ABSL_FLAG(double, training_seconds, 0.0,
          "Optional positive finite training wall-clock budget in seconds; "
          "omitted disables the time limit. Includes periodic evaluation and "
          "checkpointing; excludes setup and final evaluation/checkpointing");
ABSL_FLAG(double, learning_rate, 3e-4, "AdamW learning rate");
ABSL_FLAG(double, adam_beta1, 0.9, "AdamW first-moment decay");
ABSL_FLAG(double, adam_beta2, 0.95, "AdamW second-moment decay");
ABSL_FLAG(double, adam_epsilon, 1e-8, "AdamW numerical-stability epsilon");
ABSL_FLAG(double, weight_decay, 0.1, "Decoupled AdamW weight decay");
ABSL_FLAG(int, eval_batches, 4,
          "Number of fixed batches used for each train/test loss evaluation");
ABSL_FLAG(double, test_fraction, 0.1,
          "Fraction of the corpus reserved as contiguous held-out test data");
ABSL_FLAG(double, train_until_loss, -1.0,
          "When nonnegative, stop once training loss reaches this value; "
          "--steps, when supplied, remains the hard iteration cap");
ABSL_FLAG(int, training_eval_interval, 100,
          "Steps between training-loss checks and progress reports");
ABSL_FLAG(int, seed, 17, "Deterministic initialization and sampling seed");
ABSL_FLAG(std::string, prompt, "",
          "One inference prompt; empty starts the inference prompt loop");
ABSL_FLAG(int, generation_tokens, 300, "Tokens generated after each prompt");
ABSL_FLAG(double, temperature, 0.8,
          "Sampling temperature; zero uses deterministic greedy decoding");
ABSL_FLAG(int, batch_size, 1,
          "Number of context-length sequences per training/evaluation batch");
ABSL_FLAG(std::string, log_file, "/tmp/train.log",
          "File that receives a copy of stdout; truncated at startup");

namespace pluto::llm {
namespace {

using tokenizer::Gpt2Detokenizer;
using tokenizer::Gpt2Tokenizer;

constexpr int kSparseAutoEncoderActivationBlockCount = 4;
constexpr int kSparseAutoEncoderFeatureDimension = 8 * kGpt2ModelWidth;
constexpr float kSparseAutoEncoderPenalty = 0.5f;
static_assert(kSparseAutoEncoderActivationBlockCount <=
              kGpt2TransformerBlockCount);

template <class T>
void AddIfExplicitlySet(const absl::Flag<T>& flag,
                        std::vector<absl::string_view>* names) {
  if (flag.IsSpecifiedOnCommandLine())
    names->push_back(flag.Name());
}

// Validates only explicit command-line uses. Defaults for flags owned by other
// modes are harmless because the selected mode never reads them.
absl::StatusOr<Gpt2ShakespeareMode> ParseAndValidateRunMode() {
  ASSIGN_OR_RETURN(auto mode,
                   ParseGpt2ShakespeareMode(absl::GetFlag(FLAGS_mode)));
  std::vector<absl::string_view> explicitly_set;
  AddIfExplicitlySet(FLAGS_corpus, &explicitly_set);
  AddIfExplicitlySet(FLAGS_resume_from, &explicitly_set);
  AddIfExplicitlySet(FLAGS_inference_from, &explicitly_set);
  AddIfExplicitlySet(FLAGS_sparse_autoencoder_from, &explicitly_set);
  AddIfExplicitlySet(FLAGS_checkpoint_dir, &explicitly_set);
  AddIfExplicitlySet(FLAGS_checkpoint_every, &explicitly_set);
  AddIfExplicitlySet(FLAGS_checkpoint_initial, &explicitly_set);
  AddIfExplicitlySet(FLAGS_steps, &explicitly_set);
  AddIfExplicitlySet(FLAGS_training_seconds, &explicitly_set);
  AddIfExplicitlySet(FLAGS_learning_rate, &explicitly_set);
  AddIfExplicitlySet(FLAGS_adam_beta1, &explicitly_set);
  AddIfExplicitlySet(FLAGS_adam_beta2, &explicitly_set);
  AddIfExplicitlySet(FLAGS_adam_epsilon, &explicitly_set);
  AddIfExplicitlySet(FLAGS_weight_decay, &explicitly_set);
  AddIfExplicitlySet(FLAGS_eval_batches, &explicitly_set);
  AddIfExplicitlySet(FLAGS_test_fraction, &explicitly_set);
  AddIfExplicitlySet(FLAGS_train_until_loss, &explicitly_set);
  AddIfExplicitlySet(FLAGS_training_eval_interval, &explicitly_set);
  AddIfExplicitlySet(FLAGS_prompt, &explicitly_set);
  AddIfExplicitlySet(FLAGS_generation_tokens, &explicitly_set);
  AddIfExplicitlySet(FLAGS_temperature, &explicitly_set);
  AddIfExplicitlySet(FLAGS_batch_size, &explicitly_set);
  AddIfExplicitlySet(FLAGS_log_file, &explicitly_set);
  RETURN_IF_ERROR(ValidateGpt2ShakespeareModeFlags(
      mode, explicitly_set, absl::GetFlag(FLAGS_inference_from),
      absl::GetFlag(FLAGS_sparse_autoencoder_from)));
  if (FLAGS_training_seconds.IsSpecifiedOnCommandLine()) {
    RETURN_IF_ERROR(ValidateGpt2ShakespeareTrainingSeconds(
        absl::GetFlag(FLAGS_training_seconds)));
  }
  return mode;
}

std::optional<double> TrainingSecondsFromFlags() {
  if (!FLAGS_training_seconds.IsSpecifiedOnCommandLine())
    return std::nullopt;
  return absl::GetFlag(FLAGS_training_seconds);
}

const char* TrainingStopReason(const TrainingResult& result) {
  if (result.reached_time_limit)
    return "time_limit";
  if (result.reached_stop_loss)
    return "stop_loss";
  return "step_limit";
}

std::string CurrentTimestamp() {
  return absl::FormatTime("%Y-%m-%d %H:%M:%S UTC", absl::Now(),
                          absl::UTCTimeZone());
}

struct ModelConfig {
  int batch_size;

  int token_count() const { return batch_size * kGpt2ContextLength; }
  int padded_vocabulary_size() const { return kGpt2PaddedVocabularySize; }

  absl::Status Validate() const {
    if (batch_size <= 0)
      return absl::InvalidArgumentError("batch_size must be positive");
    if (batch_size > std::numeric_limits<int>::max() / kGpt2ContextLength)
      return absl::InvalidArgumentError("batch_size is too large");
    return absl::OkStatus();
  }
};

AdamWConfig OptimizerConfigFromFlags() {
  return AdamWConfig{
      .learning_rate = static_cast<float>(absl::GetFlag(FLAGS_learning_rate)),
      .beta1 = static_cast<float>(absl::GetFlag(FLAGS_adam_beta1)),
      .beta2 = static_cast<float>(absl::GetFlag(FLAGS_adam_beta2)),
      .epsilon = static_cast<float>(absl::GetFlag(FLAGS_adam_epsilon)),
      .weight_decay = static_cast<float>(absl::GetFlag(FLAGS_weight_decay)),
  };
}

std::string CorpusPath() {
  const std::string requested = absl::GetFlag(FLAGS_corpus);
  if (!requested.empty())
    return requested;

  if (const char* test_srcdir = std::getenv("TEST_SRCDIR")) {
    if (const char* workspace = std::getenv("TEST_WORKSPACE")) {
      const std::string runfile = absl::StrCat(test_srcdir, "/", workspace,
                                               "/testdata/shakespeare.txt");
      if (std::filesystem::exists(runfile))
        return runfile;
    }
  }
  return "testdata/shakespeare.txt";
}

absl::StatusOr<std::filesystem::path> TokenizerDirectory() {
  const std::string requested = absl::GetFlag(FLAGS_tokenizer_dir);
  if (!requested.empty())
    return std::filesystem::path(requested);
  if (const char* environment = std::getenv("PLUTO_GPT2_TOKENIZER_DIR"))
    return std::filesystem::path(environment);
  return absl::FailedPreconditionError(
      "set --tokenizer_dir or PLUTO_GPT2_TOKENIZER_DIR to the GPT-2 "
      "tokenizer directory");
}

// Training and evaluation stay device-resident. This recipe crosses the
// synchronization boundary only when a scalar must be printed or reused as a
// host-side stopping value.
absl::StatusOr<double> ReadEvaluationLoss(cuda::Executor& executor,
                                          const Buffer& loss) {
  if (&loss.executor() != &executor || loss.size_bytes() != sizeof(float)) {
    return absl::InvalidArgumentError(
        "evaluation must return one FP32 scalar on its CUDA Executor");
  }
  ASSIGN_OR_RETURN(auto host_loss,
                   cuda::PageLockedHostArray<float>::Allocate(executor, 1));
  RETURN_IF_ERROR(cuda::CudaStatus(
      cudaMemcpyAsync(host_loss.data(), loss.data(), loss.size_bytes(),
                      cudaMemcpyDeviceToHost, executor.stream()),
      "cudaMemcpyAsync(evaluation loss for logging)"));
  RETURN_IF_ERROR(executor.Synchronize());
  return host_loss[0];
}

absl::StatusOr<cuda::PageLockedHostArray<float>> Predict(
    cuda::Executor& executor, const ModelConfig& config, const Layer& model,
    const std::vector<int>& context, const Buffer& token_buffer) {
  if (&token_buffer.executor() != &executor) {
    return absl::InvalidArgumentError(
        "prediction requires its token buffer's CUDA Executor");
  }
  if (context.empty())
    return absl::InvalidArgumentError("prediction context must not be empty");
  const size_t context_size =
      std::min(context.size(), static_cast<size_t>(kGpt2ContextLength));
  const size_t context_start = context.size() - context_size;
  ASSIGN_OR_RETURN(
      auto repeated_context,
      cuda::PageLockedHostArray<int>::Allocate(executor, config.token_count()));
  for (int sequence = 0; sequence < config.batch_size; ++sequence) {
    for (size_t position = 0; position < context_size; ++position) {
      repeated_context[sequence * kGpt2ContextLength + position] =
          context[context_start + position];
    }
    // Later rows are causally invisible to the selected output row.
    for (size_t position = context_size;
         position < static_cast<size_t>(kGpt2ContextLength); ++position) {
      repeated_context[sequence * kGpt2ContextLength + position] =
          context.back();
    }
  }
  RETURN_IF_ERROR(cuda::CudaStatus(
      cudaMemcpyAsync(token_buffer.data(), repeated_context.data(),
                      token_buffer.size_bytes(), cudaMemcpyHostToDevice,
                      executor.stream()),
      "cudaMemcpyAsync(prompt context)"));

  BufferVec inputs = {token_buffer};
  ASSIGN_OR_RETURN(auto logits_fwd, model.fwd(executor, inputs));
  auto logits = std::move(logits_fwd.output);

  ASSIGN_OR_RETURN(auto host_logits, cuda::PageLockedHostArray<float>::Allocate(
                                         executor, kGpt2VocabularySize));
  const size_t output_row = context_size - 1;
  const auto* selected_logits = static_cast<const float*>(logits.data()) +
                                output_row * config.padded_vocabulary_size();
  RETURN_IF_ERROR(cuda::CudaStatus(
      cudaMemcpyAsync(host_logits.data(), selected_logits,
                      host_logits.size() * sizeof(float),
                      cudaMemcpyDeviceToHost, executor.stream()),
      "cudaMemcpyAsync(prompt logits)"));
  RETURN_IF_ERROR(executor.Synchronize());
  return host_logits;
}

absl::StatusOr<std::string> Generate(cuda::Executor& executor,
                                     const ModelConfig& config,
                                     const Layer& model,
                                     const tokenizer::Tokenizer& tokenizer,
                                     const tokenizer::Detokenizer& detokenizer,
                                     std::string prompt, int generation_tokens,
                                     double temperature, std::mt19937& random,
                                     const Buffer& token_buffer) {
  RETURN_IF_ERROR(ValidateGenerationOptions(generation_tokens, temperature));
  if (prompt.empty())
    prompt = "\n";
  ASSIGN_OR_RETURN(auto encoded_prompt, tokenizer.Encode(executor, prompt));
  std::vector<int> context(encoded_prompt.begin(), encoded_prompt.end());
  std::vector<int> generated;
  generated.reserve(generation_tokens);

  for (int index = 0; index < generation_tokens; ++index) {
    ASSIGN_OR_RETURN(auto logits,
                     Predict(executor, config, model, context, token_buffer));
    ASSIGN_OR_RETURN(const int next,
                     SelectNextToken(logits.span(), temperature, random));
    context.push_back(next);
    generated.push_back(next);
  }
  return detokenizer.Decode(generated);
}

absl::Status RunTraining(cuda::Executor& executor,
                         const std::filesystem::path& resume_from) {
  const std::string log_path = absl::GetFlag(FLAGS_log_file);
  if (log_path.empty())
    return absl::InvalidArgumentError("log_file must not be empty");
  std::ofstream log_file(log_path, std::ios::out | std::ios::trunc);
  if (!log_file.is_open()) {
    return absl::FailedPreconditionError(
        absl::StrCat("cannot open training log for writing: ", log_path));
  }
  util::TeeStream logger(std::cout, log_file);
  logger << "training log: " << log_path << '\n';
  const int checkpoint_every = absl::GetFlag(FLAGS_checkpoint_every);
  std::filesystem::path checkpoint_root = absl::GetFlag(FLAGS_checkpoint_dir);
  if (checkpoint_root.empty() && !resume_from.empty())
    checkpoint_root = resume_from;
  if (checkpoint_every < 0)
    return absl::InvalidArgumentError("checkpoint_every must be non-negative");
  if ((checkpoint_every > 0 || absl::GetFlag(FLAGS_checkpoint_initial)) &&
      checkpoint_root.empty()) {
    return absl::InvalidArgumentError(
        "checkpoint_dir must not be empty when checkpoint_every is positive "
        "or checkpoint_initial is enabled");
  }

  ASSIGN_OR_RETURN(auto corpus, LoadTextCorpus(CorpusPath()));
  ASSIGN_OR_RETURN(auto tokenizer_directory, TokenizerDirectory());
  ASSIGN_OR_RETURN(auto tokenizer, Gpt2Tokenizer::Load(tokenizer_directory));
  if (tokenizer->vocab_size() != kGpt2VocabularySize) {
    return absl::FailedPreconditionError(absl::StrCat(
        "the model requires the GPT-2 vocabulary of ", kGpt2VocabularySize,
        " tokens; encoder reports ", tokenizer->vocab_size()));
  }

  const ModelConfig config{.batch_size = absl::GetFlag(FLAGS_batch_size)};
  RETURN_IF_ERROR(config.Validate());
  ASSIGN_OR_RETURN(auto corpus_split,
                   SplitCorpus(corpus, absl::GetFlag(FLAGS_test_fraction)));

  ASSIGN_OR_RETURN(auto model, CreateGpt2(executor, DataType::BF16,
                                          absl::GetFlag(FLAGS_seed)));
  int initial_step = 0;
  if (!resume_from.empty()) {
    ASSIGN_OR_RETURN(const CheckpointInfo checkpoint,
                     ReadLatestCheckpoint(
                         executor, *model, resume_from,
                         [&logger](const CheckpointInfo& malformed,
                                   const absl::Status& status) {
                           logger << '[' << CurrentTimestamp()
                                  << "] WARNING: cannot load checkpoint "
                                  << malformed.directory.string() << " (step "
                                  << malformed.step << "): " << status
                                  << "; trying the previous checkpoint\n";
                         }));
    initial_step = checkpoint.step;
    logger << '[' << CurrentTimestamp()
           << "] resumed from checkpoint: " << checkpoint.directory.string()
           << " (step " << initial_step << ")\n";
  }
  ASSIGN_OR_RETURN(auto loss_layer,
                   CrossEntropyLossLayer::Create(executor, kGpt2VocabularySize,
                                                 DataType::BF16));
  LanguageModelingObjective objective(*model, *loss_layer);
  ASSIGN_OR_RETURN(
      auto optimizer,
      Optimizer::Create(executor, *model, OptimizerConfigFromFlags()));
  const InMemoryDataSetOptions training_data_options{
      .batch_size = config.batch_size,
      .context_length = kGpt2ContextLength,
      .order = InMemoryDataSetOrder::kRandom,
      .seed = static_cast<uint64_t>(absl::GetFlag(FLAGS_seed)),
  };
  const InMemoryDataSetOptions evaluation_data_options{
      .batch_size = config.batch_size,
      .context_length = kGpt2ContextLength,
      .order = InMemoryDataSetOrder::kSequential,
      .seed = static_cast<uint64_t>(absl::GetFlag(FLAGS_seed)),
  };
  ASSIGN_OR_RETURN(auto training_data, MakeInMemoryDataSetIterator(
                                           executor, corpus_split.training,
                                           *tokenizer, training_data_options));
  ASSIGN_OR_RETURN(
      auto training_evaluation_data,
      MakeInMemoryDataSetIterator(executor, corpus_split.training, *tokenizer,
                                  evaluation_data_options));
  ASSIGN_OR_RETURN(
      auto test_evaluation_data,
      MakeInMemoryDataSetIterator(executor, corpus_split.test, *tokenizer,
                                  evaluation_data_options));

  const int eval_batches = absl::GetFlag(FLAGS_eval_batches);
  const EvaluationOptions evaluation_options{.batches = eval_batches};
  ASSIGN_OR_RETURN(auto initial_training_loss_buffer,
                   Evaluate(executor, objective, *training_evaluation_data,
                            evaluation_options));
  ASSIGN_OR_RETURN(double initial_training_loss,
                   ReadEvaluationLoss(executor, initial_training_loss_buffer));
  ASSIGN_OR_RETURN(
      auto initial_test_loss_buffer,
      Evaluate(executor, objective, *test_evaluation_data, evaluation_options));
  ASSIGN_OR_RETURN(double initial_test_loss,
                   ReadEvaluationLoss(executor, initial_test_loss_buffer));
  logger << "model: GPT-2 vocabulary=" << kGpt2VocabularySize
         << ", context=" << kGpt2ContextLength
         << ", layers=" << kGpt2TransformerBlockCount
         << ", width=" << kGpt2ModelWidth << ", heads=" << kGpt2AttentionHeads
         << ", head_dim=" << kGpt2AttentionHeadDimension
         << ", MLP=" << kGpt2FeedForwardWidth << ", BF16 compute\n"
         << "batch: " << config.batch_size << " sequences ("
         << config.token_count() << " tokens)\n"
         << "corpus tokens: "
         << training_data->token_count() + test_evaluation_data->token_count()
         << " (training: " << training_data->token_count()
         << ", test: " << test_evaluation_data->token_count() << ")\n"
         << "starting step: " << initial_step << '\n'
         << "initial training loss: " << initial_training_loss << '\n'
         << "initial test loss: " << initial_test_loss << '\n';

  int last_checkpoint_step = -1;
  const auto save_checkpoint = [&](int step) -> absl::Status {
    if (checkpoint_root.empty() || step == last_checkpoint_step)
      return absl::OkStatus();
    const auto checkpoint = checkpoint_root / absl::StrCat("step_", step);
    RETURN_IF_ERROR(WriteToDirectory(executor, *model, checkpoint));
    last_checkpoint_step = step;
    logger << '[' << CurrentTimestamp()
           << "] wrote checkpoint: " << checkpoint.string() << '\n';
    return absl::OkStatus();
  };
  if (absl::GetFlag(FLAGS_checkpoint_initial))
    RETURN_IF_ERROR(save_checkpoint(initial_step));
  TrainingOptions training_options{
      .max_steps = absl::GetFlag(FLAGS_steps),
      .initial_step = initial_step,
      .training_seconds = TrainingSecondsFromFlags(),
      .evaluation_interval = absl::GetFlag(FLAGS_training_eval_interval),
      .evaluation_batches = eval_batches,
      .stop_loss = absl::GetFlag(FLAGS_train_until_loss),
      .evaluation_tokens = training_evaluation_data.get(),
      .initial_loss = initial_training_loss,
      .evaluation_callback =
          [&logger](int steps_completed, double loss) {
            logger << '[' << CurrentTimestamp() << "] training loss after "
                   << steps_completed << " steps: " << loss << '\n';
          },
  };
  if (checkpoint_every > 0) {
    training_options.step_callback = [&save_checkpoint,
                                      checkpoint_every](int steps_completed) {
      if (steps_completed % checkpoint_every != 0)
        return absl::OkStatus();
      return save_checkpoint(steps_completed);
    };
  }
  if (training_options.training_seconds.has_value()) {
    logger << "training budget seconds: " << *training_options.training_seconds
           << '\n';
  }
  logger << '[' << CurrentTimestamp()
         << "] training started at step: " << initial_step << '\n';
  ASSIGN_OR_RETURN(
      auto training_result,
      Train(executor, objective, *optimizer, *training_data, training_options));
  logger << '[' << CurrentTimestamp()
         << "] training stopped at step: " << training_result.steps_completed
         << '\n'
         << "training stop reason: " << TrainingStopReason(training_result)
         << '\n'
         << "training elapsed seconds: "
         << training_result.elapsed_training_seconds << '\n';
  RETURN_IF_ERROR(save_checkpoint(training_result.steps_completed));
  ASSIGN_OR_RETURN(auto final_training_loss_buffer,
                   Evaluate(executor, objective, *training_evaluation_data,
                            evaluation_options));
  ASSIGN_OR_RETURN(double final_training_loss,
                   ReadEvaluationLoss(executor, final_training_loss_buffer));
  ASSIGN_OR_RETURN(
      auto final_test_loss_buffer,
      Evaluate(executor, objective, *test_evaluation_data, evaluation_options));
  ASSIGN_OR_RETURN(double final_test_loss,
                   ReadEvaluationLoss(executor, final_test_loss_buffer));
  logger << "final training loss: " << final_training_loss << '\n'
         << "final test loss: " << final_test_loss << '\n';

  return absl::OkStatus();
}

absl::Status RunSparseAutoEncoderTraining(
    cuda::Executor& executor, const std::filesystem::path& gpt2_checkpoint_path,
    const std::filesystem::path& resume_from) {
  ASSIGN_OR_RETURN(const CheckpointInfo gpt2_checkpoint,
                   InspectCheckpointDirectory(gpt2_checkpoint_path));
  const std::string log_path = absl::GetFlag(FLAGS_log_file);
  if (log_path.empty())
    return absl::InvalidArgumentError("log_file must not be empty");
  std::ofstream log_file(log_path, std::ios::out | std::ios::trunc);
  if (!log_file.is_open()) {
    return absl::FailedPreconditionError(
        absl::StrCat("cannot open training log for writing: ", log_path));
  }
  util::TeeStream logger(std::cout, log_file);
  logger << "training log: " << log_path << '\n';

  const int checkpoint_every = absl::GetFlag(FLAGS_checkpoint_every);
  std::filesystem::path checkpoint_root = absl::GetFlag(FLAGS_checkpoint_dir);
  if (checkpoint_root.empty() && !resume_from.empty())
    checkpoint_root = resume_from;
  if (checkpoint_every < 0)
    return absl::InvalidArgumentError("checkpoint_every must be non-negative");
  if ((checkpoint_every > 0 || absl::GetFlag(FLAGS_checkpoint_initial)) &&
      checkpoint_root.empty()) {
    return absl::InvalidArgumentError(
        "checkpoint_dir must not be empty when checkpoint_every is positive "
        "or checkpoint_initial is enabled");
  }

  ASSIGN_OR_RETURN(auto corpus, LoadTextCorpus(CorpusPath()));
  ASSIGN_OR_RETURN(auto tokenizer_directory, TokenizerDirectory());
  ASSIGN_OR_RETURN(auto tokenizer, Gpt2Tokenizer::Load(tokenizer_directory));
  if (tokenizer->vocab_size() != kGpt2VocabularySize) {
    return absl::FailedPreconditionError(absl::StrCat(
        "the model requires the GPT-2 vocabulary of ", kGpt2VocabularySize,
        " tokens; encoder reports ", tokenizer->vocab_size()));
  }

  const ModelConfig config{.batch_size = absl::GetFlag(FLAGS_batch_size)};
  RETURN_IF_ERROR(config.Validate());
  const InMemoryDataSetOptions training_source_options{
      .batch_size = config.batch_size,
      .context_length = kGpt2ContextLength,
      .order = InMemoryDataSetOrder::kRandom,
      .seed = static_cast<uint64_t>(absl::GetFlag(FLAGS_seed)),
  };
  const InMemoryDataSetOptions evaluation_source_options{
      .batch_size = config.batch_size,
      .context_length = kGpt2ContextLength,
      .order = InMemoryDataSetOrder::kSequential,
      .seed = static_cast<uint64_t>(absl::GetFlag(FLAGS_seed)),
  };
  ASSIGN_OR_RETURN(auto training_source,
                   MakeInMemoryDataSetIterator(executor, corpus, *tokenizer,
                                               training_source_options));
  ASSIGN_OR_RETURN(auto evaluation_source,
                   MakeInMemoryDataSetIterator(executor, corpus, *tokenizer,
                                               evaluation_source_options));

  ASSIGN_OR_RETURN(auto activation_generator,
                   CreateActivationGenerator(
                       executor, kSparseAutoEncoderActivationBlockCount,
                       DataType::BF16, absl::GetFlag(FLAGS_seed)));
  ASSIGN_OR_RETURN(auto training_activations,
                   SparseAutoEncoderDataSetIterator::Create(
                       executor, *activation_generator, *training_source,
                       gpt2_checkpoint.directory));
  ASSIGN_OR_RETURN(auto evaluation_activations,
                   SparseAutoEncoderDataSetIterator::Create(
                       executor, *activation_generator, *evaluation_source,
                       gpt2_checkpoint.directory));

  ASSIGN_OR_RETURN(auto autoencoder,
                   SparseAutoEncoderLayer::Create(
                       executor, kGpt2ModelWidth,
                       kSparseAutoEncoderFeatureDimension, DataType::BF16));
  RETURN_IF_ERROR(autoencoder->InitializeNormal(
      1.0f / std::sqrt(static_cast<float>(kGpt2ModelWidth)),
      static_cast<uint64_t>(absl::GetFlag(FLAGS_seed)) + 20'000));
  int initial_step = 0;
  if (!resume_from.empty()) {
    ASSIGN_OR_RETURN(const CheckpointInfo checkpoint,
                     ReadLatestCheckpoint(
                         executor, *autoencoder, resume_from,
                         [&logger](const CheckpointInfo& malformed,
                                   const absl::Status& status) {
                           logger << '[' << CurrentTimestamp()
                                  << "] WARNING: cannot load SAE checkpoint "
                                  << malformed.directory.string() << " (step "
                                  << malformed.step << "): " << status
                                  << "; trying the previous checkpoint\n";
                         }));
    initial_step = checkpoint.step;
    logger << '[' << CurrentTimestamp()
           << "] resumed SAE from checkpoint: " << checkpoint.directory.string()
           << " (step " << initial_step << ")\n";
  }
  ASSIGN_OR_RETURN(
      auto loss_layer,
      SparseAutoEncoderLossLayer::Create(
          executor, kGpt2ModelWidth, kSparseAutoEncoderFeatureDimension,
          kSparseAutoEncoderPenalty, DataType::BF16));
  SparseAutoEncoderObjective objective(*autoencoder, *loss_layer);
  ASSIGN_OR_RETURN(
      auto optimizer,
      Optimizer::Create(executor, *autoencoder, OptimizerConfigFromFlags()));

  const int eval_batches = absl::GetFlag(FLAGS_eval_batches);
  const EvaluationOptions evaluation_options{.batches = eval_batches};
  ASSIGN_OR_RETURN(auto initial_loss_buffer,
                   Evaluate(executor, objective, *evaluation_activations,
                            evaluation_options));
  ASSIGN_OR_RETURN(double initial_loss,
                   ReadEvaluationLoss(executor, initial_loss_buffer));
  logger << "mode: sparse autoencoder training\n"
         << "GPT-2 checkpoint: " << gpt2_checkpoint.directory.string()
         << " (step " << gpt2_checkpoint.step << ")\n"
         << "activation tap: after transformer block "
         << kSparseAutoEncoderActivationBlockCount << '\n'
         << "SAE dimensions: d=" << kGpt2ModelWidth
         << ", m=" << kSparseAutoEncoderFeatureDimension
         << ", sparsity penalty=" << kSparseAutoEncoderPenalty << '\n'
         << "batch: " << config.batch_size << " sequences ("
         << config.token_count() << " activations)\n"
         << "corpus tokens: " << training_source->token_count() << '\n'
         << "starting SAE step: " << initial_step << '\n'
         << "initial loss per activation: " << initial_loss << '\n';

  int last_checkpoint_step = -1;
  const auto save_checkpoint = [&](int step) -> absl::Status {
    if (checkpoint_root.empty() || step == last_checkpoint_step)
      return absl::OkStatus();
    const auto checkpoint = checkpoint_root / absl::StrCat("step_", step);
    RETURN_IF_ERROR(WriteToDirectory(executor, *autoencoder, checkpoint));
    last_checkpoint_step = step;
    logger << '[' << CurrentTimestamp()
           << "] wrote SAE checkpoint: " << checkpoint.string() << '\n';
    return absl::OkStatus();
  };
  if (absl::GetFlag(FLAGS_checkpoint_initial))
    RETURN_IF_ERROR(save_checkpoint(initial_step));
  TrainingOptions training_options{
      .max_steps = absl::GetFlag(FLAGS_steps),
      .initial_step = initial_step,
      .training_seconds = TrainingSecondsFromFlags(),
      .evaluation_interval = absl::GetFlag(FLAGS_training_eval_interval),
      .evaluation_batches = eval_batches,
      .stop_loss = absl::GetFlag(FLAGS_train_until_loss),
      .evaluation_tokens = evaluation_activations.get(),
      .initial_loss = initial_loss,
      .evaluation_callback =
          [&logger](int steps_completed, double loss) {
            logger << '[' << CurrentTimestamp() << "] SAE loss after "
                   << steps_completed << " steps: " << loss << '\n';
          },
  };
  if (checkpoint_every > 0) {
    training_options.step_callback = [&save_checkpoint,
                                      checkpoint_every](int steps_completed) {
      if (steps_completed % checkpoint_every != 0)
        return absl::OkStatus();
      return save_checkpoint(steps_completed);
    };
  }
  if (training_options.training_seconds.has_value()) {
    logger << "training budget seconds: " << *training_options.training_seconds
           << '\n';
  }
  logger << '[' << CurrentTimestamp()
         << "] SAE training started at step: " << initial_step << '\n';
  ASSIGN_OR_RETURN(auto training_result,
                   Train(executor, objective, *optimizer, *training_activations,
                         training_options));
  logger << '[' << CurrentTimestamp() << "] SAE training stopped at step: "
         << training_result.steps_completed << '\n'
         << "training stop reason: " << TrainingStopReason(training_result)
         << '\n'
         << "training elapsed seconds: "
         << training_result.elapsed_training_seconds << '\n';
  RETURN_IF_ERROR(save_checkpoint(training_result.steps_completed));
  ASSIGN_OR_RETURN(auto final_loss_buffer,
                   Evaluate(executor, objective, *evaluation_activations,
                            evaluation_options));
  ASSIGN_OR_RETURN(double final_loss,
                   ReadEvaluationLoss(executor, final_loss_buffer));
  logger << "final loss per activation: " << final_loss << '\n';
  return absl::OkStatus();
}

absl::Status RunInference(cuda::Executor& executor,
                          const std::filesystem::path& checkpoint_path) {
  ASSIGN_OR_RETURN(const CheckpointInfo checkpoint,
                   InspectCheckpointDirectory(checkpoint_path));
  ASSIGN_OR_RETURN(auto tokenizer_directory, TokenizerDirectory());
  ASSIGN_OR_RETURN(auto tokenizer, Gpt2Tokenizer::Load(tokenizer_directory));
  ASSIGN_OR_RETURN(auto detokenizer, Gpt2Detokenizer::Load(tokenizer_directory));
  if (tokenizer->vocab_size() != kGpt2VocabularySize ||
      detokenizer->vocab_size() != kGpt2VocabularySize) {
    return absl::FailedPreconditionError(absl::StrCat(
        "the model requires the GPT-2 vocabulary of ", kGpt2VocabularySize,
        " tokens; encoder reports ", tokenizer->vocab_size(),
        " and decoder reports ", detokenizer->vocab_size()));
  }

  const int generation_tokens = absl::GetFlag(FLAGS_generation_tokens);
  const double temperature = absl::GetFlag(FLAGS_temperature);
  RETURN_IF_ERROR(ValidateGenerationOptions(generation_tokens, temperature));

  const ModelConfig config{.batch_size = 1};
  RETURN_IF_ERROR(config.Validate());
  ASSIGN_OR_RETURN(auto model, CreateGpt2(executor, DataType::BF16,
                                          absl::GetFlag(FLAGS_seed)));
  RETURN_IF_ERROR(ReadFromDirectory(executor, *model, checkpoint.directory));
  ASSIGN_OR_RETURN(
      auto token_buffer,
      Buffer::Allocate(executor, config.token_count() * sizeof(int)));
  std::cout << "loaded checkpoint: " << checkpoint.directory.string()
            << " (step " << checkpoint.step << ")\n";

  // Offset in the engine's unsigned type: an INT_MAX CLI seed must not cause
  // signed overflow before conversion to the generator's seed type.
  std::mt19937 random(
      static_cast<std::mt19937::result_type>(absl::GetFlag(FLAGS_seed)) + 1);
  const std::string one_shot_prompt = absl::GetFlag(FLAGS_prompt);
  if (!one_shot_prompt.empty()) {
    ASSIGN_OR_RETURN(auto completion,
                     Generate(executor, config, *model, *tokenizer,
                              *detokenizer, one_shot_prompt, generation_tokens,
                              temperature, random, token_buffer));
    std::cout << one_shot_prompt << completion << '\n';
    return absl::OkStatus();
  }

  std::cout << "Enter a prompt (Ctrl-C or Ctrl-D to quit).\n";
  std::string prompt;
  while (true) {
    std::cout << "> " << std::flush;
    if (!std::getline(std::cin, prompt))
      break;
    ASSIGN_OR_RETURN(
        auto completion,
        Generate(executor, config, *model, *tokenizer, *detokenizer, prompt,
                 generation_tokens, temperature, random, token_buffer));
    std::cout << prompt << completion << '\n';
  }
  return absl::OkStatus();
}

// The same embeddings and first four blocks used to train the SAE produce
// one activation per prompt token. Pad to the model's fixed context length;
// causal attention makes the padding invisible to all real prompt positions.
// ReadZStatistics excludes these trailing rows, including a partial SAE tile.
absl::Status PrintSparseAutoEncoderStatistics(
    cuda::Executor& executor, const tokenizer::Tokenizer& tokenizer,
    const Layer& activation_generator,
    const SparseAutoEncoderLayer& autoencoder, absl::string_view prompt,
    const Buffer& token_buffer) {
  ASSIGN_OR_RETURN(auto tokens, tokenizer.Encode(executor, prompt));
  if (tokens.empty()) {
    std::cout << "Prompt contains no tokens; no Z statistics.\n";
    return absl::OkStatus();
  }
  const int rows = static_cast<int>(
      std::min(tokens.size(), static_cast<size_t>(kGpt2ContextLength)));
  const size_t start = tokens.size() - rows;
  if (start != 0) {
    std::cout << "Prompt has " << tokens.size() << " tokens; using its last "
              << rows << " tokens (the GPT-2 context limit).\n";
  }
  ASSIGN_OR_RETURN(auto context, cuda::PageLockedHostArray<int>::Allocate(
                                     executor, kGpt2ContextLength));
  std::copy_n(tokens.data() + start, rows, context.data());
  std::fill(context.begin() + rows, context.end(), context[rows - 1]);
  RETURN_IF_ERROR(cuda::CudaStatus(
      cudaMemcpyAsync(token_buffer.data(), context.data(), context.size_bytes(),
                      cudaMemcpyHostToDevice, executor.stream()),
      "copy SAE prompt context"));

  ASSIGN_OR_RETURN(auto activations_fwd,
                   activation_generator.fwd(executor, BufferVec{token_buffer}));
  auto activations = std::move(activations_fwd.output);

  // Inference never runs backward through the frozen GPT-2 prefix.
  activations_fwd.state = {};

  ASSIGN_OR_RETURN(auto reconstruction_fwd,
                   autoencoder.fwd(executor, BufferVec{activations}));
  auto reconstruction = std::move(reconstruction_fwd.output);

  ASSIGN_OR_RETURN(auto stats, autoencoder.ReadZStatistics(
                                   executor, reconstruction_fwd.state, rows));
  std::cout << "Z statistics (prompt tokens only):\n"
            << "  tokens: " << stats.rows << ", features: " << stats.feature_dim
            << '\n'
            << "  active values (Z > 0): " << stats.active_count << " / "
            << static_cast<int64_t>(stats.rows) * stats.feature_dim << '\n'
            << "  zero fraction: " << stats.zero_fraction() << '\n'
            << "  mean active features per token (L0): "
            << stats.mean_active_features() << '\n'
            << "  mean: " << stats.mean << '\n'
            << "  stddev: " << stats.standard_deviation << '\n'
            << "  max: " << stats.maximum << '\n';
  return absl::OkStatus();
}

absl::Status RunSparseAutoEncoderInference(
    cuda::Executor& executor, const std::filesystem::path& gpt2_path,
    const std::filesystem::path& sae_path) {
  ASSIGN_OR_RETURN(auto gpt2_checkpoint, InspectCheckpointDirectory(gpt2_path));
  ASSIGN_OR_RETURN(auto sae_checkpoint, InspectCheckpointDirectory(sae_path));
  ASSIGN_OR_RETURN(auto tokenizer_directory, TokenizerDirectory());
  ASSIGN_OR_RETURN(auto tokenizer, Gpt2Tokenizer::Load(tokenizer_directory));
  if (tokenizer->vocab_size() != kGpt2VocabularySize) {
    return absl::FailedPreconditionError(
        "SAE inference requires the GPT-2 vocabulary");
  }
  ASSIGN_OR_RETURN(auto activation_generator,
                   CreateActivationGenerator(
                       executor, kSparseAutoEncoderActivationBlockCount,
                       DataType::BF16, absl::GetFlag(FLAGS_seed)));
  RETURN_IF_ERROR(ReadFromDirectory(executor, *activation_generator,
                                    gpt2_checkpoint.directory));
  ASSIGN_OR_RETURN(
      auto autoencoder,
      SparseAutoEncoderLayer::Create(
          executor, kGpt2ModelWidth, kSparseAutoEncoderFeatureDimension,
          DataType::BF16, SparseAutoEncoderLayer::Mode::kCollectStatistics));
  RETURN_IF_ERROR(
      ReadFromDirectory(executor, *autoencoder, sae_checkpoint.directory));
  ASSIGN_OR_RETURN(auto token_buffer,
                   Buffer::Allocate(executor, kGpt2ContextLength * sizeof(int)));
  std::cout << "GPT-2 checkpoint: " << gpt2_checkpoint.directory.string()
            << '\n'
            << "SAE checkpoint: " << sae_checkpoint.directory.string() << '\n'
            << "activation tap: after transformer block "
            << kSparseAutoEncoderActivationBlockCount << '\n';
  const std::string one_shot_prompt = absl::GetFlag(FLAGS_prompt);
  if (!one_shot_prompt.empty()) {
    return PrintSparseAutoEncoderStatistics(executor, *tokenizer,
                                            *activation_generator, *autoencoder,
                                            one_shot_prompt, token_buffer);
  }
  std::cout << "Enter a prompt (Ctrl-C or Ctrl-D to quit).\n";
  std::string prompt;
  while (true) {
    std::cout << "> " << std::flush;
    if (!std::getline(std::cin, prompt))
      break;
    RETURN_IF_ERROR(PrintSparseAutoEncoderStatistics(
        executor, *tokenizer, *activation_generator, *autoencoder, prompt,
        token_buffer));
  }
  return absl::OkStatus();
}

absl::Status Run(cuda::Executor& executor, Gpt2ShakespeareMode mode) {
  const std::filesystem::path resume_from = absl::GetFlag(FLAGS_resume_from);
  switch (mode) {
    case Gpt2ShakespeareMode::kTrainModel:
      return RunTraining(executor, resume_from);
    case Gpt2ShakespeareMode::kInferModel:
      return RunInference(executor, absl::GetFlag(FLAGS_inference_from));
    case Gpt2ShakespeareMode::kTrainSparseAutoEncoder:
      return RunSparseAutoEncoderTraining(
          executor, absl::GetFlag(FLAGS_sparse_autoencoder_from), resume_from);
    case Gpt2ShakespeareMode::kInferSparseAutoEncoder:
      return RunSparseAutoEncoderInference(
          executor, absl::GetFlag(FLAGS_sparse_autoencoder_from),
          absl::GetFlag(FLAGS_inference_from));
  }
  return absl::InternalError("unknown GPT-2 Shakespeare run mode");
}

}  // namespace
}  // namespace pluto::llm

int main(int argc, char** argv) {
  const std::vector<char*> positional = absl::ParseCommandLine(argc, argv);
  if (positional.size() != 1) {
    std::cerr << "This binary accepts flags only.\n";
    return 2;
  }
  auto mode = pluto::llm::ParseAndValidateRunMode();
  if (!mode.ok()) {
    std::cerr << mode.status() << '\n';
    return 2;
  }
  auto executor = pluto::cuda::Executor::Create();
  if (!executor.ok()) {
    std::cerr << executor.status() << '\n';
    return 1;
  }
  const absl::Status status = pluto::llm::Run(**executor, *mode);
  // Run() destroys every Buffer, queueing stream-ordered frees before this
  // synchronization. Executor destruction then releases the native stream.
  const absl::Status sync_status = (*executor)->Synchronize();
  if (!status.ok()) {
    std::cerr << status << '\n';
    return 1;
  }
  if (!sync_status.ok()) {
    std::cerr << sync_status << '\n';
    return 1;
  }
  return 0;
}
