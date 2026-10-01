#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <system_error>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "src/cuda/buffer.h"
#include "src/cuda/executor.h"
#include "src/cuda/page_locked_host_array.h"
#include "src/dataset/dataset.h"
#include "src/llm/adamw_optimizer.h"
#include "src/llm/checkpoint.h"
#include "src/llm/experiments/finite_state_machine/dataset.h"
#include "src/llm/experiments/finite_state_machine/tokenizer.h"
#include "src/llm/gpt2.h"
#include "src/llm/layer.h"
#include "src/llm/layers/cross_entropy_loss.h"
#include "src/llm/optimizer.h"
#include "src/llm/trainer.h"
#include "src/util/status_macros.h"
#include "src/util/tee_stream.h"

ABSL_FLAG(std::string, training_data, "",
          "Training data; defaults to "
          "testdata/finite_state_machine_training_data.txt");
ABSL_FLAG(std::string, test_data, "",
          "Test data; defaults to testdata/finite_state_machine_test_data.txt");
ABSL_FLAG(std::string, checkpoint_dir, "",
          "Required empty or new directory for this run's step_N checkpoints");
ABSL_FLAG(int, checkpoint_every, 100,
          "Save weights every N updates; zero disables periodic saves. Initial "
          "and final weights are always saved");
ABSL_FLAG(int, eval_every, 100,
          "Report fixed training-sample and full test loss every N updates");
ABSL_FLAG(int, eval_samples, 128,
          "Training examples evaluated, rounded up to a full batch and capped "
          "at the dataset size; zero evaluates the entire training dataset");
ABSL_FLAG(int, batch_size, 4, "Independent sequences per batch");
ABSL_FLAG(int, layers, pluto::llm::kGpt2TransformerBlockCount,
          "Number of transformer blocks; model and MLP widths are unchanged");
ABSL_FLAG(int, attention_heads, pluto::llm::kGpt2AttentionHeads,
          "Attention heads per block; must divide model width 512 (2 gives "
          "256 dimensions per head)");
ABSL_FLAG(int, steps, 10000,
          "Number of optimizer updates; zero evaluates only");
ABSL_FLAG(double, learning_rate, 3e-4, "AdamW learning rate");
ABSL_FLAG(int, seed, 17, "Model initialization and training shuffle seed");
ABSL_FLAG(bool, answer_only, true,
          "Train/evaluate only output tokens after > (final answer or full "
          "state trace); "
          "false supervises every next token in the sample (no EOS)");
ABSL_FLAG(std::string, log_file, "",
          "New log file; defaults to checkpoint_dir/train.log");

namespace pluto::llm::fsm {
namespace {

using Clock = std::chrono::steady_clock;

std::string Timestamp() {
  return absl::FormatTime("%Y-%m-%d %H:%M:%S UTC", absl::Now(),
                          absl::UTCTimeZone());
}

absl::Status FileSystemError(const char* operation,
                             const std::filesystem::path& path,
                             const std::error_code& error) {
  return absl::FailedPreconditionError(
      absl::StrCat(operation, " ", path.string(), ": ", error.message()));
}

absl::Status ValidateFlags() {
  if (absl::GetFlag(FLAGS_checkpoint_dir).empty())
    return absl::InvalidArgumentError("--checkpoint_dir is required");
  if (absl::GetFlag(FLAGS_layers) <= 0)
    return absl::InvalidArgumentError("--layers must be positive");
  const int attention_heads = absl::GetFlag(FLAGS_attention_heads);
  if (attention_heads <= 0 || kGpt2ModelWidth % attention_heads != 0)
    return absl::InvalidArgumentError(
        "--attention_heads must be positive and divide model width 512");
  if (absl::GetFlag(FLAGS_steps) < 0)
    return absl::InvalidArgumentError("--steps must be non-negative");
  if (absl::GetFlag(FLAGS_checkpoint_every) < 0)
    return absl::InvalidArgumentError(
        "--checkpoint_every must be non-negative");
  if (absl::GetFlag(FLAGS_eval_every) <= 0)
    return absl::InvalidArgumentError("--eval_every must be positive");
  if (absl::GetFlag(FLAGS_eval_samples) < 0)
    return absl::InvalidArgumentError("--eval_samples must be non-negative");
  const int batch_size = absl::GetFlag(FLAGS_batch_size);
  if (batch_size <= 0 ||
      batch_size > std::numeric_limits<int>::max() / kGpt2ContextLength) {
    return absl::InvalidArgumentError(
        "--batch_size must be positive and fit the context");
  }
  const double learning_rate = absl::GetFlag(FLAGS_learning_rate);
  if (!std::isfinite(learning_rate) || learning_rate <= 0.0 ||
      !std::isfinite(static_cast<float>(learning_rate)) ||
      static_cast<float>(learning_rate) <= 0.0f) {
    return absl::InvalidArgumentError(
        "--learning_rate must be finite, positive, and representable in FP32");
  }
  return absl::OkStatus();
}

std::string DataPath(const std::string& requested, const char* filename) {
  if (!requested.empty())
    return requested;
  if (const char* test_srcdir = std::getenv("TEST_SRCDIR")) {
    if (const char* workspace = std::getenv("TEST_WORKSPACE")) {
      const auto runfile = std::filesystem::path(test_srcdir) / workspace /
                           "testdata" / filename;
      std::error_code error;
      if (std::filesystem::exists(runfile, error) && !error)
        return runfile.string();
    }
  }
  return (std::filesystem::path("testdata") / filename).string();
}

// Each invocation starts from random weights. A populated destination is never
// interpreted as a resume request or silently reused for a different run.
absl::Status PrepareCheckpointRoot(const std::filesystem::path& root) {
  std::error_code error;
  const bool exists = std::filesystem::exists(root, error);
  if (error)
    return FileSystemError("cannot inspect", root, error);
  if (exists) {
    const bool directory = std::filesystem::is_directory(root, error);
    if (error)
      return FileSystemError("cannot inspect", root, error);
    if (!directory)
      return absl::FailedPreconditionError(
          "checkpoint_dir must be a directory");
    const bool empty = std::filesystem::is_empty(root, error);
    if (error)
      return FileSystemError("cannot list", root, error);
    if (!empty) {
      return absl::AlreadyExistsError(absl::StrCat(
          "refusing to overwrite a nonempty checkpoint directory: ",
          root.string()));
    }
  } else {
    std::filesystem::create_directories(root, error);
    if (error)
      return FileSystemError("cannot create", root, error);
  }
  return absl::OkStatus();
}

size_t ParameterCount(const Layer& model) {
  absl::flat_hash_set<const void*> seen;
  size_t count = 0;
  for (const Buffer& weight : model.weights())
    if (seen.insert(weight.data()).second)
      count += weight.size_bytes() / sizeof(float);
  return count;
}

absl::StatusOr<double> EvaluationLoss(cuda::Executor& executor,
                                      const Layer& model,
                                      const EvaluationOptions& options) {
  ASSIGN_OR_RETURN(auto loss, Evaluate(executor, model, options));
  if (&loss.executor() != &executor || loss.size_bytes() != sizeof(float))
    return absl::InternalError("evaluation must return one FP32 loss scalar");
  ASSIGN_OR_RETURN(auto host_loss,
                   cuda::PageLockedHostArray<float>::Allocate(executor, 1));
  RETURN_IF_ERROR(cuda::CudaStatus(
      cudaMemcpyAsync(host_loss.data(), loss.data(), sizeof(float),
                      cudaMemcpyDeviceToHost, executor.stream()),
      "cudaMemcpyAsync(FSM evaluation loss)"));
  RETURN_IF_ERROR(executor.Synchronize());
  if (!std::isfinite(host_loss[0]) || host_loss[0] < 0.0f)
    return absl::DataLossError(
        "evaluation produced a nonfinite or negative loss");
  return host_loss[0];
}

// The public step_N name appears only after every weight has been written.
// On failure, preserve the incomplete directory for inspection; it cannot be
// mistaken for a complete checkpoint by shared checkpoint readers.
absl::Status SaveCheckpoint(cuda::Executor& executor, const Layer& model,
                            const std::filesystem::path& root, int step) {
  const auto destination = root / absl::StrCat("step_", step);
  const auto staging = root / absl::StrCat(".step_", step, ".incomplete");
  std::error_code error;
  const bool exists = std::filesystem::exists(destination, error);
  if (error)
    return FileSystemError("cannot inspect", destination, error);
  if (exists)
    return absl::AlreadyExistsError(
        absl::StrCat("checkpoint exists: ", destination.string()));
  const bool created = std::filesystem::create_directory(staging, error);
  if (error)
    return FileSystemError("cannot create", staging, error);
  if (!created)
    return absl::AlreadyExistsError(
        absl::StrCat("checkpoint staging exists: ", staging.string()));
  RETURN_IF_ERROR(WriteToDirectory(executor, model, staging));
  std::filesystem::rename(staging, destination, error);
  if (error)
    return FileSystemError("cannot publish checkpoint", destination, error);
  return absl::OkStatus();
}

absl::Status TrainModel(cuda::Executor& executor,
                        const std::filesystem::path& checkpoint_root,
                        util::TeeStream& logger) {
  const FsmTokenizer tokenizer;
  Gpt2Config model_config;
  model_config.transformer_block_count = absl::GetFlag(FLAGS_layers);
  model_config.attention_heads = absl::GetFlag(FLAGS_attention_heads);
  model_config.vocabulary_size = tokenizer.vocab_size();
  // Keep Shakespeare's model/MLP widths and initialization recipe, while
  // allowing different depth and head partitioning. The projection weights
  // have the same shapes regardless of the head count, so changing heads does
  // not change the parameter count. Store 1,029 meaningful embedding rows;
  // padded logit lanes are handled by the layers.
  model_config.pad_vocabulary = false;
  RETURN_IF_ERROR(model_config.Validate());

  const int batch_size = absl::GetFlag(FLAGS_batch_size);
  const int seed = absl::GetFlag(FLAGS_seed);
  const bool answer_only = absl::GetFlag(FLAGS_answer_only);
  const std::string training_path =
      DataPath(absl::GetFlag(FLAGS_training_data),
               "finite_state_machine_training_data.txt");
  const std::string test_path = DataPath(absl::GetFlag(FLAGS_test_data),
                                         "finite_state_machine_test_data.txt");
  DataSetOptions data_options;
  data_options.batch_size = batch_size;
  data_options.context_length = model_config.context_length;
  data_options.shuffle = true;
  data_options.seed = static_cast<uint64_t>(seed);
  data_options.answer_only = answer_only;
  logger << '[' << Timestamp() << "] loading training and test examples\n";
  ASSIGN_OR_RETURN(auto training_corpus, LoadTextCorpus(training_path));
  ASSIGN_OR_RETURN(auto test_corpus, LoadTextCorpus(test_path));
  ASSIGN_OR_RETURN(auto training_data,
                   FsmDataSetIterator::Create(executor, training_corpus.text(),
                                              tokenizer, data_options));
  data_options.shuffle = false;
  ASSIGN_OR_RETURN(auto training_evaluation_data,
                   FsmDataSetIterator::Create(executor, training_corpus.text(),
                                              tokenizer, data_options));
  ASSIGN_OR_RETURN(auto test_evaluation_data,
                   FsmDataSetIterator::Create(executor, test_corpus.text(),
                                              tokenizer, data_options));

  if (training_evaluation_data->batches_per_epoch() >
          static_cast<size_t>(std::numeric_limits<int>::max()) ||
      test_evaluation_data->batches_per_epoch() >
          static_cast<size_t>(std::numeric_limits<int>::max())) {
    return absl::OutOfRangeError("evaluation batch count is too large");
  }

  const int requested_eval_samples = absl::GetFlag(FLAGS_eval_samples);
  const size_t selected_eval_samples =
      requested_eval_samples == 0
          ? training_evaluation_data->sample_count()
          : std::min(training_evaluation_data->sample_count(),
                     static_cast<size_t>(requested_eval_samples));
  const int training_eval_batches =
      static_cast<int>((selected_eval_samples + batch_size - 1) / batch_size);
  const size_t actual_eval_samples =
      std::min(training_evaluation_data->sample_count(),
               static_cast<size_t>(training_eval_batches) * batch_size);
  const int test_eval_batches = test_evaluation_data->batches_per_epoch();
  const AdamWConfig optimizer_config{
      .learning_rate = static_cast<float>(absl::GetFlag(FLAGS_learning_rate)),
      .beta1 = 0.9f,
      .beta2 = 0.95f,
      .epsilon = 1e-8f,
      .weight_decay = 0.1f};
  logger << '[' << Timestamp() << "] initializing from scratch: GPT-2"
         << " layers=" << model_config.transformer_block_count
         << " width=" << model_config.model_width
         << " heads=" << model_config.attention_heads << " head_dim="
         << model_config.model_width / model_config.attention_heads
         << " feed_forward=" << model_config.feed_forward_width
         << " context=" << model_config.context_length
         << " vocabulary=" << model_config.vocabulary_size
         << " compute=BF16 parameters=FP32 seed=" << seed << '\n'
         << "optimizer=AdamW learning_rate=" << optimizer_config.learning_rate
         << " beta1=" << optimizer_config.beta1
         << " beta2=" << optimizer_config.beta2
         << " epsilon=" << optimizer_config.epsilon
         << " weight_decay=" << optimizer_config.weight_decay << '\n'
         << "batch_size=" << batch_size
         << " steps=" << absl::GetFlag(FLAGS_steps)
         << " answer_only=" << (answer_only ? "true" : "false")
         << " checkpoint_every=" << absl::GetFlag(FLAGS_checkpoint_every)
         << " eval_every=" << absl::GetFlag(FLAGS_eval_every) << '\n'
         << "tokenizer=FSM states=000..999 letters=A..Z symbols=;,>,ERR "
            "spaces_ignored no_EOS\n"
         << "training_data=" << training_path
         << " samples=" << training_data->sample_count()
         << " batches_per_epoch=" << training_data->batches_per_epoch()
         << " max_tokens=" << training_data->max_tokens()
         << " supervised_rows=" << training_data->supervised_row_count() << '\n'
         << "test_data=" << test_path
         << " samples=" << test_evaluation_data->sample_count()
         << " batches_per_epoch=" << test_eval_batches
         << " max_tokens=" << test_evaluation_data->max_tokens()
         << " supervised_rows=" << test_evaluation_data->supervised_row_count()
         << '\n'
         << "evaluation: fixed_training_samples=" << actual_eval_samples
         << " training_batches=" << training_eval_batches
         << " test_samples=" << test_evaluation_data->sample_count()
         << " test_batches=" << test_eval_batches
         << " loss=mean_cross_entropy_per_supervised_token\n";

  ASSIGN_OR_RETURN(auto model,
                   CreateGpt2(executor, DataType::BF16, seed, model_config));
  ASSIGN_OR_RETURN(auto loss_layer,
                   CrossEntropyLossLayer::Create(
                       executor, model_config.vocabulary_size, DataType::BF16,
                       model_config.context_length));
  ASSIGN_OR_RETURN(auto optimizer,
                   Optimizer::Create(executor, *model, optimizer_config));
  logger << "unique_parameters=" << ParameterCount(*model)
         << " parameter_tensors=" << optimizer->parameter_tensor_count()
         << '\n';
  const EvaluationOptions training_evaluation_options{
      .loss_layer = *loss_layer,
      .eval_data = *training_evaluation_data,
      .batches = training_eval_batches};
  const EvaluationOptions test_evaluation_options{
      .loss_layer = *loss_layer,
      .eval_data = *test_evaluation_data,
      .batches = test_eval_batches};
  int last_evaluation_step = -1;
  const auto evaluate = [&](int step) -> absl::Status {
    if (step == last_evaluation_step)
      return absl::OkStatus();
    logger << '[' << Timestamp() << "] evaluating step=" << step << '\n';
    ASSIGN_OR_RETURN(
        const double training_loss,
        EvaluationLoss(executor, *model, training_evaluation_options));
    ASSIGN_OR_RETURN(const double test_loss,
                     EvaluationLoss(executor, *model, test_evaluation_options));
    logger << '[' << Timestamp() << "] step=" << step
           << " training_loss=" << training_loss << " test_loss=" << test_loss
           << '\n';
    last_evaluation_step = step;
    return absl::OkStatus();
  };
  int last_checkpoint_step = -1;
  const auto save_checkpoint = [&](int step) -> absl::Status {
    if (step == last_checkpoint_step)
      return absl::OkStatus();
    RETURN_IF_ERROR(SaveCheckpoint(executor, *model, checkpoint_root, step));
    last_checkpoint_step = step;
    logger << '[' << Timestamp() << "] saved checkpoint="
           << (checkpoint_root / absl::StrCat("step_", step)).string() << '\n';
    return absl::OkStatus();
  };
  RETURN_IF_ERROR(evaluate(0));
  RETURN_IF_ERROR(save_checkpoint(0));

  const int eval_every = absl::GetFlag(FLAGS_eval_every);
  const int checkpoint_every = absl::GetFlag(FLAGS_checkpoint_every);
  int last_progress_step = 0;
  auto last_progress_time = Clock::now();
  TrainingOptions training_options{.loss_layer = *loss_layer,
                                   .optimizer = *optimizer,
                                   .training_data = *training_data,
                                   .max_steps = absl::GetFlag(FLAGS_steps)};
  // Evaluate through the status-returning callback: nonfinite losses and I/O
  // failures must stop the run instead of disappearing in a void callback.
  training_options.step_callback = [&](int step) -> absl::Status {
    if (step == 1 || step % 10 == 0) {
      RETURN_IF_ERROR(executor.Synchronize());
      const auto now = Clock::now();
      const double seconds =
          std::chrono::duration<double>(now - last_progress_time).count();
      logger << '[' << Timestamp() << "] completed step=" << step
             << " seconds_per_step=" << seconds / (step - last_progress_step)
             << " (includes evaluation/checkpoint time since prior progress)\n";
      last_progress_step = step;
      last_progress_time = now;
    }
    if (step % eval_every == 0)
      RETURN_IF_ERROR(evaluate(step));
    if (checkpoint_every > 0 && step % checkpoint_every == 0)
      RETURN_IF_ERROR(save_checkpoint(step));
    return absl::OkStatus();
  };
  logger << '[' << Timestamp() << "] training started\n";
  ASSIGN_OR_RETURN(const auto result, Train(executor, *model, training_options));
  RETURN_IF_ERROR(evaluate(result.steps_completed));
  RETURN_IF_ERROR(save_checkpoint(result.steps_completed));
  logger << '[' << Timestamp()
         << "] training complete steps=" << result.steps_completed
         << " elapsed_training_seconds=" << result.elapsed_training_seconds
         << '\n';
  return absl::OkStatus();
}

absl::Status Run(cuda::Executor& executor) {
  const std::filesystem::path checkpoint_root =
      absl::GetFlag(FLAGS_checkpoint_dir);
  RETURN_IF_ERROR(PrepareCheckpointRoot(checkpoint_root));
  const std::string requested_log = absl::GetFlag(FLAGS_log_file);
  const std::filesystem::path log_path =
      requested_log.empty() ? checkpoint_root / "train.log"
                            : std::filesystem::path(requested_log);
  std::error_code error;
  const bool log_exists = std::filesystem::exists(log_path, error);
  if (error)
    return FileSystemError("cannot inspect", log_path, error);
  if (log_exists)
    return absl::AlreadyExistsError(
        absl::StrCat("refusing to overwrite log: ", log_path.string()));
  std::ofstream log_file(log_path);
  if (!log_file.is_open())
    return absl::FailedPreconditionError(
        absl::StrCat("cannot open log: ", log_path.string()));
  util::TeeStream logger(std::cout, log_file);
  logger << std::setprecision(8) << '[' << Timestamp()
         << "] FSM training log=" << log_path.string() << '\n';
  const absl::Status status = TrainModel(executor, checkpoint_root, logger);
  if (!status.ok())
    logger << '[' << Timestamp() << "] training failed: " << status << '\n';
  return status;
}

}  // namespace
}  // namespace pluto::llm::fsm

int main(int argc, char** argv) {
  const std::vector<char*> positional = absl::ParseCommandLine(argc, argv);
  if (positional.size() != 1) {
    std::cerr << "This binary accepts flags only.\n";
    return 2;
  }
  const absl::Status flags_status = pluto::llm::fsm::ValidateFlags();
  if (!flags_status.ok()) {
    std::cerr << flags_status << '\n';
    return 2;
  }
  auto executor = pluto::cuda::Executor::Create();
  if (!executor.ok()) {
    std::cerr << executor.status() << '\n';
    return 1;
  }
  const absl::Status status = pluto::llm::fsm::Run(**executor);
  // Destroy model and dataset buffers, then finish their stream-ordered frees
  // before the executor goes away, including when training returns an error.
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
