#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <string>
#include <system_error>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "src/cuda/executor.h"
#include "src/cuda/page_locked_host_array.h"
#include "src/dataset/qwen_tokenizer.h"
#include "src/llm/badam_optimizer.h"
#include "src/llm/checkpoint.h"
#include "src/llm/layers/cross_entropy_loss.h"
#include "src/llm/qwen/checkpoint.h"
#include "src/llm/qwen/embedding_algebra.h"
#include "src/llm/qwen/model.h"
#include "src/llm/qwen/qwen_cli.h"
#include "src/llm/qwen/training_model.h"
#include "src/util/status_macros.h"

ABSL_FLAG(
    std::string, mode, "",
    "Required execution mode: infer_model, train_model, or embedding_algebra");
ABSL_FLAG(std::string, expression, "",
          "Embedding-algebra expression; omit for an interactive prompt");
ABSL_FLAG(int, top_n, 3,
          "Number of embedding-algebra nearest matches; must be positive");
ABSL_FLAG(std::string, checkpoint, "",
          "Downloaded Qwen3.8-27B-FP8 HF directory");
ABSL_FLAG(std::string, prompt, "What is the capital of France? Answer briefly.",
          "Text prompt");
ABSL_FLAG(int, max_new_tokens, 32, "Maximum greedy continuation tokens");
ABSL_FLAG(int, context_length, 512,
          "Maximum total prompt plus generated tokens");
ABSL_FLAG(bool, raw_prompt, false, "Do not apply the model's chat template");
ABSL_FLAG(bool, thinking, false, "Enable the chat template's thinking mode");

ABSL_FLAG(
    std::string, text,
    "The capital of France is Paris. The capital of Greece is Athens.",
    "Raw training text; this small training driver repeats its first sequence");
ABSL_FLAG(int, sequence_length, 8,
          "Training input tokens, 1..128; text must include one extra target");
ABSL_FLAG(int, batch_size, 1, "Samples per batch; currently must be one");
ABSL_FLAG(int, steps, 2, "Number of BAdam updates");
ABSL_FLAG(int, switch_every, 50,
          "Updates per active block before resetting Adam and switching");
ABSL_FLAG(int, start_block, -1,
          "BAdam block index: 0 embedding, 1..64 decoders, 65 head; -1 starts "
          "at head");
ABSL_FLAG(double, learning_rate, 1e-5, "Fixed Adam learning rate");
ABSL_FLAG(double, max_active_gib, 0,
          "Optional cap on FP32 master/gradient/moments; 0 disables the cap");
ABSL_FLAG(std::string, resume_weights, "",
          "Optional Pluto resident-weight directory (not optimizer state)");
ABSL_FLAG(
    std::string, save_weights, "",
    "Optional NEW Pluto resident-weight directory to write after training");

namespace {

using pluto::llm::qwen::CommandLineOptions;

// Keep the command intentionally small: native tokenization, cached decoder
// steps, and greedy selection. No Python runtime or external serving engine.
absl::Status RunInference(const CommandLineOptions& options) {
  ASSIGN_OR_RETURN(auto tokenizer,
                   pluto::tokenizer::QwenTokenizer::Load(options.checkpoint));
  std::string prompt = options.prompt;
  if (!options.raw_prompt)
    prompt =
        pluto::tokenizer::QwenTokenizer::ChatPrompt(prompt, options.thinking);
  ASSIGN_OR_RETURN(auto tokens, tokenizer->Encode(prompt));
  if (tokens.empty() || tokens.size() + size_t(options.max_new_tokens) >
                            size_t(options.context_length))
    return absl::InvalidArgumentError(
        "prompt plus requested continuation exceeds --context_length");
  ASSIGN_OR_RETURN(auto executor, pluto::cuda::Executor::Create());
  pluto::llm::qwen::InferenceOptions model_options;
  model_options.context_length = options.context_length;
  model_options.load_progress = [](int loaded, int total) {
    if (loaded % 8 == 0 || loaded == total)
      std::cerr << "Loaded decoder blocks " << loaded << '/' << total << '\n';
  };
  auto start = std::chrono::steady_clock::now();
  ASSIGN_OR_RETURN(auto model,
                   pluto::llm::qwen::Model::Load(*executor, options.checkpoint,
                                                 model_options));
  if (tokenizer->vocab_size() > model->config().vocab_size)
    return absl::InvalidArgumentError("tokenizer exceeds model vocabulary");
  std::cerr << "GPU weight storage: " << model->weight_bytes()
            << " bytes; prompt: " << tokens.size() << " tokens\n";
  for (int token : tokens)
    RETURN_IF_ERROR(model->Step(token));
  ASSIGN_OR_RETURN(auto scores,
                   pluto::cuda::PageLockedHostArray<float>::Allocate(
                       *executor, model->config().vocab_size));
  std::vector<int> continuation;
  bool eos = false;
  for (int step = 0; step < options.max_new_tokens; ++step) {
    ASSIGN_OR_RETURN(auto logits, model->Logits());
    RETURN_IF_ERROR(pluto::cuda::CudaStatus(
        cudaMemcpyAsync(scores.data(), logits.data(), scores.size_bytes(),
                        cudaMemcpyDeviceToHost, executor->stream()),
        "copy Qwen logits"));
    RETURN_IF_ERROR(executor->Synchronize());
    for (float score : scores)
      if (!std::isfinite(score))
        return absl::DataLossError("non-finite Qwen logits");
    // Padded head rows have no token spelling and cannot be generated.
    int next = static_cast<int>(
        std::max_element(scores.begin(),
                         scores.begin() + tokenizer->vocab_size()) -
        scores.begin());
    if (next == tokenizer->eos_token_id() ||
        next == model->config().eos_token_id) {
      eos = true;
      break;
    }
    // Concatenate token bytes before printing, since byte-BPE tokens can end
    // inside a Unicode scalar. This also keeps diagnostic output on stderr.
    continuation.push_back(next);
    if (step + 1 < options.max_new_tokens)
      RETURN_IF_ERROR(model->Step(next));
  }
  ASSIGN_OR_RETURN(auto text, tokenizer->Decode(continuation));
  std::cout << text << '\n';
  double seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
          .count();
  std::cerr << "Generated " << continuation.size() << " tokens; "
            << (eos ? "EOS" : "token limit") << "; load + inference " << seconds
            << " seconds\n";
  return absl::OkStatus();
}

// Transfer through pinned memory, on the same executor as every model buffer.
// Host storage remains alive until the training/evaluation synchronization.
absl::Status Copy(pluto::cuda::Executor& executor, void* to, const void* from,
                  size_t bytes, cudaMemcpyKind direction) {
  return pluto::cuda::CudaStatus(
      cudaMemcpyAsync(to, from, bytes, direction, executor.stream()),
      "Qwen training transfer");
}

// Evaluate every supervised next-token loss and reject non-finite values.
// This uses the standard Pluto terminal cross entropy, averaged over tokens.
absl::StatusOr<double> MeanLoss(pluto::cuda::Executor& executor,
                                const pluto::llm::Buffer& loss, int sequence) {
  ASSIGN_OR_RETURN(auto host, pluto::cuda::PageLockedHostArray<float>::Allocate(
                                  executor, sequence));
  RETURN_IF_ERROR(Copy(executor, host.data(), loss.data(), host.size_bytes(),
                       cudaMemcpyDeviceToHost));
  RETURN_IF_ERROR(executor.Synchronize());
  double sum = 0;
  for (float x : host) {
    if (!std::isfinite(x))
      return absl::DataLossError("non-finite Qwen training loss");
    sum += x;
  }
  return sum / sequence;
}

// Load once and repeatedly optimize a short batch, switching whole parameter
// blocks. No optimizer tensors are allocated for frozen blocks.
absl::Status RunTraining(const CommandLineOptions& options) {
  using namespace pluto::llm;
  const int sequence = options.sequence_length;
  const int steps = options.steps;
  const double rate = options.learning_rate;
  const double cap = options.max_active_gib;
  const auto save = options.save_weights;
  if (!save.empty()) {
    std::error_code error;
    const bool exists = std::filesystem::exists(save, error);
    if (error)
      return absl::InternalError("cannot inspect save_weights: " +
                                 error.message());
    if (exists)
      return absl::AlreadyExistsError("save_weights must be a new directory");
  }
  ASSIGN_OR_RETURN(const auto architecture,
                   qwen::LoadConfig(options.checkpoint));
  if (options.start_block > architecture.num_hidden_layers + 1)
    return absl::InvalidArgumentError("start_block exceeds model block count");
  ASSIGN_OR_RETURN(auto tokenizer,
                   pluto::tokenizer::QwenTokenizer::Load(options.checkpoint));
  ASSIGN_OR_RETURN(auto tokens, tokenizer->Encode(options.text));
  if (tokens.size() < static_cast<size_t>(sequence) + 1)
    return absl::InvalidArgumentError(
        "training text needs sequence_length+1 tokens");
  ASSIGN_OR_RETURN(auto executor, pluto::cuda::Executor::Create());
  qwen::TrainingModelOptions model_options;
  model_options.sequence_length = sequence;
  model_options.load_progress = [](int done, int total) {
    if (done % 8 == 0 || done == total)
      std::cout << "Loaded training decoder blocks " << done << '/' << total
                << '\n'
                << std::flush;
  };
  ASSIGN_OR_RETURN(
      auto model,
      qwen::TrainingModel::Load(*executor, options.checkpoint, model_options));
  if (!options.resume_weights.empty())
    RETURN_IF_ERROR(
        ReadFromDirectory(*executor, *model, options.resume_weights, false));
  BAdamConfig config;
  config.adam.learning_rate = rate;
  config.adam.beta2 = 0.999f;
  config.adam.weight_decay = 0;
  config.switch_every = options.switch_every;
  config.start_block = options.start_block;
  config.max_active_bytes = static_cast<size_t>(cap * 1024 * 1024 * 1024);
  ASSIGN_OR_RETURN(
      auto optimizer,
      BAdamOptimizer::Create(*executor, model->parameter_blocks(), config));
  ASSIGN_OR_RETURN(auto loss, CrossEntropyLossLayer::Create(
                                  *executor, model->config().vocab_size,
                                  DataType::BF16, sequence));
  ASSIGN_OR_RETURN(
      auto host_input,
      pluto::cuda::PageLockedHostArray<int32_t>::Allocate(*executor, sequence));
  ASSIGN_OR_RETURN(
      auto host_target,
      pluto::cuda::PageLockedHostArray<int32_t>::Allocate(*executor, sequence));
  for (int i = 0; i < sequence; ++i) {
    host_input[i] = tokens[i];
    host_target[i] = tokens[i + 1];
  }
  ASSIGN_OR_RETURN(auto inputs,
                   Buffer::Allocate(*executor, host_input.size_bytes()));
  ASSIGN_OR_RETURN(auto targets,
                   Buffer::Allocate(*executor, host_target.size_bytes()));
  RETURN_IF_ERROR(Copy(*executor, inputs.data(), host_input.data(),
                       host_input.size_bytes(), cudaMemcpyHostToDevice));
  RETURN_IF_ERROR(Copy(*executor, targets.data(), host_target.data(),
                       host_target.size_bytes(), cudaMemcpyHostToDevice));
  std::cout << "resident_weight_bytes=" << model->resident_weight_bytes()
            << " parameter_blocks=" << model->parameter_blocks().size()
            << " batch_size=1 sequence_length=" << sequence << '\n'
            << std::flush;
  const auto start = std::chrono::steady_clock::now();
  for (int step = 0; step < steps; ++step) {
    RETURN_IF_ERROR(optimizer->ZeroGrad());
    RETURN_IF_ERROR(model->SetBackwardStart(optimizer->active_block()));
    ASSIGN_OR_RETURN(auto forward, model->fwd(*executor, {inputs}));
    ASSIGN_OR_RETURN(auto objective,
                     loss->fwd(*executor, {forward.outputs[0], targets}));
    ASSIGN_OR_RETURN(const double before,
                     MeanLoss(*executor, objective.outputs[0], sequence));
    ASSIGN_OR_RETURN(auto dy,
                     loss->bwd(*executor, {}, std::move(objective.state)));
    ASSIGN_OR_RETURN(auto unused,
                     model->bwd(*executor, dy, std::move(forward.state)));
    (void)unused;
    RETURN_IF_ERROR(optimizer->ApplyStep());
    RETURN_IF_ERROR(executor->Synchronize());
    size_t free = 0, total = 0;
    RETURN_IF_ERROR(pluto::cuda::CudaStatus(cudaMemGetInfo(&free, &total),
                                            "query training GPU memory"));
    std::cout << "step=" << optimizer->step()
              << " block=" << optimizer->active_block()
              << " name=" << optimizer->blocks()[optimizer->active_block()].name
              << " block_step=" << optimizer->block_step()
              << " mean_ce_before_update=" << before
              << " active_state_bytes=" << optimizer->active_state_bytes()
              << " device_used_bytes=" << total - free << " elapsed_seconds="
              << std::chrono::duration<double>(
                     std::chrono::steady_clock::now() - start)
                     .count()
              << '\n'
              << std::flush;
  }
  ASSIGN_OR_RETURN(auto final, model->fwd(*executor, {inputs}));
  ASSIGN_OR_RETURN(auto final_loss,
                   loss->fwd(*executor, {final.outputs[0], targets}));
  ASSIGN_OR_RETURN(const double after,
                   MeanLoss(*executor, final_loss.outputs[0], sequence));
  std::cout << "final_mean_ce=" << after
            << " completed_updates=" << optimizer->step() << '\n';
  if (!save.empty())
    RETURN_IF_ERROR(WriteToDirectory(*executor, *model, save));
  return absl::OkStatus();
}

// Capture actual defaults and explicit presence together. In particular,
// an explicitly default-valued flag still participates in mode validation.
template <class T>
void ReadFlag(const absl::Flag<T>& flag, T& value,
              std::vector<absl::string_view>& explicitly_set) {
  value = absl::GetFlag(flag);
  if (flag.IsSpecifiedOnCommandLine())
    explicitly_set.push_back(flag.Name());
}

// Validate all mode-specific options before accessing files or creating an
// executor, then run just the requested path with its immutable snapshot.
absl::Status Run() {
  CommandLineOptions options;
  std::vector<absl::string_view> explicitly_set;
  ReadFlag(FLAGS_mode, options.mode, explicitly_set);
  ReadFlag(FLAGS_checkpoint, options.checkpoint, explicitly_set);
  ReadFlag(FLAGS_prompt, options.prompt, explicitly_set);
  ReadFlag(FLAGS_max_new_tokens, options.max_new_tokens, explicitly_set);
  ReadFlag(FLAGS_context_length, options.context_length, explicitly_set);
  ReadFlag(FLAGS_raw_prompt, options.raw_prompt, explicitly_set);
  ReadFlag(FLAGS_thinking, options.thinking, explicitly_set);
  ReadFlag(FLAGS_text, options.text, explicitly_set);
  ReadFlag(FLAGS_sequence_length, options.sequence_length, explicitly_set);
  ReadFlag(FLAGS_batch_size, options.batch_size, explicitly_set);
  ReadFlag(FLAGS_steps, options.steps, explicitly_set);
  ReadFlag(FLAGS_switch_every, options.switch_every, explicitly_set);
  ReadFlag(FLAGS_start_block, options.start_block, explicitly_set);
  ReadFlag(FLAGS_learning_rate, options.learning_rate, explicitly_set);
  ReadFlag(FLAGS_max_active_gib, options.max_active_gib, explicitly_set);
  ReadFlag(FLAGS_resume_weights, options.resume_weights, explicitly_set);
  ReadFlag(FLAGS_save_weights, options.save_weights, explicitly_set);
  ReadFlag(FLAGS_expression, options.expression, explicitly_set);
  ReadFlag(FLAGS_top_n, options.top_n, explicitly_set);
  ASSIGN_OR_RETURN(auto mode, pluto::llm::qwen::ParseAndValidateRunMode(
                                  options, explicitly_set));
  switch (mode) {
    case pluto::llm::qwen::Mode::kInferModel:
      return RunInference(options);
    case pluto::llm::qwen::Mode::kTrainModel:
      return RunTraining(options);
    case pluto::llm::qwen::Mode::kEmbeddingAlgebra:
      return pluto::llm::qwen::RunEmbeddingAlgebra(
          options.checkpoint, options.expression, options.top_n);
  }
  return absl::InternalError("invalid run mode");
}

}  // namespace

int main(int argc, char** argv) {
  if (absl::ParseCommandLine(argc, argv).size() != 1) {
    std::cerr << absl::InvalidArgumentError(
                     "positional arguments are not supported; use --prompt "
                     "(infer_model), --text (train_model), or --expression "
                     "(embedding_algebra) for input")
              << '\n';
    return 1;
  }
  const auto status = Run();
  if (!status.ok()) {
    std::cerr << status << '\n';
    return 1;
  }
  return 0;
}
