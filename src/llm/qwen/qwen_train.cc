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
#include "src/cuda/executor.h"
#include "src/cuda/page_locked_host_array.h"
#include "src/dataset/qwen_tokenizer.h"
#include "src/llm/badam_optimizer.h"
#include "src/llm/checkpoint.h"
#include "src/llm/layers/cross_entropy_loss.h"
#include "src/llm/qwen/training_model.h"
#include "src/util/status_macros.h"

ABSL_FLAG(std::string, checkpoint, "", "Original Hugging Face Qwen checkpoint");
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
absl::Status Run() {
  using namespace pluto::llm;
  const int sequence = absl::GetFlag(FLAGS_sequence_length);
  const int steps = absl::GetFlag(FLAGS_steps);
  const double rate = absl::GetFlag(FLAGS_learning_rate);
  const double cap = absl::GetFlag(FLAGS_max_active_gib);
  if (absl::GetFlag(FLAGS_checkpoint).empty() || sequence <= 0 ||
      sequence > 128 || steps <= 0 || absl::GetFlag(FLAGS_batch_size) != 1 ||
      absl::GetFlag(FLAGS_switch_every) <= 0 ||
      absl::GetFlag(FLAGS_start_block) < -1 || !(rate > 0) ||
      !std::isfinite(rate) || rate > 1 || !std::isfinite(cap) || cap < 0 ||
      cap > 1024)
    return absl::InvalidArgumentError(
        "require checkpoint, batch_size=1, sequence_length in [1,128], "
        "positive steps/rate/switch_every, start_block >= -1 and valid memory "
        "cap");
  const auto save = absl::GetFlag(FLAGS_save_weights);
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
                   qwen::LoadConfig(absl::GetFlag(FLAGS_checkpoint)));
  if (absl::GetFlag(FLAGS_start_block) > architecture.num_hidden_layers + 1)
    return absl::InvalidArgumentError("start_block exceeds model block count");
  ASSIGN_OR_RETURN(auto tokenizer, pluto::tokenizer::QwenTokenizer::Load(
                                       absl::GetFlag(FLAGS_checkpoint)));
  ASSIGN_OR_RETURN(auto tokens, tokenizer->Encode(absl::GetFlag(FLAGS_text)));
  if (tokens.size() < static_cast<size_t>(sequence) + 1)
    return absl::InvalidArgumentError(
        "training text needs sequence_length+1 tokens");
  ASSIGN_OR_RETURN(auto executor, pluto::cuda::Executor::Create());
  qwen::TrainingModelOptions options;
  options.sequence_length = sequence;
  options.load_progress = [](int done, int total) {
    if (done % 8 == 0 || done == total)
      std::cout << "Loaded training decoder blocks " << done << '/' << total
                << '\n'
                << std::flush;
  };
  ASSIGN_OR_RETURN(auto model,
                   qwen::TrainingModel::Load(
                       *executor, absl::GetFlag(FLAGS_checkpoint), options));
  if (!absl::GetFlag(FLAGS_resume_weights).empty())
    RETURN_IF_ERROR(ReadFromDirectory(
        *executor, *model, absl::GetFlag(FLAGS_resume_weights), false));
  BAdamConfig config;
  config.adam.learning_rate = rate;
  config.adam.beta2 = 0.999f;
  config.adam.weight_decay = 0;
  config.switch_every = absl::GetFlag(FLAGS_switch_every);
  config.start_block = absl::GetFlag(FLAGS_start_block);
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

}  // namespace

int main(int argc, char** argv) {
  if (absl::ParseCommandLine(argc, argv).size() != 1) {
    std::cerr << "Use --text for training input\n";
    return 1;
  }
  const auto status = Run();
  if (!status.ok()) {
    std::cerr << status << '\n';
    return 1;
  }
  return 0;
}
