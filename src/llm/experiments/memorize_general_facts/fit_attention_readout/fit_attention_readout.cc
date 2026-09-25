// Fits only one residual MLP to frozen native post-attention activations.
// Neither the original checkpoint nor its prefix/head parameters are updated.
#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "src/cuda/page_locked_host_array.h"
#include "src/dataset/compact_vocabulary.h"
#include "src/dataset/dataset.h"
#include "src/dataset/gpt2_tokenizer.h"
#include "src/dataset/padded_line_dataset.h"
#include "src/llm/adamw_optimizer.h"
#include "src/llm/checkpoint.h"
#include "src/llm/experiments/memorize_general_facts/fit_attention_readout/margin_loss.h"
#include "src/llm/experiments/memorize_general_facts/fit_attention_readout/readout.h"
#include "src/llm/extract_top1_ids.h"
#include "src/llm/layer_hooks.h"
#include "src/llm/layers/cross_entropy_loss.h"
#include "src/util/status_macros.h"

ABSL_FLAG(std::string, checkpoint, "", "Frozen source checkpoint.");
ABSL_FLAG(std::string, tokenizer, "", "Original GPT-2 tokenizer directory.");
ABSL_FLAG(std::string, corpus, "", "Corpus, one fact per line.");
ABSL_FLAG(std::string, output, "",
          "New directory for the best MLP checkpoint.");
ABSL_FLAG(std::string, readout_checkpoint, "",
          "Optional six-tensor branch checkpoint to initialize from.");
ABSL_FLAG(int, block, 1, "Zero-based frozen attention boundary.");
ABSL_FLAG(int, layers, 4, "Source checkpoint block count.");
ABSL_FLAG(int, model_width, 10, "Source hidden width.");
ABSL_FLAG(int, feed_forward_width, 20, "Source/replacement MLP width.");
ABSL_FLAG(int, context_length, 27, "Source context length.");
ABSL_FLAG(int, batch_size, 32, "Facts per update; must divide corpus size.");
ABSL_FLAG(
    int, steps, 20000,
    "Maximum optimizer updates; zero only evaluates the initial weights.");
ABSL_FLAG(int, eval_every, 200, "Updates between complete corpus evaluations.");
ABSL_FLAG(double, seconds, 600,
          "Wall-clock cap on fitting, excluding capture.");
ABSL_FLAG(float, learning_rate, 0.001f, "Initial Adam learning rate.");
ABSL_FLAG(float, final_rate_ratio, 0.1f, "Cosine decay end/start ratio.");
ABSL_FLAG(float, margin, 0.1f, "Desired target-minus-competitor logit margin.");
ABSL_FLAG(std::string, objective, "squared_margin",
          "Training loss: squared_margin or cross_entropy. Evaluations always "
          "report margin loss and top-1.");
ABSL_FLAG(int, seed, 0, "Minibatch shuffle seed.");
ABSL_FLAG(bool, fresh_branch, false,
          "With random_init >= 0, also reset pre-LN and biases; copy no "
          "trained branch parameters.");
ABSL_FLAG(
    int, random_init, -1,
    "-1: checkpoint MLP; otherwise matrix init seed (biases/LN retained).");

namespace pluto::llm::fit_attention_readout {
namespace {

constexpr int kPromptTokens = 5;

absl::Status Copy(cuda::Executor& executor, void* to, const void* from,
                  size_t bytes, cudaMemcpyKind kind) {
  return cuda::CudaStatus(
      cudaMemcpyAsync(to, from, bytes, kind, executor.stream()),
      "copy experiment buffer");
}

// Native BF16 snapshots, retaining padding so fitting uses the same sequence
// shape and layer kernels as inference. Only suffix-plus-EOS rows have targets.
struct Cache {
  Buffer activations;
  Buffer targets;
  cuda::PageLockedHostArray<int> host_targets;
  int sample_count;
  int scored_count;
};

// The original graph is evaluated for convenience, but only this captured
// pre-MLP buffer is consumed by the replacement. Later source blocks never
// feed the replacement; all prefix computations remain frozen and causal.
absl::StatusOr<Buffer> Capture(cuda::Executor& executor, const Layer& source,
                               int block, const Buffer& tokens,
                               std::optional<FwdResult>& full) {
  std::vector<std::string> scopes;
  std::optional<Buffer> result;
  const std::string block_name = absl::StrCat("transformer_block_", block);
  LayerHooks hooks;
  hooks.enter_combinator = [&](auto&, auto name) {
    scopes.emplace_back(name);
    return absl::OkStatus();
  };
  hooks.exit_combinator = [&](auto&) {
    if (scopes.empty())
      return absl::DataLossError("unbalanced capture scope");
    scopes.pop_back();
    return absl::OkStatus();
  };
  hooks.activation_hook = [&](auto&, auto name, auto, auto buffers) {
    if (!result && name == "ResidualLayer" && scopes.size() == 2 &&
        scopes[0] == "gpt2" && scopes[1] == block_name) {
      if (buffers.size() != 1)
        return absl::DataLossError("unexpected attention output count");
      result = buffers[0];
    }
    return absl::OkStatus();
  };
  ASSIGN_OR_RETURN(auto forward, source.fwd(executor, {&tokens, 1}, &hooks));
  forward.state = BackwardState{};
  full = std::move(forward);
  if (!result || !scopes.empty())
    return absl::DataLossError("missing post-attention capture");
  return *result;
}

absl::StatusOr<Cache> CaptureCorpus(cuda::Executor& executor,
                                    const Layer& source,
                                    const Gpt2Config& config, int block,
                                    const PaddedLineDataSetIterator& data,
                                    int eos) {
  const int samples = data.sample_count();
  const int sequence = config.context_length;
  const size_t row_bytes = config.model_width * sizeof(uint16_t);
  ASSIGN_OR_RETURN(auto inputs,
                   cuda::PageLockedHostArray<int>::Allocate(executor, sequence));
  ASSIGN_OR_RETURN(auto predictions,
                   cuda::PageLockedHostArray<int>::Allocate(executor, sequence));
  ASSIGN_OR_RETURN(auto targets, cuda::PageLockedHostArray<int>::Allocate(
                                     executor, samples * sequence));
  std::fill(targets.begin(), targets.end(), -1);
  ASSIGN_OR_RETURN(auto device_input,
                   Buffer::Allocate(executor, inputs.size_bytes()));
  ASSIGN_OR_RETURN(auto device_mask,
                   Buffer::Allocate(executor, inputs.size_bytes()));
  RETURN_IF_ERROR(cuda::CudaStatus(
      cudaMemsetAsync(device_mask.data(), 0, device_mask.size_bytes(),
                      executor.stream()),
      "clear mask"));
  ASSIGN_OR_RETURN(auto activations,
                   Buffer::Allocate(executor, samples * sequence * row_bytes));
  ASSIGN_OR_RETURN(auto device_targets,
                   Buffer::Allocate(executor, targets.size_bytes()));
  int scored = 0;
  for (int sample = 0; sample < samples; ++sample) {
    const auto tokens = data.sample_tokens(sample);
    std::fill(inputs.begin(), inputs.end(), eos);
    std::copy(tokens.begin(), tokens.end(), inputs.begin());
    RETURN_IF_ERROR(Copy(executor, device_input.data(), inputs.data(),
                         inputs.size_bytes(), cudaMemcpyHostToDevice));
    std::optional<FwdResult> full;
    ASSIGN_OR_RETURN(auto captured,
                     Capture(executor, source, block, device_input, full));
    if (captured.size_bytes() != sequence * row_bytes)
      return absl::DataLossError("unexpected activation shape");
    RETURN_IF_ERROR(Copy(
        executor,
        static_cast<char*>(activations.data()) + sample * sequence * row_bytes,
        captured.data(), captured.size_bytes(), cudaMemcpyDeviceToDevice));
    ASSIGN_OR_RETURN(auto ids,
                     ExtractTop1Ids(executor, full->outputs[0], device_mask,
                                    config.vocabulary_size));
    RETURN_IF_ERROR(Copy(executor, predictions.data(), ids.data(),
                         predictions.size_bytes(), cudaMemcpyDeviceToHost));
    RETURN_IF_ERROR(
        executor.Synchronize());  // Host input is reused next sample.
    for (size_t row = kPromptTokens - 1; row < tokens.size(); ++row) {
      const int target = row + 1 < tokens.size() ? tokens[row + 1] : eos;
      if (predictions[row] != target)
        return absl::FailedPreconditionError(
            "source checkpoint is not fully memorized");
      targets[sample * sequence + row] = target;
      ++scored;
    }
  }
  RETURN_IF_ERROR(Copy(executor, device_targets.data(), targets.data(),
                       targets.size_bytes(), cudaMemcpyHostToDevice));
  return Cache{std::move(activations), std::move(device_targets),
               std::move(targets), samples, scored};
}

// A fixed-size reusable GPU minibatch. Loading is D2D; host labels are
// consulted only to count scored rows, so padding does not change the loss
// normalization.
struct Batch {
  Buffer x;
  Buffer targets;
  int normalizer = 0;
};

absl::Status LoadBatch(cuda::Executor& executor, const Cache& cache,
                       const Gpt2Config& config, absl::Span<const int> indices,
                       Batch& batch) {
  const int sequence = config.context_length;
  const size_t x_bytes = sequence * config.model_width * sizeof(uint16_t);
  const size_t target_bytes = sequence * sizeof(int);
  batch.normalizer = 0;
  for (size_t i = 0; i < indices.size(); ++i) {
    const int index = indices[i];
    RETURN_IF_ERROR(
        Copy(executor, static_cast<char*>(batch.x.data()) + i * x_bytes,
             static_cast<char*>(cache.activations.data()) + index * x_bytes,
             x_bytes, cudaMemcpyDeviceToDevice));
    RETURN_IF_ERROR(Copy(
        executor, static_cast<char*>(batch.targets.data()) + i * target_bytes,
        static_cast<char*>(cache.targets.data()) + index * target_bytes,
        target_bytes, cudaMemcpyDeviceToDevice));
    for (int row = 0; row < sequence; ++row)
      batch.normalizer += cache.host_targets[index * sequence + row] >= 0;
  }
  if (batch.normalizer == 0)
    return absl::DataLossError("batch has no scored rows");
  return absl::OkStatus();
}

struct Metrics {
  int wrong = 0;    // Scored positions whose top-1 ID is not the target.
  int failed = 0;   // Facts with at least one wrong teacher-forced position.
  double loss = 0;  // Mean squared margin violation over scored rows.
  float min_margin = std::numeric_limits<float>::infinity();
};

absl::StatusOr<Metrics> Evaluate(cuda::Executor& executor, const Layer& tail,
                                 const Cache& cache, const Gpt2Config& config,
                                 int batch_size, float margin, Batch& batch) {
  const int rows = batch_size * config.context_length;
  ASSIGN_OR_RETURN(auto losses,
                   cuda::PageLockedHostArray<float>::Allocate(executor, rows));
  ASSIGN_OR_RETURN(auto margins,
                   cuda::PageLockedHostArray<float>::Allocate(executor, rows));
  ASSIGN_OR_RETURN(auto ids,
                   cuda::PageLockedHostArray<int>::Allocate(executor, rows));
  std::vector<int> indices(batch_size);
  Metrics result;
  for (int start = 0; start < cache.sample_count; start += batch_size) {
    std::iota(indices.begin(), indices.end(), start);
    RETURN_IF_ERROR(LoadBatch(executor, cache, config, indices, batch));
    ASSIGN_OR_RETURN(auto forward, tail.fwd(executor, {&batch.x, 1}));
    ASSIGN_OR_RETURN(
        auto loss,
        SquaredMarginLoss(executor, forward.outputs[0], batch.targets,
                          config.vocabulary_size, margin, batch.normalizer));
    RETURN_IF_ERROR(Copy(executor, losses.data(), loss.losses.data(),
                         losses.size_bytes(), cudaMemcpyDeviceToHost));
    RETURN_IF_ERROR(Copy(executor, margins.data(), loss.margins.data(),
                         margins.size_bytes(), cudaMemcpyDeviceToHost));
    RETURN_IF_ERROR(Copy(executor, ids.data(), loss.predicted_ids.data(),
                         ids.size_bytes(), cudaMemcpyDeviceToHost));
    RETURN_IF_ERROR(executor.Synchronize());
    for (int sample = 0; sample < batch_size; ++sample) {
      bool failed = false;
      for (int row = 0; row < config.context_length; ++row) {
        const int local = sample * config.context_length + row;
        const int target =
            cache.host_targets[(start + sample) * config.context_length + row];
        if (target < 0)
          continue;  // Prompt and right-padding rows are not completion
                     // targets.
        if (!std::isfinite(losses[local]) || !std::isfinite(margins[local]) ||
            ids[local] < 0)
          return absl::DataLossError("nonfinite fit output");
        result.loss += losses[local];
        result.min_margin = std::min(result.min_margin, margins[local]);
        result.wrong += ids[local] != target;
        failed |= ids[local] != target;
      }
      result.failed += failed;
    }
  }
  result.loss /= cache.scored_count;
  return result;
}

// Evaluate actual generated prefixes, not the fixed cached training vectors.
// Stop each fact at its first mistake: its exact completion is then disproved.
absl::StatusOr<int> VerifyCompletions(cuda::Executor& executor,
                                      const Layer& source, const Layer& tail,
                                      const Gpt2Config& config, int block,
                                      const PaddedLineDataSetIterator& data,
                                      int eos) {
  ASSIGN_OR_RETURN(auto input, cuda::PageLockedHostArray<int>::Allocate(
                                   executor, config.context_length));
  ASSIGN_OR_RETURN(auto ids, cuda::PageLockedHostArray<int>::Allocate(
                                 executor, config.context_length));
  ASSIGN_OR_RETURN(auto device_input,
                   Buffer::Allocate(executor, input.size_bytes()));
  ASSIGN_OR_RETURN(auto mask, Buffer::Allocate(executor, input.size_bytes()));
  RETURN_IF_ERROR(cuda::CudaStatus(
      cudaMemsetAsync(mask.data(), 0, mask.size_bytes(), executor.stream()),
      "clear verification mask"));
  int correct = 0;
  for (size_t sample = 0; sample < data.sample_count(); ++sample) {
    auto tokens = data.sample_tokens(sample);
    std::fill(input.begin(), input.end(), eos);
    std::copy_n(tokens.begin(), kPromptTokens, input.begin());
    bool failed = false;
    for (size_t row = kPromptTokens - 1; row < tokens.size(); ++row) {
      RETURN_IF_ERROR(Copy(executor, device_input.data(), input.data(),
                           input.size_bytes(), cudaMemcpyHostToDevice));
      std::optional<FwdResult> unused;
      ASSIGN_OR_RETURN(auto x,
                       Capture(executor, source, block, device_input, unused));
      ASSIGN_OR_RETURN(auto forward, tail.fwd(executor, {&x, 1}));
      ASSIGN_OR_RETURN(auto predicted,
                       ExtractTop1Ids(executor, forward.outputs[0], mask,
                                      config.vocabulary_size));
      RETURN_IF_ERROR(Copy(executor, ids.data(), predicted.data(),
                           ids.size_bytes(), cudaMemcpyDeviceToHost));
      RETURN_IF_ERROR(executor.Synchronize());
      const int expected = row + 1 < tokens.size() ? tokens[row + 1] : eos;
      if (ids[row] != expected) {
        failed = true;
        break;
      }
      if (row + 1 < tokens.size())
        input[row + 1] = ids[row];
    }
    correct += !failed;
  }
  return correct;
}

// Snapshot only the frozen copy's weights to catch accidental optimizer scope
// expansion. All six trainable tensors precede final LN and tied embedding.
absl::StatusOr<std::vector<uint8_t>> FrozenWeights(cuda::Executor& executor,
                                                   const Readout& readout) {
  std::vector<uint8_t> result;
  for (const auto& weight : readout.model->weights().subspan(6)) {
    ASSIGN_OR_RETURN(auto host, cuda::PageLockedHostArray<uint8_t>::Allocate(
                                    executor, weight.size_bytes()));
    RETURN_IF_ERROR(Copy(executor, host.data(), weight.data(),
                         host.size_bytes(), cudaMemcpyDeviceToHost));
    RETURN_IF_ERROR(executor.Synchronize());
    result.insert(result.end(), host.begin(), host.end());
  }
  return result;
}

absl::Status Run() {
  const auto output = absl::GetFlag(FLAGS_output);
  const auto checkpoint = absl::GetFlag(FLAGS_checkpoint);
  const int batch_size = absl::GetFlag(FLAGS_batch_size);
  const int steps = absl::GetFlag(FLAGS_steps);
  const int eval_every = absl::GetFlag(FLAGS_eval_every);
  const float margin = absl::GetFlag(FLAGS_margin);
  const float lr = absl::GetFlag(FLAGS_learning_rate);
  const float ratio = absl::GetFlag(FLAGS_final_rate_ratio);
  const double seconds = absl::GetFlag(FLAGS_seconds);
  const std::string objective = absl::GetFlag(FLAGS_objective);
  if (objective != "squared_margin" && objective != "cross_entropy")
    return absl::InvalidArgumentError("unknown training objective");
  if (!absl::GetFlag(FLAGS_readout_checkpoint).empty() &&
      (absl::GetFlag(FLAGS_random_init) >= 0 ||
       absl::GetFlag(FLAGS_fresh_branch)))
    return absl::InvalidArgumentError(
        "readout_checkpoint cannot be combined with random initialization");
  if (output.empty() || checkpoint.empty() || batch_size <= 0 || steps < 0 ||
      eval_every <= 0 || !(seconds > 0) || !std::isfinite(seconds) ||
      !(margin > 0) || !std::isfinite(margin) || !(ratio > 0 && ratio <= 1) ||
      !(lr > 0) || !std::isfinite(lr))
    return absl::InvalidArgumentError("invalid paths or fitting options");
  if (std::filesystem::exists(output))
    return absl::AlreadyExistsError(
        "output directory must be new; source is never overwritten");
  ASSIGN_OR_RETURN(auto original, tokenizer::Gpt2Tokenizer::Load(
                                      absl::GetFlag(FLAGS_tokenizer)));
  ASSIGN_OR_RETURN(auto vocabulary,
                   tokenizer::CompactVocabularyTokenizer::LoadFromFile(
                       *original, std::filesystem::path(checkpoint) /
                                      "compact_vocabulary.tsv"));
  const Gpt2Config config{
      .transformer_block_count = absl::GetFlag(FLAGS_layers),
      .model_width = absl::GetFlag(FLAGS_model_width),
      .attention_heads = 1,
      .feed_forward_width = absl::GetFlag(FLAGS_feed_forward_width),
      .vocabulary_size = vocabulary->vocab_size(),
      .pad_vocabulary = false,
      .context_length = absl::GetFlag(FLAGS_context_length)};
  RETURN_IF_ERROR(config.Validate());
  ASSIGN_OR_RETURN(auto executor, cuda::Executor::Create());
  ASSIGN_OR_RETURN(auto corpus, LoadTextCorpus(absl::GetFlag(FLAGS_corpus)));
  ASSIGN_OR_RETURN(auto data, PaddedLineDataSetIterator::Create(
                                  *executor, corpus.text(), *vocabulary,
                                  {.batch_size = 1,
                                   .context_length = config.context_length,
                                   .prompt_tokens = kPromptTokens,
                                   .eos_token = vocabulary->eos_token_id()}));
  if (data->sample_count() % batch_size != 0)
    return absl::InvalidArgumentError("batch size must divide fact count");
  ASSIGN_OR_RETURN(auto source,
                   CreateGpt2(*executor, DataType::BF16, 0, config));
  RETURN_IF_ERROR(ReadFromDirectory(*executor, *source, checkpoint, false));
  ASSIGN_OR_RETURN(
      auto readout,
      CreateReadout(*executor, *source, config, absl::GetFlag(FLAGS_block),
                    absl::GetFlag(FLAGS_random_init),
                    absl::GetFlag(FLAGS_fresh_branch)));
  if (!absl::GetFlag(FLAGS_readout_checkpoint).empty())
    RETURN_IF_ERROR(ReadFromDirectory(*executor, *readout.trainable,
                                      absl::GetFlag(FLAGS_readout_checkpoint),
                                      false));
  ASSIGN_OR_RETURN(const auto frozen, FrozenWeights(*executor, readout));
  size_t parameters = 0;
  for (const auto& weight : readout.trainable->weights())
    parameters += weight.size_bytes() / sizeof(float);
  std::cout << "Capturing block " << absl::GetFlag(FLAGS_block)
            << "; trainable parameters=" << parameters << std::endl;
  ASSIGN_OR_RETURN(auto cache, CaptureCorpus(*executor, *source, config,
                                             absl::GetFlag(FLAGS_block), *data,
                                             vocabulary->eos_token_id()));
  std::cout << "Source verified: " << cache.sample_count << " facts; "
            << cache.scored_count << " scored targets.\n";
  const int rows = batch_size * config.context_length;
  ASSIGN_OR_RETURN(
      auto x,
      Buffer::Allocate(*executor, rows * config.model_width * sizeof(uint16_t)));
  ASSIGN_OR_RETURN(auto targets,
                   Buffer::Allocate(*executor, rows * sizeof(int)));
  Batch batch{std::move(x), std::move(targets)};
  ASSIGN_OR_RETURN(auto optimizer,
                   AdamWOptimizer::Create(*executor, *readout.trainable,
                                          {.learning_rate = lr,
                                           .beta1 = 0.9f,
                                           .beta2 = 0.999f,
                                           .epsilon = 1e-8f,
                                           .weight_decay = 0}));
  RETURN_IF_ERROR(optimizer->ZeroGrad());
  ASSIGN_OR_RETURN(
      auto cross_entropy,
      CrossEntropyLossLayer::Create(*executor, config.vocabulary_size,
                                    DataType::BF16, config.context_length));
  std::cout << "objective=" << objective << " learning_rate=" << lr
            << " batch_size=" << batch_size << " margin=" << margin
            << std::endl;
  const auto begin = std::chrono::steady_clock::now();
  auto elapsed = [&] {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                         begin)
        .count();
  };
  Metrics best;
  best.wrong = std::numeric_limits<int>::max();
  int best_step = 0;
  int last_eval = -1;
  auto evaluate = [&](int step) -> absl::StatusOr<bool> {
    ASSIGN_OR_RETURN(auto metrics, Evaluate(*executor, *readout.model, cache,
                                            config, batch_size, margin, batch));
    last_eval = step;
    std::cout << absl::StrFormat(
                     "step=%d seconds=%.2f wrong=%d/%d failed_facts=%d/%d "
                     "margin_loss=%.8f min_margin=%.6f\n",
                     step, elapsed(), metrics.wrong, cache.scored_count,
                     metrics.failed, cache.sample_count, metrics.loss,
                     metrics.min_margin)
              << std::flush;
    if (metrics.wrong < best.wrong ||
        (metrics.wrong == best.wrong && metrics.loss < best.loss)) {
      best = metrics;
      best_step = step;
      RETURN_IF_ERROR(
          WriteToDirectory(*executor, *readout.trainable,
                           std::filesystem::path(output) / "best_mlp"));
    }
    return metrics.wrong == 0;
  };
  ASSIGN_OR_RETURN(bool done, evaluate(0));
  std::vector<int> order(cache.sample_count);
  std::iota(order.begin(), order.end(), 0);
  std::mt19937 random(absl::GetFlag(FLAGS_seed));
  int cursor = cache.sample_count;
  int completed = 0;
  for (int step = 1; step <= steps && !done && elapsed() < seconds; ++step) {
    if (cursor == cache.sample_count) {
      std::shuffle(order.begin(), order.end(), random);
      cursor = 0;
    }
    RETURN_IF_ERROR(LoadBatch(
        *executor, cache, config,
        absl::MakeConstSpan(order).subspan(cursor, batch_size), batch));
    cursor += batch_size;
    const float progress = static_cast<float>(step - 1) / steps;
    const float current_lr =
        lr * (ratio + (1 - ratio) * 0.5f *
                          (1 + std::cos(3.14159265358979323846f * progress)));
    RETURN_IF_ERROR(optimizer->SetLearningRate(current_lr));
    ASSIGN_OR_RETURN(auto forward, readout.model->fwd(*executor, {&batch.x, 1}));
    BufferVec logit_gradients;
    if (objective == "squared_margin") {
      ASSIGN_OR_RETURN(
          auto loss,
          SquaredMarginLoss(*executor, forward.outputs[0], batch.targets,
                            config.vocabulary_size, margin, batch.normalizer));
      logit_gradients.push_back(std::move(loss.gradients));
    } else {
      const BufferVec loss_inputs{forward.outputs[0], batch.targets};
      ASSIGN_OR_RETURN(auto loss, cross_entropy->fwd(*executor, loss_inputs));
      ASSIGN_OR_RETURN(logit_gradients,
                       cross_entropy->bwd(*executor, {}, std::move(loss.state)));
    }
    // Fixed head/LN gradients may be computed to propagate into the branch,
    // but they never enter the optimizer. Clear them to avoid accumulation.
    for (const auto& gradient : readout.model->gradients().subspan(6))
      RETURN_IF_ERROR(cuda::CudaStatus(
          cudaMemsetAsync(gradient.data(), 0, gradient.size_bytes(),
                          executor->stream()),
          "clear frozen gradients"));
    RETURN_IF_ERROR(
        readout.model->bwd(*executor, logit_gradients, std::move(forward.state))
            .status());
    RETURN_IF_ERROR(optimizer->ApplyStep());
    completed = step;
    if (step % eval_every == 0) {
      ASSIGN_OR_RETURN(done, evaluate(step));
    }
    // Bound enqueued work and enforce the wall-clock cap without allowing an
    // unbounded async backlog. This also exposes CUDA failures promptly.
    if (step % 32 == 0)
      RETURN_IF_ERROR(executor->Synchronize());
  }
  if (last_eval != completed) {
    ASSIGN_OR_RETURN(done, evaluate(completed));
  }
  RETURN_IF_ERROR(ReadFromDirectory(*executor, *readout.trainable,
                                    std::filesystem::path(output) / "best_mlp",
                                    false));
  ASSIGN_OR_RETURN(auto after, FrozenWeights(*executor, readout));
  if (frozen != after)
    return absl::InternalError("frozen final LN or embedding changed");
  ASSIGN_OR_RETURN(auto final_metrics,
                   Evaluate(*executor, *readout.model, cache, config,
                            batch_size, margin, batch));
  ASSIGN_OR_RETURN(const int correct,
                   VerifyCompletions(*executor, *source, *readout.model, config,
                                     absl::GetFlag(FLAGS_block), *data,
                                     vocabulary->eos_token_id()));
  std::cout
      << absl::StrFormat(
             "FINAL best_step=%d updates=%d seconds=%.2f wrong=%d/%d "
             "autoregressive_complete=%d/%d frozen_weights_unchanged=true\n",
             best_step, completed, elapsed(), final_metrics.wrong,
             cache.scored_count, correct, cache.sample_count)
      << std::flush;
  return absl::OkStatus();
}

}  // namespace
}  // namespace pluto::llm::fit_attention_readout

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  const auto status = pluto::llm::fit_attention_readout::Run();
  if (!status.ok()) {
    std::cerr << status << '\n';
    return 1;
  }
  return 0;
}
