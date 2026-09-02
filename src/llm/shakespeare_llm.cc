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
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "src/cuda/buffer.h"
#include "src/cuda/executor.h"
#include "src/dataset/dataset.h"
#include "src/dataset/detokenizer.h"
#include "src/dataset/tokenizer.h"
#include "src/llm/checkpoint.h"
#include "src/llm/layer.h"
#include "src/llm/layers/attention.h"
#include "src/llm/layers/combinators.h"
#include "src/llm/layers/cross_entropy_loss.h"
#include "src/llm/layers/embedding.h"
#include "src/llm/layers/fully_connected.h"
#include "src/llm/layers/gelu.h"
#include "src/llm/layers/norm.h"
#include "src/llm/optimizer.h"
#include "src/llm/trainer.h"
#include "src/util/status_macros.h"
#include "src/util/tee_stream.h"

ABSL_FLAG(std::string, corpus, "",
          "Shakespeare corpus path; defaults to the Bazel testdata runfile");
ABSL_FLAG(std::string, tokenizer_dir, "",
          "GPT-2 tokenizer directory; defaults to "
          "PLUTO_GPT2_TOKENIZER_DIR");
ABSL_FLAG(std::string, resume_from, "",
          "Parent directory whose numerically latest step_N resumes training");
ABSL_FLAG(std::string, inference_from, "",
          "Exact step_N checkpoint directory to load for inference; selecting "
          "this mode disables training");
ABSL_FLAG(std::string, checkpoint_dir, "",
          "Root directory for periodic step_N checkpoint directories");
ABSL_FLAG(int, checkpoint_every, 0,
          "Write a checkpoint every N optimizer steps; zero disables writes");
ABSL_FLAG(int, steps, -1,
          "Maximum AdamW updates in this invocation; omitted runs forever");
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
ABSL_FLAG(double, temperature, 0.8, "Sampling temperature");
ABSL_FLAG(int, batch_size, 1,
          "Number of context-length sequences per training/evaluation batch");
ABSL_FLAG(std::string, log_file, "/tmp/train.log",
          "File that receives a copy of stdout; truncated at startup");

namespace pluto::llm {
namespace {

using tokenizer::Gpt2Detokenizer;
using tokenizer::Gpt2Tokenizer;

std::string CurrentTimestamp() {
  return absl::FormatTime("%Y-%m-%d %H:%M:%S UTC", absl::Now(),
                          absl::UTCTimeZone());
}

// This binary deliberately exposes no architecture flags: these constants are
// the model contract requested for Shakespeare. The vocabulary is physically
// padded to 50,272 only inside tiled output kernels; padded logits are masked
// and are never valid token IDs.
constexpr int kVocabularySize = 50'257;
constexpr int kContextLength = 1'024;
constexpr int kTransformerBlockCount = 8;
constexpr int kModelWidth = 512;
constexpr int kAttentionHeads = 8;
constexpr int kAttentionHeadDimension = 64;
constexpr int kFeedForwardWidth = 2'048;
constexpr float kLayerNormEpsilon = 1e-5f;
constexpr float kInitializationStandardDeviation = 0.02f;
constexpr int kTileSize = 16;
static_assert(kModelWidth == kAttentionHeads * kAttentionHeadDimension);
static_assert(kFeedForwardWidth == 4 * kModelWidth);

struct ModelConfig {
  int batch_size;

  int token_batch_size() const { return batch_size * kContextLength; }
  int padded_vocabulary_size() const {
    return ((kVocabularySize + kTileSize - 1) / kTileSize) * kTileSize;
  }

  absl::Status Validate() const {
    if (batch_size <= 0) {
      return absl::InvalidArgumentError("batch_size must be positive");
    }
    if (batch_size > std::numeric_limits<int>::max() / kContextLength) {
      return absl::InvalidArgumentError("batch_size is too large");
    }
    return absl::OkStatus();
  }
};

// Builds one pre-LayerNorm GPT-2 transformer block:
//
//   x = x + W_o CausalMHA(W_qkv LayerNorm(x))
//   x = x + W_2 GELU(W_1 LayerNorm(x))
//
// W_qkv maps 512 to three independent 512-wide Q/K/V tensors. CausalMHA has
// eight 64-wide heads and uses online FP32 softmax statistics. The MLP expands
// 512 -> 2048 -> 512. Dropout and attention dropout are exactly zero, so no
// dropout layers appear. Each block is independently parameterized.
absl::StatusOr<std::unique_ptr<ComposedLayer>> CreateTransformerBlock(
    cuda::Executor& executor, DataType output_type, int initialization_seed,
    int block_index) {
  const float residual_standard_deviation =
      kInitializationStandardDeviation /
      std::sqrt(2.0f * kTransformerBlockCount);
  const uint64_t seed_base =
      static_cast<uint64_t>(static_cast<uint32_t>(initialization_seed)) +
      1'000 + static_cast<uint64_t>(block_index) * 100;

  ComposedLayerBuilder attention_builder;
  RETURN_IF_ERROR(attention_builder.add(LayerNormLayer::Create(
      executor, kModelWidth, kLayerNormEpsilon, output_type)));
  RETURN_IF_ERROR(attention_builder.add(FullyConnectedLayer::Create(
      executor, kModelWidth, 3 * kModelWidth, output_type)));
  auto* qkv_projection =
      static_cast<FullyConnectedLayer*>(attention_builder.back());
  RETURN_IF_ERROR(qkv_projection->InitializeNormal(
      kInitializationStandardDeviation, seed_base + 1));
  RETURN_IF_ERROR(attention_builder.add(AttentionLayer::Create(
      executor, kContextLength, kAttentionHeads, kModelWidth, output_type)));
  RETURN_IF_ERROR(attention_builder.add(FullyConnectedLayer::Create(
      executor, kModelWidth, kModelWidth, output_type)));
  auto* attention_projection =
      static_cast<FullyConnectedLayer*>(attention_builder.back());
  RETURN_IF_ERROR(attention_projection->InitializeNormal(
      residual_standard_deviation, seed_base + 2));

  ComposedLayerBuilder mlp_builder;
  RETURN_IF_ERROR(mlp_builder.add(LayerNormLayer::Create(
      executor, kModelWidth, kLayerNormEpsilon, output_type)));
  RETURN_IF_ERROR(mlp_builder.add(FullyConnectedLayer::Create(
      executor, kModelWidth, kFeedForwardWidth, output_type)));
  auto* mlp_input = static_cast<FullyConnectedLayer*>(mlp_builder.back());
  RETURN_IF_ERROR(mlp_input->InitializeNormal(kInitializationStandardDeviation,
                                              seed_base + 3));
  RETURN_IF_ERROR(mlp_builder.add(GeluLayer::Create(executor, output_type)));
  RETURN_IF_ERROR(mlp_builder.add(FullyConnectedLayer::Create(
      executor, kFeedForwardWidth, kModelWidth, output_type)));
  auto* mlp_output = static_cast<FullyConnectedLayer*>(mlp_builder.back());
  RETURN_IF_ERROR(
      mlp_output->InitializeNormal(residual_standard_deviation, seed_base + 4));

  ASSIGN_OR_RETURN(auto attention, attention_builder.create());
  ASSIGN_OR_RETURN(auto mlp, mlp_builder.create());
  ComposedLayerBuilder block_builder;
  RETURN_IF_ERROR(
      block_builder.add(std::make_unique<ResidualLayer>(std::move(attention))));
  RETURN_IF_ERROR(
      block_builder.add(std::make_unique<ResidualLayer>(std::move(mlp))));
  return block_builder.create();
}

// Constructs the exact eight-block model described above. Token and position
// embeddings, block activations, and final normalized activations are BF16.
// Parameters are FP32 master weights, reductions/statistics remain FP32, and
// the terminal projection reuses the token embedding table.
absl::StatusOr<std::unique_ptr<ComposedLayer>> CreateShakespeareLlm(
    cuda::Executor& executor, DataType output_type, int seed) {
  ComposedLayerBuilder builder;
  RETURN_IF_ERROR(builder.add(EmbeddingLookupLayer::Create(
      executor, kVocabularySize, kModelWidth, output_type)));
  auto* embedding = static_cast<EmbeddingLookupLayer*>(builder.back());
  RETURN_IF_ERROR(embedding->InitializeNormal(kInitializationStandardDeviation,
                                              static_cast<uint64_t>(seed)));

  RETURN_IF_ERROR(builder.add(PositionEmbeddingLayer::Create(
      executor, kContextLength, kModelWidth, output_type)));
  auto* positions = static_cast<PositionEmbeddingLayer*>(builder.back());
  RETURN_IF_ERROR(positions->InitializeNormal(kInitializationStandardDeviation,
                                              static_cast<uint64_t>(seed) + 1));

  for (int index = 0; index < kTransformerBlockCount; ++index) {
    RETURN_IF_ERROR(builder.add(
        CreateTransformerBlock(executor, output_type, seed, index)));
  }

  RETURN_IF_ERROR(builder.add(LayerNormLayer::Create(
      executor, kModelWidth, kLayerNormEpsilon, output_type)));
  RETURN_IF_ERROR(builder.add(LanguageModelingHeadLayer::Create(embedding)));
  return builder.create();
}

std::string CorpusPath() {
  const std::string requested = absl::GetFlag(FLAGS_corpus);
  if (!requested.empty()) return requested;

  if (const char* test_srcdir = std::getenv("TEST_SRCDIR")) {
    if (const char* workspace = std::getenv("TEST_WORKSPACE")) {
      const std::string runfile = absl::StrCat(test_srcdir, "/", workspace,
                                               "/testdata/shakespeare.txt");
      if (std::filesystem::exists(runfile)) return runfile;
    }
  }
  return "testdata/shakespeare.txt";
}

absl::StatusOr<std::filesystem::path> TokenizerDirectory() {
  const std::string requested = absl::GetFlag(FLAGS_tokenizer_dir);
  if (!requested.empty()) return std::filesystem::path(requested);
  if (const char* environment = std::getenv("PLUTO_GPT2_TOKENIZER_DIR")) {
    return std::filesystem::path(environment);
  }
  return absl::FailedPreconditionError(
      "set --tokenizer_dir or PLUTO_GPT2_TOKENIZER_DIR to the GPT-2 "
      "tokenizer directory");
}

absl::StatusOr<std::vector<float>> Predict(cuda::Executor& executor,
                                           const ModelConfig& config,
                                           const Layer& model,
                                           const std::vector<int>& context,
                                           const Buffer& token_buffer) {
  if (&token_buffer.executor() != &executor) {
    return absl::InvalidArgumentError(
        "prediction requires its token buffer's CUDA Executor");
  }
  if (context.empty()) {
    return absl::InvalidArgumentError("prediction context must not be empty");
  }
  const size_t context_size =
      std::min(context.size(), static_cast<size_t>(kContextLength));
  const size_t context_start = context.size() - context_size;
  std::vector<int> repeated_context(config.token_batch_size());
  for (int sequence = 0; sequence < config.batch_size; ++sequence) {
    for (size_t position = 0; position < context_size; ++position) {
      repeated_context[sequence * kContextLength + position] =
          context[context_start + position];
    }
    // Later rows are causally invisible to the selected output row.
    for (size_t position = context_size;
         position < static_cast<size_t>(kContextLength); ++position) {
      repeated_context[sequence * kContextLength + position] = context.back();
    }
  }
  RETURN_IF_ERROR(cuda::CudaStatus(
      cudaMemcpyAsync(token_buffer.data(), repeated_context.data(),
                      token_buffer.size_bytes(), cudaMemcpyHostToDevice,
                      executor.stream()),
      "cudaMemcpyAsync(prompt context)"));
  Tape tape;
  BufferVec inputs = {token_buffer};
  ASSIGN_OR_RETURN(auto logits, model.fwd(executor, inputs, &tape));
  std::vector<float> host_logits(kVocabularySize);
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

absl::StatusOr<std::string> Generate(
    cuda::Executor& executor, const ModelConfig& config, const Layer& model,
    const Gpt2Tokenizer& tokenizer, const Gpt2Detokenizer& detokenizer,
    std::string prompt, int generation_tokens, double temperature,
    std::mt19937& random, const Buffer& token_buffer) {
  if (generation_tokens < 0 || !std::isfinite(temperature) ||
      temperature <= 0.0) {
    return absl::InvalidArgumentError(
        "generation_tokens must be non-negative and temperature finite and "
        "positive");
  }
  if (prompt.empty()) prompt = "\n";
  ASSIGN_OR_RETURN(std::vector<int> context, tokenizer.Encode(prompt));
  std::vector<int> generated;
  generated.reserve(generation_tokens);

  for (int index = 0; index < generation_tokens; ++index) {
    ASSIGN_OR_RETURN(auto logits,
                     Predict(executor, config, model, context, token_buffer));
    const float maximum = *std::max_element(logits.begin(), logits.end());
    std::vector<double> probabilities(kVocabularySize);
    for (int token = 0; token < kVocabularySize; ++token) {
      probabilities[token] = std::exp((logits[token] - maximum) / temperature);
    }
    std::discrete_distribution<int> sample(probabilities.begin(),
                                           probabilities.end());
    const int next = sample(random);
    context.push_back(next);
    generated.push_back(next);
  }
  return detokenizer.Decode(generated);
}

absl::Status RunTraining(cuda::Executor& executor,
                         const std::filesystem::path& resume_from) {
  const std::string log_path = absl::GetFlag(FLAGS_log_file);
  if (log_path.empty()) {
    return absl::InvalidArgumentError("log_file must not be empty");
  }
  std::ofstream log_file(log_path, std::ios::out | std::ios::trunc);
  if (!log_file.is_open()) {
    return absl::FailedPreconditionError(
        absl::StrCat("cannot open training log for writing: ", log_path));
  }
  util::TeeStream logger(std::cout, log_file);
  logger << "training log: " << log_path << '\n';
  const int checkpoint_every = absl::GetFlag(FLAGS_checkpoint_every);
  std::filesystem::path checkpoint_root = absl::GetFlag(FLAGS_checkpoint_dir);
  if (checkpoint_root.empty() && !resume_from.empty()) {
    checkpoint_root = resume_from;
  }
  if (checkpoint_every < 0) {
    return absl::InvalidArgumentError("checkpoint_every must be non-negative");
  }
  if (checkpoint_every > 0 && checkpoint_root.empty()) {
    return absl::InvalidArgumentError(
        "checkpoint_dir must not be empty when checkpoint_every is positive");
  }

  ASSIGN_OR_RETURN(auto corpus, LoadTextCorpus(CorpusPath()));
  ASSIGN_OR_RETURN(auto tokenizer_directory, TokenizerDirectory());
  ASSIGN_OR_RETURN(auto tokenizer, Gpt2Tokenizer::Load(tokenizer_directory));
  if (tokenizer->vocab_size() != kVocabularySize) {
    return absl::FailedPreconditionError(absl::StrCat(
        "the model requires the GPT-2 vocabulary of ", kVocabularySize,
        " tokens; encoder reports ", tokenizer->vocab_size()));
  }

  const ModelConfig config{.batch_size = absl::GetFlag(FLAGS_batch_size)};
  RETURN_IF_ERROR(config.Validate());
  ASSIGN_OR_RETURN(auto corpus_split,
                   SplitCorpus(corpus, absl::GetFlag(FLAGS_test_fraction)));

  ASSIGN_OR_RETURN(auto model, CreateShakespeareLlm(executor, DataType::BF16,
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
  ASSIGN_OR_RETURN(
      auto loss_layer,
      CrossEntropyLossLayer::Create(executor, kVocabularySize, DataType::BF16));
  const AdamWConfig optimizer_config{
      .learning_rate = static_cast<float>(absl::GetFlag(FLAGS_learning_rate)),
      .beta1 = static_cast<float>(absl::GetFlag(FLAGS_adam_beta1)),
      .beta2 = static_cast<float>(absl::GetFlag(FLAGS_adam_beta2)),
      .epsilon = static_cast<float>(absl::GetFlag(FLAGS_adam_epsilon)),
      .weight_decay = static_cast<float>(absl::GetFlag(FLAGS_weight_decay)),
  };
  ASSIGN_OR_RETURN(auto optimizer,
                   Optimizer::Create(executor, *model, optimizer_config));
  const InMemoryDataSetOptions training_data_options{
      .batch_size = config.token_batch_size(),
      .context_length = kContextLength,
      .order = InMemoryDataSetOrder::kRandom,
      .seed = static_cast<uint64_t>(absl::GetFlag(FLAGS_seed)),
  };
  const InMemoryDataSetOptions evaluation_data_options{
      .batch_size = config.token_batch_size(),
      .context_length = kContextLength,
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
  ASSIGN_OR_RETURN(double initial_training_loss,
                   Evaluate(executor, *model, *loss_layer,
                            *training_evaluation_data, evaluation_options));
  ASSIGN_OR_RETURN(double initial_test_loss,
                   Evaluate(executor, *model, *loss_layer,
                            *test_evaluation_data, evaluation_options));
  logger << "model: GPT-2 vocabulary=" << kVocabularySize
         << ", context=" << kContextLength
         << ", layers=" << kTransformerBlockCount << ", width=" << kModelWidth
         << ", heads=" << kAttentionHeads
         << ", head_dim=" << kAttentionHeadDimension
         << ", MLP=" << kFeedForwardWidth << ", BF16 compute\n"
         << "batch: " << config.batch_size << " sequences ("
         << config.token_batch_size() << " tokens)\n"
         << "corpus tokens: "
         << training_data->token_count() + test_evaluation_data->token_count()
         << " (training: " << training_data->token_count()
         << ", test: " << test_evaluation_data->token_count() << ")\n"
         << "starting step: " << initial_step << '\n'
         << "initial training loss: " << initial_training_loss << '\n'
         << "initial test loss: " << initial_test_loss << '\n';

  TrainingOptions training_options{
      .max_steps = absl::GetFlag(FLAGS_steps),
      .initial_step = initial_step,
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
    training_options.step_callback =
        [&executor, &logger, model_ptr = model.get(), checkpoint_every,
         checkpoint_root](int steps_completed) -> absl::Status {
      if (steps_completed % checkpoint_every != 0) {
        return absl::OkStatus();
      }
      const std::filesystem::path checkpoint =
          checkpoint_root / absl::StrCat("step_", steps_completed);
      RETURN_IF_ERROR(WriteToDirectory(executor, *model_ptr, checkpoint));
      logger << '[' << CurrentTimestamp()
             << "] wrote checkpoint: " << checkpoint.string() << '\n';
      return absl::OkStatus();
    };
  }
  ASSIGN_OR_RETURN(auto training_result,
                   Train(executor, *model, *loss_layer, *optimizer,
                         *training_data, training_options));
  ASSIGN_OR_RETURN(double final_training_loss,
                   Evaluate(executor, *model, *loss_layer,
                            *training_evaluation_data, evaluation_options));
  ASSIGN_OR_RETURN(double final_test_loss,
                   Evaluate(executor, *model, *loss_layer,
                            *test_evaluation_data, evaluation_options));
  logger << "training stopped at step: " << training_result.steps_completed
         << '\n'
         << "final training loss: " << final_training_loss << '\n'
         << "final test loss: " << final_test_loss << '\n';

  return absl::OkStatus();
}

absl::Status RunInference(cuda::Executor& executor,
                          const std::filesystem::path& checkpoint_path) {
  ASSIGN_OR_RETURN(const CheckpointInfo checkpoint,
                   InspectCheckpointDirectory(checkpoint_path));
  ASSIGN_OR_RETURN(auto tokenizer_directory, TokenizerDirectory());
  ASSIGN_OR_RETURN(auto tokenizer, Gpt2Tokenizer::Load(tokenizer_directory));
  ASSIGN_OR_RETURN(auto detokenizer,
                   Gpt2Detokenizer::Load(tokenizer_directory));
  if (tokenizer->vocab_size() != kVocabularySize ||
      detokenizer->vocab_size() != kVocabularySize) {
    return absl::FailedPreconditionError(absl::StrCat(
        "the model requires the GPT-2 vocabulary of ", kVocabularySize,
        " tokens; encoder reports ", tokenizer->vocab_size(),
        " and decoder reports ", detokenizer->vocab_size()));
  }

  const int generation_tokens = absl::GetFlag(FLAGS_generation_tokens);
  const double temperature = absl::GetFlag(FLAGS_temperature);
  if (generation_tokens < 0 || !std::isfinite(temperature) ||
      temperature <= 0.0) {
    return absl::InvalidArgumentError(
        "generation_tokens must be non-negative and temperature finite and "
        "positive");
  }

  const ModelConfig config{.batch_size = 1};
  RETURN_IF_ERROR(config.Validate());
  ASSIGN_OR_RETURN(auto model, CreateShakespeareLlm(executor, DataType::BF16,
                                                    absl::GetFlag(FLAGS_seed)));
  RETURN_IF_ERROR(ReadFromDirectory(executor, *model, checkpoint.directory));
  ASSIGN_OR_RETURN(
      auto token_buffer,
      Buffer::Allocate(executor, config.token_batch_size() * sizeof(int)));
  std::cout << "loaded checkpoint: " << checkpoint.directory.string()
            << " (step " << checkpoint.step << ")\n";

  std::mt19937 random(absl::GetFlag(FLAGS_seed) + 1);
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
    if (!std::getline(std::cin, prompt)) break;
    ASSIGN_OR_RETURN(
        auto completion,
        Generate(executor, config, *model, *tokenizer, *detokenizer, prompt,
                 generation_tokens, temperature, random, token_buffer));
    std::cout << prompt << completion << '\n';
  }
  return absl::OkStatus();
}

absl::Status Run(cuda::Executor& executor) {
  const std::filesystem::path resume_from = absl::GetFlag(FLAGS_resume_from);
  const std::filesystem::path inference_from =
      absl::GetFlag(FLAGS_inference_from);
  if (!resume_from.empty() && !inference_from.empty()) {
    return absl::InvalidArgumentError(
        "resume_from and inference_from are mutually exclusive");
  }
  if (!inference_from.empty()) {
    return RunInference(executor, inference_from);
  }
  return RunTraining(executor, resume_from);
}

}  // namespace
}  // namespace pluto::llm

int main(int argc, char** argv) {
  const std::vector<char*> positional = absl::ParseCommandLine(argc, argv);
  if (positional.size() != 1) {
    std::cerr << "This binary accepts flags only.\n";
    return 2;
  }
  auto executor = pluto::cuda::Executor::Create();
  if (!executor.ok()) {
    std::cerr << executor.status() << '\n';
    return 1;
  }
  const absl::Status status = pluto::llm::Run(**executor);
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
