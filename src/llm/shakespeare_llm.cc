#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
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
#include "src/common/status_macros.h"
#include "src/cuda/buffer.h"
#include "src/llm/layer.h"
#include "src/llm/layers/attention.h"
#include "src/llm/layers/combinators.h"
#include "src/llm/layers/cross_entropy_loss.h"
#include "src/llm/layers/embedding.h"
#include "src/llm/layers/fully_connected.h"
#include "src/llm/layers/gelu.h"
#include "src/llm/layers/norm.h"
#include "src/llm/optimizer.h"
#include "src/tokenization/detokenizer.h"
#include "src/tokenization/tokenizer.h"

ABSL_FLAG(std::string, corpus, "",
          "Shakespeare corpus path; defaults to the Bazel testdata runfile");
ABSL_FLAG(std::string, tokenizer_dir, "",
          "GPT-2 tokenizer directory; defaults to "
          "PLUTO_GPT2_TOKENIZER_DIR");
ABSL_FLAG(int, steps, 1200, "Maximum number of AdamW training steps");
ABSL_FLAG(double, learning_rate, 3e-4, "AdamW learning rate");
ABSL_FLAG(double, adam_beta1, 0.9, "AdamW first-moment decay");
ABSL_FLAG(double, adam_beta2, 0.95, "AdamW second-moment decay");
ABSL_FLAG(double, adam_epsilon, 1e-8, "AdamW numerical-stability epsilon");
ABSL_FLAG(double, weight_decay, 0.1, "Decoupled AdamW weight decay");
ABSL_FLAG(double, target_loss, 2.8,
          "Fail unless held-out average token loss is at most this value; a "
          "negative value disables this check");
ABSL_FLAG(int, eval_batches, 4,
          "Number of fixed batches used for each train/test loss evaluation");
ABSL_FLAG(double, test_fraction, 0.1,
          "Fraction of the corpus reserved as contiguous held-out test data");
ABSL_FLAG(double, train_until_loss, -1.0,
          "When nonnegative, stop once training loss reaches this value; "
          "--steps remains the hard iteration cap");
ABSL_FLAG(int, training_eval_interval, 100,
          "Steps between training-loss checks and progress reports");
ABSL_FLAG(int, seed, 17, "Deterministic initialization and sampling seed");
ABSL_FLAG(bool, interactive, true,
          "Read prompts after training; disabled by the Bazel tests");
ABSL_FLAG(int, generation_tokens, 300,
          "Tokens generated after each prompt");
ABSL_FLAG(double, temperature, 0.8, "Sampling temperature");
ABSL_FLAG(int, batch_size, 1024,
          "Token rows per batch; must be a multiple of context length 1024");

namespace pluto::llm {
namespace {

using tokenizer::Gpt2Detokenizer;
using tokenizer::Gpt2Tokenizer;

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

  int sequence_batch_size() const { return batch_size / kContextLength; }
  int padded_vocabulary_size() const {
    return ((kVocabularySize + kTileSize - 1) / kTileSize) * kTileSize;
  }

  absl::Status Validate() const {
    if (batch_size <= 0 || batch_size % kContextLength != 0) {
      return absl::InvalidArgumentError(
          "batch_size must be a positive multiple of context length 1024");
    }
    return absl::OkStatus();
  }
};

struct CorpusSplit {
  std::vector<int> training;
  std::vector<int> test;
};

// Preserves temporal order: the prefix is used for fitting and the suffix is
// held out. A contiguous split avoids leaking overlapping context windows.
absl::StatusOr<CorpusSplit> SplitCorpus(const std::vector<int>& corpus_tokens,
                                        double test_fraction) {
  if (!std::isfinite(test_fraction) || test_fraction <= 0.0 ||
      test_fraction >= 1.0) {
    return absl::InvalidArgumentError(
        "test_fraction must be finite and strictly between zero and one");
  }
  const size_t training_size = static_cast<size_t>(
      static_cast<double>(corpus_tokens.size()) * (1.0 - test_fraction));
  constexpr size_t kMinimumSplitSize = kContextLength + 1;
  if (training_size < kMinimumSplitSize ||
      corpus_tokens.size() - training_size < kMinimumSplitSize) {
    return absl::InvalidArgumentError(
        "both corpus splits must contain more than 1024 GPT-2 tokens");
  }
  return CorpusSplit{
      .training =
          std::vector<int>(corpus_tokens.begin(),
                           corpus_tokens.begin() + training_size),
      .test = std::vector<int>(corpus_tokens.begin() + training_size,
                              corpus_tokens.end()),
  };
}

struct TrainingOptions {
  int max_steps;
  int seed;
  int evaluation_batches;
  int evaluation_interval;
  // A negative value disables loss-based early stopping. Zero requests exact
  // zero according to the FP32 cross-entropy evaluation.
  double stop_loss;
};

struct TrainingResult {
  int steps_completed;
  bool reached_stop_loss;
};

absl::Status CudaStatus(cudaError_t error, const char* operation) {
  if (error == cudaSuccess) return absl::OkStatus();
  return absl::InternalError(
      absl::StrCat(operation, " failed: ", cudaGetErrorName(error), ": ",
                   cudaGetErrorString(error)));
}

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
    DataType output_type, int initialization_seed, int block_index,
    cudaStream_t stream) {
  const float residual_standard_deviation =
      kInitializationStandardDeviation /
      std::sqrt(2.0f * kTransformerBlockCount);
  const uint64_t seed_base =
      static_cast<uint64_t>(static_cast<uint32_t>(initialization_seed)) +
      1'000 + static_cast<uint64_t>(block_index) * 100;

  ComposedLayerBuilder attention_builder;
  RETURN_IF_ERROR(attention_builder.add(LayerNormLayer::Create(
      kModelWidth, kLayerNormEpsilon, output_type, stream)));
  RETURN_IF_ERROR(attention_builder.add(FullyConnectedLayer::Create(
      kModelWidth, 3 * kModelWidth, output_type, stream)));
  auto* qkv_projection =
      static_cast<FullyConnectedLayer*>(attention_builder.back());
  RETURN_IF_ERROR(qkv_projection->InitializeNormal(
      kInitializationStandardDeviation, seed_base + 1));
  RETURN_IF_ERROR(attention_builder.add(AttentionLayer::Create(
      kContextLength, kAttentionHeads, kModelWidth, output_type, stream)));
  RETURN_IF_ERROR(attention_builder.add(FullyConnectedLayer::Create(
      kModelWidth, kModelWidth, output_type, stream)));
  auto* attention_projection =
      static_cast<FullyConnectedLayer*>(attention_builder.back());
  RETURN_IF_ERROR(attention_projection->InitializeNormal(
      residual_standard_deviation, seed_base + 2));

  ComposedLayerBuilder mlp_builder;
  RETURN_IF_ERROR(mlp_builder.add(LayerNormLayer::Create(
      kModelWidth, kLayerNormEpsilon, output_type, stream)));
  RETURN_IF_ERROR(mlp_builder.add(FullyConnectedLayer::Create(
      kModelWidth, kFeedForwardWidth, output_type, stream)));
  auto* mlp_input =
      static_cast<FullyConnectedLayer*>(mlp_builder.back());
  RETURN_IF_ERROR(mlp_input->InitializeNormal(
      kInitializationStandardDeviation, seed_base + 3));
  RETURN_IF_ERROR(mlp_builder.add(GeluLayer::Create(output_type, stream)));
  RETURN_IF_ERROR(mlp_builder.add(FullyConnectedLayer::Create(
      kFeedForwardWidth, kModelWidth, output_type, stream)));
  auto* mlp_output =
      static_cast<FullyConnectedLayer*>(mlp_builder.back());
  RETURN_IF_ERROR(mlp_output->InitializeNormal(
      residual_standard_deviation, seed_base + 4));

  ASSIGN_OR_RETURN(auto attention, attention_builder.create());
  ASSIGN_OR_RETURN(auto mlp, mlp_builder.create());
  ComposedLayerBuilder block_builder;
  RETURN_IF_ERROR(block_builder.add(
      std::make_unique<ResidualLayer>(std::move(attention))));
  RETURN_IF_ERROR(
      block_builder.add(std::make_unique<ResidualLayer>(std::move(mlp))));
  return block_builder.create();
}

// Constructs the exact eight-block model described above. Token and position
// embeddings, block activations, and final normalized activations are BF16.
// Parameters are FP32 master weights, reductions/statistics remain FP32, and
// the terminal projection reuses the token embedding table.
absl::StatusOr<std::unique_ptr<ComposedLayer>> CreateShakespeareLlm(
    DataType output_type, int seed, cudaStream_t stream) {
  ComposedLayerBuilder builder;
  RETURN_IF_ERROR(builder.add(EmbeddingLookupLayer::Create(
      kVocabularySize, kModelWidth, output_type, stream)));
  auto* embedding = static_cast<EmbeddingLookupLayer*>(builder.back());
  RETURN_IF_ERROR(embedding->InitializeNormal(
      kInitializationStandardDeviation, static_cast<uint64_t>(seed)));

  RETURN_IF_ERROR(builder.add(PositionEmbeddingLayer::Create(
      kContextLength, kModelWidth, output_type, stream)));
  auto* positions = static_cast<PositionEmbeddingLayer*>(builder.back());
  RETURN_IF_ERROR(positions->InitializeNormal(
      kInitializationStandardDeviation, static_cast<uint64_t>(seed) + 1));

  for (int index = 0; index < kTransformerBlockCount; ++index) {
    RETURN_IF_ERROR(builder.add(
        CreateTransformerBlock(output_type, seed, index, stream)));
  }

  RETURN_IF_ERROR(builder.add(LayerNormLayer::Create(
      kModelWidth, kLayerNormEpsilon, output_type, stream)));
  RETURN_IF_ERROR(builder.add(LanguageModelingHeadLayer::Create(embedding)));
  return builder.create();
}

absl::StatusOr<std::string> ReadFile(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    return absl::NotFoundError(absl::StrCat("cannot open corpus: ", path));
  }
  input.seekg(0, std::ios::end);
  const std::streamoff size = input.tellg();
  if (size < 0) {
    return absl::InternalError(
        absl::StrCat("cannot determine corpus size: ", path));
  }
  std::string contents(static_cast<size_t>(size), '\0');
  input.seekg(0, std::ios::beg);
  input.read(contents.data(), size);
  if (!input) {
    return absl::InternalError(absl::StrCat("cannot read corpus: ", path));
  }
  return contents;
}

absl::StatusOr<std::string> LoadCorpus() {
  const std::string requested = absl::GetFlag(FLAGS_corpus);
  if (!requested.empty()) return ReadFile(requested);

  if (const char* test_srcdir = std::getenv("TEST_SRCDIR")) {
    if (const char* workspace = std::getenv("TEST_WORKSPACE")) {
      auto corpus = ReadFile(absl::StrCat(test_srcdir, "/", workspace,
                                          "/testdata/shakespeare.txt"));
      if (corpus.ok()) return corpus;
    }
  }
  return ReadFile("testdata/shakespeare.txt");
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

absl::Status CopyBatch(const std::vector<int>& tokens,
                       const std::vector<int>& targets,
                       const Buffer& token_buffer, const Buffer& target_buffer,
                       cudaStream_t stream) {
  RETURN_IF_ERROR(CudaStatus(
      cudaMemcpyAsync(token_buffer.data(), tokens.data(),
                      token_buffer.size_bytes(), cudaMemcpyHostToDevice,
                      stream),
      "cudaMemcpyAsync(tokens)"));
  return CudaStatus(cudaMemcpyAsync(target_buffer.data(), targets.data(),
                                    target_buffer.size_bytes(),
                                    cudaMemcpyHostToDevice, stream),
                    "cudaMemcpyAsync(targets)");
}

absl::StatusOr<double> Evaluate(
    const ModelConfig& config, Layer& model, CrossEntropyLossLayer& loss_layer,
    const std::vector<int>& corpus_tokens, int eval_batches,
    const Buffer& token_buffer, const Buffer& target_buffer,
    cudaStream_t stream) {
  if (eval_batches <= 0) {
    return absl::InvalidArgumentError("eval_batches must be positive");
  }
  std::vector<int> tokens(config.batch_size);
  std::vector<int> targets(config.batch_size);
  std::vector<float> losses(config.batch_size);
  double total = 0.0;
  const size_t sequence_start_count = corpus_tokens.size() - kContextLength;

  for (int batch = 0; batch < eval_batches; ++batch) {
    for (int sequence = 0; sequence < config.sequence_batch_size();
         ++sequence) {
      const size_t ordinal =
          static_cast<size_t>(batch) * config.sequence_batch_size() + sequence;
      const size_t start =
          (ordinal * sequence_start_count) /
          (static_cast<size_t>(eval_batches) *
           config.sequence_batch_size());
      for (int position = 0; position < kContextLength; ++position) {
        const int row = sequence * kContextLength + position;
        tokens[row] = corpus_tokens[start + position];
        targets[row] = corpus_tokens[start + position + 1];
      }
    }
    RETURN_IF_ERROR(
        CopyBatch(tokens, targets, token_buffer, target_buffer, stream));
    Tape model_tape;
    BufferVec model_inputs = {token_buffer};
    ASSIGN_OR_RETURN(auto logits, model.fwd(model_inputs, &model_tape));
    Tape loss_tape;
    BufferVec loss_inputs = {logits, target_buffer};
    ASSIGN_OR_RETURN(auto device_losses,
                     loss_layer.fwd(loss_inputs, &loss_tape));
    RETURN_IF_ERROR(CudaStatus(
        cudaMemcpyAsync(losses.data(), device_losses.data(),
                        device_losses.size_bytes(), cudaMemcpyDeviceToHost,
                        stream),
        "cudaMemcpyAsync(evaluation losses)"));
    RETURN_IF_ERROR(CudaStatus(cudaStreamSynchronize(stream),
                               "cudaStreamSynchronize(evaluation)"));
    for (float loss : losses) total += loss;
  }
  return total / (static_cast<double>(eval_batches) * config.batch_size);
}

// Runs stochastic next-token training and updates model with optimizer.
//
// config determines how many independent 1024-token sequences are packed in
// one batch. loss_layer computes an FP32 cross-entropy per row and seeds the
// mean-loss logits gradient. training_tokens is the host-resident GPT-2-token
// training split. options controls the random window sampler, hard step cap,
// and deterministic evaluations used for optional loss-based early stopping.
// initial_training_loss avoids repeating Run()'s initial evaluation.
//
// token_buffer and target_buffer are reusable GPU staging allocations with
// config.batch_size native int values. Viewed as
// [sequence_batch_size, 1024], token_buffer contains input sequences and
// target_buffer contains the same sequences shifted by one token. Both buffers
// are tied to stream; that stream orders copies, forward/backward kernels,
// AdamW updates, and synchronization. Buffer is reference-counted, so this
// function only borrows the handles and does not own their allocations.
absl::StatusOr<TrainingResult> Train(
    const ModelConfig& config, Layer& model,
    CrossEntropyLossLayer& loss_layer, AdamWOptimizer& optimizer,
    const std::vector<int>& training_tokens,
    const TrainingOptions& options, double initial_training_loss,
    const Buffer& token_buffer, const Buffer& target_buffer,
    cudaStream_t stream) {
  if (options.max_steps < 0) {
    return absl::InvalidArgumentError("max_steps must be non-negative");
  }
  if (options.evaluation_batches <= 0 || options.evaluation_interval <= 0) {
    return absl::InvalidArgumentError(
        "training evaluation counts must be positive");
  }
  if (!std::isfinite(options.stop_loss)) {
    return absl::InvalidArgumentError("train_until_loss must be finite");
  }
  if (options.stop_loss >= 0.0 &&
      initial_training_loss <= options.stop_loss) {
    std::cout << "training loss already reached " << options.stop_loss
              << "; no updates needed\n";
    return TrainingResult{.steps_completed = 0, .reached_stop_loss = true};
  }

  std::mt19937 random(options.seed);
  std::uniform_int_distribution<size_t> sequence_start(
      0, training_tokens.size() - kContextLength - 1);
  std::vector<int> tokens(config.batch_size);
  std::vector<int> targets(config.batch_size);
  std::vector<float> losses(config.batch_size);

  for (int step = 0; step < options.max_steps; ++step) {
    for (int sequence = 0; sequence < config.sequence_batch_size();
         ++sequence) {
      const size_t start = sequence_start(random);
      for (int position = 0; position < kContextLength; ++position) {
        const int row = sequence * kContextLength + position;
        tokens[row] = training_tokens[start + position];
        targets[row] = training_tokens[start + position + 1];
      }
    }
    RETURN_IF_ERROR(
        CopyBatch(tokens, targets, token_buffer, target_buffer, stream));

    Tape model_tape;
    BufferVec model_inputs = {token_buffer};
    ASSIGN_OR_RETURN(auto logits, model.fwd(model_inputs, &model_tape));
    Tape loss_tape;
    BufferVec loss_inputs = {logits, target_buffer};
    ASSIGN_OR_RETURN(auto device_losses,
                     loss_layer.fwd(loss_inputs, &loss_tape));
    ASSIGN_OR_RETURN(auto logits_gradient,
                     loss_layer.bwd({}, std::move(loss_tape)));
    auto input_gradient = model.bwd(logits_gradient, std::move(model_tape));
    if (!input_gradient.ok()) return input_gradient.status();
    RETURN_IF_ERROR(optimizer.Step());

    const bool evaluate =
        (step + 1) % options.evaluation_interval == 0 ||
        step + 1 == options.max_steps;
    if (evaluate) {
      RETURN_IF_ERROR(CudaStatus(
          cudaMemcpyAsync(losses.data(), device_losses.data(),
                          device_losses.size_bytes(), cudaMemcpyDeviceToHost,
                          stream),
          "cudaMemcpyAsync(training losses)"));
      RETURN_IF_ERROR(CudaStatus(cudaStreamSynchronize(stream),
                                 "cudaStreamSynchronize(training)"));
      double mean = 0.0;
      for (float loss : losses) mean += loss;
      mean /= config.batch_size;
      std::cout << "step " << step + 1 << "/" << options.max_steps
                << ", pre-update batch loss: " << mean << '\n';

      if (options.stop_loss >= 0.0) {
        ASSIGN_OR_RETURN(
            double training_loss,
            Evaluate(config, model, loss_layer, training_tokens,
                     options.evaluation_batches, token_buffer, target_buffer,
                     stream));
        std::cout << "training evaluation loss after step " << step + 1
                  << ": " << training_loss << '\n';
        if (training_loss <= options.stop_loss) {
          return TrainingResult{.steps_completed = step + 1,
                                .reached_stop_loss = true};
        }
      }
    }
  }
  RETURN_IF_ERROR(CudaStatus(cudaStreamSynchronize(stream),
                             "cudaStreamSynchronize(after training)"));
  return TrainingResult{.steps_completed = options.max_steps,
                        .reached_stop_loss = false};
}

absl::StatusOr<std::vector<float>> Predict(
    const ModelConfig& config, Layer& model, const std::vector<int>& context,
    const Buffer& token_buffer, cudaStream_t stream) {
  if (context.empty()) {
    return absl::InvalidArgumentError("prediction context must not be empty");
  }
  const size_t context_size =
      std::min(context.size(), static_cast<size_t>(kContextLength));
  const size_t context_start = context.size() - context_size;
  std::vector<int> repeated_context(config.batch_size);
  for (int sequence = 0; sequence < config.sequence_batch_size();
       ++sequence) {
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
  RETURN_IF_ERROR(CudaStatus(
      cudaMemcpyAsync(token_buffer.data(), repeated_context.data(),
                      token_buffer.size_bytes(), cudaMemcpyHostToDevice,
                      stream),
      "cudaMemcpyAsync(prompt context)"));
  Tape tape;
  BufferVec inputs = {token_buffer};
  ASSIGN_OR_RETURN(auto logits, model.fwd(inputs, &tape));
  std::vector<float> host_logits(kVocabularySize);
  const size_t output_row = context_size - 1;
  const auto* selected_logits =
      static_cast<const float*>(logits.data()) +
      output_row * config.padded_vocabulary_size();
  RETURN_IF_ERROR(CudaStatus(
      cudaMemcpyAsync(host_logits.data(), selected_logits,
                      host_logits.size() * sizeof(float),
                      cudaMemcpyDeviceToHost, stream),
      "cudaMemcpyAsync(prompt logits)"));
  RETURN_IF_ERROR(CudaStatus(cudaStreamSynchronize(stream),
                             "cudaStreamSynchronize(prompt)"));
  return host_logits;
}

absl::StatusOr<std::string> Generate(
    const ModelConfig& config, Layer& model, const Gpt2Tokenizer& tokenizer,
    const Gpt2Detokenizer& detokenizer, std::string prompt,
    int generation_tokens, double temperature, std::mt19937* random,
    const Buffer& token_buffer, cudaStream_t stream) {
  if (generation_tokens < 0 || temperature <= 0.0) {
    return absl::InvalidArgumentError(
        "generation_tokens must be non-negative and temperature positive");
  }
  if (prompt.empty()) prompt = "\n";
  ASSIGN_OR_RETURN(std::vector<int> context, tokenizer.Encode(prompt));
  std::vector<int> generated;
  generated.reserve(generation_tokens);

  for (int index = 0; index < generation_tokens; ++index) {
    ASSIGN_OR_RETURN(auto logits,
                     Predict(config, model, context, token_buffer, stream));
    const float maximum = *std::max_element(logits.begin(), logits.end());
    std::vector<double> probabilities(kVocabularySize);
    for (int token = 0; token < kVocabularySize; ++token) {
      probabilities[token] =
          std::exp((logits[token] - maximum) / temperature);
    }
    std::discrete_distribution<int> sample(probabilities.begin(),
                                            probabilities.end());
    const int next = sample(*random);
    context.push_back(next);
    generated.push_back(next);
  }
  return detokenizer.Decode(generated);
}

absl::Status Run(cudaStream_t stream) {
  ASSIGN_OR_RETURN(auto corpus, LoadCorpus());
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

  const ModelConfig config{.batch_size = absl::GetFlag(FLAGS_batch_size)};
  RETURN_IF_ERROR(config.Validate());
  ASSIGN_OR_RETURN(std::vector<int> corpus_tokens, tokenizer->Encode(corpus));
  ASSIGN_OR_RETURN(
      auto corpus_split,
      SplitCorpus(corpus_tokens, absl::GetFlag(FLAGS_test_fraction)));

  ASSIGN_OR_RETURN(auto model,
                   CreateShakespeareLlm(DataType::BF16,
                                        absl::GetFlag(FLAGS_seed), stream));
  ASSIGN_OR_RETURN(auto loss_layer,
                   CrossEntropyLossLayer::Create(kVocabularySize,
                                                 DataType::BF16, stream));
  const AdamWConfig optimizer_config{
      .learning_rate =
          static_cast<float>(absl::GetFlag(FLAGS_learning_rate)),
      .beta1 = static_cast<float>(absl::GetFlag(FLAGS_adam_beta1)),
      .beta2 = static_cast<float>(absl::GetFlag(FLAGS_adam_beta2)),
      .epsilon = static_cast<float>(absl::GetFlag(FLAGS_adam_epsilon)),
      .weight_decay =
          static_cast<float>(absl::GetFlag(FLAGS_weight_decay)),
  };
  ASSIGN_OR_RETURN(auto optimizer,
                   AdamWOptimizer::Create(*model, optimizer_config, stream));
  ASSIGN_OR_RETURN(
      auto token_buffer,
      Buffer::Allocate(config.batch_size * sizeof(int), stream));
  ASSIGN_OR_RETURN(
      auto target_buffer,
      Buffer::Allocate(config.batch_size * sizeof(int), stream));

  const int eval_batches = absl::GetFlag(FLAGS_eval_batches);
  ASSIGN_OR_RETURN(
      double initial_training_loss,
      Evaluate(config, *model, *loss_layer, corpus_split.training,
               eval_batches, token_buffer, target_buffer, stream));
  ASSIGN_OR_RETURN(
      double initial_test_loss,
      Evaluate(config, *model, *loss_layer, corpus_split.test, eval_batches,
               token_buffer, target_buffer, stream));
  std::cout << "model: GPT-2 vocabulary=" << kVocabularySize
            << ", context=" << kContextLength
            << ", layers=" << kTransformerBlockCount
            << ", width=" << kModelWidth << ", heads=" << kAttentionHeads
            << ", head_dim=" << kAttentionHeadDimension
            << ", MLP=" << kFeedForwardWidth << ", BF16 compute\n"
            << "corpus tokens: " << corpus_tokens.size()
            << " (training: " << corpus_split.training.size()
            << ", test: " << corpus_split.test.size() << ")\n"
            << "initial training loss: " << initial_training_loss << '\n'
            << "initial test loss: " << initial_test_loss << '\n';

  const TrainingOptions training_options{
      .max_steps = absl::GetFlag(FLAGS_steps),
      .seed = absl::GetFlag(FLAGS_seed),
      .evaluation_batches = eval_batches,
      .evaluation_interval = absl::GetFlag(FLAGS_training_eval_interval),
      .stop_loss = absl::GetFlag(FLAGS_train_until_loss),
  };
  ASSIGN_OR_RETURN(
      auto training_result,
      Train(config, *model, *loss_layer, *optimizer, corpus_split.training,
            training_options, initial_training_loss, token_buffer,
            target_buffer, stream));
  ASSIGN_OR_RETURN(
      double final_training_loss,
      Evaluate(config, *model, *loss_layer, corpus_split.training,
               eval_batches, token_buffer, target_buffer, stream));
  ASSIGN_OR_RETURN(
      double final_test_loss,
      Evaluate(config, *model, *loss_layer, corpus_split.test, eval_batches,
               token_buffer, target_buffer, stream));
  std::cout << "completed training steps: " << training_result.steps_completed
            << '\n'
            << "final training loss: " << final_training_loss << '\n'
            << "final test loss: " << final_test_loss << '\n';

  if (training_options.stop_loss >= 0.0 &&
      (!training_result.reached_stop_loss ||
       final_training_loss > training_options.stop_loss)) {
    return absl::FailedPreconditionError(absl::StrCat(
        "training loss did not reach ", training_options.stop_loss, " within ",
        training_options.max_steps, " steps; final training loss was ",
        final_training_loss));
  }
  const double target_loss = absl::GetFlag(FLAGS_target_loss);
  if (target_loss >= 0.0 && final_test_loss > target_loss) {
    return absl::FailedPreconditionError(
        absl::StrCat("model did not reach target loss ", target_loss,
                     "; final test loss was ", final_test_loss));
  }
  if (training_result.steps_completed > 0 &&
      final_training_loss >= initial_training_loss) {
    return absl::FailedPreconditionError(
        "training did not reduce training loss");
  }

  std::mt19937 random(absl::GetFlag(FLAGS_seed) + 1);
  ASSIGN_OR_RETURN(
      auto sample,
      Generate(config, *model, *tokenizer, *detokenizer, "To be",
               std::min(120, absl::GetFlag(FLAGS_generation_tokens)),
               absl::GetFlag(FLAGS_temperature), &random, token_buffer,
               stream));
  std::cout << "sample:\nTo be" << sample << "\n";

  if (!absl::GetFlag(FLAGS_interactive)) return absl::OkStatus();
  std::cout << "\nEnter a prompt (Ctrl-C or Ctrl-D to quit).\n";
  std::string prompt;
  while (true) {
    std::cout << "> " << std::flush;
    if (!std::getline(std::cin, prompt)) break;
    ASSIGN_OR_RETURN(
        auto completion,
        Generate(config, *model, *tokenizer, *detokenizer, prompt,
                 absl::GetFlag(FLAGS_generation_tokens),
                 absl::GetFlag(FLAGS_temperature), &random, token_buffer,
                 stream));
    std::cout << prompt << completion << "\n";
  }
  return absl::OkStatus();
}

}  // namespace
}  // namespace pluto::llm

int main(int argc, char** argv) {
  const std::vector<char*> positional = absl::ParseCommandLine(argc, argv);
  if (positional.size() != 1) {
    std::cerr << "This binary accepts flags only; use --corpus=PATH.\n";
    return 2;
  }
  cudaStream_t stream = nullptr;
  if (const absl::Status status = pluto::llm::CudaStatus(
          cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
          "cudaStreamCreateWithFlags");
      !status.ok()) {
    std::cerr << status << '\n';
    return 1;
  }
  const absl::Status status = pluto::llm::Run(stream);
  // Run() destroys every Buffer, queueing stream-ordered frees before this
  // synchronization and stream destruction.
  const cudaError_t sync_error = cudaStreamSynchronize(stream);
  const cudaError_t destroy_error = cudaStreamDestroy(stream);
  if (!status.ok()) {
    std::cerr << status << '\n';
    return 1;
  }
  if (sync_error != cudaSuccess || destroy_error != cudaSuccess) {
    std::cerr << "CUDA stream cleanup failed\n";
    return 1;
  }
  return 0;
}
