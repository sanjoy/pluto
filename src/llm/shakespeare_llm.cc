#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
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
#include "src/common/status_macros.h"
#include "src/gpu/buffer.h"
#include "src/llm/layer.h"
#include "src/llm/layers/attention.h"
#include "src/llm/layers/combinators.h"
#include "src/llm/layers/cross_entropy_loss.h"
#include "src/llm/layers/embedding.h"
#include "src/llm/layers/fully_connected.h"
#include "src/llm/layers/gelu.h"
#include "src/llm/layers/norm.h"
#include "src/tokenization/plain_text_tokenizer.h"

ABSL_FLAG(std::string, corpus, "",
          "Shakespeare corpus path; defaults to the Bazel testdata runfile");
ABSL_FLAG(int, steps, 1200, "Number of stochastic-gradient training steps");
ABSL_FLAG(double, learning_rate, 0.05, "SGD learning rate");
ABSL_FLAG(double, target_loss, 2.8,
          "Fail unless held-out average next-byte loss is at most this value");
ABSL_FLAG(int, eval_batches, 32,
          "Number of fixed batches used for loss evaluation");
ABSL_FLAG(int, seed, 17, "Deterministic training and sampling seed");
ABSL_FLAG(bool, interactive, true,
          "Read prompts after training; disabled by the Bazel test");
ABSL_FLAG(int, generation_tokens, 300,
          "Bytes generated after each prompt");
ABSL_FLAG(double, temperature, 0.8, "Sampling temperature");
ABSL_FLAG(int, batch_size, 256,
          "Token rows per training batch; must be a multiple of 16");
ABSL_FLAG(int, model_width, 256,
          "Transformer hidden width; must be a multiple of 16");
ABSL_FLAG(int, context_length, 16,
          "Tokens per training sequence; must divide batch_size");
ABSL_FLAG(int, attention_heads, 4,
          "Attention heads; each head dimension must be a multiple of 16");

namespace pluto::llm {
namespace {

using tokenization::PlainTextTokenizer;

constexpr int kTransformerBlockCount = 12;

// All data/model dimensions live in this binary-level configuration. Layers
// retain only the dimensions inherent to their own weights and infer the batch
// row count from their input buffers.
struct ModelConfig {
  int batch_size;
  int model_width;
  int vocabulary_size;
  int context_length;
  int attention_heads;

  int sequence_batch_size() const { return batch_size / context_length; }

  absl::Status Validate() const {
    if (batch_size <= 0 || batch_size % 16 != 0) {
      return absl::InvalidArgumentError(
          "batch_size must be a positive multiple of 16");
    }
    if (context_length <= 0 || batch_size % context_length != 0) {
      return absl::InvalidArgumentError(
          "context_length must be positive and divide batch_size");
    }
    if (model_width <= 0 || model_width % 16 != 0) {
      return absl::InvalidArgumentError(
          "model_width must be a positive multiple of 16");
    }
    if (vocabulary_size <= 0 || vocabulary_size % 16 != 0) {
      return absl::InvalidArgumentError(
          "vocabulary_size must be a positive multiple of 16");
    }
    if (attention_heads <= 0 || model_width % attention_heads != 0 ||
        (model_width / attention_heads) % 16 != 0) {
      return absl::InvalidArgumentError(
          "attention_heads must divide model_width and produce a head "
          "dimension that is a multiple of 16");
    }
    return absl::OkStatus();
  }
};

absl::Status CudaStatus(cudaError_t error, const char* operation) {
  if (error == cudaSuccess) return absl::OkStatus();
  return absl::InternalError(
      absl::StrCat(operation, " failed: ", cudaGetErrorName(error), ": ",
                   cudaGetErrorString(error)));
}

// Builds one pre-norm, GPT-2-style transformer block. Given an input x, the
// block applies these two residual branches in sequence:
//
//   attention: x <- x + W_o CausalAttention(W_qkv LayerNorm(x))
//   MLP:       x <- x + W_2 GELU(W_1 LayerNorm(x))
//
// CausalAttention is the fused FlashAttention layer. Its single projected
// activation supplies Q, K, and V, so this scaled-down model shares their
// projection rather than creating three matrices. The MLP is also
// width-preserving instead of expanding to GPT-2's usual four-times width.
// Both residual-output projections are scaled at initialization to keep
// activation variance stable across the model's 12 blocks.
absl::StatusOr<std::unique_ptr<ComposedLayer>> CreateTransformerBlock(
    const ModelConfig& config, DataType output_type, float learning_rate,
    cudaStream_t stream) {
  // GPT-2 scales residual projections by 1/sqrt(2 * layer_count) so variance
  // does not grow with depth.
  const float residual_projection_scale =
      1.0f / std::sqrt(2.0f * kTransformerBlockCount);
  ComposedLayerBuilder attention_builder;
  RETURN_IF_ERROR(attention_builder.add(LayerNormLayer::Create(
      config.model_width, 1e-5f, output_type, stream)));
  RETURN_IF_ERROR(attention_builder.add(FullyConnectedLayer::Create(
      config.model_width, output_type, learning_rate, stream)));
  auto* qkv_projection =
      static_cast<FullyConnectedLayer*>(attention_builder.back());
  RETURN_IF_ERROR(qkv_projection->InitializeIdentity());
  RETURN_IF_ERROR(attention_builder.add(AttentionLayer::Create(
      config.context_length, config.attention_heads, config.model_width,
      output_type, stream)));
  RETURN_IF_ERROR(attention_builder.add(FullyConnectedLayer::Create(
      config.model_width, output_type, learning_rate, stream)));
  auto* attention_projection =
      static_cast<FullyConnectedLayer*>(attention_builder.back());
  RETURN_IF_ERROR(
      attention_projection->InitializeIdentity(residual_projection_scale));

  ComposedLayerBuilder mlp_builder;
  RETURN_IF_ERROR(mlp_builder.add(LayerNormLayer::Create(
      config.model_width, 1e-5f, output_type, stream)));
  RETURN_IF_ERROR(mlp_builder.add(FullyConnectedLayer::Create(
      config.model_width, output_type, learning_rate, stream)));
  auto* mlp_input =
      static_cast<FullyConnectedLayer*>(mlp_builder.back());
  RETURN_IF_ERROR(mlp_input->InitializeIdentity());
  RETURN_IF_ERROR(mlp_builder.add(GeluLayer::Create(output_type, stream)));
  RETURN_IF_ERROR(mlp_builder.add(FullyConnectedLayer::Create(
      config.model_width, output_type, learning_rate, stream)));
  auto* mlp_output =
      static_cast<FullyConnectedLayer*>(mlp_builder.back());
  RETURN_IF_ERROR(
      mlp_output->InitializeIdentity(residual_projection_scale));

  ASSIGN_OR_RETURN(auto attention, attention_builder.create());
  ASSIGN_OR_RETURN(auto mlp, mlp_builder.create());
  ComposedLayerBuilder block_builder;
  RETURN_IF_ERROR(block_builder.add(
      std::make_unique<ResidualLayer>(std::move(attention))));
  RETURN_IF_ERROR(
      block_builder.add(std::make_unique<ResidualLayer>(std::move(mlp))));
  return block_builder.create();
}

// Constructs the model here because its topology is specific to this training
// binary. Generic layer implementations remain in //src/llm:layers.
absl::StatusOr<std::unique_ptr<ComposedLayer>> CreateShakespeareLlm(
    const ModelConfig& config, DataType output_type, float learning_rate,
    cudaStream_t stream) {
  ComposedLayerBuilder builder;
  RETURN_IF_ERROR(builder.add(EmbeddingLookupLayer::Create(
      config.vocabulary_size, config.model_width, output_type, learning_rate,
      stream)));
  auto* embedding = static_cast<EmbeddingLookupLayer*>(builder.back());
  // GPT-2 uses small initial embeddings. A scaled identity is deterministic,
  // breaks the tied E * E^T zero-gradient symmetry, and keeps the final
  // layer-normalized logits in a stable range.
  RETURN_IF_ERROR(embedding->InitializeIdentity(0.02f));

  RETURN_IF_ERROR(builder.add(PositionEmbeddingLayer::Create(
      config.context_length, config.model_width, output_type, learning_rate,
      stream)));

  // Each block is independently parameterized and participates directly in
  // the model's sequential composition. The compact variant retains GPT-2's
  // 12-block depth, pre-norm residual topology, and causal attention, but
  // shares Q/K/V within each block and keeps the MLP width-preserving.
  for (int index = 0; index < kTransformerBlockCount; ++index) {
    RETURN_IF_ERROR(builder.add(
        CreateTransformerBlock(config, output_type, learning_rate, stream)));
  }

  RETURN_IF_ERROR(builder.add(LayerNormLayer::Create(
      config.model_width, 1e-5f, output_type, stream)));

  RETURN_IF_ERROR(
      builder.add(LanguageModelingHeadLayer::Create(embedding)));

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

  // Bazel tests expose the fixture beneath TEST_SRCDIR/TEST_WORKSPACE. `bazel
  // run` normally starts in the workspace, so the relative fallback handles
  // both direct execution and interactive development.
  if (const char* test_srcdir = std::getenv("TEST_SRCDIR")) {
    const char* workspace = std::getenv("TEST_WORKSPACE");
    if (workspace != nullptr) {
      auto corpus = ReadFile(absl::StrCat(test_srcdir, "/", workspace,
                                          "/testdata/shakespeare.txt"));
      if (corpus.ok()) return corpus;
    }
  }
  return ReadFile("testdata/shakespeare.txt");
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
    const std::vector<uint32_t>& corpus_tokens, int eval_batches,
    const Buffer& token_buffer, const Buffer& target_buffer,
    cudaStream_t stream) {
  if (eval_batches <= 0) {
    return absl::InvalidArgumentError("eval_batches must be positive");
  }
  std::vector<int> tokens(config.batch_size);
  std::vector<int> targets(config.batch_size);
  std::vector<float> losses(config.batch_size);
  double total = 0.0;
  const size_t sequence_start_count =
      corpus_tokens.size() - config.context_length;

  for (int batch = 0; batch < eval_batches; ++batch) {
    for (int sequence = 0; sequence < config.sequence_batch_size(); ++sequence) {
      const size_t ordinal =
          static_cast<size_t>(batch) * config.sequence_batch_size() + sequence;
      const size_t start =
          (ordinal * sequence_start_count) /
          (static_cast<size_t>(eval_batches) * config.sequence_batch_size());
      for (int position = 0; position < config.context_length; ++position) {
        const int row = sequence * config.context_length + position;
        tokens[row] = static_cast<int>(corpus_tokens[start + position]);
        targets[row] =
            static_cast<int>(corpus_tokens[start + position + 1]);
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

// Runs stochastic next-token training and updates the model in place.
//
// `config` defines the batch, sequence, vocabulary, and model dimensions.
// `model` maps token IDs to vocabulary logits; its backward pass applies the
// parameter updates using the learning rate supplied when it was constructed.
// `loss_layer` computes one cross-entropy value per token and seeds the logits
// gradient. `corpus_tokens` is the complete host-resident tokenized corpus.
// `steps` is the number of randomly sampled batches, while `seed` makes their
// starting offsets reproducible.
//
// `token_buffer` and `target_buffer` are reusable GPU staging allocations,
// each containing `config.batch_size` native `int` values. They are interpreted
// as flattened [config.sequence_batch_size(), config.context_length] arrays:
// token_buffer holds each input sequence and target_buffer holds the same
// sequence shifted forward by one token. Reusing these buffers avoids a device
// allocation on every step. They must belong to `stream`, which orders the
// host-to-device copies, forward/backward kernels, and final synchronization.
// Buffer is reference-counted, so Train borrows these handles without taking
// ownership of the underlying allocations.
absl::Status Train(const ModelConfig& config, Layer& model,
                   CrossEntropyLossLayer& loss_layer,
                   const std::vector<uint32_t>& corpus_tokens, int steps,
                   int seed, const Buffer& token_buffer,
                   const Buffer& target_buffer, cudaStream_t stream) {
  if (steps < 0) {
    return absl::InvalidArgumentError("steps must be non-negative");
  }
  std::mt19937 random(seed);
  std::uniform_int_distribution<size_t> sequence_start(
      0, corpus_tokens.size() - config.context_length - 1);
  std::vector<int> tokens(config.batch_size);
  std::vector<int> targets(config.batch_size);
  std::vector<float> losses(config.batch_size);

  for (int step = 0; step < steps; ++step) {
    for (int sequence = 0; sequence < config.sequence_batch_size(); ++sequence) {
      const size_t start = sequence_start(random);
      for (int position = 0; position < config.context_length; ++position) {
        const int row = sequence * config.context_length + position;
        tokens[row] = static_cast<int>(corpus_tokens[start + position]);
        targets[row] =
            static_cast<int>(corpus_tokens[start + position + 1]);
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

    if ((step + 1) % 200 == 0 || step + 1 == steps) {
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
      std::cout << "step " << step + 1 << "/" << steps
                << ", batch loss: " << mean << '\n';
    }
  }
  return CudaStatus(cudaStreamSynchronize(stream),
                    "cudaStreamSynchronize(after training)");
}

absl::StatusOr<std::vector<float>> Predict(
    const ModelConfig& config, Layer& model,
    const std::vector<uint32_t>& context,
    const Buffer& token_buffer, cudaStream_t stream) {
  if (context.empty()) {
    return absl::InvalidArgumentError("prediction context must not be empty");
  }
  const size_t context_size =
      std::min(context.size(), static_cast<size_t>(config.context_length));
  const size_t context_start = context.size() - context_size;
  std::vector<int> repeated_context(config.batch_size);
  for (int sequence = 0; sequence < config.sequence_batch_size(); ++sequence) {
    for (size_t position = 0; position < context_size; ++position) {
      repeated_context[sequence * config.context_length + position] =
          static_cast<int>(context[context_start + position]);
    }
    // These rows are causally invisible to the selected output row. Filling
    // them still gives every kernel valid token IDs.
    for (size_t position = context_size;
         position < static_cast<size_t>(config.context_length); ++position) {
      repeated_context[sequence * config.context_length + position] =
          static_cast<int>(context.back());
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
  std::vector<float> host_logits(config.vocabulary_size);
  const size_t output_row = context_size - 1;
  const auto* selected_logits =
      static_cast<const float*>(logits.data()) +
      output_row * config.vocabulary_size;
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
    const ModelConfig& config, Layer& model,
    const PlainTextTokenizer& tokenizer, const std::vector<bool>& observed,
    std::string prompt,
    int generation_tokens, double temperature, std::mt19937* random,
    const Buffer& token_buffer, cudaStream_t stream) {
  if (generation_tokens < 0 || temperature <= 0.0) {
    return absl::InvalidArgumentError(
        "generation_tokens must be non-negative and temperature positive");
  }
  if (prompt.empty()) prompt = "\n";
  std::vector<uint32_t> prompt_tokens = tokenizer.Encode(prompt);
  std::vector<uint32_t> context = prompt_tokens;
  std::vector<uint32_t> generated;
  generated.reserve(generation_tokens);

  for (int index = 0; index < generation_tokens; ++index) {
    ASSIGN_OR_RETURN(auto logits,
                     Predict(config, model, context, token_buffer, stream));
    const float maximum = *std::max_element(logits.begin(), logits.end());
    std::vector<double> probabilities(config.vocabulary_size);
    for (int token = 0; token < config.vocabulary_size; ++token) {
      if (observed[token]) {
        probabilities[token] =
            std::exp((logits[token] - maximum) / temperature);
      }
    }
    std::discrete_distribution<int> sample(probabilities.begin(),
                                            probabilities.end());
    const uint32_t next = static_cast<uint32_t>(sample(*random));
    context.push_back(next);
    generated.push_back(next);
  }
  return tokenizer.Decode(generated);
}

absl::Status Run(cudaStream_t stream) {
  ASSIGN_OR_RETURN(auto corpus, LoadCorpus());
  const PlainTextTokenizer tokenizer;
  const ModelConfig config{
      .batch_size = absl::GetFlag(FLAGS_batch_size),
      .model_width = absl::GetFlag(FLAGS_model_width),
      .vocabulary_size = static_cast<int>(tokenizer.vocab_size()),
      .context_length = absl::GetFlag(FLAGS_context_length),
      .attention_heads = absl::GetFlag(FLAGS_attention_heads),
  };
  RETURN_IF_ERROR(config.Validate());
  const std::vector<uint32_t> corpus_tokens = tokenizer.Encode(corpus);
  if (corpus_tokens.size() <= static_cast<size_t>(config.context_length)) {
    return absl::InvalidArgumentError(
        "the training corpus must be longer than the model context");
  }
  std::vector<bool> observed(config.vocabulary_size);
  for (uint32_t token : corpus_tokens) observed[token] = true;

  ASSIGN_OR_RETURN(
      auto model,
      CreateShakespeareLlm(
          config, DataType::FP16,
          static_cast<float>(absl::GetFlag(FLAGS_learning_rate)), stream));
  ASSIGN_OR_RETURN(
      auto loss_layer,
      CrossEntropyLossLayer::Create(config.vocabulary_size, DataType::FP16,
                                    stream));
  ASSIGN_OR_RETURN(
      auto token_buffer,
      Buffer::Allocate(config.batch_size * sizeof(int), stream));
  ASSIGN_OR_RETURN(
      auto target_buffer,
      Buffer::Allocate(config.batch_size * sizeof(int), stream));

  const int eval_batches = absl::GetFlag(FLAGS_eval_batches);
  ASSIGN_OR_RETURN(double initial_loss,
                   Evaluate(config, *model, *loss_layer, corpus_tokens,
                            eval_batches, token_buffer, target_buffer, stream));
  std::cout << "corpus bytes: " << corpus_tokens.size()
            << ", vocabulary: " << tokenizer.vocab_size()
            << ", initial loss: " << initial_loss << '\n';

  RETURN_IF_ERROR(Train(config, *model, *loss_layer, corpus_tokens,
                        absl::GetFlag(FLAGS_steps), absl::GetFlag(FLAGS_seed),
                        token_buffer, target_buffer, stream));
  ASSIGN_OR_RETURN(double final_loss,
                   Evaluate(config, *model, *loss_layer, corpus_tokens,
                            eval_batches, token_buffer, target_buffer, stream));
  std::cout << "final evaluation loss: " << final_loss << '\n';
  const double target_loss = absl::GetFlag(FLAGS_target_loss);
  if (final_loss > target_loss) {
    return absl::FailedPreconditionError(
        absl::StrCat("model did not reach target loss ", target_loss,
                     "; final loss was ", final_loss));
  }
  if (final_loss >= initial_loss) {
    return absl::FailedPreconditionError(
        "training did not reduce evaluation loss");
  }

  std::mt19937 random(absl::GetFlag(FLAGS_seed) + 1);
  ASSIGN_OR_RETURN(
      auto sample,
      Generate(config, *model, tokenizer, observed, "To be",
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
        Generate(config, *model, tokenizer, observed, prompt,
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
  // Run() owns every Buffer, so its return queues all stream-ordered frees
  // before the final synchronization and stream destruction below.
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
