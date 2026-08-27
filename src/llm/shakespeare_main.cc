#include <cuda_runtime.h>

#include <algorithm>
#include <array>
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
#include "src/gpu/buffer.h"
#include "src/llm/layer.h"
#include "src/llm/layers.h"
#include "src/llm/shakespeare_llm.h"
#include "src/tokenization/plain_text_tokenizer.h"

ABSL_FLAG(std::string, corpus, "",
          "Shakespeare corpus path; defaults to the Bazel testdata runfile");
ABSL_FLAG(int, steps, 1200, "Number of stochastic-gradient training steps");
ABSL_FLAG(double, learning_rate, 6.0, "SGD learning rate");
ABSL_FLAG(double, target_loss, 3.35,
          "Fail unless held-out average next-byte loss is at most this value");
ABSL_FLAG(int, eval_batches, 32,
          "Number of fixed batches used for loss evaluation");
ABSL_FLAG(int, seed, 17, "Deterministic training and sampling seed");
ABSL_FLAG(bool, interactive, true,
          "Read prompts after training; disabled by the Bazel test");
ABSL_FLAG(int, generation_tokens, 300,
          "Bytes generated after each prompt");
ABSL_FLAG(double, temperature, 0.8, "Sampling temperature");

namespace pluto::llm {
namespace {

using tokenization::PlainTextTokenizer;

absl::Status CudaStatus(cudaError_t error, const char* operation) {
  if (error == cudaSuccess) return absl::OkStatus();
  return absl::InternalError(
      absl::StrCat(operation, " failed: ", cudaGetErrorName(error), ": ",
                   cudaGetErrorString(error)));
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

absl::Status CopyBatch(const std::array<int, kBatchSize>& tokens,
                       const std::array<int, kBatchSize>& targets,
                       const Buffer& token_buffer, const Buffer& target_buffer,
                       cudaStream_t stream) {
  if (auto status = CudaStatus(
          cudaMemcpyAsync(token_buffer.data(), tokens.data(),
                          token_buffer.size_bytes(), cudaMemcpyHostToDevice,
                          stream),
          "cudaMemcpyAsync(tokens)");
      !status.ok()) {
    return status;
  }
  return CudaStatus(cudaMemcpyAsync(target_buffer.data(), targets.data(),
                                    target_buffer.size_bytes(),
                                    cudaMemcpyHostToDevice, stream),
                    "cudaMemcpyAsync(targets)");
}

absl::StatusOr<double> Evaluate(
    Layer& model, CrossEntropyLossLayer& loss_layer,
    const std::vector<uint32_t>& corpus_tokens, int eval_batches,
    const Buffer& token_buffer, const Buffer& target_buffer,
    cudaStream_t stream) {
  if (eval_batches <= 0) {
    return absl::InvalidArgumentError("eval_batches must be positive");
  }
  std::array<int, kBatchSize> tokens{};
  std::array<int, kBatchSize> targets{};
  std::array<float, kBatchSize> losses{};
  double total = 0.0;
  const size_t pair_count = corpus_tokens.size() - 1;

  for (int batch = 0; batch < eval_batches; ++batch) {
    for (int row = 0; row < kBatchSize; ++row) {
      const size_t ordinal =
          static_cast<size_t>(batch) * kBatchSize + row;
      const size_t position =
          (ordinal * pair_count) /
          (static_cast<size_t>(eval_batches) * kBatchSize);
      tokens[row] = static_cast<int>(corpus_tokens[position]);
      targets[row] = static_cast<int>(corpus_tokens[position + 1]);
    }
    if (auto status = CopyBatch(tokens, targets, token_buffer, target_buffer,
                                stream);
        !status.ok()) {
      return status;
    }
    Tape model_tape;
    BufferVec model_inputs = {token_buffer};
    auto logits = model.fwd(model_inputs, &model_tape);
    if (!logits.ok()) return logits.status();
    Tape loss_tape;
    BufferVec loss_inputs = {*logits, target_buffer};
    auto device_losses = loss_layer.fwd(loss_inputs, &loss_tape);
    if (!device_losses.ok()) return device_losses.status();
    if (auto status = CudaStatus(
            cudaMemcpyAsync(losses.data(), device_losses->data(),
                            device_losses->size_bytes(), cudaMemcpyDeviceToHost,
                            stream),
            "cudaMemcpyAsync(evaluation losses)");
        !status.ok()) {
      return status;
    }
    if (auto status = CudaStatus(cudaStreamSynchronize(stream),
                                 "cudaStreamSynchronize(evaluation)");
        !status.ok()) {
      return status;
    }
    for (float loss : losses) total += loss;
  }
  return total / (static_cast<double>(eval_batches) * kBatchSize);
}

absl::Status Train(Layer& model, CrossEntropyLossLayer& loss_layer,
                   const std::vector<uint32_t>& corpus_tokens, int steps,
                   int seed, const Buffer& token_buffer,
                   const Buffer& target_buffer, cudaStream_t stream) {
  if (steps < 0) {
    return absl::InvalidArgumentError("steps must be non-negative");
  }
  std::mt19937 random(seed);
  std::uniform_int_distribution<size_t> position(0,
                                                  corpus_tokens.size() - 2);
  std::array<int, kBatchSize> tokens{};
  std::array<int, kBatchSize> targets{};
  std::array<float, kBatchSize> losses{};

  for (int step = 0; step < steps; ++step) {
    for (int row = 0; row < kBatchSize; ++row) {
      const size_t index = position(random);
      tokens[row] = static_cast<int>(corpus_tokens[index]);
      targets[row] = static_cast<int>(corpus_tokens[index + 1]);
    }
    if (auto status = CopyBatch(tokens, targets, token_buffer, target_buffer,
                                stream);
        !status.ok()) {
      return status;
    }

    Tape model_tape;
    BufferVec model_inputs = {token_buffer};
    auto logits = model.fwd(model_inputs, &model_tape);
    if (!logits.ok()) return logits.status();
    Tape loss_tape;
    BufferVec loss_inputs = {*logits, target_buffer};
    auto device_losses = loss_layer.fwd(loss_inputs, &loss_tape);
    if (!device_losses.ok()) return device_losses.status();
    auto logits_gradient = loss_layer.bwd({}, std::move(loss_tape));
    if (!logits_gradient.ok()) return logits_gradient.status();
    auto input_gradient =
        model.bwd(*logits_gradient, std::move(model_tape));
    if (!input_gradient.ok()) return input_gradient.status();

    if ((step + 1) % 200 == 0 || step + 1 == steps) {
      if (auto status = CudaStatus(
              cudaMemcpyAsync(losses.data(), device_losses->data(),
                              device_losses->size_bytes(),
                              cudaMemcpyDeviceToHost, stream),
              "cudaMemcpyAsync(training losses)");
          !status.ok()) {
        return status;
      }
      if (auto status = CudaStatus(cudaStreamSynchronize(stream),
                                   "cudaStreamSynchronize(training)");
          !status.ok()) {
        return status;
      }
      double mean = 0.0;
      for (float loss : losses) mean += loss;
      mean /= kBatchSize;
      std::cout << "step " << step + 1 << "/" << steps
                << ", batch loss: " << mean << '\n';
    }
  }
  return CudaStatus(cudaStreamSynchronize(stream),
                    "cudaStreamSynchronize(after training)");
}

absl::StatusOr<std::array<float, kVocabularySize>> Predict(
    Layer& model, uint32_t token, const Buffer& token_buffer,
    cudaStream_t stream) {
  std::array<int, kBatchSize> repeated_tokens{};
  repeated_tokens.fill(static_cast<int>(token));
  if (auto status = CudaStatus(
          cudaMemcpyAsync(token_buffer.data(), repeated_tokens.data(),
                          token_buffer.size_bytes(), cudaMemcpyHostToDevice,
                          stream),
          "cudaMemcpyAsync(prompt token)");
      !status.ok()) {
    return status;
  }
  Tape tape;
  BufferVec inputs = {token_buffer};
  auto logits = model.fwd(inputs, &tape);
  if (!logits.ok()) return logits.status();
  std::array<float, kVocabularySize> host_logits{};
  if (auto status = CudaStatus(
          cudaMemcpyAsync(host_logits.data(), logits->data(),
                          host_logits.size() * sizeof(float),
                          cudaMemcpyDeviceToHost, stream),
          "cudaMemcpyAsync(prompt logits)");
      !status.ok()) {
    return status;
  }
  if (auto status = CudaStatus(cudaStreamSynchronize(stream),
                               "cudaStreamSynchronize(prompt)");
      !status.ok()) {
    return status;
  }
  return host_logits;
}

absl::StatusOr<std::string> Generate(
    Layer& model, const PlainTextTokenizer& tokenizer,
    const std::array<bool, kVocabularySize>& observed, std::string prompt,
    int generation_tokens, double temperature, std::mt19937* random,
    const Buffer& token_buffer, cudaStream_t stream) {
  if (generation_tokens < 0 || temperature <= 0.0) {
    return absl::InvalidArgumentError(
        "generation_tokens must be non-negative and temperature positive");
  }
  if (prompt.empty()) prompt = "\n";
  std::vector<uint32_t> prompt_tokens = tokenizer.Encode(prompt);
  uint32_t current = prompt_tokens.back();
  std::vector<uint32_t> generated;
  generated.reserve(generation_tokens);

  for (int index = 0; index < generation_tokens; ++index) {
    auto logits = Predict(model, current, token_buffer, stream);
    if (!logits.ok()) return logits.status();
    const float maximum = *std::max_element(logits->begin(), logits->end());
    std::array<double, kVocabularySize> probabilities{};
    for (int token = 0; token < kVocabularySize; ++token) {
      if (observed[token]) {
        probabilities[token] =
            std::exp(((*logits)[token] - maximum) / temperature);
      }
    }
    std::discrete_distribution<int> sample(probabilities.begin(),
                                            probabilities.end());
    current = static_cast<uint32_t>(sample(*random));
    generated.push_back(current);
  }
  return tokenizer.Decode(generated);
}

absl::Status Run(cudaStream_t stream) {
  auto corpus = LoadCorpus();
  if (!corpus.ok()) return corpus.status();
  const PlainTextTokenizer tokenizer;
  const std::vector<uint32_t> corpus_tokens = tokenizer.Encode(*corpus);
  if (corpus_tokens.size() < 2) {
    return absl::InvalidArgumentError(
        "the training corpus must contain at least two bytes");
  }
  std::array<bool, kVocabularySize> observed{};
  for (uint32_t token : corpus_tokens) observed[token] = true;

  auto model = CreateShakespeareLlm(
      DataType::FP16,
      static_cast<float>(absl::GetFlag(FLAGS_learning_rate)), stream);
  if (!model.ok()) return model.status();
  auto loss_layer = CrossEntropyLossLayer::Create(DataType::FP16, stream);
  if (!loss_layer.ok()) return loss_layer.status();
  auto token_buffer = Buffer::Allocate(kBatchSize * sizeof(int), stream);
  if (!token_buffer.ok()) return token_buffer.status();
  auto target_buffer = Buffer::Allocate(kBatchSize * sizeof(int), stream);
  if (!target_buffer.ok()) return target_buffer.status();

  const int eval_batches = absl::GetFlag(FLAGS_eval_batches);
  auto initial_loss = Evaluate(**model, **loss_layer, corpus_tokens,
                               eval_batches, *token_buffer, *target_buffer,
                               stream);
  if (!initial_loss.ok()) return initial_loss.status();
  std::cout << "corpus bytes: " << corpus_tokens.size()
            << ", vocabulary: " << tokenizer.vocab_size()
            << ", initial loss: " << *initial_loss << '\n';

  if (auto status =
          Train(**model, **loss_layer, corpus_tokens,
                absl::GetFlag(FLAGS_steps), absl::GetFlag(FLAGS_seed),
                *token_buffer, *target_buffer, stream);
      !status.ok()) {
    return status;
  }
  auto final_loss = Evaluate(**model, **loss_layer, corpus_tokens, eval_batches,
                             *token_buffer, *target_buffer, stream);
  if (!final_loss.ok()) return final_loss.status();
  std::cout << "final evaluation loss: " << *final_loss << '\n';
  const double target_loss = absl::GetFlag(FLAGS_target_loss);
  if (*final_loss > target_loss) {
    return absl::FailedPreconditionError(
        absl::StrCat("model did not reach target loss ", target_loss,
                     "; final loss was ", *final_loss));
  }
  if (*final_loss >= *initial_loss) {
    return absl::FailedPreconditionError(
        "training did not reduce evaluation loss");
  }

  std::mt19937 random(absl::GetFlag(FLAGS_seed) + 1);
  auto sample = Generate(**model, tokenizer, observed, "To be",
                         std::min(120,
                                  absl::GetFlag(FLAGS_generation_tokens)),
                         absl::GetFlag(FLAGS_temperature), &random,
                         *token_buffer, stream);
  if (!sample.ok()) return sample.status();
  std::cout << "sample:\nTo be" << *sample << "\n";

  if (!absl::GetFlag(FLAGS_interactive)) return absl::OkStatus();
  std::cout << "\nEnter a prompt (Ctrl-C or Ctrl-D to quit).\n";
  std::string prompt;
  while (true) {
    std::cout << "> " << std::flush;
    if (!std::getline(std::cin, prompt)) break;
    auto completion = Generate(
        **model, tokenizer, observed, prompt,
        absl::GetFlag(FLAGS_generation_tokens),
        absl::GetFlag(FLAGS_temperature), &random, *token_buffer, stream);
    if (!completion.ok()) return completion.status();
    std::cout << prompt << *completion << "\n";
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
