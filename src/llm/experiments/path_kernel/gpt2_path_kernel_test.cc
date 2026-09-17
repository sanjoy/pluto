#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "gtest/gtest.h"
#include "src/cuda/page_locked_host_array.h"
#include "src/dataset/dataset.h"
#include "src/dataset/gpt2_tokenizer.h"
#include "src/llm/experiments/path_kernel/path_kernel.h"
#include "src/llm/recipes/gpt2.h"
#include "src/util/status_macros.h"

namespace pluto::llm::path_kernel {
namespace {

constexpr size_t kPromptLength = 16;
constexpr size_t kLogitOffset = (kPromptLength - 1) * kGpt2PaddedVocabularySize;

struct ParameterLayout {
  std::vector<size_t> indices;
  size_t count = 0;
};

// Deliberately do not consume Run()'s parameter metadata or Jacobian helpers.
// This separately discovers the tied embedding/LM-head allocation, and reads
// gradients from an ordinary forward/backward pass on the actual full model.
ParameterLayout FindUniqueParameters(Layer& model) {
  ParameterLayout layout;
  std::vector<void*> addresses;
  const auto weights = model.weights();
  for (size_t index = 0; index < weights.size(); ++index) {
    if (std::find(addresses.begin(), addresses.end(), weights[index].data()) !=
        addresses.end())
      continue;
    addresses.push_back(weights[index].data());
    layout.indices.push_back(index);
    layout.count += weights[index].size_bytes() / sizeof(float);
  }
  return layout;
}

absl::StatusOr<cuda::PageLockedHostArray<float>> ReadParameters(
    cuda::Executor& executor, Layer& model, const ParameterLayout& layout,
    bool gradients) {
  ASSIGN_OR_RETURN(auto host, cuda::PageLockedHostArray<float>::Allocate(
                                  executor, layout.count));
  const auto buffers = gradients ? model.gradients() : model.weights();
  size_t offset = 0;
  for (size_t index : layout.indices) {
    RETURN_IF_ERROR(cuda::CudaStatus(
        cudaMemcpyAsync(host.data() + offset, buffers[index].data(),
                        buffers[index].size_bytes(), cudaMemcpyDeviceToHost,
                        executor.stream()),
        "read independent path-kernel test parameters"));
    offset += buffers[index].size_bytes() / sizeof(float);
  }
  RETURN_IF_ERROR(executor.Synchronize());
  return host;
}

absl::StatusOr<cuda::PageLockedHostArray<float>> ReadLogitRow(
    cuda::Executor& executor, const Buffer& logits) {
  ASSIGN_OR_RETURN(auto host, cuda::PageLockedHostArray<float>::Allocate(
                                  executor, kGpt2VocabularySize));
  RETURN_IF_ERROR(cuda::CudaStatus(
      cudaMemcpyAsync(
          host.data(), static_cast<const float*>(logits.data()) + kLogitOffset,
          host.size_bytes(), cudaMemcpyDeviceToHost, executor.stream()),
      "read independent GPT-2 vocabulary row"));
  RETURN_IF_ERROR(executor.Synchronize());
  return host;
}

struct Softmax {
  double maximum;
  double denominator;
  double loss;
};

// A scalar, CPU-double full-vocabulary oracle. No production CrossEntropy()
// call is used here. In particular, the denominator includes all 50,257
// logical classes, never just the one queried logit or padded -infinity slots.
Softmax ReferenceSoftmax(absl::Span<const float> logits, size_t target) {
  const double maximum = *std::max_element(logits.begin(), logits.end());
  double denominator = 0;
  for (float logit : logits)
    denominator += std::exp(static_cast<double>(logit) - maximum);
  return {maximum, denominator,
          maximum - logits[target] + std::log(denominator)};
}

struct ManualDerivative {
  cuda::PageLockedHostArray<float> values;
  double scalar;
};

absl::StatusOr<ManualDerivative> DifferentiateManually(
    cuda::Executor& executor, Layer& model, const Buffer& tokens,
    const ParameterLayout& layout, size_t target, bool cross_entropy) {
  for (size_t index : layout.indices) {
    const Buffer& gradient = model.gradients()[index];
    RETURN_IF_ERROR(cuda::CudaStatus(
        cudaMemsetAsync(gradient.data(), 0, gradient.size_bytes(),
                        executor.stream()),
        "clear independent path-kernel gradient"));
  }
  ASSIGN_OR_RETURN(auto forward, model.fwd(executor, {tokens}));
  const Buffer& logits = forward.outputs[0];
  ASSIGN_OR_RETURN(auto row, ReadLogitRow(executor, logits));
  double scalar = row[target];
  if (cross_entropy) {
    const Softmax softmax = ReferenceSoftmax(row.span(), target);
    scalar = softmax.loss;
    for (size_t token = 0; token < row.size(); ++token) {
      const double probability =
          std::exp(static_cast<double>(row[token]) - softmax.maximum) /
          softmax.denominator;
      row[token] = static_cast<float>(probability - (token == target ? 1 : 0));
    }
  } else {
    std::fill(row.begin(), row.end(), 0);
    row[target] = 1;
  }
  ASSIGN_OR_RETURN(auto seed, Buffer::Allocate(executor, logits.size_bytes()));
  RETURN_IF_ERROR(cuda::CudaStatus(
      cudaMemsetAsync(seed.data(), 0, seed.size_bytes(), executor.stream()),
      "clear independent full-logit seed"));
  RETURN_IF_ERROR(cuda::CudaStatus(
      cudaMemcpyAsync(static_cast<float*>(seed.data()) + kLogitOffset,
                      row.data(), row.size_bytes(), cudaMemcpyHostToDevice,
                      executor.stream()),
      "seed independent full-logit backward"));
  ASSIGN_OR_RETURN(auto ignored,
                   model.bwd(executor, {seed}, std::move(forward.state)));
  (void)ignored;
  ASSIGN_OR_RETURN(auto derivatives,
                   ReadParameters(executor, model, layout, true));
  return ManualDerivative{std::move(derivatives), scalar};
}

TEST(Gpt2PathKernelTest,
     ShakespeareFullVocabularyUpdatesAndContributionsMatchIndependentOracle) {
  const char* tokenizer_directory = std::getenv("PLUTO_GPT2_TOKENIZER_DIR");
  const char* runfiles = std::getenv("TEST_SRCDIR");
  const char* workspace = std::getenv("TEST_WORKSPACE");
  ASSERT_NE(tokenizer_directory, nullptr)
      << "set PLUTO_GPT2_TOKENIZER_DIR to the saved GPT-2 tokenizer";
  ASSERT_STRNE(tokenizer_directory, "");
  ASSERT_NE(runfiles, nullptr);
  ASSERT_NE(workspace, nullptr);
  const auto path =
      std::filesystem::path(runfiles) / workspace / "testdata/shakespeare.txt";
  auto corpus = LoadTextCorpus(path.string());
  auto tokenizer = tokenizer::Gpt2Tokenizer::Load(tokenizer_directory);
  ASSERT_TRUE(corpus.ok()) << corpus.status();
  ASSERT_TRUE(tokenizer.ok()) << tokenizer.status();
  ASSERT_EQ((*tokenizer)->vocab_size(), kGpt2VocabularySize);
  auto executor = cuda::Executor::Create();
  ASSERT_TRUE(executor.ok()) << executor.status();
  auto encoded = (*tokenizer)->Encode(**executor, corpus->text());
  ASSERT_TRUE(encoded.ok()) << encoded.status();
  constexpr std::array<size_t, 3> kOffsets{0, 64, 128};
  ASSERT_GT(encoded->size(), kOffsets.back() + kPromptLength);
  BufferVec inputs;
  std::vector<size_t> targets;
  for (size_t offset : kOffsets) {
    auto host = cuda::PageLockedHostArray<int>::Allocate(**executor,
                                                         kGpt2ContextLength);
    ASSERT_TRUE(host.ok()) << host.status();
    std::fill(host->begin(), host->end(), (*tokenizer)->eos_token_id());
    std::copy_n(encoded->data() + offset, kPromptLength, host->data());
    auto buffer = Buffer::Allocate(**executor, host->size_bytes());
    ASSERT_TRUE(buffer.ok()) << buffer.status();
    ASSERT_EQ(cudaMemcpyAsync(buffer->data(), host->data(), host->size_bytes(),
                              cudaMemcpyHostToDevice, (*executor)->stream()),
              cudaSuccess);
    inputs.push_back(std::move(*buffer));
    targets.push_back((*encoded)[offset + kPromptLength]);
  }

  constexpr int kSeed = 1729;
  auto model = CreateGpt2(**executor, DataType::FP16, kSeed);
  ASSERT_TRUE(model.ok()) << model.status();
  ASSERT_EQ((*model)->weights().size(), 101);
  ASSERT_EQ((*model)->weights().front().data(),
            (*model)->weights().back().data());
  const ParameterLayout layout = FindUniqueParameters(**model);
  ASSERT_EQ(layout.indices.size(), 100);
  auto initial_weights = ReadParameters(**executor, **model, layout, false);
  ASSERT_TRUE(initial_weights.ok()) << initial_weights.status();
  auto first = DifferentiateManually(**executor, **model, inputs[0], layout,
                                     targets[0], true);
  ASSERT_TRUE(first.ok()) << first.status();
  auto second = DifferentiateManually(**executor, **model, inputs[1], layout,
                                      targets[1], true);
  ASSERT_TRUE(second.ok()) << second.status();
  auto query = DifferentiateManually(**executor, **model, inputs[2], layout,
                                     targets[2], false);
  ASSERT_TRUE(query.ok()) << query.status();
  // The manual query leaves nonzero model gradients. Run() must restore them,
  // rather than assuming the caller initially supplied zero accumulators.
  const std::vector<TrainingExample> training{
      {{inputs[0]},
       CrossEntropy({0, kLogitOffset}, kGpt2VocabularySize, targets[0])},
      {{inputs[1]},
       CrossEntropy({0, kLogitOffset}, kGpt2VocabularySize, targets[1])}};
  const std::vector<ntk::Sample> queries{
      {{inputs[2]}, {{0, kLogitOffset + targets[2]}}}};
  Options options;
  options.steps = 1;
  options.learning_rate = 1e-5;
  auto result =
      path_kernel::Run(**executor, **model, training, queries, options);
  ASSERT_TRUE(result.ok()) << result.status();
  ASSERT_EQ(result->steps.size(), 1);
  ASSERT_EQ(result->initial_values.size(), 1);
  ASSERT_EQ(result->final_values.size(), 1);
  ASSERT_EQ(result->contributions.rows, 1);
  ASSERT_EQ(result->contributions.columns, 2);
  ASSERT_EQ(result->path_kernel.rows, 1);
  EXPECT_EQ(result->parameter_count, layout.count);
  ASSERT_EQ(result->parameters.size(), layout.indices.size());
  for (size_t index = 0; index < layout.indices.size(); ++index)
    EXPECT_EQ(result->parameters[index].weight_index, layout.indices[index]);
  EXPECT_DOUBLE_EQ(result->initial_values[0], query->scalar);
  EXPECT_DOUBLE_EQ(result->steps[0].training_losses[0], first->scalar);
  EXPECT_DOUBLE_EQ(result->steps[0].training_losses[1], second->scalar);

  // Dot every unique parameter in CPU double, without using ComputeGram(),
  // Run()'s parameter layout, or the production contribution implementation.
  std::array<double, 2> dots{};
  double squared_norm = 0;
  for (size_t parameter = 0; parameter < layout.count; ++parameter) {
    const double derivative = query->values[parameter];
    dots[0] += derivative * first->values[parameter];
    dots[1] += derivative * second->values[parameter];
    squared_norm += derivative * derivative;
  }
  double reconstructed = query->scalar;
  for (size_t example = 0; example < dots.size(); ++example) {
    const double expected = -options.learning_rate / 2 * dots[example];
    EXPECT_NEAR(result->contributions(0, example), expected,
                std::max(1e-12, std::abs(expected) * 1e-9));
    reconstructed += expected;
  }
  EXPECT_NEAR(result->path_kernel(0, 0), options.learning_rate * squared_norm,
              options.learning_rate * squared_norm * 1e-9);
  EXPECT_NEAR(result->reconstructed_values[0], reconstructed, 1e-9);

  auto final_weights = ReadParameters(**executor, **model, layout, false);
  auto restored_gradients = ReadParameters(**executor, **model, layout, true);
  ASSERT_TRUE(final_weights.ok()) << final_weights.status();
  ASSERT_TRUE(restored_gradients.ok()) << restored_gradients.status();
  size_t incorrect_updates = 0;
  size_t changed_parameters = 0;
  size_t incorrect_gradients = 0;
  for (size_t parameter = 0; parameter < layout.count; ++parameter) {
    const double gradient = static_cast<double>(first->values[parameter]) +
                            static_cast<double>(second->values[parameter]);
    const double original = (*initial_weights)[parameter];
    const double scale = options.learning_rate / 2;
    // Either host/device compiler may contract multiply/subtract to a double
    // FMA. Accept exactly either correctly rounded FP32 result, not an error
    // bound large enough to hide an omitted or twice-applied parameter update.
    const float separate = static_cast<float>(original - scale * gradient);
    const float fused =
        static_cast<float>(std::fma(-scale, gradient, original));
    const float actual = (*final_weights)[parameter];
    incorrect_updates += actual != separate && actual != fused;
    changed_parameters += actual != (*initial_weights)[parameter];
    incorrect_gradients +=
        (*restored_gradients)[parameter] != query->values[parameter];
  }
  EXPECT_EQ(incorrect_updates, 0);
  EXPECT_EQ(incorrect_gradients, 0);
  EXPECT_GT(changed_parameters, 1000);

  // Independently measure the real updated model, including full-vocabulary
  // loss. Reconstruction is a tangent approximation; reduced precision and a
  // finite step need not produce zero residual, and the test does not claim so.
  auto final_query = (*model)->fwd(**executor, {inputs[2]});
  ASSERT_TRUE(final_query.ok()) << final_query.status();
  auto final_logits = ReadLogitRow(**executor, final_query->outputs[0]);
  ASSERT_TRUE(final_logits.ok()) << final_logits.status();
  EXPECT_DOUBLE_EQ(result->final_values[0], (*final_logits)[targets[2]]);
  EXPECT_NEAR(result->residual[0], result->final_values[0] - reconstructed,
              1e-9);
  EXPECT_TRUE(std::isfinite(result->residual[0]));
  RecordProperty("path_reconstruction_residual",
                 std::to_string(result->residual[0]));
  ASSERT_EQ(result->final_training_losses.size(), 2);
  double final_mean_loss = 0;
  for (size_t example = 0; example < 2; ++example) {
    auto forward = (*model)->fwd(**executor, {inputs[example]});
    ASSERT_TRUE(forward.ok()) << forward.status();
    auto row = ReadLogitRow(**executor, forward->outputs[0]);
    ASSERT_TRUE(row.ok()) << row.status();
    const double loss = ReferenceSoftmax(row->span(), targets[example]).loss;
    EXPECT_DOUBLE_EQ(result->final_training_losses[example], loss);
    final_mean_loss += loss / 2;
  }
  EXPECT_LT(final_mean_loss, (first->scalar + second->scalar) / 2);

  // A separately initialized model with the same seed must reproduce actual
  // master weights AND the path accounting bitwise, despite different initial
  // gradient accumulators (zero here, a manual derivative in the first run).
  auto repeat_model = CreateGpt2(**executor, DataType::FP16, kSeed);
  ASSERT_TRUE(repeat_model.ok()) << repeat_model.status();
  auto repeat =
      path_kernel::Run(**executor, **repeat_model, training, queries, options);
  ASSERT_TRUE(repeat.ok()) << repeat.status();
  EXPECT_EQ(repeat->initial_values, result->initial_values);
  EXPECT_EQ(repeat->final_values, result->final_values);
  EXPECT_EQ(repeat->contributions.values, result->contributions.values);
  EXPECT_EQ(repeat->path_kernel.values, result->path_kernel.values);
  EXPECT_EQ(repeat->reconstructed_values, result->reconstructed_values);
  EXPECT_EQ(repeat->residual, result->residual);
  EXPECT_EQ(repeat->final_training_losses, result->final_training_losses);
  auto repeat_weights =
      ReadParameters(**executor, **repeat_model, layout, false);
  ASSERT_TRUE(repeat_weights.ok()) << repeat_weights.status();
  EXPECT_TRUE(std::equal(final_weights->begin(), final_weights->end(),
                         repeat_weights->begin()));
}

}  // namespace
}  // namespace pluto::llm::path_kernel
