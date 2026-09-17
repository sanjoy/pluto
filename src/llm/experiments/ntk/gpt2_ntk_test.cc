#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "gtest/gtest.h"
#include "src/cuda/page_locked_host_array.h"
#include "src/dataset/dataset.h"
#include "src/dataset/gpt2_tokenizer.h"
#include "src/llm/experiments/ntk/empirical_ntk.h"
#include "src/llm/layer.h"
#include "src/llm/recipes/gpt2.h"
#include "src/util/status_macros.h"

namespace pluto::llm::ntk {
namespace {

struct ManualJacobianRow {
  cuda::PageLockedHostArray<float> gradient;
  double initial_value;
};

// Intentionally independent of ComputeEmpiricalKernel and its parameter-block
// metadata. Seed one chosen logit directly, then read every independently
// enumerated unique parameter gradient. This checks full backpropagation and
// concatenation, not just properties that any positive-semidefinite matrix
// could satisfy. The helper does mutate test-owned gradient accumulators.
absl::StatusOr<ManualJacobianRow> ManualGradient(
    cuda::Executor& executor, Layer& model, const Sample& sample,
    const OutputCoordinate& coordinate,
    const std::vector<size_t>& unique_weight_indices, size_t parameter_count) {
  for (size_t index : unique_weight_indices) {
    const Buffer& gradient = model.gradients()[index];
    RETURN_IF_ERROR(cuda::CudaStatus(
        cudaMemsetAsync(gradient.data(), 0, gradient.size_bytes(),
                        executor.stream()),
        "clear independent GPT-2 test gradient"));
  }
  ASSIGN_OR_RETURN(auto forward, model.fwd(executor, sample.inputs));
  if (forward.outputs.size() != 1 || coordinate.output_index != 0)
    return absl::InvalidArgumentError("GPT-2 test expects one logit tensor");
  const Buffer& logits = forward.outputs[0];
  if (coordinate.element >= logits.size_bytes() / sizeof(float))
    return absl::InvalidArgumentError(
        "GPT-2 test logit coordinate is out of bounds");
  ASSIGN_OR_RETURN(auto scalar,
                   cuda::PageLockedHostArray<float>::Allocate(executor, 2));
  scalar[0] = 1.0f;
  ASSIGN_OR_RETURN(auto seed, Buffer::Allocate(executor, logits.size_bytes()));
  RETURN_IF_ERROR(cuda::CudaStatus(
      cudaMemsetAsync(seed.data(), 0, seed.size_bytes(), executor.stream()),
      "clear independent GPT-2 output seed"));
  RETURN_IF_ERROR(cuda::CudaStatus(
      cudaMemcpyAsync(static_cast<float*>(seed.data()) + coordinate.element,
                      scalar.data(), sizeof(float), cudaMemcpyHostToDevice,
                      executor.stream()),
      "seed independent GPT-2 scalar output"));
  RETURN_IF_ERROR(cuda::CudaStatus(
      cudaMemcpyAsync(
          scalar.data() + 1,
          static_cast<const float*>(logits.data()) + coordinate.element,
          sizeof(float), cudaMemcpyDeviceToHost, executor.stream()),
      "read independent GPT-2 scalar output"));
  ASSIGN_OR_RETURN(auto input_gradients,
                   model.bwd(executor, {seed}, std::move(forward.state)));
  (void)input_gradients;

  ASSIGN_OR_RETURN(
      auto host_gradient,
      cuda::PageLockedHostArray<float>::Allocate(executor, parameter_count));
  size_t offset = 0;
  for (size_t index : unique_weight_indices) {
    const Buffer& gradient = model.gradients()[index];
    RETURN_IF_ERROR(cuda::CudaStatus(
        cudaMemcpyAsync(host_gradient.data() + offset, gradient.data(),
                        gradient.size_bytes(), cudaMemcpyDeviceToHost,
                        executor.stream()),
        "read independent GPT-2 parameter gradient"));
    offset += gradient.size_bytes() / sizeof(float);
  }
  RETURN_IF_ERROR(executor.Synchronize());
  return ManualJacobianRow{std::move(host_gradient), scalar[1]};
}

// Gaussian elimination is ample for the at-most-four-dimensional principal
// submatrices here. Normalize first so the acceptance threshold does not depend
// on the large, unnormalized NTK diagonals.
double NormalizedPrincipalMinor(const Matrix& gram, unsigned subset) {
  std::vector<size_t> indices;
  for (size_t i = 0; i < gram.rows; ++i)
    if ((subset & (1u << i)) != 0)
      indices.push_back(i);
  const size_t n = indices.size();
  std::vector<double> values(n * n);
  for (size_t i = 0; i < n; ++i)
    for (size_t j = 0; j < n; ++j)
      values[i * n + j] = gram(indices[i], indices[j]) /
                          std::sqrt(gram(indices[i], indices[i]) *
                                    gram(indices[j], indices[j]));
  double determinant = 1.0;
  for (size_t column = 0; column < n; ++column) {
    size_t pivot = column;
    for (size_t row = column + 1; row < n; ++row)
      if (std::abs(values[row * n + column]) >
          std::abs(values[pivot * n + column]))
        pivot = row;
    if (std::abs(values[pivot * n + column]) < 1e-15)
      return 0.0;
    if (pivot != column) {
      for (size_t j = 0; j < n; ++j)
        std::swap(values[pivot * n + j], values[column * n + j]);
      determinant = -determinant;
    }
    const double diagonal = values[column * n + column];
    determinant *= diagonal;
    for (size_t row = column + 1; row < n; ++row) {
      const double factor = values[row * n + column] / diagonal;
      for (size_t j = column + 1; j < n; ++j)
        values[row * n + j] -= factor * values[column * n + j];
    }
  }
  return determinant;
}

TEST(Gpt2NtkTest, RealShakespeareFullModelMatchesIndependentGradientDots) {
  const char* tokenizer_directory = std::getenv("PLUTO_GPT2_TOKENIZER_DIR");
  ASSERT_NE(tokenizer_directory, nullptr)
      << "set PLUTO_GPT2_TOKENIZER_DIR to the saved GPT-2 tokenizer";
  ASSERT_STRNE(tokenizer_directory, "");
  const char* runfiles = std::getenv("TEST_SRCDIR");
  const char* workspace = std::getenv("TEST_WORKSPACE");
  ASSERT_NE(runfiles, nullptr);
  ASSERT_NE(workspace, nullptr);
  const auto corpus_path =
      std::filesystem::path(runfiles) / workspace / "testdata/shakespeare.txt";
  auto corpus = LoadTextCorpus(corpus_path.string());
  ASSERT_TRUE(corpus.ok()) << corpus.status();
  auto tokenizer = tokenizer::Gpt2Tokenizer::Load(tokenizer_directory);
  ASSERT_TRUE(tokenizer.ok()) << tokenizer.status();
  ASSERT_EQ((*tokenizer)->vocab_size(), kGpt2VocabularySize);
  auto executor = cuda::Executor::Create();
  ASSERT_TRUE(executor.ok()) << executor.status();
  auto encoded = (*tokenizer)->Encode(**executor, corpus->text());
  ASSERT_TRUE(encoded.ok()) << encoded.status();

  constexpr int kPromptLength = 16;
  constexpr std::array<size_t, 2> kOffsets{0, 64};
  ASSERT_GT(encoded->size(), kOffsets.back() + kPromptLength);
  std::vector<Sample> samples;
  for (size_t offset : kOffsets) {
    auto host = cuda::PageLockedHostArray<int>::Allocate(**executor,
                                                         kGpt2ContextLength);
    ASSERT_TRUE(host.ok()) << host.status();
    std::fill(host->begin(), host->end(), (*tokenizer)->eos_token_id());
    std::copy_n(encoded->data() + offset, kPromptLength, host->data());
    auto tokens = Buffer::Allocate(**executor, host->size_bytes());
    ASSERT_TRUE(tokens.ok()) << tokens.status();
    ASSERT_EQ(cudaMemcpyAsync(tokens->data(), host->data(), host->size_bytes(),
                              cudaMemcpyHostToDevice, (*executor)->stream()),
              cudaSuccess);
    const int actual_next = (*encoded)[offset + kPromptLength];
    const int alternative = (actual_next + 1) % kGpt2VocabularySize;
    const size_t row_offset =
        static_cast<size_t>(kPromptLength - 1) * kGpt2PaddedVocabularySize;
    // Two classes at the same prediction position give a vector-valued kernel,
    // not a scalar-output trace. Both selected classes are logical vocabulary
    // entries, never the -infinity padding at the end of the LM head.
    samples.push_back({.inputs = {*tokens},
                       .coordinates = {{0, row_offset + actual_next},
                                       {0, row_offset + alternative}}});
  }

  auto model = CreateGpt2(**executor, DataType::FP16, 1729);
  ASSERT_TRUE(model.ok()) << model.status();
  ASSERT_EQ((*model)->weights().size(), 101);
  ASSERT_EQ((*model)->gradients().size(), 101);
  ASSERT_EQ((*model)->weights().front().data(),
            (*model)->weights().back().data());
  ASSERT_EQ((*model)->gradients().front().data(),
            (*model)->gradients().back().data());
  std::vector<size_t> unique_indices;
  std::vector<void*> addresses;
  size_t parameter_count = 0;
  for (size_t i = 0; i < (*model)->weights().size(); ++i) {
    const Buffer& weight = (*model)->weights()[i];
    if (std::find(addresses.begin(), addresses.end(), weight.data()) !=
        addresses.end())
      continue;
    addresses.push_back(weight.data());
    unique_indices.push_back(i);
    parameter_count += weight.size_bytes() / sizeof(float);
  }
  ASSERT_EQ(unique_indices.size(), 100);
  auto result = ComputeEmpiricalKernel(**executor, **model, samples);
  ASSERT_TRUE(result.ok()) << result.status();
  ASSERT_EQ(result->gram.rows, 4);
  ASSERT_EQ(result->gram.columns, 4);
  ASSERT_EQ(result->initial_values.size(), 4);
  ASSERT_EQ(result->parameters.size(), 100);
  EXPECT_EQ(result->parameter_count, parameter_count);
  size_t parameter_offset = 0;
  for (size_t i = 0; i < unique_indices.size(); ++i) {
    EXPECT_EQ(result->parameters[i].weight_index, unique_indices[i]);
    EXPECT_EQ(result->parameters[i].offset, parameter_offset);
    const size_t count =
        (*model)->weights()[unique_indices[i]].size_bytes() / sizeof(float);
    EXPECT_EQ(result->parameters[i].elements, count);
    parameter_offset += count;
  }
  for (size_t i = 0; i < 4; ++i) {
    EXPECT_TRUE(std::isfinite(result->initial_values[i]));
    ASSERT_GT(result->gram(i, i), 0.0);
    for (size_t j = 0; j < 4; ++j) {
      ASSERT_TRUE(std::isfinite(result->gram(i, j)));
      EXPECT_DOUBLE_EQ(result->gram(i, j), result->gram(j, i));
    }
  }
  for (unsigned subset = 1; subset < 16; ++subset)
    EXPECT_GE(NormalizedPrincipalMinor(result->gram, subset), -1e-9) << subset;
  EXPECT_GT(std::abs(result->gram(0, 1)),
            1e-8 * std::sqrt(result->gram(0, 0) * result->gram(1, 1)));

  auto first =
      ManualGradient(**executor, **model, samples[0], samples[0].coordinates[0],
                     unique_indices, parameter_count);
  auto second =
      ManualGradient(**executor, **model, samples[1], samples[1].coordinates[1],
                     unique_indices, parameter_count);
  ASSERT_TRUE(first.ok()) << first.status();
  ASSERT_TRUE(second.ok()) << second.status();
  EXPECT_DOUBLE_EQ(first->initial_value, result->initial_values[0]);
  EXPECT_DOUBLE_EQ(second->initial_value, result->initial_values[3]);
  double first_norm = 0.0, second_norm = 0.0, cross = 0.0;
  for (size_t parameter = 0; parameter < parameter_count; ++parameter) {
    const double a = first->gradient[parameter];
    const double b = second->gradient[parameter];
    first_norm += a * a;
    second_norm += b * b;
    cross += a * b;
  }
  // CPU sequential sums and GPU tree reductions differ only in FP64 addition
  // order. These tolerances are far tighter than FP16 compute error because
  // both paths deliberately compare the SAME implemented backward gradients.
  EXPECT_NEAR(result->gram(0, 0), first_norm, 1e-8 * std::max(1.0, first_norm));
  EXPECT_NEAR(result->gram(3, 3), second_norm,
              1e-8 * std::max(1.0, second_norm));
  EXPECT_NEAR(result->gram(0, 3), cross,
              1e-8 * std::max(1.0, std::sqrt(first_norm * second_norm)));
}

}  // namespace
}  // namespace pluto::llm::ntk
