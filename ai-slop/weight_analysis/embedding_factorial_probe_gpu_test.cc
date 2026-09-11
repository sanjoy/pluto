#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cstring>

#include "ai-slop/weight_analysis/causal_probe.h"
#include "ai-slop/weight_analysis/embedding_factorial_probe.h"
#include "gtest/gtest.h"
#include "src/llm/recipes/gpt2.h"

namespace pluto::weight_analysis {
namespace {

// This full-architecture test is intentionally separate from CPU validation.
// Do not run it concurrently with the controlled four-hour training arms.
TEST(EmbeddingFactorialGpuTest, NativeDiagonalsAndCausalInputExposure) {
  auto executor = cuda::Executor::Create();
  ASSERT_TRUE(executor.ok()) << executor.status();
  auto a = llm::CreateGpt2(**executor, llm::DataType::BF16, 17);
  auto j = llm::CreateGpt2(**executor, llm::DataType::BF16, 17);
  ASSERT_TRUE(a.ok()) << a.status();
  ASSERT_TRUE(j.ok()) << j.status();
  auto weights_a = UniqueWeights(**executor, (*a)->weights());
  auto weights_j = UniqueWeights(**executor, (*j)->weights());
  ASSERT_TRUE(weights_a.ok());
  ASSERT_TRUE(weights_j.ok());
  constexpr int width = llm::kGpt2ModelWidth;
  constexpr int vocabulary = llm::kGpt2VocabularySize;
  constexpr int context = llm::kGpt2ContextLength;
  constexpr int patched_token = 5;
  auto replacement_row = cuda::PageLockedHostArray<float>::Allocate(width);
  ASSERT_TRUE(replacement_row.ok());
  for (int i = 0; i < width; ++i)
    (*replacement_row)[i] = ((i % 13) - 6) * 0.05f;
  ASSERT_EQ(
      cudaMemcpyAsync(
          static_cast<float*>((*weights_j)[0].data()) + patched_token * width,
          replacement_row->data(), replacement_row->size_bytes(),
          cudaMemcpyHostToDevice, (*executor)->stream()),
      cudaSuccess);
  auto lens_a = NativeLogitLens::Create(**executor, *weights_a);
  auto lens_j = NativeLogitLens::Create(**executor, *weights_j);
  ASSERT_TRUE(lens_a.ok());
  ASSERT_TRUE(lens_j.ok());
  auto tokens = cuda::PageLockedHostArray<int32_t>::Allocate(context);
  ASSERT_TRUE(tokens.ok());
  std::fill(tokens->begin(), tokens->end(), 0);
  // No selected weight appears before position 10. An input-side effect at
  // rows 0 or 5 would violate causality or confuse the input/head cell labels.
  (*tokens)[10] = patched_token;
  auto input = cuda::Buffer::Allocate(**executor, tokens->size_bytes());
  ASSERT_TRUE(input.ok());
  ASSERT_EQ(cudaMemcpyAsync(input->data(), tokens->data(), tokens->size_bytes(),
                            cudaMemcpyHostToDevice, (*executor)->stream()),
            cudaSuccess);
  llm::Tape tape_a, tape_j;
  auto native_a =
      (*a)->fwd(**executor, absl::MakeConstSpan(&*input, 1), &tape_a);
  auto native_j =
      (*j)->fwd(**executor, absl::MakeConstSpan(&*input, 1), &tape_j);
  ASSERT_TRUE(native_a.ok());
  ASSERT_TRUE(native_j.ok());
  ASSERT_TRUE(ValidateGpt2Tape(tape_a).ok());
  ASSERT_TRUE(ValidateGpt2Tape(tape_j).ok());
  const std::array<cuda::Buffer, 2> residuals{
      tape_a.children[10].intermediates[0],
      tape_j.children[10].intermediates[0]};
  const std::array<cuda::Buffer, 2> native{*native_a, *native_j};
  const std::array<const NativeLogitLens*, 2> lenses{lens_a->get(),
                                                     lens_j->get()};
  const std::array<int32_t, 4> rows{0, 5, 10, 11};
  const std::array<int32_t, 4> targets{5, 5, 6, 9};
  auto result = EvaluateEmbeddingFactorial(**executor, lenses, residuals,
                                           native, rows, targets);
  ASSERT_TRUE(result.ok()) << result.status();
  for (int cell = 0; cell < 4; ++cell) {
    ASSERT_EQ(result->logits[cell].size(), 4U * vocabulary);
    ASSERT_EQ(result->scores[cell].size(), 4U);
    for (int i = 0; i < 4; ++i) {
      auto independent = ScoreFactorialToken(
          result->logits[cell].span().subspan(i * vocabulary, vocabulary),
          targets[i]);
      ASSERT_TRUE(independent.ok());
      EXPECT_EQ(independent->nll, result->scores[cell][i].nll);
      EXPECT_EQ(independent->target_rank, result->scores[cell][i].target_rank);
    }
  }
  for (int i : {0, 1}) {
    EXPECT_EQ(std::memcmp(result->logits[0].data() + i * vocabulary,
                          result->logits[2].data() + i * vocabulary,
                          vocabulary * sizeof(float)),
              0);
    EXPECT_EQ(std::memcmp(result->logits[1].data() + i * vocabulary,
                          result->logits[3].data() + i * vocabulary,
                          vocabulary * sizeof(float)),
              0);
    EXPECT_NE(result->logits[0][i * vocabulary + patched_token],
              result->logits[1][i * vocabulary + patched_token]);
    for (int token = 0; token < vocabulary; ++token) {
      if (token != patched_token) {
        EXPECT_EQ(result->logits[0][i * vocabulary + token],
                  result->logits[1][i * vocabulary + token]);
      }
    }
  }
  EXPECT_NE(std::memcmp(result->logits[0].data() + 2 * vocabulary,
                        result->logits[2].data() + 2 * vocabulary,
                        vocabulary * sizeof(float)),
            0);
  // Identity control: every cell must reduce to the same native model.
  auto identity = EvaluateEmbeddingFactorial(
      **executor, {lens_a->get(), lens_a->get()}, {residuals[0], residuals[0]},
      {*native_a, *native_a}, {0, 10, 11}, {5, 6, 9});
  ASSERT_TRUE(identity.ok()) << identity.status();
  for (int cell = 1; cell < 4; ++cell) {
    EXPECT_EQ(
        std::memcmp(identity->logits[0].data(), identity->logits[cell].data(),
                    identity->logits[0].size_bytes()),
        0);
  }
  // Wrong diagonal evidence must fail closed, not silently certify a mixed
  // cell as the recipient. At row 10 the changed input is already visible.
  EXPECT_FALSE(EvaluateEmbeddingFactorial(**executor, lenses, residuals,
                                          {*native_a, *native_a}, {10}, {5})
                   .ok());
  EXPECT_FALSE(EvaluateEmbeddingFactorial(**executor, {nullptr, lens_j->get()},
                                          residuals, native, rows, targets)
                   .ok());
  EXPECT_FALSE(EvaluateEmbeddingFactorial(**executor, lenses, residuals, native,
                                          {1024}, {5})
                   .ok());
  EXPECT_FALSE(EvaluateEmbeddingFactorial(**executor, lenses, residuals, native,
                                          {0}, {vocabulary})
                   .ok());

  // Exercise genuinely global selected-row indices across two independent
  // sequences. Only sequence 1 contains the patched token. Its rows 0/5/10/11
  // must match the single-sequence execution above exactly, even though their
  // flattened positions now start at context. In particular, neither future
  // exposure nor a different sequence may contaminate an earlier prediction.
  auto two_tokens = cuda::PageLockedHostArray<int32_t>::Allocate(2 * context);
  ASSERT_TRUE(two_tokens.ok());
  std::fill(two_tokens->begin(), two_tokens->end(), 0);
  (*two_tokens)[context + 10] = patched_token;
  auto two_input = cuda::Buffer::Allocate(**executor, two_tokens->size_bytes());
  ASSERT_TRUE(two_input.ok());
  ASSERT_EQ(cudaMemcpyAsync(two_input->data(), two_tokens->data(),
                            two_tokens->size_bytes(), cudaMemcpyHostToDevice,
                            (*executor)->stream()),
            cudaSuccess);
  llm::Tape two_tape_a, two_tape_j;
  auto two_native_a =
      (*a)->fwd(**executor, absl::MakeConstSpan(&*two_input, 1), &two_tape_a);
  auto two_native_j =
      (*j)->fwd(**executor, absl::MakeConstSpan(&*two_input, 1), &two_tape_j);
  ASSERT_TRUE(two_native_a.ok()) << two_native_a.status();
  ASSERT_TRUE(two_native_j.ok()) << two_native_j.status();
  ASSERT_TRUE(ValidateGpt2Tape(two_tape_a).ok());
  ASSERT_TRUE(ValidateGpt2Tape(two_tape_j).ok());
  const std::array<int32_t, 7> two_rows{0,
                                        context - 1,
                                        context,
                                        context + 5,
                                        context + 10,
                                        context + 11,
                                        2 * context - 1};
  const std::array<int32_t, 7> two_targets{5, 7, 5, 5, 6, 9, 11};
  auto two_result = EvaluateEmbeddingFactorial(
      **executor, lenses,
      {two_tape_a.children[10].intermediates[0],
       two_tape_j.children[10].intermediates[0]},
      {*two_native_a, *two_native_j}, two_rows, two_targets);
  ASSERT_TRUE(two_result.ok()) << two_result.status();
  for (int cell = 0; cell < 4; ++cell) {
    ASSERT_EQ(two_result->logits[cell].size(), two_rows.size() * vocabulary);
    ASSERT_EQ(two_result->scores[cell].size(), two_rows.size());
    for (int i = 0; i < 4; ++i) {
      EXPECT_EQ(
          std::memcmp(result->logits[cell].data() + i * vocabulary,
                      two_result->logits[cell].data() + (i + 2) * vocabulary,
                      vocabulary * sizeof(float)),
          0)
          << "cell=" << cell << " single row=" << rows[i];
      EXPECT_EQ(result->scores[cell][i].nll,
                two_result->scores[cell][i + 2].nll);
      EXPECT_EQ(result->scores[cell][i].argmax,
                two_result->scores[cell][i + 2].argmax);
      EXPECT_EQ(result->scores[cell][i].target_rank,
                two_result->scores[cell][i + 2].target_rank);
    }
  }
  for (int i : {0, 1, 2, 3}) {
    for (int head = 0; head < 2; ++head) {
      EXPECT_EQ(
          std::memcmp(two_result->logits[head].data() + i * vocabulary,
                      two_result->logits[2 + head].data() + i * vocabulary,
                      vocabulary * sizeof(float)),
          0)
          << "global row=" << two_rows[i] << " head=" << head;
    }
  }
  EXPECT_NE(std::memcmp(two_result->logits[0].data() + 4 * vocabulary,
                        two_result->logits[2].data() + 4 * vocabulary,
                        vocabulary * sizeof(float)),
            0);
  ASSERT_TRUE((*executor)->Synchronize().ok());
}

}  // namespace
}  // namespace pluto::weight_analysis
