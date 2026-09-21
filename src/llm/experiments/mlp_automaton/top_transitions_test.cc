#include "src/llm/experiments/mlp_automaton/top_transitions.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "gtest/gtest.h"
#include "src/cuda/page_locked_host_array.h"

namespace pluto::llm::mlp_automaton {
namespace {

TopTransition Reference(const float* row, int vocabulary) {
  int winner = 0;
  for (int token = 0; token < vocabulary; ++token) {
    if (!std::isfinite(row[token]))
      return {-1, std::numeric_limits<float>::quiet_NaN()};
    if (row[token] > row[winner])
      winner = token;
  }
  double denominator = 0;
  for (int token = 0; token < vocabulary; ++token)
    denominator += std::exp(static_cast<double>(row[token]) - row[winner]);
  return {winner, static_cast<float>(1.0 / denominator)};
}

class TopTransitionsTest : public testing::Test {
 protected:
  void SetUp() override {
    auto executor = cuda::Executor::Create();
    ASSERT_TRUE(executor.ok()) << executor.status();
    executor_ = std::move(*executor);
  }

  void TearDown() override {
    if (executor_)
      EXPECT_TRUE(executor_->Synchronize().ok());
  }

  void Check(const std::vector<float>& logits, int rows, int vocabulary,
             int padded_vocabulary) {
    ASSERT_EQ(logits.size(), static_cast<size_t>(rows) * padded_vocabulary);
    auto host_input =
        cuda::PageLockedHostArray<float>::Allocate(*executor_, logits.size());
    ASSERT_TRUE(host_input.ok()) << host_input.status();
    std::copy(logits.begin(), logits.end(), host_input->begin());
    auto input = cuda::Buffer::Allocate(*executor_, host_input->size_bytes());
    ASSERT_TRUE(input.ok()) << input.status();
    ASSERT_EQ(cudaMemcpyAsync(input->data(), host_input->data(),
                              host_input->size_bytes(), cudaMemcpyHostToDevice,
                              executor_->stream()),
              cudaSuccess);
    auto output = ReadTopTransitions(*executor_, *input, rows, vocabulary,
                                     padded_vocabulary);
    ASSERT_TRUE(output.ok()) << output.status();
    EXPECT_EQ(output->size_bytes(),
              static_cast<size_t>(rows) * sizeof(TopTransition));
    EXPECT_EQ(&output->executor(), executor_.get());
    auto host_output =
        cuda::PageLockedHostArray<TopTransition>::Allocate(*executor_, rows);
    ASSERT_TRUE(host_output.ok()) << host_output.status();
    ASSERT_EQ(cudaMemcpyAsync(host_output->data(), output->data(),
                              output->size_bytes(), cudaMemcpyDeviceToHost,
                              executor_->stream()),
              cudaSuccess);
    ASSERT_TRUE(executor_->Synchronize().ok());
    for (int row = 0; row < rows; ++row) {
      SCOPED_TRACE(row);
      const auto expected = Reference(
          logits.data() + static_cast<size_t>(row) * padded_vocabulary,
          vocabulary);
      const auto actual = (*host_output)[row];
      EXPECT_EQ(actual.token, expected.token);
      if (expected.token < 0) {
        EXPECT_TRUE(std::isnan(actual.probability));
      } else {
        EXPECT_NEAR(actual.probability, expected.probability,
                    2e-6f * expected.probability);
        EXPECT_GT(actual.probability, 0);
        EXPECT_LE(actual.probability, 1);
      }
    }
  }

  std::unique_ptr<cuda::Executor> executor_;
};

TEST_F(TopTransitionsTest, IgnoresPaddingAndResolvesTiesByLowestToken) {
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const float inf = std::numeric_limits<float>::infinity();
  Check({-4,   2,    2,     0,    inf,    nan,    10000, 9999,
         9998, 9997, 1e30f, -inf, -10000, -10001, -9999, -10002,
         nan,  inf,  0,     0,    0,      0,      1e30f, 1e30f},
        4, 4, 6);
}

TEST_F(TopTransitionsTest, HandlesFullVocabularyAndCrossThreadTies) {
  constexpr int vocabulary = 50257;
  constexpr int padded = 50272;
  constexpr int rows = 4;
  std::vector<float> logits(rows * padded,
                            std::numeric_limits<float>::infinity());
  for (int row = 0; row < rows; ++row) {
    for (int token = 0; token < vocabulary; ++token) {
      logits[row * padded + token] =
          static_cast<float>((token * 17 + row * 37) % 997) / 100.0f - 8.0f;
    }
  }
  logits[3] = logits[256] = logits[49155] = 12;
  std::fill(logits.begin() + padded, logits.begin() + padded + vocabulary,
            5.0f);
  logits[2 * padded + vocabulary - 1] = 20;
  Check(logits, rows, vocabulary, padded);
}

TEST_F(TopTransitionsTest, InvalidatesOnlyRowsWithNonfiniteLogicalValues) {
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const float inf = std::numeric_limits<float>::infinity();
  Check({0, nan, 1,   99,  0,   inf, 1, 99, 0, -inf,
         1, 99,  nan, nan, nan, 99,  0, 1,  2, nan},
        5, 3, 4);
}

TEST_F(TopTransitionsTest, HandlesOneTokenAndExtremeFiniteRanges) {
  Check({-3e38f, 3e38f, 7, -9}, 2, 1, 2);
  Check({-3e38f, 3e38f, 0, -100, 3e38f, 3e38f, -3e38f, 0}, 2, 4, 4);
}

TEST_F(TopTransitionsTest, PartialTilesLowestFiniteAndSignedZeroTies) {
  for (int vocabulary : {1, 255, 256, 257, 4475}) {
    SCOPED_TRACE(vocabulary);
    const int stride = vocabulary + 3;
    std::vector<float> logits(4 * stride,
                              std::numeric_limits<float>::quiet_NaN());
    for (int token = 0; token < vocabulary; ++token) {
      logits[token] = std::numeric_limits<float>::lowest();
      logits[stride + token] = std::numeric_limits<float>::max();
      logits[2 * stride + token] = token % 2 == 0 ? -0.0f : 0.0f;
      logits[3 * stride + token] = -17;
    }
    logits[3 * stride + vocabulary - 1] = -1;
    Check(logits, 4, vocabulary, stride);
  }
}

TEST_F(TopTransitionsTest, ResultsAreBitwiseRepeatable) {
  constexpr int kRows = 3;
  constexpr int kVocabulary = 1031;
  constexpr int kStride = 1033;
  auto host_input =
      cuda::PageLockedHostArray<float>::Allocate(*executor_, kRows * kStride);
  auto first =
      cuda::PageLockedHostArray<TopTransition>::Allocate(*executor_, kRows);
  auto next =
      cuda::PageLockedHostArray<TopTransition>::Allocate(*executor_, kRows);
  ASSERT_TRUE(host_input.ok());
  ASSERT_TRUE(first.ok());
  ASSERT_TRUE(next.ok());
  std::fill(host_input->begin(), host_input->end(), -3);
  (*host_input)[257] = (*host_input)[1030] = 2;
  (*host_input)[2 * kStride] = std::numeric_limits<float>::quiet_NaN();
  auto input = cuda::Buffer::Allocate(*executor_, host_input->size_bytes());
  ASSERT_TRUE(input.ok());
  ASSERT_EQ(cudaMemcpyAsync(input->data(), host_input->data(),
                            host_input->size_bytes(), cudaMemcpyHostToDevice,
                            executor_->stream()),
            cudaSuccess);
  for (int repetition = 0; repetition < 4; ++repetition) {
    auto output =
        ReadTopTransitions(*executor_, *input, kRows, kVocabulary, kStride);
    ASSERT_TRUE(output.ok()) << output.status();
    auto& destination = repetition == 0 ? *first : *next;
    ASSERT_EQ(cudaMemcpyAsync(destination.data(), output->data(),
                              output->size_bytes(), cudaMemcpyDeviceToHost,
                              executor_->stream()),
              cudaSuccess);
    ASSERT_TRUE(executor_->Synchronize().ok());
    if (repetition != 0)
      EXPECT_EQ(std::memcmp(first->data(), next->data(), first->size_bytes()),
                0);
  }
}

TEST_F(TopTransitionsTest, RejectsInvalidDimensionsAndExactSizeMismatch) {
  auto input = cuda::Buffer::Allocate(*executor_, 16 * sizeof(float));
  ASSERT_TRUE(input.ok()) << input.status();
  for (const auto& dimensions : {std::vector<int>{0, 4, 4},
                                 {-1, 4, 4},
                                 {1, 0, 4},
                                 {1, -1, 4},
                                 {1, 5, 4},
                                 {1, 4, -1},
                                 {3, 4, 4},
                                 {5, 4, 4}}) {
    auto result = ReadTopTransitions(*executor_, *input, dimensions[0],
                                     dimensions[1], dimensions[2]);
    EXPECT_EQ(result.status().code(), absl::StatusCode::kInvalidArgument);
  }
  auto empty = cuda::Buffer::Allocate(*executor_, 0);
  ASSERT_TRUE(empty.ok()) << empty.status();
  EXPECT_EQ(ReadTopTransitions(*executor_, *empty, 1, 1, 1).status().code(),
            absl::StatusCode::kInvalidArgument);
}

TEST_F(TopTransitionsTest, RejectsInputFromAnotherExecutor) {
  auto other = cuda::Executor::Create();
  ASSERT_TRUE(other.ok()) << other.status();
  auto input = cuda::Buffer::Allocate(**other, 4 * sizeof(float));
  ASSERT_TRUE(input.ok()) << input.status();
  EXPECT_EQ(ReadTopTransitions(*executor_, *input, 1, 4, 4).status().code(),
            absl::StatusCode::kInvalidArgument);
}

}  // namespace
}  // namespace pluto::llm::mlp_automaton
