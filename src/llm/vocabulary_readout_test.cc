#include "src/llm/vocabulary_readout.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <numeric>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "gtest/gtest.h"
#include "src/cuda/page_locked_host_array.h"
#include "src/util/status_macros.h"

namespace pluto::llm {
namespace {

TopThreeTokens ReferenceSoftmax(absl::Span<const double> logits) {
  TopThreeTokens result{};
  for (double value : logits)
    if (!std::isfinite(value)) {
      for (int rank = 0; rank < 3; ++rank) {
        result.tokens[rank] = -1;
        result.probabilities[rank] = std::numeric_limits<float>::quiet_NaN();
      }
      return result;
    }
  std::vector<int> order(logits.size());
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](int left, int right) {
    return logits[left] == logits[right] ? left < right
                                         : logits[left] > logits[right];
  });
  double denominator = 0;
  for (double value : logits)
    denominator += std::exp(value - logits[order[0]]);
  for (int rank = 0; rank < 3; ++rank) {
    result.tokens[rank] = order[rank];
    result.probabilities[rank] =
        std::exp(logits[order[rank]] - logits[order[0]]) / denominator;
  }
  return result;
}

void ExpectNear(const TopThreeTokens& actual, const TopThreeTokens& expected) {
  for (int rank = 0; rank < 3; ++rank) {
    EXPECT_EQ(actual.tokens[rank], expected.tokens[rank]);
    if (expected.tokens[rank] == -1)
      EXPECT_TRUE(std::isnan(actual.probabilities[rank]));
    else
      EXPECT_NEAR(actual.probabilities[rank], expected.probabilities[rank],
                  3e-6 * expected.probabilities[rank] + 2e-8);
  }
}

// CPU bit conversion avoids requiring CUDA-specific scalar types in a normal
// C++ test binary. Round finite values to BF16 using round-to-nearest/even.
uint16_t Bf16Bits(float value) {
  uint32_t bits = std::bit_cast<uint32_t>(value);
  if ((bits & 0x7f800000) == 0x7f800000)
    return static_cast<uint16_t>((bits >> 16) |
                                 ((bits & 0x7fffff) != 0 ? 0x40 : 0));
  bits += 0x7fff + ((bits >> 16) & 1);
  return static_cast<uint16_t>(bits >> 16);
}

float Bf16Value(uint16_t bits) {
  return std::bit_cast<float>(static_cast<uint32_t>(bits) << 16);
}

class VocabularyReadoutTest : public testing::Test {
 protected:
  void SetUp() override {
    auto executor = cuda::Executor::Create();
    ASSERT_TRUE(executor.ok()) << executor.status();
    executor_ = std::move(*executor);
  }

  void TearDown() override {
    if (executor_ != nullptr)
      EXPECT_TRUE(executor_->Synchronize().ok());
  }

  template <class T>
  absl::StatusOr<Buffer> Upload(const std::vector<T>& values) {
    ASSIGN_OR_RETURN(auto staging,
                     cuda::PageLockedHostArray<T>::CopyFrom(*executor_, values));
    ASSIGN_OR_RETURN(auto device,
                     Buffer::Allocate(*executor_, staging.size_bytes()));
    RETURN_IF_ERROR(cuda::CudaStatus(
        cudaMemcpyAsync(device.data(), staging.data(), staging.size_bytes(),
                        cudaMemcpyHostToDevice, executor_->stream()),
        "upload readout fixture"));
    return device;
  }

  absl::StatusOr<cuda::PageLockedHostArray<TopThreeTokens>> Download(
      const Buffer& result) {
    ASSIGN_OR_RETURN(
        auto host,
        cuda::PageLockedHostArray<TopThreeTokens>::Allocate(
            *executor_, result.size_bytes() / sizeof(TopThreeTokens)));
    RETURN_IF_ERROR(cuda::CudaStatus(
        cudaMemcpyAsync(host.data(), result.data(), result.size_bytes(),
                        cudaMemcpyDeviceToHost, executor_->stream()),
        "download readout fixture"));
    RETURN_IF_ERROR(executor_->Synchronize());
    return host;
  }

  std::unique_ptr<cuda::Executor> executor_;
};

TEST_F(VocabularyReadoutTest,
       FullVocabularySoftmaxAndTiesIgnorePaddingAndTail) {
  constexpr int kVocab = 4;
  constexpr int kStride = 7;
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const std::vector<float> values{0,
                                  std::log(2.0f),
                                  std::log(3.0f),
                                  std::log(4.0f),
                                  nan,
                                  1000,
                                  nan,
                                  9,
                                  9,
                                  9,
                                  9,
                                  nan,
                                  nan,
                                  nan,
                                  nan,
                                  nan,
                                  nan,
                                  nan,
                                  nan,
                                  nan,
                                  nan};
  auto logits = Upload(values);
  ASSERT_TRUE(logits.ok()) << logits.status();
  auto output = ReadTopThreeTokens(*executor_, *logits, 2, kVocab, kStride);
  ASSERT_TRUE(output.ok()) << output.status();
  EXPECT_EQ(&output->executor(), executor_.get());
  EXPECT_EQ(output->size_bytes(), 2 * sizeof(TopThreeTokens));
  auto host = Download(*output);
  ASSERT_TRUE(host.ok()) << host.status();
  for (int row = 0; row < 2; ++row) {
    std::vector<double> logical(values.begin() + row * kStride,
                                values.begin() + row * kStride + kVocab);
    ExpectNear((*host)[row], ReferenceSoftmax(logical));
  }
  EXPECT_EQ((*host)[0].tokens[0], 3);
  EXPECT_NEAR((*host)[0].probabilities[0], 0.4f, 1e-7f);
  EXPECT_EQ((*host)[1].tokens[0], 0);
  EXPECT_EQ((*host)[1].tokens[1], 1);
  EXPECT_EQ((*host)[1].tokens[2], 2);
  EXPECT_FLOAT_EQ((*host)[1].probabilities[0], 0.25f);
}

TEST_F(VocabularyReadoutTest, ReductionSpansThreadsAndUsesStableLargeLogits) {
  constexpr int kVocab = 773;
  std::vector<float> logits(2 * kVocab);
  for (int row = 0; row < 2; ++row)
    for (int token = 0; token < kVocab; ++token)
      logits[row * kVocab + token] =
          (row == 0 ? 1000.0f : -1000.0f) - (token % 11) * 0.125f;
  auto device = Upload(logits);
  ASSERT_TRUE(device.ok()) << device.status();
  auto output = ReadTopThreeTokens(*executor_, *device, 2, kVocab, kVocab);
  ASSERT_TRUE(output.ok()) << output.status();
  auto host = Download(*output);
  ASSERT_TRUE(host.ok()) << host.status();
  for (int row = 0; row < 2; ++row) {
    std::vector<double> values(logits.begin() + row * kVocab,
                               logits.begin() + (row + 1) * kVocab);
    ExpectNear((*host)[row], ReferenceSoftmax(values));
  }
}

TEST_F(VocabularyReadoutTest, NonfiniteLogicalLogitInvalidatesOnlyItsRow) {
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const float inf = std::numeric_limits<float>::infinity();
  const std::vector<float> values{nan, 0, 1,    2, 0, inf, 1, 2,
                                  0,   1, -inf, 2, 0, 1,   2, 3};
  auto input = Upload(values);
  ASSERT_TRUE(input.ok()) << input.status();
  auto output = ReadTopThreeTokens(*executor_, *input, 4, 4, 4);
  ASSERT_TRUE(output.ok()) << output.status();
  auto host = Download(*output);
  ASSERT_TRUE(host.ok()) << host.status();
  for (int row = 0; row < 4; ++row) {
    std::vector<double> logical(values.begin() + row * 4,
                                values.begin() + row * 4 + 4);
    ExpectNear((*host)[row], ReferenceSoftmax(logical));
  }
}

TEST_F(VocabularyReadoutTest,
       EmbeddingProjectionMatchesScalarOracleAcrossChunks) {
  constexpr int kRows = 35;
  constexpr int kVocab = 19;
  constexpr int kWidth = 33;
  const float nan = std::numeric_limits<float>::quiet_NaN();
  std::vector<float> embedding(32 * kWidth, nan);
  for (int token = 0; token < kVocab; ++token)
    for (int column = 0; column < kWidth; ++column)
      embedding[token * kWidth + column] =
          (static_cast<int>((token * 13 + column * 7) % 37) - 18) / 32.0f;
  // The last physical activation row is outside the requested prefix.
  std::vector<float> activations((kRows + 1) * kWidth, nan);
  for (int row = 0; row < kRows; ++row)
    for (int column = 0; column < kWidth; ++column)
      activations[row * kWidth + column] =
          0.17f * std::sin(row * 0.71f + column * 0.31f);
  // An exactly zero row produces a full-vocabulary tie, not a top-three-only
  // normalization. A duplicate across chunks must be bit-identical.
  std::fill_n(activations.begin(), kWidth, 0.0f);
  std::copy_n(activations.begin() + kWidth, kWidth,
              activations.begin() + (kRows - 1) * kWidth);
  auto device_embedding = Upload(embedding);
  ASSERT_TRUE(device_embedding.ok()) << device_embedding.status();

  for (DataType storage : {DataType::FP32, DataType::BF16}) {
    SCOPED_TRACE(static_cast<int>(storage));
    std::vector<float> decoded = activations;
    absl::StatusOr<Buffer> input = absl::InternalError("unset test input");
    if (storage == DataType::BF16) {
      std::vector<uint16_t> bits(activations.size());
      for (size_t i = 0; i < bits.size(); ++i) {
        bits[i] = Bf16Bits(activations[i]);
        decoded[i] = Bf16Value(bits[i]);
      }
      input = Upload(bits);
    } else {
      input = Upload(activations);
    }
    ASSERT_TRUE(input.ok()) << input.status();
    auto output = ReadEmbeddingNeighbors(
        *executor_, *input, storage, *device_embedding, kRows, kVocab, kWidth);
    ASSERT_TRUE(output.ok()) << output.status();
    EXPECT_EQ(output->size_bytes(), kRows * sizeof(TopThreeTokens));
    auto host = Download(*output);
    ASSERT_TRUE(host.ok()) << host.status();
    for (int row = 0; row < kRows; ++row) {
      SCOPED_TRACE(row);
      std::vector<double> logits(kVocab, 0);
      for (int token = 0; token < kVocab; ++token)
        for (int column = 0; column < kWidth; ++column)
          logits[token] += static_cast<double>(decoded[row * kWidth + column]) *
                           embedding[token * kWidth + column];
      ExpectNear((*host)[row], ReferenceSoftmax(logits));
    }
    EXPECT_EQ(
        std::memcmp(&(*host)[1], &(*host)[kRows - 1], sizeof(TopThreeTokens)),
        0);
    auto repeated = ReadEmbeddingNeighbors(
        *executor_, *input, storage, *device_embedding, kRows, kVocab, kWidth);
    ASSERT_TRUE(repeated.ok()) << repeated.status();
    auto repeated_host = Download(*repeated);
    ASSERT_TRUE(repeated_host.ok()) << repeated_host.status();
    EXPECT_EQ(
        std::memcmp(host->data(), repeated_host->data(), host->size_bytes()),
        0);
  }
}

TEST_F(VocabularyReadoutTest, EmbeddingReadoutAddsNoNormalizationOrScaling) {
  const std::vector<float> embedding{1, 0, 0, 2, -1, 0, 0, -1};
  auto table = Upload(embedding);
  auto input = Upload<float>({2, 1});
  ASSERT_TRUE(table.ok()) << table.status();
  ASSERT_TRUE(input.ok()) << input.status();
  auto output = ReadEmbeddingNeighbors(*executor_, *input, DataType::FP32,
                                       *table, 1, 4, 2);
  ASSERT_TRUE(output.ok()) << output.status();
  auto host = Download(*output);
  ASSERT_TRUE(host.ok()) << host.status();
  ExpectNear((*host)[0], ReferenceSoftmax(std::vector<double>{2, 2, -2, -1}));
}

TEST_F(VocabularyReadoutTest, NonfiniteEmbeddingProjectionUsesInvalidSentinel) {
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const float inf = std::numeric_limits<float>::infinity();
  auto table = Upload<float>({1, 0, 0, 1, -1, -1});
  ASSERT_TRUE(table.ok()) << table.status();
  for (DataType storage : {DataType::FP32, DataType::BF16}) {
    const std::vector<float> values{1, 2, nan, 0, inf, 1};
    absl::StatusOr<Buffer> input = absl::InternalError("unset test input");
    if (storage == DataType::FP32)
      input = Upload(values);
    else {
      std::vector<uint16_t> bits;
      for (float value : values)
        bits.push_back(Bf16Bits(value));
      input = Upload(bits);
    }
    ASSERT_TRUE(input.ok()) << input.status();
    auto output =
        ReadEmbeddingNeighbors(*executor_, *input, storage, *table, 3, 3, 2);
    ASSERT_TRUE(output.ok()) << output.status();
    auto host = Download(*output);
    ASSERT_TRUE(host.ok()) << host.status();
    EXPECT_GE((*host)[0].tokens[0], 0);
    for (int row : {1, 2})
      for (int rank = 0; rank < 3; ++rank) {
        EXPECT_EQ((*host)[row].tokens[rank], -1);
        EXPECT_TRUE(std::isnan((*host)[row].probabilities[rank]));
      }
  }

  auto invalid_table = Upload<float>({1, 0, 0, nan, -1, -1});
  auto finite_input = Upload<float>({1, 2});
  ASSERT_TRUE(invalid_table.ok()) << invalid_table.status();
  ASSERT_TRUE(finite_input.ok()) << finite_input.status();
  auto output = ReadEmbeddingNeighbors(*executor_, *finite_input,
                                       DataType::FP32, *invalid_table, 1, 3, 2);
  ASSERT_TRUE(output.ok()) << output.status();
  auto host = Download(*output);
  ASSERT_TRUE(host.ok()) << host.status();
  EXPECT_EQ((*host)[0].tokens[0], -1);
  EXPECT_TRUE(std::isnan((*host)[0].probabilities[0]));
}

TEST_F(VocabularyReadoutTest, RejectsInvalidDimensionsSizesAndPhysicalTypes) {
  auto input = Upload<float>(std::vector<float>(8, 0));
  auto table = Upload<float>(std::vector<float>(8, 0));
  ASSERT_TRUE(input.ok()) << input.status();
  ASSERT_TRUE(table.ok()) << table.status();
  struct TopArguments {
    int rows;
    int vocab;
    int stride;
  };
  for (const auto args :
       {TopArguments{0, 4, 4}, TopArguments{-1, 4, 4}, TopArguments{1, 2, 4},
        TopArguments{1, 4, 3}, TopArguments{3, 4, 4}, TopArguments{1, 4, 5},
        TopArguments{std::numeric_limits<int>::max(),
                     std::numeric_limits<int>::max(),
                     std::numeric_limits<int>::max()}})
    EXPECT_EQ(ReadTopThreeTokens(*executor_, *input, args.rows, args.vocab,
                                 args.stride)
                  .status()
                  .code(),
              absl::StatusCode::kInvalidArgument);
  struct EmbeddingArguments {
    int rows;
    int vocab;
    int width;
  };
  for (const auto args :
       {EmbeddingArguments{0, 4, 2}, EmbeddingArguments{1, 2, 2},
        EmbeddingArguments{1, 4, 0}, EmbeddingArguments{1, 4, -1},
        EmbeddingArguments{5, 4, 2}, EmbeddingArguments{1, 5, 2},
        EmbeddingArguments{1, 3, 3},
        EmbeddingArguments{std::numeric_limits<int>::max(),
                           std::numeric_limits<int>::max(),
                           std::numeric_limits<int>::max()}})
    EXPECT_EQ(ReadEmbeddingNeighbors(*executor_, *input, DataType::FP32, *table,
                                     args.rows, args.vocab, args.width)
                  .status()
                  .code(),
              absl::StatusCode::kInvalidArgument);
  for (DataType unsupported : {DataType::FP16, DataType::FP8, DataType::INT32})
    EXPECT_EQ(
        ReadEmbeddingNeighbors(*executor_, *input, unsupported, *table, 1, 4, 2)
            .status()
            .code(),
        absl::StatusCode::kUnimplemented);
}

TEST_F(VocabularyReadoutTest, RejectsBuffersFromAnotherExecutor) {
  auto other = cuda::Executor::Create();
  ASSERT_TRUE(other.ok()) << other.status();
  auto input = Buffer::Allocate(*executor_, 8 * sizeof(float));
  auto table = Buffer::Allocate(*executor_, 8 * sizeof(float));
  auto foreign = Buffer::Allocate(**other, 8 * sizeof(float));
  ASSERT_TRUE(input.ok()) << input.status();
  ASSERT_TRUE(table.ok()) << table.status();
  ASSERT_TRUE(foreign.ok()) << foreign.status();
  EXPECT_EQ(ReadTopThreeTokens(*executor_, *foreign, 1, 4, 4).status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(ReadEmbeddingNeighbors(*executor_, *foreign, DataType::FP32, *table,
                                   1, 4, 2)
                .status()
                .code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(ReadEmbeddingNeighbors(*executor_, *input, DataType::FP32, *foreign,
                                   1, 4, 2)
                .status()
                .code(),
            absl::StatusCode::kInvalidArgument);
}

}  // namespace
}  // namespace pluto::llm
