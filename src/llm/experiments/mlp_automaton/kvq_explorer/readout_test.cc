#include "src/llm/experiments/mlp_automaton/kvq_explorer/readout.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

#include "absl/types/span.h"
#include "gtest/gtest.h"
#include "src/util/status_macros.h"

namespace pluto::llm::kvq_explorer {
namespace {

// Non-tiled width exercises the warp tail; the physical vocabulary has 32
// rows, so 13 large padding embeddings must never enter the readout softmax.
constexpr Dimensions kDimensions{19, 33, 8};
constexpr int kWidth = kDimensions.model_width;
constexpr std::array<int, 8> kMatrices{4, 16, 28, 40, 52, 64, 76, 88};
constexpr std::array<int, 8> kBiases{5, 17, 29, 41, 53, 65, 77, 89};

TopThree ReferenceSoftmax(const std::vector<double>& logits) {
  std::vector<int> order(logits.size());
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](int a, int b) {
    return logits[a] == logits[b] ? a < b : logits[a] > logits[b];
  });
  double denominator = 0;
  for (double value : logits)
    denominator += std::exp(value - logits[order[0]]);
  TopThree result{};
  for (int rank = 0; rank < 3; ++rank) {
    result.tokens[rank] = order[rank];
    result.probabilities[rank] =
        std::exp(logits[order[rank]] - logits[order[0]]) / denominator;
  }
  return result;
}

void ExpectNear(const TopThree& actual, const TopThree& expected) {
  for (int rank = 0; rank < 3; ++rank) {
    EXPECT_EQ(actual.tokens[rank], expected.tokens[rank]);
    EXPECT_NEAR(actual.probabilities[rank], expected.probabilities[rank],
                3e-6f * expected.probabilities[rank] + 1e-8f);
  }
}

class KvqReadoutTest : public testing::Test {
 protected:
  void SetUp() override {
    auto executor = cuda::Executor::Create();
    ASSERT_TRUE(executor.ok()) << executor.status();
    executor_ = std::move(*executor);
    std::string pattern =
        (std::filesystem::path(testing::TempDir()) / "kvq-explorer-XXXXXX")
            .string();
    std::vector<char> name(pattern.begin(), pattern.end());
    name.push_back('\0');
    const char* created = mkdtemp(name.data());
    ASSERT_NE(created, nullptr);
    directory_ = created;

    embedding_.resize(32 * kWidth, 1000);
    for (int token = 0; token < kDimensions.vocab_size; ++token)
      for (int column = 0; column < kWidth; ++column)
        embedding_[token * kWidth + column] =
            0.4f * std::sin(0.71f * token + 0.11f * column) +
            0.15f * std::cos(0.39f * token - 0.7f * column);
    for (int block = 0; block < 8; ++block) {
      matrices_[block].resize(kWidth * 3 * kWidth);
      biases_[block].resize(3 * kWidth);
      for (size_t i = 0; i < matrices_[block].size(); ++i)
        matrices_[block][i] = 0.17f * std::sin(i * 0.317f + block * 0.831f);
      for (size_t i = 0; i < biases_[block].size(); ++i)
        biases_[block][i] = 0.13f * std::cos(i * 0.793f + block * 0.213f);
    }
    WriteCheckpoint();
  }

  void TearDown() override {
    if (executor_)
      EXPECT_TRUE(executor_->Synchronize().ok());
    if (!directory_.empty()) {
      std::error_code error;
      std::filesystem::remove_all(directory_, error);
      EXPECT_FALSE(error) << error.message();
    }
  }

  void Write(int index, absl::Span<const float> values) {
    std::ofstream output(
        directory_ / ("weight_" + std::to_string(index) + ".bin"),
        std::ios::binary);
    output.write(reinterpret_cast<const char*>(values.data()),
                 values.size() * sizeof(float));
    output.close();
    ASSERT_TRUE(output.good());
  }

  void WriteCheckpoint() {
    Write(0, embedding_);
    for (int block = 0; block < 8; ++block) {
      Write(kMatrices[block], matrices_[block]);
      Write(kBiases[block], biases_[block]);
    }
  }

  // Scalar, double-precision specification: first form every projected
  // coordinate with the checkpoint's [input, Q/K/V, output] layout, then
  // take a dot product with every logical E row. No GPU helpers are used.
  TopThree Reference(int token, int block, int projection) {
    std::vector<double> projected(kWidth);
    for (int output = 0; output < kWidth; ++output) {
      projected[output] = biases_[block][projection * kWidth + output];
      for (int input = 0; input < kWidth; ++input)
        projected[output] +=
            static_cast<double>(embedding_[token * kWidth + input]) *
            matrices_[block][(input * 3 + projection) * kWidth + output];
    }
    std::vector<double> logits(kDimensions.vocab_size, 0);
    for (int target = 0; target < kDimensions.vocab_size; ++target)
      for (int column = 0; column < kWidth; ++column)
        logits[target] +=
            projected[column] * embedding_[target * kWidth + column];
    return ReferenceSoftmax(logits);
  }

  absl::StatusOr<cuda::PageLockedHostArray<TopThree>> Reduce(
      absl::Span<const float> values, int rows, int vocab) {
    ASSIGN_OR_RETURN(auto input, cuda::PageLockedHostArray<float>::CopyFrom(
                                     *executor_, values));
    ASSIGN_OR_RETURN(auto device,
                     cuda::Buffer::Allocate(*executor_, input.size_bytes()));
    RETURN_IF_ERROR(cuda::CudaStatus(
        cudaMemcpyAsync(device.data(), input.data(), input.size_bytes(),
                        cudaMemcpyHostToDevice, executor_->stream()),
        "test upload"));
    ASSIGN_OR_RETURN(auto output, ReadTopThree(*executor_, device, rows, vocab));
    ASSIGN_OR_RETURN(auto result, cuda::PageLockedHostArray<TopThree>::Allocate(
                                      *executor_, rows));
    RETURN_IF_ERROR(cuda::CudaStatus(
        cudaMemcpyAsync(result.data(), output.data(), output.size_bytes(),
                        cudaMemcpyDeviceToHost, executor_->stream()),
        "test download"));
    RETURN_IF_ERROR(executor_->Synchronize());
    return result;
  }

  std::unique_ptr<cuda::Executor> executor_;
  std::filesystem::path directory_;
  std::vector<float> embedding_;
  std::array<std::vector<float>, 8> matrices_;
  std::array<std::vector<float>, 8> biases_;
};

TEST_F(KvqReadoutTest, AllBlocksAndSlicesMatchScalarCpuIncludingChunkTail) {
  auto model = Readout::Load(*executor_, directory_, kDimensions);
  ASSERT_TRUE(model.ok()) << model.status();
  auto tokens = cuda::PageLockedHostArray<int>::Allocate(*executor_, 35);
  ASSERT_TRUE(tokens.ok()) << tokens.status();
  for (size_t row = 0; row < tokens->size(); ++row)
    (*tokens)[row] = (row * 7 + 3) % kDimensions.vocab_size;
  (*tokens)[0] = (*tokens)[34] = kDimensions.vocab_size - 1;
  auto output = (*model)->Explore(*executor_, *tokens);
  ASSERT_TRUE(output.ok()) << output.status();
  ASSERT_EQ(output->size(), 35 * 8 * 3);
  EXPECT_EQ(&output->executor(), executor_.get());
  for (size_t row = 0; row < tokens->size(); ++row)
    for (int block = 0; block < 8; ++block)
      for (int projection = 0; projection < 3; ++projection) {
        SCOPED_TRACE(testing::Message()
                     << row << '/' << block << '/' << projection);
        ExpectNear((*output)[(row * 8 + block) * 3 + projection],
                   Reference((*tokens)[row], block, projection));
      }
  // The diagnostic is independent of position and other tokens. A duplicate
  // across chunk boundaries must produce exactly the same values.
  EXPECT_EQ(std::memcmp(output->data(), output->data() + 34 * 8 * 3,
                        8 * 3 * sizeof(TopThree)),
            0);
  auto repeated = (*model)->Explore(*executor_, *tokens);
  ASSERT_TRUE(repeated.ok()) << repeated.status();
  EXPECT_EQ(std::memcmp(output->data(), repeated->data(), output->size_bytes()),
            0);
}

TEST_F(KvqReadoutTest, BiasesMatterAndUnneededCheckpointFilesAreIgnored) {
  for (auto& matrix : matrices_)
    std::fill(matrix.begin(), matrix.end(), 0);
  WriteCheckpoint();
  Write(1, {42});  // Invalid unused position tensor, deliberately ignored.
  // No attention LayerNorm, output projection, MLP or final norm files exist.
  ASSERT_FALSE(std::filesystem::exists(directory_ / "weight_2.bin"));
  auto model = Readout::Load(*executor_, directory_, kDimensions);
  ASSERT_TRUE(model.ok()) << model.status();
  auto tokens = cuda::PageLockedHostArray<int>::CopyFrom(*executor_, {0, 18});
  ASSERT_TRUE(tokens.ok()) << tokens.status();
  auto output = (*model)->Explore(*executor_, *tokens);
  ASSERT_TRUE(output.ok()) << output.status();
  for (int block = 0; block < 8; ++block)
    for (int projection = 0; projection < 3; ++projection) {
      ExpectNear((*output)[block * 3 + projection],
                 Reference(0, block, projection));
      ExpectNear((*output)[(8 + block) * 3 + projection],
                 Reference(18, block, projection));
    }
}

TEST_F(KvqReadoutTest, TopThreeUsesFullVocabularyAndStableTieBreaking) {
  constexpr int vocab = 50257;
  constexpr int rows = 4;
  std::vector<float> logits(rows * vocab);
  for (int row = 0; row < rows; ++row)
    for (int token = 0; token < vocab; ++token)
      logits[row * vocab + token] = ((token * 17 + row * 31) % 997) * 0.01f - 8;
  logits[3] = logits[256] = logits[49155] = 12;  // Cross-thread ties.
  std::fill(logits.begin() + vocab, logits.begin() + 2 * vocab, 10000);
  logits[3 * vocab - 1] = 25;  // Last logical token.
  logits[3 * vocab] = 3e38f;
  logits[3 * vocab + 1] = 0;
  logits[3 * vocab + 2] = -3e38f;
  auto output = Reduce(logits, rows, vocab);
  ASSERT_TRUE(output.ok()) << output.status();
  for (int row = 0; row < rows; ++row) {
    const std::vector<double> expected(logits.begin() + row * vocab,
                                       logits.begin() + (row + 1) * vocab);
    ExpectNear((*output)[row], ReferenceSoftmax(expected));
  }
  EXPECT_FLOAT_EQ((*output)[1].probabilities[0], 1.0f / vocab);
  EXPECT_LT((*output)[1].probabilities[0] + (*output)[1].probabilities[1] +
                (*output)[1].probabilities[2],
            0.001f);
  EXPECT_EQ((*output)[0].tokens[0], 3);
  EXPECT_EQ((*output)[0].tokens[1], 256);
  EXPECT_EQ((*output)[0].tokens[2], 49155);
}

TEST_F(KvqReadoutTest, ThreeElementRowsAndNonfiniteRows) {
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const float inf = std::numeric_limits<float>::infinity();
  const std::vector<float> logits{2, 2, 1, nan, 1, 0, 0, inf, 1, 0, 1, -inf};
  auto output = Reduce(logits, 4, 3);
  ASSERT_TRUE(output.ok()) << output.status();
  ExpectNear((*output)[0], ReferenceSoftmax({2, 2, 1}));
  for (int row = 1; row < 4; ++row)
    for (int rank = 0; rank < 3; ++rank) {
      EXPECT_EQ((*output)[row].tokens[rank], -1);
      EXPECT_TRUE(std::isnan((*output)[row].probabilities[rank]));
    }
}

TEST_F(KvqReadoutTest, RejectsMissingWrongSizedAndNonfiniteWeights) {
  const auto missing = directory_ / "weight_89.bin";
  ASSERT_TRUE(std::filesystem::remove(missing));
  EXPECT_EQ(Readout::Load(*executor_, directory_, kDimensions).status().code(),
            absl::StatusCode::kNotFound);
  Write(89, {0});
  EXPECT_EQ(Readout::Load(*executor_, directory_, kDimensions).status().code(),
            absl::StatusCode::kDataLoss);
  auto oversized = biases_.back();
  oversized.push_back(0);
  Write(89, oversized);
  EXPECT_EQ(Readout::Load(*executor_, directory_, kDimensions).status().code(),
            absl::StatusCode::kDataLoss);
  Write(89, biases_.back());
  for (float value : {std::numeric_limits<float>::infinity(),
                      std::numeric_limits<float>::quiet_NaN()}) {
    auto invalid = matrices_[0];
    invalid.back() = value;
    Write(4, invalid);
    EXPECT_EQ(
        Readout::Load(*executor_, directory_, kDimensions).status().code(),
        absl::StatusCode::kDataLoss);
  }
}

TEST_F(KvqReadoutTest, RejectsBadDimensionsTokenIdsAndExecutors) {
  EXPECT_FALSE(Readout::Load(*executor_, "", kDimensions).ok());
  for (const Dimensions invalid :
       {Dimensions{2, 33, 8}, Dimensions{19, 0, 8}, Dimensions{19, -1, 8},
        Dimensions{19, 33, 0}, Dimensions{19, 33, 9},
        Dimensions{std::numeric_limits<int>::max(), 33, 8},
        Dimensions{19, std::numeric_limits<int>::max(), 8}})
    EXPECT_EQ(Readout::Load(*executor_, directory_, invalid).status().code(),
              absl::StatusCode::kInvalidArgument);
  auto model = Readout::Load(*executor_, directory_, kDimensions);
  ASSERT_TRUE(model.ok()) << model.status();
  auto empty = cuda::PageLockedHostArray<int>::Allocate(*executor_, 0);
  ASSERT_TRUE(empty.ok()) << empty.status();
  auto output = (*model)->Explore(*executor_, *empty);
  ASSERT_TRUE(output.ok()) << output.status();
  EXPECT_TRUE(output->empty());
  auto tokens = cuda::PageLockedHostArray<int>::CopyFrom(*executor_, {-1});
  ASSERT_TRUE(tokens.ok()) << tokens.status();
  EXPECT_FALSE((*model)->Explore(*executor_, *tokens).ok());
  (*tokens)[0] = 19;  // Padding is not a valid token.
  EXPECT_FALSE((*model)->Explore(*executor_, *tokens).ok());
  (*tokens)[0] = 0;
  auto other = cuda::Executor::Create();
  ASSERT_TRUE(other.ok()) << other.status();
  auto foreign = cuda::PageLockedHostArray<int>::CopyFrom(**other, {0});
  ASSERT_TRUE(foreign.ok()) << foreign.status();
  EXPECT_FALSE((*model)->Explore(*executor_, *foreign).ok());
  EXPECT_FALSE((*model)->Explore(**other, *tokens).ok());
  auto logits = cuda::Buffer::Allocate(*executor_, 3 * sizeof(float));
  ASSERT_TRUE(logits.ok()) << logits.status();
  EXPECT_FALSE(ReadTopThree(*executor_, *logits, 0, 3).ok());
  EXPECT_FALSE(ReadTopThree(*executor_, *logits, 1, 2).ok());
  EXPECT_FALSE(ReadTopThree(*executor_, *logits, 2, 3).ok());
  EXPECT_FALSE(ReadTopThree(**other, *logits, 1, 3).ok());
}

TEST_F(KvqReadoutTest, FiniteWeightsWithOverflowingProjectionAreRejected) {
  std::fill(matrices_[0].begin(), matrices_[0].end(), 3e38f);
  std::fill(embedding_.begin(), embedding_.end(), 3e38f);
  WriteCheckpoint();
  auto model = Readout::Load(*executor_, directory_, kDimensions);
  ASSERT_TRUE(model.ok()) << model.status();
  auto tokens = cuda::PageLockedHostArray<int>::CopyFrom(*executor_, {0});
  ASSERT_TRUE(tokens.ok()) << tokens.status();
  EXPECT_EQ((*model)->Explore(*executor_, *tokens).status().code(),
            absl::StatusCode::kDataLoss);
}

}  // namespace
}  // namespace pluto::llm::kvq_explorer
