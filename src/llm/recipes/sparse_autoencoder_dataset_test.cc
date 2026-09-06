#include "src/llm/recipes/sparse_autoencoder_dataset.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "gtest/gtest.h"
#include "src/cuda/buffer.h"
#include "src/cuda/executor.h"
#include "src/dataset/dataset.h"
#include "src/llm/checkpoint.h"
#include "src/llm/layer.h"
#include "src/llm/layers/embedding.h"

namespace pluto::llm {
namespace {

constexpr int kBatchSize = 16;
constexpr int kVocabularySize = 32;
constexpr int kEmbeddingDimension = 16;

class FixedTokenDataSetIterator final : public DataSetIterator {
 public:
  explicit FixedTokenDataSetIterator(Buffer tokens)
      : tokens_(std::move(tokens)) {}

  absl::StatusOr<TokenBatch> Next() override {
    ++next_calls_;
    return TokenBatch{
        .tokens = tokens_,
        .targets = tokens_,
        .batch_size = kBatchSize,
    };
  }

  absl::Status Reset() override {
    ++reset_calls_;
    return absl::OkStatus();
  }

  int next_calls() const { return next_calls_; }
  int reset_calls() const { return reset_calls_; }

 private:
  Buffer tokens_;
  int next_calls_ = 0;
  int reset_calls_ = 0;
};

class SparseAutoEncoderDataSetTest : public testing::Test {
 protected:
  void SetUp() override {
    auto executor = cuda::Executor::Create();
    ASSERT_TRUE(executor.ok()) << executor.status();
    executor_ = std::move(*executor);
  }

  void TearDown() override {
    if (executor_ == nullptr) return;
    EXPECT_TRUE(executor_->Synchronize().ok());
    executor_.reset();
  }

  absl::StatusOr<Buffer> MakeTokens() {
    std::vector<int> tokens(kBatchSize);
    std::iota(tokens.begin(), tokens.end(), 0);
    auto buffer = Buffer::Allocate(*executor_, tokens.size() * sizeof(int));
    if (!buffer.ok()) return buffer.status();
    const cudaError_t error =
        cudaMemcpyAsync(buffer->data(), tokens.data(), buffer->size_bytes(),
                        cudaMemcpyHostToDevice, executor_->stream());
    if (error != cudaSuccess) {
      return cuda::CudaStatus(error, "cudaMemcpyAsync(test tokens)");
    }
    const absl::Status sync = executor_->Synchronize();
    if (!sync.ok()) return sync;
    return *buffer;
  }

  std::unique_ptr<cuda::Executor> executor_;
};

TEST_F(SparseAutoEncoderDataSetTest,
       LoadsPrefixCheckpointAndGeneratesActivationsLazily) {
  auto checkpoint_embedding = EmbeddingLookupLayer::Create(
      *executor_, kVocabularySize, kEmbeddingDimension, DataType::FP16);
  ASSERT_TRUE(checkpoint_embedding.ok()) << checkpoint_embedding.status();
  std::vector<float> table(kVocabularySize * kEmbeddingDimension);
  for (int token = 0; token < kVocabularySize; ++token) {
    for (int column = 0; column < kEmbeddingDimension; ++column) {
      table[token * kEmbeddingDimension + column] =
          static_cast<float>(token * 100 + column);
    }
  }
  ASSERT_EQ(cudaMemcpyAsync((*checkpoint_embedding)->weight().data(),
                            table.data(), table.size() * sizeof(float),
                            cudaMemcpyHostToDevice, executor_->stream()),
            cudaSuccess);

  const std::filesystem::path checkpoint =
      std::filesystem::path(testing::TempDir()) / "sae-activation-checkpoint";
  ASSERT_TRUE(
      WriteToDirectory(*executor_, **checkpoint_embedding, checkpoint).ok());
  // Simulate later weights from the complete model. The activation generator
  // needs only the valid weight_0.bin prefix.
  std::ofstream(checkpoint / "weight_1.bin", std::ios::binary) << "unused";

  auto activation_generator = EmbeddingLookupLayer::Create(
      *executor_, kVocabularySize, kEmbeddingDimension, DataType::FP16);
  ASSERT_TRUE(activation_generator.ok()) << activation_generator.status();
  auto tokens = MakeTokens();
  ASSERT_TRUE(tokens.ok()) << tokens.status();
  FixedTokenDataSetIterator source(*tokens);

  auto dataset = SparseAutoEncoderDataSetIterator::Create(
      *executor_, **activation_generator, source, checkpoint);
  ASSERT_TRUE(dataset.ok()) << dataset.status();
  EXPECT_EQ(source.next_calls(), 0);

  auto batch = (*dataset)->Next();
  ASSERT_TRUE(batch.ok()) << batch.status();
  EXPECT_EQ(source.next_calls(), 1);
  EXPECT_EQ(batch->batch_size, kBatchSize);
  EXPECT_EQ(
      batch->activations.size_bytes(),
      static_cast<size_t>(kBatchSize) * kEmbeddingDimension * sizeof(float));

  std::vector<float> actual(kBatchSize * kEmbeddingDimension);
  ASSERT_EQ(cudaMemcpyAsync(actual.data(), batch->activations.data(),
                            batch->activations.size_bytes(),
                            cudaMemcpyDeviceToHost, executor_->stream()),
            cudaSuccess);
  ASSERT_TRUE(executor_->Synchronize().ok());
  for (int row = 0; row < kBatchSize; ++row) {
    for (int column = 0; column < kEmbeddingDimension; ++column) {
      EXPECT_FLOAT_EQ(actual[row * kEmbeddingDimension + column],
                      table[row * kEmbeddingDimension + column]);
    }
  }

  EXPECT_TRUE((*dataset)->Reset().ok());
  EXPECT_EQ(source.reset_calls(), 1);
}

TEST_F(SparseAutoEncoderDataSetTest,
       RejectsMalformedCheckpointBeforeReadingSource) {
  const std::filesystem::path checkpoint =
      std::filesystem::path(testing::TempDir()) / "sae-malformed-checkpoint";
  ASSERT_TRUE(std::filesystem::create_directories(checkpoint));
  std::ofstream(checkpoint / "weight_0.bin", std::ios::binary) << "too short";

  auto activation_generator = EmbeddingLookupLayer::Create(
      *executor_, kVocabularySize, kEmbeddingDimension, DataType::FP16);
  ASSERT_TRUE(activation_generator.ok()) << activation_generator.status();
  auto tokens = MakeTokens();
  ASSERT_TRUE(tokens.ok()) << tokens.status();
  FixedTokenDataSetIterator source(*tokens);

  auto dataset = SparseAutoEncoderDataSetIterator::Create(
      *executor_, **activation_generator, source, checkpoint);
  EXPECT_EQ(dataset.status().code(), absl::StatusCode::kDataLoss);
  EXPECT_EQ(source.next_calls(), 0);
}

}  // namespace
}  // namespace pluto::llm
