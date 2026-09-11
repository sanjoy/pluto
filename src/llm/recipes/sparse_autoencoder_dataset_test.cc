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
#include "src/cuda/page_locked_host_array.h"
#include "src/dataset/dataset.h"
#include "src/llm/checkpoint.h"
#include "src/llm/layer.h"
#include "src/llm/layers/combinators.h"
#include "src/llm/layers/embedding.h"
#include "src/util/status_macros.h"

namespace pluto::llm {
namespace {

constexpr int kBatchSize = 2;
constexpr int kSequenceLength = 8;
constexpr int kTokenCount = kBatchSize * kSequenceLength;
constexpr int kVocabularySize = 32;
constexpr int kEmbeddingDimension = 16;

class FixedTokenDataSetIterator final : public DataSetIterator {
 public:
  explicit FixedTokenDataSetIterator(Buffer data, int batch_size = kBatchSize,
                                     int sequence_length = kSequenceLength)
      : data_(std::move(data)),
        batch_size_(batch_size),
        sequence_length_(sequence_length) {}

  absl::StatusOr<DataBatch> Next() override {
    ++next_calls_;
    return DataBatch{
        .data = data_,
        .batch_size = batch_size_,
        .sequence_length = sequence_length_,
    };
  }

  absl::Status Reset() override {
    ++reset_calls_;
    return absl::OkStatus();
  }

  int next_calls() const { return next_calls_; }
  int reset_calls() const { return reset_calls_; }

 private:
  Buffer data_;
  int batch_size_;
  int sequence_length_;
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
    if (executor_ == nullptr)
      return;
    EXPECT_TRUE(executor_->Synchronize().ok());
    executor_.reset();
  }

  absl::StatusOr<Buffer> MakeData() {
    ASSIGN_OR_RETURN(auto data,
                     cuda::PageLockedHostArray<int>::Allocate(2 * kTokenCount));
    std::iota(data.begin(), data.begin() + kTokenCount, 0);
    std::iota(data.begin() + kTokenCount, data.end(), 1);
    auto buffer = Buffer::Allocate(*executor_, data.size() * sizeof(int));
    if (!buffer.ok())
      return buffer.status();
    const cudaError_t error =
        cudaMemcpyAsync(buffer->data(), data.data(), buffer->size_bytes(),
                        cudaMemcpyHostToDevice, executor_->stream());
    if (error != cudaSuccess)
      return cuda::CudaStatus(error, "cudaMemcpyAsync(test tokens)");
    const absl::Status sync = executor_->Synchronize();
    if (!sync.ok())
      return sync;
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
  auto pinned_table = cuda::PageLockedHostArray<float>::CopyFrom(table);
  ASSERT_TRUE(pinned_table.ok()) << pinned_table.status();
  ASSERT_EQ(cudaMemcpyAsync((*checkpoint_embedding)->weight().data(),
                            pinned_table->data(), table.size() * sizeof(float),
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
  auto data = MakeData();
  ASSERT_TRUE(data.ok()) << data.status();
  FixedTokenDataSetIterator source(*data);

  auto concrete_dataset = SparseAutoEncoderDataSetIterator::Create(
      *executor_, **activation_generator, source, checkpoint);
  ASSERT_TRUE(concrete_dataset.ok()) << concrete_dataset.status();
  std::unique_ptr<DataSetIterator> dataset = std::move(*concrete_dataset);
  EXPECT_EQ(source.next_calls(), 0);

  auto batch = dataset->Next();
  ASSERT_TRUE(batch.ok()) << batch.status();
  EXPECT_EQ(source.next_calls(), 1);
  EXPECT_EQ(batch->batch_size, kBatchSize);
  EXPECT_EQ(batch->sequence_length, kSequenceLength);
  ASSERT_TRUE(batch->token_count().ok());
  EXPECT_EQ(*batch->token_count(), kTokenCount);
  EXPECT_EQ(batch->data.size_bytes(), static_cast<size_t>(kTokenCount) *
                                          kEmbeddingDimension * sizeof(float));

  auto actual = cuda::PageLockedHostArray<float>::Allocate(kTokenCount *
                                                           kEmbeddingDimension);
  ASSERT_TRUE(actual.ok()) << actual.status();
  ASSERT_EQ(cudaMemcpyAsync(actual->data(), batch->data.data(),
                            batch->data.size_bytes(), cudaMemcpyDeviceToHost,
                            executor_->stream()),
            cudaSuccess);
  ASSERT_TRUE(executor_->Synchronize().ok());
  for (int row = 0; row < kTokenCount; ++row) {
    for (int column = 0; column < kEmbeddingDimension; ++column) {
      EXPECT_FLOAT_EQ((*actual)[row * kEmbeddingDimension + column],
                      table[row * kEmbeddingDimension + column]);
    }
  }

  EXPECT_TRUE(dataset->Reset().ok());
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
  auto data = MakeData();
  ASSERT_TRUE(data.ok()) << data.status();
  FixedTokenDataSetIterator source(*data);

  auto dataset = SparseAutoEncoderDataSetIterator::Create(
      *executor_, **activation_generator, source, checkpoint);
  EXPECT_EQ(dataset.status().code(), absl::StatusCode::kDataLoss);
  EXPECT_EQ(source.next_calls(), 0);
}

TEST_F(SparseAutoEncoderDataSetTest,
       RejectsSamplesThatWouldSharePositionBoundaries) {
  auto embedding = EmbeddingLookupLayer::Create(
      *executor_, kVocabularySize, kEmbeddingDimension, DataType::FP16);
  auto positions = PositionEmbeddingLayer::Create(
      *executor_, kTokenCount, kEmbeddingDimension, DataType::FP16);
  ASSERT_TRUE(embedding.ok()) << embedding.status();
  ASSERT_TRUE(positions.ok()) << positions.status();
  ComposedLayerBuilder builder;
  ASSERT_TRUE(builder.add(std::move(*embedding)).ok());
  ASSERT_TRUE(builder.add(std::move(*positions)).ok());
  auto generator = builder.create();
  ASSERT_TRUE(generator.ok()) << generator.status();
  const std::filesystem::path checkpoint =
      std::filesystem::path(testing::TempDir()) / "sae-sequence-checkpoint";
  ASSERT_TRUE(WriteToDirectory(*executor_, **generator, checkpoint).ok());
  auto data = MakeData();
  ASSERT_TRUE(data.ok()) << data.status();

  // Two eight-token samples fit exactly in one sixteen-token model context.
  // A flat buffer-size check alone would accept this and silently give the
  // second sample positions 8..15 instead of resetting positions to zero.
  FixedTokenDataSetIterator shorter_source(*data);
  auto shorter = SparseAutoEncoderDataSetIterator::Create(
      *executor_, **generator, shorter_source, checkpoint);
  ASSERT_TRUE(shorter.ok()) << shorter.status();
  auto invalid = (*shorter)->Next();
  EXPECT_EQ(invalid.status().code(), absl::StatusCode::kInvalidArgument);

  FixedTokenDataSetIterator matching_source(*data, 1, kTokenCount);
  auto matching = SparseAutoEncoderDataSetIterator::Create(
      *executor_, **generator, matching_source, checkpoint);
  ASSERT_TRUE(matching.ok()) << matching.status();
  auto valid = (*matching)->Next();
  ASSERT_TRUE(valid.ok()) << valid.status();
  EXPECT_EQ(valid->batch_size, 1);
  EXPECT_EQ(valid->sequence_length, kTokenCount);
  EXPECT_EQ(valid->data.size_bytes(),
            kTokenCount * kEmbeddingDimension * sizeof(float));
}

}  // namespace
}  // namespace pluto::llm
