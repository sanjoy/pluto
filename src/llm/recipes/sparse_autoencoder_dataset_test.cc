#include "src/llm/recipes/sparse_autoencoder_dataset.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <numeric>
#include <string>
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
  explicit FixedTokenDataSetIterator(DataBatch batch,
                                     int batch_size = kBatchSize,
                                     int sequence_length = kSequenceLength)
      : batch_(std::move(batch)) {
    batch_.batch_size = batch_size;
    batch_.sequence_length = sequence_length;
  }

  absl::StatusOr<DataBatch> Next() override {
    ++next_calls_;
    return batch_;
  }

  absl::Status Reset() override {
    ++reset_calls_;
    return absl::OkStatus();
  }

  int next_calls() const { return next_calls_; }
  int reset_calls() const { return reset_calls_; }

 private:
  DataBatch batch_;
  int next_calls_ = 0;
  int reset_calls_ = 0;
};

// Records allocation identities while allowing arbitrary forward output counts.
class RecordingActivationGenerator final : public Layer {
 public:
  explicit RecordingActivationGenerator(BufferVec outputs)
      : outputs_(std::move(outputs)) {}

  absl::Span<Buffer> weights() override { return {}; }
  DataType output_type() const override { return DataType::FP16; }
  const void* input_address() const { return input_address_; }
  int forward_calls() const { return forward_calls_; }
  int backward_calls() const { return backward_calls_; }

 private:
  absl::StatusOr<FwdResult> fwd_impl(
      cuda::Executor&, absl::Span<const Buffer> inputs) const override {
    ++forward_calls_;
    if (inputs.size() != 1)
      return absl::InvalidArgumentError("expected one generator input");
    input_address_ = inputs.front().data();
    return FwdResult{
        .outputs = outputs_,
        .state = {.intermediates = {inputs.front()}},
    };
  }

  absl::StatusOr<BufferVec> bwd_impl(cuda::Executor&, absl::Span<const Buffer>,
                                     BackwardState) override {
    ++backward_calls_;
    return absl::InternalError("frozen generator must not run backward");
  }

  BufferVec outputs_;
  mutable const void* input_address_ = nullptr;
  mutable int forward_calls_ = 0;
  int backward_calls_ = 0;
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

  absl::StatusOr<DataBatch> MakeData() {
    ASSIGN_OR_RETURN(auto data, cuda::PageLockedHostArray<int>::Allocate(
                                    *executor_, kTokenCount + 1));
    std::iota(data.begin(), data.end(), 0);
    const size_t token_bytes = kTokenCount * sizeof(int);
    ASSIGN_OR_RETURN(auto inputs, Buffer::Allocate(*executor_, token_bytes));
    ASSIGN_OR_RETURN(auto targets, Buffer::Allocate(*executor_, token_bytes));
    RETURN_IF_ERROR(cuda::CudaStatus(
        cudaMemcpyAsync(inputs.data(), data.data(), token_bytes,
                        cudaMemcpyHostToDevice, executor_->stream()),
        "cudaMemcpyAsync(test inputs)"));
    RETURN_IF_ERROR(cuda::CudaStatus(
        cudaMemcpyAsync(targets.data(), data.data() + 1, token_bytes,
                        cudaMemcpyHostToDevice, executor_->stream()),
        "cudaMemcpyAsync(test targets)"));
    RETURN_IF_ERROR(executor_->Synchronize());
    return DataBatch{.inputs = std::move(inputs),
                     .targets = std::move(targets),
                     .batch_size = kBatchSize,
                     .sequence_length = kSequenceLength};
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
  auto pinned_table =
      cuda::PageLockedHostArray<float>::CopyFrom(*executor_, table);
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
  EXPECT_EQ(batch->inputs.data(), batch->targets.data());
  EXPECT_EQ(batch->inputs.size_bytes(), batch->targets.size_bytes());
  EXPECT_EQ(
      batch->inputs.size_bytes(),
      static_cast<size_t>(kTokenCount) * kEmbeddingDimension * sizeof(float));

  auto actual = cuda::PageLockedHostArray<float>::Allocate(
      *executor_, kTokenCount * kEmbeddingDimension);
  ASSERT_TRUE(actual.ok()) << actual.status();
  ASSERT_EQ(cudaMemcpyAsync(actual->data(), batch->inputs.data(),
                            batch->inputs.size_bytes(), cudaMemcpyDeviceToHost,
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
  EXPECT_EQ(valid->inputs.size_bytes(),
            kTokenCount * kEmbeddingDimension * sizeof(float));
}

TEST_F(SparseAutoEncoderDataSetTest,
       ForwardsInputsAndAliasesBothActivationHandlesWithoutCopies) {
  auto source_batch = MakeData();
  ASSERT_TRUE(source_batch.ok()) << source_batch.status();
  auto activations = Buffer::Allocate(
      *executor_, kTokenCount * kEmbeddingDimension * sizeof(float));
  ASSERT_TRUE(activations.ok()) << activations.status();
  RecordingActivationGenerator generator({*activations});
  FixedTokenDataSetIterator source(*source_batch);
  const std::filesystem::path checkpoint =
      std::filesystem::path(testing::TempDir()) / "sae-no-copy-checkpoint";
  ASSERT_TRUE(WriteToDirectory(*executor_, generator, checkpoint).ok());
  auto dataset = SparseAutoEncoderDataSetIterator::Create(*executor_, generator,
                                                          source, checkpoint);
  ASSERT_TRUE(dataset.ok()) << dataset.status();

  auto first = (*dataset)->Next();
  ASSERT_TRUE(first.ok()) << first.status();
  EXPECT_EQ(generator.input_address(), source_batch->inputs.data());
  EXPECT_NE(generator.input_address(), source_batch->targets.data());
  EXPECT_EQ(first->inputs.data(), activations->data());
  EXPECT_EQ(first->targets.data(), activations->data());
  EXPECT_EQ(first->batch_size, kBatchSize);
  EXPECT_EQ(first->sequence_length, kSequenceLength);

  auto second = (*dataset)->Next();
  ASSERT_TRUE(second.ok()) << second.status();
  EXPECT_EQ(second->inputs.data(), first->inputs.data());
  EXPECT_EQ(second->targets.data(), first->targets.data());
  EXPECT_EQ(generator.forward_calls(), 2);
  EXPECT_EQ(generator.backward_calls(), 0);
  dataset->reset();
  EXPECT_EQ(first->inputs.data(), activations->data());
  EXPECT_EQ(first->targets.data(), activations->data());
}

TEST_F(SparseAutoEncoderDataSetTest,
       RejectsMalformedSourceBufferSizesBeforeForward) {
  auto data = MakeData();
  ASSERT_TRUE(data.ok()) << data.status();
  RecordingActivationGenerator generator({data->inputs});
  const std::filesystem::path checkpoint =
      std::filesystem::path(testing::TempDir()) / "sae-source-size-checkpoint";
  ASSERT_TRUE(WriteToDirectory(*executor_, generator, checkpoint).ok());
  for (bool invalid_inputs : {false, true}) {
    for (size_t bytes : {size_t{0}, (kTokenCount - 1) * sizeof(int),
                         2 * kTokenCount * sizeof(int)}) {
      SCOPED_TRACE(testing::Message() << invalid_inputs << " " << bytes);
      auto malformed = Buffer::Allocate(*executor_, bytes);
      ASSERT_TRUE(malformed.ok()) << malformed.status();
      DataBatch batch = *data;
      if (invalid_inputs)
        batch.inputs = *malformed;
      else
        batch.targets = *malformed;
      FixedTokenDataSetIterator source(std::move(batch));
      auto dataset = SparseAutoEncoderDataSetIterator::Create(
          *executor_, generator, source, checkpoint);
      ASSERT_TRUE(dataset.ok()) << dataset.status();
      auto result = (*dataset)->Next();
      EXPECT_EQ(result.status().code(), absl::StatusCode::kInvalidArgument);
    }
  }
  EXPECT_EQ(generator.forward_calls(), 0);
}

TEST_F(SparseAutoEncoderDataSetTest,
       RejectsSourceBuffersFromAnotherExecutorBeforeForward) {
  auto other = cuda::Executor::Create();
  ASSERT_TRUE(other.ok()) << other.status();
  auto foreign = Buffer::Allocate(**other, kTokenCount * sizeof(int));
  ASSERT_TRUE(foreign.ok()) << foreign.status();
  auto data = MakeData();
  ASSERT_TRUE(data.ok()) << data.status();
  RecordingActivationGenerator generator({data->inputs});
  const std::filesystem::path checkpoint =
      std::filesystem::path(testing::TempDir()) /
      "sae-source-executor-checkpoint";
  ASSERT_TRUE(WriteToDirectory(*executor_, generator, checkpoint).ok());
  for (bool foreign_inputs : {false, true}) {
    DataBatch batch = *data;
    if (foreign_inputs)
      batch.inputs = *foreign;
    else
      batch.targets = *foreign;
    FixedTokenDataSetIterator source(std::move(batch));
    auto dataset = SparseAutoEncoderDataSetIterator::Create(
        *executor_, generator, source, checkpoint);
    ASSERT_TRUE(dataset.ok()) << dataset.status();
    auto result = (*dataset)->Next();
    EXPECT_EQ(result.status().code(), absl::StatusCode::kInvalidArgument);
  }
  EXPECT_EQ(generator.forward_calls(), 0);
}

TEST_F(SparseAutoEncoderDataSetTest,
       RejectsInvalidSourceDimensionsBeforeForward) {
  auto data = MakeData();
  ASSERT_TRUE(data.ok()) << data.status();
  RecordingActivationGenerator generator({data->inputs});
  const std::filesystem::path checkpoint =
      std::filesystem::path(testing::TempDir()) / "sae-source-shape-checkpoint";
  ASSERT_TRUE(WriteToDirectory(*executor_, generator, checkpoint).ok());
  for (const auto [samples, length] :
       {std::pair{0, kSequenceLength}, std::pair{-1, kSequenceLength},
        std::pair{kBatchSize, 0}, std::pair{kBatchSize, -1},
        std::pair{std::numeric_limits<int>::max(), 2}}) {
    SCOPED_TRACE(testing::Message() << samples << " x " << length);
    FixedTokenDataSetIterator source(*data, samples, length);
    auto dataset = SparseAutoEncoderDataSetIterator::Create(
        *executor_, generator, source, checkpoint);
    ASSERT_TRUE(dataset.ok()) << dataset.status();
    auto result = (*dataset)->Next();
    EXPECT_EQ(result.status().code(), absl::StatusCode::kInvalidArgument);
  }
  EXPECT_EQ(generator.forward_calls(), 0);
}

TEST_F(SparseAutoEncoderDataSetTest, RequiresExactlyOneGeneratorOutput) {
  auto data = MakeData();
  ASSERT_TRUE(data.ok()) << data.status();
  for (int count : {0, 2}) {
    SCOPED_TRACE(count);
    RecordingActivationGenerator generator(BufferVec(count, data->inputs));
    FixedTokenDataSetIterator source(*data);
    const std::filesystem::path checkpoint =
        std::filesystem::path(testing::TempDir()) /
        ("sae-output-count-" + std::to_string(count));
    ASSERT_TRUE(WriteToDirectory(*executor_, generator, checkpoint).ok());
    auto dataset = SparseAutoEncoderDataSetIterator::Create(
        *executor_, generator, source, checkpoint);
    ASSERT_TRUE(dataset.ok()) << dataset.status();
    auto result = (*dataset)->Next();
    EXPECT_EQ(result.status().code(), absl::StatusCode::kInvalidArgument);
    EXPECT_EQ(generator.forward_calls(), 1);
    EXPECT_EQ(generator.backward_calls(), 0);
  }
}

TEST_F(SparseAutoEncoderDataSetTest,
       RejectsGeneratorOutputFromAnotherExecutor) {
  auto other = cuda::Executor::Create();
  ASSERT_TRUE(other.ok()) << other.status();
  auto foreign = Buffer::Allocate(
      **other, kTokenCount * kEmbeddingDimension * sizeof(float));
  ASSERT_TRUE(foreign.ok()) << foreign.status();
  auto data = MakeData();
  ASSERT_TRUE(data.ok()) << data.status();
  RecordingActivationGenerator generator({*foreign});
  FixedTokenDataSetIterator source(*data);
  const std::filesystem::path checkpoint =
      std::filesystem::path(testing::TempDir()) /
      "sae-generator-executor-checkpoint";
  ASSERT_TRUE(WriteToDirectory(*executor_, generator, checkpoint).ok());
  auto dataset = SparseAutoEncoderDataSetIterator::Create(*executor_, generator,
                                                          source, checkpoint);
  ASSERT_TRUE(dataset.ok()) << dataset.status();
  auto result = (*dataset)->Next();
  EXPECT_EQ(result.status().code(), absl::StatusCode::kInvalidArgument);
}

}  // namespace
}  // namespace pluto::llm
