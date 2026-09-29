#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "src/cuda/page_locked_host_array.h"
#include "src/llm/layer_hooks.h"
#include "src/llm/layers/embedding.h"
#include "src/util/status_macros.h"

namespace pluto::llm {
namespace {

class ImportedEmbeddingTest : public testing::Test {
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

  template <class T>
  absl::StatusOr<Buffer> Upload(const std::vector<T>& values) {
    ASSIGN_OR_RETURN(auto host,
                     cuda::PageLockedHostArray<T>::CopyFrom(*executor_, values));
    ASSIGN_OR_RETURN(auto device,
                     Buffer::Allocate(*executor_, values.size() * sizeof(T)));
    RETURN_IF_ERROR(cuda::CudaStatus(
        cudaMemcpyAsync(device.data(), host.data(), device.size_bytes(),
                        cudaMemcpyHostToDevice, executor_->stream()),
        "upload imported embedding test"));
    RETURN_IF_ERROR(executor_->Synchronize());
    return device;
  }

  absl::StatusOr<Buffer> UploadWeights(const std::vector<float>& values,
                                       MatrixStorage storage) {
    if (storage == MatrixStorage::kFloat32)
      return Upload(values);
    std::vector<__nv_bfloat16> bf16;
    for (float value : values)
      bf16.emplace_back(value);
    return Upload(bf16);
  }

  template <class T>
  std::vector<float> Read(const Buffer& buffer) {
    auto host = cuda::PageLockedHostArray<T>::Allocate(
        *executor_, buffer.size_bytes() / sizeof(T));
    EXPECT_TRUE(host.ok()) << host.status();
    if (!host.ok())
      return {};
    EXPECT_EQ(cudaMemcpyAsync(host->data(), buffer.data(), buffer.size_bytes(),
                              cudaMemcpyDeviceToHost, executor_->stream()),
              cudaSuccess);
    EXPECT_TRUE(executor_->Synchronize().ok());
    std::vector<float> result;
    for (auto value : *host)
      result.push_back(static_cast<float>(value));
    return result;
  }

  std::unique_ptr<cuda::Executor> executor_;
};

TEST_F(ImportedEmbeddingTest,
       BothStorageFormatsMatchScalarLookupIncludingTails) {
  constexpr int kVocabulary = 3;
  for (MatrixStorage storage :
       {MatrixStorage::kFloat32, MatrixStorage::kBFloat16}) {
    for (int width : {1, 7, 257, 5120}) {
      SCOPED_TRACE(static_cast<int>(storage));
      SCOPED_TRACE(width);
      std::vector<float> values(kVocabulary * width);
      for (size_t i = 0; i < values.size(); ++i)
        values[i] = (static_cast<int>(i % 113) - 56) * 0.01731f;
      auto weights = UploadWeights(values, storage);
      ASSERT_TRUE(weights.ok()) << weights.status();
      auto layer = EmbeddingLookupLayer::Create(*executor_, *weights, storage,
                                                kVocabulary, width);
      ASSERT_TRUE(layer.ok()) << layer.status();
      ASSERT_EQ((*layer)->weights().size(), 1);
      EXPECT_EQ((*layer)->weights()[0].data(), weights->data());
      EXPECT_EQ((*layer)->weights()[0].size_bytes(), weights->size_bytes());
      EXPECT_TRUE((*layer)->gradients().empty());
      EXPECT_EQ((*layer)->stored_vocab_size(), kVocabulary);
      EXPECT_EQ((*layer)->output_type(), DataType::BF16);
      EXPECT_EQ((*layer)->input_types()[0],
                ActivationType(DataType::INT32, {-2, 1}));
      EXPECT_EQ((*layer)->output_types()[0],
                ActivationType(DataType::BF16, {-2, 1, width}));

      for (int32_t id : {-1, 0, 1, 2, 3, std::numeric_limits<int32_t>::max()}) {
        SCOPED_TRACE(id);
        auto token = Upload<int32_t>({id});
        ASSERT_TRUE(token.ok()) << token.status();
        auto result = (*layer)->fwd(*executor_, {*token});
        ASSERT_TRUE(result.ok()) << result.status();
        ASSERT_EQ(result->outputs.size(), 1);
        EXPECT_EQ(result->outputs[0].size_bytes(),
                  width * sizeof(__nv_bfloat16));
        EXPECT_TRUE(result->state.intermediates.empty());
        const auto actual = Read<__nv_bfloat16>(result->outputs[0]);
        ASSERT_EQ(actual.size(), width);
        for (int column = 0; column < width; ++column) {
          const float expected = id >= 0 && id < kVocabulary
                                     ? static_cast<float>(__nv_bfloat16(
                                           values[id * width + column]))
                                     : 0.0f;
          EXPECT_EQ(actual[column], expected) << "column " << column;
        }
      }
    }
  }
}

TEST_F(ImportedEmbeddingTest, UnsupportedTrainingCannotMutateSharedWeights) {
  const std::vector<float> original = {0.101f, -0.401f, 0.601f, 1.201f};
  for (MatrixStorage storage :
       {MatrixStorage::kFloat32, MatrixStorage::kBFloat16}) {
    SCOPED_TRACE(static_cast<int>(storage));
    auto weights = UploadWeights(original, storage);
    ASSERT_TRUE(weights.ok());
    auto layer =
        EmbeddingLookupLayer::Create(*executor_, *weights, storage, 2, 2);
    ASSERT_TRUE(layer.ok()) << layer.status();
    EXPECT_TRUE(absl::IsUnimplemented((*layer)->InitializeIdentity()));
    EXPECT_TRUE(absl::IsUnimplemented((*layer)->InitializeNormal(0.02f, 17)));
    EXPECT_TRUE(absl::IsUnimplemented(
        LanguageModelingHeadLayer::Create(layer->get()).status()));
    auto token = Upload<int32_t>({1});
    ASSERT_TRUE(token.ok());
    auto result = (*layer)->fwd(*executor_, {*token});
    ASSERT_TRUE(result.ok()) << result.status();
    EXPECT_TRUE(absl::IsUnimplemented(
        (*layer)->bwd(*executor_, {}, std::move(result->state)).status()));
    EXPECT_TRUE((*layer)->gradients().empty());
    auto expected = original;
    if (storage == MatrixStorage::kBFloat16)
      for (float& value : expected)
        value = static_cast<float>(__nv_bfloat16(value));
    const auto actual = storage == MatrixStorage::kFloat32
                            ? Read<float>(*weights)
                            : Read<__nv_bfloat16>(*weights);
    EXPECT_EQ(actual, expected);
  }
}

TEST_F(ImportedEmbeddingTest, RejectsBadStorageShapeSizeAndWeightExecutor) {
  auto weights = Upload<float>({1, 2, 3, 4});
  ASSERT_TRUE(weights.ok());
  for (auto [vocab, width] :
       {std::pair{0, 2}, std::pair{2, -1}, std::pair{3, 2},
        std::pair{2, (1 << 20) + 1}, std::pair{(1 << 20) + 1, 2}})
    EXPECT_TRUE(absl::IsInvalidArgument(
        EmbeddingLookupLayer::Create(*executor_, *weights,
                                     MatrixStorage::kFloat32, vocab, width)
            .status()));
  for (auto storage : {MatrixStorage::kBFloat16, MatrixStorage::kFp8E4M3,
                       static_cast<MatrixStorage>(99)})
    EXPECT_TRUE(absl::IsInvalidArgument(
        EmbeddingLookupLayer::Create(*executor_, *weights, storage, 2, 2)
            .status()));
  auto other = cuda::Executor::Create();
  ASSERT_TRUE(other.ok()) << other.status();
  EXPECT_TRUE(absl::IsInvalidArgument(
      EmbeddingLookupLayer::Create(**other, *weights, MatrixStorage::kFloat32,
                                   2, 2)
          .status()));
  EXPECT_TRUE((*other)->Synchronize().ok());
}

TEST_F(ImportedEmbeddingTest, RejectsInvalidInputsAndExecutor) {
  auto weights = Upload<float>({1, 2, 3, 4});
  ASSERT_TRUE(weights.ok());
  auto layer = EmbeddingLookupLayer::Create(*executor_, *weights,
                                            MatrixStorage::kFloat32, 2, 2);
  ASSERT_TRUE(layer.ok());
  auto token = Upload<int32_t>({0});
  auto too_many = Upload<int32_t>({0, 1});
  auto wrong_bytes = Upload<__nv_bfloat16>({__nv_bfloat16(0.f)});
  ASSERT_TRUE(token.ok());
  ASSERT_TRUE(too_many.ok());
  ASSERT_TRUE(wrong_bytes.ok());
  EXPECT_TRUE(absl::IsInvalidArgument((*layer)->fwd(*executor_, {}).status()));
  EXPECT_TRUE(absl::IsInvalidArgument(
      (*layer)->fwd(*executor_, {*token, *token}).status()));
  EXPECT_TRUE(
      absl::IsInvalidArgument((*layer)->fwd(*executor_, {*too_many}).status()));
  EXPECT_TRUE(absl::IsInvalidArgument(
      (*layer)->fwd(*executor_, {*wrong_bytes}).status()));
  auto other = cuda::Executor::Create();
  ASSERT_TRUE(other.ok()) << other.status();
  {
    auto foreign = Buffer::Allocate(**other, sizeof(int32_t));
    ASSERT_TRUE(foreign.ok());
    EXPECT_TRUE(absl::IsInvalidArgument(
        (*layer)->fwd(*executor_, {*foreign}).status()));
  }
  EXPECT_TRUE(
      absl::IsInvalidArgument((*layer)->fwd(**other, {*token}).status()));
  EXPECT_TRUE((*other)->Synchronize().ok());
}

TEST_F(ImportedEmbeddingTest, ImportedLookupUsesOrdinaryLayerHooks) {
  auto weights = Upload<float>({1, 2, 3, 4});
  auto token = Upload<int32_t>({1});
  ASSERT_TRUE(weights.ok());
  ASSERT_TRUE(token.ok());
  auto layer = EmbeddingLookupLayer::Create(*executor_, *weights,
                                            MatrixStorage::kFloat32, 2, 2);
  ASSERT_TRUE(layer.ok());
  LayerHooks hooks;
  int calls = 0;
  hooks.activation_hook = [&](cuda::Executor& executor, absl::string_view name,
                              absl::Span<const ActivationType> types,
                              absl::Span<Buffer> outputs) {
    EXPECT_EQ(&executor, executor_.get());
    EXPECT_EQ(name, "EmbeddingLookupLayer");
    EXPECT_EQ(types[0], ActivationType(DataType::BF16, {-2, 1, 2}));
    EXPECT_EQ(outputs.size(), 1);
    ++calls;
    return absl::OkStatus();
  };
  auto result = (*layer)->fwd(*executor_, {*token}, &hooks);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(calls, 1);
  EXPECT_EQ(Read<__nv_bfloat16>(result->outputs[0]),
            (std::vector<float>{3, 4}));
}

}  // namespace
}  // namespace pluto::llm
