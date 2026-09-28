#include "src/llm/layers/inference.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "src/cuda/page_locked_host_array.h"
#include "src/llm/layer_hooks.h"
#include "src/util/status_macros.h"

namespace pluto::llm {
namespace {

using Storage = inference_ops::MatrixStorage;

float Bf16(float x) { return static_cast<float>(__nv_bfloat16(x)); }

class InferenceTest : public testing::Test {
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
    ASSIGN_OR_RETURN(Buffer buffer,
                     Buffer::Allocate(*executor_, values.size() * sizeof(T)));
    RETURN_IF_ERROR(cuda::CudaStatus(
        cudaMemcpyAsync(buffer.data(), host.data(), buffer.size_bytes(),
                        cudaMemcpyHostToDevice, executor_->stream()),
        "upload inference test"));
    RETURN_IF_ERROR(executor_->Synchronize());
    return buffer;
  }

  absl::StatusOr<Buffer> Input(std::vector<float> values) {
    std::vector<__nv_bfloat16> bf16;
    for (float value : values)
      bf16.emplace_back(value);
    return Upload(bf16);
  }

  std::vector<float> Read(const Buffer& buffer) {
    auto host = cuda::PageLockedHostArray<__nv_bfloat16>::Allocate(
        *executor_, buffer.size_bytes() / sizeof(__nv_bfloat16));
    EXPECT_TRUE(host.ok());
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

TEST_F(InferenceTest, LinearImportedStorageHasNoMasterWeightsOrGradients) {
  auto input = Input({1, 0, -1});
  ASSERT_TRUE(input.ok());
  for (Storage storage :
       {Storage::kFloat32, Storage::kBFloat16, Storage::kFp8E4M3}) {
    SCOPED_TRACE(static_cast<int>(storage));
    auto weights =
        storage == Storage::kFloat32 ? Upload<float>({1, 2, 3, 4, 5, 6})
        : storage == Storage::kBFloat16
            ? Upload<__nv_bfloat16>({__nv_bfloat16(1.f), __nv_bfloat16(2.f),
                                     __nv_bfloat16(3.f), __nv_bfloat16(4.f),
                                     __nv_bfloat16(5.f), __nv_bfloat16(6.f)})
            : Upload<__nv_fp8_e4m3>({__nv_fp8_e4m3(1.f), __nv_fp8_e4m3(2.f),
                                     __nv_fp8_e4m3(3.f), __nv_fp8_e4m3(4.f),
                                     __nv_fp8_e4m3(5.f), __nv_fp8_e4m3(6.f)});
    ASSERT_TRUE(weights.ok());
    auto scale = Upload<float>({1});
    ASSERT_TRUE(scale.ok());
    auto layer = InferenceLinearLayer::Create(
        *executor_, *weights, storage, 3, 2,
        storage == Storage::kFp8E4M3 ? std::optional<Buffer>(*scale)
                                     : std::nullopt);
    ASSERT_TRUE(layer.ok()) << layer.status();
    EXPECT_EQ((*layer)->weights()[0].data(), weights->data());
    EXPECT_EQ((*layer)->weights().size(), storage == Storage::kFp8E4M3 ? 2 : 1);
    EXPECT_TRUE((*layer)->gradients().empty());
    EXPECT_EQ((*layer)->input_types()[0],
              ActivationType(DataType::BF16, {-2, 1, 3}));
    EXPECT_EQ((*layer)->output_types()[0],
              ActivationType(DataType::BF16, {-2, 1, 2}));
    auto result = (*layer)->fwd(*executor_, {*input});
    ASSERT_TRUE(result.ok()) << result.status();
    EXPECT_EQ(Read(result->outputs[0]), (std::vector<float>{-2, -2}));
    EXPECT_TRUE(absl::IsUnimplemented(
        (*layer)->bwd(*executor_, {}, std::move(result->state)).status()));
  }
}

TEST_F(InferenceTest, EmbeddingDeviceTokenAndOutOfBoundsAreSafe) {
  for (Storage storage : {Storage::kFloat32, Storage::kBFloat16}) {
    auto weights = storage == Storage::kFloat32
                       ? Upload<float>({1, 2, 3, 4, 5, 6})
                       : Input({1, 2, 3, 4, 5, 6});
    ASSERT_TRUE(weights.ok());
    auto layer =
        InferenceEmbeddingLayer::Create(*executor_, *weights, storage, 2, 3);
    ASSERT_TRUE(layer.ok()) << layer.status();
    EXPECT_EQ((*layer)->input_types()[0],
              ActivationType(DataType::INT32, {-2, 1}));
    for (int32_t token : {-1, 0, 1, 2, std::numeric_limits<int32_t>::max()}) {
      auto id = Upload<int32_t>({token});
      ASSERT_TRUE(id.ok());
      auto result = (*layer)->fwd(*executor_, {*id});
      ASSERT_TRUE(result.ok()) << result.status();
      EXPECT_EQ(Read(result->outputs[0]),
                token == 0   ? (std::vector<float>{1, 2, 3})
                : token == 1 ? (std::vector<float>{4, 5, 6})
                             : (std::vector<float>{0, 0, 0}));
    }
  }
}

TEST_F(InferenceTest, RmsNormPreservesZeroCenteredRuleAndBfloat16Boundary) {
  auto weight = Upload<float>({0.f, 0.5f, -0.25f});
  auto input = Input({1.f, -2.f, 0.5f});
  ASSERT_TRUE(weight.ok());
  ASSERT_TRUE(input.ok());
  auto layer = RmsNormLayer::Create(*executor_, 3, *weight, 1e-6f);
  ASSERT_TRUE(layer.ok());
  auto result = (*layer)->fwd(*executor_, {*input});
  ASSERT_TRUE(result.ok()) << result.status();
  const float inverse = 1.f / std::sqrt(1.75f + 1e-6f);
  EXPECT_EQ(Read(result->outputs[0]),
            (std::vector<float>{Bf16(inverse), Bf16(-3.f * inverse),
                                Bf16(0.375f * inverse)}));
  EXPECT_EQ((*layer)->weights()[0].data(), weight->data());
  EXPECT_TRUE(absl::IsUnimplemented(
      (*layer)->bwd(*executor_, {}, std::move(result->state)).status()));
}

TEST_F(InferenceTest, SwiGluSupportsHooksAndRejectsBadInputs) {
  auto gate = Input({-3.f, -0.5f, 1.5f});
  auto up = Input({2.f, -4.f, 0.25f});
  ASSERT_TRUE(gate.ok());
  ASSERT_TRUE(up.ok());
  auto layer = SwiGluLayer::Create(3);
  ASSERT_TRUE(layer.ok());
  LayerHooks hooks;
  int calls = 0;
  hooks.activation_hook = [&](cuda::Executor& executor, absl::string_view name,
                              absl::Span<const ActivationType> types,
                              absl::Span<Buffer> outputs) {
    EXPECT_EQ(&executor, executor_.get());
    EXPECT_EQ(name, "SwiGluLayer");
    EXPECT_EQ(types[0], ActivationType(DataType::BF16, {-2, 1, 3}));
    EXPECT_EQ(outputs.size(), 1);
    ++calls;
    return absl::OkStatus();
  };
  auto result = (*layer)->fwd(*executor_, {*gate, *up}, &hooks);
  ASSERT_TRUE(result.ok());
  EXPECT_EQ(calls, 1);
  EXPECT_EQ(
      Read(result->outputs[0]),
      (std::vector<float>{Bf16(Bf16(-3.f / (1 + std::exp(3.f))) * 2),
                          Bf16(Bf16(-0.5f / (1 + std::exp(0.5f))) * -4),
                          Bf16(Bf16(1.5f / (1 + std::exp(-1.5f))) * 0.25f)}));
  EXPECT_TRUE(absl::IsUnimplemented(
      (*layer)->bwd(*executor_, {}, std::move(result->state)).status()));
  EXPECT_FALSE((*layer)->fwd(*executor_, {*gate}).ok());
  auto wrong = Input({1, 2});
  ASSERT_TRUE(wrong.ok());
  EXPECT_FALSE((*layer)->fwd(*executor_, {*gate, *wrong}).ok());
  auto other = cuda::Executor::Create();
  ASSERT_TRUE(other.ok());
  EXPECT_FALSE((*layer)->fwd(**other, {*gate, *up}).ok());
  EXPECT_TRUE((*other)->Synchronize().ok());
}

TEST_F(InferenceTest, ValidationCatchesStorageDimensionsScalesAndExecutor) {
  auto weight = Upload<float>({1, 2, 3, 4});
  auto scale = Upload<float>({1});
  auto fp8 = Upload<uint8_t>({0, 0, 0, 0});
  auto input = Input({1, 2});
  ASSERT_TRUE(weight.ok());
  ASSERT_TRUE(scale.ok());
  ASSERT_TRUE(fp8.ok());
  ASSERT_TRUE(input.ok());
  EXPECT_FALSE(
      InferenceLinearLayer::Create(*executor_, *weight, Storage::kFloat32, 0, 2)
          .ok());
  EXPECT_FALSE(
      InferenceLinearLayer::Create(*executor_, *weight, Storage::kFloat32, 3, 2)
          .ok());
  EXPECT_FALSE(InferenceLinearLayer::Create(*executor_, *weight,
                                            Storage::kFloat32, 2, 2, *scale)
                   .ok());
  EXPECT_FALSE(
      InferenceLinearLayer::Create(*executor_, *fp8, Storage::kFp8E4M3, 2, 2)
          .ok());
  EXPECT_FALSE(InferenceLinearLayer::Create(*executor_, *fp8, Storage::kFp8E4M3,
                                            2, 2, *weight)
                   .ok());
  EXPECT_FALSE(InferenceLinearLayer::Create(*executor_, *weight,
                                            static_cast<Storage>(99), 2, 2)
                   .ok());
  EXPECT_FALSE(
      InferenceEmbeddingLayer::Create(*executor_, *fp8, Storage::kFp8E4M3, 2, 2)
          .ok());
  EXPECT_FALSE(RmsNormLayer::Create(*executor_, 4, *weight, 0).ok());
  EXPECT_FALSE(RmsNormLayer::Create(*executor_, 4, *weight,
                                    std::numeric_limits<float>::quiet_NaN())
                   .ok());
  EXPECT_FALSE(SwiGluLayer::Create(-1).ok());
  auto layer = InferenceLinearLayer::Create(*executor_, *weight,
                                            Storage::kFloat32, 2, 2);
  ASSERT_TRUE(layer.ok());
  EXPECT_FALSE((*layer)->fwd(*executor_, {}).ok());
  EXPECT_FALSE((*layer)->fwd(*executor_, {*weight}).ok());
  auto other = cuda::Executor::Create();
  ASSERT_TRUE(other.ok());
  EXPECT_FALSE(
      InferenceLinearLayer::Create(**other, *weight, Storage::kFloat32, 2, 2)
          .ok());
  EXPECT_FALSE((*layer)->fwd(**other, {*input}).ok());
  EXPECT_TRUE((*other)->Synchronize().ok());
}

TEST_F(InferenceTest, ConversionHandlesTailsAndDoesNotAliasInput) {
  std::vector<float> values(513);
  for (size_t i = 0; i < values.size(); ++i)
    values[i] = std::sin(float(i)) * 1.2345f;
  auto input = Upload(values);
  ASSERT_TRUE(input.ok());
  auto packed =
      inference_internal::ToBFloat16(*executor_, *input, values.size());
  ASSERT_TRUE(packed.ok()) << packed.status();
  EXPECT_NE(packed->data(), input->data());
  const auto actual = Read(*packed);
  ASSERT_EQ(actual.size(), values.size());
  for (size_t i = 0; i < values.size(); ++i)
    EXPECT_EQ(actual[i], Bf16(values[i]));
  auto expanded =
      inference_internal::ToFloat(*executor_, *packed, values.size());
  ASSERT_TRUE(expanded.ok());
  auto host =
      cuda::PageLockedHostArray<float>::Allocate(*executor_, values.size());
  ASSERT_TRUE(host.ok());
  ASSERT_EQ(
      cudaMemcpyAsync(host->data(), expanded->data(), expanded->size_bytes(),
                      cudaMemcpyDeviceToHost, executor_->stream()),
      cudaSuccess);
  ASSERT_TRUE(executor_->Synchronize().ok());
  for (size_t i = 0; i < values.size(); ++i)
    EXPECT_EQ((*host)[i], Bf16(values[i]));
  EXPECT_FALSE(
      inference_internal::ToFloat(*executor_, *packed, values.size() - 1).ok());
  EXPECT_FALSE(inference_internal::AllocateFloatVector(*executor_, -1).ok());
}

}  // namespace
}  // namespace pluto::llm
