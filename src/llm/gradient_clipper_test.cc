#include "src/llm/gradient_clipper.h"

#include <cuda_runtime_api.h>

#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/types/span.h"
#include "gtest/gtest.h"
#include "src/cuda/page_locked_host_array.h"
#include "src/util/status_macros.h"

namespace pluto::llm {
namespace {

// The clipper operates on parameter storage only. A minimal fake layer makes
// allocation aliasing and malformed parameter lists easy to test explicitly.
class GradientModel final : public Layer {
 public:
  absl::string_view name() const override { return "GradientModel"; }
  absl::Span<const ActivationType> input_types() const override { return {}; }
  absl::Span<const ActivationType> output_types() const override { return {}; }
  absl::Span<Buffer> weights() override { return absl::MakeSpan(weights_); }
  absl::Span<Buffer> gradients() override { return absl::MakeSpan(gradients_); }
  DataType output_type() const override { return DataType::FP32; }

  std::vector<Buffer> weights_;
  std::vector<Buffer> gradients_;

 private:
  absl::StatusOr<FwdResult> fwd_impl(cuda::Executor&, absl::Span<const Buffer>,
                                     LayerHooks*) const override {
    return absl::UnimplementedError("storage-only test model");
  }
  absl::StatusOr<BufferVec> bwd_impl(cuda::Executor&, absl::Span<const Buffer>,
                                     BackwardState, LayerHooks*) override {
    return absl::UnimplementedError("storage-only test model");
  }
};

class GradientClipperTest : public testing::Test {
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
  }

  absl::Status Upload(const Buffer& buffer, absl::Span<const float> values) {
    if (buffer.size_bytes() != values.size() * sizeof(float))
      return absl::InvalidArgumentError("test upload shape mismatch");
    ASSIGN_OR_RETURN(auto host, cuda::PageLockedHostArray<float>::CopyFrom(
                                    *executor_, values));
    return cuda::CudaStatus(
        cudaMemcpyAsync(buffer.data(), host.data(), buffer.size_bytes(),
                        cudaMemcpyHostToDevice, executor_->stream()),
        "upload test values");
  }

  absl::StatusOr<Buffer> MakeBuffer(absl::Span<const float> values) {
    ASSIGN_OR_RETURN(
        auto buffer,
        Buffer::Allocate(*executor_, values.size() * sizeof(float)));
    RETURN_IF_ERROR(Upload(buffer, values));
    return buffer;
  }

  absl::StatusOr<std::unique_ptr<GradientModel>> MakeModel(
      const std::vector<std::vector<float>>& tensors) {
    auto model = absl::make_unique<GradientModel>();
    for (const auto& tensor : tensors) {
      ASSIGN_OR_RETURN(auto gradient, MakeBuffer(tensor));
      ASSIGN_OR_RETURN(auto weight,
                       MakeBuffer(std::vector<float>(tensor.size(), 42.0f)));
      model->weights_.push_back(std::move(weight));
      model->gradients_.push_back(std::move(gradient));
    }
    return model;
  }

  absl::StatusOr<cuda::PageLockedHostArray<float>> Download(
      const Buffer& buffer) {
    ASSIGN_OR_RETURN(auto host,
                     cuda::PageLockedHostArray<float>::Allocate(
                         *executor_, buffer.size_bytes() / sizeof(float)));
    RETURN_IF_ERROR(cuda::CudaStatus(
        cudaMemcpyAsync(host.data(), buffer.data(), buffer.size_bytes(),
                        cudaMemcpyDeviceToHost, executor_->stream()),
        "download test values"));
    RETURN_IF_ERROR(executor_->Synchronize());
    return host;
  }

  std::unique_ptr<cuda::Executor> executor_;
};

TEST_F(GradientClipperTest, ClipsOneGlobalNormAndDoesNotModifyWeights) {
  auto model = MakeModel({{3.0f}, {4.0f, 0.0f}});
  ASSERT_TRUE(model.ok()) << model.status();
  auto clipper = GradientClipper::Create(*executor_, **model, 2.0f);
  ASSERT_TRUE(clipper.ok()) << clipper.status();
  ASSERT_TRUE((*clipper)->Clip().ok());
  auto first = Download((*model)->gradients_[0]);
  auto second = Download((*model)->gradients_[1]);
  ASSERT_TRUE(first.ok()) << first.status();
  ASSERT_TRUE(second.ok()) << second.status();
  EXPECT_NEAR((*first)[0], 1.2f, 1e-6f);
  EXPECT_NEAR((*second)[0], 1.6f, 1e-6f);
  EXPECT_EQ((*second)[1], 0.0f);
  for (const auto& weight : (*model)->weights_) {
    auto values = Download(weight);
    ASSERT_TRUE(values.ok()) << values.status();
    for (float value : values->span())
      EXPECT_EQ(value, 42.0f);
  }
}

TEST_F(GradientClipperTest, ZeroAndBelowThresholdGradientsAreUnchanged) {
  for (const std::vector<float>& values :
       {std::vector<float>{0.0f, 0.0f, 0.0f},
        std::vector<float>{0.1f, -0.2f, 0.3f}}) {
    auto model = MakeModel({values});
    ASSERT_TRUE(model.ok()) << model.status();
    auto clipper = GradientClipper::Create(*executor_, **model);
    ASSERT_TRUE(clipper.ok()) << clipper.status();
    ASSERT_TRUE((*clipper)->Clip().ok());
    auto actual = Download((*model)->gradients_[0]);
    ASSERT_TRUE(actual.ok()) << actual.status();
    EXPECT_EQ(std::memcmp(actual->data(), values.data(),
                          values.size() * sizeof(float)),
              0);
  }
}

TEST_F(GradientClipperTest, TiedGradientCountsAndScalesOnlyOnce) {
  auto model = MakeModel({{3.0f}, {4.0f}});
  ASSERT_TRUE(model.ok()) << model.status();
  (*model)->weights_.push_back((*model)->weights_.front());
  (*model)->gradients_.push_back((*model)->gradients_.front());
  auto clipper = GradientClipper::Create(*executor_, **model);
  ASSERT_TRUE(clipper.ok()) << clipper.status();
  ASSERT_TRUE((*clipper)->Clip().ok());
  auto first = Download((*model)->gradients_[0]);
  auto second = Download((*model)->gradients_[1]);
  ASSERT_TRUE(first.ok()) << first.status();
  ASSERT_TRUE(second.ok()) << second.status();
  EXPECT_NEAR((*first)[0], 0.6f, 1e-6f);
  EXPECT_NEAR((*second)[0], 0.8f, 1e-6f);
}

TEST_F(GradientClipperTest, PartialChunksAndManyChunksRepeatExactly) {
  // More than 256 chunks exercises each final-reduction thread's loop, while
  // nonmultiples of 1024 exercise both tensors' partial final chunks.
  std::vector<std::vector<float>> tensors{std::vector<float>(300017),
                                          std::vector<float>(1025)};
  double squared_norm = 0;
  for (size_t tensor = 0; tensor < tensors.size(); ++tensor)
    for (size_t index = 0; index < tensors[tensor].size(); ++index) {
      const float value = std::sin(index * 0.03 + tensor) * 0.25f;
      tensors[tensor][index] = value;
      squared_norm += static_cast<double>(value) * value;
    }
  const float scale =
      static_cast<float>(1.0 / (std::sqrt(squared_norm) + 1e-6));
  auto model = MakeModel(tensors);
  ASSERT_TRUE(model.ok()) << model.status();
  auto clipper = GradientClipper::Create(*executor_, **model);
  ASSERT_TRUE(clipper.ok()) << clipper.status();
  std::vector<cuda::PageLockedHostArray<float>> first_result;
  for (int repeat = 0; repeat < 4; ++repeat) {
    for (size_t tensor = 0; tensor < tensors.size(); ++tensor)
      ASSERT_TRUE(Upload((*model)->gradients_[tensor], tensors[tensor]).ok());
    ASSERT_TRUE((*clipper)->Clip().ok());
    for (size_t tensor = 0; tensor < tensors.size(); ++tensor) {
      auto actual = Download((*model)->gradients_[tensor]);
      ASSERT_TRUE(actual.ok()) << actual.status();
      if (repeat == 0) {
        for (size_t index = 0; index < actual->size(); ++index)
          EXPECT_NEAR((*actual)[index], tensors[tensor][index] * scale, 1e-8f);
        first_result.push_back(std::move(*actual));
      } else {
        EXPECT_EQ(std::memcmp(actual->data(), first_result[tensor].data(),
                              actual->size_bytes()),
                  0);
      }
    }
  }
}

TEST_F(GradientClipperTest, FiniteLargeGradientsDoNotOverflowSquaredNorm) {
  auto model = MakeModel({{3e20f, 4e20f}});
  ASSERT_TRUE(model.ok()) << model.status();
  auto clipper = GradientClipper::Create(*executor_, **model);
  ASSERT_TRUE(clipper.ok()) << clipper.status();
  ASSERT_TRUE((*clipper)->Clip().ok());
  auto actual = Download((*model)->gradients_[0]);
  ASSERT_TRUE(actual.ok()) << actual.status();
  EXPECT_NEAR((*actual)[0], 0.6f, 1e-6f);
  EXPECT_NEAR((*actual)[1], 0.8f, 1e-6f);
}

TEST_F(GradientClipperTest, MasksEachTensorTailAtTileBoundaries) {
  // The pointer table describes distinct allocations, not consecutive chunks
  // of one flat tensor. In particular a short tensor's masked lanes must not
  // load or scale the following allocation's data.
  std::vector<std::vector<float>> tensors;
  double squared_norm = 0;
  for (int size : {1, 15, 16, 255, 256, 257, 1023, 1024, 1025, 2049}) {
    std::vector<float> values(size);
    for (int index = 0; index < size; ++index) {
      const float value = static_cast<float>((index + size) % 13 - 6) / 8.0f;
      values[index] = value;
      squared_norm += static_cast<double>(value) * value;
    }
    tensors.push_back(std::move(values));
  }
  const float scale =
      static_cast<float>(0.75 / (std::sqrt(squared_norm) + 1e-6));
  auto model = MakeModel(tensors);
  ASSERT_TRUE(model.ok()) << model.status();
  auto clipper = GradientClipper::Create(*executor_, **model, 0.75f);
  ASSERT_TRUE(clipper.ok()) << clipper.status();
  ASSERT_TRUE((*clipper)->Clip().ok());
  for (size_t tensor = 0; tensor < tensors.size(); ++tensor) {
    auto actual = Download((*model)->gradients_[tensor]);
    ASSERT_TRUE(actual.ok()) << actual.status();
    SCOPED_TRACE(tensors[tensor].size());
    for (size_t index = 0; index < actual->size(); ++index)
      EXPECT_NEAR((*actual)[index], tensors[tensor][index] * scale, 1e-8f);
  }
}

TEST_F(GradientClipperTest, NormBeyondFloatRangePreservesSubnormalScale) {
  // The FP64 norm exceeds FLT_MAX, and the final FP32 scale is subnormal.
  // Neither narrowing the norm nor flushing that scale to zero is acceptable.
  const float largest = std::numeric_limits<float>::max();
  auto model = MakeModel({{largest, -largest}, {largest / 2}});
  ASSERT_TRUE(model.ok()) << model.status();
  auto clipper = GradientClipper::Create(*executor_, **model);
  ASSERT_TRUE(clipper.ok()) << clipper.status();
  ASSERT_TRUE((*clipper)->Clip().ok());
  auto first = Download((*model)->gradients_[0]);
  auto second = Download((*model)->gradients_[1]);
  ASSERT_TRUE(first.ok()) << first.status();
  ASSERT_TRUE(second.ok()) << second.status();
  EXPECT_NEAR((*first)[0], 2.0f / 3.0f, 1e-6f);
  EXPECT_NEAR((*first)[1], -2.0f / 3.0f, 1e-6f);
  EXPECT_NEAR((*second)[0], 1.0f / 3.0f, 1e-6f);
}

TEST_F(GradientClipperTest, EpsilonIsAppliedAtTheClippingBoundary) {
  auto model = MakeModel({{1.0f}});
  ASSERT_TRUE(model.ok()) << model.status();
  auto clipper = GradientClipper::Create(*executor_, **model);
  ASSERT_TRUE(clipper.ok()) << clipper.status();
  ASSERT_TRUE((*clipper)->Clip().ok());
  auto actual = Download((*model)->gradients_[0]);
  ASSERT_TRUE(actual.ok()) << actual.status();
  EXPECT_FLOAT_EQ((*actual)[0], static_cast<float>(1.0 / (1.0 + 1e-6)));
  EXPECT_LT((*actual)[0], 1.0f);
}

TEST_F(GradientClipperTest, NaNNormPreservesLegacyUnchangedGradientBehavior) {
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const float infinity = std::numeric_limits<float>::infinity();
  for (const std::vector<float>& values :
       {std::vector<float>{nan, 3.0f, -4.0f},
        std::vector<float>{nan, infinity, 3.0f}}) {
    auto model = MakeModel({values, {6.0f}});
    ASSERT_TRUE(model.ok()) << model.status();
    auto clipper = GradientClipper::Create(*executor_, **model);
    ASSERT_TRUE(clipper.ok()) << clipper.status();
    ASSERT_TRUE((*clipper)->Clip().ok());
    auto first = Download((*model)->gradients_[0]);
    auto second = Download((*model)->gradients_[1]);
    ASSERT_TRUE(first.ok()) << first.status();
    ASSERT_TRUE(second.ok()) << second.status();
    // Legacy fmin(1, NaN) returns 1. Clipping therefore leaves every tensor
    // unchanged, including the NaN's bits; it does not repair bad gradients.
    EXPECT_EQ(std::memcmp(first->data(), values.data(), first->size_bytes()),
              0);
    EXPECT_EQ((*second)[0], 6.0f);
  }
}

TEST_F(GradientClipperTest, InfiniteNormPreservesLegacyZeroScalingBehavior) {
  const float infinity = std::numeric_limits<float>::infinity();
  auto model = MakeModel({{infinity, 3.0f, -4.0f}, {-infinity, 2.0f}});
  ASSERT_TRUE(model.ok()) << model.status();
  auto clipper = GradientClipper::Create(*executor_, **model);
  ASSERT_TRUE(clipper.ok()) << clipper.status();
  ASSERT_TRUE((*clipper)->Clip().ok());
  auto first = Download((*model)->gradients_[0]);
  auto second = Download((*model)->gradients_[1]);
  ASSERT_TRUE(first.ok()) << first.status();
  ASSERT_TRUE(second.ok()) << second.status();
  // With no input NaN, the norm is infinite and the factor is zero. Preserve
  // IEEE Inf*0 -> NaN and the signs of the finite gradients' resulting zeros.
  EXPECT_TRUE(std::isnan((*first)[0]));
  EXPECT_TRUE(std::isnan((*second)[0]));
  EXPECT_EQ((*first)[1], 0.0f);
  EXPECT_FALSE(std::signbit((*first)[1]));
  EXPECT_EQ((*first)[2], 0.0f);
  EXPECT_TRUE(std::signbit((*first)[2]));
  EXPECT_EQ((*second)[1], 0.0f);
}

TEST_F(GradientClipperTest, ReusesScratchAcrossQueuedUpdatesWithoutHostWait) {
  auto model = MakeModel({{3.0f, 4.0f}});
  ASSERT_TRUE(model.ok()) << model.status();
  auto clipper = GradientClipper::Create(*executor_, **model);
  ASSERT_TRUE(clipper.ok()) << clipper.status();
  ASSERT_TRUE((*clipper)->Clip().ok());
  // Queue a new gradient and another clip before the first readback/wait.
  ASSERT_TRUE(
      Upload((*model)->gradients_[0], std::vector<float>{-8.0f, 6.0f}).ok());
  ASSERT_TRUE((*clipper)->Clip().ok());
  auto actual = Download((*model)->gradients_[0]);
  ASSERT_TRUE(actual.ok()) << actual.status();
  EXPECT_NEAR((*actual)[0], -0.8f, 1e-6f);
  EXPECT_NEAR((*actual)[1], 0.6f, 1e-6f);
}

TEST_F(GradientClipperTest, RejectsInvalidLimitsAndParameterStorage) {
  auto model = MakeModel({{1.0f}});
  ASSERT_TRUE(model.ok()) << model.status();
  for (float invalid : {0.0f, -1.0f, std::numeric_limits<float>::infinity(),
                        std::numeric_limits<float>::quiet_NaN()})
    EXPECT_EQ(
        GradientClipper::Create(*executor_, **model, invalid).status().code(),
        absl::StatusCode::kInvalidArgument);
  GradientModel empty;
  EXPECT_EQ(GradientClipper::Create(*executor_, empty).status().code(),
            absl::StatusCode::kInvalidArgument);
  (*model)->weights_.push_back((*model)->weights_.front());
  EXPECT_EQ(GradientClipper::Create(*executor_, **model).status().code(),
            absl::StatusCode::kInvalidArgument);
  (*model)->weights_.pop_back();
  auto mismatched = MakeBuffer(std::vector<float>{1.0f, 2.0f});
  ASSERT_TRUE(mismatched.ok()) << mismatched.status();
  (*model)->gradients_[0] = std::move(*mismatched);
  EXPECT_EQ(GradientClipper::Create(*executor_, **model).status().code(),
            absl::StatusCode::kInvalidArgument);
  auto other_executor = cuda::Executor::Create();
  ASSERT_TRUE(other_executor.ok()) << other_executor.status();
  auto foreign = Buffer::Allocate(**other_executor, sizeof(float));
  ASSERT_TRUE(foreign.ok()) << foreign.status();
  (*model)->gradients_[0] = *foreign;
  EXPECT_EQ(GradientClipper::Create(*executor_, **model).status().code(),
            absl::StatusCode::kInvalidArgument);
  // Release the fake model's foreign handle before its executor.
  (*model)->gradients_.clear();
}

}  // namespace
}  // namespace pluto::llm
