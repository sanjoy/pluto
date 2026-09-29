#include "src/llm/layers/util.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <vector>

#include "gtest/gtest.h"
#include "src/llm/layers/matrix_common.h"
#include "src/llm/layers/test_util.h"

namespace pluto::llm {
namespace {

using LayerUtilTest = LayersTest;

TEST_F(LayerUtilTest, ConversionsPreserveBfloat16RoundingAndMaskTails) {
  for (int size : {1, 255, 256, 257, 513}) {
    SCOPED_TRACE(size);
    auto host = cuda::PageLockedHostArray<float>::Allocate(*executor_, size);
    ASSERT_TRUE(host.ok());
    for (int i = 0; i < size; ++i)
      (*host)[i] = std::sin(float(i)) * 1.2345f;
    auto source = internal::AllocateFloatVector(*executor_, size);
    ASSERT_TRUE(source.ok());
    ASSERT_EQ(
        cudaMemcpyAsync(source->data(), host->data(), source->size_bytes(),
                        cudaMemcpyHostToDevice, executor_->stream()),
        cudaSuccess);
    auto packed = internal::ToBFloat16(*executor_, *source, size);
    ASSERT_TRUE(packed.ok()) << packed.status();
    EXPECT_NE(source->data(), packed->data());
    EXPECT_EQ(packed->size_bytes(), size_t(size) * sizeof(__nv_bfloat16));
    auto expanded = internal::ToFloat(*executor_, *packed, size);
    ASSERT_TRUE(expanded.ok()) << expanded.status();
    EXPECT_NE(expanded->data(), source->data());
    EXPECT_EQ(&expanded->executor(), executor_.get());
    auto result = cuda::PageLockedHostArray<float>::Allocate(*executor_, size);
    ASSERT_TRUE(result.ok());
    ASSERT_EQ(cudaMemcpyAsync(result->data(), expanded->data(),
                              expanded->size_bytes(), cudaMemcpyDeviceToHost,
                              executor_->stream()),
              cudaSuccess);
    ASSERT_TRUE(executor_->Synchronize().ok());
    for (int i = 0; i < size; ++i)
      EXPECT_EQ((*result)[i], static_cast<float>(__nv_bfloat16((*host)[i])));
  }
}

TEST_F(LayerUtilTest, ValidationRejectsBadSizesCountsAndExecutors) {
  auto source = internal::AllocateFloatVector(*executor_, 3);
  auto bf16 = Buffer::Allocate(*executor_, 3 * sizeof(__nv_bfloat16));
  auto other = cuda::Executor::Create();
  ASSERT_TRUE(source.ok());
  ASSERT_TRUE(bf16.ok());
  ASSERT_TRUE(other.ok());
  EXPECT_TRUE(internal::ValidateBFloat16Inputs(*executor_, {*bf16}, {3}).ok());
  EXPECT_FALSE(internal::ValidateBFloat16Inputs(*executor_, {}, {3}).ok());
  EXPECT_FALSE(internal::ValidateBFloat16Inputs(*executor_, {*bf16}, {}).ok());
  EXPECT_FALSE(internal::ValidateBFloat16Inputs(*executor_, {*bf16}, {2}).ok());
  EXPECT_FALSE(
      internal::ValidateBFloat16Inputs(*executor_, {*source}, {3}).ok());
  EXPECT_FALSE(internal::ValidateBFloat16Inputs(**other, {*bf16}, {3}).ok());
  for (int size : {-1, 0}) {
    EXPECT_FALSE(internal::AllocateFloatVector(*executor_, size).ok());
    EXPECT_FALSE(internal::ToFloat(*executor_, *bf16, size).ok());
    EXPECT_FALSE(internal::ToBFloat16(*executor_, *source, size).ok());
    EXPECT_FALSE(
        internal::ValidateBFloat16Inputs(*executor_, {*bf16}, {size}).ok());
  }
  EXPECT_FALSE(internal::ToFloat(*executor_, *bf16, 2).ok());
  EXPECT_FALSE(internal::ToBFloat16(*executor_, *source, 2).ok());
  EXPECT_FALSE(internal::ToFloat(**other, *bf16, 3).ok());
  EXPECT_FALSE(internal::ToBFloat16(**other, *source, 3).ok());
  EXPECT_TRUE((*other)->Synchronize().ok());
}

TEST_F(LayerUtilTest, AllocationDoesNotImposeAnUnrelatedLayersShapeLimit) {
  // The old shared inference limit was 2^20; individual layer factories now
  // own their restrictions instead of imposing them on unrelated helpers.
  auto buffer = internal::AllocateFloatVector(*executor_, 1048577);
  ASSERT_TRUE(buffer.ok()) << buffer.status();
  EXPECT_EQ(buffer->size_bytes(), size_t(1048577) * sizeof(float));
}

TEST(MatrixCommonTest, ElementSizeReflectsPhysicalStorageAndRejectsUnknown) {
  ASSERT_EQ(*MatrixElementBytes(MatrixStorage::kFloat32), 4);
  ASSERT_EQ(*MatrixElementBytes(MatrixStorage::kBFloat16), 2);
  ASSERT_EQ(*MatrixElementBytes(MatrixStorage::kFp8E4M3), 1);
  EXPECT_EQ(MatrixElementBytes(static_cast<MatrixStorage>(-1)).status().code(),
            absl::StatusCode::kInvalidArgument);
}

}  // namespace
}  // namespace pluto::llm
