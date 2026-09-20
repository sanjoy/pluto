#include "src/llm/batch_validation.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "gtest/gtest.h"

namespace pluto::llm {
namespace {

constexpr int64_t kBatch = ActivationType::kBatchDimension;

class BatchValidationTest : public testing::Test {
 protected:
  void SetUp() override {
    auto executor = cuda::Executor::Create();
    ASSERT_TRUE(executor.ok()) << executor.status();
    executor_ = std::move(*executor);
    auto inputs = Buffer::Allocate(*executor_, 2 * 4 * 3 * sizeof(float));
    auto targets = Buffer::Allocate(*executor_, 2 * 4 * sizeof(int32_t));
    ASSERT_TRUE(inputs.ok()) << inputs.status();
    ASSERT_TRUE(targets.ok()) << targets.status();
    batch_.emplace(DataBatch{.inputs = std::move(*inputs),
                             .targets = std::move(*targets),
                             .batch_size = 2,
                             .sequence_length = 4});
  }

  // The executor outlives every batch buffer.
  std::unique_ptr<cuda::Executor> executor_;
  std::optional<DataBatch> batch_;
};

TEST_F(BatchValidationTest, ChecksShapeWithoutInferringBatchFromBytes) {
  const ActivationType type(DataType::FP32, {kBatch, 4, 3});
  EXPECT_TRUE(
      ValidateBatchInput(*executor_, *batch_, batch_->inputs, type, "input")
          .ok());
  // Same flattened bytes, different sample boundaries: 2x4 is not 1x8.
  batch_->batch_size = 1;
  batch_->sequence_length = 8;
  EXPECT_EQ(
      ValidateBatchInput(*executor_, *batch_, batch_->inputs, type, "input")
          .code(),
      absl::StatusCode::kInvalidArgument);

  batch_->sequence_length = 4;
  // Never infer a second, larger batch size from the buffer to accept it.
  const auto wrong_batch = ValidateBatchInput(
      *executor_, *batch_, batch_->inputs, type, "encoder input");
  EXPECT_EQ(wrong_batch.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_NE(wrong_batch.message().find("encoder input"),
            absl::string_view::npos);
  EXPECT_NE(wrong_batch.message().find("expected 48"), absl::string_view::npos);
}

TEST_F(BatchValidationTest,
       UsesSameBatchForAllOutputsAndAllowsUnbatchedWeights) {
  auto decoder = Buffer::Allocate(*executor_, 3 * 5 * sizeof(float));
  ASSERT_TRUE(decoder.ok()) << decoder.status();
  const ActivationType types[] = {{DataType::FP32, {kBatch, 4, 3}},
                                  {DataType::INT32, {kBatch, 4}},
                                  {DataType::FP32, {3, 5}}};
  BufferVec buffers{batch_->inputs, batch_->targets, *decoder};
  EXPECT_TRUE(
      ValidateBatchBuffers(*executor_, *batch_, buffers, types, "outputs")
          .ok());
  auto too_many_rows = Buffer::Allocate(*executor_, 3 * 4 * sizeof(int32_t));
  ASSERT_TRUE(too_many_rows.ok()) << too_many_rows.status();
  buffers[1] = *too_many_rows;
  EXPECT_EQ(ValidateBatchBuffers(*executor_, *batch_, buffers, types, "outputs")
                .code(),
            absl::StatusCode::kInvalidArgument);
}

TEST_F(BatchValidationTest, ScalarsAndFixedTensorsDoNotHaveSequenceAxes) {
  auto scalar = Buffer::Allocate(*executor_, sizeof(float));
  auto matrix = Buffer::Allocate(*executor_, 3 * 7 * sizeof(float));
  ASSERT_TRUE(scalar.ok()) << scalar.status();
  ASSERT_TRUE(matrix.ok()) << matrix.status();
  const ActivationType types[] = {{DataType::FP32, {}},
                                  {DataType::FP32, {3, 7}}};
  EXPECT_TRUE(ValidateBatchBuffers(*executor_, *batch_,
                                   BufferVec{*scalar, *matrix}, types,
                                   "parameters")
                  .ok());
  // Only actual dataset inputs/targets require a batch + sequence prefix.
  EXPECT_EQ(ValidateBatchInput(*executor_, *batch_, *scalar, types[0], "target")
                .code(),
            absl::StatusCode::kInvalidArgument);
  const ActivationType fixed_batch(DataType::FP32, {2, 4, 3});
  EXPECT_EQ(ValidateBatchInput(*executor_, *batch_, batch_->inputs, fixed_batch,
                               "input")
                .code(),
            absl::StatusCode::kInvalidArgument);
}

TEST_F(BatchValidationTest, CountsPhysicalElementBytesForEveryDataType) {
  for (DataType dtype : {DataType::FP8, DataType::FP16, DataType::BF16,
                         DataType::FP32, DataType::INT32}) {
    const size_t element_bytes =
        dtype == DataType::FP8                                 ? 1
        : (dtype == DataType::FP16 || dtype == DataType::BF16) ? 2
                                                               : 4;
    auto good = Buffer::Allocate(*executor_, 2 * 4 * 3 * element_bytes);
    auto short_buffer =
        Buffer::Allocate(*executor_, 2 * 4 * 3 * element_bytes - 1);
    ASSERT_TRUE(good.ok()) << good.status();
    ASSERT_TRUE(short_buffer.ok()) << short_buffer.status();
    const ActivationType type(dtype, {kBatch, 4, 3});
    EXPECT_TRUE(
        ValidateBatchInput(*executor_, *batch_, *good, type, "input").ok());
    EXPECT_EQ(
        ValidateBatchInput(*executor_, *batch_, *short_buffer, type, "input")
            .code(),
        absl::StatusCode::kInvalidArgument);
  }
}

TEST_F(BatchValidationTest, RejectsArityAndExecutorMismatch) {
  const ActivationType types[] = {{DataType::FP32, {kBatch, 4, 3}}};
  EXPECT_EQ(
      ValidateBatchBuffers(*executor_, *batch_, {}, types, "input").code(),
      absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(ValidateBatchBuffers(*executor_, *batch_, BufferVec{batch_->inputs},
                                 {}, "input")
                .code(),
            absl::StatusCode::kInvalidArgument);
  auto other = cuda::Executor::Create();
  ASSERT_TRUE(other.ok()) << other.status();
  auto foreign = Buffer::Allocate(**other, batch_->inputs.size_bytes());
  ASSERT_TRUE(foreign.ok()) << foreign.status();
  EXPECT_EQ(ValidateBatchInput(*executor_, *batch_, *foreign, types[0], "input")
                .code(),
            absl::StatusCode::kInvalidArgument);
}

TEST_F(BatchValidationTest, RejectsInvalidMetadataAndShapeOverflow) {
  const ActivationType valid[] = {{DataType::FP32, {kBatch, 4, 3}}};
  for (int batch_size : {0, -1}) {
    batch_->batch_size = batch_size;
    EXPECT_EQ(ValidateBatchTypes(*batch_, valid, "input").code(),
              absl::StatusCode::kInvalidArgument);
  }
  batch_->batch_size = 2;
  for (int length : {0, -1, std::numeric_limits<int>::max()}) {
    batch_->sequence_length = length;
    EXPECT_EQ(ValidateBatchTypes(*batch_, valid, "input").code(),
              absl::StatusCode::kInvalidArgument);
  }
  batch_->sequence_length = 4;
  const std::vector<ActivationType> invalid = {
      {DataType::FP32, {kBatch}},
      {DataType::FP32, {kBatch, 0, 3}},
      {DataType::FP32, {kBatch, 4, -1}},
      {DataType::FP32, {2, kBatch, 3}},
      {static_cast<DataType>(999), {kBatch, 4, 3}},
      {DataType::FP32, {std::numeric_limits<int64_t>::max()}},
      {DataType::FP8, {std::numeric_limits<int64_t>::max(), 3}},
      {DataType::BF16, {kBatch, 4, std::numeric_limits<int64_t>::max()}}};
  for (const auto& type : invalid)
    EXPECT_EQ(
        ValidateBatchTypes(*batch_, absl::MakeConstSpan(&type, 1), "input")
            .code(),
        absl::StatusCode::kInvalidArgument);
}

}  // namespace
}  // namespace pluto::llm
