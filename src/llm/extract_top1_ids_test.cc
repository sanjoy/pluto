#include "src/llm/extract_top1_ids.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "gtest/gtest.h"
#include "src/cuda/page_locked_host_array.h"
#include "src/util/status_macros.h"

namespace pluto::llm {
namespace {

// Deliberately scalar reference: any invalid logical value rejects the row,
// and strict greater-than naturally preserves the first (lowest) tied ID.
int ReferenceTop1(absl::Span<const float> row, int target) {
  if (target == -1)
    return -1;
  int best = 0;
  for (int id = 0; id < static_cast<int>(row.size()); ++id) {
    if (!std::isfinite(row[id]))
      return -2;
    if (row[id] > row[best])
      best = id;
  }
  return best;
}

class ExtractTop1IdsTest : public testing::Test {
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

  template <class T>
  absl::StatusOr<cuda::Buffer> Upload(const std::vector<T>& values) {
    ASSIGN_OR_RETURN(auto staging,
                     cuda::PageLockedHostArray<T>::CopyFrom(*executor_, values));
    ASSIGN_OR_RETURN(auto device,
                     cuda::Buffer::Allocate(*executor_, staging.size_bytes()));
    RETURN_IF_ERROR(cuda::CudaStatus(
        cudaMemcpyAsync(device.data(), staging.data(), staging.size_bytes(),
                        cudaMemcpyHostToDevice, executor_->stream()),
        "upload top-1 fixture"));
    return device;
  }

  absl::StatusOr<cuda::PageLockedHostArray<int>> Download(
      const cuda::Buffer& device) {
    ASSIGN_OR_RETURN(auto host,
                     cuda::PageLockedHostArray<int>::Allocate(
                         *executor_, device.size_bytes() / sizeof(int)));
    RETURN_IF_ERROR(cuda::CudaStatus(
        cudaMemcpyAsync(host.data(), device.data(), device.size_bytes(),
                        cudaMemcpyDeviceToHost, executor_->stream()),
        "download top-1 result"));
    RETURN_IF_ERROR(executor_->Synchronize());
    return host;
  }

  void Check(const std::vector<float>& values, const std::vector<int>& targets,
             int vocabulary_size, int stride) {
    ASSERT_EQ(values.size(), targets.size() * stride);
    auto logits = Upload(values);
    auto device_targets = Upload(targets);
    ASSERT_TRUE(logits.ok()) << logits.status();
    ASSERT_TRUE(device_targets.ok()) << device_targets.status();
    // Repeated launches must produce exactly the same IDs, not merely close
    // floating-point results. Also exercise output allocation reuse/order.
    for (int repetition = 0; repetition < 3; ++repetition) {
      SCOPED_TRACE(repetition);
      auto ids =
          ExtractTop1Ids(*executor_, *logits, *device_targets, vocabulary_size);
      ASSERT_TRUE(ids.ok()) << ids.status();
      EXPECT_EQ(&ids->executor(), executor_.get());
      EXPECT_EQ(ids->size_bytes(), targets.size() * sizeof(int));
      auto actual = Download(*ids);
      ASSERT_TRUE(actual.ok()) << actual.status();
      for (size_t row = 0; row < targets.size(); ++row) {
        SCOPED_TRACE(row);
        EXPECT_EQ(
            (*actual)[row],
            ReferenceTop1(absl::MakeConstSpan(values.data() + row * stride,
                                              vocabulary_size),
                          targets[row]));
      }
    }
  }

  std::unique_ptr<cuda::Executor> executor_;
};

TEST_F(ExtractTop1IdsTest, MasksPromptsAndVocabularyPadding) {
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const float inf = std::numeric_limits<float>::infinity();
  // Five real tokens in each eight-wide row. Padding can contain anything,
  // including NaN or infinity, without invalidating the logical row.
  Check({-3,  -3,  -3,   -3,  2,   1000, nan, inf,  -3,  -2,  -2,
         -3,  -3,  1000, nan, inf, nan,  inf, -inf, nan, nan, nan,
         nan, nan, nan,  -3,  -3,  -3,   -3,  1000, nan, inf},
        {4, 1, -1, 0}, 5, 8);
}

TEST_F(ExtractTop1IdsTest, MatchesReferenceAcrossTileAndVocabularySizes) {
  for (const auto& [vocabulary, stride] : {std::pair{1, 1},
                                           {5, 7},
                                           {255, 259},
                                           {256, 256},
                                           {257, 271},
                                           {4475, 4480},
                                           {50257, 50272}}) {
    SCOPED_TRACE(vocabulary);
    constexpr int kRows = 7;
    std::vector<float> values(kRows * stride,
                              std::numeric_limits<float>::quiet_NaN());
    for (int row = 0; row < kRows; ++row)
      for (int token = 0; token < vocabulary; ++token)
        values[row * stride + token] =
            static_cast<float>((token * 37 + row * 13) % 509 - 700);
    // Ensure both the first and last logical positions can win; specifically
    // include the tail of a partial tile and an all-negative vocabulary.
    values[0] = -1;
    values[stride + vocabulary - 1] = -1;
    Check(values, {0, 0, -1, 0, 0, -1, 0}, vocabulary, stride);
  }
}

TEST_F(ExtractTop1IdsTest, TiesAcrossTilesAndLanesUseLowestId) {
  constexpr int kVocabulary = 1031;
  constexpr int kStride = 1033;
  std::vector<float> values(3 * kStride, -100);
  values[255] = values[256] = values[1024] = 17;
  values[kStride + 17] = values[kStride + 273] = 2;
  values[kStride + 400] = values[kStride + 1030] = 2;
  // An exact zero tie includes mixed signs of zero and crosses tile bounds.
  std::fill(values.begin() + 2 * kStride, values.end(), -0.0f);
  values[2 * kStride + 513] = 0.0f;
  Check(values, {0, 0, 0}, kVocabulary, kStride);
}

TEST_F(ExtractTop1IdsTest, RejectsEveryNonfiniteLogicalLogitButNotMaskedRows) {
  constexpr int kVocabulary = 515;
  constexpr int kStride = 519;
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const float inf = std::numeric_limits<float>::infinity();
  std::vector<float> values(7 * kStride, -5);
  values[0] = nan;
  values[kStride + 255] = inf;
  values[2 * kStride + 256] = -inf;
  values[3 * kStride + 514] = nan;
  std::fill(values.begin() + 4 * kStride, values.begin() + 5 * kStride, -inf);
  std::fill(values.begin() + 5 * kStride, values.begin() + 6 * kStride, nan);
  values[6 * kStride] = inf;
  Check(values, {0, 0, 0, 0, 0, -1, -1}, kVocabulary, kStride);
}

TEST_F(ExtractTop1IdsTest, FiniteExtremesAndSignedZeroRemainValid) {
  const float smallest = std::numeric_limits<float>::lowest();
  const float largest = std::numeric_limits<float>::max();
  const float tiny = std::numeric_limits<float>::denorm_min();
  Check({smallest, smallest, smallest, smallest, largest, largest, -0.0f, 0.0f,
         -0.0f, 0, tiny, -tiny},
        {0, 0, 0, 0}, 3, 3);
}

TEST_F(ExtractTop1IdsTest, RejectsMalformedBufferDimensions) {
  auto logits = Upload<float>({0, 1, 2, 3, 4, 5});
  auto targets = Upload<int>({0, 0});
  auto three_targets = Upload<int>({0, 0, 0});
  auto partial_target = cuda::Buffer::Allocate(*executor_, sizeof(int) - 1);
  auto partial_logits = cuda::Buffer::Allocate(*executor_, 13);
  auto empty = cuda::Buffer::Allocate(*executor_, 0);
  ASSERT_TRUE(logits.ok());
  ASSERT_TRUE(targets.ok());
  ASSERT_TRUE(three_targets.ok());
  ASSERT_TRUE(partial_target.ok());
  ASSERT_TRUE(partial_logits.ok());
  ASSERT_TRUE(empty.ok());
  for (int vocabulary : {-1, 0, 4})
    EXPECT_EQ(ExtractTop1Ids(*executor_, *logits, *targets, vocabulary)
                  .status()
                  .code(),
              absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(ExtractTop1Ids(*executor_, *logits, *empty, 1).status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(ExtractTop1Ids(*executor_, *empty, *targets, 1).status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(
      ExtractTop1Ids(*executor_, *logits, *partial_target, 1).status().code(),
      absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(
      ExtractTop1Ids(*executor_, *partial_logits, *targets, 1).status().code(),
      absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(ExtractTop1Ids(*executor_, *partial_logits, *three_targets, 1)
                .status()
                .code(),
            absl::StatusCode::kInvalidArgument);
}

TEST_F(ExtractTop1IdsTest, RejectsInputsFromAnotherExecutor) {
  auto other = cuda::Executor::Create();
  ASSERT_TRUE(other.ok()) << other.status();
  auto local_logits = Upload<float>({0, 1, 2});
  auto local_target = Upload<int>({0});
  auto foreign_logits = cuda::Buffer::Allocate(**other, 3 * sizeof(float));
  auto foreign_target = cuda::Buffer::Allocate(**other, sizeof(int));
  ASSERT_TRUE(local_logits.ok());
  ASSERT_TRUE(local_target.ok());
  ASSERT_TRUE(foreign_logits.ok());
  ASSERT_TRUE(foreign_target.ok());
  EXPECT_EQ(ExtractTop1Ids(*executor_, *foreign_logits, *local_target, 3)
                .status()
                .code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(ExtractTop1Ids(*executor_, *local_logits, *foreign_target, 3)
                .status()
                .code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(
      ExtractTop1Ids(**other, *local_logits, *local_target, 3).status().code(),
      absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE((*other)->Synchronize().ok());
}

}  // namespace
}  // namespace pluto::llm
