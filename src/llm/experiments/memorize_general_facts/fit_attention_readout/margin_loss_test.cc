#include "src/llm/experiments/memorize_general_facts/fit_attention_readout/margin_loss.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "absl/types/span.h"
#include "gtest/gtest.h"
#include "src/cuda/page_locked_host_array.h"
#include "src/util/status_macros.h"

namespace pluto::llm::fit_attention_readout {
namespace {

struct HostResult {
  std::vector<float> losses;
  std::vector<float> gradients;
  std::vector<int> predictions;
  std::vector<float> margins;
};

// Deliberately scalar, keeping the earliest ID whenever values are tied.
HostResult Reference(const std::vector<float>& logits,
                     const std::vector<int>& targets, int vocabulary,
                     float margin, int normalizer) {
  const size_t stride = logits.size() / targets.size();
  HostResult result{
      std::vector<float>(targets.size()), std::vector<float>(logits.size()),
      std::vector<int>(targets.size(), -1), std::vector<float>(targets.size())};
  for (size_t row = 0; row < targets.size(); ++row) {
    const int target = targets[row];
    if (target == -1)
      continue;
    bool invalid = target < 0 || target >= vocabulary;
    for (int id = 0; id < vocabulary; ++id)
      invalid |= !std::isfinite(logits[row * stride + id]);
    if (invalid) {
      const float nan = std::numeric_limits<float>::quiet_NaN();
      result.losses[row] = result.margins[row] = nan;
      result.predictions[row] = -2;
      std::fill_n(result.gradients.begin() + row * stride, vocabulary, nan);
      continue;
    }
    int best = 0;
    int competitor = target == 0 ? 1 : 0;
    for (int id = 0; id < vocabulary; ++id) {
      if (logits[row * stride + id] > logits[row * stride + best])
        best = id;
      if (id != target &&
          logits[row * stride + id] > logits[row * stride + competitor])
        competitor = id;
    }
    const float actual_margin =
        logits[row * stride + target] - logits[row * stride + competitor];
    const float violation = std::max(0.0f, margin - actual_margin);
    result.losses[row] = violation * violation;
    result.margins[row] = actual_margin;
    result.predictions[row] = best;
    result.gradients[row * stride + competitor] = 2 * violation / normalizer;
    result.gradients[row * stride + target] = -2 * violation / normalizer;
  }
  return result;
}

class MarginLossTest : public testing::Test {
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
    ASSIGN_OR_RETURN(auto host,
                     cuda::PageLockedHostArray<T>::CopyFrom(*executor_, values));
    ASSIGN_OR_RETURN(auto device,
                     cuda::Buffer::Allocate(*executor_, host.size_bytes()));
    RETURN_IF_ERROR(cuda::CudaStatus(
        cudaMemcpyAsync(device.data(), host.data(), host.size_bytes(),
                        cudaMemcpyHostToDevice, executor_->stream()),
        "upload margin loss fixture"));
    return device;
  }

  template <class T>
  absl::StatusOr<std::vector<T>> CopyD2H(const cuda::Buffer& device) {
    ASSIGN_OR_RETURN(auto host,
                     cuda::PageLockedHostArray<T>::Allocate(
                         *executor_, device.size_bytes() / sizeof(T)));
    RETURN_IF_ERROR(cuda::CudaStatus(
        cudaMemcpyAsync(host.data(), device.data(), device.size_bytes(),
                        cudaMemcpyDeviceToHost, executor_->stream()),
        "download margin loss result"));
    RETURN_IF_ERROR(executor_->Synchronize());
    return std::vector<T>(host.begin(), host.end());
  }

  absl::StatusOr<HostResult> Run(const std::vector<float>& values,
                                 const std::vector<int>& targets,
                                 int vocabulary, float margin, int normalizer) {
    ASSIGN_OR_RETURN(auto logits, Upload(values));
    ASSIGN_OR_RETURN(auto target_ids, Upload(targets));
    ASSIGN_OR_RETURN(auto result,
                     SquaredMarginLoss(*executor_, logits, target_ids,
                                       vocabulary, margin, normalizer));
    ASSIGN_OR_RETURN(auto losses, CopyD2H<float>(result.losses));
    ASSIGN_OR_RETURN(auto gradients, CopyD2H<float>(result.gradients));
    ASSIGN_OR_RETURN(auto predictions, CopyD2H<int>(result.predicted_ids));
    ASSIGN_OR_RETURN(auto margins, CopyD2H<float>(result.margins));
    return HostResult{std::move(losses), std::move(gradients),
                      std::move(predictions), std::move(margins)};
  }

  void Check(const std::vector<float>& values, const std::vector<int>& targets,
             int vocabulary, float margin, int normalizer) {
    const auto expected =
        Reference(values, targets, vocabulary, margin, normalizer);
    for (int repetition = 0; repetition < 3; ++repetition) {
      auto actual = Run(values, targets, vocabulary, margin, normalizer);
      ASSERT_TRUE(actual.ok()) << actual.status();
      EXPECT_EQ(actual->predictions, expected.predictions);
      for (const auto& [got, want] :
           {std::pair{&actual->losses, &expected.losses},
            std::pair{&actual->gradients, &expected.gradients},
            std::pair{&actual->margins, &expected.margins}}) {
        ASSERT_EQ(got->size(), want->size());
        for (size_t i = 0; i < want->size(); ++i) {
          SCOPED_TRACE(i);
          if (std::isnan((*want)[i]))
            EXPECT_TRUE(std::isnan((*got)[i]));
          else
            EXPECT_NEAR((*got)[i], (*want)[i],
                        1e-6f * std::max(1.0f, std::abs((*want)[i])));
        }
      }
    }
  }

  std::unique_ptr<cuda::Executor> executor_;
};

TEST_F(MarginLossTest, CorrectIncorrectTiedIgnoredAndPaddedRows) {
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const float inf = std::numeric_limits<float>::infinity();
  Check(
      {3,   0,    -1,  1000, nan,   // Correct with more than required margin.
       1,   2,    -4,  inf,  1000,  // Incorrect, competitor ID 1.
       1,   1,    1,   nan,  inf,  // Three-way tie, competitor/prediction ID 0.
       nan, inf,  nan, inf,  0,    // Ignored: logits must not be inspected.
       0,   0.5f, -1,  1000, 1},
      {0, 0, 2, -1, 1}, 3, 1.0f, 4);
}

TEST_F(MarginLossTest, ExplicitNormalizerScalesGradientsNotLosses) {
  const std::vector<float> logits{0, 2, -1, 1, -2, 0};
  const std::vector<int> targets{0, 2};
  auto one = Run(logits, targets, 3, 0.5f, 1);
  auto seven = Run(logits, targets, 3, 0.5f, 7);
  ASSERT_TRUE(one.ok()) << one.status();
  ASSERT_TRUE(seven.ok()) << seven.status();
  EXPECT_EQ(one->losses, seven->losses);
  EXPECT_EQ(one->margins, seven->margins);
  for (size_t i = 0; i < logits.size(); ++i)
    EXPECT_FLOAT_EQ(one->gradients[i] / 7, seven->gradients[i]);
}

TEST_F(MarginLossTest, ExactHingeBoundaryAndLowestTargetTie) {
  // The first two rows meet the requested margin exactly, so their loss and
  // gradients are zero. The last has correct top-1 via ID tie breaking but a
  // positive margin violation, whose competitor is the other tied token.
  Check({2, 1, 0, 100, -3, -2, -1, 100, 1, 1, 0, 100}, {0, 2, 0}, 3, 1.0f, 3);
}

TEST_F(MarginLossTest, PartialTilesAndLowestIdTiesAcrossTiles) {
  for (int vocabulary : {2, 5, 255, 256, 257, 1031, 4475}) {
    SCOPED_TRACE(vocabulary);
    const int stride = vocabulary + 5;
    std::vector<float> values(2 * stride,
                              std::numeric_limits<float>::quiet_NaN());
    for (int row = 0; row < 2; ++row)
      for (int id = 0; id < vocabulary; ++id)
        values[row * stride + id] = -10.0f;
    values[1] = values[vocabulary - 1] = 4;
    values[stride] = values[stride + vocabulary - 1] = 4;
    if (vocabulary > 256)
      values[256] = 4;
    Check(values, {0, vocabulary - 1}, vocabulary, 0.5f, 2);
  }
}

TEST_F(MarginLossTest, AllIgnoredRowsHaveZeroLossAndGradients) {
  Check(std::vector<float>(24, std::numeric_limits<float>::quiet_NaN()),
        {-1, -1, -1}, 5, 1.0f, 1);
}

TEST_F(MarginLossTest, InvalidDeviceValuesAreExplicitRatherThanSilentSuccess) {
  const float inf = std::numeric_limits<float>::infinity();
  const float nan = std::numeric_limits<float>::quiet_NaN();
  Check({0, 1,    0, 1000, 0, 1,    0, 1000, nan,  0,
         1, 1000, 0, inf,  1, 1000, 0, 1,    -inf, 1000},
        {-2, 3, 0, 0, 1}, 3, 1.0f, 5);
}

TEST_F(MarginLossTest, GradientMatchesFiniteDifferencesAwayFromTiesAndHinge) {
  const std::vector<float> logits{0.2f, 1.5f, -1, 100, 2, -2, 0, 100};
  const std::vector<int> targets{0, 2};
  constexpr int kNormalizer = 2;
  auto base = Run(logits, targets, 3, 0.7f, kNormalizer);
  ASSERT_TRUE(base.ok()) << base.status();
  constexpr float kEpsilon = 0.01f;
  for (size_t index = 0; index < logits.size(); ++index) {
    auto plus = logits, minus = logits;
    plus[index] += kEpsilon;
    minus[index] -= kEpsilon;
    auto forward = Run(plus, targets, 3, 0.7f, kNormalizer);
    auto backward = Run(minus, targets, 3, 0.7f, kNormalizer);
    ASSERT_TRUE(forward.ok()) << forward.status();
    ASSERT_TRUE(backward.ok()) << backward.status();
    float difference = 0;
    for (size_t row = 0; row < targets.size(); ++row)
      difference += forward->losses[row] - backward->losses[row];
    const float numerical = difference / (2 * kEpsilon * kNormalizer);
    EXPECT_NEAR(base->gradients[index], numerical, 5e-4f) << index;
  }
}

TEST_F(MarginLossTest, RejectsInvalidArgumentsAndShapes) {
  auto logits = Upload<float>({0, 1, 2, 3, 4, 5});
  auto targets = Upload<int>({0, 1});
  auto empty = cuda::Buffer::Allocate(*executor_, 0);
  auto partial_logits = cuda::Buffer::Allocate(*executor_, 13);
  auto partial_target = cuda::Buffer::Allocate(*executor_, 3);
  ASSERT_TRUE(logits.ok());
  ASSERT_TRUE(targets.ok());
  ASSERT_TRUE(empty.ok());
  ASSERT_TRUE(partial_logits.ok());
  ASSERT_TRUE(partial_target.ok());
  for (int vocabulary : {-1, 0, 1, 4})
    EXPECT_EQ(SquaredMarginLoss(*executor_, *logits, *targets, vocabulary, 1, 2)
                  .status()
                  .code(),
              absl::StatusCode::kInvalidArgument);
  for (float margin : {0.0f, -1.0f, std::numeric_limits<float>::infinity(),
                       std::numeric_limits<float>::quiet_NaN()})
    EXPECT_EQ(SquaredMarginLoss(*executor_, *logits, *targets, 3, margin, 2)
                  .status()
                  .code(),
              absl::StatusCode::kInvalidArgument);
  for (int normalizer : {0, -1})
    EXPECT_EQ(SquaredMarginLoss(*executor_, *logits, *targets, 3, 1, normalizer)
                  .status()
                  .code(),
              absl::StatusCode::kInvalidArgument);
  for (const cuda::Buffer* values : {&*empty, &*partial_logits})
    EXPECT_EQ(SquaredMarginLoss(*executor_, *values, *targets, 3, 1, 2)
                  .status()
                  .code(),
              absl::StatusCode::kInvalidArgument);
  for (const cuda::Buffer* ids : {&*empty, &*partial_target})
    EXPECT_EQ(
        SquaredMarginLoss(*executor_, *logits, *ids, 3, 1, 2).status().code(),
        absl::StatusCode::kInvalidArgument);
}

TEST_F(MarginLossTest, RejectsAnotherExecutor) {
  auto other = cuda::Executor::Create();
  ASSERT_TRUE(other.ok()) << other.status();
  auto logits = Upload<float>({0, 1});
  auto targets = Upload<int>({0});
  ASSERT_TRUE(logits.ok());
  ASSERT_TRUE(targets.ok());
  auto foreign = cuda::Buffer::Allocate(**other, sizeof(float) * 2);
  ASSERT_TRUE(foreign.ok());
  EXPECT_EQ(SquaredMarginLoss(*executor_, *foreign, *targets, 2, 1, 1)
                .status()
                .code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(
      SquaredMarginLoss(**other, *logits, *targets, 2, 1, 1).status().code(),
      absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE((*other)->Synchronize().ok());
}

}  // namespace
}  // namespace pluto::llm::fit_attention_readout
