// Prepared for explicit post-training GPU validation; this target has not been
// run while the timed training arm is active. Building it is not validation.
// A future evidence controller must reject skipped tests (including all-skip).
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <utility>

#include "ai-slop/weight_analysis/head_context/probe.h"
#include "ai-slop/weight_analysis/head_context/selection.h"
#include "gtest/gtest.h"
#include "src/cuda/page_locked_host_array.h"
#include "src/llm/recipes/gpt2.h"
#include "src/util/status_macros.h"

namespace pluto::weight_analysis::head_context {
namespace {
constexpr int kContext = llm::kGpt2ContextLength;
constexpr int kWidth = llm::kGpt2ModelWidth;
constexpr int kHeadWidth = llm::kGpt2AttentionHeadDimension;
constexpr int kPadded = llm::kGpt2PaddedVocabularySize;

template <class T>
absl::StatusOr<cuda::PageLockedHostArray<T>> Download(
    cuda::Executor& executor, const cuda::Buffer& buffer) {
  if (&buffer.executor() != &executor || buffer.size_bytes() % sizeof(T)) {
    return absl::InvalidArgumentError(
        "GPU test download shape/executor differs");
  }
  ASSIGN_OR_RETURN(auto host, cuda::PageLockedHostArray<T>::Allocate(
                                  executor, buffer.size_bytes() / sizeof(T)));
  RETURN_IF_ERROR(cuda::CudaStatus(
      cudaMemcpyAsync(host.data(), buffer.data(), buffer.size_bytes(),
                      cudaMemcpyDeviceToHost, executor.stream()),
      "GPU test pinned download"));
  RETURN_IF_ERROR(executor.Synchronize());
  return host;
}

template <class T>
absl::Status Upload(cuda::Executor& executor, const cuda::Buffer& destination,
                    const cuda::PageLockedHostArray<T>& source) {
  if (&destination.executor() != &executor ||
      destination.size_bytes() != source.size_bytes()) {
    return absl::InvalidArgumentError("GPU test upload shape/executor differs");
  }
  RETURN_IF_ERROR(cuda::CudaStatus(
      cudaMemcpyAsync(destination.data(), source.data(), source.size_bytes(),
                      cudaMemcpyHostToDevice, executor.stream()),
      "GPU test pinned upload"));
  return executor.Synchronize();
}

class HeadContextGpuTest : public ::testing::Test {
 protected:
  void SetUp() override {
    int devices = 0;
    const auto availability = cudaGetDeviceCount(&devices);
    if (availability != cudaSuccess || devices == 0)
      GTEST_SKIP() << "CUDA device unavailable; no native replay validated";
    auto executor = cuda::Executor::Create();
    ASSERT_TRUE(executor.ok()) << executor.status();
    executor_ = std::move(*executor);
  }

  void Initialize(int sequences) {
    rows_ = sequences * kContext;
    auto model = llm::CreateGpt2(*executor_, llm::DataType::BF16, 17);
    ASSERT_TRUE(model.ok()) << model.status();
    model_ = std::move(*model);
    auto tokens =
        cuda::PageLockedHostArray<int32_t>::Allocate(*executor_, rows_);
    ASSERT_TRUE(tokens.ok()) << tokens.status();
    tokens_ = std::move(*tokens);
    for (int i = 0; i < rows_; ++i)
      tokens_[i] = (17 * i + 3) % 1000;
    auto input = cuda::Buffer::Allocate(*executor_, tokens_.size_bytes());
    ASSERT_TRUE(input.ok()) << input.status();
    input_ = std::move(*input);
    ASSERT_TRUE(Upload(*executor_, *input_, tokens_).ok());
    auto clean = model_->fwd(*executor_, absl::MakeConstSpan(&*input_, 1));
    if (clean.ok()) state_ = std::move(clean->state);
    ASSERT_TRUE(clean.ok()) << clean.status();
    clean_ = std::move(clean->outputs[0]);
    auto clean_values = Download<float>(*executor_, *clean_);
    ASSERT_TRUE(clean_values.ok()) << clean_values.status();
    clean_values_ = std::move(*clean_values);
  }

  // Expected masking is specified independently from ScaleContext's loops.
  // Exact scalar BF16 arithmetic was exhaustively checked by the CPU target.
  void CheckResult(const Probe& probe, const Result& result,
                   const Selection& selection) {
    EXPECT_NE(result.context.data(), probe.original_context().data());
    EXPECT_NE(result.logits.data(), clean_->data());
    auto original = Download<uint16_t>(*executor_, probe.original_context());
    auto actual = Download<uint16_t>(*executor_, result.context);
    auto values = Download<float>(*executor_, result.logits);
    ASSERT_TRUE(original.ok()) << original.status();
    ASSERT_TRUE(actual.ok()) << actual.status();
    ASSERT_TRUE(values.ok()) << values.status();
    ASSERT_EQ(actual->size(), static_cast<size_t>(rows_) * kWidth);
    ASSERT_EQ(values->size(), static_cast<size_t>(rows_) * kPadded);
    auto expected = cuda::PageLockedHostArray<uint16_t>::CopyFrom(
        *executor_, original->span());
    ASSERT_TRUE(expected.ok()) << expected.status();
    const int first_row = selection.sequence * kContext;
    for (int row = first_row; row < first_row + kContext; ++row) {
      if ((selection.scope == QueryScope::kSelectedQuery &&
           row != first_row + selection.query_position) ||
          (selection.scope == QueryScope::kOtherQueries &&
           row == first_row + selection.query_position))
        continue;
      const size_t first =
          static_cast<size_t>(row) * kWidth + selection.head * kHeadWidth;
      for (int lane = 0; lane < kHeadWidth; ++lane) {
        auto scaled = ScaleBf16((*original)[first + lane], selection.scale);
        ASSERT_TRUE(scaled.ok()) << scaled.status();
        (*expected)[first + lane] = *scaled;
      }
    }
    EXPECT_EQ(
        std::memcmp(actual->data(), expected->data(), actual->size_bytes()), 0);
    if (selection.scale == 1) {
      EXPECT_EQ(std::memcmp(values->data(), clean_values_.data(),
                            values->size_bytes()),
                0);
    }
    for (int row = 0; row < rows_; ++row) {
      const bool other_sequence = row / kContext != selection.sequence;
      const bool earlier = selection.scope == QueryScope::kSelectedQuery &&
                           row % kContext < selection.query_position;
      const bool future_only = selection.scope == QueryScope::kOtherQueries &&
                               selection.query_position == 0 &&
                               row % kContext == 0;
      const size_t offset = static_cast<size_t>(row) * kPadded;
      if (other_sequence || earlier || future_only) {
        EXPECT_EQ(
            std::memcmp(values->data() + offset, clean_values_.data() + offset,
                        kPadded * sizeof(float)),
            0)
            << row;
      }
      constexpr int logical = llm::kGpt2VocabularySize;
      EXPECT_EQ(std::memcmp(values->data() + offset + logical,
                            clean_values_.data() + offset + logical,
                            (kPadded - logical) * sizeof(float)),
                0)
          << row;
    }
    EXPECT_TRUE(probe.VerifyOriginals(*executor_).ok());
  }

  // Only this test-owned model is modified. Restore BEFORE assertions about
  // the counterfactual forward so a failed comparison cannot leave it edited.
  void CheckProjectionZeroEquivalence(Probe& probe, int block, int head) {
    ASSERT_EQ(rows_, kContext);  // A weight edit affects all packed sequences.
    auto zero_context = probe.Apply(
        *executor_, {block, head, 0, 511, QueryScope::kAllQueries, 0});
    ASSERT_TRUE(zero_context.ok()) << zero_context.status();
    auto expected = Download<float>(*executor_, zero_context->logits);
    ASSERT_TRUE(expected.ok()) << expected.status();
    const auto matrix = model_->weights()[6 + 12 * block];
    auto backup = Download<float>(*executor_, matrix);
    ASSERT_TRUE(backup.ok()) << backup.status();
    auto patched =
        cuda::PageLockedHostArray<float>::CopyFrom(*executor_, backup->span());
    ASSERT_TRUE(patched.ok()) << patched.status();
    // W_o is physically [input_width, output_width]. Zero only the selected
    // head's 64 input rows; output bias and all other model tensors stay
    // intact.
    const size_t first = static_cast<size_t>(head * kHeadWidth) * kWidth;
    std::fill(patched->begin() + first,
              patched->begin() + first + kHeadWidth * kWidth, 0.0f);
    const auto written = Upload(*executor_, matrix, *patched);

    auto edited = model_->fwd(*executor_, absl::MakeConstSpan(&*input_, 1));

    const auto restored = Upload(*executor_, matrix, *backup);
    ASSERT_TRUE(written.ok()) << written;
    ASSERT_TRUE(restored.ok()) << restored;
    ASSERT_TRUE(edited.ok()) << edited.status();
    auto actual = Download<float>(*executor_, edited->outputs[0]);
    ASSERT_TRUE(actual.ok()) << actual.status();
    ASSERT_EQ(actual->size_bytes(), expected->size_bytes());
    EXPECT_EQ(
        std::memcmp(actual->data(), expected->data(), actual->size_bytes()), 0);
    EXPECT_TRUE(probe.VerifyOriginals(*executor_).ok());
    // No analogous half-weight equivalence is claimed: rounding a BF16
    // context by one-half need not equal rounding scaled FP32 master weights.
  }

  // Destruction order keeps model/states/device buffers ahead of the executor.
  std::unique_ptr<cuda::Executor> executor_;
  std::unique_ptr<llm::ComposedLayer> model_;
  cuda::PageLockedHostArray<int32_t> tokens_;
  std::optional<cuda::Buffer> input_, clean_;
  llm::BackwardState state_;
  cuda::PageLockedHostArray<float> clean_values_;
  int rows_ = 0;
};

TEST_F(HeadContextGpuTest, NativeScopesDosesAndIndependentProjectionZeroAgree) {
  ASSERT_NO_FATAL_FAILURE(Initialize(1));
  // Earliest block, historical sign-reversal candidate and final-block tail.
  for (const auto& [block, head] :
       {std::pair{0, 5}, std::pair{3, 6}, std::pair{7, 0}}) {
    auto probe =
        Probe::Create(*executor_, *model_, *input_, state_, *clean_, block);
    ASSERT_TRUE(probe.ok()) << probe.status();
    for (int query : {0, 511, kContext - 1}) {
      for (auto scope : {QueryScope::kSelectedQuery, QueryScope::kOtherQueries,
                         QueryScope::kAllQueries}) {
        for (float dose : {1.0f, 0.5f, 0.0f}) {
          SCOPED_TRACE(::testing::Message()
                       << block << ':' << head << ':' << query << ':'
                       << static_cast<int>(scope) << ':' << dose);
          const Selection selection{block, head, 0, query, scope, dose};
          auto result = (*probe)->Apply(*executor_, selection);
          ASSERT_TRUE(result.ok()) << result.status();
          ASSERT_NO_FATAL_FAILURE(CheckResult(**probe, *result, selection));
        }
      }
    }
    ASSERT_NO_FATAL_FAILURE(
        CheckProjectionZeroEquivalence(**probe, block, head));
    // After changing/restoring actual model weights, identity still genuinely
    // executes the native tail and must recover every original output bit.
    auto after = (*probe)->Apply(
        *executor_, {block, head, 0, 0, QueryScope::kAllQueries, 1});
    ASSERT_TRUE(after.ok()) << after.status();
  }
}

TEST_F(HeadContextGpuTest,
       OtherSequenceAndFutureOnlyQueriesCannotChangeLogits) {
  ASSERT_NO_FATAL_FAILURE(Initialize(2));
  auto probe = Probe::Create(*executor_, *model_, *input_, state_, *clean_, 3);
  ASSERT_TRUE(probe.ok()) << probe.status();
  for (int query : {0, 511}) {
    for (auto scope : {QueryScope::kSelectedQuery, QueryScope::kOtherQueries,
                       QueryScope::kAllQueries}) {
      for (float dose : {1.0f, 0.5f, 0.0f}) {
        const Selection selection{3, 6, 1, query, scope, dose};
        auto result = (*probe)->Apply(*executor_, selection);
        ASSERT_TRUE(result.ok()) << result.status();
        ASSERT_NO_FATAL_FAILURE(CheckResult(**probe, *result, selection));
      }
    }
  }
}

TEST_F(HeadContextGpuTest,
       RejectsWrongIdentityMalformedInputsAndOriginalMutation) {
  ASSERT_NO_FATAL_FAILURE(Initialize(1));
  auto probe = Probe::Create(*executor_, *model_, *input_, state_, *clean_, 3);
  ASSERT_TRUE(probe.ok()) << probe.status();
  const Selection selection{3, 6, 0, 511, QueryScope::kSelectedQuery, 0};
  auto other_executor = cuda::Executor::Create();
  ASSERT_TRUE(other_executor.ok()) << other_executor.status();
  EXPECT_FALSE((*probe)->Apply(**other_executor, selection).ok());
  EXPECT_FALSE((*probe)->VerifyOriginals(**other_executor).ok());
  EXPECT_FALSE(
      Probe::Create(**other_executor, *model_, *input_, state_, *clean_, 3)
          .ok());
  EXPECT_FALSE(Probe::Create(*executor_, *model_, *input_, llm::BackwardState{},
                             *clean_, 3)
                   .ok());
  auto wrong_shape = cuda::Buffer::Allocate(*executor_, sizeof(int32_t));
  ASSERT_TRUE(wrong_shape.ok()) << wrong_shape.status();
  EXPECT_FALSE(
      Probe::Create(*executor_, *model_, *wrong_shape, state_, *clean_, 3)
          .ok());
  auto other_input = cuda::Buffer::Allocate(*executor_, tokens_.size_bytes());
  ASSERT_TRUE(other_input.ok()) << other_input.status();
  ASSERT_TRUE(Upload(*executor_, *other_input, tokens_).ok());
  // Same bytes are insufficient: this state must retain this exact token
  // buffer.
  EXPECT_FALSE(
      Probe::Create(*executor_, *model_, *other_input, state_, *clean_, 3)
          .ok());
  auto other_model = llm::CreateGpt2(*executor_, llm::DataType::BF16, 18);
  ASSERT_TRUE(other_model.ok()) << other_model.status();
  EXPECT_FALSE(
      Probe::Create(*executor_, **other_model, *input_, state_, *clean_, 3)
          .ok());
  for (const Selection invalid :
       {Selection{2, 6, 0, 511, QueryScope::kSelectedQuery, 0},
        Selection{3, 8, 0, 511, QueryScope::kSelectedQuery, 0},
        Selection{3, 6, 1, 511, QueryScope::kSelectedQuery, 0},
        Selection{3, 6, 0, 1024, QueryScope::kSelectedQuery, 0},
        Selection{3, 6, 0, 511, QueryScope::kSelectedQuery, 0.25f}}) {
    EXPECT_FALSE((*probe)->Apply(*executor_, invalid).ok());
  }

  // Even a weight from BEFORE the selected block is protected, although it
  // is not used by the standalone replay tail. Restore before expectations.
  const auto weight = model_->weights()[1];
  auto backup = Download<float>(*executor_, weight);
  ASSERT_TRUE(backup.ok()) << backup.status();
  auto altered =
      cuda::PageLockedHostArray<float>::CopyFrom(*executor_, backup->span());
  ASSERT_TRUE(altered.ok()) << altered.status();
  (*altered)[0] += 0.25f;
  const auto weight_write = Upload(*executor_, weight, *altered);
  const auto weight_rejected = (*probe)->Apply(*executor_, selection);
  const auto weight_restored = Upload(*executor_, weight, *backup);
  ASSERT_TRUE(weight_write.ok()) << weight_write;
  ASSERT_TRUE(weight_restored.ok()) << weight_restored;
  EXPECT_FALSE(weight_rejected.ok());
  EXPECT_TRUE((*probe)->VerifyOriginals(*executor_).ok());

  auto changed_tokens =
      cuda::PageLockedHostArray<int32_t>::CopyFrom(*executor_, tokens_.span());
  ASSERT_TRUE(changed_tokens.ok()) << changed_tokens.status();
  (*changed_tokens)[0] += 1;
  const auto token_write = Upload(*executor_, *input_, *changed_tokens);
  const auto token_rejected = (*probe)->Apply(*executor_, selection);
  const auto token_restored = Upload(*executor_, *input_, tokens_);
  ASSERT_TRUE(token_write.ok()) << token_write;
  ASSERT_TRUE(token_restored.ok()) << token_restored;
  EXPECT_FALSE(token_rejected.ok());
  EXPECT_TRUE((*probe)->VerifyOriginals(*executor_).ok());
  auto restored = (*probe)->Apply(
      *executor_, {3, 6, 0, 511, QueryScope::kSelectedQuery, 1});
  ASSERT_TRUE(restored.ok()) << restored.status();
}

}  // namespace
}  // namespace pluto::weight_analysis::head_context
