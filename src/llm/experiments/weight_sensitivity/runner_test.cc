#include "src/llm/experiments/weight_sensitivity/runner.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>

#include "gtest/gtest.h"

namespace pluto::llm::weight_sensitivity {
namespace {

TEST(CorruptionRunnerTest, RestoresPackedTensorAfterSuccessAndFailure) {
  auto executor = cuda::Executor::Create();
  ASSERT_TRUE(executor.ok()) << executor.status();
  const std::array<float, 12> values{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  auto original =
      cuda::PageLockedHostArray<float>::CopyFrom(**executor, values);
  auto observed = cuda::PageLockedHostArray<float>::Allocate(**executor, 12);
  auto weight = cuda::Buffer::Allocate(**executor, sizeof(values));
  ASSERT_TRUE(original.ok()) << original.status();
  ASSERT_TRUE(observed.ok()) << observed.status();
  ASSERT_TRUE(weight.ok()) << weight.status();
  WeightTarget target{.name = "K matrix",
                      .checkpoint_index = 0,
                      .shape = {2, 2},
                      .tensor_elements = 12,
                      .offset = 2,
                      .rows = 2,
                      .columns = 2,
                      .row_stride = 6};
  auto download = [&] {
    EXPECT_EQ(cudaMemcpyAsync(observed->data(), weight->data(), sizeof(values),
                              cudaMemcpyDeviceToHost, (*executor)->stream()),
              cudaSuccess);
    EXPECT_TRUE((*executor)->Synchronize().ok());
  };
  for (bool fail : {false, true}) {
    bool called = false;
    auto result = EvaluateCorruption(
        **executor, *weight, target, *original, 123, 1, 0.02,
        [&]() -> absl::StatusOr<CompletionScores> {
          called = true;
          download();
          bool changed = false;
          for (size_t index = 0; index < values.size(); ++index)
            if (index % 6 >= 2 && index % 6 < 4)
              changed |= (*observed)[index] != values[index];
            else
              EXPECT_EQ((*observed)[index], values[index]);
          EXPECT_TRUE(changed);
          if (fail)
            return absl::InternalError("deliberate evaluation failure");
          return CompletionScores{.exact = {1}, .scored_tokens = 1};
        });
    EXPECT_TRUE(called);
    EXPECT_EQ(result.ok(), !fail);
    if (!fail)
      EXPECT_GT(result->noise_stddev, 0);
    else
      EXPECT_NE(result.status().message().find("deliberate"),
                absl::string_view::npos);
    download();
    EXPECT_TRUE(std::equal(values.begin(), values.end(), observed->begin()));
  }
}

}  // namespace
}  // namespace pluto::llm::weight_sensitivity
