#pragma once

#include <cstddef>
#include <memory>

#include "absl/types/span.h"
#include "gtest/gtest.h"
#include "src/cuda/executor.h"
#include "src/cuda/page_locked_host_array.h"

namespace pluto::llm {

// Deliberately differ from the Shakespeare binary's defaults. Exercising a
// second shape catches accidental compile-time coupling in the cuTile backend.
inline constexpr int kTestBatchSize = 32;
inline constexpr int kTestModelWidth = 32;
inline constexpr int kTestVocabularySize = 32;
inline constexpr int kTestContextLength = 4;
inline constexpr int kTestAttentionHeads = 2;

template <class Container>
auto CopyToPageLockedHostArray(const Container& values)
    -> cuda::PageLockedHostArray<typename Container::value_type> {
  using Element = typename Container::value_type;
  auto result = cuda::PageLockedHostArray<Element>::CopyFrom(
      absl::MakeConstSpan(values.data(), values.size()));
  EXPECT_TRUE(result.ok()) << result.status();
  return result.ok() ? *result : cuda::PageLockedHostArray<Element>();
}

template <class Element>
cuda::PageLockedHostArray<Element> AllocatePageLockedHostArray(size_t size) {
  auto result = cuda::PageLockedHostArray<Element>::Allocate(size);
  EXPECT_TRUE(result.ok()) << result.status();
  return result.ok() ? *result : cuda::PageLockedHostArray<Element>();
}

class LayersTest : public testing::Test {
 protected:
  void SetUp() override {
    auto executor = cuda::Executor::Create();
    ASSERT_TRUE(executor.ok()) << executor.status();
    executor_ = std::move(*executor);
  }

  void TearDown() override {
    if (executor_ == nullptr) return;
    EXPECT_TRUE(executor_->Synchronize().ok());
    executor_.reset();
  }

  std::unique_ptr<cuda::Executor> executor_;
};

}  // namespace pluto::llm
