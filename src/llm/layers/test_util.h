#pragma once

#include <memory>

#include "gtest/gtest.h"
#include "src/cuda/executor.h"

namespace pluto::llm {

// Deliberately differ from the Shakespeare binary's defaults. Exercising a
// second shape catches accidental compile-time coupling in the cuTile backend.
inline constexpr int kTestBatchSize = 32;
inline constexpr int kTestModelWidth = 32;
inline constexpr int kTestVocabularySize = 32;
inline constexpr int kTestContextLength = 4;
inline constexpr int kTestAttentionHeads = 2;

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
