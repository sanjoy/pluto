#ifndef PLUTO_SRC_LLM_LAYERS_TEST_UTIL_H_
#define PLUTO_SRC_LLM_LAYERS_TEST_UTIL_H_

#include <cuda_runtime.h>

#include "gtest/gtest.h"

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
    ASSERT_EQ(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
              cudaSuccess);
  }

  void TearDown() override {
    if (stream_ == nullptr) return;
    EXPECT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
    EXPECT_EQ(cudaStreamDestroy(stream_), cudaSuccess);
  }

  cudaStream_t stream_ = nullptr;
};

}  // namespace pluto::llm

#endif  // PLUTO_SRC_LLM_LAYERS_TEST_UTIL_H_
