#include "src/llm/layers/combinators.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "src/gpu/buffer.h"
#include "src/llm/layer.h"
#include "src/llm/layers/test_util.h"
#include "src/llm/layers/fully_connected.h"

namespace pluto::llm {
namespace {

TEST_F(LayersTest, RepeatedLayerCollectsIndependentChildWeights) {
  auto first = FullyConnectedLayer::Create(
      kTestModelWidth, DataType::FP16, 0.0f, stream_);
  auto second = FullyConnectedLayer::Create(
      kTestModelWidth, DataType::FP16, 0.0f, stream_);
  ASSERT_TRUE(first.ok()) << first.status();
  ASSERT_TRUE(second.ok()) << second.status();
  std::vector<std::unique_ptr<FullyConnectedLayer>> repetitions;
  repetitions.push_back(std::move(*first));
  repetitions.push_back(std::move(*second));

  RepeatedLayer<FullyConnectedLayer> repeated(DataType::FP16,
                                               std::move(repetitions));
  // Each dense repetition contributes its own matrix and bias.
  EXPECT_EQ(repeated.weights().size(), 4u);
}


}  // namespace
}  // namespace pluto::llm
