#include "src/llm/experiments/weight_sensitivity/weights.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include "gtest/gtest.h"
#include "src/cuda/executor.h"

namespace pluto::llm::weight_sensitivity {
namespace {

Gpt2Config SmallConfig() {
  Gpt2Config config;
  config.transformer_block_count = 8;
  config.model_width = 16;
  config.attention_heads = 1;
  config.feed_forward_width = 64;
  config.vocabulary_size = 4475;
  config.pad_vocabulary = false;
  return config;
}

TEST(WeightsTest, LogicalTargetsPartitionAllStoredScalarsExactlyOnce) {
  const auto config = SmallConfig();
  auto targets = DescribeWeights(config);
  ASSERT_TRUE(targets.ok()) << targets.status();
  ASSERT_EQ(targets->size(), 132u);
  ASSERT_EQ(targets->front().shape, (std::vector<int64_t>{4475, 16}));
  ASSERT_EQ(targets->back().checkpoint_index, 99);
  std::vector<std::vector<int>> counts(100);
  for (const auto& target : *targets) {
    auto& tensor = counts[target.checkpoint_index];
    if (tensor.empty())
      tensor.resize(target.tensor_elements);
    ASSERT_EQ(tensor.size(), target.tensor_elements);
    size_t shape_elements = 1;
    for (int64_t dimension : target.shape)
      shape_elements *= dimension;
    ASSERT_EQ(shape_elements, target.rows * target.columns);
    for (size_t row = 0; row < target.rows; ++row)
      for (size_t column = 0; column < target.columns; ++column) {
        const size_t index = target.offset + row * target.row_stride + column;
        ASSERT_LT(index, tensor.size());
        ++tensor[index];
      }
  }
  for (const auto& tensor : counts) {
    ASSERT_FALSE(tensor.empty());
    EXPECT_TRUE(std::all_of(tensor.begin(), tensor.end(),
                            [](int count) { return count == 1; }));
  }
}

TEST(WeightsTest, QkvAreStridedColumnSlicesAndBiasIsContiguous) {
  auto targets = DescribeWeights(SmallConfig());
  ASSERT_TRUE(targets.ok());
  for (int i = 0; i < 3; ++i) {
    const auto& matrix = (*targets)[4 + i];
    EXPECT_EQ(matrix.checkpoint_index, 4);
    EXPECT_EQ(matrix.offset, i * 16);
    EXPECT_EQ(matrix.rows, 16);
    EXPECT_EQ(matrix.columns, 16);
    EXPECT_EQ(matrix.row_stride, 48);
    const auto& bias = (*targets)[7 + i];
    EXPECT_EQ(bias.checkpoint_index, 5);
    EXPECT_EQ(bias.offset, i * 16);
    EXPECT_EQ(bias.rows, 1);
    EXPECT_EQ(bias.columns, 16);
  }
}

TEST(WeightsTest, ZeroBlocksAndOptionalEmbeddingPadding) {
  auto config = SmallConfig();
  config.transformer_block_count = 0;
  config.pad_vocabulary = true;
  auto targets = DescribeWeights(config);
  ASSERT_TRUE(targets.ok());
  ASSERT_EQ(targets->size(), 4u);
  EXPECT_EQ(targets->front().shape, (std::vector<int64_t>{4480, 16}));
  EXPECT_EQ(targets->back().checkpoint_index, 3);
}

TEST(WeightsTest, InvalidConfigurationsAreRejected) {
  auto config = SmallConfig();
  config.attention_heads = 3;
  EXPECT_FALSE(DescribeWeights(config).ok());
  config = SmallConfig();
  config.model_width = 0;
  EXPECT_FALSE(DescribeWeights(config).ok());
  config = SmallConfig();
  config.transformer_block_count = std::numeric_limits<int>::max();
  EXPECT_FALSE(DescribeWeights(config).ok());
}

TEST(WeightsTest, NoiseChangesOnlySelectedSliceAndUsesItsRms) {
  auto targets = DescribeWeights(SmallConfig());
  ASSERT_TRUE(targets.ok());
  const auto& target = (*targets)[5];  // Packed K matrix.
  std::vector<float> original(target.tensor_elements, 100.0f);
  for (size_t row = 0; row < target.rows; ++row)
    for (size_t column = 0; column < target.columns; ++column)
      original[target.offset + row * target.row_stride + column] = 2.0f;
  std::vector<float> first(original.size()), second(original.size());
  auto stddev = ReplaceWithNoise(target, original, absl::MakeSpan(first), 42);
  ASSERT_TRUE(stddev.ok()) << stddev.status();
  EXPECT_DOUBLE_EQ(*stddev, 2.0);
  ASSERT_TRUE(
      ReplaceWithNoise(target, original, absl::MakeSpan(second), 42).ok());
  EXPECT_EQ(first, second);
  ASSERT_TRUE(
      ReplaceWithNoise(target, original, absl::MakeSpan(second), 43).ok());
  EXPECT_NE(first, second);
  for (size_t row = 0; row < target.rows; ++row)
    for (size_t column = 0; column < target.row_stride; ++column) {
      const size_t index = row * target.row_stride + column;
      if (column < 16 || column >= 32) {
        EXPECT_EQ(first[index], original[index]);
        EXPECT_EQ(second[index], original[index]);
      } else {
        EXPECT_TRUE(std::isfinite(first[index]));
        EXPECT_NE(first[index], original[index]);
      }
    }
  auto scaled =
      ReplaceWithNoise(target, original, absl::MakeSpan(second), 42, 0.5);
  ASSERT_TRUE(scaled.ok());
  EXPECT_DOUBLE_EQ(*scaled, 1.0);
  for (size_t row = 0; row < target.rows; ++row)
    for (size_t column = 0; column < target.columns; ++column) {
      const size_t index = target.offset + row * target.row_stride + column;
      EXPECT_FLOAT_EQ(second[index], first[index] * 0.5f);
    }
}

TEST(WeightsTest, ZeroBiasUsesExplicitFallbackAndSupportsInPlace) {
  auto targets = DescribeWeights(SmallConfig());
  ASSERT_TRUE(targets.ok());
  const auto& target = (*targets)[7];
  std::vector<float> zero(target.tensor_elements, 0.0f);
  auto stddev =
      ReplaceWithNoise(target, zero, absl::MakeSpan(zero), 7, 2.0, 0.125);
  ASSERT_TRUE(stddev.ok());
  EXPECT_DOUBLE_EQ(*stddev, 0.25);
  EXPECT_NE(zero.front(), 0.0f);
  for (size_t i = target.columns; i < zero.size(); ++i)
    EXPECT_EQ(zero[i], 0.0f);
}

TEST(WeightsTest, RejectsInvalidNoiseAndTensorSizes) {
  auto targets = DescribeWeights(SmallConfig());
  ASSERT_TRUE(targets.ok());
  auto target = (*targets)[7];
  std::vector<float> original(target.tensor_elements, 1.0f);
  std::vector<float> output(target.tensor_elements);
  EXPECT_FALSE(
      ReplaceWithNoise(target, original, absl::MakeSpan(output), 1, 0).ok());
  EXPECT_FALSE(ReplaceWithNoise(target, original, absl::MakeSpan(output), 1, 1,
                                std::numeric_limits<double>::infinity())
                   .ok());
  EXPECT_FALSE(ReplaceWithNoise(target, original, absl::MakeSpan(output), 1,
                                std::numeric_limits<double>::quiet_NaN())
                   .ok());
  std::fill(original.begin(), original.end(), 0.0f);
  EXPECT_FALSE(ReplaceWithNoise(target, original, absl::MakeSpan(output), 1,
                                std::numeric_limits<double>::denorm_min())
                   .ok());
  std::fill(original.begin(), original.end(), 1.0f);
  original[0] = std::numeric_limits<float>::infinity();
  EXPECT_FALSE(
      ReplaceWithNoise(target, original, absl::MakeSpan(output), 1).ok());
  original[0] = 1;
  original.pop_back();
  EXPECT_FALSE(
      ReplaceWithNoise(target, original, absl::MakeSpan(output), 1).ok());
  original.push_back(1);
  output.push_back(0);
  EXPECT_FALSE(
      ReplaceWithNoise(target, original, absl::MakeSpan(output), 1).ok());
  output.pop_back();
  target.row_stride = 0;
  EXPECT_FALSE(
      ReplaceWithNoise(target, original, absl::MakeSpan(output), 1).ok());
  target = (*targets)[7];
  target.rows = std::numeric_limits<size_t>::max();
  EXPECT_FALSE(
      ReplaceWithNoise(target, original, absl::MakeSpan(output), 1).ok());
}

TEST(WeightsTest, ShapeDescriptionMatchesActualModelAndRejectsBadBuffers) {
  auto config = SmallConfig();
  config.transformer_block_count = 1;
  config.model_width = 3;
  config.feed_forward_width = 5;
  config.vocabulary_size = 7;
  auto executor = cuda::Executor::Create();
  ASSERT_TRUE(executor.ok()) << executor.status();
  {
    auto model = CreateGpt2(**executor, DataType::BF16, 42, config);
    ASSERT_TRUE(model.ok()) << model.status();
    std::vector<Buffer> unique;
    for (const auto& weight : (*model)->weights())
      if (std::none_of(unique.begin(), unique.end(), [&](const Buffer& other) {
            return other.data() == weight.data();
          }))
        unique.push_back(weight);
    ASSERT_EQ(unique.size(), 16u);
    EXPECT_TRUE(ValidateWeights(config, unique).ok());
    auto missing = unique;
    missing.pop_back();
    EXPECT_FALSE(ValidateWeights(config, missing).ok());
    auto alias = unique;
    alias.back() = alias[2];  // Both LayerNorm vectors have identical sizes.
    EXPECT_FALSE(ValidateWeights(config, alias).ok());
    auto wrong_shape = unique;
    std::swap(wrong_shape[0], wrong_shape[1]);
    EXPECT_FALSE(ValidateWeights(config, wrong_shape).ok());
  }
  EXPECT_TRUE((*executor)->Synchronize().ok());
}

}  // namespace
}  // namespace pluto::llm::weight_sensitivity
