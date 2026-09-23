#include "src/llm/experiments/one_shot_memorizer/sentence_ablation.h"

#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "gtest/gtest.h"

namespace pluto::llm::one_shot_memorizer {
namespace {

constexpr Gpt2ParameterDimensions kDimensions{.vocabulary_size = 7,
                                              .model_width = 3,
                                              .feed_forward_width = 5,
                                              .context_length = 11,
                                              .transformer_block_count = 2};

std::vector<std::vector<float>> ZeroValues(
    absl::Span<const TensorSpec> layout) {
  std::vector<std::vector<float>> values;
  for (const auto& tensor : layout)
    values.emplace_back(tensor.element_count, 0);
  return values;
}

std::vector<absl::Span<const float>> Views(
    const std::vector<std::vector<float>>& values) {
  std::vector<absl::Span<const float>> views;
  for (const auto& tensor : values)
    views.emplace_back(tensor);
  return views;
}

TEST(SentenceAblationTest,
     LayoutUsesLogicalAsymmetricShapesAndUniqueWeightOrder) {
  auto layout = BuildGpt2ParameterLayout(kDimensions);
  ASSERT_TRUE(layout.ok()) << layout.status();
  ASSERT_EQ(layout->size(), 28u);
  EXPECT_EQ((*layout)[0].name, "token_embedding.weight");
  EXPECT_EQ((*layout)[0].shape, (std::vector<size_t>{7, 3}));
  EXPECT_TRUE((*layout)[0].token_embedding);
  EXPECT_EQ((*layout)[1].shape, (std::vector<size_t>{11, 3}));
  EXPECT_FALSE((*layout)[1].token_embedding);
  for (size_t block = 0; block < 2; ++block) {
    const size_t first = 2 + 12 * block;
    EXPECT_EQ((*layout)[first].shape, (std::vector<size_t>{3}));
    EXPECT_EQ((*layout)[first + 1].shape, (std::vector<size_t>{3}));
    EXPECT_EQ((*layout)[first + 2].shape, (std::vector<size_t>{3, 9}));
    EXPECT_EQ((*layout)[first + 2].qkv_width, 3u);
    EXPECT_EQ((*layout)[first + 3].shape, (std::vector<size_t>{9}));
    EXPECT_EQ((*layout)[first + 3].qkv_width, 3u);
    EXPECT_EQ((*layout)[first + 4].shape, (std::vector<size_t>{3, 3}));
    EXPECT_EQ((*layout)[first + 5].shape, (std::vector<size_t>{3}));
    EXPECT_EQ((*layout)[first + 6].shape, (std::vector<size_t>{3}));
    EXPECT_EQ((*layout)[first + 7].shape, (std::vector<size_t>{3}));
    EXPECT_EQ((*layout)[first + 8].shape, (std::vector<size_t>{3, 5}));
    EXPECT_EQ((*layout)[first + 9].shape, (std::vector<size_t>{5}));
    EXPECT_EQ((*layout)[first + 10].shape, (std::vector<size_t>{5, 3}));
    EXPECT_EQ((*layout)[first + 11].shape, (std::vector<size_t>{3}));
  }
  EXPECT_EQ((*layout)[14].name,
            "transformer_block_1.attention.layer_norm.gamma");
  EXPECT_EQ((*layout)[26].name, "final_layer_norm.gamma");
  EXPECT_EQ((*layout)[27].name, "final_layer_norm.beta");
  size_t offset = 0;
  for (size_t tensor = 0; tensor < layout->size(); ++tensor) {
    EXPECT_EQ((*layout)[tensor].checkpoint_index, tensor);
    EXPECT_EQ((*layout)[tensor].flat_offset, offset);
    size_t elements = 1;
    for (size_t dimension : (*layout)[tensor].shape)
      elements *= dimension;
    EXPECT_EQ((*layout)[tensor].element_count, elements);
    offset += elements;
  }
  EXPECT_EQ(offset, 256u);
  const auto values = ZeroValues(*layout);
  EXPECT_TRUE(ValidateParameterValues(*layout, Views(values)).ok());
}

TEST(SentenceAblationTest, MemorizedModelHas100UniqueTensorsAnd114256Scalars) {
  auto layout = BuildGpt2ParameterLayout({.vocabulary_size = 4475,
                                          .model_width = 16,
                                          .feed_forward_width = 64,
                                          .context_length = 1024,
                                          .transformer_block_count = 8});
  ASSERT_TRUE(layout.ok()) << layout.status();
  ASSERT_EQ(layout->size(), 100u);
  EXPECT_EQ(layout->back().flat_offset + layout->back().element_count, 114256u);
  EXPECT_EQ((*layout)[98].name, "final_layer_norm.gamma");
  EXPECT_EQ((*layout)[99].name, "final_layer_norm.beta");
  auto no_blocks = kDimensions;
  no_blocks.transformer_block_count = 0;
  auto small = BuildGpt2ParameterLayout(no_blocks);
  ASSERT_TRUE(small.ok());
  ASSERT_EQ(small->size(), 4u);
  EXPECT_EQ((*small)[2].name, "final_layer_norm.gamma");
}

TEST(SentenceAblationTest, CoordinatesDistinguishTokenRowsAndAllQkvPartitions) {
  auto layout = BuildGpt2ParameterLayout(kDimensions);
  ASSERT_TRUE(layout.ok());
  auto embedding = LocateParameter((*layout)[0], 20);
  ASSERT_TRUE(embedding.ok());
  EXPECT_EQ(embedding->checkpoint_index, 0u);
  EXPECT_EQ(embedding->element_index, 20u);
  EXPECT_EQ(embedding->row, 6u);
  EXPECT_EQ(embedding->column, 2u);
  EXPECT_EQ(embedding->compact_token_id, 6);
  EXPECT_EQ(embedding->flat_index, 20u);
  EXPECT_EQ(embedding->qkv_component, QkvComponent::kNone);
  EXPECT_FALSE(embedding->qkv_channel.has_value());
  auto position = LocateParameter((*layout)[1], 31);
  ASSERT_TRUE(position.ok());
  EXPECT_EQ(position->row, 10u);
  EXPECT_EQ(position->column, 1u);
  EXPECT_FALSE(position->compact_token_id.has_value());
  constexpr QkvComponent parts[]{QkvComponent::kQuery, QkvComponent::kKey,
                                 QkvComponent::kValue};
  constexpr const char* names[]{"query", "key", "value"};
  for (size_t component = 0; component < 3; ++component) {
    auto weight = LocateParameter((*layout)[4], 2 * 9 + component * 3 + 1);
    auto bias = LocateParameter((*layout)[5], component * 3 + 1);
    ASSERT_TRUE(weight.ok());
    ASSERT_TRUE(bias.ok());
    EXPECT_EQ(weight->row, 2u);
    EXPECT_EQ(weight->column, component * 3 + 1);
    EXPECT_EQ(weight->qkv_component, parts[component]);
    EXPECT_EQ(weight->qkv_channel, 1u);
    EXPECT_EQ(weight->flat_index,
              (*layout)[4].flat_offset + weight->element_index);
    EXPECT_EQ(bias->row, component * 3 + 1);
    EXPECT_FALSE(bias->column.has_value());
    EXPECT_EQ(bias->qkv_component, parts[component]);
    EXPECT_EQ(bias->qkv_channel, 1u);
    EXPECT_STREQ(QkvComponentName(parts[component]), names[component]);
  }
  auto contraction = LocateParameter((*layout)[12], 13);
  ASSERT_TRUE(contraction.ok());
  EXPECT_EQ(contraction->row, 4u);
  EXPECT_EQ(contraction->column, 1u);
}

TEST(SentenceAblationTest,
     ReportsEveryDeltaWithSignedDirectionAndCorrectNorms) {
  auto layout = BuildGpt2ParameterLayout(kDimensions);
  ASSERT_TRUE(layout.ok());
  auto baseline = ZeroValues(*layout), ablated = ZeroValues(*layout);
  baseline[0][0] = 3;
  baseline[0][1] = 4;
  ablated[0][1] = -4;
  baseline[4][22] = -5;
  ablated[4][22] = 2;
  auto report =
      CompareParameterValues(*layout, Views(baseline), Views(ablated), 1);
  ASSERT_TRUE(report.ok()) << report.status();
  EXPECT_EQ(report->total.element_count, 256u);
  EXPECT_EQ(report->total.bitwise_changed_count, 3u);
  EXPECT_EQ(report->total.numerically_changed_count, 3u);
  EXPECT_DOUBLE_EQ(report->total.baseline_l2, std::sqrt(50.0));
  EXPECT_DOUBLE_EQ(report->total.ablated_l2, std::sqrt(20.0));
  EXPECT_DOUBLE_EQ(report->total.delta_l1, 18);
  EXPECT_DOUBLE_EQ(report->total.delta_l2, std::sqrt(122.0));
  EXPECT_DOUBLE_EQ(report->total.maximum_absolute_delta, 8);
  ASSERT_TRUE(report->total.relative_l2.has_value());
  EXPECT_DOUBLE_EQ(*report->total.relative_l2,
                   std::sqrt(122.0) / std::sqrt(50.0));
  const auto& embedding = report->tensors[0];
  EXPECT_EQ(embedding.deltas.size(), baseline[0].size());
  EXPECT_EQ(embedding.deltas[0], 3);
  EXPECT_EQ(embedding.deltas[1], 8);
  EXPECT_EQ(embedding.summary.bitwise_changed_count, 2u);
  EXPECT_DOUBLE_EQ(embedding.summary.delta_l2, std::sqrt(73.0));
  ASSERT_EQ(embedding.top_coordinates.size(), 1u);
  const auto& top = embedding.top_coordinates[0];
  EXPECT_EQ(top.coordinate.element_index, 1u);
  EXPECT_EQ(top.coordinate.compact_token_id, 0);
  EXPECT_EQ(top.coordinate.column, 1u);
  EXPECT_EQ(top.baseline, 4);
  EXPECT_EQ(top.ablated, -4);
  EXPECT_EQ(top.delta, 8);
  const auto& qkv = report->tensors[4];
  ASSERT_EQ(qkv.top_coordinates.size(), 1u);
  EXPECT_EQ(qkv.deltas[22], -7);
  EXPECT_EQ(qkv.top_coordinates[0].coordinate.qkv_component,
            QkvComponent::kKey);
  EXPECT_EQ(qkv.top_coordinates[0].coordinate.qkv_channel, 1u);
  // The result owns its copied deltas even after caller data changes.
  baseline[0][0] = 100;
  EXPECT_EQ(embedding.deltas[0], 3);
}

TEST(SentenceAblationTest, SignedZeroChangesBitsButNotNumericDeltaOrNorms) {
  auto layout = BuildGpt2ParameterLayout(kDimensions);
  ASSERT_TRUE(layout.ok());
  auto baseline = ZeroValues(*layout), ablated = ZeroValues(*layout);
  ablated[0][0] = -0.0f;
  baseline[0][1] = -0.0f;
  auto report =
      CompareParameterValues(*layout, Views(baseline), Views(ablated));
  ASSERT_TRUE(report.ok());
  EXPECT_EQ(report->total.bitwise_changed_count, 2u);
  EXPECT_EQ(report->total.numerically_changed_count, 0u);
  EXPECT_EQ(report->total.delta_l1, 0);
  EXPECT_EQ(report->total.delta_l2, 0);
  EXPECT_EQ(report->total.maximum_absolute_delta, 0);
  EXPECT_FALSE(report->total.relative_l2.has_value());
  const auto& tensor = report->tensors[0];
  EXPECT_TRUE(tensor.bitwise_changed[0]);
  EXPECT_TRUE(tensor.bitwise_changed[1]);
  EXPECT_FALSE(tensor.bitwise_changed[2]);
  ASSERT_EQ(tensor.top_coordinates.size(), 2u);
  EXPECT_EQ(tensor.top_coordinates[0].coordinate.element_index, 0u);
  EXPECT_EQ(tensor.top_coordinates[1].coordinate.element_index, 1u);
  EXPECT_EQ(tensor.top_coordinates[0].delta, 0);
  EXPECT_EQ(std::bit_cast<uint32_t>(tensor.top_coordinates[0].ablated),
            0x80000000u);
}

TEST(SentenceAblationTest,
     TopCoordinatesAreDeterministicAndZeroLimitKeepsFullData) {
  auto layout = BuildGpt2ParameterLayout(kDimensions);
  ASSERT_TRUE(layout.ok());
  auto baseline = ZeroValues(*layout), ablated = ZeroValues(*layout);
  baseline[0][0] = 5;
  baseline[0][1] = -5;
  baseline[0][2] = 4;
  baseline[0][3] = 4;
  auto first =
      CompareParameterValues(*layout, Views(baseline), Views(ablated), 3);
  auto second =
      CompareParameterValues(*layout, Views(baseline), Views(ablated), 3);
  auto no_top =
      CompareParameterValues(*layout, Views(baseline), Views(ablated), 0);
  auto identical =
      CompareParameterValues(*layout, Views(baseline), Views(baseline));
  ASSERT_TRUE(first.ok());
  ASSERT_TRUE(second.ok());
  ASSERT_TRUE(no_top.ok());
  ASSERT_TRUE(identical.ok());
  ASSERT_EQ(first->tensors[0].top_coordinates.size(), 3u);
  for (size_t index = 0; index < 3; ++index) {
    EXPECT_EQ(first->tensors[0].top_coordinates[index].coordinate.element_index,
              index);
    EXPECT_EQ(
        second->tensors[0].top_coordinates[index].coordinate.element_index,
        index);
    EXPECT_EQ(first->tensors[0].top_coordinates[index].delta,
              second->tensors[0].top_coordinates[index].delta);
  }
  EXPECT_TRUE(no_top->tensors[0].top_coordinates.empty());
  EXPECT_EQ(no_top->tensors[0].deltas, first->tensors[0].deltas);
  EXPECT_EQ(no_top->tensors[0].bitwise_changed,
            first->tensors[0].bitwise_changed);
  EXPECT_EQ(identical->total.bitwise_changed_count, 0u);
  ASSERT_TRUE(identical->total.relative_l2.has_value());
  EXPECT_EQ(*identical->total.relative_l2, 0);
  EXPECT_TRUE(identical->tensors[0].top_coordinates.empty());
}

TEST(SentenceAblationTest,
     RejectsInvalidDimensionsAndImpossibleGpuTensorSizes) {
  auto dimensions = kDimensions;
  dimensions.vocabulary_size = 0;
  EXPECT_FALSE(BuildGpt2ParameterLayout(dimensions).ok());
  dimensions = kDimensions;
  dimensions.model_width = -1;
  EXPECT_FALSE(BuildGpt2ParameterLayout(dimensions).ok());
  dimensions = kDimensions;
  dimensions.feed_forward_width = 0;
  EXPECT_FALSE(BuildGpt2ParameterLayout(dimensions).ok());
  dimensions = kDimensions;
  dimensions.context_length = 0;
  EXPECT_FALSE(BuildGpt2ParameterLayout(dimensions).ok());
  dimensions = kDimensions;
  dimensions.transformer_block_count = -1;
  EXPECT_FALSE(BuildGpt2ParameterLayout(dimensions).ok());
  dimensions.transformer_block_count = std::numeric_limits<int>::max();
  EXPECT_FALSE(BuildGpt2ParameterLayout(dimensions).ok());
  dimensions = kDimensions;
  dimensions.model_width = std::numeric_limits<int>::max();
  EXPECT_FALSE(BuildGpt2ParameterLayout(dimensions).ok());
  dimensions = kDimensions;
  dimensions.vocabulary_size = std::numeric_limits<int>::max();
  EXPECT_FALSE(BuildGpt2ParameterLayout(dimensions).ok());
}

TEST(SentenceAblationTest,
     RejectsMalformedLayoutStorageCoordinatesAndNonfiniteValues) {
  auto layout = BuildGpt2ParameterLayout(kDimensions);
  ASSERT_TRUE(layout.ok());
  auto values = ZeroValues(*layout);
  EXPECT_FALSE(ValidateParameterValues({}, Views(values)).ok());
  EXPECT_FALSE(ValidateParameterValues(*layout, {}).ok());
  EXPECT_FALSE(LocateParameter((*layout)[0], (*layout)[0].element_count).ok());
  auto invalid = *layout;
  invalid[0].checkpoint_index = 1;
  EXPECT_FALSE(ValidateParameterValues(invalid, Views(values)).ok());
  invalid = *layout;
  invalid[1].flat_offset = 0;
  EXPECT_FALSE(ValidateParameterValues(invalid, Views(values)).ok());
  invalid = *layout;
  invalid[1].name = invalid[0].name;
  EXPECT_FALSE(ValidateParameterValues(invalid, Views(values)).ok());
  invalid = *layout;
  invalid[0].shape = {0, 3};
  EXPECT_FALSE(ValidateParameterValues(invalid, Views(values)).ok());
  invalid = *layout;
  invalid[0].shape = {std::numeric_limits<size_t>::max(), 2};
  EXPECT_FALSE(ValidateParameterValues(invalid, Views(values)).ok());
  invalid = *layout;
  invalid[0].flat_offset = std::numeric_limits<size_t>::max();
  EXPECT_FALSE(ValidateParameterValues(invalid, Views(values)).ok());
  invalid = *layout;
  invalid[0].element_count += 1;
  EXPECT_FALSE(ValidateParameterValues(invalid, Views(values)).ok());
  invalid = *layout;
  invalid[4].qkv_width = 2;
  EXPECT_FALSE(ValidateParameterValues(invalid, Views(values)).ok());
  invalid = *layout;
  invalid[2].token_embedding = true;
  EXPECT_FALSE(ValidateParameterValues(invalid, Views(values)).ok());
  auto short_values = values;
  short_values[0].pop_back();
  EXPECT_FALSE(ValidateParameterValues(*layout, Views(short_values)).ok());
  EXPECT_FALSE(
      CompareParameterValues(*layout, Views(values), Views(short_values)).ok());
  for (float nonfinite : {std::numeric_limits<float>::infinity(),
                          std::numeric_limits<float>::quiet_NaN()}) {
    auto bad = values;
    bad[3][0] = nonfinite;
    EXPECT_FALSE(ValidateParameterValues(*layout, Views(bad)).ok());
    EXPECT_FALSE(
        CompareParameterValues(*layout, Views(bad), Views(values)).ok());
    EXPECT_FALSE(
        CompareParameterValues(*layout, Views(values), Views(bad)).ok());
  }
}

TEST(SentenceAblationTest, FiniteFloatExtremesUseDoubleDifferencesAndNorms) {
  auto layout = BuildGpt2ParameterLayout(kDimensions);
  ASSERT_TRUE(layout.ok());
  auto baseline = ZeroValues(*layout), ablated = ZeroValues(*layout);
  baseline[0][0] = std::numeric_limits<float>::max();
  ablated[0][0] = -std::numeric_limits<float>::max();
  auto report =
      CompareParameterValues(*layout, Views(baseline), Views(ablated));
  ASSERT_TRUE(report.ok()) << report.status();
  EXPECT_TRUE(std::isfinite(report->total.delta_l2));
  EXPECT_EQ(report->tensors[0].deltas[0],
            2 * static_cast<double>(std::numeric_limits<float>::max()));
  ASSERT_TRUE(report->total.relative_l2.has_value());
  EXPECT_DOUBLE_EQ(*report->total.relative_l2, 2);
}

}  // namespace
}  // namespace pluto::llm::one_shot_memorizer
