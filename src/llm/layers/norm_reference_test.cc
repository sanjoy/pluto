#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <tuple>
#include <vector>

#include "absl/status/status.h"
#include "gtest/gtest.h"
#include "src/llm/layer.h"
#include "src/llm/layers/norm.h"
#include "src/llm/layers/reference_test_util.h"

namespace pluto::llm {
namespace {

testing::AssertionResult LayerNormActivationsNear(
    absl::Span<const float> actual, absl::Span<const float> expected,
    DataType type) {
  if (type != DataType::BF16)
    return VectorsNear(actual, expected, 3e-4f, 3e-4f);
  if (actual.size() != expected.size())
    return testing::AssertionFailure() << "activation size mismatch";
  for (size_t index = 0; index < actual.size(); ++index) {
    if (actual[index] == expected[index]) continue;
    if (!std::isfinite(actual[index]) || !std::isfinite(expected[index]))
      return testing::AssertionFailure() << "nonfinite activation " << index;
    const float magnitude =
        std::max(std::abs(actual[index]), std::abs(expected[index]));
    // Changing the FP32 reduction order can move a value across a BF16
    // rounding midpoint. Permit one storage ULP, with a small FP32 absolute
    // roundoff floor for affine outputs that cancel almost to zero.
    const float storage_ulp = std::ldexp(1.0f, std::ilogb(magnitude) - 7);
    const float tolerance = std::max(3e-5f, storage_ulp);
    if (std::abs(actual[index] - expected[index]) > tolerance)
      return testing::AssertionFailure()
             << "element " << index << ": " << actual[index] << " vs "
             << expected[index] << " (one BF16 ULP/FP32 floor " << tolerance
             << ")";
  }
  return testing::AssertionSuccess();
}

testing::AssertionResult SavedStatisticsMatchDoubleOracle(
    cuda::Executor& executor, const BackwardState& state,
    const HostBuffer& input_pattern, int width, DataType type) {
  if (state.intermediates.size() != 3)
    return testing::AssertionFailure() << "expected three saved buffers";
  auto means = ReadDeviceFloats(executor, state.intermediates[1]);
  auto inverse_stddevs = ReadDeviceFloats(executor, state.intermediates[2]);
  if (!means.ok()) return testing::AssertionFailure() << means.status();
  if (!inverse_stddevs.ok())
    return testing::AssertionFailure() << inverse_stddevs.status();
  auto input = ReadHostActivations(input_pattern, type);
  const size_t pattern_rows = input.size() / width;
  if (pattern_rows == 0 || means->size() % pattern_rows != 0 ||
      means->size() != inverse_stddevs->size())
    return testing::AssertionFailure() << "invalid statistics dimensions";
  std::vector<double> expected_means(pattern_rows);
  std::vector<double> expected_inverse_stddevs(pattern_rows);
  for (size_t row = 0; row < pattern_rows; ++row) {
    double mean = 0.0;
    for (int column = 0; column < width; ++column)
      mean += static_cast<double>(input[row * width + column]);
    mean /= width;
    double variance = 0.0;
    for (int column = 0; column < width; ++column) {
      const double centered =
          static_cast<double>(input[row * width + column]) - mean;
      variance += centered * centered;
    }
    expected_means[row] = mean;
    expected_inverse_stddevs[row] =
        1.0 / std::sqrt(variance / width + static_cast<double>(1e-5f));
  }
  // This is independent of BF16 output rounding and the scalar FP32 reference:
  // both saved reductions must agree with the stored-input double oracle.
  for (size_t row = 0; row < means->size(); ++row) {
    const double expected_mean = expected_means[row % pattern_rows];
    const double expected_inverse_stddev =
        expected_inverse_stddevs[row % pattern_rows];
    if (!std::isfinite((*means)[row]) ||
        std::abs((*means)[row] - expected_mean) >
            2e-6 + 5e-7 * std::abs(expected_mean))
      return testing::AssertionFailure()
             << "mean row " << row << ": " << (*means)[row]
             << " vs double oracle " << expected_mean;
    if (!std::isfinite((*inverse_stddevs)[row]) ||
        std::abs((*inverse_stddevs)[row] - expected_inverse_stddev) >
            2e-6 + 5e-7 * std::abs(expected_inverse_stddev))
      return testing::AssertionFailure()
             << "inverse stddev row " << row << ": " << (*inverse_stddevs)[row]
             << " vs double oracle " << expected_inverse_stddev;
  }
  return testing::AssertionSuccess();
}

TEST_F(LayerReferenceTest, ForwardAndBackwardMatchAcrossShapesAndTypes) {
  for (DataType type : {DataType::FP16, DataType::BF16}) {
    for (const auto& [rows, width] :
         {std::tuple{3, 1}, std::tuple{5, 3}, std::tuple{7, 7},
          std::tuple{3, 8}, std::tuple{17, 15}, std::tuple{3, 24},
          std::tuple{17, 33}, std::tuple{1, 16}, std::tuple{3, 16},
          std::tuple{16, 32}, std::tuple{17, 48}, std::tuple{31, 80},
          std::tuple{127, 512}, std::tuple{129, 144}, std::tuple{257, 528}}) {
      SCOPED_TRACE(testing::Message()
                   << "type=" << static_cast<int>(type) << " rows=" << rows
                   << " width=" << width);
      auto device_layer =
          LayerNormLayer::Create(*executor_, width, 1e-5f, type);
      auto reference_layer =
          LayerNormLayerReference::Create(width, 1e-5f, type);
      ASSERT_TRUE(device_layer.ok()) << device_layer.status();
      ASSERT_TRUE(reference_layer.ok()) << reference_layer.status();
      auto device_weights = (*device_layer)->weights();
      auto reference_weights = (*reference_layer)->weights();
      std::vector<float> gamma(width);
      std::vector<float> beta(width);
      for (int column = 0; column < width; ++column) {
        gamma[column] = 0.7f + 0.2f * std::cos(column * 0.19f);
        beta[column] = 0.1f * std::sin(column * 0.31f);
      }
      ASSERT_TRUE(SetFloatBufferPair(*executor_, device_weights[0],
                                     &reference_weights[0], gamma)
                      .ok());
      ASSERT_TRUE(SetFloatBufferPair(*executor_, device_weights[1],
                                     &reference_weights[1], beta)
                      .ok());

      std::vector<float> input(static_cast<size_t>(rows) * width);
      std::vector<float> gradient(input.size());
      for (int row = 0; row < rows; ++row) {
        for (int column = 0; column < width; ++column) {
          input[static_cast<size_t>(row) * width + column] =
              0.4f * std::sin(row * 0.71f + column * 0.17f) + row * 0.03f;
          gradient[static_cast<size_t>(row) * width + column] =
              0.2f * std::cos(row * 0.37f - column * 0.11f);
        }
      }
      auto input_pair = MakeActivationBufferPair(*executor_, input, type);
      auto gradient_pair = MakeRawBufferPair<float>(*executor_, gradient);
      ASSERT_TRUE(input_pair.ok()) << input_pair.status();
      ASSERT_TRUE(gradient_pair.ok()) << gradient_pair.status();

      BufferVec device_inputs = {input_pair->device};
      HostBufferVec reference_inputs = {input_pair->host};
      auto device_output = (*device_layer)->fwd(*executor_, device_inputs);

      auto reference_output = (*reference_layer)->fwd(reference_inputs);

      ASSERT_TRUE(device_output.ok()) << device_output.status();
      ASSERT_TRUE(reference_output.ok()) << reference_output.status();
      EXPECT_TRUE(SavedStatisticsMatchDoubleOracle(
          *executor_, device_output->state, input_pair->host, width, type));
      auto actual_output =
          ReadDeviceActivations(*executor_, device_output->outputs[0], type);
      ASSERT_TRUE(actual_output.ok()) << actual_output.status();
      EXPECT_TRUE(LayerNormActivationsNear(
          actual_output->span(),
          ReadHostActivations(reference_output->outputs[0], type), type));

      BufferVec device_gradients = {gradient_pair->device};
      HostBufferVec reference_gradients = {gradient_pair->host};
      auto device_input = (*device_layer)
                              ->bwd(*executor_, device_gradients,
                                    std::move(device_output->state));
      auto reference_input =
          (*reference_layer)
              ->bwd(reference_gradients, std::move(reference_output->state));
      ASSERT_TRUE(device_input.ok()) << device_input.status();
      ASSERT_TRUE(reference_input.ok()) << reference_input.status();
      EXPECT_TRUE(FloatBuffersNear(device_input->front(),
                                   reference_input->front(), 5e-3f, 4e-3f));
      auto device_parameter_gradients = (*device_layer)->gradients();
      auto reference_parameter_gradients = (*reference_layer)->gradients();
      for (size_t index = 0; index < device_parameter_gradients.size();
           ++index) {
        EXPECT_TRUE(FloatBuffersNear(device_parameter_gradients[index],
                                     reference_parameter_gradients[index],
                                     5e-3f, 3e-3f));
      }
    }
  }
}

TEST_F(LayerReferenceTest, ExtremeFiniteConstantsExcludeMaskedChannels) {
  constexpr float kInput = 0x1p120f;
  constexpr float kEpsilon = 0x1p-20f;
  for (DataType type : {DataType::FP16, DataType::BF16}) {
    for (int rows : {1, 17}) {
      for (int width : {16, 80}) {
        SCOPED_TRACE(testing::Message()
                     << "type=" << static_cast<int>(type) << " rows=" << rows
                     << " width=" << width);
        // Both values are exactly representable in either activation type.
        // Valid lanes normalize to zero, but masked zero lanes would normalize
        // to -infinity. Their zero upstream gradient must not create 0 * inf
        // in the channel reduction for dx.
        auto device_layer =
            LayerNormLayer::Create(*executor_, width, kEpsilon, type);
        auto reference_layer =
            LayerNormLayerReference::Create(width, kEpsilon, type);
        ASSERT_TRUE(device_layer.ok()) << device_layer.status();
        ASSERT_TRUE(reference_layer.ok()) << reference_layer.status();
        // Default affine parameters are gamma=1 and beta=0.
        std::vector<float> input(static_cast<size_t>(rows) * width, kInput);
        std::vector<float> gradient(input.size(), 1.0f);
        auto input_pair = MakeActivationBufferPair(*executor_, input, type);
        auto gradient_pair = MakeRawBufferPair<float>(*executor_, gradient);
        ASSERT_TRUE(input_pair.ok()) << input_pair.status();
        ASSERT_TRUE(gradient_pair.ok()) << gradient_pair.status();
        auto device_output =
            (*device_layer)->fwd(*executor_, BufferVec{input_pair->device});
        auto reference_output =
            (*reference_layer)->fwd(HostBufferVec{input_pair->host});
        ASSERT_TRUE(device_output.ok()) << device_output.status();
        ASSERT_TRUE(reference_output.ok()) << reference_output.status();
        auto actual_output =
            ReadDeviceActivations(*executor_, device_output->outputs[0], type);
        ASSERT_TRUE(actual_output.ok()) << actual_output.status();
        EXPECT_TRUE(VectorsNear(
            actual_output->span(),
            ReadHostActivations(reference_output->outputs[0], type), 0.0f));
        EXPECT_TRUE(VectorsNear(actual_output->span(),
                                std::vector<float>(input.size(), 0.0f), 0.0f));

        auto device_input =
            (*device_layer)
                ->bwd(*executor_, BufferVec{gradient_pair->device},
                      std::move(device_output->state));
        auto reference_input = (*reference_layer)
                                   ->bwd(HostBufferVec{gradient_pair->host},
                                         std::move(reference_output->state));
        ASSERT_TRUE(device_input.ok()) << device_input.status();
        ASSERT_TRUE(reference_input.ok()) << reference_input.status();
        ASSERT_EQ(device_input->size(), 1u);
        ASSERT_EQ(reference_input->size(), 1u);
        const BufferVec device_results = {device_input->front(),
                                          (*device_layer)->gradients()[0],
                                          (*device_layer)->gradients()[1]};
        const HostBufferVec reference_results = {
            reference_input->front(), (*reference_layer)->gradients()[0],
            (*reference_layer)->gradients()[1]};
        for (size_t index = 0; index < device_results.size(); ++index) {
          SCOPED_TRACE(testing::Message() << "dx/dgamma/dbeta index=" << index);
          auto actual = ReadDeviceFloats(*executor_, device_results[index]);
          ASSERT_TRUE(actual.ok()) << actual.status();
          EXPECT_TRUE(VectorsNear(
              actual->span(), ReadHostFloats(reference_results[index]), 0.0f));
          // Exact finite values independently check the reference: dx and
          // dgamma are zero; each row contributes one to every beta gradient.
          const float expected = index == 2 ? static_cast<float>(rows) : 0.0f;
          EXPECT_TRUE(VectorsNear(actual->span(),
                                  std::vector<float>(actual->size(), expected),
                                  0.0f));
        }
      }
    }
  }
}

TEST_F(LayerReferenceTest,
       ProductionRowsMatchRepeatedOracleAndAreDeterministic) {
  constexpr int kRows = 20480;
  constexpr int kWidth = 512;
  constexpr int kPatternRows = 128;
  constexpr int kRepeats = kRows / kPatternRows;
  constexpr size_t kPatternSize = kPatternRows * kWidth;
  for (DataType type : {DataType::FP16, DataType::BF16}) {
    SCOPED_TRACE(testing::Message() << "type=" << static_cast<int>(type));
    auto device_layer = LayerNormLayer::Create(*executor_, kWidth, 1e-5f, type);
    auto reference_layer = LayerNormLayerReference::Create(kWidth, 1e-5f, type);
    ASSERT_TRUE(device_layer.ok()) << device_layer.status();
    ASSERT_TRUE(reference_layer.ok()) << reference_layer.status();
    std::vector<float> gamma(kWidth);
    std::vector<float> beta(kWidth);
    for (int column = 0; column < kWidth; ++column) {
      gamma[column] = 0.8f + 0.3f * std::sin(column * 0.13f);
      beta[column] = 0.1f * std::cos(column * 0.17f);
    }
    auto device_weights = (*device_layer)->weights();
    auto reference_weights = (*reference_layer)->weights();
    ASSERT_TRUE(SetFloatBufferPair(*executor_, device_weights[0],
                                   &reference_weights[0], gamma)
                    .ok());
    ASSERT_TRUE(SetFloatBufferPair(*executor_, device_weights[1],
                                   &reference_weights[1], beta)
                    .ok());

    std::vector<float> pattern(kPatternSize);
    std::vector<float> gradient_pattern(kPatternSize);
    for (int row = 0; row < kPatternRows; ++row)
      for (int column = 0; column < kWidth; ++column) {
        const size_t index = static_cast<size_t>(row) * kWidth + column;
        pattern[index] = 0.3f * std::sin(row * 0.27f + column * 0.11f) +
                         0.2f * std::cos(column * 0.07f) + row * 0.002f;
        gradient_pattern[index] =
            0.04f + 0.09f * std::cos(row * 0.19f - column * 0.03f);
      }
    std::vector<float> input(static_cast<size_t>(kRows) * kWidth);
    std::vector<float> gradient(input.size());
    for (int repeat = 0; repeat < kRepeats; ++repeat) {
      std::copy(pattern.begin(), pattern.end(),
                input.begin() + repeat * kPatternSize);
      std::copy(gradient_pattern.begin(), gradient_pattern.end(),
                gradient.begin() + repeat * kPatternSize);
    }
    auto input_pair = MakeActivationBufferPair(*executor_, input, type);
    auto gradient_pair = MakeRawBufferPair<float>(*executor_, gradient);
    auto pattern_pair = MakeActivationBufferPair(*executor_, pattern, type);
    auto gradient_pattern_pair =
        MakeRawBufferPair<float>(*executor_, gradient_pattern);
    ASSERT_TRUE(input_pair.ok()) << input_pair.status();
    ASSERT_TRUE(gradient_pair.ok()) << gradient_pair.status();
    ASSERT_TRUE(pattern_pair.ok()) << pattern_pair.status();
    ASSERT_TRUE(gradient_pattern_pair.ok()) << gradient_pattern_pair.status();
    auto output =
        (*device_layer)->fwd(*executor_, BufferVec{input_pair->device});
    auto reference_output =
        (*reference_layer)->fwd(HostBufferVec{pattern_pair->host});
    ASSERT_TRUE(output.ok()) << output.status();
    ASSERT_TRUE(reference_output.ok()) << reference_output.status();
    ASSERT_EQ(output->state.intermediates.size(), 3);
    ASSERT_TRUE(output->state.children.empty());
    for (size_t index : {1, 2})
      EXPECT_EQ(output->state.intermediates[index].size_bytes(),
                static_cast<size_t>(kRows) * sizeof(float));

    EXPECT_TRUE(SavedStatisticsMatchDoubleOracle(
        *executor_, output->state, pattern_pair->host, kWidth, type));
    auto actual_output =
        ReadDeviceActivations(*executor_, output->outputs[0], type);
    ASSERT_TRUE(actual_output.ok()) << actual_output.status();
    auto expected_output =
        ReadHostActivations(reference_output->outputs[0], type);
    for (int repeat = 0; repeat < kRepeats; ++repeat) {
      SCOPED_TRACE(repeat);
      ASSERT_TRUE(LayerNormActivationsNear(
          actual_output->span().subspan(repeat * kPatternSize, kPatternSize),
          expected_output, type));
    }

    // A subsequent forward has different statistics and a different row count.
    // It must not overwrite the first forward's retained per-row statistics.
    BackwardState retained = output->state;
    auto means_before = ReadDeviceFloats(*executor_, retained.intermediates[1]);
    auto inverse_stddevs_before =
        ReadDeviceFloats(*executor_, retained.intermediates[2]);
    ASSERT_TRUE(means_before.ok()) << means_before.status();
    ASSERT_TRUE(inverse_stddevs_before.ok()) << inverse_stddevs_before.status();
    for (float& value : pattern) value = value * 3.0f + 2.0f;
    auto later_pair = MakeActivationBufferPair(*executor_, pattern, type);
    ASSERT_TRUE(later_pair.ok()) << later_pair.status();
    auto later_output =
        (*device_layer)->fwd(*executor_, BufferVec{later_pair->device});
    ASSERT_TRUE(later_output.ok()) << later_output.status();
    EXPECT_NE(later_output->state.intermediates[1].data(),
              retained.intermediates[1].data());
    EXPECT_NE(later_output->state.intermediates[2].data(),
              retained.intermediates[2].data());

    auto device_parameter_gradients = (*device_layer)->gradients();
    auto reference_parameter_gradients = (*reference_layer)->gradients();
    // Backward historically replaces gamma/beta gradients, even if nonzero.
    // Seed them to exercise that contract as well as all partial-reduction
    // tiles.
    std::vector<float> sentinel(kWidth, 31.0f);
    for (size_t index = 0; index < device_parameter_gradients.size(); ++index)
      ASSERT_TRUE(
          SetFloatBufferPair(*executor_, device_parameter_gradients[index],
                             &reference_parameter_gradients[index], sentinel)
              .ok());
    auto actual_input =
        (*device_layer)
            ->bwd(*executor_, BufferVec{gradient_pair->device}, retained);
    auto expected_input = (*reference_layer)
                              ->bwd(HostBufferVec{gradient_pattern_pair->host},
                                    std::move(reference_output->state));
    ASSERT_TRUE(actual_input.ok()) << actual_input.status();
    ASSERT_TRUE(expected_input.ok()) << expected_input.status();
    auto actual_dx = ReadDeviceFloats(*executor_, actual_input->front());
    ASSERT_TRUE(actual_dx.ok()) << actual_dx.status();
    auto expected_dx = ReadHostFloats(expected_input->front());
    for (int repeat = 0; repeat < kRepeats; ++repeat) {
      SCOPED_TRACE(repeat);
      ASSERT_TRUE(VectorsNear(
          actual_dx->span().subspan(repeat * kPatternSize, kPatternSize),
          expected_dx, 5e-3f, 4e-3f));
    }
    std::vector<std::vector<float>> first_parameter_gradients;
    for (size_t index = 0; index < device_parameter_gradients.size(); ++index) {
      auto actual =
          ReadDeviceFloats(*executor_, device_parameter_gradients[index]);
      ASSERT_TRUE(actual.ok()) << actual.status();
      auto expected = ReadHostFloats(reference_parameter_gradients[index]);
      for (float& value : expected) value *= kRepeats;
      EXPECT_TRUE(VectorsNear(actual->span(), expected, 8e-3f, 3e-4f));
      first_parameter_gradients.emplace_back(actual->begin(), actual->end());
    }

    // Reuse the exact state without clearing gradients: all outputs must be
    // bit-for-bit identical, independently of kernel scheduling.
    auto repeated_input =
        (*device_layer)
            ->bwd(*executor_, BufferVec{gradient_pair->device}, retained);
    ASSERT_TRUE(repeated_input.ok()) << repeated_input.status();
    auto repeated_dx = ReadDeviceFloats(*executor_, repeated_input->front());
    ASSERT_TRUE(repeated_dx.ok()) << repeated_dx.status();
    EXPECT_EQ(std::memcmp(actual_dx->data(), repeated_dx->data(),
                          actual_input->front().size_bytes()),
              0);
    for (size_t index = 0; index < device_parameter_gradients.size(); ++index) {
      auto repeated =
          ReadDeviceFloats(*executor_, device_parameter_gradients[index]);
      ASSERT_TRUE(repeated.ok()) << repeated.status();
      EXPECT_EQ(
          std::memcmp(first_parameter_gradients[index].data(), repeated->data(),
                      device_parameter_gradients[index].size_bytes()),
          0);
    }
    auto means_after = ReadDeviceFloats(*executor_, retained.intermediates[1]);
    auto inverse_stddevs_after =
        ReadDeviceFloats(*executor_, retained.intermediates[2]);
    ASSERT_TRUE(means_after.ok()) << means_after.status();
    ASSERT_TRUE(inverse_stddevs_after.ok()) << inverse_stddevs_after.status();
    EXPECT_EQ(std::memcmp(means_before->data(), means_after->data(),
                          retained.intermediates[1].size_bytes()),
              0);
    EXPECT_EQ(std::memcmp(inverse_stddevs_before->data(),
                          inverse_stddevs_after->data(),
                          retained.intermediates[2].size_bytes()),
              0);

    // Missing or incorrectly sized statistics must be rejected before launch.
    BackwardState invalid = retained;
    invalid.intermediates.pop_back();
    EXPECT_FALSE(
        (*device_layer)
            ->bwd(*executor_, BufferVec{gradient_pair->device}, invalid)
            .ok());
    invalid = retained;
    invalid.intermediates[1] = later_output->state.intermediates[1];
    EXPECT_FALSE(
        (*device_layer)
            ->bwd(*executor_, BufferVec{gradient_pair->device}, invalid)
            .ok());
    invalid = retained;
    invalid.children.emplace_back();
    EXPECT_FALSE(
        (*device_layer)
            ->bwd(*executor_, BufferVec{gradient_pair->device}, invalid)
            .ok());
  }
}

TEST_F(LayerReferenceTest, FP8IsRejectedConsistently) {
  auto device = LayerNormLayer::Create(*executor_, 16, 1e-5f, DataType::FP8);
  auto reference = LayerNormLayerReference::Create(16, 1e-5f, DataType::FP8);
  ASSERT_FALSE(device.ok());
  ASSERT_FALSE(reference.ok());
  EXPECT_EQ(device.status().code(), absl::StatusCode::kUnimplemented);
  EXPECT_EQ(reference.status().code(), absl::StatusCode::kUnimplemented);
}

TEST_F(LayerReferenceTest, LayerNormSignaturesPreserveSampleDimensions) {
  for (DataType compute : {DataType::FP16, DataType::BF16}) {
    const DataType storage =
        compute == DataType::BF16 ? DataType::BF16 : DataType::FP32;
    auto device = LayerNormLayer::Create(*executor_, 32, 1e-5f, compute, 7);
    auto reference = LayerNormLayerReference::Create(32, 1e-5f, compute, 7);
    ASSERT_TRUE(device.ok()) << device.status();
    ASSERT_TRUE(reference.ok()) << reference.status();
    const ActivationType activation(storage, {-2, 7, 32});
    ASSERT_EQ((*device)->input_types().size(), 1);
    ASSERT_EQ((*device)->output_types().size(), 1);
    ASSERT_EQ((*reference)->input_types().size(), 1);
    ASSERT_EQ((*reference)->output_types().size(), 1);
    EXPECT_EQ((*device)->input_types()[0], activation);
    EXPECT_EQ((*device)->output_types()[0], activation);
    EXPECT_EQ((*reference)->input_types()[0], activation);
    EXPECT_EQ((*reference)->output_types()[0], activation);
  }
}

TEST_F(LayerReferenceTest, LayerNormRejectsNonpositiveSequenceLength) {
  for (int length : {0, -1, -2}) {
    EXPECT_EQ(
        LayerNormLayer::Create(*executor_, 32, 1e-5f, DataType::FP16, length)
            .status()
            .code(),
        absl::StatusCode::kInvalidArgument);
    EXPECT_EQ(LayerNormLayerReference::Create(32, 1e-5f, DataType::FP16, length)
                  .status()
                  .code(),
              absl::StatusCode::kInvalidArgument);
  }
}

}  // namespace
}  // namespace pluto::llm
