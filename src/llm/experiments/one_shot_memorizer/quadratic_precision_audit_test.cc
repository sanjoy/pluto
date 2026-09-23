#include "src/llm/experiments/one_shot_memorizer/quadratic_precision_audit.h"

#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "absl/status/status.h"
#include "gtest/gtest.h"

namespace pluto::llm::one_shot_memorizer {
namespace {

float Bf16(float value) {
  uint32_t bits = std::bit_cast<uint32_t>(value);
  bits += 0x7fff + ((bits >> 16) & 1);
  return std::bit_cast<float>(bits & 0xffff0000);
}

struct Samples {
  std::vector<size_t> lengths{2, 1, 3, 2, 1, 3};  // Held rows0,1,9,10,11.
  std::vector<float> inputs = std::vector<float>(12 * 16);
  std::vector<float> targets = std::vector<float>(12 * 16);
  Samples() {
    for (int row = 0; row < 12; ++row) {
      const float x = static_cast<float>(row % 7 - 3);
      inputs[row * 16] = x;
      targets[row * 16] = x * x;
      targets[row * 16 + 1] = x;
      targets[row * 16 + 2] = 2;
    }
  }
};

Samples FullRankSamples() {
  Samples samples;
  // An unisolvent degree-two design: origin, +/- every axis, and every pair
  // of positive axes. Duplicate it as one held and one fitting sentence so
  // the fit has all 153 independent affine-quadratic rows, not a random rank.
  samples.lengths = {153, 153};
  samples.inputs.assign(306 * 16, 0);
  size_t row = 1;
  for (int i = 0; i < 16; ++i)
    for (float sign : {-1.0f, 1.0f})
      samples.inputs[row++ * 16 + i] = sign;
  for (int i = 0; i < 16; ++i)
    for (int j = i + 1; j < 16; ++j) {
      samples.inputs[row * 16 + i] = 1;
      samples.inputs[row++ * 16 + j] = 1;
    }
  for (size_t index = 0; index < 153 * 16; ++index)
    samples.inputs[153 * 16 + index] = samples.inputs[index];
  samples.targets.assign(samples.inputs.size(), 0);
  for (size_t r = 0; r < 306; ++r) {
    const float* x = samples.inputs.data() + r * 16;
    samples.targets[r * 16] = x[0] * x[1] + 2 * x[2] * x[2] - 3 * x[3] + 4;
    samples.targets[r * 16 + 1] = x[4] * x[5] - x[6];
  }
  return samples;
}

TEST(QuadraticPrecisionAuditTest, BasisHasFixedOrderAndNoScaling) {
  std::vector<float> inputs(32);
  for (int coordinate = 0; coordinate < 16; ++coordinate) {
    inputs[coordinate] = coordinate;
    inputs[16 + coordinate] = -coordinate;
  }
  inputs[16] = -0.0f;
  for (auto precision :
       {QuadraticProductPrecision::kFp32, QuadraticProductPrecision::kBf16}) {
    auto features = MakeQuadraticAuditFeatures(inputs, precision);
    ASSERT_TRUE(features.ok()) << features.status();
    ASSERT_EQ(features->size(), 304u);
    for (int row = 0; row < 2; ++row) {
      for (int i = 0; i < 16; ++i)
        EXPECT_EQ(std::bit_cast<uint32_t>((*features)[row * 152 + i]),
                  std::bit_cast<uint32_t>(inputs[row * 16 + i]));
      int feature = 16;
      for (int i = 0; i < 16; ++i)
        for (int j = i; j < 16; ++j)
          EXPECT_FLOAT_EQ((*features)[row * 152 + feature++],
                          inputs[row * 16 + i] * inputs[row * 16 + j]);
      EXPECT_EQ(feature, 152);
    }
  }
}

TEST(QuadraticPrecisionAuditTest, ChangesOnlyProductRoundingIncludingEvenTies) {
  std::vector<float> inputs(16);
  inputs[0] = 1.0078125f;
  inputs[1] = 1.0234375f;
  inputs[2] = 1.5f;
  inputs[3] = -0.0f;
  auto real =
      MakeQuadraticAuditFeatures(inputs, QuadraticProductPrecision::kFp32);
  auto rounded =
      MakeQuadraticAuditFeatures(inputs, QuadraticProductPrecision::kBf16);
  ASSERT_TRUE(real.ok()) << real.status();
  ASSERT_TRUE(rounded.ok()) << rounded.status();
  EXPECT_FLOAT_EQ((*real)[18], 1.51171875f);
  EXPECT_FLOAT_EQ((*rounded)[18], 1.515625f);
  EXPECT_FLOAT_EQ((*real)[33], 1.53515625f);
  EXPECT_FLOAT_EQ((*rounded)[33], 1.53125f);
  for (int column = 0; column < 16; ++column)
    EXPECT_EQ(std::bit_cast<uint32_t>((*real)[column]),
              std::bit_cast<uint32_t>((*rounded)[column]));
  EXPECT_TRUE(std::signbit((*real)[19]));
  EXPECT_TRUE(std::signbit((*rounded)[19]));
}

TEST(QuadraticPrecisionAuditTest, FixedSentenceSplitAndKnownQuadraticFit) {
  const Samples samples;
  auto audit =
      AuditQuadraticPrecision(samples.inputs, samples.targets, samples.lengths);
  ASSERT_TRUE(audit.ok()) << audit.status();
  EXPECT_EQ(audit->fitting_sentences, 4u);
  EXPECT_EQ(audit->held_sentences, 2u);
  for (const auto* variant :
       {&audit->unrounded_products, &audit->bf16_products}) {
    EXPECT_EQ(variant->fitted_map.input_dim, 152);
    EXPECT_EQ(variant->fitted_map.output_dim, 16);
    EXPECT_EQ(variant->fitted_map.sample_count, 7u);
    EXPECT_DOUBLE_EQ(variant->fitted_map.options.ridge, 1e-6);
    EXPECT_EQ(variant->fp64_parameters.fitting.rows, 7u);
    EXPECT_EQ(variant->fp64_parameters.held.rows, 5u);
    EXPECT_LT(variant->fp64_parameters.fitting.rmse, 1e-5);
    EXPECT_LT(variant->fp64_parameters.held.rmse, 1e-5);
  }
  // Products of these small integers are all exactly representable in BF16.
  EXPECT_EQ(audit->unrounded_products.fitted_map.weights,
            audit->bf16_products.fitted_map.weights);
  EXPECT_EQ(audit->unrounded_products.fitted_map.biases,
            audit->bf16_products.fitted_map.biases);
}

TEST(QuadraticPrecisionAuditTest, RidgeZeroRecoversExactKnownQuadratic) {
  const auto samples = FullRankSamples();
  const auto audit =
      AuditQuadraticPrecision(samples.inputs, samples.targets, samples.lengths);
  ASSERT_TRUE(audit.ok()) << audit.status();
  ASSERT_TRUE(audit->ridge_zero_unrounded.ok())
      << audit->ridge_zero_unrounded.status();
  const auto& diagnostic = *audit->ridge_zero_unrounded;
  EXPECT_DOUBLE_EQ(diagnostic.fitted_map.options.ridge, 0);
  EXPECT_DOUBLE_EQ(diagnostic.fitted_map.options.relative_rank_tolerance,
                   1e-12);
  EXPECT_EQ(diagnostic.fitted_map.numerical_rank, 152u);
  EXPECT_EQ(diagnostic.fitted_map.sample_count, 153u);
  EXPECT_EQ(diagnostic.evaluation.fitting.rows, 153u);
  EXPECT_EQ(diagnostic.evaluation.held.rows, 153u);
  EXPECT_LT(diagnostic.evaluation.fitting.rmse, 1e-13);
  EXPECT_LT(diagnostic.evaluation.held.rmse, 1e-13);
  EXPECT_NEAR(diagnostic.fitted_map.biases[0], 4, 1e-13);
  EXPECT_NEAR(diagnostic.fitted_map.weights[3 * 16], -3, 1e-13);
  EXPECT_NEAR(diagnostic.fitted_map.weights[17 * 16], 1, 1e-13);
  EXPECT_LT(diagnostic.evaluation.fitting.rmse,
            audit->unrounded_products.fp64_parameters.fitting.rmse);
  // The separate diagnostic must not change either primary fit's penalty.
  EXPECT_DOUBLE_EQ(audit->unrounded_products.fitted_map.options.ridge, 1e-6);
  EXPECT_DOUBLE_EQ(audit->bf16_products.fitted_map.options.ridge, 1e-6);
}

TEST(QuadraticPrecisionAuditTest, RidgeZeroReportsRankFailureWithoutRetry) {
  auto samples = FullRankSamples();
  samples.inputs.assign(samples.inputs.size(), 0);
  const auto audit =
      AuditQuadraticPrecision(samples.inputs, samples.targets, samples.lengths);
  ASSERT_TRUE(audit.ok()) << audit.status();
  EXPECT_TRUE(absl::IsFailedPrecondition(audit->ridge_zero_unrounded.status()));
  EXPECT_NE(audit->ridge_zero_unrounded.status().message().find("rank 0"),
            absl::string_view::npos);
  EXPECT_EQ(audit->unrounded_products.fitted_map.numerical_rank, 152u);
  const Samples insufficient;
  const auto small = AuditQuadraticPrecision(
      insufficient.inputs, insufficient.targets, insufficient.lengths);
  ASSERT_TRUE(small.ok()) << small.status();
  EXPECT_TRUE(absl::IsFailedPrecondition(small->ridge_zero_unrounded.status()));
}

TEST(QuadraticPrecisionAuditTest, HeldRowsCannotChangeUnregularizedFit) {
  auto samples = FullRankSamples();
  const auto original =
      AuditQuadraticPrecision(samples.inputs, samples.targets, samples.lengths);
  ASSERT_TRUE(original.ok()) << original.status();
  ASSERT_TRUE(original->ridge_zero_unrounded.ok())
      << original->ridge_zero_unrounded.status();
  for (size_t index = 0; index < 153 * 16; ++index) {
    samples.inputs[index] = 7;
    samples.targets[index] = -100;
  }
  const auto changed =
      AuditQuadraticPrecision(samples.inputs, samples.targets, samples.lengths);
  ASSERT_TRUE(changed.ok()) << changed.status();
  ASSERT_TRUE(changed->ridge_zero_unrounded.ok())
      << changed->ridge_zero_unrounded.status();
  EXPECT_EQ(original->ridge_zero_unrounded->fitted_map.weights,
            changed->ridge_zero_unrounded->fitted_map.weights);
  EXPECT_EQ(original->ridge_zero_unrounded->fitted_map.biases,
            changed->ridge_zero_unrounded->fitted_map.biases);
  EXPECT_DOUBLE_EQ(original->ridge_zero_unrounded->evaluation.fitting.rmse,
                   changed->ridge_zero_unrounded->evaluation.fitting.rmse);
  EXPECT_GT(changed->ridge_zero_unrounded->evaluation.held.rmse, 100);
}

TEST(QuadraticPrecisionAuditTest, HeldTargetsCannotChangeEitherFit) {
  Samples samples;
  auto original =
      AuditQuadraticPrecision(samples.inputs, samples.targets, samples.lengths);
  ASSERT_TRUE(original.ok()) << original.status();
  for (int row : {0, 1, 9, 10, 11})
    for (int coordinate = 0; coordinate < 16; ++coordinate)
      samples.targets[row * 16 + coordinate] += 128;
  auto changed =
      AuditQuadraticPrecision(samples.inputs, samples.targets, samples.lengths);
  ASSERT_TRUE(changed.ok()) << changed.status();
  EXPECT_EQ(original->unrounded_products.fitted_map.weights,
            changed->unrounded_products.fitted_map.weights);
  EXPECT_EQ(original->unrounded_products.fitted_map.biases,
            changed->unrounded_products.fitted_map.biases);
  EXPECT_EQ(original->bf16_products.fitted_map.weights,
            changed->bf16_products.fitted_map.weights);
  EXPECT_EQ(original->bf16_products.fitted_map.biases,
            changed->bf16_products.fitted_map.biases);
  EXPECT_DOUBLE_EQ(original->unrounded_products.fp64_parameters.fitting.rmse,
                   changed->unrounded_products.fp64_parameters.fitting.rmse);
  EXPECT_GT(changed->unrounded_products.fp64_parameters.held.rmse, 127);
}

TEST(QuadraticPrecisionAuditTest, HeldInputsCannotChangeEitherFit) {
  Samples samples;
  auto original =
      AuditQuadraticPrecision(samples.inputs, samples.targets, samples.lengths);
  ASSERT_TRUE(original.ok()) << original.status();
  for (int row : {0, 1, 9, 10, 11})
    samples.inputs[row * 16] = 31.5f;
  auto changed =
      AuditQuadraticPrecision(samples.inputs, samples.targets, samples.lengths);
  ASSERT_TRUE(changed.ok()) << changed.status();
  EXPECT_EQ(original->unrounded_products.fitted_map.weights,
            changed->unrounded_products.fitted_map.weights);
  EXPECT_EQ(original->unrounded_products.fitted_map.biases,
            changed->unrounded_products.fitted_map.biases);
  EXPECT_EQ(original->bf16_products.fitted_map.weights,
            changed->bf16_products.fitted_map.weights);
  EXPECT_GT(changed->unrounded_products.fp64_parameters.held.rmse, 200);
}

TEST(QuadraticPrecisionAuditTest,
     ParameterRoundingAuditsStillUseFp64DotProducts) {
  Samples samples;
  for (int row = 0; row < 12; ++row)
    samples.targets[row * 16] = static_cast<float>((row * 7 + 3) % 11);
  auto audit =
      AuditQuadraticPrecision(samples.inputs, samples.targets, samples.lengths);
  ASSERT_TRUE(audit.ok()) << audit.status();
  auto features = MakeQuadraticAuditFeatures(samples.inputs,
                                             QuadraticProductPrecision::kFp32);
  ASSERT_TRUE(features.ok()) << features.status();
  const auto& variant = audit->unrounded_products;
  for (bool bf16 : {false, true}) {
    double square_error = 0;
    for (int row : {0, 1, 9, 10, 11})
      for (int coordinate = 0; coordinate < 16; ++coordinate) {
        double prediction =
            static_cast<float>(variant.fitted_map.biases[coordinate]);
        for (int feature = 0; feature < 152; ++feature) {
          float coefficient = static_cast<float>(
              variant.fitted_map.weights[feature * 16 + coordinate]);
          if (bf16)
            coefficient = Bf16(coefficient);
          prediction += static_cast<double>((*features)[row * 152 + feature]) *
                        coefficient;
        }
        const double delta =
            prediction - samples.targets[row * 16 + coordinate];
        square_error += delta * delta;
      }
    const auto& actual =
        bf16 ? variant.bf16_weights_fp32_bias : variant.fp32_parameters;
    EXPECT_NEAR(actual.held.rmse, std::sqrt(square_error / (5 * 16)), 1e-13);
  }
}

TEST(QuadraticPrecisionAuditTest, ZeroAndConstantTargetMetricsAreDefined) {
  const std::vector<float> inputs(6 * 16);
  std::vector<float> targets(inputs.size());
  const std::vector<size_t> lengths(6, 1);
  auto zero = AuditQuadraticPrecision(inputs, targets, lengths);
  ASSERT_TRUE(zero.ok()) << zero.status();
  EXPECT_DOUBLE_EQ(zero->unrounded_products.fp64_parameters.fitting.rmse, 0);
  EXPECT_DOUBLE_EQ(zero->bf16_products.fp64_parameters.held.relative_rmse, 0);
  EXPECT_DOUBLE_EQ(
      zero->bf16_products.fp64_parameters.held.centered_relative_rmse, 0);
  for (int row = 1; row < 5; ++row)
    for (int coordinate = 0; coordinate < 16; ++coordinate)
      targets[row * 16 + coordinate] = 1;
  auto constant = AuditQuadraticPrecision(inputs, targets, lengths);
  ASSERT_TRUE(constant.ok()) << constant.status();
  EXPECT_DOUBLE_EQ(constant->unrounded_products.fp64_parameters.held.rmse, 1);
  EXPECT_TRUE(std::isinf(
      constant->unrounded_products.fp64_parameters.held.relative_rmse));
  EXPECT_TRUE(std::isinf(constant->unrounded_products.fp64_parameters.held
                             .centered_relative_rmse));
  EXPECT_DOUBLE_EQ(constant->unrounded_products.fp64_parameters.fitting
                       .centered_relative_rmse,
                   0);
}

TEST(QuadraticPrecisionAuditTest, RejectsInvalidShapesSplitsAndNonBf16Values) {
  const Samples samples;
  EXPECT_FALSE(AuditQuadraticPrecision({}, {}, {}).ok());
  EXPECT_FALSE(
      AuditQuadraticPrecision(samples.inputs, {}, samples.lengths).ok());
  EXPECT_FALSE(
      AuditQuadraticPrecision(samples.inputs, samples.targets, {}).ok());
  EXPECT_FALSE(
      AuditQuadraticPrecision(samples.inputs, samples.targets, {12}).ok());
  EXPECT_FALSE(
      AuditQuadraticPrecision(samples.inputs, samples.targets, {0, 12}).ok());
  EXPECT_FALSE(
      AuditQuadraticPrecision(samples.inputs, samples.targets, {1, 10}).ok());
  EXPECT_FALSE(
      AuditQuadraticPrecision(samples.inputs, samples.targets, {1, 12}).ok());
  EXPECT_FALSE(AuditQuadraticPrecision(samples.inputs, samples.targets,
                                       {std::numeric_limits<size_t>::max(), 1})
                   .ok());
  for (float value : {1.1f, std::numeric_limits<float>::infinity(),
                      std::numeric_limits<float>::quiet_NaN()}) {
    auto invalid = samples.inputs;
    invalid[0] = value;
    EXPECT_FALSE(
        AuditQuadraticPrecision(invalid, samples.targets, samples.lengths)
            .ok());
    auto bad_targets = samples.targets;
    bad_targets[0] = value;
    EXPECT_FALSE(
        AuditQuadraticPrecision(samples.inputs, bad_targets, samples.lengths)
            .ok());
  }
  EXPECT_FALSE(
      MakeQuadraticAuditFeatures({1}, QuadraticProductPrecision::kFp32).ok());
  EXPECT_FALSE(MakeQuadraticAuditFeatures(
                   samples.inputs, static_cast<QuadraticProductPrecision>(99))
                   .ok());
  auto excessive = samples.inputs;
  excessive[0] = std::bit_cast<float>(uint32_t{0x7f7f0000});
  EXPECT_FALSE(
      MakeQuadraticAuditFeatures(excessive, QuadraticProductPrecision::kFp32)
          .ok());
  EXPECT_FALSE(
      MakeQuadraticAuditFeatures(excessive, QuadraticProductPrecision::kBf16)
          .ok());
}

}  // namespace
}  // namespace pluto::llm::one_shot_memorizer
