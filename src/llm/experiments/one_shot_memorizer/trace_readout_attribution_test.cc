#include "src/llm/experiments/one_shot_memorizer/trace_readout_attribution.h"

#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "gtest/gtest.h"

namespace pluto::llm::one_shot_memorizer {
namespace {

struct Inputs {
  std::vector<std::vector<float>> rows{
      {1, -2, 3, 4}, {2, 1, -1, 5}, {4, -3, 2, 7}};
  std::vector<float> gamma{2, -0.5, 0, 1.25};
  std::vector<float> beta{0.25, -1, 2, -0.5};
  std::vector<float> target{1.25, -2, 0.5, 3};
  std::vector<float> rival{-0.75, 1, 0.25, -1};
  std::vector<float> actual{1.5, -0.25, 2, 0.5};
  double epsilon = 0.03125;

  absl::StatusOr<TraceReadoutAttribution> Compute() const {
    std::vector<absl::Span<const float>> views;
    for (const auto& row : rows)
      views.push_back(row);
    return ComputeTraceReadoutAttribution(views, gamma, beta, target, rival,
                                          actual, epsilon);
  }
};

std::vector<double> LiteralNormalize(const Inputs& input) {
  const auto& row = input.rows.back();
  const double mean = std::accumulate(row.begin(), row.end(), 0.0) / row.size();
  double variance = 0;
  for (float value : row)
    variance += (value - mean) * (value - mean) / row.size();
  std::vector<double> normalized;
  for (size_t i = 0; i < row.size(); ++i)
    normalized.push_back(input.gamma[i] * (row[i] - mean) /
                             std::sqrt(variance + input.epsilon) +
                         input.beta[i]);
  return normalized;
}

float RoundBf16(double input) {
  const float value = static_cast<float>(input);
  uint32_t bits = std::bit_cast<uint32_t>(value);
  bits += 0x7fffu + ((bits >> 16) & 1u);
  return std::bit_cast<float>(bits & 0xffff0000u);
}

TEST(TraceReadoutAttributionTest, TelescopesToLiteralFinalLayerNormAndHead) {
  Inputs input;
  const auto normalized = LiteralNormalize(input);
  for (size_t i = 0; i < normalized.size(); ++i)
    input.actual[i] = RoundBf16(normalized[i]);
  auto result = input.Compute();
  ASSERT_TRUE(result.ok()) << result.status();
  ASSERT_EQ(result->boundaries.size(), input.rows.size());
  ASSERT_EQ(result->direction.size(), input.gamma.size());
  double literal_margin = 0, beta_margin = 0, actual_margin = 0;
  for (size_t i = 0; i < input.gamma.size(); ++i) {
    const double difference =
        static_cast<double>(input.target[i]) - input.rival[i];
    EXPECT_DOUBLE_EQ(result->embedding_difference[i], difference);
    literal_margin += normalized[i] * difference;
    beta_margin += input.beta[i] * difference;
    actual_margin += input.actual[i] * difference;
    EXPECT_DOUBLE_EQ(result->actual_dimension_contributions[i],
                     input.actual[i] * difference);
  }
  EXPECT_NEAR(result->ideal_margin, literal_margin, 1e-12);
  EXPECT_NEAR(result->accounted_margin, literal_margin, 1e-12);
  EXPECT_NEAR(result->accounting_residual, 0, 1e-12);
  EXPECT_DOUBLE_EQ(result->beta_contribution, beta_margin);
  EXPECT_NEAR(result->actual_normalized_margin, actual_margin, 1e-12);
  EXPECT_NEAR(result->normalization_residual, actual_margin - literal_margin,
              1e-12);
  EXPECT_NE(result->normalization_residual, 0);
  EXPECT_NEAR(
      std::accumulate(result->direction.begin(), result->direction.end(), 0.0),
      0, 1e-14);
  for (size_t boundary = 0; boundary < input.rows.size(); ++boundary) {
    double total = 0;
    for (size_t i = 0; i < input.gamma.size(); ++i) {
      const double value =
          static_cast<double>(input.rows[boundary][i]) -
          (boundary == 0 ? 0.0
                         : static_cast<double>(input.rows[boundary - 1][i]));
      const double term = result->direction[i] * value;
      EXPECT_DOUBLE_EQ(result->boundaries[boundary].dimension_contributions[i],
                       term);
      total += term;
    }
    EXPECT_NEAR(result->boundaries[boundary].total, total, 1e-12);
  }
}

TEST(TraceReadoutAttributionTest,
     ConstantOffsetsAtEachBoundaryDoNotChangeTotals) {
  Inputs original;
  Inputs shifted = original;
  const std::vector<float> offsets{16, -8, 32};
  for (size_t boundary = 0; boundary < shifted.rows.size(); ++boundary)
    for (float& value : shifted.rows[boundary])
      value += offsets[boundary];
  auto before = original.Compute();
  auto after = shifted.Compute();
  ASSERT_TRUE(before.ok()) << before.status();
  ASSERT_TRUE(after.ok()) << after.status();
  EXPECT_EQ(before->direction, after->direction);
  EXPECT_DOUBLE_EQ(before->final_variance, after->final_variance);
  EXPECT_DOUBLE_EQ(after->final_mean, before->final_mean + offsets.back());
  for (size_t boundary = 0; boundary < original.rows.size(); ++boundary)
    EXPECT_NEAR(before->boundaries[boundary].total,
                after->boundaries[boundary].total, 1e-12);
  EXPECT_NEAR(before->ideal_margin, after->ideal_margin, 1e-12);
  EXPECT_NEAR(before->accounted_margin, after->accounted_margin, 1e-12);
}

TEST(TraceReadoutAttributionTest,
     IntermediateChoicesOnlyRedistributeAccounting) {
  Inputs original;
  Inputs changed = original;
  changed.rows[1] = {-100, 45, 87, -12};
  auto before = original.Compute();
  auto after = changed.Compute();
  ASSERT_TRUE(before.ok()) << before.status();
  ASSERT_TRUE(after.ok()) << after.status();
  EXPECT_EQ(before->direction, after->direction);
  EXPECT_NE(before->boundaries[1].total, after->boundaries[1].total);
  EXPECT_NEAR(before->boundaries[1].total + before->boundaries[2].total,
              after->boundaries[1].total + after->boundaries[2].total, 1e-12);
  EXPECT_DOUBLE_EQ(before->ideal_margin, after->ideal_margin);
  EXPECT_NEAR(before->accounted_margin, after->accounted_margin, 1e-12);
}

TEST(TraceReadoutAttributionTest,
     ConstantFinalStateRetainsBetaAndPositiveEpsilon) {
  Inputs input{.rows = {{1, -1}, {5, 5}},
               .gamma = {2, 1},
               .beta = {3, 4},
               .target = {1, -1},
               .rival = {0, 0},
               .actual = {3, 4},
               .epsilon = 0.25};
  auto result = input.Compute();
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result->direction, (std::vector<double>{3, -3}));
  EXPECT_DOUBLE_EQ(result->final_variance, 0);
  EXPECT_DOUBLE_EQ(result->normalization_denominator, 0.5);
  EXPECT_DOUBLE_EQ(result->boundaries[0].total, 6);
  EXPECT_DOUBLE_EQ(result->boundaries[1].total, -6);
  EXPECT_DOUBLE_EQ(result->beta_contribution, -1);
  EXPECT_DOUBLE_EQ(result->accounted_margin, -1);
  EXPECT_DOUBLE_EQ(result->ideal_margin, -1);
  EXPECT_DOUBLE_EQ(result->actual_normalized_margin, -1);
  EXPECT_DOUBLE_EQ(result->normalization_residual, 0);
}

TEST(TraceReadoutAttributionTest, SingletonWidthAndIdenticalEmbeddings) {
  Inputs scalar{.rows = {{-100}, {50}},
                .gamma = {9},
                .beta = {7},
                .target = {2},
                .rival = {-3},
                .actual = {7}};
  auto result = scalar.Compute();
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result->direction, (std::vector<double>{0}));
  EXPECT_DOUBLE_EQ(result->ideal_margin, 35);
  EXPECT_DOUBLE_EQ(result->accounted_margin, 35);
  EXPECT_DOUBLE_EQ(result->actual_normalized_margin, 35);
  for (const auto& boundary : result->boundaries)
    EXPECT_DOUBLE_EQ(boundary.total, 0);

  Inputs identical;
  identical.rival = identical.target;
  result = identical.Compute();
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_DOUBLE_EQ(result->ideal_margin, 0);
  EXPECT_DOUBLE_EQ(result->actual_normalized_margin, 0);
  EXPECT_DOUBLE_EQ(result->accounted_margin, 0);
  EXPECT_DOUBLE_EQ(result->beta_contribution, 0);
  for (const auto& boundary : result->boundaries)
    EXPECT_DOUBLE_EQ(boundary.total, 0);
}

TEST(TraceReadoutAttributionTest,
     SwappingTargetAndRivalReversesEveryMarginTerm) {
  Inputs input;
  auto forward = input.Compute();
  std::swap(input.target, input.rival);
  auto reverse = input.Compute();
  ASSERT_TRUE(forward.ok()) << forward.status();
  ASSERT_TRUE(reverse.ok()) << reverse.status();
  for (size_t dim = 0; dim < input.gamma.size(); ++dim) {
    EXPECT_DOUBLE_EQ(forward->direction[dim], -reverse->direction[dim]);
    EXPECT_DOUBLE_EQ(forward->actual_dimension_contributions[dim],
                     -reverse->actual_dimension_contributions[dim]);
    for (size_t boundary = 0; boundary < input.rows.size(); ++boundary)
      EXPECT_DOUBLE_EQ(
          forward->boundaries[boundary].dimension_contributions[dim],
          -reverse->boundaries[boundary].dimension_contributions[dim]);
  }
  EXPECT_DOUBLE_EQ(forward->beta_contribution, -reverse->beta_contribution);
  EXPECT_DOUBLE_EQ(forward->ideal_margin, -reverse->ideal_margin);
  EXPECT_DOUBLE_EQ(forward->normalization_residual,
                   -reverse->normalization_residual);
}

TEST(TraceReadoutAttributionTest, FiniteExtremeInputsDoNotSubtractInFloat) {
  const float large = std::numeric_limits<float>::max();
  Inputs input{.rows = {{large, -large}, {-large, large}},
               .gamma = {1, 1},
               .beta = {0, 0},
               .target = {1, -1},
               .rival = {0, 0},
               .actual = {-1, 1}};
  auto result = input.Compute();
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_NEAR(result->boundaries[0].total, 2, 1e-14);
  EXPECT_NEAR(result->boundaries[1].total, -4, 1e-14);
  EXPECT_NEAR(result->ideal_margin, -2, 1e-14);
  EXPECT_NEAR(result->accounted_margin, -2, 1e-14);
  EXPECT_NEAR(result->actual_normalized_margin, -2, 1e-14);
}

TEST(TraceReadoutAttributionTest, RejectsEmptyMismatchedAndNonfiniteInputs) {
  Inputs empty;
  empty.rows.clear();
  EXPECT_EQ(empty.Compute().status().code(),
            absl::StatusCode::kInvalidArgument);
  empty = Inputs{};
  empty.gamma.clear();
  EXPECT_EQ(empty.Compute().status().code(),
            absl::StatusCode::kInvalidArgument);
  auto field = [](Inputs& input, int index) -> std::vector<float>& {
    std::vector<float>* fields[]{&input.rows[0], &input.rows[1], &input.rows[2],
                                 &input.gamma,   &input.beta,    &input.target,
                                 &input.rival,   &input.actual};
    return *fields[index];
  };
  for (int vector = 0; vector < 8; ++vector) {
    Inputs bad;
    field(bad, vector).pop_back();
    EXPECT_EQ(bad.Compute().status().code(),
              absl::StatusCode::kInvalidArgument);
    for (float value : {std::numeric_limits<float>::quiet_NaN(),
                        std::numeric_limits<float>::infinity(),
                        -std::numeric_limits<float>::infinity()}) {
      bad = Inputs{};
      field(bad, vector)[0] = value;
      EXPECT_EQ(bad.Compute().status().code(),
                absl::StatusCode::kInvalidArgument);
    }
  }
  for (double epsilon : {0.0, -1.0, std::numeric_limits<double>::infinity(),
                         std::numeric_limits<double>::quiet_NaN()}) {
    Inputs bad;
    bad.epsilon = epsilon;
    EXPECT_EQ(bad.Compute().status().code(),
              absl::StatusCode::kInvalidArgument);
  }
}

struct DenseInputs {
  std::vector<float> gelu{2, -1, 0.5};
  std::vector<float> weights{1, 2, -3, 4, 5, -2};  // [hidden=3,width=2].
  std::vector<float> bias{0.25, -0.5};
  std::vector<float> actual{7.75, -1.5};
  std::vector<double> direction{2, -1};

  absl::StatusOr<DenseProjectionReadoutAttribution> Compute() const {
    return ComputeDenseProjectionReadoutAttribution(gelu, weights, bias, actual,
                                                    direction);
  }
};

TEST(DenseProjectionReadoutAttributionTest,
     ClosesToLiteralRowMajorDenseProjectionIncludingBias) {
  DenseInputs input;
  auto result = input.Compute();
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result->neuron_directions, (std::vector<double>{0, -10, 12}));
  EXPECT_EQ(result->neuron_contributions, (std::vector<double>{0, 10, 6}));
  EXPECT_DOUBLE_EQ(result->bias_contribution, 1);
  double literal = 0;
  for (size_t dimension = 0; dimension < input.direction.size(); ++dimension) {
    double output = input.bias[dimension];
    for (size_t neuron = 0; neuron < input.gelu.size(); ++neuron)
      output += static_cast<double>(input.gelu[neuron]) *
                input.weights[neuron * input.direction.size() + dimension];
    EXPECT_DOUBLE_EQ(output, input.actual[dimension]);
    literal += output * input.direction[dimension];
  }
  EXPECT_DOUBLE_EQ(result->ideal_directional_sum, literal);
  EXPECT_DOUBLE_EQ(result->actual_directional_sum, literal);
  EXPECT_DOUBLE_EQ(result->rounding_residual, 0);
}

TEST(DenseProjectionReadoutAttributionTest,
     ReportsCapturedNumericalResidualWithoutRedistributingIt) {
  DenseInputs input;
  input.actual = {7.5, -1.25};
  auto result = input.Compute();
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result->neuron_contributions, (std::vector<double>{0, 10, 6}));
  EXPECT_DOUBLE_EQ(result->bias_contribution, 1);
  EXPECT_DOUBLE_EQ(result->ideal_directional_sum, 17);
  EXPECT_DOUBLE_EQ(result->actual_directional_sum, 16.25);
  EXPECT_DOUBLE_EQ(result->rounding_residual, -0.75);
}

TEST(DenseProjectionReadoutAttributionTest, ZeroNeuronRemovesOnlyItsOwnTerm) {
  DenseInputs input;
  auto before = input.Compute();
  ASSERT_TRUE(before.ok()) << before.status();
  input.gelu[1] = 0;
  input.actual = {4.75, 2.5};  // Literal projection with neuron 1 set to zero.
  auto after = input.Compute();
  ASSERT_TRUE(after.ok()) << after.status();
  EXPECT_EQ(after->neuron_directions, before->neuron_directions);
  EXPECT_EQ(after->neuron_contributions, (std::vector<double>{0, 0, 6}));
  EXPECT_DOUBLE_EQ(after->bias_contribution, before->bias_contribution);
  EXPECT_DOUBLE_EQ(before->ideal_directional_sum - after->ideal_directional_sum,
                   before->neuron_contributions[1]);
  EXPECT_DOUBLE_EQ(after->actual_directional_sum, 7);
  EXPECT_DOUBLE_EQ(after->rounding_residual, 0);
}

TEST(DenseProjectionReadoutAttributionTest, RetainsSignedNeuronContributions) {
  DenseInputs input;
  input.gelu[1] = 1;
  input.actual = {1.75, 6.5};
  auto result = input.Compute();
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result->neuron_contributions, (std::vector<double>{0, -10, 6}));
  EXPECT_DOUBLE_EQ(result->ideal_directional_sum, -3);
  EXPECT_DOUBLE_EQ(result->actual_directional_sum, -3);
  EXPECT_DOUBLE_EQ(result->rounding_residual, 0);
  for (double& value : input.direction)
    value = -value;
  auto reverse = input.Compute();
  ASSERT_TRUE(reverse.ok()) << reverse.status();
  EXPECT_EQ(reverse->neuron_contributions, (std::vector<double>{0, 10, -6}));
  EXPECT_DOUBLE_EQ(reverse->bias_contribution, -result->bias_contribution);
  EXPECT_DOUBLE_EQ(reverse->ideal_directional_sum,
                   -result->ideal_directional_sum);
  EXPECT_DOUBLE_EQ(reverse->actual_directional_sum,
                   -result->actual_directional_sum);
}

TEST(DenseProjectionReadoutAttributionTest, ZeroDirectionAccountsForNothing) {
  DenseInputs input;
  input.direction = {0, 0};
  auto result = input.Compute();
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result->neuron_contributions, (std::vector<double>{0, 0, 0}));
  EXPECT_DOUBLE_EQ(result->bias_contribution, 0);
  EXPECT_DOUBLE_EQ(result->ideal_directional_sum, 0);
  EXPECT_DOUBLE_EQ(result->actual_directional_sum, 0);
  EXPECT_DOUBLE_EQ(result->rounding_residual, 0);
}

TEST(DenseProjectionReadoutAttributionTest,
     PreservesSmallTermBetweenLargeCancellingNeuronTerms) {
  DenseInputs input{.gelu = {1e20f, 1, -1e20f},
                    .weights = {1, 1, 1},
                    .bias = {0},
                    .actual = {1},
                    .direction = {1}};
  auto result = input.Compute();
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_DOUBLE_EQ(result->ideal_directional_sum, 1);
  EXPECT_DOUBLE_EQ(result->actual_directional_sum, 1);
  EXPECT_DOUBLE_EQ(result->rounding_residual, 0);
}

TEST(DenseProjectionReadoutAttributionTest,
     RetainsCompensatedDirectionalCoefficientForZeroGeluNeuron) {
  DenseInputs input{.gelu = {0, 2},
                    .weights = {1e20f, 1, -1e20f, 1e20f, 1, -1e20f},
                    .bias = {0, 0, 0},
                    .actual = {2e20f, 2, -2e20f},
                    .direction = {1, 1, 1}};
  auto result = input.Compute();
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result->neuron_directions, (std::vector<double>{1, 1}));
  EXPECT_EQ(result->neuron_contributions, (std::vector<double>{0, 2}));
  EXPECT_DOUBLE_EQ(result->ideal_directional_sum, 2);
  EXPECT_DOUBLE_EQ(result->actual_directional_sum, 2);
  EXPECT_DOUBLE_EQ(result->rounding_residual, 0);
}

TEST(DenseProjectionReadoutAttributionTest,
     DoesNotMultiplyEffectiveInputsInFloat) {
  const float large = std::numeric_limits<float>::max();
  DenseInputs input{.gelu = {large},
                    .weights = {large},
                    .bias = {0},
                    .actual = {0},
                    .direction = {1}};
  auto result = input.Compute();
  ASSERT_TRUE(result.ok()) << result.status();
  const double product = static_cast<double>(large) * large;
  EXPECT_DOUBLE_EQ(result->neuron_contributions[0], product);
  EXPECT_DOUBLE_EQ(result->ideal_directional_sum, product);
  EXPECT_DOUBLE_EQ(result->rounding_residual, -product);
}

TEST(DenseProjectionReadoutAttributionTest,
     RejectsEmptyMismatchedAndNonfiniteInputs) {
  auto field = [](DenseInputs& input, int index) -> std::vector<float>& {
    std::vector<float>* fields[]{&input.gelu, &input.weights, &input.bias,
                                 &input.actual};
    return *fields[index];
  };
  for (int index = 0; index < 4; ++index) {
    DenseInputs bad;
    field(bad, index).clear();
    EXPECT_EQ(bad.Compute().status().code(),
              absl::StatusCode::kInvalidArgument);
    bad = DenseInputs{};
    field(bad, index).pop_back();
    EXPECT_EQ(bad.Compute().status().code(),
              absl::StatusCode::kInvalidArgument);
    for (float value : {std::numeric_limits<float>::quiet_NaN(),
                        std::numeric_limits<float>::infinity(),
                        -std::numeric_limits<float>::infinity()}) {
      bad = DenseInputs{};
      field(bad, index)[0] = value;
      EXPECT_EQ(bad.Compute().status().code(),
                absl::StatusCode::kInvalidArgument);
    }
  }
  DenseInputs bad;
  bad.direction.clear();
  EXPECT_EQ(bad.Compute().status().code(), absl::StatusCode::kInvalidArgument);
  bad = DenseInputs{};
  bad.direction.pop_back();
  EXPECT_EQ(bad.Compute().status().code(), absl::StatusCode::kInvalidArgument);
  for (double value : {std::numeric_limits<double>::quiet_NaN(),
                       std::numeric_limits<double>::infinity(),
                       -std::numeric_limits<double>::infinity()}) {
    bad = DenseInputs{};
    bad.direction[0] = value;
    EXPECT_EQ(bad.Compute().status().code(),
              absl::StatusCode::kInvalidArgument);
  }
}

TEST(DenseProjectionReadoutAttributionTest, RejectsArithmeticOverflow) {
  const double large = std::numeric_limits<double>::max();
  DenseInputs input{.gelu = {1},
                    .weights = {2},
                    .bias = {0},
                    .actual = {0},
                    .direction = {large}};
  EXPECT_EQ(input.Compute().status().code(), absl::StatusCode::kOutOfRange);
  input.weights = {0};
  input.bias = {2};
  EXPECT_EQ(input.Compute().status().code(), absl::StatusCode::kOutOfRange);
  input.bias = {0};
  input.actual = {2};
  EXPECT_EQ(input.Compute().status().code(), absl::StatusCode::kOutOfRange);
  input.weights = {-1};
  input.actual = {1};
  // Both sums are finite, but their difference is not.
  EXPECT_EQ(input.Compute().status().code(), absl::StatusCode::kOutOfRange);
  input.gelu = {1, 1};
  input.weights = {1, 1};
  input.actual = {0};
  // Individual neuron terms are finite, but the ideal sum is not.
  EXPECT_EQ(input.Compute().status().code(), absl::StatusCode::kOutOfRange);
}

}  // namespace
}  // namespace pluto::llm::one_shot_memorizer
