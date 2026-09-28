#include "src/llm/layers/sequence_delta_net.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "src/cuda/page_locked_host_array.h"
#include "src/util/status_macros.h"

namespace pluto::llm {
namespace {

using Parameters = cached_attention_ops::DeltaNetParameters;

struct Example {
  Parameters p;
  int length;
  // QKV, Z, A, B, convolution, A_log, dt_bias, and norm, respectively.
  std::array<std::vector<double>, 8> values;
  std::vector<float> upstream;
};

Example MakeExample(int length, int kh, int vh, int kd, int vd, bool round) {
  Example example;
  example.p = {kh, vh, kd, vd, 3, 0.03f, round};
  example.length = length;
  const int channels = 2 * kh * kd + vh * vd;
  const int sizes[] = {length * channels,
                       length * vh * vd,
                       length * vh,
                       length * vh,
                       channels * 3,
                       vh,
                       vh,
                       vd};
  for (int tensor = 0; tensor < 8; ++tensor)
    for (int i = 0; i < sizes[tensor]; ++i) {
      float value = 0.15f + 0.4f * std::sin(0.37f * i + 0.41f * tensor);
      if (tensor == 5)
        value -= 0.7f;
      if (tensor == 7)
        value += 1.f;
      // The CPU reference starts at exactly the same physical input values.
      example.values[tensor].push_back(
          tensor < 4 ? static_cast<float>(__nv_bfloat16(value)) : value);
    }
  example.upstream.resize(length * vh * vd);
  for (int i = (length - 1) * vh * vd; i < length * vh * vd; ++i)
    example.upstream[i] = 0.4f * std::cos(0.2f * i);
  return example;
}

// Deliberately scalar, double-precision specification of every operation.
// Rounding offsets let numerical differentiation test BF16 STE: record each
// quantizer's (rounded - unrounded) at the base point, then hold those offsets
// constant for +/-epsilon evaluations. This differentiates the rounded forward
// graph with identity quantizer derivatives, rather than its staircase values.
std::vector<double> Reference(const Example& example,
                              std::vector<double>* recorded_offsets = nullptr,
                              const std::vector<double>* offsets = nullptr) {
  const auto& p = example.p;
  const int kh = p.key_heads, vh = p.value_heads;
  const int kd = p.key_head_dim, vd = p.value_head_dim;
  const int kernel = p.conv_kernel_dim;
  const int channels = 2 * kh * kd + vh * vd;
  const auto& v = example.values;
  size_t rounding_index = 0;
  auto round = [&](double x, bool enabled) {
    if (!enabled)
      return x;
    if (offsets)
      return x + (*offsets)[rounding_index++];
    const double rounded =
        static_cast<float>(__nv_bfloat16(static_cast<float>(x)));
    if (recorded_offsets)
      recorded_offsets->push_back(rounded - x);
    return rounded;
  };
  std::vector<double> convolved(example.length * channels);
  for (int t = 0; t < example.length; ++t)
    for (int c = 0; c < channels; ++c) {
      double sum = 0;
      for (int tap = 0; tap < kernel; ++tap) {
        const int source = t + tap - kernel + 1;
        if (source >= 0)
          sum += v[0][source * channels + c] * v[4][c * kernel + tap];
      }
      sum = round(sum, p.round_to_bfloat16);
      convolved[t * channels + c] =
          round(sum / (1 + std::exp(-sum)), p.round_to_bfloat16);
    }
  std::vector<double> states(vh * kd * vd);
  std::vector<double> core(example.length * vh * vd);
  for (int head = 0; head < vh; ++head) {
    const int key_head = head / (vh / kh);
    for (int t = 0; t < example.length; ++t) {
      std::vector<double> q(kd), k(kd);
      double q_norm = 1e-6, k_norm = 1e-6;
      for (int d = 0; d < kd; ++d) {
        q[d] = convolved[t * channels + key_head * kd + d];
        k[d] = convolved[t * channels + (kh + key_head) * kd + d];
        q_norm += q[d] * q[d];
        k_norm += k[d] * k[d];
      }
      for (int d = 0; d < kd; ++d) {
        q[d] /= std::sqrt(q_norm * kd);
        k[d] /= std::sqrt(k_norm);
      }
      const double alpha = v[2][t * vh + head] + v[6][head];
      const double softplus =
          std::max(alpha, 0.) + std::log1p(std::exp(-std::abs(alpha)));
      const double decay = std::exp(-std::exp(v[5][head]) * softplus);
      const double beta =
          round(1 / (1 + std::exp(-v[3][t * vh + head])), p.round_to_bfloat16);
      for (int channel = 0; channel < vd; ++channel) {
        double prediction = 0;
        for (int d = 0; d < kd; ++d) {
          const int i = (head * kd + d) * vd + channel;
          states[i] *= decay;
          prediction += states[i] * k[d];
        }
        const double delta =
            (convolved[t * channels + 2 * kh * kd + head * vd + channel] -
             prediction) *
            beta;
        double output = 0;
        for (int d = 0; d < kd; ++d) {
          const int i = (head * kd + d) * vd + channel;
          states[i] += k[d] * delta;
          output += states[i] * q[d];
        }
        core[(t * vh + head) * vd + channel] =
            round(output, p.round_to_bfloat16);
      }
    }
  }
  std::vector<double> result(core.size());
  for (int row = 0; row < example.length * vh; ++row) {
    double sum = 0;
    for (int d = 0; d < vd; ++d)
      sum += core[row * vd + d] * core[row * vd + d];
    const double inverse = 1 / std::sqrt(sum / vd + p.rms_norm_epsilon);
    for (int d = 0; d < vd; ++d) {
      const int i = row * vd + d;
      const double normalized = round(core[i] * inverse, p.round_to_bfloat16);
      const double affine = round(normalized * v[7][d], p.round_to_bfloat16);
      result[i] = round(affine * v[1][i] / (1 + std::exp(-v[1][i])), true);
    }
  }
  return result;
}

double Loss(const Example& example, const std::vector<double>& offsets) {
  const auto output = Reference(example, nullptr, &offsets);
  return std::inner_product(output.begin(), output.end(),
                            example.upstream.begin(), 0.);
}

class SequenceDeltaNetTest : public testing::Test {
 protected:
  void SetUp() override {
    auto executor = cuda::Executor::Create();
    ASSERT_TRUE(executor.ok()) << executor.status();
    executor_ = std::move(*executor);
  }
  void TearDown() override {
    if (executor_)
      EXPECT_TRUE(executor_->Synchronize().ok());
  }

  template <class T>
  absl::StatusOr<Buffer> Upload(const std::vector<T>& values) {
    ASSIGN_OR_RETURN(auto host,
                     cuda::PageLockedHostArray<T>::CopyFrom(*executor_, values));
    ASSIGN_OR_RETURN(auto buffer,
                     Buffer::Allocate(*executor_, values.size() * sizeof(T)));
    RETURN_IF_ERROR(cuda::CudaStatus(
        cudaMemcpyAsync(buffer.data(), host.data(), buffer.size_bytes(),
                        cudaMemcpyHostToDevice, executor_->stream()),
        "test upload"));
    return buffer;
  }

  template <class T>
  std::vector<float> Download(const Buffer& buffer) {
    auto host = cuda::PageLockedHostArray<T>::Allocate(
        *executor_, buffer.size_bytes() / sizeof(T));
    EXPECT_TRUE(host.ok()) << host.status();
    if (!host.ok())
      return {};
    EXPECT_EQ(cudaMemcpyAsync(host->data(), buffer.data(), buffer.size_bytes(),
                              cudaMemcpyDeviceToHost, executor_->stream()),
              cudaSuccess);
    EXPECT_TRUE(executor_->Synchronize().ok());
    std::vector<float> result;
    for (auto value : *host)
      result.push_back(static_cast<float>(value));
    return result;
  }

  absl::StatusOr<std::unique_ptr<SequenceDeltaNetLayer>> MakeLayer(
      const Example& e, bool active = true) {
    inputs_.clear();
    parameters_.clear();
    for (int i = 0; i < 8; ++i) {
      if (i < 4) {
        std::vector<__nv_bfloat16> values;
        for (double value : e.values[i])
          values.emplace_back(static_cast<float>(value));
        ASSIGN_OR_RETURN(auto input, Upload(values));
        inputs_.push_back(std::move(input));
      } else {
        const std::vector<float> values(e.values[i].begin(), e.values[i].end());
        ASSIGN_OR_RETURN(auto weight, Upload(values));
        ASSIGN_OR_RETURN(auto parameter,
                         BlockParameter::Create(*executor_, std::move(weight),
                                                DataType::FP32));
        if (active)
          RETURN_IF_ERROR(parameter->Activate());
        parameters_.push_back(std::move(parameter));
      }
    }
    return SequenceDeltaNetLayer::Create(*executor_, e.p, parameters_[0],
                                         parameters_[1], parameters_[2],
                                         parameters_[3], e.length);
  }

  std::unique_ptr<cuda::Executor> executor_;
  BufferVec inputs_;
  std::vector<std::shared_ptr<BlockParameter>> parameters_;
};

TEST_F(SequenceDeltaNetTest, ForwardMatchesScalarAndCachedTokenInference) {
  for (const auto& shape :
       {std::array<int, 4>{1, 2, 3, 5}, {2, 4, 5, 35}, {16, 48, 128, 128}}) {
    const auto e = MakeExample(3, shape[0], shape[1], shape[2], shape[3], true);
    auto layer = MakeLayer(e);
    ASSERT_TRUE(layer.ok()) << layer.status();
    auto result = (*layer)->fwd(*executor_, inputs_);
    ASSERT_TRUE(result.ok()) << result.status();
    const auto actual = Download<__nv_bfloat16>(result->outputs[0]);
    const auto expected = Reference(e);
    ASSERT_EQ(actual.size(), expected.size());
    for (size_t i = 0; i < actual.size(); ++i)
      EXPECT_NEAR(actual[i], expected[i],
                  0.00015 + std::abs(expected[i]) * 0.015)
          << i;

    auto cache = cached_attention_ops::DeltaNetState::Create(*executor_, e.p);
    ASSERT_TRUE(cache.ok()) << cache.status();
    const int channels = 2 * e.p.key_heads * e.p.key_head_dim +
                         e.p.value_heads * e.p.value_head_dim;
    const int width = e.p.value_heads * e.p.value_head_dim;
    const int counts[] = {channels, width, e.p.value_heads, e.p.value_heads};
    for (int t = 0; t < e.length; ++t) {
      BufferVec converted;
      for (int input = 0; input < 4; ++input) {
        const auto begin = e.values[input].begin() + t * counts[input];
        const std::vector<float> slice(begin, begin + counts[input]);
        auto buffer = Upload(slice);
        ASSERT_TRUE(buffer.ok());
        converted.push_back(std::move(*buffer));
      }
      auto output = Buffer::Allocate(*executor_, width * sizeof(float));
      ASSERT_TRUE(output.ok());
      auto read = [](const Buffer& b) {
        return static_cast<const float*>(b.data());
      };
      ASSERT_TRUE((*cache)
                      ->Step(read(converted[0]), read(converted[1]),
                             read(converted[2]), read(converted[3]),
                             read(parameters_[0]->value()),
                             read(parameters_[1]->value()),
                             read(parameters_[2]->value()),
                             read(parameters_[3]->value()),
                             static_cast<float*>(output->data()))
                      .ok());
      const auto cached = Download<float>(*output);
      for (int i = 0; i < width; ++i)
        EXPECT_NEAR(actual[t * width + i], cached[i],
                    0.00015 + std::abs(cached[i]) * 0.015)
            << t << ":" << i;
    }
    // A second sequence must not inherit cache state from the first one.
    auto repeat = (*layer)->fwd(*executor_, inputs_);
    ASSERT_TRUE(repeat.ok());
    EXPECT_EQ(Download<__nv_bfloat16>(repeat->outputs[0]), actual);
  }
}

TEST_F(SequenceDeltaNetTest, AllInputAndParameterGradientsMatchNumericalBptt) {
  for (bool round : {false, true}) {
    SCOPED_TRACE(round);
    auto e = MakeExample(5, 1, 2, 3, 5, round);
    auto layer = MakeLayer(e);
    ASSERT_TRUE(layer.ok()) << layer.status();
    auto result = (*layer)->fwd(*executor_, inputs_);
    ASSERT_TRUE(result.ok()) << result.status();
    auto upstream = Upload(e.upstream);
    ASSERT_TRUE(upstream.ok());
    auto gradient =
        (*layer)->bwd(*executor_, {*upstream}, std::move(result->state));
    ASSERT_TRUE(gradient.ok()) << gradient.status();
    std::vector<double> offsets;
    Reference(e, &offsets);
    for (int tensor = 0; tensor < 8; ++tensor) {
      const auto actual =
          Download<float>(tensor < 4 ? (*gradient)[tensor]
                                     : parameters_[tensor - 4]->gradient());
      ASSERT_EQ(actual.size(), e.values[tensor].size());
      for (size_t i = 0; i < actual.size(); ++i) {
        const double original = e.values[tensor][i];
        constexpr double epsilon = 1e-5;
        e.values[tensor][i] = original + epsilon;
        const double plus = Loss(e, offsets);
        e.values[tensor][i] = original - epsilon;
        const double minus = Loss(e, offsets);
        e.values[tensor][i] = original;
        const double expected = (plus - minus) / (2 * epsilon);
        EXPECT_NEAR(actual[i], expected, 2e-5 + std::abs(expected) * 0.002)
            << "tensor=" << tensor << " index=" << i;
      }
    }
    const auto dqkv = Download<float>((*gradient)[0]);
    const int channels = dqkv.size() / e.length;
    double first_token_gradient = 0;
    for (int c = 0; c < channels; ++c)
      first_token_gradient += std::abs(dqkv[c]);
    // Loss exists only at the last token; this catches accidentally truncating
    // either the recurrent or convolutional dependency on earlier positions.
    EXPECT_GT(first_token_gradient, 1e-5);
  }
}

TEST_F(SequenceDeltaNetTest,
       GroupedHeadsAndPartialValueTilesDifferentiateCorrectly) {
  auto e = MakeExample(3, 2, 4, 5, 35, true);
  auto layer = MakeLayer(e);
  ASSERT_TRUE(layer.ok());
  auto result = (*layer)->fwd(*executor_, inputs_);
  ASSERT_TRUE(result.ok());
  auto upstream = Upload(e.upstream);
  ASSERT_TRUE(upstream.ok());
  auto gradient =
      (*layer)->bwd(*executor_, {*upstream}, std::move(result->state));
  ASSERT_TRUE(gradient.ok()) << gradient.status();
  std::vector<double> offsets;
  Reference(e, &offsets);
  for (int tensor = 0; tensor < 8; ++tensor) {
    const auto actual = Download<float>(
        tensor < 4 ? (*gradient)[tensor] : parameters_[tensor - 4]->gradient());
    for (size_t i = 0; i < actual.size(); i += 7) {
      const double original = e.values[tensor][i];
      constexpr double epsilon = 1e-5;
      e.values[tensor][i] = original + epsilon;
      const double plus = Loss(e, offsets);
      e.values[tensor][i] = original - epsilon;
      const double minus = Loss(e, offsets);
      e.values[tensor][i] = original;
      const double expected = (plus - minus) / (2 * epsilon);
      EXPECT_NEAR(actual[i], expected, 3e-5 + std::abs(expected) * 0.003)
          << "tensor=" << tensor << " index=" << i;
    }
  }
}

TEST_F(SequenceDeltaNetTest,
       FreezingKeepsInputGradientsAndAccumulationIsAdditive) {
  const auto e = MakeExample(4, 1, 2, 3, 5, true);
  auto layer = MakeLayer(e);
  ASSERT_TRUE(layer.ok());
  auto upstream = Upload(e.upstream);
  ASSERT_TRUE(upstream.ok());
  auto fwd = (*layer)->fwd(*executor_, inputs_);
  ASSERT_TRUE(fwd.ok());
  auto first = (*layer)->bwd(*executor_, {*upstream}, fwd->state);
  ASSERT_TRUE(first.ok()) << first.status();
  std::vector<std::vector<float>> parameter_gradients;
  for (auto& parameter : parameters_)
    parameter_gradients.push_back(Download<float>(parameter->gradient()));
  auto second = (*layer)->bwd(*executor_, {*upstream}, fwd->state);
  ASSERT_TRUE(second.ok());
  for (int p = 0; p < 4; ++p) {
    const auto twice = Download<float>(parameters_[p]->gradient());
    for (size_t i = 0; i < twice.size(); ++i)
      EXPECT_FLOAT_EQ(twice[i], 2 * parameter_gradients[p][i]);
    ASSERT_TRUE(parameters_[p]->Deactivate().ok());
  }
  auto frozen = (*layer)->bwd(*executor_, {*upstream}, std::move(fwd->state));
  ASSERT_TRUE(frozen.ok());
  for (int input = 0; input < 4; ++input)
    EXPECT_EQ(Download<float>((*first)[input]),
              Download<float>((*frozen)[input]));
  for (auto& parameter : parameters_)
    EXPECT_FALSE(parameter->active());
}

TEST_F(SequenceDeltaNetTest, RejectsBadShapesAndPreservesTypedSignatures) {
  auto e = MakeExample(3, 1, 2, 3, 5, true);
  auto layer = MakeLayer(e);
  ASSERT_TRUE(layer.ok());
  EXPECT_EQ((*layer)->input_types()[0],
            ActivationType(DataType::BF16, {-2, 3, 16}));
  EXPECT_EQ((*layer)->output_types()[0],
            ActivationType(DataType::BF16, {-2, 3, 10}));
  EXPECT_TRUE((*layer)->gradients().empty());
  EXPECT_EQ((*layer)->weights().size(), 4);
  EXPECT_TRUE(absl::IsInvalidArgument((*layer)->fwd(*executor_, {}).status()));
  auto doubled = Buffer::Allocate(*executor_, inputs_[0].size_bytes() * 2);
  ASSERT_TRUE(doubled.ok());
  auto bad = inputs_;
  bad[0] = *doubled;
  EXPECT_TRUE(absl::IsInvalidArgument((*layer)->fwd(*executor_, bad).status()));
  for (int length : {0, 129})
    EXPECT_FALSE(SequenceDeltaNetLayer::Create(*executor_, e.p, parameters_[0],
                                               parameters_[1], parameters_[2],
                                               parameters_[3], length)
                     .ok());
  e.p.key_heads = 3;
  EXPECT_FALSE(SequenceDeltaNetLayer::Create(*executor_, e.p, parameters_[0],
                                             parameters_[1], parameters_[2],
                                             parameters_[3], 3)
                   .ok());
}

}  // namespace
}  // namespace pluto::llm
