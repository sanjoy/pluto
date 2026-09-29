#include "src/llm/layers/qwen_attention.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

#include "gtest/gtest.h"
#include "src/cuda/page_locked_host_array.h"
#include "src/llm/key_value_cache.h"
#include "src/llm/layer_hooks.h"
#include "src/util/status_macros.h"

namespace pluto::llm {
namespace {

float Bf16(float x) { return static_cast<float>(__nv_bfloat16(x)); }

// Deliberately scalar double-precision math. To check mixed-precision STE,
// record the rounding error at every cast, then hold those errors constant
// while perturbing inputs. Thus casts have derivative one without pretending
// that finite differences of the discontinuous BF16 quantizer are gradients.
std::vector<double> Reference(
    const FullAttentionParameters& p, int length, const std::vector<double>& qg,
    const std::vector<double>& k, const std::vector<double>& v,
    const std::vector<double>& qw, const std::vector<double>& kw,
    std::vector<double>& rounding_errors, bool record_rounding = false) {
  const int d = p.head_dim, qh = p.query_heads, kh = p.key_value_heads;
  size_t rounding_index = 0;
  const auto round = [&](double value) {
    if (!p.round_to_bfloat16)
      return value;
    if (record_rounding)
      rounding_errors.push_back(Bf16(value) - value);
    return value + rounding_errors[rounding_index++];
  };
  const auto prepare = [&](const std::vector<double>& x,
                           const std::vector<double>& w, int heads,
                           int stride) {
    std::vector<double> result(length * heads * d);
    for (int row = 0; row < length; ++row)
      for (int h = 0; h < heads; ++h) {
        int base = (row * heads + h) * stride;
        int out = (row * heads + h) * d;
        double ss = 0;
        for (int c = 0; c < d; ++c)
          ss += x[base + c] * x[base + c];
        double inverse = 1 / std::sqrt(ss / d + p.rms_norm_epsilon);
        for (int c = 0; c < d; ++c)
          result[out + c] = round(x[base + c] * inverse * (1 + w[c]));
        for (int c = 0; c < p.rotary_dim / 2; ++c) {
          int partner = c + p.rotary_dim / 2;
          double a = row * std::pow(p.rope_theta, -2. * c / p.rotary_dim);
          double first = result[out + c], second = result[out + partner];
          double cosine = round(std::cos(a)), sine = round(std::sin(a));
          double fc = round(first * cosine), ss = round(second * sine);
          double sc = round(second * cosine), fs = round(first * sine);
          result[out + c] = round(fc - ss);
          result[out + partner] = round(sc + fs);
        }
      }
    return result;
  };
  auto query = prepare(qg, qw, qh, 2 * d);
  auto key = prepare(k, kw, kh, d);
  std::vector<double> output(length * qh * d);
  for (int row = 0; row < length; ++row)
    for (int h = 0; h < qh; ++h) {
      int kv = h / (qh / kh);
      std::vector<double> prob(row + 1);
      for (int t = 0; t <= row; ++t)
        for (int c = 0; c < d; ++c)
          prob[t] += query[(row * qh + h) * d + c] *
                     key[(t * kh + kv) * d + c] / std::sqrt(d);
      double maximum = *std::max_element(prob.begin(), prob.end());
      double sum = 0;
      for (double& value : prob) {
        value = std::exp(value - maximum);
        sum += value;
      }
      for (int c = 0; c < d; ++c) {
        double core = 0;
        for (int t = 0; t <= row; ++t)
          core += prob[t] / sum * v[(t * kh + kv) * d + c];
        double rounded_core = round(core);
        double gate =
            round(1 / (1 + std::exp(-qg[((row * qh + h) * 2 + 1) * d + c])));
        output[(row * qh + h) * d + c] = round(rounded_core * gate);
      }
    }
  return output;
}

std::vector<double> Values(int n, double phase) {
  std::vector<double> result(n);
  for (int i = 0; i < n; ++i)
    result[i] = Bf16(0.8 * std::sin(phase + 0.29 * i));
  return result;
}

class QwenAttentionTest : public testing::Test {
 protected:
  void SetUp() override {
    auto result = cuda::Executor::Create();
    ASSERT_TRUE(result.ok()) << result.status();
    executor_ = std::move(*result);
  }
  void TearDown() override { EXPECT_TRUE(executor_->Synchronize().ok()); }

  template <class T>
  absl::StatusOr<Buffer> Upload(const std::vector<double>& values) {
    ASSIGN_OR_RETURN(auto host, cuda::PageLockedHostArray<T>::Allocate(
                                    *executor_, values.size()));
    for (size_t i = 0; i < values.size(); ++i)
      host[i] = T(static_cast<float>(values[i]));
    ASSIGN_OR_RETURN(auto buffer,
                     Buffer::Allocate(*executor_, values.size() * sizeof(T)));
    RETURN_IF_ERROR(cuda::CudaStatus(
        cudaMemcpyAsync(buffer.data(), host.data(), buffer.size_bytes(),
                        cudaMemcpyHostToDevice, executor_->stream()),
        "upload sequence attention test"));
    RETURN_IF_ERROR(executor_->Synchronize());
    return buffer;
  }

  template <class T>
  std::vector<float> Read(const Buffer& buffer) {
    auto host = cuda::PageLockedHostArray<T>::Allocate(
        *executor_, buffer.size_bytes() / sizeof(T));
    EXPECT_TRUE(host.ok());
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

  absl::StatusOr<std::shared_ptr<BlockParameter>> Parameter(
      const std::vector<double>& values) {
    ASSIGN_OR_RETURN(auto data, Upload<float>(values));
    return BlockParameter::Create(*executor_, data, DataType::FP32);
  }

  std::unique_ptr<cuda::Executor> executor_;
};

TEST_F(QwenAttentionTest, ForwardMatchesCachedAttentionAndCausalHook) {
  // Includes partial/non-power-of-two heads, grouped heads, real Qwen width,
  // and a prefix spanning two 32-token online-softmax tiles.
  for (int d : {3, 256}) {
    FullAttentionParameters p;
    p.query_heads = 2;
    p.key_value_heads = 1;
    p.head_dim = d;
    p.rotary_dim = d == 3 ? 2 : 64;
    p.capacity = 40;
    int length = d == 3 ? 33 : 3;
    auto qg = Values(length * 2 * p.query_heads * d, .1);
    auto k = Values(length * d, 1.2), v = Values(length * d, 2.3);
    auto qw = Parameter(Values(d, .2)), kw = Parameter(Values(d, .4));
    ASSERT_TRUE(qw.ok());
    ASSERT_TRUE(kw.ok());
    auto layer = QwenAttentionLayer::Create(*executor_, p, *qw, *kw, length);
    ASSERT_TRUE(layer.ok()) << layer.status();
    auto qgb = Upload<__nv_bfloat16>(qg), kb = Upload<__nv_bfloat16>(k),
         vb = Upload<__nv_bfloat16>(v);
    ASSERT_TRUE(qgb.ok());
    ASSERT_TRUE(kb.ok());
    ASSERT_TRUE(vb.ok());
    LayerHooks hooks;
    int hook_calls = 0;
    hooks.attention_probabilities_hook = [&](cuda::Executor&, absl::string_view,
                                             const ActivationType& type,
                                             const Buffer& buffer) {
      EXPECT_EQ(type, ActivationType(DataType::FP32, {1, 2, length, length}));
      auto probs = Read<float>(buffer);
      for (int h = 0; h < 2; ++h)
        for (int row = 0; row < length; ++row) {
          double sum = 0;
          for (int t = 0; t < length; ++t) {
            float value = probs[(h * length + row) * length + t];
            if (t > row)
              EXPECT_EQ(value, 0);
            sum += value;
          }
          EXPECT_NEAR(sum, 1, 2e-6);
        }
      ++hook_calls;
      return absl::OkStatus();
    };
    auto fwd = (*layer)->fwd(*executor_, {*qgb, *kb, *vb}, &hooks);
    ASSERT_TRUE(fwd.ok()) << fwd.status();
    EXPECT_EQ(hook_calls, 1);
    auto actual = Read<__nv_bfloat16>(fwd->outputs[0]);
    auto cache = KeyValueCache::Create(*executor_, p.capacity,
                                       p.key_value_heads, p.head_dim);
    ASSERT_TRUE(cache.ok());
    auto cached_layer =
        QwenAttentionLayer::Create(*executor_, p, *qw, *kw, 1, cache->get());
    ASSERT_TRUE(cached_layer.ok()) << cached_layer.status();
    for (int row = 0; row < length; ++row) {
      auto qi = Upload<__nv_bfloat16>(
          {qg.begin() + row * 4 * d, qg.begin() + (row + 1) * 4 * d});
      auto ki = Upload<__nv_bfloat16>(
          {k.begin() + row * d, k.begin() + (row + 1) * d});
      auto vi = Upload<__nv_bfloat16>(
          {v.begin() + row * d, v.begin() + (row + 1) * d});
      ASSERT_TRUE(qi.ok());
      ASSERT_TRUE(ki.ok());
      ASSERT_TRUE(vi.ok());
      auto output = (*cached_layer)->fwd(*executor_, {*qi, *ki, *vi});
      ASSERT_TRUE(output.ok());
      EXPECT_EQ((*cache)->position(), row + 1);
      auto expected = Read<__nv_bfloat16>(output->outputs[0]);
      for (int c = 0; c < 2 * d; ++c)
        EXPECT_EQ(actual[row * 2 * d + c], expected[c])
            << "d=" << d << " row=" << row << " c=" << c;
    }
  }
}

TEST_F(QwenAttentionTest, FullBackwardMatchesIndependentFiniteDifference) {
  for (bool use_bf16_rounding : {false, true}) {
    SCOPED_TRACE(use_bf16_rounding);
    FullAttentionParameters p;
    p.query_heads = 2;
    p.key_value_heads = 1;
    p.head_dim = 3;
    p.rotary_dim = 2;
    p.round_to_bfloat16 = use_bf16_rounding;
    constexpr int t = 3;
    std::vector<std::vector<double>> values = {
        Values(t * 12, .1), Values(t * 3, 1.2), Values(t * 3, 2.3),
        Values(3, .2), Values(3, .4)};
    auto qw = Parameter(values[3]), kw = Parameter(values[4]);
    ASSERT_TRUE(qw.ok());
    ASSERT_TRUE(kw.ok());
    ASSERT_TRUE((*qw)->Activate().ok());
    ASSERT_TRUE((*kw)->Activate().ok());
    auto layer = QwenAttentionLayer::Create(*executor_, p, *qw, *kw, t);
    ASSERT_TRUE(layer.ok());
    auto qgb = Upload<__nv_bfloat16>(values[0]),
         kb = Upload<__nv_bfloat16>(values[1]),
         vb = Upload<__nv_bfloat16>(values[2]);
    ASSERT_TRUE(qgb.ok());
    ASSERT_TRUE(kb.ok());
    ASSERT_TRUE(vb.ok());
    auto fwd = (*layer)->fwd(*executor_, {*qgb, *kb, *vb});
    ASSERT_TRUE(fwd.ok());
    auto upstream = Values(t * 6, .7);
    auto dy = Upload<float>(upstream);
    ASSERT_TRUE(dy.ok());
    auto backward = (*layer)->bwd(*executor_, {*dy}, std::move(fwd->state));
    ASSERT_TRUE(backward.ok()) << backward.status();
    std::vector<std::vector<float>> actual;
    for (const auto& b : *backward)
      actual.push_back(Read<float>(b));
    actual.push_back(Read<float>((*qw)->gradient()));
    actual.push_back(Read<float>((*kw)->gradient()));
    std::vector<double> rounding_errors;
    Reference(p, t, values[0], values[1], values[2], values[3], values[4],
              rounding_errors, true);
    const auto loss = [&] {
      auto output = Reference(p, t, values[0], values[1], values[2], values[3],
                              values[4], rounding_errors);
      double sum = 0;
      for (size_t i = 0; i < output.size(); ++i)
        sum += output[i] * upstream[i];
      return sum;
    };
    for (size_t input = 0; input < values.size(); ++input)
      for (size_t i = 0; i < values[input].size(); ++i) {
        constexpr double epsilon = 1e-5;
        double original = values[input][i];
        values[input][i] = original + epsilon;
        double plus = loss();
        values[input][i] = original - epsilon;
        double minus = loss();
        values[input][i] = original;
        EXPECT_NEAR(actual[input][i], (plus - minus) / (2 * epsilon), 3e-5)
            << "input=" << input << " coordinate=" << i;
      }
    // Shared norm gradients accumulate across backward calls, not overwrite.
    auto second = (*layer)->fwd(*executor_, {*qgb, *kb, *vb});
    ASSERT_TRUE(second.ok());
    auto repeated = (*layer)->bwd(*executor_, {*dy}, std::move(second->state));
    ASSERT_TRUE(repeated.ok());
    for (size_t i = 0; i < repeated->size(); ++i)
      EXPECT_EQ(Read<float>((*repeated)[i]), actual[i]);
    auto summed = Read<float>((*qw)->gradient());
    for (size_t i = 0; i < summed.size(); ++i)
      EXPECT_NEAR(summed[i], 2 * actual[3][i], 1e-6);
  }
}

TEST_F(QwenAttentionTest, FrozenNormsStillPropagateToEarlierTokens) {
  FullAttentionParameters p;
  p.query_heads = 2;
  p.key_value_heads = 1;
  p.head_dim = 4;
  p.rotary_dim = 2;
  auto qw = Parameter(Values(4, .2)), kw = Parameter(Values(4, .4));
  ASSERT_TRUE(qw.ok());
  ASSERT_TRUE(kw.ok());
  auto layer = QwenAttentionLayer::Create(*executor_, p, *qw, *kw, 3);
  ASSERT_TRUE(layer.ok());
  auto q = Upload<__nv_bfloat16>(Values(48, .1)),
       k = Upload<__nv_bfloat16>(Values(12, 1.2)),
       v = Upload<__nv_bfloat16>(Values(12, 2.3));
  ASSERT_TRUE(q.ok());
  ASSERT_TRUE(k.ok());
  ASSERT_TRUE(v.ok());
  auto result = (*layer)->fwd(*executor_, {*q, *k, *v});
  ASSERT_TRUE(result.ok());
  std::vector<double> grad(24, 0);
  for (int i = 16; i < 24; ++i)
    grad[i] = .5;
  auto dy = Upload<float>(grad);
  ASSERT_TRUE(dy.ok());
  auto backward = (*layer)->bwd(*executor_, {*dy}, std::move(result->state));
  ASSERT_TRUE(backward.ok());
  auto dk = Read<float>((*backward)[1]), dv = Read<float>((*backward)[2]);
  float key_magnitude = 0, value_magnitude = 0;
  for (int i = 0; i < 4; ++i) {
    key_magnitude += std::abs(dk[i]);
    value_magnitude += std::abs(dv[i]);
  }
  EXPECT_GT(key_magnitude, 1e-5);
  EXPECT_GT(value_magnitude, 1e-5);
  EXPECT_FALSE((*qw)->active());
  EXPECT_FALSE((*kw)->active());
  EXPECT_FALSE((*layer)->fwd(*executor_, {*q, *k}).ok());
  EXPECT_FALSE(QwenAttentionLayer::Create(*executor_, p, *qw, *kw, 129).ok());
  auto other = cuda::Executor::Create();
  ASSERT_TRUE(other.ok());
  EXPECT_FALSE((*layer)->fwd(**other, {*q, *k, *v}).ok());
}

TEST_F(QwenAttentionTest, CachePointerSelectsBackwardContractAtLengthOne) {
  FullAttentionParameters p;
  p.query_heads = 2;
  p.key_value_heads = 1;
  p.head_dim = 4;
  p.rotary_dim = 2;
  p.capacity = 2;
  auto qw = Parameter(Values(4, .2)), kw = Parameter(Values(4, .4));
  ASSERT_TRUE(qw.ok());
  ASSERT_TRUE(kw.ok());
  ASSERT_TRUE((*qw)->Activate().ok());
  ASSERT_TRUE((*kw)->Activate().ok());
  auto cache = KeyValueCache::Create(*executor_, p.capacity, p.key_value_heads,
                                     p.head_dim);
  ASSERT_TRUE(cache.ok());
  auto sequence = QwenAttentionLayer::Create(*executor_, p, *qw, *kw, 1);
  auto cached =
      QwenAttentionLayer::Create(*executor_, p, *qw, *kw, 1, cache->get());
  ASSERT_TRUE(sequence.ok()) << sequence.status();
  ASSERT_TRUE(cached.ok()) << cached.status();
  EXPECT_EQ((*sequence)->input_types()[0], (*cached)->input_types()[0]);
  EXPECT_EQ((*sequence)->output_types()[0], (*cached)->output_types()[0]);
  auto q = Upload<__nv_bfloat16>(Values(16, .1));
  auto k = Upload<__nv_bfloat16>(Values(4, 1.2));
  auto v = Upload<__nv_bfloat16>(Values(4, 2.3));
  auto dy = Upload<float>(Values(8, .7));
  ASSERT_TRUE(q.ok());
  ASSERT_TRUE(k.ok());
  ASSERT_TRUE(v.ok());
  ASSERT_TRUE(dy.ok());
  auto sf = (*sequence)->fwd(*executor_, {*q, *k, *v});
  auto cf = (*cached)->fwd(*executor_, {*q, *k, *v});
  ASSERT_TRUE(sf.ok()) << sf.status();
  ASSERT_TRUE(cf.ok()) << cf.status();
  EXPECT_EQ(Read<__nv_bfloat16>(sf->outputs[0]),
            Read<__nv_bfloat16>(cf->outputs[0]));
  EXPECT_EQ((*cache)->position(), 1);

  // Active BlockParameters do not turn cached decoding into training: the
  // explicit cache pointer alone selects the forward/backward contract.
  EXPECT_EQ(
      (*cached)->bwd(*executor_, {*dy}, std::move(cf->state)).status().code(),
      absl::StatusCode::kUnimplemented);
  for (float gradient : Read<float>((*qw)->gradient()))
    EXPECT_EQ(gradient, 0);
  for (float gradient : Read<float>((*kw)->gradient()))
    EXPECT_EQ(gradient, 0);
  auto backward = (*sequence)->bwd(*executor_, {*dy}, std::move(sf->state));
  ASSERT_TRUE(backward.ok()) << backward.status();
  ASSERT_EQ(backward->size(), 3);
  float value_magnitude = 0;
  for (float gradient : Read<float>((*backward)[2])) {
    EXPECT_TRUE(std::isfinite(gradient));
    value_magnitude += std::abs(gradient);
  }
  EXPECT_GT(value_magnitude, 0);
  EXPECT_EQ((*cache)->position(), 1);
}

TEST_F(QwenAttentionTest,
       NullCacheWithRawNormsRemainsStatelessAndDifferentiable) {
  FullAttentionParameters p;
  p.query_heads = 2;
  p.key_value_heads = 1;
  p.head_dim = 4;
  p.rotary_dim = 2;
  p.capacity = 0;  // A sequence-only layer has no cache capacity to validate.
  auto qw = Parameter(Values(4, .2)), kw = Parameter(Values(4, .4));
  ASSERT_TRUE(qw.ok());
  ASSERT_TRUE(kw.ok());
  auto reference = QwenAttentionLayer::Create(*executor_, p, *qw, *kw, 3);
  auto raw = QwenAttentionLayer::Create(*executor_, p, (*qw)->value(),
                                        (*kw)->value(), 3, nullptr);
  ASSERT_TRUE(reference.ok()) << reference.status();
  ASSERT_TRUE(raw.ok()) << raw.status();
  EXPECT_EQ((*raw)->weights()[0].data(), (*qw)->value().data());
  EXPECT_EQ((*raw)->weights()[1].data(), (*kw)->value().data());
  EXPECT_TRUE((*raw)->gradients().empty());
  auto q = Upload<__nv_bfloat16>(Values(48, .1));
  auto k = Upload<__nv_bfloat16>(Values(12, 1.2));
  auto v = Upload<__nv_bfloat16>(Values(12, 2.3));
  auto dy = Upload<float>(Values(24, .7));
  ASSERT_TRUE(q.ok());
  ASSERT_TRUE(k.ok());
  ASSERT_TRUE(v.ok());
  ASSERT_TRUE(dy.ok());
  auto expected = (*reference)->fwd(*executor_, {*q, *k, *v});
  ASSERT_TRUE(expected.ok());
  auto expected_gradients =
      (*reference)->bwd(*executor_, {*dy}, std::move(expected->state));
  ASSERT_TRUE(expected_gradients.ok());
  // Repeated sequence forwards restart at position zero, unlike decoding.
  for (int repetition = 0; repetition < 2; ++repetition) {
    auto actual = (*raw)->fwd(*executor_, {*q, *k, *v});
    ASSERT_TRUE(actual.ok()) << actual.status();
    EXPECT_EQ(Read<__nv_bfloat16>(actual->outputs[0]),
              Read<__nv_bfloat16>(expected->outputs[0]));
    auto gradients = (*raw)->bwd(*executor_, {*dy}, std::move(actual->state));
    ASSERT_TRUE(gradients.ok()) << gradients.status();
    ASSERT_EQ(gradients->size(), 3);
    for (size_t i = 0; i < gradients->size(); ++i)
      EXPECT_EQ(Read<float>((*gradients)[i]),
                Read<float>((*expected_gradients)[i]));
  }
}

TEST_F(QwenAttentionTest, EitherNormMayBeActiveWhileTheOtherRemainsFrozen) {
  for (bool train_query : {false, true}) {
    SCOPED_TRACE(train_query);
    FullAttentionParameters p;
    p.query_heads = 2;
    p.key_value_heads = 1;
    p.head_dim = 4;
    p.rotary_dim = 2;
    auto qw = Parameter(Values(4, .2)), kw = Parameter(Values(4, .4));
    ASSERT_TRUE(qw.ok());
    ASSERT_TRUE(kw.ok());
    const auto& active = train_query ? *qw : *kw;
    const auto& frozen = train_query ? *kw : *qw;
    ASSERT_TRUE(active->Activate().ok());
    auto layer = QwenAttentionLayer::Create(*executor_, p, *qw, *kw, 3);
    ASSERT_TRUE(layer.ok());
    auto q = Upload<__nv_bfloat16>(Values(48, .1));
    auto k = Upload<__nv_bfloat16>(Values(12, 1.2));
    auto v = Upload<__nv_bfloat16>(Values(12, 2.3));
    auto dy = Upload<float>(Values(24, .7));
    ASSERT_TRUE(q.ok());
    ASSERT_TRUE(k.ok());
    ASSERT_TRUE(v.ok());
    ASSERT_TRUE(dy.ok());
    auto forward = (*layer)->fwd(*executor_, {*q, *k, *v});
    ASSERT_TRUE(forward.ok());
    auto backward = (*layer)->bwd(*executor_, {*dy}, std::move(forward->state));
    ASSERT_TRUE(backward.ok()) << backward.status();
    ASSERT_EQ(backward->size(), 3);
    for (const Buffer& gradient : *backward)
      for (float value : Read<float>(gradient))
        EXPECT_TRUE(std::isfinite(value));
    float magnitude = 0;
    for (float value : Read<float>(active->gradient())) {
      EXPECT_TRUE(std::isfinite(value));
      magnitude += std::abs(value);
    }
    EXPECT_GT(magnitude, 1e-5);
    EXPECT_FALSE(frozen->active());
    EXPECT_TRUE((*layer)->gradients().empty());
  }
}

}  // namespace
}  // namespace pluto::llm
