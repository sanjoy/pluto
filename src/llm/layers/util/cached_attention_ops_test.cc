#include "src/llm/layers/util/cached_attention_ops.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "gtest/gtest.h"
#include "src/cuda/buffer.h"
#include "src/cuda/executor.h"
#include "src/cuda/page_locked_host_array.h"
#include "src/util/status_macros.h"

namespace pluto::llm::cached_attention_ops {
namespace {

float RoundBfloat16(float value) {
  uint32_t bits = std::bit_cast<uint32_t>(value);
  bits += 0x7fff + ((bits >> 16) & 1);
  return std::bit_cast<float>(bits & 0xffff0000);
}

float Sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }
float Silu(float x) { return x * Sigmoid(x); }

std::vector<float> Values(int count, float phase, float scale = 1.0f) {
  std::vector<float> result(count);
  for (int i = 0; i < count; ++i)
    result[i] = scale * std::sin(phase + i * 0.29f);
  return result;
}

// A scalar specification with explicit per-head vectors and complete softmax,
// independent of the GPU's padded tiles and online-softmax decomposition.
class AttentionReference {
 public:
  explicit AttentionReference(FullAttentionParameters p) : p_(p) {}

  std::vector<float> Step(const std::vector<float>& q_gate,
                          const std::vector<float>& key,
                          const std::vector<float>& value,
                          const std::vector<float>& q_norm,
                          const std::vector<float>& k_norm) {
    const int position = static_cast<int>(keys_.size());
    std::vector<float> queries(p_.query_heads * p_.head_dim);
    std::vector<float> keys(p_.key_value_heads * p_.head_dim);
    const auto normalize_rotate = [&](const float* input,
                                      const std::vector<float>& weight,
                                      float* output) {
      double square = 0.0;
      for (int d = 0; d < p_.head_dim; ++d)
        square += input[d] * input[d];
      const double inverse =
          1.0 / std::sqrt(square / p_.head_dim + p_.rms_norm_epsilon);
      for (int d = 0; d < p_.head_dim; ++d)
        output[d] =
            Round(static_cast<float>(input[d] * inverse * (1.0 + weight[d])));
      for (int d = 0; d < p_.rotary_dim / 2; ++d) {
        const int partner = d + p_.rotary_dim / 2;
        const double angle =
            position * std::pow(p_.rope_theta, -2.0 * d / p_.rotary_dim);
        const float first = output[d];
        const float second = output[partner];
        const float cosine = Round(std::cos(angle));
        const float sine = Round(std::sin(angle));
        output[d] = Round(Round(first * cosine) - Round(second * sine));
        output[partner] = Round(Round(second * cosine) + Round(first * sine));
      }
    };
    for (int h = 0; h < p_.query_heads; ++h)
      normalize_rotate(q_gate.data() + h * 2 * p_.head_dim, q_norm,
                       queries.data() + h * p_.head_dim);
    for (int h = 0; h < p_.key_value_heads; ++h)
      normalize_rotate(key.data() + h * p_.head_dim, k_norm,
                       keys.data() + h * p_.head_dim);
    keys_.push_back(std::move(keys));
    values_.push_back(value);
    for (float& x : values_.back())
      x = Round(x);
    std::vector<float> output(queries.size());
    probabilities_.resize(p_.query_heads * keys_.size());
    for (int h = 0; h < p_.query_heads; ++h) {
      const int kv = h / (p_.query_heads / p_.key_value_heads);
      std::vector<double> probabilities(keys_.size());
      for (size_t t = 0; t < keys_.size(); ++t) {
        double score = 0;
        for (int d = 0; d < p_.head_dim; ++d)
          score +=
              queries[h * p_.head_dim + d] * keys_[t][kv * p_.head_dim + d];
        probabilities[t] = score / std::sqrt(p_.head_dim);
      }
      const double maximum =
          *std::max_element(probabilities.begin(), probabilities.end());
      double normalizer = 0;
      for (double& probability : probabilities) {
        probability = std::exp(probability - maximum);
        normalizer += probability;
      }
      for (size_t t = 0; t < probabilities.size(); ++t)
        probabilities_[h * keys_.size() + t] = probabilities[t] / normalizer;
      for (int d = 0; d < p_.head_dim; ++d) {
        double sum = 0;
        for (size_t t = 0; t < probabilities.size(); ++t)
          sum +=
              probabilities[t] * values_[t][kv * p_.head_dim + d] / normalizer;
        output[h * p_.head_dim + d] = Round(
            Round(sum) * Round(Sigmoid(q_gate[(2 * h + 1) * p_.head_dim + d])));
      }
    }
    return output;
  }

  const std::vector<float>& probabilities() const { return probabilities_; }

 private:
  float Round(float x) const {
    return p_.round_to_bfloat16 ? RoundBfloat16(x) : x;
  }
  FullAttentionParameters p_;
  std::vector<std::vector<float>> keys_;
  std::vector<std::vector<float>> values_;
  std::vector<float> probabilities_;
};

class DeltaReference {
 public:
  explicit DeltaReference(DeltaNetParameters p)
      : p_(p),
        convolution_((2 * p.key_heads * p.key_head_dim +
                      p.value_heads * p.value_head_dim) *
                     p.conv_kernel_dim),
        state_(p.value_heads * p.key_head_dim * p.value_head_dim) {}

  std::vector<float> Step(
      const std::vector<float>& qkv, const std::vector<float>& z,
      const std::vector<float>& a, const std::vector<float>& b,
      const std::vector<float>& weights, const std::vector<float>& a_log,
      const std::vector<float>& dt_bias, const std::vector<float>& norm) {
    std::vector<float> convolved(qkv.size());
    for (size_t c = 0; c < qkv.size(); ++c) {
      for (int tap = 0; tap < p_.conv_kernel_dim - 1; ++tap)
        convolution_[c * p_.conv_kernel_dim + tap] =
            convolution_[c * p_.conv_kernel_dim + tap + 1];
      convolution_[(c + 1) * p_.conv_kernel_dim - 1] = Round(qkv[c]);
      double sum = 0;
      for (int tap = 0; tap < p_.conv_kernel_dim; ++tap)
        sum += convolution_[c * p_.conv_kernel_dim + tap] *
               weights[c * p_.conv_kernel_dim + tap];
      convolved[c] = Round(Silu(Round(sum)));
    }
    std::vector<float> output(p_.value_heads * p_.value_head_dim);
    for (int h = 0; h < p_.value_heads; ++h) {
      const int kh = h / (p_.value_heads / p_.key_heads);
      std::vector<double> q(p_.key_head_dim), k(p_.key_head_dim);
      double q_square = 1e-6, k_square = 1e-6;
      for (int d = 0; d < p_.key_head_dim; ++d) {
        q[d] = convolved[kh * p_.key_head_dim + d];
        k[d] = convolved[(p_.key_heads + kh) * p_.key_head_dim + d];
        q_square += q[d] * q[d];
        k_square += k[d] * k[d];
      }
      for (int d = 0; d < p_.key_head_dim; ++d) {
        q[d] /= std::sqrt(q_square * p_.key_head_dim);
        k[d] /= std::sqrt(k_square);
      }
      const double alpha = a[h] + dt_bias[h];
      const double softplus =
          std::max(alpha, 0.0) + std::log1p(std::exp(-std::abs(alpha)));
      const double decay = std::exp(-std::exp(a_log[h]) * softplus);
      for (int v = 0; v < p_.value_head_dim; ++v) {
        double prediction = 0;
        for (int d = 0; d < p_.key_head_dim; ++d) {
          const int index = (h * p_.key_head_dim + d) * p_.value_head_dim + v;
          state_[index] *= decay;
          prediction += state_[index] * k[d];
        }
        const float value = convolved[2 * p_.key_heads * p_.key_head_dim +
                                      h * p_.value_head_dim + v];
        const double delta = (value - prediction) * Round(Sigmoid(b[h]));
        double core = 0;
        for (int d = 0; d < p_.key_head_dim; ++d) {
          const int index = (h * p_.key_head_dim + d) * p_.value_head_dim + v;
          state_[index] += k[d] * delta;
          core += state_[index] * q[d];
        }
        output[h * p_.value_head_dim + v] = Round(core);
      }
      double variance = 0;
      for (int v = 0; v < p_.value_head_dim; ++v) {
        const double x = output[h * p_.value_head_dim + v];
        variance += x * x;
      }
      const double inverse =
          1.0 / std::sqrt(variance / p_.value_head_dim + p_.rms_norm_epsilon);
      for (int v = 0; v < p_.value_head_dim; ++v) {
        const int index = h * p_.value_head_dim + v;
        const float normalized = Round(output[index] * inverse);
        const float affine = Round(normalized * norm[v]);
        output[index] = Round(affine * Silu(z[index]));
      }
    }
    return output;
  }

 private:
  float Round(float x) const {
    return p_.round_to_bfloat16 ? RoundBfloat16(x) : x;
  }
  DeltaNetParameters p_;
  std::vector<float> convolution_;
  std::vector<double> state_;
};

class AttentionOpsTest : public testing::Test {
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

  absl::StatusOr<cuda::Buffer> Upload(const std::vector<float>& input) {
    ASSIGN_OR_RETURN(auto staging, cuda::PageLockedHostArray<float>::CopyFrom(
                                       *executor_, input));
    ASSIGN_OR_RETURN(auto buffer, cuda::Buffer::Allocate(
                                      *executor_, input.size() * sizeof(float)));
    RETURN_IF_ERROR(cuda::CudaStatus(
        cudaMemcpyAsync(buffer.data(), staging.data(), buffer.size_bytes(),
                        cudaMemcpyHostToDevice, executor_->stream()),
        "upload attention test input"));
    RETURN_IF_ERROR(executor_->Synchronize());
    return buffer;
  }

  std::vector<float> Read(const cuda::Buffer& buffer) {
    auto staging = cuda::PageLockedHostArray<float>::Allocate(
        *executor_, buffer.size_bytes() / sizeof(float));
    EXPECT_TRUE(staging.ok()) << staging.status();
    if (!staging.ok())
      return {};
    const auto copied =
        cudaMemcpyAsync(staging->data(), buffer.data(), buffer.size_bytes(),
                        cudaMemcpyDeviceToHost, executor_->stream());
    EXPECT_EQ(copied, cudaSuccess);
    if (copied != cudaSuccess)
      return {};
    const auto synchronized = executor_->Synchronize();
    EXPECT_TRUE(synchronized.ok()) << synchronized;
    if (!synchronized.ok())
      return {};
    return std::vector<float>(staging->begin(), staging->end());
  }

  void ExpectNear(const std::vector<float>& actual,
                  const std::vector<float>& expected, bool bfloat16 = false) {
    ASSERT_EQ(actual.size(), expected.size());
    for (size_t i = 0; i < actual.size(); ++i)
      EXPECT_NEAR(actual[i], expected[i],
                  bfloat16 ? 0.0002f + 0.008f * std::abs(expected[i])
                           : 3e-5f + 1e-4f * std::abs(expected[i]))
          << "element " << i;
  }

  static float* Data(cuda::Buffer& buffer) {
    return static_cast<float*>(buffer.data());
  }
  std::unique_ptr<cuda::Executor> executor_;
};

TEST_F(AttentionOpsTest, GqaPartialRopeGatingAndCacheMatchScalarAcrossTiles) {
  for (int dim : {6, 256}) {
    SCOPED_TRACE(dim);
    for (bool round : {false, true}) {
      SCOPED_TRACE(round);
      FullAttentionParameters p;
      p.query_heads = dim == 6 ? 4 : 24;
      p.key_value_heads = dim == 6 ? 2 : 4;
      p.head_dim = dim;
      p.rotary_dim = dim == 6 ? 4 : 64;
      p.capacity = 35;
      p.rope_theta = dim == 6 ? 73.0f : 10000000.0f;
      p.round_to_bfloat16 = round;
      auto cache = KeyValueCache::Create(*executor_, p.capacity,
                                         p.key_value_heads, p.head_dim);
      ASSERT_TRUE(cache.ok()) << cache.status();
      ASSERT_TRUE(ValidateCachedAttention(*executor_, p, **cache).ok());
      auto output = Upload(std::vector<float>(p.query_heads * p.head_dim));
      auto qnorm = Values(p.head_dim, 0.2f, 0.3f);
      auto knorm = Values(p.head_dim, 0.8f, 0.4f);
      auto dqnorm = Upload(qnorm);
      auto dknorm = Upload(knorm);
      ASSERT_TRUE(output.ok());
      ASSERT_TRUE(dqnorm.ok());
      ASSERT_TRUE(dknorm.ok());
      AttentionReference reference(p);
      std::vector<float> first;
      for (int t = 0; t < p.capacity; ++t) {
        SCOPED_TRACE(t);
        auto q = Values(p.query_heads * 2 * p.head_dim, 0.15f + t * 0.21f);
        auto k = Values(p.key_value_heads * p.head_dim, 0.78f + t * 0.13f);
        auto v = Values(p.key_value_heads * p.head_dim, -0.31f + t * 0.37f);
        auto dq = Upload(q);
        auto dk = Upload(k);
        auto dv = Upload(v);
        auto probabilities =
            Upload(std::vector<float>(p.query_heads * (t + 1)));
        ASSERT_TRUE(dq.ok());
        ASSERT_TRUE(dk.ok());
        ASSERT_TRUE(dv.ok());
        ASSERT_TRUE(probabilities.ok());
        ASSERT_TRUE(CachedAttentionStep(*executor_, p, **cache, Data(*dq),
                                        Data(*dk), Data(*dv), Data(*dqnorm),
                                        Data(*dknorm), Data(*output),
                                        Data(*probabilities))
                        .ok());
        const auto actual = Read(*output);
        ExpectNear(actual, reference.Step(q, k, v, qnorm, knorm), round);
        ExpectNear(Read(*probabilities), reference.probabilities(), round);
        if (t == 0)
          first = actual;
        EXPECT_EQ((*cache)->position(), t + 1);
        if (t + 1 == p.capacity) {
          EXPECT_TRUE(ValidateCachedAttention(*executor_, p, **cache).ok());
          EXPECT_EQ(CachedAttentionStep(*executor_, p, **cache, Data(*dq),
                                        Data(*dk), Data(*dv), Data(*dqnorm),
                                        Data(*dknorm), Data(*output))
                        .code(),
                    absl::StatusCode::kResourceExhausted);
          EXPECT_EQ((*cache)->position(), p.capacity);
          EXPECT_EQ(Read(*output), actual);
        }
      }
      (*cache)->Reset();
      EXPECT_EQ((*cache)->position(), 0);
      auto dq = Upload(Values(p.query_heads * 2 * p.head_dim, 0.15f));
      auto dk = Upload(Values(p.key_value_heads * p.head_dim, 0.78f));
      auto dv = Upload(Values(p.key_value_heads * p.head_dim, -0.31f));
      ASSERT_TRUE(dq.ok());
      ASSERT_TRUE(dk.ok());
      ASSERT_TRUE(dv.ok());
      ASSERT_TRUE(CachedAttentionStep(*executor_, p, **cache, Data(*dq),
                                      Data(*dk), Data(*dv), Data(*dqnorm),
                                      Data(*dknorm), Data(*output))
                      .ok());
      EXPECT_EQ((*cache)->position(), 1);
      EXPECT_EQ(Read(*output), first);
    }
  }
}

TEST_F(AttentionOpsTest, FullAttentionBfloat16OutputUsesPerHeadSigmoidGates) {
  FullAttentionParameters p;
  p.query_heads = 2;
  p.key_value_heads = 1;
  p.head_dim = 4;
  p.rotary_dim = 2;
  auto cache = KeyValueCache::Create(*executor_, p.capacity, p.key_value_heads,
                                     p.head_dim);
  ASSERT_TRUE(cache.ok());
  const std::vector<float> q = {1, 2, 3, 4, -2, 0,  1, 3,
                                4, 3, 2, 1, 2,  -1, 0, -3};
  const std::vector<float> value = {0.31f, -0.9f, 1.2f, 2.7f};
  auto dq = Upload(q);
  auto dk = Upload({1, 1, 1, 1});
  auto dv = Upload(value);
  auto norm = Upload({0, 0, 0, 0});
  auto output = Upload(std::vector<float>(8));
  ASSERT_TRUE(dq.ok());
  ASSERT_TRUE(dk.ok());
  ASSERT_TRUE(dv.ok());
  ASSERT_TRUE(norm.ok());
  ASSERT_TRUE(output.ok());
  ASSERT_TRUE(CachedAttentionStep(*executor_, p, **cache, Data(*dq), Data(*dk),
                                  Data(*dv), Data(*norm), Data(*norm),
                                  Data(*output))
                  .ok());
  EXPECT_EQ((*cache)->position(), 1);
  std::vector<float> expected(8);
  for (int h = 0; h < 2; ++h)
    for (int d = 0; d < 4; ++d)
      expected[h * 4 + d] =
          RoundBfloat16(RoundBfloat16(value[d]) *
                        RoundBfloat16(Sigmoid(q[(h * 2 + 1) * 4 + d])));
  EXPECT_EQ(Read(*output), expected);
}

TEST_F(AttentionOpsTest, DeltaNetConvolutionRecurrenceAndResetMatchScalar) {
  for (bool round : {false, true}) {
    SCOPED_TRACE(round);
    for (int dim : {5, 128}) {
      SCOPED_TRACE(dim);
      DeltaNetParameters p;
      p.key_heads = 2;
      p.value_heads = 6;
      p.key_head_dim = dim;
      p.value_head_dim = dim == 5 ? 7 : 128;
      p.conv_kernel_dim = 4;
      p.round_to_bfloat16 = round;
      const int channels =
          2 * p.key_heads * p.key_head_dim + p.value_heads * p.value_head_dim;
      auto state = DeltaNetState::Create(*executor_, p);
      ASSERT_TRUE(state.ok()) << state.status();
      auto weights = Values(channels * p.conv_kernel_dim, 0.73f, 0.6f);
      auto alog = Values(p.value_heads, 0.18f, 0.5f);
      auto dt = Values(p.value_heads, -0.7f, 0.3f);
      auto norm = Values(p.value_head_dim, 0.5f, 0.6f);
      auto dw = Upload(weights);
      auto dalog = Upload(alog);
      auto ddt = Upload(dt);
      auto dn = Upload(norm);
      auto output =
          Upload(std::vector<float>(p.value_heads * p.value_head_dim));
      ASSERT_TRUE(dw.ok());
      ASSERT_TRUE(dalog.ok());
      ASSERT_TRUE(ddt.ok());
      ASSERT_TRUE(dn.ok());
      ASSERT_TRUE(output.ok());
      DeltaReference reference(p);
      std::vector<float> first;
      for (int step = 0; step < 7; ++step) {
        SCOPED_TRACE(step);
        const int t = step == 6 ? 0 : step;
        if (step == 6)
          ASSERT_TRUE((*state)->Reset().ok());
        auto qkv = Values(channels, 0.3f + t * 0.71f);
        auto z = Values(p.value_heads * p.value_head_dim, -0.6f + t * 0.27f);
        auto a = Values(p.value_heads, 0.1f + t * 0.1f);
        auto b = Values(p.value_heads, -0.7f + t * 0.3f);
        // Exercise softplus at both extremes without poisoning the recurrence.
        a[0] = 100.0f;
        a[1] = -100.0f;
        auto dqkv = Upload(qkv);
        auto dz = Upload(z);
        auto da = Upload(a);
        auto db = Upload(b);
        ASSERT_TRUE(dqkv.ok());
        ASSERT_TRUE(dz.ok());
        ASSERT_TRUE(da.ok());
        ASSERT_TRUE(db.ok());
        ASSERT_TRUE((*state)
                        ->Step(Data(*dqkv), Data(*dz), Data(*da), Data(*db),
                               Data(*dw), Data(*dalog), Data(*ddt), Data(*dn),
                               Data(*output))
                        .ok());
        const auto actual = Read(*output);
        if (step < 6)
          ExpectNear(actual,
                     reference.Step(qkv, z, a, b, weights, alog, dt, norm),
                     round);
        if (step == 0)
          first = actual;
        if (step == 6)
          EXPECT_EQ(actual, first);
      }
    }
  }
}

TEST_F(AttentionOpsTest, InspectedProbabilitiesNormalizeAcrossCacheTiles) {
  FullAttentionParameters p;
  p.query_heads = 2;
  p.key_value_heads = 1;
  p.head_dim = 4;
  p.rotary_dim = 2;
  p.capacity = 35;
  auto cache = KeyValueCache::Create(*executor_, p.capacity, p.key_value_heads,
                                     p.head_dim);
  auto q = Upload(std::vector<float>(16));
  auto k = Upload({1, 1, 1, 1});
  auto v = Upload({1, 2, 3, 4});
  auto norm = Upload({0, 0, 0, 0});
  auto output = Upload(std::vector<float>(8));
  ASSERT_TRUE(cache.ok());
  ASSERT_TRUE(q.ok());
  ASSERT_TRUE(k.ok());
  ASSERT_TRUE(v.ok());
  ASSERT_TRUE(norm.ok());
  ASSERT_TRUE(output.ok());
  for (int length = 1; length <= p.capacity; ++length) {
    auto probabilities = Upload(std::vector<float>(2 * length));
    ASSERT_TRUE(probabilities.ok());
    ASSERT_TRUE(CachedAttentionStep(*executor_, p, **cache, Data(*q), Data(*k),
                                    Data(*v), Data(*norm), Data(*norm),
                                    Data(*output), Data(*probabilities))
                    .ok());
    EXPECT_EQ((*cache)->position(), length);
    for (float value : Read(*probabilities))
      EXPECT_NEAR(value, 1.0f / length, 1e-7f);
  }
}

TEST_F(AttentionOpsTest, CachedAttentionRejectsInvalidParametersBeforeWork) {
  const FullAttentionParameters p{.query_heads = 2,
                                  .key_value_heads = 1,
                                  .head_dim = 4,
                                  .rotary_dim = 2,
                                  .capacity = 2};
  auto cache = KeyValueCache::Create(*executor_, p.capacity, p.key_value_heads,
                                     p.head_dim);
  auto output = Upload(std::vector<float>(16, -13.f));
  ASSERT_TRUE(cache.ok());
  ASSERT_TRUE(output.ok());
  const auto check_invalid = [&](const FullAttentionParameters& invalid) {
    EXPECT_TRUE(absl::IsInvalidArgument(
        ValidateCachedAttention(*executor_, invalid, **cache)));
    EXPECT_TRUE(absl::IsInvalidArgument(CachedAttentionStep(
        *executor_, invalid, **cache, Data(*output), Data(*output),
        Data(*output), Data(*output), Data(*output), Data(*output))));
    EXPECT_EQ((*cache)->position(), 0);
  };
  for (int count : {0, -1, std::numeric_limits<int>::max()}) {
    auto invalid = p;
    invalid.query_heads = count;
    check_invalid(invalid);
    invalid = p;
    invalid.key_value_heads = count;
    check_invalid(invalid);
  }
  auto invalid = p;
  invalid.query_heads = 3;
  invalid.key_value_heads = 2;
  check_invalid(invalid);
  for (int dim : {0, -1, 257, std::numeric_limits<int>::max()}) {
    invalid = p;
    invalid.head_dim = dim;
    check_invalid(invalid);
  }
  for (int dim : {0, -1, 3, 6}) {
    invalid = p;
    invalid.rotary_dim = dim;
    check_invalid(invalid);
  }
  for (int capacity : {0, -1, std::numeric_limits<int>::max()}) {
    invalid = p;
    invalid.capacity = capacity;
    check_invalid(invalid);
  }
  for (float value : {0.f, -1.f, std::numeric_limits<float>::infinity(),
                      std::numeric_limits<float>::quiet_NaN()}) {
    invalid = p;
    invalid.rope_theta = value;
    check_invalid(invalid);
    invalid = p;
    invalid.rms_norm_epsilon = value;
    check_invalid(invalid);
  }
  EXPECT_EQ(Read(*output), (std::vector<float>(16, -13.f)));
}

TEST_F(AttentionOpsTest, CachedAttentionRejectsMismatchedCacheBeforeWork) {
  const FullAttentionParameters p{.query_heads = 2,
                                  .key_value_heads = 1,
                                  .head_dim = 4,
                                  .rotary_dim = 2,
                                  .capacity = 2};
  auto output = Upload(std::vector<float>(16, -17.f));
  ASSERT_TRUE(output.ok());
  const int wrong_shapes[][3] = {{3, 1, 4}, {2, 2, 4}, {2, 1, 6}};
  for (const auto& shape : wrong_shapes) {
    auto cache =
        KeyValueCache::Create(*executor_, shape[0], shape[1], shape[2]);
    ASSERT_TRUE(cache.ok());
    EXPECT_TRUE(absl::IsInvalidArgument(
        ValidateCachedAttention(*executor_, p, **cache)));
    EXPECT_TRUE(absl::IsInvalidArgument(CachedAttentionStep(
        *executor_, p, **cache, Data(*output), Data(*output), Data(*output),
        Data(*output), Data(*output), Data(*output))));
    EXPECT_EQ((*cache)->position(), 0);
  }
  auto other = cuda::Executor::Create();
  ASSERT_TRUE(other.ok());
  auto foreign_cache =
      KeyValueCache::Create(**other, p.capacity, p.key_value_heads, p.head_dim);
  ASSERT_TRUE(foreign_cache.ok());
  EXPECT_TRUE(absl::IsInvalidArgument(
      ValidateCachedAttention(*executor_, p, **foreign_cache)));
  EXPECT_TRUE(absl::IsInvalidArgument(CachedAttentionStep(
      *executor_, p, **foreign_cache, Data(*output), Data(*output),
      Data(*output), Data(*output), Data(*output), Data(*output))));
  EXPECT_EQ((*foreign_cache)->position(), 0);
  EXPECT_EQ(Read(*output), (std::vector<float>(16, -17.f)));
  EXPECT_TRUE((*other)->Synchronize().ok());
}

TEST_F(AttentionOpsTest, CachedAttentionRejectsNullPointersWithoutAdvancing) {
  const FullAttentionParameters p{.query_heads = 2,
                                  .key_value_heads = 1,
                                  .head_dim = 4,
                                  .rotary_dim = 2,
                                  .capacity = 2};
  auto cache = KeyValueCache::Create(*executor_, p.capacity, p.key_value_heads,
                                     p.head_dim);
  auto output = Upload(std::vector<float>(16, -19.f));
  ASSERT_TRUE(cache.ok());
  ASSERT_TRUE(output.ok());
  for (int missing = 0; missing < 6; ++missing) {
    SCOPED_TRACE(missing);
    float* pointers[6];
    std::fill_n(pointers, 6, Data(*output));
    pointers[missing] = nullptr;
    EXPECT_TRUE(absl::IsInvalidArgument(CachedAttentionStep(
        *executor_, p, **cache, pointers[0], pointers[1], pointers[2],
        pointers[3], pointers[4], pointers[5])));
    EXPECT_EQ((*cache)->position(), 0);
  }
  EXPECT_EQ(Read(*output), (std::vector<float>(16, -19.f)));
}

TEST_F(AttentionOpsTest, DeltaNetRejectsUnsupportedShapesAndNullInputs) {
  DeltaNetParameters delta;
  delta.key_head_dim = 129;
  EXPECT_EQ(DeltaNetState::Create(*executor_, delta).status().code(),
            absl::StatusCode::kInvalidArgument);
  delta.key_head_dim = 5;
  delta.value_head_dim = 7;
  auto state = DeltaNetState::Create(*executor_, delta);
  ASSERT_TRUE(state.ok());
  EXPECT_EQ((*state)
                ->Step(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                       nullptr, nullptr, nullptr)
                .code(),
            absl::StatusCode::kInvalidArgument);
}

}  // namespace
}  // namespace pluto::llm::cached_attention_ops
