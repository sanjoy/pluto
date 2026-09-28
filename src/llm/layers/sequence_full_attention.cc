#include "src/llm/layers/sequence_full_attention.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda_tile.h>

#include <cmath>
#include <utility>

#include "absl/memory/memory.h"
#include "src/llm/layer_hooks.h"
#include "src/util/status_macros.h"

namespace pluto::llm {
namespace {

template <class Tile>
__tile__ auto Round(Tile value, bool round) {
  namespace ct = ::cuda::tiles;
  if (round)
    return ct::element_cast<float>(ct::element_cast<__nv_bfloat16>(value));
  return value;
}

// Q/K normalization is per head, followed by split-half partial RoPE. The
// repeated BF16 boundaries intentionally match cached Qwen inference.
__tile_global__ void PrepareKernel(const __nv_bfloat16* qg,
                                   const __nv_bfloat16* k, const float* qw,
                                   const float* kw, int qh, int kh, int d,
                                   int rotary, float epsilon, float log_theta,
                                   bool round, float* q, float* key) {
  namespace ct = ::cuda::tiles;
  using namespace ct::literals;
  int row = ct::bid().x / (qh + kh);
  int head = ct::bid().x % (qh + kh);
  bool query = head < qh;
  int h = query ? head : head - qh;
  auto c = ct::iota<ct::tile<int, ct::shape<256>>>();
  const __nv_bfloat16* input =
      query ? qg + (row * qh + h) * 2 * d : k + (row * kh + h) * d;
  const float* w = query ? qw : kw;
  auto x = ct::element_cast<float>(ct::load_masked(input + c, c < d));
  auto inv = ct::rsqrt(ct::sum(x * x, 0_ic) / float(d) + epsilon);
  auto n = Round(x * inv * (1.f + ct::load_masked(w + c, c < d)), round);
  auto partner = ct::select(c < rotary / 2, c + rotary / 2, c - rotary / 2);
  auto xp =
      ct::element_cast<float>(ct::load_masked(input + partner, c < rotary));
  auto np =
      Round(xp * inv * (1.f + ct::load_masked(w + partner, c < rotary)), round);
  auto frequency = ct::select(c < rotary / 2, c, c - rotary / 2);
  auto angle = float(row) * ct::exp(ct::element_cast<float>(frequency) *
                                    (-2.f * log_theta / float(rotary)));
  auto co = Round(ct::cos(angle), round);
  auto si = Round(ct::sin(angle), round);
  np = ct::select(c < rotary / 2, -np, np);
  auto rotated = Round(Round(n * co, round) + Round(np * si, round), round);
  auto result = ct::select(c < rotary, rotated, n);
  float* output = query ? q + (row * qh + h) * d : key + (row * kh + h) * d;
  ct::store_masked(output + c, result, c < d);
}

// One program owns one query. Online FP32 softmax follows the cache's 32-key
// tiles. Save unrounded attention for its derivative and probabilities for
// backward (and optional inspection), then apply the rounded sigmoid gate.
__tile_global__ void ForwardKernel(const float* q, const float* k,
                                   const __nv_bfloat16* v,
                                   const __nv_bfloat16* qg, int qh, int kh,
                                   int d, int length, bool round, float scale,
                                   float* probabilities, float* core,
                                   __nv_bfloat16* output) {
  namespace ct = ::cuda::tiles;
  using namespace ct::literals;
  int row = ct::bid().x / qh, h = ct::bid().x % qh;
  int kv = h / (qh / kh);
  auto c = ct::iota<ct::tile<int, ct::shape<1, 256>>>();
  auto r = ct::iota<ct::tile<int, ct::shape<32, 1>>>();
  auto query = ct::load_masked(q + (row * qh + h) * d + c, c < d);
  auto maximum = ct::full<ct::tile<float, ct::shape<1, 1>>>(-3.402823466e38f);
  auto denominator = ct::zeros<ct::tile<float, ct::shape<1, 1>>>();
  auto sum = ct::zeros<ct::tile<float, ct::shape<1, 256>>>();
  for (int start = 0; start <= row; start += 32) {
    auto t = r + start;
    auto valid = (t <= row) && (c < d);
    auto key = ct::load_masked(k + (t * kh + kv) * d + c, valid);
    auto score = ct::sum(key * query, 1_ic) * scale;
    score = ct::select(t <= row, score,
                       ct::full<decltype(score)>(-3.402823466e38f));
    auto next_max = ct::max(maximum, ct::reduce_max(score, 0_ic));
    auto correction = ct::exp(maximum - next_max);
    auto p = ct::select(t <= row, ct::exp(score - next_max),
                        ct::zeros<decltype(score)>());
    auto value = ct::element_cast<float>(
        ct::load_masked(v + (t * kh + kv) * d + c, valid));
    sum = sum * correction + ct::sum(p * value, 0_ic);
    denominator = denominator * correction + ct::sum(p, 0_ic);
    maximum = next_max;
  }
  auto attention = sum / denominator;
  ct::store_masked(core + (row * qh + h) * d + c, attention, c < d);
  auto gate = ct::element_cast<float>(
      ct::load_masked(qg + ((row * qh + h) * 2 + 1) * d + c, c < d));
  auto sigmoid = Round(1.f / (1.f + ct::exp(-gate)), round);
  auto out = Round(Round(attention, round) * sigmoid, round);
  ct::store_masked(output + (row * qh + h) * d + c,
                   ct::element_cast<__nv_bfloat16>(out), c < d);
  for (int start = 0; start < length; start += 32) {
    auto t = r + start;
    auto key =
        ct::load_masked(k + (t * kh + kv) * d + c, (t <= row) && (c < d));
    auto score = ct::sum(key * query, 1_ic) * scale;
    auto p = ct::select(t <= row, ct::exp(score - maximum) / denominator,
                        ct::zeros<decltype(score)>());
    ct::store_masked(probabilities + (h * length + row) * length + t, p,
                     t < length);
  }
}

// Convert the upstream gradient through the sigmoid gate. The derivative of
// each BF16 conversion is identity; sigmoid's own derivative uses its FP32
// value, while the product derivative uses the rounded forward operands.
__tile_global__ void GateBackwardKernel(const __nv_bfloat16* qg,
                                        const float* core, const float* dy,
                                        int qh, int d, bool round, float* dc,
                                        float* dqg) {
  namespace ct = ::cuda::tiles;
  int i = ct::bid().x;
  auto c = ct::iota<ct::tile<int, ct::shape<256>>>();
  auto gate =
      ct::element_cast<float>(ct::load_masked(qg + (i * 2 + 1) * d + c, c < d));
  auto sigmoid = 1.f / (1.f + ct::exp(-gate));
  auto grad = ct::load_masked(dy + i * d + c, c < d);
  auto value = Round(ct::load_masked(core + i * d + c, c < d), round);
  ct::store_masked(dc + i * d + c, grad * Round(sigmoid, round), c < d);
  ct::store_masked(dqg + (i * 2 + 1) * d + c,
                   grad * value * sigmoid * (1.f - sigmoid), c < d);
}

// dS_ij = P_ij * (dC_i dot V_j - dC_i dot C_i). Each query owns dQ and
// the saved score-gradient row. No atomic accumulation or causal truncation.
__tile_global__ void QueryBackwardKernel(const float* k, const __nv_bfloat16* v,
                                         const float* p, const float* core,
                                         const float* dc, int qh, int kh, int d,
                                         int length, float scale, float* ds,
                                         float* dq) {
  namespace ct = ::cuda::tiles;
  using namespace ct::literals;
  int row = ct::bid().x / qh, h = ct::bid().x % qh, kv = h / (qh / kh);
  auto c = ct::iota<ct::tile<int, ct::shape<1, 256>>>();
  auto r = ct::iota<ct::tile<int, ct::shape<32, 1>>>();
  auto grad = ct::load_masked(dc + (row * qh + h) * d + c, c < d);
  auto value = ct::load_masked(core + (row * qh + h) * d + c, c < d);
  auto delta = ct::sum(grad * value, 1_ic);
  auto sum = ct::zeros<ct::tile<float, ct::shape<1, 256>>>();
  for (int start = 0; start < length; start += 32) {
    auto t = r + start;
    auto valid = (t <= row) && (c < d);
    auto values = ct::element_cast<float>(
        ct::load_masked(v + (t * kh + kv) * d + c, valid));
    auto prob =
        ct::load_masked(p + (h * length + row) * length + t, t < length);
    auto score_grad = prob * (ct::sum(grad * values, 1_ic) - delta);
    auto keys = ct::load_masked(k + (t * kh + kv) * d + c, valid);
    sum = sum + ct::sum(score_grad * keys, 0_ic) * scale;
    ct::store_masked(ds + (h * length + row) * length + t, score_grad,
                     t < length);
  }
  ct::store_masked(dq + (row * qh + h) * d + c, sum, c < d);
}

// A KV head can serve multiple query heads. One program owns its key/value
// gradient and sums every later query and grouped head in a fixed order.
__tile_global__ void KeyValueBackwardKernel(const float* q, const float* p,
                                            const float* ds, const float* dc,
                                            int qh, int kh, int d, int length,
                                            float scale, float* dk, float* dv) {
  namespace ct = ::cuda::tiles;
  int row = ct::bid().x / kh, kv = ct::bid().x % kh;
  auto c = ct::iota<ct::tile<int, ct::shape<256>>>();
  auto kg = ct::zeros<ct::tile<float, ct::shape<256>>>();
  auto vg = ct::zeros<ct::tile<float, ct::shape<256>>>();
  auto index = ct::full<ct::tile<int, ct::shape<1>>>(row);
  for (int h = kv * (qh / kh); h < (kv + 1) * (qh / kh); ++h)
    for (int t = row; t < length; ++t) {
      auto prob = ct::load(p + (h * length + t) * length + index);
      auto score_grad = ct::load(ds + (h * length + t) * length + index);
      auto query = ct::load_masked(q + (t * qh + h) * d + c, c < d);
      auto grad = ct::load_masked(dc + (t * qh + h) * d + c, c < d);
      kg = kg + score_grad * query * scale;
      vg = vg + prob * grad;
    }
  ct::store_masked(dk + (row * kh + kv) * d + c, kg, c < d);
  ct::store_masked(dv + (row * kh + kv) * d + c, vg, c < d);
}

// Reverse partial RoPE, then weighted RMSNorm. Norm partials are one row per
// position/head, so the subsequent shared-weight reduction is deterministic.
__tile_global__ void NormBackwardKernel(const __nv_bfloat16* input,
                                        const float* w, const float* grad,
                                        int heads, int d, int rotary,
                                        int input_stride, float epsilon,
                                        float log_theta, bool round, float* dx,
                                        float* dw_parts) {
  namespace ct = ::cuda::tiles;
  using namespace ct::literals;
  int i = ct::bid().x, row = i / heads;
  auto c = ct::iota<ct::tile<int, ct::shape<256>>>();
  auto g = ct::load_masked(grad + i * d + c, c < d);
  auto partner = ct::select(c < rotary / 2, c + rotary / 2, c - rotary / 2);
  auto gp = ct::load_masked(grad + i * d + partner, c < rotary);
  auto frequency = ct::select(c < rotary / 2, c, c - rotary / 2);
  auto angle = float(row) * ct::exp(ct::element_cast<float>(frequency) *
                                    (-2.f * log_theta / float(rotary)));
  auto co = Round(ct::cos(angle), round), si = Round(ct::sin(angle), round);
  auto normalized_grad = ct::select(
      c < rotary, g * co + ct::select(c < rotary / 2, gp * si, -gp * si), g);
  auto x = ct::element_cast<float>(
      ct::load_masked(input + i * input_stride + c, c < d));
  auto inv = ct::rsqrt(ct::sum(x * x, 0_ic) / float(d) + epsilon);
  auto weighted_grad = normalized_grad * (1.f + ct::load_masked(w + c, c < d));
  auto dot = ct::sum(weighted_grad * x, 0_ic);
  auto result = inv * weighted_grad - x * (inv * inv * inv / float(d)) * dot;
  ct::store_masked(dx + i * input_stride + c, result, c < d);
  if (dw_parts)
    ct::store_masked(dw_parts + i * d + c, normalized_grad * x * inv, c < d);
}

__tile_global__ void ReduceNormKernel(const float* parts, int rows, int d,
                                      float* dw) {
  namespace ct = ::cuda::tiles;
  auto c = ct::iota<ct::tile<int, ct::shape<256>>>();
  auto sum = ct::zeros<ct::tile<float, ct::shape<256>>>();
  for (int row = 0; row < rows; ++row)
    sum = sum + ct::load_masked(parts + row * d + c, c < d);
  auto previous = ct::load_masked(dw + c, c < d);
  ct::store_masked(dw + c, previous + sum, c < d);
}

absl::Status Validate(cuda::Executor& executor,
                      absl::Span<const Buffer> buffers,
                      absl::Span<const size_t> sizes) {
  if (buffers.size() != sizes.size())
    return absl::InvalidArgumentError(
        "sequence attention buffer count mismatch");
  for (size_t i = 0; i < sizes.size(); ++i)
    if (&buffers[i].executor() != &executor ||
        buffers[i].size_bytes() != sizes[i])
      return absl::InvalidArgumentError(
          "sequence attention expects batch-one buffers on its Executor");
  return absl::OkStatus();
}
absl::StatusOr<Buffer> Floats(cuda::Executor& executor, size_t count) {
  return Buffer::Allocate(executor, count * sizeof(float));
}
const float* F(const Buffer& b) { return static_cast<const float*>(b.data()); }
float* F(Buffer& b) { return static_cast<float*>(b.data()); }
const __nv_bfloat16* B(const Buffer& b) {
  return static_cast<const __nv_bfloat16*>(b.data());
}

}  // namespace

SequenceFullAttentionLayer::SequenceFullAttentionLayer(
    cuda::Executor& executor, FullAttentionParameters parameters,
    std::shared_ptr<BlockParameter> q_norm,
    std::shared_ptr<BlockParameter> k_norm, int sequence_length)
    : executor_(executor),
      parameters_(parameters),
      q_norm_(std::move(q_norm)),
      k_norm_(std::move(k_norm)),
      sequence_length_(sequence_length),
      weights_{q_norm_->value(), k_norm_->value()},
      input_types_{{DataType::BF16,
                    {-2, sequence_length,
                     2LL * parameters.query_heads * parameters.head_dim}},
                   {DataType::BF16,
                    {-2, sequence_length,
                     1LL * parameters.key_value_heads * parameters.head_dim}},
                   {DataType::BF16,
                    {-2, sequence_length,
                     1LL * parameters.key_value_heads * parameters.head_dim}}},
      output_types_{{DataType::BF16,
                     {-2, sequence_length,
                      1LL * parameters.query_heads * parameters.head_dim}}} {}

absl::StatusOr<std::unique_ptr<SequenceFullAttentionLayer>>
SequenceFullAttentionLayer::Create(cuda::Executor& executor,
                                   FullAttentionParameters p,
                                   std::shared_ptr<BlockParameter> q_norm,
                                   std::shared_ptr<BlockParameter> k_norm,
                                   int sequence_length) {
  if (p.query_heads <= 0 || p.query_heads > 256 || p.key_value_heads <= 0 ||
      p.query_heads % p.key_value_heads || p.head_dim <= 0 ||
      p.head_dim > 256 || p.rotary_dim <= 0 || p.rotary_dim > p.head_dim ||
      p.rotary_dim % 2 || sequence_length <= 0 || sequence_length > 128 ||
      !(p.rms_norm_epsilon > 0) || !std::isfinite(p.rms_norm_epsilon) ||
      !(p.rope_theta > 0) || !std::isfinite(p.rope_theta))
    return absl::InvalidArgumentError(
        "sequence attention requires valid GQA dimensions, D<=256 and "
        "1<=T<=128");
  for (const auto& norm : {q_norm, k_norm})
    if (!norm || norm->storage() != DataType::FP32 ||
        norm->elements() != static_cast<size_t>(p.head_dim) ||
        &norm->value().executor() != &executor)
      return absl::InvalidArgumentError(
          "sequence attention norm must be FP32 head-width parameters on its "
          "Executor");
  return absl::WrapUnique(new SequenceFullAttentionLayer(
      executor, p, std::move(q_norm), std::move(k_norm), sequence_length));
}

absl::StatusOr<FwdResult> SequenceFullAttentionLayer::fwd_impl(
    cuda::Executor& executor, absl::Span<const Buffer> inputs,
    LayerHooks* hooks) const {
  if (&executor != &executor_)
    return absl::InvalidArgumentError("sequence attention Executor mismatch");
  const auto& p = parameters_;
  int t = sequence_length_, d = p.head_dim, qh = p.query_heads,
      kh = p.key_value_heads;
  size_t qn = size_t(t) * qh * d, kn = size_t(t) * kh * d;
  RETURN_IF_ERROR(Validate(executor, inputs, {4 * qn, 2 * kn, 2 * kn}));
  ASSIGN_OR_RETURN(auto q, Floats(executor, qn));
  ASSIGN_OR_RETURN(auto k, Floats(executor, kn));
  ASSIGN_OR_RETURN(auto probs, Floats(executor, size_t(t) * t * qh));
  ASSIGN_OR_RETURN(auto core, Floats(executor, qn));
  ASSIGN_OR_RETURN(auto output, Buffer::Allocate(executor, 2 * qn));
  PrepareKernel<<<t*(qh + kh), 1, 0, executor.stream()>>>(
      B(inputs[0]), B(inputs[1]), F(q_norm_->value()), F(k_norm_->value()), qh,
      kh, d, p.rotary_dim, p.rms_norm_epsilon, std::log(p.rope_theta),
      p.round_to_bfloat16, F(q), F(k));
  ForwardKernel<<<t * qh, 1, 0, executor.stream()>>>(
      F(q), F(k), B(inputs[2]), B(inputs[0]), qh, kh, d, t, p.round_to_bfloat16,
      1.f / std::sqrt(float(d)), F(probs), F(core),
      static_cast<__nv_bfloat16*>(output.data()));
  RETURN_IF_ERROR(
      cuda::CudaStatus(cudaGetLastError(), "sequence full attention forward"));
  if (hooks && hooks->attention_probabilities_hook)
    RETURN_IF_ERROR(hooks->attention_probabilities_hook(
        executor, name(), ActivationType(DataType::FP32, {1, qh, t, t}),
        probs));
  return FwdResult{{std::move(output)},
                   {nullptr,
                    {inputs[0], inputs[1], inputs[2], std::move(q),
                     std::move(k), std::move(probs), std::move(core)},
                    {}}};
}

absl::StatusOr<BufferVec> SequenceFullAttentionLayer::bwd_impl(
    cuda::Executor& executor, absl::Span<const Buffer> gradients,
    BackwardState state, LayerHooks*) {
  if (&executor != &executor_ || state.intermediates.size() != 7)
    return absl::InvalidArgumentError(
        "sequence attention backward Executor/state mismatch");
  const auto& p = parameters_;
  int t = sequence_length_, d = p.head_dim, qh = p.query_heads,
      kh = p.key_value_heads;
  size_t qn = size_t(t) * qh * d, kn = size_t(t) * kh * d;
  RETURN_IF_ERROR(Validate(executor, gradients, {4 * qn}));
  auto& s = state.intermediates;
  ASSIGN_OR_RETURN(auto dqg, Floats(executor, 2 * qn));
  ASSIGN_OR_RETURN(auto dk, Floats(executor, kn));
  ASSIGN_OR_RETURN(auto dv, Floats(executor, kn));
  ASSIGN_OR_RETURN(auto dq_rope, Floats(executor, qn));
  ASSIGN_OR_RETURN(auto dk_rope, Floats(executor, kn));
  ASSIGN_OR_RETURN(auto dc, Floats(executor, qn));
  ASSIGN_OR_RETURN(auto ds, Floats(executor, size_t(t) * t * qh));
  ASSIGN_OR_RETURN(auto qw_parts, Floats(executor, q_norm_->active() ? qn : 0));
  ASSIGN_OR_RETURN(auto kw_parts, Floats(executor, k_norm_->active() ? kn : 0));
  GateBackwardKernel<<<t * qh, 1, 0, executor.stream()>>>(
      B(s[0]), F(s[6]), F(gradients[0]), qh, d, p.round_to_bfloat16, F(dc),
      F(dqg));
  QueryBackwardKernel<<<t * qh, 1, 0, executor.stream()>>>(
      F(s[4]), B(s[2]), F(s[5]), F(s[6]), F(dc), qh, kh, d, t,
      1.f / std::sqrt(float(d)), F(ds), F(dq_rope));
  KeyValueBackwardKernel<<<t * kh, 1, 0, executor.stream()>>>(
      F(s[3]), F(s[5]), F(ds), F(dc), qh, kh, d, t, 1.f / std::sqrt(float(d)),
      F(dk_rope), F(dv));
  NormBackwardKernel<<<t * qh, 1, 0, executor.stream()>>>(
      B(s[0]), F(q_norm_->value()), F(dq_rope), qh, d, p.rotary_dim, 2 * d,
      p.rms_norm_epsilon, std::log(p.rope_theta), p.round_to_bfloat16, F(dqg),
      q_norm_->active() ? F(qw_parts) : nullptr);
  NormBackwardKernel<<<t * kh, 1, 0, executor.stream()>>>(
      B(s[1]), F(k_norm_->value()), F(dk_rope), kh, d, p.rotary_dim, d,
      p.rms_norm_epsilon, std::log(p.rope_theta), p.round_to_bfloat16, F(dk),
      k_norm_->active() ? F(kw_parts) : nullptr);
  if (q_norm_->active())
    ReduceNormKernel<<<1, 1, 0, executor.stream()>>>(
        F(qw_parts), t * qh, d,
        static_cast<float*>(q_norm_->gradient().data()));
  if (k_norm_->active())
    ReduceNormKernel<<<1, 1, 0, executor.stream()>>>(
        F(kw_parts), t * kh, d,
        static_cast<float*>(k_norm_->gradient().data()));
  RETURN_IF_ERROR(
      cuda::CudaStatus(cudaGetLastError(), "sequence full attention backward"));
  return BufferVec{std::move(dqg), std::move(dk), std::move(dv)};
}

}  // namespace pluto::llm
