#include "src/llm/qwen/attention_ops.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda_tile.h>

#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <utility>

#include "absl/memory/memory.h"
#include "src/util/status_macros.h"

namespace pluto::llm::qwen {
namespace {

template <class Tile>
__tile__ auto Activation(Tile value, bool round) {
  namespace ct = ::cuda::tiles;
  if (round)
    return ct::element_cast<float>(ct::element_cast<__nv_bfloat16>(value));
  return value;
}

// RMS normalization and partial, split-half RoPE match Qwen3_5Attention. Text
// positions have equal temporal/height/width coordinates, so multimodal RoPE's
// interleaved frequency selection reduces to ordinary one-dimensional RoPE.
__tile_global__ void PrepareAttentionKernel(
    const float* q_gate, const float* k, const float* v, const float* q_norm,
    const float* k_norm, int query_heads, int kv_heads, int dim, int rotary_dim,
    int position, float epsilon, float log_theta, bool round, float* queries,
    float* keys, float* values) {
  namespace ct = ::cuda::tiles;
  using namespace ct::literals;
  const int head = ct::bid().x;
  const bool is_query = head < query_heads;
  const int local_head = is_query ? head : head - query_heads;
  const float* input =
      is_query ? q_gate + local_head * 2 * dim : k + local_head * dim;
  const float* norm = is_query ? q_norm : k_norm;
  auto channel = ct::iota<ct::tile<int, ct::shape<256>>>();
  auto x = ct::load_masked(input + channel, channel < dim);
  auto inverse =
      ct::rsqrt(ct::sum(x * x, 0_ic) / static_cast<float>(dim) + epsilon);
  auto weight = ct::load_masked(norm + channel, channel < dim);
  auto normalized = Activation(x * inverse * (1.0f + weight), round);
  const int half_rotary = rotary_dim / 2;
  auto partner = ct::select(channel < half_rotary, channel + half_rotary,
                            channel - half_rotary);
  auto partner_x = ct::load_masked(input + partner, channel < rotary_dim);
  auto partner_weight = ct::load_masked(norm + partner, channel < rotary_dim);
  auto rotated =
      Activation(partner_x * inverse * (1.0f + partner_weight), round);
  rotated = ct::select(channel < half_rotary, -rotated, rotated);
  auto frequency_index =
      ct::select(channel < half_rotary, channel, channel - half_rotary);
  auto angle = static_cast<float>(position) *
               ct::exp(ct::element_cast<float>(frequency_index) *
                       (-2.0f * log_theta / static_cast<float>(rotary_dim)));
  auto cosine = Activation(ct::cos(angle), round);
  auto sine = Activation(ct::sin(angle), round);
  auto rope = Activation(Activation(normalized * cosine, round) +
                             Activation(rotated * sine, round),
                         round);
  auto result = ct::select(channel < rotary_dim, rope, normalized);
  if (is_query) {
    ct::store_masked(queries + local_head * dim + channel, result,
                     channel < dim);
  } else {
    const size_t offset =
        (static_cast<size_t>(position) * kv_heads + local_head) * dim;
    ct::store_masked(keys + offset + channel, result, channel < dim);
    auto value = ct::load_masked(v + local_head * dim + channel, channel < dim);
    ct::store_masked(values + offset + channel, Activation(value, round),
                     channel < dim);
  }
}

// Online softmax over 32 cached tokens at a time, with one program per query
// head. Scores and sums stay FP32, and no quadratic attention buffer exists.
__tile_global__ void FullAttentionKernel(const float* queries,
                                         const float* q_gate, const float* keys,
                                         const float* values, int query_heads,
                                         int kv_heads, int dim, int length,
                                         float scale, bool round,
                                         float* output) {
  namespace ct = ::cuda::tiles;
  using namespace ct::literals;
  const int head = ct::bid().x;
  const int kv_head = head / (query_heads / kv_heads);
  auto channel = ct::iota<ct::tile<int, ct::shape<1, 256>>>();
  auto rows = ct::iota<ct::tile<int, ct::shape<32, 1>>>();
  auto query = ct::load_masked(queries + head * dim + channel, channel < dim);
  auto maximum = ct::full<ct::tile<float, ct::shape<1, 1>>>(-3.402823466e+38f);
  auto denominator = ct::zeros<ct::tile<float, ct::shape<1, 1>>>();
  auto result = ct::zeros<ct::tile<float, ct::shape<1, 256>>>();
  for (int start = 0; start < length; start += 32) {
    auto token = rows + start;
    auto offsets =
        (ct::element_cast<size_t>(token) * static_cast<size_t>(kv_heads) +
         static_cast<size_t>(kv_head)) *
            static_cast<size_t>(dim) +
        ct::element_cast<size_t>(channel);
    auto visible = (token < length) && (channel < dim);
    auto key = ct::load_masked(keys + offsets, visible);
    auto score = ct::sum(key * query, 1_ic) * scale;
    score = ct::select(token < length, score,
                       ct::full<decltype(score)>(-3.402823466e+38f));
    auto next_maximum = ct::max(maximum, ct::reduce_max(score, 0_ic));
    auto correction = ct::exp(maximum - next_maximum);
    auto probability = ct::select(token < length, ct::exp(score - next_maximum),
                                  ct::zeros<decltype(score)>());
    auto value = ct::load_masked(values + offsets, visible);
    result = result * correction + ct::sum(probability * value, 0_ic);
    denominator = denominator * correction + ct::sum(probability, 0_ic);
    maximum = next_maximum;
  }
  auto gate =
      ct::load_masked(q_gate + (head * 2 + 1) * dim + channel, channel < dim);
  auto sigmoid = Activation(1.0f / (1.0f + ct::exp(-gate)), round);
  auto gated =
      Activation(Activation(result / denominator, round) * sigmoid, round);
  ct::store_masked(output + head * dim + channel, gated, channel < dim);
}

__tile_global__ void ConvolutionKernel(const float* input, const float* weight,
                                       int channels, int width, bool round,
                                       float* state, float* output) {
  namespace ct = ::cuda::tiles;
  const int block = ct::bid().x;
  auto channel = ct::iota<ct::tile<int, ct::shape<128>>>() + block * 128;
  auto valid = channel < channels;
  auto sum = ct::zeros<ct::tile<float, ct::shape<128>>>();
  // Each channel is owned by one program. Shifting in ascending order cannot
  // overwrite a future source; channels never share state entries.
  for (int tap = 0; tap < width - 1; ++tap) {
    auto previous = ct::load_masked(state + channel * width + tap + 1, valid);
    auto w = ct::load_masked(weight + channel * width + tap, valid);
    sum = sum + previous * w;
    ct::store_masked(state + channel * width + tap, previous, valid);
  }
  auto current = Activation(ct::load_masked(input + channel, valid), round);
  auto last_weight =
      ct::load_masked(weight + channel * width + width - 1, valid);
  ct::store_masked(state + channel * width + width - 1, current, valid);
  sum = Activation(sum + current * last_weight, round);
  ct::store_masked(output + channel,
                   Activation(sum / (1.0f + ct::exp(-sum)), round), valid);
}

__tile_global__ void DeltaRecurrenceKernel(const float* qkv, const float* a,
                                           const float* b, const float* a_log,
                                           const float* dt_bias, int key_heads,
                                           int value_heads, int key_dim,
                                           int value_dim, bool round,
                                           float* state, float* output) {
  namespace ct = ::cuda::tiles;
  using namespace ct::literals;
  const int value_tiles = (value_dim + 31) / 32;
  const int head = ct::bid().x / value_tiles;
  const int value_tile = ct::bid().x % value_tiles;
  const int key_head = head / (value_heads / key_heads);
  auto key_index = ct::iota<ct::tile<int, ct::shape<128, 1>>>();
  auto value_index =
      ct::iota<ct::tile<int, ct::shape<1, 32>>>() + value_tile * 32;
  auto q = ct::load_masked(qkv + key_head * key_dim + key_index,
                           key_index < key_dim);
  auto k = ct::load_masked(qkv + (key_heads + key_head) * key_dim + key_index,
                           key_index < key_dim);
  q = q * ct::rsqrt(ct::sum(q * q, 0_ic) + 1e-6f) /
      ct::sqrt(ct::full<ct::tile<float, ct::shape<1, 1>>>(
          static_cast<float>(key_dim)));
  k = k * ct::rsqrt(ct::sum(k * k, 0_ic) + 1e-6f);
  auto v = ct::load_masked(
      qkv + 2 * key_heads * key_dim + head * value_dim + value_index,
      value_index < value_dim);
  auto scalar_index = ct::full<ct::tile<int, ct::shape<1, 1>>>(head);
  auto alpha = ct::load(a + scalar_index) + ct::load(dt_bias + scalar_index);
  // Stable softplus prevents overflow for strongly positive time steps.
  auto softplus = ct::max(alpha, ct::zeros<decltype(alpha)>()) +
                  ct::log(1.0f + ct::exp(-ct::abs(alpha)));
  auto decay = ct::exp(-ct::exp(ct::load(a_log + scalar_index)) * softplus);
  auto beta =
      Activation(1.0f / (1.0f + ct::exp(-ct::load(b + scalar_index))), round);
  auto offsets = (static_cast<size_t>(head) * key_dim +
                  ct::element_cast<size_t>(key_index)) *
                     static_cast<size_t>(value_dim) +
                 ct::element_cast<size_t>(value_index);
  auto valid = (key_index < key_dim) && (value_index < value_dim);
  auto matrix = ct::load_masked(state + offsets, valid) * decay;
  auto delta = (v - ct::sum(matrix * k, 0_ic)) * beta;
  matrix = matrix + k * delta;
  ct::store_masked(state + offsets, matrix, valid);
  ct::store_masked(output + head * value_dim + value_index,
                   Activation(ct::sum(matrix * q, 0_ic), round),
                   value_index < value_dim);
}

__tile_global__ void DeltaOutputKernel(const float* core, const float* z,
                                       const float* norm_weight, int dim,
                                       float epsilon, bool round,
                                       float* output) {
  namespace ct = ::cuda::tiles;
  using namespace ct::literals;
  const int head = ct::bid().x;
  auto channel = ct::iota<ct::tile<int, ct::shape<256>>>();
  auto valid = channel < dim;
  auto x = ct::load_masked(core + head * dim + channel, valid);
  auto normalized = Activation(
      x * ct::rsqrt(ct::sum(x * x, 0_ic) / static_cast<float>(dim) + epsilon),
      round);
  auto affine = Activation(
      normalized * ct::load_masked(norm_weight + channel, valid), round);
  auto gate = ct::load_masked(z + head * dim + channel, valid);
  auto silu = gate / (1.0f + ct::exp(-gate));
  ct::store_masked(output + head * dim + channel,
                   Activation(affine * silu, round), valid);
}

absl::StatusOr<cuda::Buffer> AllocateFloats(cuda::Executor& executor,
                                            size_t count) {
  if (count > std::numeric_limits<size_t>::max() / sizeof(float))
    return absl::ResourceExhaustedError("attention allocation size overflows");
  return cuda::Buffer::Allocate(executor, count * sizeof(float));
}

}  // namespace

FullAttentionState::FullAttentionState(cuda::Executor& executor,
                                       FullAttentionParameters parameters,
                                       cuda::Buffer keys, cuda::Buffer values,
                                       cuda::Buffer queries)
    : executor_(executor),
      parameters_(parameters),
      keys_(std::move(keys)),
      values_(std::move(values)),
      queries_(std::move(queries)) {}

absl::StatusOr<std::unique_ptr<FullAttentionState>> FullAttentionState::Create(
    cuda::Executor& executor, FullAttentionParameters p) {
  if (p.query_heads <= 0 || p.key_value_heads <= 0 ||
      p.query_heads % p.key_value_heads != 0 || p.head_dim <= 0 ||
      p.head_dim > 256 || p.rotary_dim <= 0 || p.rotary_dim > p.head_dim ||
      p.rotary_dim % 2 != 0 || p.capacity <= 0 ||
      p.capacity > std::numeric_limits<int>::max() - 32 ||
      !std::isfinite(p.rope_theta) || p.rope_theta <= 0.0f ||
      !std::isfinite(p.rms_norm_epsilon) || p.rms_norm_epsilon <= 0.0f ||
      p.query_heads > std::numeric_limits<int>::max() / (2 * p.head_dim) ||
      p.query_heads > std::numeric_limits<int>::max() - p.key_value_heads)
    return absl::InvalidArgumentError(
        "invalid full-attention dimensions or parameters");
  const size_t count = static_cast<size_t>(p.capacity) * p.key_value_heads;
  if (count > std::numeric_limits<size_t>::max() / p.head_dim)
    return absl::ResourceExhaustedError("attention cache size overflows");
  ASSIGN_OR_RETURN(auto keys, AllocateFloats(executor, count * p.head_dim));
  ASSIGN_OR_RETURN(auto values, AllocateFloats(executor, count * p.head_dim));
  ASSIGN_OR_RETURN(
      auto queries,
      AllocateFloats(executor, static_cast<size_t>(p.query_heads) * p.head_dim));
  return absl::WrapUnique(new FullAttentionState(
      executor, p, std::move(keys), std::move(values), std::move(queries)));
}

absl::Status FullAttentionState::Step(const float* q_gate, const float* k,
                                      const float* v, const float* q_norm,
                                      const float* k_norm, float* output) {
  if (!q_gate || !k || !v || !q_norm || !k_norm || !output)
    return absl::InvalidArgumentError(
        "attention requires non-null device pointers");
  if (length_ >= parameters_.capacity)
    return absl::ResourceExhaustedError(
        "full-attention cache capacity reached");
  const auto& p = parameters_;
  PrepareAttentionKernel<<<p.query_heads + p.key_value_heads, 1, 0,
                           executor_.stream()>>>(
      q_gate, k, v, q_norm, k_norm, p.query_heads, p.key_value_heads,
      p.head_dim, p.rotary_dim, length_, p.rms_norm_epsilon,
      std::log(p.rope_theta), p.round_to_bfloat16,
      static_cast<float*>(queries_.data()), static_cast<float*>(keys_.data()),
      static_cast<float*>(values_.data()));
  RETURN_IF_ERROR(
      cuda::CudaStatus(cudaGetLastError(), "prepare Qwen attention"));
  FullAttentionKernel<<<p.query_heads, 1, 0, executor_.stream()>>>(
      static_cast<const float*>(queries_.data()), q_gate,
      static_cast<const float*>(keys_.data()),
      static_cast<const float*>(values_.data()), p.query_heads,
      p.key_value_heads, p.head_dim, length_ + 1,
      1.0f / std::sqrt(static_cast<float>(p.head_dim)), p.round_to_bfloat16,
      output);
  RETURN_IF_ERROR(cuda::CudaStatus(cudaGetLastError(), "Qwen full attention"));
  ++length_;
  return absl::OkStatus();
}

absl::Status FullAttentionState::Reset() {
  // Every visible cache entry is overwritten before it can be read again.
  length_ = 0;
  return absl::OkStatus();
}

DeltaNetState::DeltaNetState(cuda::Executor& executor,
                             DeltaNetParameters parameters,
                             cuda::Buffer convolution_state,
                             cuda::Buffer recurrent_state,
                             cuda::Buffer convolved_qkv,
                             cuda::Buffer core_output)
    : executor_(executor),
      parameters_(parameters),
      convolution_state_(std::move(convolution_state)),
      recurrent_state_(std::move(recurrent_state)),
      convolved_qkv_(std::move(convolved_qkv)),
      core_output_(std::move(core_output)) {}

absl::StatusOr<std::unique_ptr<DeltaNetState>> DeltaNetState::Create(
    cuda::Executor& executor, DeltaNetParameters p) {
  if (p.key_heads <= 0 || p.value_heads <= 0 ||
      p.value_heads % p.key_heads != 0 || p.key_head_dim <= 0 ||
      p.key_head_dim > 128 || p.value_head_dim <= 0 || p.value_head_dim > 256 ||
      p.conv_kernel_dim <= 0 || p.conv_kernel_dim > 32 ||
      !std::isfinite(p.rms_norm_epsilon) || p.rms_norm_epsilon <= 0.0f)
    return absl::InvalidArgumentError(
        "invalid DeltaNet dimensions or parameters");
  const size_t channels =
      2 * static_cast<size_t>(p.key_heads) * p.key_head_dim +
      static_cast<size_t>(p.value_heads) * p.value_head_dim;
  if (channels >
      static_cast<size_t>(std::numeric_limits<int>::max()) / p.conv_kernel_dim)
    return absl::ResourceExhaustedError(
        "DeltaNet dimensions exceed indexing limit");
  ASSIGN_OR_RETURN(auto convolution,
                   AllocateFloats(executor, channels * p.conv_kernel_dim));
  ASSIGN_OR_RETURN(
      auto recurrence,
      AllocateFloats(executor, static_cast<size_t>(p.value_heads) *
                                   p.key_head_dim * p.value_head_dim));
  ASSIGN_OR_RETURN(auto qkv, AllocateFloats(executor, channels));
  ASSIGN_OR_RETURN(auto core,
                   AllocateFloats(executor, static_cast<size_t>(p.value_heads) *
                                                p.value_head_dim));
  auto result = absl::WrapUnique(new DeltaNetState(
      executor, p, std::move(convolution), std::move(recurrence),
      std::move(qkv), std::move(core)));
  RETURN_IF_ERROR(result->Reset());
  return result;
}

absl::Status DeltaNetState::Step(const float* qkv, const float* z,
                                 const float* a, const float* b,
                                 const float* conv_weight, const float* a_log,
                                 const float* dt_bias, const float* norm_weight,
                                 float* output) {
  if (!qkv || !z || !a || !b || !conv_weight || !a_log || !dt_bias ||
      !norm_weight || !output)
    return absl::InvalidArgumentError(
        "DeltaNet requires non-null device pointers");
  const auto& p = parameters_;
  const int channels =
      2 * p.key_heads * p.key_head_dim + p.value_heads * p.value_head_dim;
  ConvolutionKernel<<<(channels - 1) / 128 + 1, 1, 0, executor_.stream()>>>(
      qkv, conv_weight, channels, p.conv_kernel_dim, p.round_to_bfloat16,
      static_cast<float*>(convolution_state_.data()),
      static_cast<float*>(convolved_qkv_.data()));
  RETURN_IF_ERROR(
      cuda::CudaStatus(cudaGetLastError(), "Qwen causal convolution"));
  DeltaRecurrenceKernel<<<p.value_heads*((p.value_head_dim + 31) / 32), 1, 0,
                          executor_.stream()>>>(
      static_cast<const float*>(convolved_qkv_.data()), a, b, a_log, dt_bias,
      p.key_heads, p.value_heads, p.key_head_dim, p.value_head_dim,
      p.round_to_bfloat16, static_cast<float*>(recurrent_state_.data()),
      static_cast<float*>(core_output_.data()));
  RETURN_IF_ERROR(
      cuda::CudaStatus(cudaGetLastError(), "Qwen gated delta recurrence"));
  DeltaOutputKernel<<<p.value_heads, 1, 0, executor_.stream()>>>(
      static_cast<const float*>(core_output_.data()), z, norm_weight,
      p.value_head_dim, p.rms_norm_epsilon, p.round_to_bfloat16, output);
  return cuda::CudaStatus(cudaGetLastError(), "Qwen gated delta output norm");
}

absl::Status DeltaNetState::Reset() {
  RETURN_IF_ERROR(cuda::CudaStatus(
      cudaMemsetAsync(convolution_state_.data(), 0,
                      convolution_state_.size_bytes(), executor_.stream()),
      "reset Qwen convolution state"));
  return cuda::CudaStatus(
      cudaMemsetAsync(recurrent_state_.data(), 0, recurrent_state_.size_bytes(),
                      executor_.stream()),
      "reset Qwen recurrent state");
}

}  // namespace pluto::llm::qwen
