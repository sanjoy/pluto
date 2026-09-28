#include "src/llm/layers/sequence_delta_net.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda_tile.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>

#include "absl/memory/memory.h"
#include "src/util/status_macros.h"

namespace pluto::llm {
namespace {

namespace ct = ::cuda::tiles;
using namespace ct::literals;

template <class Tile>
__tile__ Tile Round(Tile x, bool round) {
  if (round)
    return ct::element_cast<float>(ct::element_cast<__nv_bfloat16>(x));
  return x;
}

// Causal depthwise convolution, followed by SiLU. Unlike the inference cache,
// sequence indexing makes the initial zero history explicit and reproducible.
__tile_global__ void ConvForward(const __nv_bfloat16* input,
                                 const float* weight, int channels, int kernel,
                                 bool round, float* pre, float* post) {
  const int t = ct::bid().y;
  const int block = ct::bid().x;
  auto c = ct::iota<ct::tile<int, ct::shape<128>>>() + block * 128;
  auto valid = c < channels;
  auto sum = ct::zeros<ct::tile<float, ct::shape<128>>>();
  for (int tap = 0; tap < kernel; ++tap) {
    const int source = t + tap - kernel + 1;
    if (source >= 0) {
      auto x = ct::element_cast<float>(
          ct::load_masked(input + source * channels + c, valid));
      sum = sum + x * ct::load_masked(weight + c * kernel + tap, valid);
    }
  }
  sum = Round(sum, round);
  ct::store_masked(pre + t * channels + c, sum, valid);
  ct::store_masked(post + t * channels + c,
                   Round(sum / (1.f + ct::exp(-sum)), round), valid);
}

// One program owns a [key_dim,32] slice for an entire sequence. It saves S_t
// after each update; no token can observe a partially updated recurrent state.
__tile_global__ void RecurrenceForward(const float* qkv, const __nv_bfloat16* a,
                                       const __nv_bfloat16* b,
                                       const float* a_log, const float* dt_bias,
                                       int length, int kh, int vh, int kd,
                                       int vd, bool round, float* states,
                                       float* core) {
  const int tiles = (vd + 31) / 32;
  const int head = ct::bid().x / tiles;
  const int tile = ct::bid().x % tiles;
  const int key_head = head / (vh / kh);
  const int channels = 2 * kh * kd + vh * vd;
  auto ki = ct::iota<ct::tile<int, ct::shape<128, 1>>>();
  auto vi = ct::iota<ct::tile<int, ct::shape<1, 32>>>() + tile * 32;
  auto scalar = ct::full<ct::tile<int, ct::shape<1, 1>>>(head);
  auto norm_scale = ct::sqrt(
      ct::full<ct::tile<float, ct::shape<1, 1>>>(static_cast<float>(kd)));
  auto offset =
      (static_cast<size_t>(head) * kd + ct::element_cast<size_t>(ki)) *
          static_cast<size_t>(vd) +
      ct::element_cast<size_t>(vi);
  auto valid = (ki < kd) && (vi < vd);
  auto matrix = ct::zeros<ct::tile<float, ct::shape<128, 32>>>();
  const size_t stride = static_cast<size_t>(vh) * kd * vd;
  ct::store_masked(states + offset, matrix, valid);
  for (int t = 0; t < length; ++t) {
    auto q = ct::load_masked(qkv + t * channels + key_head * kd + ki, ki < kd);
    auto k = ct::load_masked(qkv + t * channels + (kh + key_head) * kd + ki,
                             ki < kd);
    q = q * ct::rsqrt(ct::sum(q * q, 0_ic) + 1e-6f) / norm_scale;
    k = k * ct::rsqrt(ct::sum(k * k, 0_ic) + 1e-6f);
    auto v = ct::load_masked(qkv + t * channels + 2 * kh * kd + head * vd + vi,
                             vi < vd);
    auto alpha = ct::element_cast<float>(ct::load(a + t * vh + scalar)) +
                 ct::load(dt_bias + scalar);
    auto softplus = ct::max(alpha, ct::zeros<decltype(alpha)>()) +
                    ct::log(1.f + ct::exp(-ct::abs(alpha)));
    auto decay = ct::exp(-ct::exp(ct::load(a_log + scalar)) * softplus);
    auto beta = Round(1.f / (1.f + ct::exp(-ct::element_cast<float>(
                                       ct::load(b + t * vh + scalar)))),
                      round);
    matrix = matrix * decay;
    auto delta = (v - ct::sum(matrix * k, 0_ic)) * beta;
    matrix = matrix + k * delta;
    ct::store_masked(states + (t + 1) * stride + offset, matrix, valid);
    ct::store_masked(core + (t * vh + head) * vd + vi,
                     Round(ct::sum(matrix * q, 0_ic), round), vi < vd);
  }
}

__tile_global__ void OutputForward(const float* core, const __nv_bfloat16* z,
                                   const float* norm, int vd, float epsilon,
                                   bool round, __nv_bfloat16* output) {
  const int row = ct::bid().x;
  auto vi = ct::iota<ct::tile<int, ct::shape<256>>>();
  auto valid = vi < vd;
  auto x = ct::load_masked(core + row * vd + vi, valid);
  auto inverse =
      ct::rsqrt(ct::sum(x * x, 0_ic) / static_cast<float>(vd) + epsilon);
  auto normalized = Round(x * inverse, round);
  auto affine = Round(normalized * ct::load_masked(norm + vi, valid), round);
  auto gate =
      ct::element_cast<float>(ct::load_masked(z + row * vd + vi, valid));
  ct::store_masked(
      output + row * vd + vi,
      ct::element_cast<__nv_bfloat16>(affine * gate / (1.f + ct::exp(-gate))),
      valid);
}

// Norm's derivative uses the unrounded normalized expression, while its
// multiply consumes the rounded forward value. This is BF16 straight-through
// differentiation, not finite-differencing the discontinuous BF16 quantizer.
__tile_global__ void OutputBackward(const float* core, const __nv_bfloat16* z,
                                    const float* norm, const float* gradient,
                                    int vd, float epsilon, bool round,
                                    float* dcore, float* dz,
                                    float* norm_partials) {
  const int row = ct::bid().x;
  auto vi = ct::iota<ct::tile<int, ct::shape<256>>>();
  auto valid = vi < vd;
  auto x = ct::load_masked(core + row * vd + vi, valid);
  auto inverse =
      ct::rsqrt(ct::sum(x * x, 0_ic) / static_cast<float>(vd) + epsilon);
  auto normalized = Round(x * inverse, round);
  auto w = ct::load_masked(norm + vi, valid);
  auto affine = Round(normalized * w, round);
  auto gate =
      ct::element_cast<float>(ct::load_masked(z + row * vd + vi, valid));
  auto sigmoid = 1.f / (1.f + ct::exp(-gate));
  auto dy = ct::load_masked(gradient + row * vd + vi, valid);
  auto dn = dy * gate * sigmoid * w;
  auto dx = inverse * dn - x * inverse * inverse * inverse /
                               static_cast<float>(vd) * ct::sum(dn * x, 0_ic);
  ct::store_masked(dcore + row * vd + vi, dx, valid);
  ct::store_masked(dz + row * vd + vi,
                   dy * affine * sigmoid * (1.f + gate * (1.f - sigmoid)),
                   valid);
  if (norm_partials != nullptr)
    ct::store_masked(norm_partials + row * vd + vi,
                     dy * gate * sigmoid * normalized, valid);
}

__tile_global__ void SumNormGradient(const float* partials, int rows, int vd,
                                     float* gradient) {
  auto vi = ct::iota<ct::tile<int, ct::shape<256>>>();
  auto valid = vi < vd;
  auto sum = ct::zeros<ct::tile<float, ct::shape<256>>>();
  for (int row = 0; row < rows; ++row)
    sum = sum + ct::load_masked(partials + row * vd + vi, valid);
  ct::store_masked(gradient + vi, ct::load_masked(gradient + vi, valid) + sum,
                   valid);
}

// Differentiate S_t = g*S_(t-1) + k*beta*(v - k^T*g*S_(t-1))^T.
// dm carries every later token's sensitivity to S_t. Q/K and scalar derivatives
// are private partials; subsequent kernels sum head/value slices without races.
__tile_global__ void RecurrenceBackward(
    const float* qkv, const __nv_bfloat16* a, const __nv_bfloat16* b,
    const float* a_log, const float* dt_bias, const float* states,
    const float* dcore, int length, int kh, int vh, int kd, int vd, bool round,
    float* qk_partials, float* scalar_partials, float* dconvolved) {
  const int tiles = (vd + 31) / 32;
  const int head = ct::bid().x / tiles;
  const int tile = ct::bid().x % tiles;
  const int key_head = head / (vh / kh);
  const int channels = 2 * kh * kd + vh * vd;
  auto ki = ct::iota<ct::tile<int, ct::shape<128, 1>>>();
  auto vi = ct::iota<ct::tile<int, ct::shape<1, 32>>>() + tile * 32;
  auto scalar = ct::full<ct::tile<int, ct::shape<1, 1>>>(head);
  auto norm_scale = ct::sqrt(
      ct::full<ct::tile<float, ct::shape<1, 1>>>(static_cast<float>(kd)));
  auto offset =
      (static_cast<size_t>(head) * kd + ct::element_cast<size_t>(ki)) *
          static_cast<size_t>(vd) +
      ct::element_cast<size_t>(vi);
  auto valid = (ki < kd) && (vi < vd);
  const size_t stride = static_cast<size_t>(vh) * kd * vd;
  auto dm = ct::zeros<ct::tile<float, ct::shape<128, 32>>>();
  for (int t = length - 1; t >= 0; --t) {
    auto raw_q =
        ct::load_masked(qkv + t * channels + key_head * kd + ki, ki < kd);
    auto raw_k = ct::load_masked(qkv + t * channels + (kh + key_head) * kd + ki,
                                 ki < kd);
    auto iq = ct::rsqrt(ct::sum(raw_q * raw_q, 0_ic) + 1e-6f);
    auto ik = ct::rsqrt(ct::sum(raw_k * raw_k, 0_ic) + 1e-6f);
    auto q = raw_q * iq / norm_scale;
    auto k = raw_k * ik;
    auto v = ct::load_masked(qkv + t * channels + 2 * kh * kd + head * vd + vi,
                             vi < vd);
    auto alpha = ct::element_cast<float>(ct::load(a + t * vh + scalar)) +
                 ct::load(dt_bias + scalar);
    auto softplus = ct::max(alpha, ct::zeros<decltype(alpha)>()) +
                    ct::log(1.f + ct::exp(-ct::abs(alpha)));
    auto rate = ct::exp(ct::load(a_log + scalar));
    auto decay = ct::exp(-rate * softplus);
    auto sigmoid = 1.f / (1.f + ct::exp(-ct::element_cast<float>(
                                    ct::load(b + t * vh + scalar))));
    auto beta = Round(sigmoid, round);
    auto previous = ct::load_masked(states + t * stride + offset, valid);
    auto decayed = previous * decay;
    auto difference = v - ct::sum(decayed * k, 0_ic);
    auto delta = difference * beta;
    auto matrix = ct::load_masked(states + (t + 1) * stride + offset, valid);
    auto dy = ct::load_masked(dcore + (t * vh + head) * vd + vi, vi < vd);
    auto dq = ct::sum(matrix * dy, 1_ic);
    dm = dm + q * dy;
    auto ddelta = ct::sum(dm * k, 0_ic);
    auto dv = ddelta * beta;
    auto dk = ct::sum(dm * delta - decayed * dv, 1_ic);
    auto ddecayed = dm - k * dv;
    auto dg = ct::sum(ct::sum(ddecayed * previous, 0_ic), 1_ic);
    auto da = -dg * decay * rate / (1.f + ct::exp(-alpha));
    auto db = ct::sum(ddelta * difference, 1_ic) * sigmoid * (1.f - sigmoid);
    auto dalog = -dg * decay * rate * softplus;
    dm = ddecayed * decay;
    auto d_raw_q =
        (dq * iq - raw_q * iq * iq * iq * ct::sum(dq * raw_q, 0_ic)) /
        norm_scale;
    auto d_raw_k = dk * ik - raw_k * ik * ik * ik * ct::sum(dk * raw_k, 0_ic);
    const int partial = (t * vh + head) * tiles + tile;
    ct::store_masked(qk_partials + partial * 2 * kd + ki, d_raw_q, ki < kd);
    ct::store_masked(qk_partials + partial * 2 * kd + kd + ki, d_raw_k,
                     ki < kd);
    auto zero = ct::zeros<ct::tile<int, ct::shape<1, 1>>>();
    ct::store(scalar_partials + partial * 3 + zero, da);
    ct::store(scalar_partials + partial * 3 + 1 + zero, db);
    ct::store(scalar_partials + partial * 3 + 2 + zero, dalog);
    ct::store_masked(dconvolved + t * channels + 2 * kh * kd + head * vd + vi,
                     dv, vi < vd);
  }
}

__tile_global__ void GatherQkGradient(const float* partials, int kh, int vh,
                                      int kd, int vd, float* gradient) {
  const int t = ct::bid().y;
  const int key_head = ct::bid().x;
  const int group = vh / kh;
  const int tiles = (vd + 31) / 32;
  const int channels = 2 * kh * kd + vh * vd;
  auto ki = ct::iota<ct::tile<int, ct::shape<128>>>();
  auto dq = ct::zeros<ct::tile<float, ct::shape<128>>>();
  auto dk = dq;
  for (int h = key_head * group; h < (key_head + 1) * group; ++h)
    for (int tile = 0; tile < tiles; ++tile) {
      const int partial = ((t * vh + h) * tiles + tile) * 2 * kd;
      dq = dq + ct::load_masked(partials + partial + ki, ki < kd);
      dk = dk + ct::load_masked(partials + partial + kd + ki, ki < kd);
    }
  ct::store_masked(gradient + t * channels + key_head * kd + ki, dq, ki < kd);
  ct::store_masked(gradient + t * channels + (kh + key_head) * kd + ki, dk,
                   ki < kd);
}

__tile_global__ void GatherScalarGradient(const float* partials, int length,
                                          int vh, int vd, float* da, float* db,
                                          float* da_log, float* ddt_bias) {
  const int head = ct::bid().x;
  const int tiles = (vd + 31) / 32;
  auto t = ct::iota<ct::tile<int, ct::shape<128>>>();
  auto valid = t < length;
  auto ga = ct::zeros<ct::tile<float, ct::shape<128>>>();
  auto gb = ga;
  auto glog = ga;
  for (int tile = 0; tile < tiles; ++tile) {
    auto index = ((t * vh + head) * tiles + tile) * 3;
    ga = ga + ct::load_masked(partials + index, valid);
    gb = gb + ct::load_masked(partials + index + 1, valid);
    glog = glog + ct::load_masked(partials + index + 2, valid);
  }
  ct::store_masked(da + t * vh + head, ga, valid);
  ct::store_masked(db + t * vh + head, gb, valid);
  auto scalar = ct::full<ct::tile<int, ct::shape<1>>>(head);
  if (da_log != nullptr)
    ct::store(da_log + scalar, ct::load(da_log + scalar) + ct::sum(glog, 0_ic));
  if (ddt_bias != nullptr)
    ct::store(ddt_bias + scalar,
              ct::load(ddt_bias + scalar) + ct::sum(ga, 0_ic));
}

__tile_global__ void ConvSiluBackward(const float* pre, const float* gradient,
                                      int elements, float* dpre) {
  const int block = ct::bid().x;
  auto i = ct::iota<ct::tile<int, ct::shape<128>>>() + block * 128;
  auto valid = i < elements;
  auto x = ct::load_masked(pre + i, valid);
  auto sigmoid = 1.f / (1.f + ct::exp(-x));
  ct::store_masked(dpre + i,
                   ct::load_masked(gradient + i, valid) * sigmoid *
                       (1.f + x * (1.f - sigmoid)),
                   valid);
}

__tile_global__ void ConvInputBackward(const float* dpre, const float* weights,
                                       int length, int channels, int kernel,
                                       float* dx) {
  const int t = ct::bid().y;
  const int block = ct::bid().x;
  auto c = ct::iota<ct::tile<int, ct::shape<128>>>() + block * 128;
  auto valid = c < channels;
  auto sum = ct::zeros<ct::tile<float, ct::shape<128>>>();
  for (int tap = 0; tap < kernel; ++tap) {
    const int future = t + kernel - 1 - tap;
    if (future < length)
      sum = sum + ct::load_masked(dpre + future * channels + c, valid) *
                      ct::load_masked(weights + c * kernel + tap, valid);
  }
  ct::store_masked(dx + t * channels + c, sum, valid);
}

__tile_global__ void ConvWeightBackward(const __nv_bfloat16* input,
                                        const float* dpre, int length,
                                        int channels, int kernel, float* dw) {
  const int tap = ct::bid().y;
  const int block = ct::bid().x;
  auto c = ct::iota<ct::tile<int, ct::shape<128>>>() + block * 128;
  auto valid = c < channels;
  auto sum = ct::zeros<ct::tile<float, ct::shape<128>>>();
  for (int t = kernel - 1 - tap; t < length; ++t) {
    const int source = t + tap - kernel + 1;
    sum = sum + ct::load_masked(dpre + t * channels + c, valid) *
                    ct::element_cast<float>(
                        ct::load_masked(input + source * channels + c, valid));
  }
  auto index = c * kernel + tap;
  ct::store_masked(dw + index, ct::load_masked(dw + index, valid) + sum, valid);
}

absl::Status Validate(cuda::Executor& executor, const Buffer& buffer,
                      size_t bytes) {
  if (buffer.size_bytes() != bytes || &buffer.executor() != &executor)
    return absl::InvalidArgumentError(
        "SequenceDeltaNetLayer requires batch-one buffers of the declared "
        "sequence shape on its Executor");
  return absl::OkStatus();
}

absl::StatusOr<Buffer> Floats(cuda::Executor& executor, size_t count) {
  return Buffer::Allocate(executor, count * sizeof(float));
}

const float* Read(const Buffer& buffer) {
  return static_cast<const float*>(buffer.data());
}
float* Write(Buffer& buffer) { return static_cast<float*>(buffer.data()); }
const __nv_bfloat16* Bf16(const Buffer& buffer) {
  return static_cast<const __nv_bfloat16*>(buffer.data());
}
float* ActiveGradient(const std::shared_ptr<BlockParameter>& parameter) {
  if (!parameter->active())
    return nullptr;
  Buffer gradient = parameter->gradient();
  return static_cast<float*>(gradient.data());
}

}  // namespace

SequenceDeltaNetLayer::SequenceDeltaNetLayer(
    cuda::Executor& executor, cached_attention_ops::DeltaNetParameters p,
    std::shared_ptr<BlockParameter> convolution,
    std::shared_ptr<BlockParameter> a_log,
    std::shared_ptr<BlockParameter> dt_bias,
    std::shared_ptr<BlockParameter> norm, int sequence_length)
    : executor_(executor),
      parameters_(p),
      sequence_length_(sequence_length),
      parameters_owned_{std::move(convolution), std::move(a_log),
                        std::move(dt_bias), std::move(norm)},
      weights_{parameters_owned_[0]->value(), parameters_owned_[1]->value(),
               parameters_owned_[2]->value(), parameters_owned_[3]->value()},
      input_types_{
          {DataType::BF16,
           {-2, sequence_length,
            2LL * p.key_heads * p.key_head_dim +
                1LL * p.value_heads * p.value_head_dim}},
          {DataType::BF16,
           {-2, sequence_length, 1LL * p.value_heads * p.value_head_dim}},
          {DataType::BF16, {-2, sequence_length, p.value_heads}},
          {DataType::BF16, {-2, sequence_length, p.value_heads}}},
      output_types_{
          {DataType::BF16,
           {-2, sequence_length, 1LL * p.value_heads * p.value_head_dim}}} {}

absl::StatusOr<std::unique_ptr<SequenceDeltaNetLayer>>
SequenceDeltaNetLayer::Create(cuda::Executor& executor,
                              cached_attention_ops::DeltaNetParameters p,
                              std::shared_ptr<BlockParameter> convolution,
                              std::shared_ptr<BlockParameter> a_log,
                              std::shared_ptr<BlockParameter> dt_bias,
                              std::shared_ptr<BlockParameter> norm,
                              int sequence_length) {
  if (sequence_length < 1 || sequence_length > 128 || p.key_heads < 1 ||
      p.value_heads < 1 || p.value_heads % p.key_heads != 0 ||
      p.key_head_dim < 1 || p.key_head_dim > 128 || p.value_head_dim < 1 ||
      p.value_head_dim > 256 || p.conv_kernel_dim < 1 ||
      p.conv_kernel_dim > 32 || !std::isfinite(p.rms_norm_epsilon) ||
      p.rms_norm_epsilon <= 0)
    return absl::InvalidArgumentError(
        "unsupported SequenceDeltaNetLayer shape");
  const int64_t channels = 2LL * p.key_heads * p.key_head_dim +
                           1LL * p.value_heads * p.value_head_dim;
  // Kernel offsets for activations/partials are int; recurrent matrices use
  // size_t. Check the largest int-indexed workspaces before doing any launch.
  if (channels * sequence_length > std::numeric_limits<int>::max() / 32 ||
      1LL * sequence_length * p.value_heads * 8 * 2 * p.key_head_dim >
          std::numeric_limits<int>::max())
    return absl::InvalidArgumentError("SequenceDeltaNetLayer shape overflows");
  const std::shared_ptr<BlockParameter> parameters[] = {convolution, a_log,
                                                        dt_bias, norm};
  const int64_t counts[] = {channels * p.conv_kernel_dim, p.value_heads,
                            p.value_heads, p.value_head_dim};
  for (int i = 0; i < 4; ++i) {
    if (!parameters[i] || parameters[i]->storage() != DataType::FP32)
      return absl::InvalidArgumentError("DeltaNet parameters must be FP32");
    RETURN_IF_ERROR(
        Validate(executor, parameters[i]->value(), counts[i] * sizeof(float)));
  }
  return absl::WrapUnique(new SequenceDeltaNetLayer(
      executor, p, std::move(convolution), std::move(a_log), std::move(dt_bias),
      std::move(norm), sequence_length));
}

absl::StatusOr<FwdResult> SequenceDeltaNetLayer::fwd_impl(
    cuda::Executor& executor, absl::Span<const Buffer> inputs,
    LayerHooks*) const {
  if (&executor != &executor_ || inputs.size() != 4)
    return absl::InvalidArgumentError(
        "DeltaNet expects four inputs on its Executor");
  const auto& p = parameters_;
  const int t = sequence_length_;
  const int channels =
      2 * p.key_heads * p.key_head_dim + p.value_heads * p.value_head_dim;
  const int output_width = p.value_heads * p.value_head_dim;
  const int widths[] = {channels, output_width, p.value_heads, p.value_heads};
  for (int i = 0; i < 4; ++i)
    RETURN_IF_ERROR(
        Validate(executor, inputs[i],
                 static_cast<size_t>(t) * widths[i] * sizeof(__nv_bfloat16)));
  ASSIGN_OR_RETURN(auto pre, Floats(executor, t * channels));
  ASSIGN_OR_RETURN(auto post, Floats(executor, t * channels));
  ASSIGN_OR_RETURN(auto core, Floats(executor, t * output_width));
  ASSIGN_OR_RETURN(auto states,
                   Floats(executor, static_cast<size_t>(t + 1) * p.value_heads *
                                        p.key_head_dim * p.value_head_dim));
  ASSIGN_OR_RETURN(
      auto output,
      Buffer::Allocate(executor, static_cast<size_t>(t) * output_width *
                                     sizeof(__nv_bfloat16)));
  ConvForward<<<dim3((channels + 127) / 128, t), 1, 0, executor.stream()>>>(
      Bf16(inputs[0]), Read(weights_[0]), channels, p.conv_kernel_dim,
      p.round_to_bfloat16, Write(pre), Write(post));
  RecurrenceForward<<<p.value_heads*((p.value_head_dim + 31) / 32), 1, 0,
                      executor.stream()>>>(
      Read(post), Bf16(inputs[2]), Bf16(inputs[3]), Read(weights_[1]),
      Read(weights_[2]), t, p.key_heads, p.value_heads, p.key_head_dim,
      p.value_head_dim, p.round_to_bfloat16, Write(states), Write(core));
  OutputForward<<<t * p.value_heads, 1, 0, executor.stream()>>>(
      Read(core), Bf16(inputs[1]), Read(weights_[3]), p.value_head_dim,
      p.rms_norm_epsilon, p.round_to_bfloat16,
      static_cast<__nv_bfloat16*>(output.data()));
  RETURN_IF_ERROR(
      cuda::CudaStatus(cudaGetLastError(), "sequence DeltaNet forward"));
  BackwardState state;
  state.intermediates = {inputs[0],       inputs[1],        inputs[2],
                         inputs[3],       std::move(pre),   std::move(post),
                         std::move(core), std::move(states)};
  return FwdResult{{std::move(output)}, std::move(state)};
}

absl::StatusOr<BufferVec> SequenceDeltaNetLayer::bwd_impl(
    cuda::Executor& executor, absl::Span<const Buffer> gradients,
    BackwardState state, LayerHooks*) {
  if (&executor != &executor_ || gradients.size() != 1 ||
      state.intermediates.size() != 8)
    return absl::InvalidArgumentError(
        "invalid DeltaNet backward state or gradient");
  const auto& p = parameters_;
  const int t = sequence_length_;
  const int channels =
      2 * p.key_heads * p.key_head_dim + p.value_heads * p.value_head_dim;
  const int output_width = p.value_heads * p.value_head_dim;
  const int tiles = (p.value_head_dim + 31) / 32;
  const auto& saved = state.intermediates;
  RETURN_IF_ERROR(
      Validate(executor, gradients[0],
               static_cast<size_t>(t) * output_width * sizeof(float)));
  const size_t expected[] = {
      static_cast<size_t>(t) * channels * sizeof(__nv_bfloat16),
      static_cast<size_t>(t) * output_width * sizeof(__nv_bfloat16),
      static_cast<size_t>(t) * p.value_heads * sizeof(__nv_bfloat16),
      static_cast<size_t>(t) * p.value_heads * sizeof(__nv_bfloat16),
      static_cast<size_t>(t) * channels * sizeof(float),
      static_cast<size_t>(t) * channels * sizeof(float),
      static_cast<size_t>(t) * output_width * sizeof(float),
      static_cast<size_t>(t + 1) * p.value_heads * p.key_head_dim *
          p.value_head_dim * sizeof(float)};
  for (int i = 0; i < 8; ++i)
    RETURN_IF_ERROR(Validate(executor, saved[i], expected[i]));
  ASSIGN_OR_RETURN(auto dqkv, Floats(executor, t * channels));
  ASSIGN_OR_RETURN(auto dz, Floats(executor, t * output_width));
  ASSIGN_OR_RETURN(auto da, Floats(executor, t * p.value_heads));
  ASSIGN_OR_RETURN(auto db, Floats(executor, t * p.value_heads));
  ASSIGN_OR_RETURN(auto dcore, Floats(executor, t * output_width));
  ASSIGN_OR_RETURN(auto dpost, Floats(executor, t * channels));
  ASSIGN_OR_RETURN(auto dpre, Floats(executor, t * channels));
  ASSIGN_OR_RETURN(
      auto qk, Floats(executor, t * p.value_heads * tiles * 2 * p.key_head_dim));
  ASSIGN_OR_RETURN(auto scalars,
                   Floats(executor, t * p.value_heads * tiles * 3));
  ASSIGN_OR_RETURN(
      auto norm_partials,
      Floats(executor, parameters_owned_[3]->active() ? t * output_width : 0));
  OutputBackward<<<t * p.value_heads, 1, 0, executor.stream()>>>(
      Read(saved[6]), Bf16(saved[1]), Read(weights_[3]), Read(gradients[0]),
      p.value_head_dim, p.rms_norm_epsilon, p.round_to_bfloat16, Write(dcore),
      Write(dz),
      parameters_owned_[3]->active() ? Write(norm_partials) : nullptr);
  if (parameters_owned_[3]->active())
    SumNormGradient<<<1, 1, 0, executor.stream()>>>(
        Read(norm_partials), t * p.value_heads, p.value_head_dim,
        ActiveGradient(parameters_owned_[3]));
  RecurrenceBackward<<<p.value_heads * tiles, 1, 0, executor.stream()>>>(
      Read(saved[5]), Bf16(saved[2]), Bf16(saved[3]), Read(weights_[1]),
      Read(weights_[2]), Read(saved[7]), Read(dcore), t, p.key_heads,
      p.value_heads, p.key_head_dim, p.value_head_dim, p.round_to_bfloat16,
      Write(qk), Write(scalars), Write(dpost));
  GatherQkGradient<<<dim3(p.key_heads, t), 1, 0, executor.stream()>>>(
      Read(qk), p.key_heads, p.value_heads, p.key_head_dim, p.value_head_dim,
      Write(dpost));
  GatherScalarGradient<<<p.value_heads, 1, 0, executor.stream()>>>(
      Read(scalars), t, p.value_heads, p.value_head_dim, Write(da), Write(db),
      ActiveGradient(parameters_owned_[1]),
      ActiveGradient(parameters_owned_[2]));
  ConvSiluBackward<<<(t * channels + 127) / 128, 1, 0, executor.stream()>>>(
      Read(saved[4]), Read(dpost), t * channels, Write(dpre));
  ConvInputBackward<<<dim3((channels + 127) / 128, t), 1, 0,
                      executor.stream()>>>(Read(dpre), Read(weights_[0]), t,
                                           channels, p.conv_kernel_dim,
                                           Write(dqkv));
  if (parameters_owned_[0]->active())
    ConvWeightBackward<<<dim3((channels + 127) / 128, p.conv_kernel_dim), 1, 0,
                         executor.stream()>>>(
        Bf16(saved[0]), Read(dpre), t, channels, p.conv_kernel_dim,
        ActiveGradient(parameters_owned_[0]));
  RETURN_IF_ERROR(
      cuda::CudaStatus(cudaGetLastError(), "sequence DeltaNet backward"));
  return BufferVec{std::move(dqkv), std::move(dz), std::move(da),
                   std::move(db)};
}

}  // namespace pluto::llm
