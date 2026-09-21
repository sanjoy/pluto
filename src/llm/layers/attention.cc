#include "src/llm/layers/attention.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda_tile.h>

#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <utility>

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "src/llm/layer_hooks.h"
#include "src/llm/layers/util.h"
#include "src/util/status_macros.h"

namespace pluto::llm {
namespace {

// A block owns 32 query (or key) rows and 64 output channels. In the GPT-2
// recipe that is a complete head: every score is shared across all 64 channels.
// Wider heads use multiple output tiles, but model dimensions remain dynamic.
constexpr int kAttentionRows = 32;
constexpr int kAttentionChannels = 64;

struct AttentionProbabilityLayout {
  size_t bytes;
  int blocks;
};

// Validate the optional quadratic tensor before allocating it or narrowing
// its launch grid. Even a valid packed Q/K/V input can have an unrepresentable
// attention matrix. No overflowing product is evaluated on either path.
// fwd has already checked that rows is positive and divisible by the positive
// context length, so every factor below (including batch size) is nonzero.
absl::StatusOr<AttentionProbabilityLayout> ProbabilityLayout(int rows,
                                                             int context_length,
                                                             int num_heads) {
  size_t bytes = static_cast<size_t>(rows);
  for (size_t factor : {static_cast<size_t>(num_heads),
                        static_cast<size_t>(context_length), sizeof(float)}) {
    if (bytes > std::numeric_limits<size_t>::max() / factor)
      return absl::ResourceExhaustedError(
          "attention probabilities tensor byte size overflows size_t");
    bytes *= factor;
  }
  const int row_tiles =
      context_length / kAttentionRows + (context_length % kAttentionRows != 0);
  int blocks = 1;
  for (int factor : {rows / context_length, num_heads, row_tiles, row_tiles}) {
    if (blocks > std::numeric_limits<int>::max() / factor)
      return absl::ResourceExhaustedError(
          "attention probabilities tile count exceeds CUDA's x-grid limit");
    blocks *= factor;
  }
  return AttentionProbabilityLayout{bytes, blocks};
}

// Slice one sequence/head with explicit strides. Masked tile loads/stores then
// handle both partial sequence tiles and partial head tiles without reading a
// neighboring sequence, another head, or another part of packed Q/K/V.
template <class T>
__tile__ auto HeadView(T* data, int context_length, int head_dimension,
                       int row_stride) {
  namespace ct = ::cuda::tiles;
  using namespace ct::literals;
  auto mapping = ct::layout_strided_mapping{
      ct::extents{context_length, head_dimension}, ct::extents{row_stride, 1}};
  return ct::partition_view{ct::tensor_span{data, mapping},
                            ct::shape{32_ic, 64_ic}};
}

template <class T>
__tile__ auto StatisticsView(T* data, int context_length) {
  namespace ct = ::cuda::tiles;
  using namespace ct::literals;
  return ct::partition_view{ct::tensor_span{data, ct::extents{context_length}},
                            ct::shape{32_ic}};
}

// A finite negative sentinel keeps padded query rows harmless (no -inf - -inf),
// and the explicit mask makes their contribution exactly zero. Valid query
// rows always have at least one visible key in the first visited tile.
__tile__ auto CausalMask(int query_tile, int key_tile, int context_length) {
  namespace ct = ::cuda::tiles;
  auto queries =
      ct::iota<ct::tile<int, ct::shape<32, 1>>>() + query_tile * kAttentionRows;
  auto keys =
      ct::iota<ct::tile<int, ct::shape<1, 32>>>() + key_tile * kAttentionRows;
  return (keys <= queries) && (keys < context_length) &&
         (queries < context_length);
}

template <class Activation>
__tile_global__ void FlashAttentionForwardKernel(
    const Activation* __restrict__ qkv, int context_length, int num_heads,
    int embedding_dim, float scale, Activation* __restrict__ output,
    float* __restrict__ maxima, float* __restrict__ normalizers) {
  namespace ct = ::cuda::tiles;
  using namespace ct::literals;
  const int head_dimension = embedding_dim / num_heads;
  const int channel_tiles =
      (head_dimension + kAttentionChannels - 1) / kAttentionChannels;
  const int row_tiles = (context_length + kAttentionRows - 1) / kAttentionRows;
  const int output_tile = ct::bid().x % channel_tiles;
  const int query_tile = ct::bid().x / channel_tiles % row_tiles;
  const int sequence_head = ct::bid().x / channel_tiles / row_tiles;
  const int sequence = sequence_head / num_heads;
  const int head = sequence_head % num_heads;
  const size_t packed_offset =
      static_cast<size_t>(sequence) * context_length * 3 * embedding_dim +
      head * head_dimension;
  const size_t output_offset =
      static_cast<size_t>(sequence) * context_length * embedding_dim +
      head * head_dimension;
  auto q = HeadView(qkv + packed_offset, context_length, head_dimension,
                    3 * embedding_dim);
  auto k = HeadView(qkv + packed_offset + embedding_dim, context_length,
                    head_dimension, 3 * embedding_dim);
  auto v = HeadView(qkv + packed_offset + 2 * embedding_dim, context_length,
                    head_dimension, 3 * embedding_dim);
  auto out = HeadView(output + output_offset, context_length, head_dimension,
                      embedding_dim);
  const size_t stats_offset =
      static_cast<size_t>(sequence_head) * context_length;
  auto max_view = StatisticsView(maxima + stats_offset, context_length);
  auto norm_view = StatisticsView(normalizers + stats_offset, context_length);
  auto maximum = ct::full<ct::tile<float, ct::shape<32, 1>>>(-3.402823466e+38f);
  auto normalizer = ct::zeros<ct::tile<float, ct::shape<32, 1>>>();
  auto accumulator = ct::zeros<ct::tile<float, ct::shape<32, 64>>>();

  // Online softmax over key tiles: rescale the old sum whenever its maximum
  // changes. Only a 32x32 score tile is live, never the quadratic matrix.
  for (int key_tile = 0; key_tile <= query_tile; ++key_tile) {
    auto score = ct::zeros<ct::tile<float, ct::shape<32, 32>>>();
    for (int dim_tile = 0; dim_tile < channel_tiles; ++dim_tile)
      score = ct::mma(q.load_masked(query_tile, dim_tile),
                      ct::transpose(k.load_masked(key_tile, dim_tile)), score);
    score = score * scale;
    auto visible = CausalMask(query_tile, key_tile, context_length);
    score = ct::select(visible, score,
                       ct::full<decltype(score)>(-3.402823466e+38f));
    auto new_maximum = ct::max(maximum, ct::reduce_max(score, 1_ic));
    auto old_scale = ct::exp(maximum - new_maximum);
    auto probability = ct::select(visible, ct::exp(score - new_maximum),
                                  ct::zeros<decltype(score)>());
    // Probabilities and sensitive products stay FP32; Q.K uses native BF16
    // tensor operands for BF16 inputs. All matrix accumulators are FP32.
    accumulator =
        ct::mma(probability,
                ct::element_cast<float>(v.load_masked(key_tile, output_tile)),
                accumulator * old_scale);
    normalizer = normalizer * old_scale + ct::sum(probability, 1_ic);
    maximum = new_maximum;
  }
  // Padded query rows have zero normalizers, but are not stored.
  auto safe_normalizer = ct::select(normalizer > 0.0f, normalizer,
                                    ct::full<decltype(normalizer)>(1.0f));
  out.store_masked(ct::element_cast<Activation>(accumulator / safe_normalizer),
                   query_tile, output_tile);
  if (output_tile == 0) {
    max_view.store_masked(ct::reshape(maximum, ct::shape{32_ic}), query_tile);
    norm_view.store_masked(ct::reshape(safe_normalizer, ct::shape{32_ic}),
                           query_tile);
  }
}

// Inspection uses the same Q.K tile products and forward softmax statistics as
// backward. One program owns a [32 queries, 32 keys] tile, so storing the full
// matrix needs no atomics. Fully masked future tiles still store zeros: leaving
// them uninitialized would expose stale device memory to the callback.
template <class Activation>
__tile_global__ void AttentionProbabilitiesKernel(
    const Activation* __restrict__ qkv, const float* __restrict__ maxima,
    const float* __restrict__ normalizers, int context_length, int num_heads,
    int embedding_dim, float scale, float* __restrict__ probabilities) {
  namespace ct = ::cuda::tiles;
  using namespace ct::literals;
  const int head_dimension = embedding_dim / num_heads;
  const int channel_tiles =
      (head_dimension + kAttentionChannels - 1) / kAttentionChannels;
  const int row_tiles =
      context_length / kAttentionRows + (context_length % kAttentionRows != 0);
  const int key_tile = ct::bid().x % row_tiles;
  const int query_tile = ct::bid().x / row_tiles % row_tiles;
  const int sequence_head = ct::bid().x / row_tiles / row_tiles;
  const int sequence = sequence_head / num_heads;
  const int head = sequence_head % num_heads;
  const size_t packed_offset =
      static_cast<size_t>(sequence) * context_length * 3 * embedding_dim +
      head * head_dimension;
  auto q = HeadView(qkv + packed_offset, context_length, head_dimension,
                    3 * embedding_dim);
  auto k = HeadView(qkv + packed_offset + embedding_dim, context_length,
                    head_dimension, 3 * embedding_dim);
  auto probability = ct::zeros<ct::tile<float, ct::shape<32, 32>>>();
  if (key_tile <= query_tile) {
    auto score = ct::zeros<ct::tile<float, ct::shape<32, 32>>>();
    for (int dim_tile = 0; dim_tile < channel_tiles; ++dim_tile)
      score = ct::mma(q.load_masked(query_tile, dim_tile),
                      ct::transpose(k.load_masked(key_tile, dim_tile)), score);
    const size_t stats_offset =
        static_cast<size_t>(sequence_head) * context_length;
    auto maximum =
        ct::reshape(StatisticsView(maxima + stats_offset, context_length)
                        .load_masked(query_tile),
                    ct::shape{32_ic, 1_ic});
    auto normalizer =
        ct::reshape(StatisticsView(normalizers + stats_offset, context_length)
                        .load_masked(query_tile),
                    ct::shape{32_ic, 1_ic});
    // Padded query rows are never stored, but avoid dividing by zero there.
    normalizer = ct::select(normalizer > 0.0f, normalizer,
                            ct::full<decltype(normalizer)>(1.0f));
    probability = ct::select(CausalMask(query_tile, key_tile, context_length),
                             ct::exp(score * scale - maximum) / normalizer,
                             ct::zeros<decltype(score)>());
  }
  const size_t probability_offset =
      static_cast<size_t>(sequence_head) * context_length * context_length;
  auto out = ct::partition_view{
      ct::tensor_span{probabilities + probability_offset,
                      ct::extents{context_length, context_length}},
      ct::shape{32_ic, 32_ic}};
  out.store_masked(probability, query_tile, key_tile);
}

// Each query tile owns its dQ tile. It also computes delta=sum(dO*O) once per
// row/head for the following key-owned kernel. O is the stored (possibly BF16)
// forward output, matching the layer's existing backward/reference convention.
template <class Activation>
__tile_global__ void FlashAttentionQueryBackwardKernel(
    const Activation* __restrict__ qkv, const Activation* __restrict__ output,
    const float* __restrict__ output_gradient, const float* __restrict__ maxima,
    const float* __restrict__ normalizers, int context_length, int num_heads,
    int embedding_dim, float scale, float* __restrict__ qkv_gradient,
    float* __restrict__ deltas) {
  namespace ct = ::cuda::tiles;
  using namespace ct::literals;
  const int head_dimension = embedding_dim / num_heads;
  const int channel_tiles =
      (head_dimension + kAttentionChannels - 1) / kAttentionChannels;
  const int row_tiles = (context_length + kAttentionRows - 1) / kAttentionRows;
  const int output_tile = ct::bid().x % channel_tiles;
  const int query_tile = ct::bid().x / channel_tiles % row_tiles;
  const int sequence_head = ct::bid().x / channel_tiles / row_tiles;
  const int sequence = sequence_head / num_heads;
  const int head = sequence_head % num_heads;
  const size_t packed_offset =
      static_cast<size_t>(sequence) * context_length * 3 * embedding_dim +
      head * head_dimension;
  const size_t output_offset =
      static_cast<size_t>(sequence) * context_length * embedding_dim +
      head * head_dimension;
  auto q = HeadView(qkv + packed_offset, context_length, head_dimension,
                    3 * embedding_dim);
  auto k = HeadView(qkv + packed_offset + embedding_dim, context_length,
                    head_dimension, 3 * embedding_dim);
  auto v = HeadView(qkv + packed_offset + 2 * embedding_dim, context_length,
                    head_dimension, 3 * embedding_dim);
  auto out = HeadView(output + output_offset, context_length, head_dimension,
                      embedding_dim);
  auto d_out = HeadView(output_gradient + output_offset, context_length,
                        head_dimension, embedding_dim);
  auto d_q = HeadView(qkv_gradient + packed_offset, context_length,
                      head_dimension, 3 * embedding_dim);
  const size_t stats_offset =
      static_cast<size_t>(sequence_head) * context_length;
  auto maximum =
      ct::reshape(StatisticsView(maxima + stats_offset, context_length)
                      .load_masked(query_tile),
                  ct::shape{32_ic, 1_ic});
  auto normalizer =
      ct::reshape(StatisticsView(normalizers + stats_offset, context_length)
                      .load_masked(query_tile),
                  ct::shape{32_ic, 1_ic});
  normalizer = ct::select(normalizer > 0.0f, normalizer,
                          ct::full<decltype(normalizer)>(1.0f));
  auto delta = ct::zeros<ct::tile<float, ct::shape<32, 1>>>();
  for (int dim_tile = 0; dim_tile < channel_tiles; ++dim_tile)
    delta = delta + ct::sum(d_out.load_masked(query_tile, dim_tile) *
                                ct::element_cast<float>(
                                    out.load_masked(query_tile, dim_tile)),
                            1_ic);
  if (output_tile == 0)
    StatisticsView(deltas + stats_offset, context_length)
        .store_masked(ct::reshape(delta, ct::shape{32_ic}), query_tile);
  auto gradient = ct::zeros<ct::tile<float, ct::shape<32, 64>>>();
  for (int key_tile = 0; key_tile <= query_tile; ++key_tile) {
    auto score = ct::zeros<ct::tile<float, ct::shape<32, 32>>>();
    auto d_probability = ct::zeros<ct::tile<float, ct::shape<32, 32>>>();
    for (int dim_tile = 0; dim_tile < channel_tiles; ++dim_tile) {
      score = ct::mma(q.load_masked(query_tile, dim_tile),
                      ct::transpose(k.load_masked(key_tile, dim_tile)), score);
      d_probability = ct::mma(d_out.load_masked(query_tile, dim_tile),
                              ct::transpose(ct::element_cast<float>(
                                  v.load_masked(key_tile, dim_tile))),
                              d_probability);
    }
    auto probability =
        ct::select(CausalMask(query_tile, key_tile, context_length),
                   ct::exp(score * scale - maximum) / normalizer,
                   ct::zeros<decltype(score)>());
    auto d_score = probability * (d_probability - delta) * scale;
    gradient = ct::mma(
        d_score, ct::element_cast<float>(k.load_masked(key_tile, output_tile)),
        gradient);
  }
  d_q.store_masked(gradient, query_tile, output_tile);
}

// Each key tile owns both dK and dV, visiting query tiles in ascending order.
// No other block updates these locations: no atomics, races, or scheduling-
// dependent sums. Recompute score tiles rather than storing O(context^2) state.
template <class Activation>
__tile_global__ void FlashAttentionKeyValueBackwardKernel(
    const Activation* __restrict__ qkv,
    const float* __restrict__ output_gradient, const float* __restrict__ maxima,
    const float* __restrict__ normalizers, const float* __restrict__ deltas,
    int context_length, int num_heads, int embedding_dim, float scale,
    float* __restrict__ qkv_gradient) {
  namespace ct = ::cuda::tiles;
  using namespace ct::literals;
  const int head_dimension = embedding_dim / num_heads;
  const int channel_tiles =
      (head_dimension + kAttentionChannels - 1) / kAttentionChannels;
  const int row_tiles = (context_length + kAttentionRows - 1) / kAttentionRows;
  const int output_tile = ct::bid().x % channel_tiles;
  const int key_tile = ct::bid().x / channel_tiles % row_tiles;
  const int sequence_head = ct::bid().x / channel_tiles / row_tiles;
  const int sequence = sequence_head / num_heads;
  const int head = sequence_head % num_heads;
  const size_t packed_offset =
      static_cast<size_t>(sequence) * context_length * 3 * embedding_dim +
      head * head_dimension;
  const size_t output_offset =
      static_cast<size_t>(sequence) * context_length * embedding_dim +
      head * head_dimension;
  auto q = HeadView(qkv + packed_offset, context_length, head_dimension,
                    3 * embedding_dim);
  auto k = HeadView(qkv + packed_offset + embedding_dim, context_length,
                    head_dimension, 3 * embedding_dim);
  auto v = HeadView(qkv + packed_offset + 2 * embedding_dim, context_length,
                    head_dimension, 3 * embedding_dim);
  auto d_out = HeadView(output_gradient + output_offset, context_length,
                        head_dimension, embedding_dim);
  auto d_k = HeadView(qkv_gradient + packed_offset + embedding_dim,
                      context_length, head_dimension, 3 * embedding_dim);
  auto d_v = HeadView(qkv_gradient + packed_offset + 2 * embedding_dim,
                      context_length, head_dimension, 3 * embedding_dim);
  const size_t stats_offset =
      static_cast<size_t>(sequence_head) * context_length;
  auto max_view = StatisticsView(maxima + stats_offset, context_length);
  auto norm_view = StatisticsView(normalizers + stats_offset, context_length);
  auto delta_view = StatisticsView(deltas + stats_offset, context_length);
  auto key_gradient = ct::zeros<ct::tile<float, ct::shape<32, 64>>>();
  auto value_gradient = ct::zeros<ct::tile<float, ct::shape<32, 64>>>();
  for (int query_tile = key_tile; query_tile < row_tiles; ++query_tile) {
    auto score = ct::zeros<ct::tile<float, ct::shape<32, 32>>>();
    auto d_probability = ct::zeros<ct::tile<float, ct::shape<32, 32>>>();
    for (int dim_tile = 0; dim_tile < channel_tiles; ++dim_tile) {
      score = ct::mma(q.load_masked(query_tile, dim_tile),
                      ct::transpose(k.load_masked(key_tile, dim_tile)), score);
      d_probability = ct::mma(d_out.load_masked(query_tile, dim_tile),
                              ct::transpose(ct::element_cast<float>(
                                  v.load_masked(key_tile, dim_tile))),
                              d_probability);
    }
    auto maximum =
        ct::reshape(max_view.load_masked(query_tile), ct::shape{32_ic, 1_ic});
    auto normalizer =
        ct::reshape(norm_view.load_masked(query_tile), ct::shape{32_ic, 1_ic});
    normalizer = ct::select(normalizer > 0.0f, normalizer,
                            ct::full<decltype(normalizer)>(1.0f));
    auto delta =
        ct::reshape(delta_view.load_masked(query_tile), ct::shape{32_ic, 1_ic});
    auto probability =
        ct::select(CausalMask(query_tile, key_tile, context_length),
                   ct::exp(score * scale - maximum) / normalizer,
                   ct::zeros<decltype(score)>());
    auto d_score = probability * (d_probability - delta) * scale;
    key_gradient =
        ct::mma(ct::transpose(d_score),
                ct::element_cast<float>(q.load_masked(query_tile, output_tile)),
                key_gradient);
    value_gradient =
        ct::mma(ct::transpose(probability),
                d_out.load_masked(query_tile, output_tile), value_gradient);
  }
  d_k.store_masked(key_gradient, key_tile, output_tile);
  d_v.store_masked(value_gradient, key_tile, output_tile);
}

}  // namespace

absl::StatusOr<std::unique_ptr<AttentionLayer>> AttentionLayer::Create(
    cuda::Executor& executor, int context_length, int num_heads,
    int embedding_dim, DataType data_type) {
  RETURN_IF_ERROR(internal::ValidateComputeType(data_type));
  if (context_length <= 0 || num_heads <= 0 || embedding_dim <= 0) {
    return absl::InvalidArgumentError(
        "attention dimensions must all be positive");
  }
  if (embedding_dim % num_heads != 0) {
    return absl::InvalidArgumentError(
        "embedding_dim must be divisible by num_heads");
  }
  // Head width is a logical extent. Every tiled Q/K/V access masks its tail;
  // neither the dot-product scale nor the model storage includes padded lanes.
  return absl::WrapUnique(new AttentionLayer(
      executor, context_length, num_heads, embedding_dim, data_type));
}

absl::StatusOr<FwdResult> AttentionLayer::fwd_impl(
    cuda::Executor& executor, absl::Span<const Buffer> inputs,
    LayerHooks* hooks) const {
  RETURN_IF_ERROR(
      internal::ValidateExecutor(executor_, executor, "AttentionLayer"));
  if (inputs.size() != 1)
    return absl::InvalidArgumentError(
        "AttentionLayer fwd expects packed Q/K/V");
  ASSIGN_OR_RETURN(int rows, internal::ActivationRows(
                                 executor, inputs[0], 3 * embedding_dim_,
                                 output_type_, "attention packed Q/K/V input"));
  if (rows % context_length_ != 0)
    return absl::InvalidArgumentError(
        "attention rows must be divisible by context_length");
  AttentionProbabilityLayout probability_layout{};
  const bool inspect_probabilities =
      hooks != nullptr &&
      static_cast<bool>(hooks->attention_probabilities_hook);
  if (inspect_probabilities) {
    ASSIGN_OR_RETURN(probability_layout,
                     ProbabilityLayout(rows, context_length_, num_heads_));
  }
  ASSIGN_OR_RETURN(
      auto output,
      Buffer::Allocate(executor,
                       static_cast<size_t>(rows) * embedding_dim_ *
                           internal::ActivationElementBytes(output_type_)));
  // Keep forward's softmax statistics instead of rescanning keys in backward.
  // Two floats per row/head are linear in context length and independent of
  // head width. Separate buffers give contiguous per-sequence/head tile loads.
  const size_t stats_bytes =
      static_cast<size_t>(rows) * num_heads_ * sizeof(float);
  ASSIGN_OR_RETURN(auto maxima, Buffer::Allocate(executor, stats_bytes));
  ASSIGN_OR_RETURN(auto normalizers, Buffer::Allocate(executor, stats_bytes));
  const int head_dimension = embedding_dim_ / num_heads_;
  const int blocks =
      (rows / context_length_) * num_heads_ *
      ((context_length_ + kAttentionRows - 1) / kAttentionRows) *
      ((head_dimension + kAttentionChannels - 1) / kAttentionChannels);
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dimension));
  if (output_type_ == DataType::BF16) {
    FlashAttentionForwardKernel<__nv_bfloat16>
        <<<blocks, 1, 0, executor.stream()>>>(
            static_cast<const __nv_bfloat16*>(inputs[0].data()),
            context_length_, num_heads_, embedding_dim_, scale,
            static_cast<__nv_bfloat16*>(output.data()),
            static_cast<float*>(maxima.data()),
            static_cast<float*>(normalizers.data()));
  } else {
    FlashAttentionForwardKernel<float><<<blocks, 1, 0, executor.stream()>>>(
        static_cast<const float*>(inputs[0].data()), context_length_,
        num_heads_, embedding_dim_, scale, static_cast<float*>(output.data()),
        static_cast<float*>(maxima.data()),
        static_cast<float*>(normalizers.data()));
  }
  RETURN_IF_ERROR(cuda::CudaStatus(cudaGetLastError(),
                                   "FlashAttentionForwardKernel launch"));
  if (inspect_probabilities) {
    ASSIGN_OR_RETURN(auto probabilities,
                     Buffer::Allocate(executor, probability_layout.bytes));
    if (output_type_ == DataType::BF16) {
      AttentionProbabilitiesKernel<__nv_bfloat16>
          <<<probability_layout.blocks, 1, 0, executor.stream()>>>(
              static_cast<const __nv_bfloat16*>(inputs[0].data()),
              static_cast<const float*>(maxima.data()),
              static_cast<const float*>(normalizers.data()), context_length_,
              num_heads_, embedding_dim_, scale,
              static_cast<float*>(probabilities.data()));
    } else {
      AttentionProbabilitiesKernel<float>
          <<<probability_layout.blocks, 1, 0, executor.stream()>>>(
              static_cast<const float*>(inputs[0].data()),
              static_cast<const float*>(maxima.data()),
              static_cast<const float*>(normalizers.data()), context_length_,
              num_heads_, embedding_dim_, scale,
              static_cast<float*>(probabilities.data()));
    }
    RETURN_IF_ERROR(cuda::CudaStatus(cudaGetLastError(),
                                     "AttentionProbabilitiesKernel launch"));
    const ActivationType probability_type{
        DataType::FP32,
        {rows / context_length_, num_heads_, context_length_, context_length_}};
    RETURN_IF_ERROR(hooks->attention_probabilities_hook(
        executor, name(), probability_type, probabilities));
  }
  BackwardState state;
  state.intermediates = {inputs[0], output, std::move(maxima),
                         std::move(normalizers)};
  return FwdResult{{std::move(output)}, std::move(state)};
}

absl::StatusOr<BufferVec> AttentionLayer::bwd_impl(
    cuda::Executor& executor, absl::Span<const Buffer> output_gradients,
    BackwardState state, LayerHooks*) {
  RETURN_IF_ERROR(
      internal::ValidateExecutor(executor_, executor, "AttentionLayer"));
  if (output_gradients.size() != 1 || state.intermediates.size() != 4)
    return absl::InvalidArgumentError(
        "AttentionLayer bwd received an incompatible gradient or state");
  ASSIGN_OR_RETURN(int rows, internal::MatrixRows(executor, output_gradients[0],
                                                  embedding_dim_,
                                                  "attention output gradient"));
  if (rows % context_length_ != 0)
    return absl::InvalidArgumentError(
        "attention gradient rows must be divisible by context_length");
  RETURN_IF_ERROR(internal::ValidateBuffer(
      executor, state.intermediates[0],
      static_cast<size_t>(rows) * 3 * embedding_dim_ *
          internal::ActivationElementBytes(output_type_),
      "attention saved Q/K/V"));
  RETURN_IF_ERROR(internal::ValidateBuffer(
      executor, state.intermediates[1],
      static_cast<size_t>(rows) * embedding_dim_ *
          internal::ActivationElementBytes(output_type_),
      "attention saved output"));
  const size_t stats_bytes =
      static_cast<size_t>(rows) * num_heads_ * sizeof(float);
  RETURN_IF_ERROR(internal::ValidateBuffer(
      executor, state.intermediates[2], stats_bytes, "attention saved maxima"));
  RETURN_IF_ERROR(internal::ValidateBuffer(executor, state.intermediates[3],
                                           stats_bytes,
                                           "attention saved normalizers"));
  ASSIGN_OR_RETURN(
      auto qkv_gradient,
      Buffer::Allocate(executor, static_cast<size_t>(rows) * 3 *
                                     embedding_dim_ * sizeof(float)));
  ASSIGN_OR_RETURN(auto deltas, Buffer::Allocate(executor, stats_bytes));
  const int head_dimension = embedding_dim_ / num_heads_;
  const int blocks =
      (rows / context_length_) * num_heads_ *
      ((context_length_ + kAttentionRows - 1) / kAttentionRows) *
      ((head_dimension + kAttentionChannels - 1) / kAttentionChannels);
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dimension));
  const auto* maxima = static_cast<const float*>(state.intermediates[2].data());
  const auto* normalizers =
      static_cast<const float*>(state.intermediates[3].data());
  if (output_type_ == DataType::BF16) {
    FlashAttentionQueryBackwardKernel<__nv_bfloat16>
        <<<blocks, 1, 0, executor.stream()>>>(
            static_cast<const __nv_bfloat16*>(state.intermediates[0].data()),
            static_cast<const __nv_bfloat16*>(state.intermediates[1].data()),
            static_cast<const float*>(output_gradients[0].data()), maxima,
            normalizers, context_length_, num_heads_, embedding_dim_, scale,
            static_cast<float*>(qkv_gradient.data()),
            static_cast<float*>(deltas.data()));
  } else {
    FlashAttentionQueryBackwardKernel<float>
        <<<blocks, 1, 0, executor.stream()>>>(
            static_cast<const float*>(state.intermediates[0].data()),
            static_cast<const float*>(state.intermediates[1].data()),
            static_cast<const float*>(output_gradients[0].data()), maxima,
            normalizers, context_length_, num_heads_, embedding_dim_, scale,
            static_cast<float*>(qkv_gradient.data()),
            static_cast<float*>(deltas.data()));
  }
  RETURN_IF_ERROR(cuda::CudaStatus(cudaGetLastError(),
                                   "FlashAttentionQueryBackwardKernel launch"));
  // Same-stream ordering makes every delta visible before the key-owned pass.
  if (output_type_ == DataType::BF16) {
    FlashAttentionKeyValueBackwardKernel<__nv_bfloat16>
        <<<blocks, 1, 0, executor.stream()>>>(
            static_cast<const __nv_bfloat16*>(state.intermediates[0].data()),
            static_cast<const float*>(output_gradients[0].data()), maxima,
            normalizers, static_cast<const float*>(deltas.data()),
            context_length_, num_heads_, embedding_dim_, scale,
            static_cast<float*>(qkv_gradient.data()));
  } else {
    FlashAttentionKeyValueBackwardKernel<float>
        <<<blocks, 1, 0, executor.stream()>>>(
            static_cast<const float*>(state.intermediates[0].data()),
            static_cast<const float*>(output_gradients[0].data()), maxima,
            normalizers, static_cast<const float*>(deltas.data()),
            context_length_, num_heads_, embedding_dim_, scale,
            static_cast<float*>(qkv_gradient.data()));
  }
  RETURN_IF_ERROR(cuda::CudaStatus(
      cudaGetLastError(), "FlashAttentionKeyValueBackwardKernel launch"));
  return BufferVec{std::move(qkv_gradient)};
}

}  // namespace pluto::llm
