#include "src/llm/layers/cross_entropy_loss.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda_tile.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "src/cuda/buffer.h"
#include "src/llm/layers/internal.h"
#include "src/util/status_macros.h"

namespace pluto::llm {

using cuda::CudaStatus;
using internal::ElementCount;
using internal::kDenseTile;
using internal::MatrixRows;
using internal::TileCount;
using internal::ValidateBuffer;
using internal::ValidateFp16;
using internal::ValidateTiledExtent;

namespace {
__tile_global__ void CrossEntropyForwardKernel(const float* __restrict__ logits,
                                               const int* __restrict__ targets,
                                               int rows, int padded_vocab_size,
                                               float* __restrict__ losses) {
  namespace ct = ::cuda::tiles;
  using namespace ct::literals;

  auto logits_view = ct::partition_view{
      ct::tensor_span{logits, ct::extents{rows, padded_vocab_size}},
      ct::shape{1_ic, 16_ic}};
  auto target_view = ct::partition_view{
      ct::tensor_span{targets, ct::extents{rows}}, ct::shape{1_ic}};
  auto loss_view = ct::partition_view{
      ct::tensor_span{losses, ct::extents{rows}}, ct::shape{1_ic}};

  const int row = ct::bid().x;
  const int target = static_cast<int>(target_view.load(row));
  const int vocabulary_tiles = padded_vocab_size / kDenseTile;
  auto maximum = ct::full<ct::tile<float, ct::shape<1, 1>>>(-3.402823466e+38f);
  for (int tile = 0; tile < vocabulary_tiles; ++tile) {
    maximum =
        ct::max(maximum, ct::reduce_max(logits_view.load(row, tile), 1_ic));
  }
  auto denominator = ct::zeros<ct::tile<float, ct::shape<1, 1>>>();
  auto target_logit = ct::zeros<ct::tile<float, ct::shape<1, 1>>>();
  for (int tile = 0; tile < vocabulary_tiles; ++tile) {
    auto row_logits = logits_view.load(row, tile);
    denominator = denominator + ct::sum(ct::exp(row_logits - maximum), 1_ic);
    auto token_ids =
        ct::iota<ct::tile<int, ct::shape<1, 16>>>() + tile * kDenseTile;
    auto one_hot = ct::element_cast<float>(token_ids == target);
    target_logit = target_logit + ct::sum(row_logits * one_hot, 1_ic);
  }
  auto loss = ct::log(denominator) + maximum - target_logit;
  loss_view.store(ct::reshape(loss, ct::shape{1_ic}), row);
}

__tile_global__ void CrossEntropyBackwardKernel(
    const float* __restrict__ logits, const int* __restrict__ targets, int rows,
    int padded_vocab_size, float* __restrict__ logits_gradient) {
  namespace ct = ::cuda::tiles;
  using namespace ct::literals;

  auto logits_view = ct::partition_view{
      ct::tensor_span{logits, ct::extents{rows, padded_vocab_size}},
      ct::shape{1_ic, 16_ic}};
  auto target_view = ct::partition_view{
      ct::tensor_span{targets, ct::extents{rows}}, ct::shape{1_ic}};
  auto gradient_view = ct::partition_view{
      ct::tensor_span{logits_gradient, ct::extents{rows, padded_vocab_size}},
      ct::shape{1_ic, 16_ic}};

  const int vocabulary_tiles = padded_vocab_size / kDenseTile;
  const int block = ct::bid().x;
  const int row = block / vocabulary_tiles;
  const int output_tile = block % vocabulary_tiles;
  const int target = static_cast<int>(target_view.load(row));
  auto maximum = ct::full<ct::tile<float, ct::shape<1, 1>>>(-3.402823466e+38f);
  for (int tile = 0; tile < vocabulary_tiles; ++tile) {
    maximum =
        ct::max(maximum, ct::reduce_max(logits_view.load(row, tile), 1_ic));
  }
  auto denominator = ct::zeros<ct::tile<float, ct::shape<1, 1>>>();
  for (int tile = 0; tile < vocabulary_tiles; ++tile) {
    denominator = denominator +
                  ct::sum(ct::exp(logits_view.load(row, tile) - maximum), 1_ic);
  }
  auto row_logits = logits_view.load(row, output_tile);
  auto exponentials = ct::exp(row_logits - maximum);
  auto token_ids =
      ct::iota<ct::tile<int, ct::shape<1, 16>>>() + output_tile * kDenseTile;
  auto one_hot = ct::element_cast<float>(token_ids == target);
  gradient_view.store(
      (exponentials / denominator - one_hot) / static_cast<float>(rows), row,
      output_tile);
}

}  // namespace

absl::StatusOr<std::unique_ptr<CrossEntropyLossLayer>>
CrossEntropyLossLayer::Create(cuda::Executor& executor, int vocabulary_size,
                              DataType data_type) {
  RETURN_IF_ERROR(internal::ValidateComputeType(data_type));
  if (vocabulary_size <= 0)
    return absl::InvalidArgumentError("vocabulary_size must be positive");
  return absl::WrapUnique(new CrossEntropyLossLayer(
      executor, vocabulary_size, internal::RoundUpToTile(vocabulary_size),
      data_type));
}

absl::StatusOr<FwdResult> CrossEntropyLossLayer::fwd_impl(
    cuda::Executor& executor, absl::Span<const Buffer> inputs) const {
  BackwardState state;
  RETURN_IF_ERROR(
      internal::ValidateExecutor(executor_, executor, "CrossEntropyLossLayer"));
  if (inputs.size() != 2) {
    return absl::InvalidArgumentError(
        "CrossEntropyLossLayer fwd expects logits, targets, and a non-null "
        "state");
  }
  ASSIGN_OR_RETURN(int rows, MatrixRows(executor, inputs[0], padded_vocab_size_,
                                        "cross-entropy logits"));
  RETURN_IF_ERROR(ValidateBuffer(executor, inputs[1],
                                 static_cast<size_t>(rows) * sizeof(int),
                                 "cross-entropy targets"));
  ASSIGN_OR_RETURN(
      auto losses,
      Buffer::Allocate(executor, static_cast<size_t>(rows) * sizeof(float)));
  state.intermediates = {inputs[0], inputs[1]};
  state.children.clear();
  CrossEntropyForwardKernel<<<rows, 1, 0, executor.stream()>>>(
      static_cast<const float*>(inputs[0].data()),
      static_cast<const int*>(inputs[1].data()), rows, padded_vocab_size_,
      static_cast<float*>(losses.data()));
  RETURN_IF_ERROR(
      CudaStatus(cudaGetLastError(), "CrossEntropyForwardKernel launch"));
  return FwdResult{std::move(losses), std::move(state)};
}

absl::StatusOr<BufferVec> CrossEntropyLossLayer::bwd_impl(
    cuda::Executor& executor, absl::Span<const Buffer> output_gradients,
    BackwardState state) {
  RETURN_IF_ERROR(
      internal::ValidateExecutor(executor_, executor, "CrossEntropyLossLayer"));
  if (!output_gradients.empty() || state.intermediates.size() != 2) {
    return absl::InvalidArgumentError(
        "terminal CrossEntropyLossLayer bwd expects no upstream gradient and "
        "a matching state");
  }
  ASSIGN_OR_RETURN(
      int rows, MatrixRows(executor, state.intermediates[0], padded_vocab_size_,
                           "cross-entropy saved logits"));
  RETURN_IF_ERROR(ValidateBuffer(executor, state.intermediates[1],
                                 static_cast<size_t>(rows) * sizeof(int),
                                 "cross-entropy saved targets"));
  ASSIGN_OR_RETURN(
      auto logits_gradient,
      Buffer::Allocate(executor, state.intermediates[0].size_bytes()));
  CrossEntropyBackwardKernel<<<rows * TileCount(padded_vocab_size_), 1, 0,
                               executor.stream()>>>(
      static_cast<const float*>(state.intermediates[0].data()),
      static_cast<const int*>(state.intermediates[1].data()), rows,
      padded_vocab_size_, static_cast<float*>(logits_gradient.data()));
  RETURN_IF_ERROR(
      CudaStatus(cudaGetLastError(), "CrossEntropyBackwardKernel launch"));
  return BufferVec{std::move(logits_gradient)};
}

}  // namespace pluto::llm
