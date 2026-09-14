#include "src/llm/layers/cross_entropy_loss.h"

#include <cuda_runtime.h>
#include <cuda_tile.h>

#include <cstddef>
#include <memory>
#include <utility>

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "src/cuda/buffer.h"
#include "src/llm/layers/util.h"
#include "src/util/status_macros.h"

namespace pluto::llm {

using cuda::CudaStatus;
using internal::kDenseTile;
using internal::MatrixRows;
using internal::ValidateBuffer;

namespace {
constexpr int kBackwardVocabularyTile = 256;

__tile_global__ void CrossEntropyForwardKernel(
    const float* __restrict__ logits, const int* __restrict__ targets, int rows,
    int padded_vocab_size, float* __restrict__ losses,
    float* __restrict__ maxima, float* __restrict__ denominators) {
  namespace ct = ::cuda::tiles;
  using namespace ct::literals;

  auto logits_view = ct::partition_view{
      ct::tensor_span{logits, ct::extents{rows, padded_vocab_size}},
      ct::shape{1_ic, 16_ic}};
  auto target_view = ct::partition_view{
      ct::tensor_span{targets, ct::extents{rows}}, ct::shape{1_ic}};
  auto loss_view = ct::partition_view{
      ct::tensor_span{losses, ct::extents{rows}}, ct::shape{1_ic}};
  auto maximum_view = ct::partition_view{
      ct::tensor_span{maxima, ct::extents{rows}}, ct::shape{1_ic}};
  auto denominator_view = ct::partition_view{
      ct::tensor_span{denominators, ct::extents{rows}}, ct::shape{1_ic}};

  const int row = ct::bid().x;
  const int target = static_cast<int>(target_view.load(row));
  const int vocabulary_tiles = padded_vocab_size / kDenseTile;
  auto maximum = ct::full<ct::tile<float, ct::shape<1, 1>>>(-3.402823466e+38f);
  for (int tile = 0; tile < vocabulary_tiles; ++tile)
    maximum =
        ct::max(maximum, ct::reduce_max(logits_view.load(row, tile), 1_ic));
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
  maximum_view.store(ct::reshape(maximum, ct::shape{1_ic}), row);
  denominator_view.store(ct::reshape(denominator, ct::shape{1_ic}), row);
}

__tile_global__ void CrossEntropyBackwardKernel(
    const float* __restrict__ logits, const int* __restrict__ targets,
    const float* __restrict__ maxima, const float* __restrict__ denominators,
    int rows, int padded_vocab_size, float* __restrict__ logits_gradient) {
  namespace ct = ::cuda::tiles;
  using namespace ct::literals;

  auto logits_view = ct::partition_view{
      ct::tensor_span{logits, ct::extents{rows, padded_vocab_size}},
      ct::shape{1_ic, 256_ic}};
  auto target_view = ct::partition_view{
      ct::tensor_span{targets, ct::extents{rows}}, ct::shape{1_ic}};
  auto maximum_view = ct::partition_view{
      ct::tensor_span{maxima, ct::extents{rows}}, ct::shape{1_ic}};
  auto denominator_view = ct::partition_view{
      ct::tensor_span{denominators, ct::extents{rows}}, ct::shape{1_ic}};
  auto gradient_view = ct::partition_view{
      ct::tensor_span{logits_gradient, ct::extents{rows, padded_vocab_size}},
      ct::shape{1_ic, 256_ic}};

  const int vocabulary_tiles =
      (padded_vocab_size + kBackwardVocabularyTile - 1) /
      kBackwardVocabularyTile;
  const int block = ct::bid().x;
  const int row = block / vocabulary_tiles;
  const int output_tile = block % vocabulary_tiles;
  const int target = static_cast<int>(target_view.load(row));
  auto maximum = ct::reshape(maximum_view.load(row), ct::shape{1_ic, 1_ic});
  auto denominator =
      ct::reshape(denominator_view.load(row), ct::shape{1_ic, 1_ic});
  // Vocabulary padding is 16-wide, so the last 256-wide tile can be partial.
  auto row_logits = logits_view.load_masked(row, output_tile);
  auto exponentials = ct::exp(row_logits - maximum);
  auto token_ids = ct::iota<ct::tile<int, ct::shape<1, 256>>>() +
                   output_tile * kBackwardVocabularyTile;
  auto one_hot = ct::element_cast<float>(token_ids == target);
  gradient_view.store_masked(
      (exponentials / denominator - one_hot) / static_cast<float>(rows), row,
      output_tile);
}

}  // namespace

absl::StatusOr<std::unique_ptr<CrossEntropyLossLayer>>
CrossEntropyLossLayer::Create(cuda::Executor& executor, int vocabulary_size,
                              DataType data_type, int sequence_length) {
  RETURN_IF_ERROR(internal::ValidateComputeType(data_type));
  if (sequence_length <= 0)
    return absl::InvalidArgumentError("sequence_length must be positive");
  if (vocabulary_size <= 0)
    return absl::InvalidArgumentError("vocabulary_size must be positive");
  return absl::WrapUnique(new CrossEntropyLossLayer(
      executor, vocabulary_size, internal::RoundUpToTile(vocabulary_size),
      data_type, sequence_length));
}

absl::Status CrossEntropyLossLayer::ValidateSequenceLength(
    int sequence_length) const {
  if (sequence_length != sequence_length_)
    return absl::InvalidArgumentError(
        "CrossEntropyLossLayer sequence length does not match its activation "
        "type");
  return absl::OkStatus();
}

absl::StatusOr<FwdResult> CrossEntropyLossLayer::fwd_impl(
    cuda::Executor& executor, absl::Span<const Buffer> inputs) const {
  BackwardState state;
  RETURN_IF_ERROR(
      internal::ValidateExecutor(executor_, executor, "CrossEntropyLossLayer"));
  if (inputs.size() != 2)
    return absl::InvalidArgumentError(
        "CrossEntropyLossLayer fwd expects logits, targets, and a non-null "
        "state");
  ASSIGN_OR_RETURN(int rows, MatrixRows(executor, inputs[0], padded_vocab_size_,
                                        "cross-entropy logits"));
  RETURN_IF_ERROR(ValidateBuffer(executor, inputs[1],
                                 static_cast<size_t>(rows) * sizeof(int),
                                 "cross-entropy targets"));
  ASSIGN_OR_RETURN(
      auto losses,
      Buffer::Allocate(executor, static_cast<size_t>(rows) * sizeof(float)));
  ASSIGN_OR_RETURN(
      auto maxima,
      Buffer::Allocate(executor, static_cast<size_t>(rows) * sizeof(float)));
  ASSIGN_OR_RETURN(
      auto denominators,
      Buffer::Allocate(executor, static_cast<size_t>(rows) * sizeof(float)));
  // Reuse the forward reduction results without repeating a vocabulary scan
  // for each output tile. Keep both values in FP32 and preserve the existing
  // reduction order and mean-loss gradient scaling.
  state.intermediates = {inputs[0], inputs[1], maxima, denominators};
  state.children.clear();
  CrossEntropyForwardKernel<<<rows, 1, 0, executor.stream()>>>(
      static_cast<const float*>(inputs[0].data()),
      static_cast<const int*>(inputs[1].data()), rows, padded_vocab_size_,
      static_cast<float*>(losses.data()), static_cast<float*>(maxima.data()),
      static_cast<float*>(denominators.data()));
  RETURN_IF_ERROR(
      CudaStatus(cudaGetLastError(), "CrossEntropyForwardKernel launch"));
  return FwdResult{{std::move(losses)}, std::move(state)};
}

absl::StatusOr<BufferVec> CrossEntropyLossLayer::bwd_impl(
    cuda::Executor& executor, absl::Span<const Buffer> output_gradients,
    BackwardState state) {
  RETURN_IF_ERROR(
      internal::ValidateExecutor(executor_, executor, "CrossEntropyLossLayer"));
  if (!output_gradients.empty() || state.intermediates.size() != 4)
    return absl::InvalidArgumentError(
        "terminal CrossEntropyLossLayer bwd expects no upstream gradient and "
        "a matching state");
  ASSIGN_OR_RETURN(
      int rows, MatrixRows(executor, state.intermediates[0], padded_vocab_size_,
                           "cross-entropy saved logits"));
  RETURN_IF_ERROR(ValidateBuffer(executor, state.intermediates[1],
                                 static_cast<size_t>(rows) * sizeof(int),
                                 "cross-entropy saved targets"));
  RETURN_IF_ERROR(ValidateBuffer(executor, state.intermediates[2],
                                 static_cast<size_t>(rows) * sizeof(float),
                                 "cross-entropy saved maxima"));
  RETURN_IF_ERROR(ValidateBuffer(executor, state.intermediates[3],
                                 static_cast<size_t>(rows) * sizeof(float),
                                 "cross-entropy saved denominators"));
  ASSIGN_OR_RETURN(
      auto logits_gradient,
      Buffer::Allocate(executor, state.intermediates[0].size_bytes()));
  const int vocabulary_tiles =
      (padded_vocab_size_ + kBackwardVocabularyTile - 1) /
      kBackwardVocabularyTile;
  CrossEntropyBackwardKernel<<<rows * vocabulary_tiles, 1, 0,
                               executor.stream()>>>(
      static_cast<const float*>(state.intermediates[0].data()),
      static_cast<const int*>(state.intermediates[1].data()),
      static_cast<const float*>(state.intermediates[2].data()),
      static_cast<const float*>(state.intermediates[3].data()), rows,
      padded_vocab_size_, static_cast<float*>(logits_gradient.data()));
  RETURN_IF_ERROR(
      CudaStatus(cudaGetLastError(), "CrossEntropyBackwardKernel launch"));
  return BufferVec{std::move(logits_gradient)};
}

}  // namespace pluto::llm
