#include "src/llm/layers.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda_tile.h>

#include <algorithm>
#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "src/gpu/buffer.h"

namespace pluto::llm {
namespace {

constexpr int kMatrixElementCount = kModelWidth * kModelWidth;
constexpr int kDenseTile = 16;
constexpr int kDenseTilesPerAxis = kModelWidth / kDenseTile;

absl::Status CudaStatus(cudaError_t error, const char* operation) {
  if (error == cudaSuccess) return absl::OkStatus();
  return absl::InternalError(
      absl::StrCat(operation, " failed: ", cudaGetErrorName(error), ": ",
                   cudaGetErrorString(error)));
}

absl::Status ValidateFp16(DataType data_type) {
  if (data_type == DataType::FP16) return absl::OkStatus();
  return absl::UnimplementedError(
      "FP8 requires an explicit scaling policy; this cuTile backend currently "
      "implements FP16 compute with FP32 master weights");
}

absl::Status ValidateBuffer(const Buffer& buffer, size_t expected_bytes,
                            cudaStream_t stream, const char* name) {
  if (buffer.size_bytes() != expected_bytes) {
    return absl::InvalidArgumentError(absl::StrCat(
        name, " has ", buffer.size_bytes(), " bytes; expected ",
        expected_bytes));
  }
  if (buffer.stream() != stream) {
    return absl::InvalidArgumentError(
        absl::StrCat(name, " belongs to a different CUDA stream"));
  }
  return absl::OkStatus();
}

__tile_global__ void EmbeddingForwardKernel(
    const int* __restrict__ tokens, const float* __restrict__ table,
    float* __restrict__ output) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  auto token_view = ct::partition_view{
      ct::tensor_span{tokens, ct::extents{256_ic}}, ct::shape{1_ic}};
  auto table_view = ct::partition_view{
      ct::tensor_span{table, ct::extents{256_ic, 256_ic}},
      ct::shape{1_ic, 256_ic}};
  auto output_view = ct::partition_view{
      ct::tensor_span{output, ct::extents{256_ic, 256_ic}},
      ct::shape{1_ic, 256_ic}};

  const int row = ct::bid().x;
  const int token = static_cast<int>(token_view.load(row));
  auto master = table_view.load(token, 0);
  auto fp16 = ct::element_cast<__half>(master);
  output_view.store(ct::element_cast<float>(fp16), row, 0);
}

__tile_global__ void EmbeddingBackwardKernel(
    const int* __restrict__ tokens, const float* __restrict__ output_gradient,
    float learning_rate, float* __restrict__ table) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  auto token_view = ct::partition_view{
      ct::tensor_span{tokens, ct::extents{256_ic}}, ct::shape{1_ic}};
  auto gradient_view = ct::partition_view{
      ct::tensor_span{output_gradient, ct::extents{256_ic, 256_ic}},
      ct::shape{1_ic, 256_ic}};

  const int row = ct::bid().x;
  const int token = static_cast<int>(token_view.load(row));
  auto offsets = ct::iota<ct::tile<int, ct::shape<1, 256>>>();
  auto pointers = table + token * kModelWidth + offsets;
  ct::atomic_sub<ct::memory_order::relaxed>(
      pointers, gradient_view.load(row, 0) * learning_rate);
}

__tile_global__ void CrossEntropyForwardKernel(
    const float* __restrict__ logits, const int* __restrict__ targets,
    float* __restrict__ losses) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  auto logits_view = ct::partition_view{
      ct::tensor_span{logits, ct::extents{256_ic, 256_ic}},
      ct::shape{1_ic, 256_ic}};
  auto target_view = ct::partition_view{
      ct::tensor_span{targets, ct::extents{256_ic}}, ct::shape{1_ic}};
  auto loss_view = ct::partition_view{
      ct::tensor_span{losses, ct::extents{256_ic}}, ct::shape{1_ic}};

  const int row = ct::bid().x;
  const int target = static_cast<int>(target_view.load(row));
  auto row_logits = logits_view.load(row, 0);
  auto maximum = ct::reduce_max(row_logits, 1_ic);
  auto exponentials = ct::exp(row_logits - maximum);
  auto denominator = ct::sum(exponentials, 1_ic);
  auto token_ids = ct::iota<ct::tile<int, ct::shape<1, 256>>>();
  auto one_hot = ct::element_cast<float>(token_ids == target);
  auto target_logit = ct::sum(row_logits * one_hot, 1_ic);
  auto loss = ct::log(denominator) + maximum - target_logit;
  loss_view.store(ct::reshape(loss, ct::shape{1_ic}), row);
}

__tile_global__ void CrossEntropyBackwardKernel(
    const float* __restrict__ logits, const int* __restrict__ targets,
    float* __restrict__ logits_gradient) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  auto logits_view = ct::partition_view{
      ct::tensor_span{logits, ct::extents{256_ic, 256_ic}},
      ct::shape{1_ic, 256_ic}};
  auto target_view = ct::partition_view{
      ct::tensor_span{targets, ct::extents{256_ic}}, ct::shape{1_ic}};
  auto gradient_view = ct::partition_view{
      ct::tensor_span{logits_gradient, ct::extents{256_ic, 256_ic}},
      ct::shape{1_ic, 256_ic}};

  const int row = ct::bid().x;
  const int target = static_cast<int>(target_view.load(row));
  auto row_logits = logits_view.load(row, 0);
  auto maximum = ct::reduce_max(row_logits, 1_ic);
  auto exponentials = ct::exp(row_logits - maximum);
  auto denominator = ct::sum(exponentials, 1_ic);
  auto token_ids = ct::iota<ct::tile<int, ct::shape<1, 256>>>();
  auto one_hot = ct::element_cast<float>(token_ids == target);
  gradient_view.store(
      (exponentials / denominator - one_hot) /
          static_cast<float>(kBatchSize),
      row, 0);
}

__tile_global__ void DenseForwardKernel(
    const float* __restrict__ input, const float* __restrict__ matrix,
    const float* __restrict__ bias, float* __restrict__ output) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  auto input_view = ct::partition_view{
      ct::tensor_span{input, ct::extents{256_ic, 256_ic}},
      ct::shape{16_ic, 16_ic}};
  auto matrix_view = ct::partition_view{
      ct::tensor_span{matrix, ct::extents{256_ic, 256_ic}},
      ct::shape{16_ic, 16_ic}};
  auto bias_view = ct::partition_view{
      ct::tensor_span{bias, ct::extents{256_ic}}, ct::shape{16_ic}};
  auto output_view = ct::partition_view{
      ct::tensor_span{output, ct::extents{256_ic, 256_ic}},
      ct::shape{16_ic, 16_ic}};

  const int block = ct::bid().x;
  const int batch_tile = block / kDenseTilesPerAxis;
  const int output_tile = block % kDenseTilesPerAxis;
  auto accumulator = ct::broadcast(bias_view.load(output_tile),
                                   ct::shape{16_ic, 16_ic});
  for (int inner_tile = 0; inner_tile < kDenseTilesPerAxis; ++inner_tile) {
    auto left =
        ct::element_cast<__half>(input_view.load(batch_tile, inner_tile));
    auto right =
        ct::element_cast<__half>(matrix_view.load(inner_tile, output_tile));
    accumulator = ct::mma(left, right, accumulator);
  }
  output_view.store(accumulator, batch_tile, output_tile);
}

__tile_global__ void DenseInputGradientKernel(
    const float* __restrict__ output_gradient,
    const float* __restrict__ matrix, float* __restrict__ input_gradient) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  auto gradient_view = ct::partition_view{
      ct::tensor_span{output_gradient, ct::extents{256_ic, 256_ic}},
      ct::shape{16_ic, 16_ic}};
  auto matrix_view = ct::partition_view{
      ct::tensor_span{matrix, ct::extents{256_ic, 256_ic}},
      ct::shape{16_ic, 16_ic}};
  auto input_gradient_view = ct::partition_view{
      ct::tensor_span{input_gradient, ct::extents{256_ic, 256_ic}},
      ct::shape{16_ic, 16_ic}};

  const int block = ct::bid().x;
  const int batch_tile = block / kDenseTilesPerAxis;
  const int input_tile = block % kDenseTilesPerAxis;
  auto accumulator = ct::zeros<ct::tile<float, ct::shape<16, 16>>>();
  for (int output_tile = 0; output_tile < kDenseTilesPerAxis;
       ++output_tile) {
    auto gradient = ct::element_cast<__half>(
        gradient_view.load(batch_tile, output_tile));
    auto matrix_transposed = ct::transpose(ct::element_cast<__half>(
        matrix_view.load(input_tile, output_tile)));
    accumulator = ct::mma(gradient, matrix_transposed, accumulator);
  }
  input_gradient_view.store(accumulator, batch_tile, input_tile);
}

__tile_global__ void DenseWeightUpdateKernel(
    const float* __restrict__ input,
    const float* __restrict__ output_gradient, float learning_rate,
    float* __restrict__ matrix) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  auto input_view = ct::partition_view{
      ct::tensor_span{input, ct::extents{256_ic, 256_ic}},
      ct::shape{16_ic, 16_ic}};
  auto gradient_view = ct::partition_view{
      ct::tensor_span{output_gradient, ct::extents{256_ic, 256_ic}},
      ct::shape{16_ic, 16_ic}};
  auto matrix_view = ct::partition_view{
      ct::tensor_span{matrix, ct::extents{256_ic, 256_ic}},
      ct::shape{16_ic, 16_ic}};

  const int block = ct::bid().x;
  const int input_tile = block / kDenseTilesPerAxis;
  const int output_tile = block % kDenseTilesPerAxis;
  auto accumulator = ct::zeros<ct::tile<float, ct::shape<16, 16>>>();
  for (int batch_tile = 0; batch_tile < kDenseTilesPerAxis; ++batch_tile) {
    auto input_transposed = ct::transpose(ct::element_cast<__half>(
        input_view.load(batch_tile, input_tile)));
    auto gradient = ct::element_cast<__half>(
        gradient_view.load(batch_tile, output_tile));
    accumulator = ct::mma(input_transposed, gradient, accumulator);
  }
  auto old_matrix = matrix_view.load(input_tile, output_tile);
  matrix_view.store(old_matrix - learning_rate * accumulator, input_tile,
                    output_tile);
}

__tile_global__ void DenseBiasUpdateKernel(
    const float* __restrict__ output_gradient, float learning_rate,
    float* __restrict__ bias) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  auto gradient_view = ct::partition_view{
      ct::tensor_span{output_gradient, ct::extents{256_ic, 256_ic}},
      ct::shape{16_ic, 16_ic}};
  auto bias_view = ct::partition_view{
      ct::tensor_span{bias, ct::extents{256_ic}}, ct::shape{16_ic}};

  const int output_tile = ct::bid().x;
  auto accumulator = ct::zeros<ct::tile<float, ct::shape<1, 16>>>();
  for (int batch_tile = 0; batch_tile < kDenseTilesPerAxis; ++batch_tile) {
    accumulator = accumulator +
                  ct::sum(gradient_view.load(batch_tile, output_tile), 0_ic);
  }
  auto old_bias = bias_view.load(output_tile);
  bias_view.store(old_bias - learning_rate *
                                 ct::reshape(accumulator, ct::shape{16_ic}),
                  output_tile);
}

}  // namespace

EmbeddingLookupLayer::EmbeddingLookupLayer(DataType data_type,
                                           float learning_rate,
                                           cudaStream_t stream, Buffer table)
    : data_type_(data_type),
      learning_rate_(learning_rate),
      stream_(stream),
      weights_{std::move(table)} {}

absl::StatusOr<std::unique_ptr<EmbeddingLookupLayer>>
EmbeddingLookupLayer::Create(DataType data_type, float learning_rate,
                             cudaStream_t stream) {
  if (auto status = ValidateFp16(data_type); !status.ok()) return status;
  if (learning_rate < 0.0f) {
    return absl::InvalidArgumentError("learning rate must be non-negative");
  }
  auto table = Buffer::Allocate(kMatrixElementCount * sizeof(float), stream);
  if (!table.ok()) return table.status();
  if (auto status = CudaStatus(cudaMemsetAsync(table->data(), 0,
                                               table->size_bytes(), stream),
                               "cudaMemsetAsync(embedding table)");
      !status.ok()) {
    return status;
  }
  return std::unique_ptr<EmbeddingLookupLayer>(new EmbeddingLookupLayer(
      data_type, learning_rate, stream, *std::move(table)));
}

absl::StatusOr<Buffer> EmbeddingLookupLayer::fwd(
    absl::Span<const Buffer> inputs, Tape* tape) {
  if (inputs.size() != 1 || tape == nullptr) {
    return absl::InvalidArgumentError(
        "EmbeddingLookupLayer fwd expects one input and a non-null tape");
  }
  if (auto status = ValidateBuffer(inputs[0], kBatchSize * sizeof(int),
                                   stream_, "embedding token input");
      !status.ok()) {
    return status;
  }
  auto output =
      Buffer::Allocate(kBatchSize * kModelWidth * sizeof(float), stream_);
  if (!output.ok()) return output.status();
  tape->intermediates = {inputs[0]};
  tape->children.clear();
  EmbeddingForwardKernel<<<kBatchSize, 1, 0, stream_>>>(
      static_cast<const int*>(inputs[0].data()),
      static_cast<const float*>(weights_[0].data()),
      static_cast<float*>(output->data()));
  if (auto status = CudaStatus(cudaGetLastError(),
                               "EmbeddingForwardKernel launch");
      !status.ok()) {
    return status;
  }
  return *std::move(output);
}

absl::StatusOr<Buffers> EmbeddingLookupLayer::bwd(
    absl::Span<const Buffer> output_gradients, Tape tape) {
  if (output_gradients.size() != 1 || tape.intermediates.size() != 1) {
    return absl::InvalidArgumentError(
        "EmbeddingLookupLayer bwd received an incompatible gradient or tape");
  }
  if (auto status =
          ValidateBuffer(output_gradients[0],
                         kBatchSize * kModelWidth * sizeof(float), stream_,
                         "embedding output gradient");
      !status.ok()) {
    return status;
  }
  EmbeddingBackwardKernel<<<kBatchSize, 1, 0, stream_>>>(
      static_cast<const int*>(tape.intermediates[0].data()),
      static_cast<const float*>(output_gradients[0].data()), learning_rate_,
      static_cast<float*>(weights_[0].data()));
  if (auto status = CudaStatus(cudaGetLastError(),
                               "EmbeddingBackwardKernel launch");
      !status.ok()) {
    return status;
  }
  // Integer token IDs are not differentiable.
  return Buffers{};
}

FullyConnectedLayer::FullyConnectedLayer(DataType data_type,
                                         float learning_rate,
                                         cudaStream_t stream, Buffer matrix,
                                         Buffer bias)
    : data_type_(data_type),
      learning_rate_(learning_rate),
      stream_(stream),
      weights_{std::move(matrix), std::move(bias)} {}

absl::StatusOr<std::unique_ptr<FullyConnectedLayer>>
FullyConnectedLayer::Create(DataType data_type, float learning_rate,
                            cudaStream_t stream) {
  if (auto status = ValidateFp16(data_type); !status.ok()) return status;
  if (learning_rate < 0.0f) {
    return absl::InvalidArgumentError("learning rate must be non-negative");
  }
  auto matrix = Buffer::Allocate(kMatrixElementCount * sizeof(float), stream);
  if (!matrix.ok()) return matrix.status();
  auto bias = Buffer::Allocate(kModelWidth * sizeof(float), stream);
  if (!bias.ok()) return bias.status();
  if (auto status = CudaStatus(cudaMemsetAsync(matrix->data(), 0,
                                               matrix->size_bytes(), stream),
                               "cudaMemsetAsync(dense matrix)");
      !status.ok()) {
    return status;
  }
  if (auto status = CudaStatus(
          cudaMemsetAsync(bias->data(), 0, bias->size_bytes(), stream),
          "cudaMemsetAsync(dense bias)");
      !status.ok()) {
    return status;
  }
  return std::unique_ptr<FullyConnectedLayer>(new FullyConnectedLayer(
      data_type, learning_rate, stream, *std::move(matrix), *std::move(bias)));
}

absl::Status FullyConnectedLayer::InitializeIdentity() {
  std::vector<float> identity(kMatrixElementCount, 0.0f);
  for (int index = 0; index < kModelWidth; ++index) {
    identity[index * kModelWidth + index] = 1.0f;
  }
  return CudaStatus(cudaMemcpyAsync(weights_[0].data(), identity.data(),
                                    weights_[0].size_bytes(),
                                    cudaMemcpyHostToDevice, stream_),
                    "cudaMemcpyAsync(identity matrix)");
}

absl::StatusOr<Buffer> FullyConnectedLayer::fwd(
    absl::Span<const Buffer> inputs, Tape* tape) {
  if (inputs.size() != 1 || tape == nullptr) {
    return absl::InvalidArgumentError(
        "FullyConnectedLayer fwd expects one input and a non-null tape");
  }
  if (auto status =
          ValidateBuffer(inputs[0],
                         kBatchSize * kModelWidth * sizeof(float), stream_,
                         "dense input");
      !status.ok()) {
    return status;
  }
  auto output =
      Buffer::Allocate(kBatchSize * kModelWidth * sizeof(float), stream_);
  if (!output.ok()) return output.status();
  tape->intermediates = {inputs[0]};
  tape->children.clear();
  DenseForwardKernel<<<kDenseTilesPerAxis * kDenseTilesPerAxis, 1, 0,
                       stream_>>>(
      static_cast<const float*>(inputs[0].data()),
      static_cast<const float*>(weights_[0].data()),
      static_cast<const float*>(weights_[1].data()),
      static_cast<float*>(output->data()));
  if (auto status = CudaStatus(cudaGetLastError(),
                               "DenseForwardKernel launch");
      !status.ok()) {
    return status;
  }
  return *std::move(output);
}

absl::StatusOr<Buffers> FullyConnectedLayer::bwd(
    absl::Span<const Buffer> output_gradients, Tape tape) {
  if (output_gradients.size() != 1 || tape.intermediates.size() != 1) {
    return absl::InvalidArgumentError(
        "FullyConnectedLayer bwd received an incompatible gradient or tape");
  }
  if (auto status =
          ValidateBuffer(output_gradients[0],
                         kBatchSize * kModelWidth * sizeof(float), stream_,
                         "dense output gradient");
      !status.ok()) {
    return status;
  }
  auto input_gradient =
      Buffer::Allocate(kBatchSize * kModelWidth * sizeof(float), stream_);
  if (!input_gradient.ok()) return input_gradient.status();
  DenseInputGradientKernel<<<kDenseTilesPerAxis * kDenseTilesPerAxis, 1, 0,
                             stream_>>>(
      static_cast<const float*>(output_gradients[0].data()),
      static_cast<const float*>(weights_[0].data()),
      static_cast<float*>(input_gradient->data()));
  if (learning_rate_ != 0.0f) {
    DenseWeightUpdateKernel<<<kDenseTilesPerAxis * kDenseTilesPerAxis, 1, 0,
                              stream_>>>(
        static_cast<const float*>(tape.intermediates[0].data()),
        static_cast<const float*>(output_gradients[0].data()), learning_rate_,
        static_cast<float*>(weights_[0].data()));
    DenseBiasUpdateKernel<<<kDenseTilesPerAxis, 1, 0, stream_>>>(
        static_cast<const float*>(output_gradients[0].data()), learning_rate_,
        static_cast<float*>(weights_[1].data()));
  }
  if (auto status = CudaStatus(cudaGetLastError(),
                               "dense backward kernel launch");
      !status.ok()) {
    return status;
  }
  return Buffers{*std::move(input_gradient)};
}

absl::StatusOr<std::unique_ptr<CrossEntropyLossLayer>>
CrossEntropyLossLayer::Create(DataType data_type, cudaStream_t stream) {
  if (auto status = ValidateFp16(data_type); !status.ok()) return status;
  return std::unique_ptr<CrossEntropyLossLayer>(
      new CrossEntropyLossLayer(data_type, stream));
}

absl::StatusOr<Buffer> CrossEntropyLossLayer::fwd(
    absl::Span<const Buffer> inputs, Tape* tape) {
  if (inputs.size() != 2 || tape == nullptr) {
    return absl::InvalidArgumentError(
        "CrossEntropyLossLayer fwd expects logits, targets, and a non-null "
        "tape");
  }
  if (auto status =
          ValidateBuffer(inputs[0],
                         kBatchSize * kVocabularySize * sizeof(float), stream_,
                         "cross-entropy logits");
      !status.ok()) {
    return status;
  }
  if (auto status = ValidateBuffer(inputs[1], kBatchSize * sizeof(int),
                                   stream_, "cross-entropy targets");
      !status.ok()) {
    return status;
  }
  auto losses = Buffer::Allocate(kBatchSize * sizeof(float), stream_);
  if (!losses.ok()) return losses.status();
  tape->intermediates = {inputs[0], inputs[1]};
  tape->children.clear();
  CrossEntropyForwardKernel<<<kBatchSize, 1, 0, stream_>>>(
      static_cast<const float*>(inputs[0].data()),
      static_cast<const int*>(inputs[1].data()),
      static_cast<float*>(losses->data()));
  if (auto status = CudaStatus(cudaGetLastError(),
                               "CrossEntropyForwardKernel launch");
      !status.ok()) {
    return status;
  }
  return *std::move(losses);
}

absl::StatusOr<Buffers> CrossEntropyLossLayer::bwd(
    absl::Span<const Buffer> output_gradients, Tape tape) {
  if (!output_gradients.empty() || tape.intermediates.size() != 2) {
    return absl::InvalidArgumentError(
        "terminal CrossEntropyLossLayer bwd expects no upstream gradient and "
        "a matching tape");
  }
  auto logits_gradient = Buffer::Allocate(
      kBatchSize * kVocabularySize * sizeof(float), stream_);
  if (!logits_gradient.ok()) return logits_gradient.status();
  CrossEntropyBackwardKernel<<<kBatchSize, 1, 0, stream_>>>(
      static_cast<const float*>(tape.intermediates[0].data()),
      static_cast<const int*>(tape.intermediates[1].data()),
      static_cast<float*>(logits_gradient->data()));
  if (auto status = CudaStatus(cudaGetLastError(),
                               "CrossEntropyBackwardKernel launch");
      !status.ok()) {
    return status;
  }
  return Buffers{*std::move(logits_gradient)};
}

ComposedLayer::ComposedLayer(DataType data_type,
                             std::vector<std::unique_ptr<Layer>> layers)
    : data_type_(data_type), layers_(std::move(layers)) {
  for (const auto& layer : layers_) {
    for (Buffer& weight : layer->weights()) weights_.push_back(weight);
  }
}

absl::StatusOr<Buffer> ComposedLayer::fwd(absl::Span<const Buffer> inputs,
                                           Tape* tape) {
  if (inputs.size() != 1 || tape == nullptr) {
    return absl::InvalidArgumentError(
        "ComposedLayer fwd expects one input and a non-null tape");
  }
  tape->intermediates.clear();
  tape->children.clear();
  Buffer activation = inputs.front();
  for (auto& layer : layers_) {
    Tape child_tape;
    Buffers child_inputs = {activation};
    auto output = layer->fwd(child_inputs, &child_tape);
    if (!output.ok()) return output.status();
    activation = *std::move(output);
    tape->children.push_back(std::move(child_tape));
  }
  return activation;
}

absl::StatusOr<Buffers> ComposedLayer::bwd(
    absl::Span<const Buffer> output_gradients, Tape tape) {
  if (output_gradients.size() != 1 || tape.children.size() != layers_.size()) {
    return absl::InvalidArgumentError(
        "ComposedLayer bwd received an incompatible gradient or tape");
  }
  Buffer gradient = output_gradients.front();
  for (size_t index = layers_.size(); index-- > 0;) {
    Buffers child_gradients = {gradient};
    auto input_gradients = layers_[index]->bwd(
        child_gradients, std::move(tape.children[index]));
    if (!input_gradients.ok()) return input_gradients.status();
    if (index == 0 && input_gradients->empty()) return Buffers{};
    if (input_gradients->size() != 1) {
      return absl::InternalError(
          "a composed unary layer returned multiple input gradients");
    }
    gradient = input_gradients->front();
  }
  return Buffers{gradient};
}

}  // namespace pluto::llm
