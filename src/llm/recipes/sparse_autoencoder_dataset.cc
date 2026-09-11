#include "src/llm/recipes/sparse_autoencoder_dataset.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <filesystem>
#include <memory>
#include <utility>

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "src/cuda/buffer.h"
#include "src/cuda/executor.h"
#include "src/dataset/dataset.h"
#include "src/llm/checkpoint.h"
#include "src/llm/layer.h"
#include "src/util/status_macros.h"

namespace pluto::llm {
namespace {

absl::Status ValidateSourceBatch(cuda::Executor& executor,
                                 const DataBatch& batch) {
  ASSIGN_OR_RETURN(const int token_count, batch.token_count());
  const size_t token_bytes = static_cast<size_t>(token_count) * sizeof(int);
  if (batch.data.size_bytes() != 2 * token_bytes) {
    return absl::InvalidArgumentError(
        "activation dataset source must contain packed input and target "
        "tokens");
  }
  if (&batch.data.executor() != &executor) {
    return absl::InvalidArgumentError(
        "activation dataset source belongs to a different executor");
  }
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<std::unique_ptr<SparseAutoEncoderDataSetIterator>>
SparseAutoEncoderDataSetIterator::Create(
    cuda::Executor& executor, Layer& activation_generator,
    DataSetIterator& source,
    const std::filesystem::path& checkpoint_directory) {
  RETURN_IF_ERROR(
      ReadFromDirectory(executor, activation_generator, checkpoint_directory));
  return absl::WrapUnique(new SparseAutoEncoderDataSetIterator(
      executor, activation_generator, source));
}

absl::StatusOr<DataBatch> SparseAutoEncoderDataSetIterator::Next() {
  ASSIGN_OR_RETURN(DataBatch batch, source_.Next());
  RETURN_IF_ERROR(ValidateSourceBatch(executor_, batch));
  // Preserve sequence boundaries, not merely the flattened activation count.
  // Fixed-context generators must reject shorter samples instead of allowing
  // attention and position kernels to silently join neighboring samples.
  RETURN_IF_ERROR(
      activation_generator_.ValidateSequenceLength(batch.sequence_length));
  ASSIGN_OR_RETURN(const int token_count, batch.token_count());

  const size_t token_bytes = static_cast<size_t>(token_count) * sizeof(int);
  ASSIGN_OR_RETURN(auto tokens, Buffer::Allocate(executor_, token_bytes));
  RETURN_IF_ERROR(cuda::CudaStatus(
      cudaMemcpyAsync(tokens.data(), batch.data.data(), token_bytes,
                      cudaMemcpyDeviceToDevice, executor_.stream()),
      "cudaMemcpyAsync(activation dataset inputs)"));

  BufferVec inputs = {std::move(tokens)};
  ASSIGN_OR_RETURN(auto activations_fwd,
                   activation_generator_.fwd(executor_, inputs));
  auto activations = std::move(activations_fwd.output);

  if (&activations.executor() != &executor_) {
    return absl::InvalidArgumentError(
        "activation generator returned a buffer on a different executor");
  }
  return DataBatch{
      .data = std::move(activations),
      .batch_size = batch.batch_size,
      .sequence_length = batch.sequence_length,
  };
}

absl::Status SparseAutoEncoderDataSetIterator::Reset() {
  return source_.Reset();
}

}  // namespace pluto::llm
