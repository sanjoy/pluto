#include "src/llm/recipes/sparse_autoencoder_dataset.h"

#include <cstddef>
#include <filesystem>
#include <memory>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "src/cuda/executor.h"
#include "src/dataset/dataset.h"
#include "src/llm/checkpoint.h"
#include "src/llm/layer.h"
#include "src/util/status_macros.h"

namespace pluto::llm {
namespace {

absl::Status ValidateSourceBatch(cuda::Executor& executor,
                                 const TokenBatch& batch) {
  if (batch.batch_size <= 0) {
    return absl::InvalidArgumentError(
        "activation dataset source returned an empty batch");
  }
  const size_t expected_bytes =
      static_cast<size_t>(batch.batch_size) * sizeof(int);
  if (batch.tokens.size_bytes() != expected_bytes ||
      batch.targets.size_bytes() != expected_bytes) {
    return absl::InvalidArgumentError(
        "activation dataset source buffers do not match batch_size");
  }
  if (&batch.tokens.executor() != &executor ||
      &batch.targets.executor() != &executor) {
    return absl::InvalidArgumentError(
        "activation dataset source buffers belong to a different executor");
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
  return std::unique_ptr<SparseAutoEncoderDataSetIterator>(
      new SparseAutoEncoderDataSetIterator(executor, activation_generator,
                                           source));
}

absl::StatusOr<ActivationBatch> SparseAutoEncoderDataSetIterator::Next() {
  ASSIGN_OR_RETURN(TokenBatch batch, source_.Next());
  RETURN_IF_ERROR(ValidateSourceBatch(executor_, batch));

  Tape tape;
  BufferVec inputs = {batch.tokens};
  ASSIGN_OR_RETURN(auto activations,
                   activation_generator_.fwd(executor_, inputs, &tape));
  if (&activations.executor() != &executor_) {
    return absl::InvalidArgumentError(
        "activation generator returned a buffer on a different executor");
  }
  return ActivationBatch{
      .activations = std::move(activations),
      .batch_size = batch.batch_size,
  };
}

absl::Status SparseAutoEncoderDataSetIterator::Reset() {
  return source_.Reset();
}

}  // namespace pluto::llm
