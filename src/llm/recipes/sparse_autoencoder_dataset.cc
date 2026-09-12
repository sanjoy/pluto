#include "src/llm/recipes/sparse_autoencoder_dataset.h"

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
  if (batch.inputs.size_bytes() != token_bytes ||
      batch.targets.size_bytes() != token_bytes) {
    return absl::InvalidArgumentError(
        "activation dataset source inputs and targets must each contain "
        "batch_size * sequence_length int32 tokens");
  }
  if (&batch.inputs.executor() != &executor ||
      &batch.targets.executor() != &executor) {
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
  BufferVec inputs = {batch.inputs};
  ASSIGN_OR_RETURN(auto activations_fwd,
                   activation_generator_.fwd(executor_, inputs));
  if (activations_fwd.outputs.size() != 1) {
    return absl::InvalidArgumentError(
        "activation generator must return exactly one output");
  }
  auto activations = std::move(activations_fwd.outputs.front());

  if (&activations.executor() != &executor_) {
    return absl::InvalidArgumentError(
        "activation generator returned a buffer on a different executor");
  }
  return DataBatch{
      .inputs = activations,
      .targets = std::move(activations),
      .batch_size = batch.batch_size,
      .sequence_length = batch.sequence_length,
  };
}

absl::Status SparseAutoEncoderDataSetIterator::Reset() {
  return source_.Reset();
}

}  // namespace pluto::llm
