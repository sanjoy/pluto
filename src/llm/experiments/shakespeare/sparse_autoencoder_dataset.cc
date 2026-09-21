#include "src/llm/experiments/shakespeare/sparse_autoencoder_dataset.h"

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
#include "src/llm/batch_validation.h"
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
  const auto input_types = activation_generator_.input_types();
  const ActivationType token_type(
      DataType::INT32,
      {ActivationType::kBatchDimension, batch.sequence_length});
  if (input_types.size() != 1 || input_types.front() != token_type)
    return absl::InvalidArgumentError(
        "activation generator must accept one INT32 [batch, sequence] input "
        "matching the source sample boundaries");
  RETURN_IF_ERROR(ValidateBatchInput(executor_, batch, batch.inputs,
                                     input_types.front(), "generator input"));

  // We forward source sample metadata unchanged, so the generator must keep
  // both leading axes. Equal flattened byte counts do not imply equal sample
  // boundaries: two short sequences must not become one long sequence.
  const auto output_types = activation_generator_.output_types();
  if (output_types.size() != 1)
    return absl::InvalidArgumentError(
        "activation generator must declare exactly one output");
  const auto dimensions = output_types.front().dimensions();
  if (dimensions.size() < 2 ||
      dimensions[0] != ActivationType::kBatchDimension ||
      dimensions[1] != batch.sequence_length)
    return absl::InvalidArgumentError(
        "activation generator output must preserve [batch, sequence] axes");
  RETURN_IF_ERROR(ValidateBatchTypes(batch, output_types, "generator output"));

  BufferVec inputs = {batch.inputs};
  ASSIGN_OR_RETURN(auto activations_fwd,
                   activation_generator_.fwd(executor_, inputs));
  RETURN_IF_ERROR(ValidateBatchBuffers(executor_, batch,
                                       activations_fwd.outputs, output_types,
                                       "generator output"));
  auto activations = std::move(activations_fwd.outputs.front());

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
