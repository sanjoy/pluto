#include "src/llm/batch_validation.h"

#include <cstddef>
#include <cstdint>
#include <limits>

#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "src/llm/layers/type_check_util.h"
#include "src/util/status_macros.h"

namespace pluto::llm {
namespace {

absl::Status Invalid(absl::string_view description, absl::string_view message) {
  return absl::InvalidArgumentError(absl::StrCat(description, ": ", message));
}

// Metadata-only sizing: using one supplied batch size is essential. Inferring
// a batch size separately from each allocation would accept mismatched inputs.
absl::StatusOr<size_t> BatchBytes(const DataBatch& batch,
                                  const ActivationType& type,
                                  absl::string_view description) {
  const auto type_status = type.Validate();
  if (!type_status.ok())
    return Invalid(description, type_status.message());
  const auto dims = type.dimensions();
  const bool batched =
      !dims.empty() && dims.front() == ActivationType::kBatchDimension;
  if (batched && (dims.size() < 2 || dims[1] != batch.sequence_length))
    return Invalid(description,
                   "declared sample shape must match batch.sequence_length");

  size_t bytes;
  switch (type.data_type()) {
    case DataType::FP8:
      bytes = 1;
      break;
    case DataType::FP16:
    case DataType::BF16:
      bytes = 2;
      break;
    case DataType::FP32:
    case DataType::INT32:
      bytes = 4;
      break;
    default:
      return Invalid(description, "unknown activation data type");
  }
  // Starting with element bytes also catches overflow for a tensor whose
  // element count fits size_t but whose complete allocation would not.
  for (int64_t dim : dims) {
    const uint64_t extent = static_cast<uint64_t>(
        dim == ActivationType::kBatchDimension ? batch.batch_size : dim);
    if (extent > std::numeric_limits<size_t>::max() / bytes)
      return Invalid(description, "activation byte size overflows size_t");
    bytes *= static_cast<size_t>(extent);
  }
  return bytes;
}

}  // namespace

absl::Status ValidateBatchTypes(const DataBatch& batch,
                                absl::Span<const ActivationType> types,
                                absl::string_view description) {
  RETURN_IF_ERROR(batch.loss_row_count().status());
  for (size_t i = 0; i < types.size(); ++i)
    RETURN_IF_ERROR(
        BatchBytes(batch, types[i], absl::StrCat(description, "[", i, "]"))
            .status());
  return absl::OkStatus();
}

absl::Status ValidateBatchBuffers(cuda::Executor& executor,
                                  const DataBatch& batch,
                                  absl::Span<const Buffer> buffers,
                                  absl::Span<const ActivationType> types,
                                  absl::string_view description) {
  RETURN_IF_ERROR(batch.loss_row_count().status());
  if (buffers.size() != types.size())
    return Invalid(description,
                   absl::StrCat("received ", buffers.size(),
                                " buffers; expected ", types.size()));
  for (size_t i = 0; i < types.size(); ++i) {
    const auto label = absl::StrCat(description, "[", i, "]");
    ASSIGN_OR_RETURN(const size_t bytes, BatchBytes(batch, types[i], label));
    if (&buffers[i].executor() != &executor)
      return Invalid(label, "buffer belongs to a different CUDA Executor");
    if (buffers[i].size_bytes() != bytes)
      return Invalid(label, absl::StrCat("buffer has ", buffers[i].size_bytes(),
                                         " bytes; expected ", bytes));
  }
  return absl::OkStatus();
}

absl::Status ValidateBatchInput(cuda::Executor& executor,
                                const DataBatch& batch, const Buffer& buffer,
                                const ActivationType& type,
                                absl::string_view description) {
  const auto dims = type.dimensions();
  if (dims.size() < 2 || dims.front() != ActivationType::kBatchDimension)
    return Invalid(description,
                   "dataset inputs and targets require a leading batch (-2) "
                   "and a sequence axis");
  return ValidateBatchBuffers(executor, batch, absl::MakeConstSpan(&buffer, 1),
                              absl::MakeConstSpan(&type, 1), description);
}

absl::Status ValidateTrainingBatch(cuda::Executor& executor, const Layer& model,
                                   const Layer& loss_layer,
                                   const DataBatch& batch) {
  RETURN_IF_ERROR(batch.loss_row_count().status());
  const auto inputs = model.input_types();
  const auto outputs = model.output_types();
  const auto loss_inputs = loss_layer.input_types();
  const auto loss_outputs = loss_layer.output_types();
  if (inputs.size() != 1)
    return Invalid(model.name(), "training model must declare one input");
  if (outputs.empty())
    return Invalid(model.name(),
                   "training model must declare at least one output");
  if (loss_inputs.empty() || loss_inputs.size() - 1 != outputs.size())
    return Invalid(loss_layer.name(),
                   "loss inputs must be model outputs followed by one target");
  RETURN_IF_ERROR(ValidateBatchInput(executor, batch, batch.inputs, inputs[0],
                                     absl::StrCat(model.name(), " input")));
  RETURN_IF_ERROR(ValidateBatchTypes(batch, outputs,
                                     absl::StrCat(model.name(), " outputs")));
  const auto connection = internal::ValidateTypeConnection(
      outputs, loss_inputs.subspan(0, outputs.size()));
  if (!connection.ok())
    return Invalid(absl::StrCat(model.name(), " -> ", loss_layer.name()),
                   connection.message());
  RETURN_IF_ERROR(
      ValidateBatchInput(executor, batch, batch.targets, loss_inputs.back(),
                         absl::StrCat(loss_layer.name(), " target")));
  const ActivationType loss_type(
      DataType::FP32, {ActivationType::kBatchDimension, batch.sequence_length});
  if (loss_outputs.size() != 1 || loss_outputs[0] != loss_type)
    return Invalid(
        loss_layer.name(),
        "loss must declare one FP32 [batch, sequence_length] output");
  return ValidateBatchTypes(batch, loss_outputs,
                            absl::StrCat(loss_layer.name(), " outputs"));
}

}  // namespace pluto::llm
