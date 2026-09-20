#pragma once

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "src/cuda/executor.h"
#include "src/dataset/dataset.h"
#include "src/llm/layer.h"

namespace pluto::llm {

// Checks a dataset-boundary signature without reading device memory. Batched
// types have shape [-2, sequence_length, ...]; unbatched tensors (e.g. an SAE
// decoder matrix) retain their exact dimensions. All -2 axes use this batch's
// one batch_size value for byte accounting, not independent sizes inferred
// from buffers. Symbolic type equality itself still matches -2 only with -2.
// Checks positivity, sample boundaries, and overflow of the implied byte sizes.
absl::Status ValidateBatchTypes(const DataBatch& batch,
                                absl::Span<const ActivationType> types,
                                absl::string_view description);

// Also checks buffer arity, byte sizes, and Executor ownership. This validates
// actual results against declared signatures without copies or synchronization.
// Buffer is untyped: equal-size FP32 and INT32 storage cannot be distinguished;
// the dataset/layer remains responsible for putting the declared dtype in it.
absl::Status ValidateBatchBuffers(cuda::Executor& executor,
                                  const DataBatch& batch,
                                  absl::Span<const Buffer> buffers,
                                  absl::Span<const ActivationType> types,
                                  absl::string_view description);

// A dataset input or target must be batched, unlike auxiliary layer outputs.
// In particular, a fixed extent equal to batch_size does not replace -2.
absl::Status ValidateBatchInput(cuda::Executor& executor,
                                const DataBatch& batch, const Buffer& buffer,
                                const ActivationType& type,
                                absl::string_view description);

// Preflight for Train/Evaluate's model -> loss connection. A model consumes
// one dataset input. Loss consumes every model output followed by the dataset
// target, and returns one FP32 value per token/activation row. Signature and
// dataset errors are rejected before any forward kernels run. The caller must
// still validate the actual forward results with ValidateBatchBuffers().
absl::Status ValidateTrainingBatch(cuda::Executor& executor, const Layer& model,
                                   const Layer& loss_layer,
                                   const DataBatch& batch);

}  // namespace pluto::llm
