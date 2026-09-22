#include "src/llm/experiments/weight_sensitivity/evaluation.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstddef>

#include "absl/status/status.h"
#include "src/cuda/page_locked_host_array.h"
#include "src/llm/batch_validation.h"
#include "src/llm/extract_top1_ids.h"
#include "src/util/status_macros.h"

namespace pluto::llm::weight_sensitivity {

absl::StatusOr<CompletionScores> EvaluateCompletions(
    cuda::Executor& executor, const Layer& model,
    PaddedLineDataSetIterator& data, int vocabulary_size) {
  if (data.options().shuffle)
    return absl::InvalidArgumentError(
        "completion evaluation requires an unshuffled dataset");
  if (vocabulary_size <= 0 || data.options().eos_token >= vocabulary_size)
    return absl::InvalidArgumentError("invalid evaluation vocabulary size");
  const ActivationType token_type(
      DataType::INT32,
      {ActivationType::kBatchDimension, data.options().context_length});
  if (model.input_types().size() != 1 || model.input_types()[0] != token_type)
    return absl::InvalidArgumentError(
        "completion model requires one INT32 [batch, context] input");
  const auto outputs = model.output_types();
  if (outputs.size() != 1 || outputs[0].data_type() != DataType::FP32 ||
      outputs[0].dimensions().size() != 3 ||
      outputs[0].dimensions()[0] != ActivationType::kBatchDimension ||
      outputs[0].dimensions()[1] != data.options().context_length ||
      outputs[0].dimensions()[2] < vocabulary_size)
    return absl::InvalidArgumentError(
        "completion model requires one FP32 [batch, context, vocabulary] "
        "output (vocabulary padding is allowed)");

  RETURN_IF_ERROR(data.Reset());
  // Reuse host staging across batches; the final batch may have fewer rows.
  const size_t capacity =
      std::min(data.sample_count(),
               static_cast<size_t>(data.options().batch_size)) *
      data.options().context_length;
  ASSIGN_OR_RETURN(auto ids, cuda::PageLockedHostArray<int32_t>::Allocate(
                                 executor, capacity));
  ASSIGN_OR_RETURN(auto targets, cuda::PageLockedHostArray<int32_t>::Allocate(
                                     executor, capacity));
  CompletionScores scores;
  scores.exact.reserve(data.sample_count());
  for (size_t i = 0; i < data.batches_per_epoch(); ++i) {
    ASSIGN_OR_RETURN(auto batch, data.Next());
    RETURN_IF_ERROR(ValidateBatchInput(executor, batch, batch.inputs,
                                       token_type, "completion inputs"));
    RETURN_IF_ERROR(ValidateBatchInput(executor, batch, batch.targets,
                                       token_type, "completion targets"));
    RETURN_IF_ERROR(ValidateBatchTypes(batch, outputs, "completion logits"));
    ASSIGN_OR_RETURN(auto forward, model.fwd(executor, {batch.inputs}));
    RETURN_IF_ERROR(ValidateBatchBuffers(executor, batch, forward.outputs,
                                         outputs, "completion logits"));
    ASSIGN_OR_RETURN(auto predictions,
                     ExtractTop1Ids(executor, forward.outputs[0], batch.targets,
                                    vocabulary_size));
    const size_t rows =
        static_cast<size_t>(batch.batch_size) * batch.sequence_length;
    if (rows > capacity)
      return absl::InternalError("dataset batch exceeds its declared size");
    RETURN_IF_ERROR(cuda::CudaStatus(
        cudaMemcpyAsync(ids.data(), predictions.data(), rows * sizeof(int32_t),
                        cudaMemcpyDeviceToHost, executor.stream()),
        "copy completion predictions"));
    RETURN_IF_ERROR(cuda::CudaStatus(
        cudaMemcpyAsync(targets.data(), batch.targets.data(),
                        rows * sizeof(int32_t), cudaMemcpyDeviceToHost,
                        executor.stream()),
        "copy completion targets"));
    RETURN_IF_ERROR(executor.Synchronize());
    int64_t counted = 0;
    for (int sample = 0; sample < batch.batch_size; ++sample) {
      bool exact = true;
      for (int position = 0; position < batch.sequence_length; ++position) {
        const size_t row =
            static_cast<size_t>(sample) * batch.sequence_length + position;
        if (targets[row] == -1)
          continue;  // Neither supplied prompt tokens nor padding are scored.
        if (targets[row] < 0 || targets[row] >= vocabulary_size)
          return absl::InvalidArgumentError(
              "evaluation target is outside the vocabulary");
        if (ids[row] < -2 || ids[row] == -1 || ids[row] >= vocabulary_size)
          return absl::InternalError("invalid top-1 prediction for scored row");
        ++counted;
        ++scores.scored_tokens;
        scores.nonfinite_rows += ids[row] == -2;
        const bool correct = ids[row] == targets[row];
        scores.token_errors += !correct;
        exact &= correct;
      }
      scores.exact.push_back(exact);
    }
    if (counted != batch.supervised_row_count)
      return absl::InternalError(
          "dataset supervised count disagrees with evaluation target mask");
  }
  if (scores.exact.size() != data.sample_count() ||
      scores.scored_tokens != data.supervised_row_count())
    return absl::InternalError(
        "completion evaluation did not visit every target exactly once");
  return scores;
}

}  // namespace pluto::llm::weight_sensitivity
