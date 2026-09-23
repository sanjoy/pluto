#include "src/llm/experiments/one_shot_memorizer/feature_capture.h"

#include <cuda_runtime_api.h>

#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include "src/cuda/page_locked_host_array.h"
#include "src/llm/extract_top1_ids.h"
#include "src/llm/layer_hooks.h"
#include "src/util/status_macros.h"

namespace pluto::llm::one_shot_memorizer {
namespace {

float ExpandBf16(uint16_t word) {
  return std::bit_cast<float>(uint32_t{word} << 16);
}

// CUDA's finite float -> BF16 conversion rounds to nearest, ties to even.
float RoundBf16(float value) {
  uint32_t word = std::bit_cast<uint32_t>(value);
  word += 0x7fff + ((word >> 16) & 1);
  return ExpandBf16(static_cast<uint16_t>(word >> 16));
}

template <typename T>
absl::StatusOr<cuda::PageLockedHostArray<T>> QueueCopyD2H(
    cuda::Executor& executor, const Buffer& source) {
  if (&source.executor() != &executor || source.size_bytes() % sizeof(T))
    return absl::InvalidArgumentError("invalid feature capture buffer");
  ASSIGN_OR_RETURN(auto host, cuda::PageLockedHostArray<T>::Allocate(
                                  executor, source.size_bytes() / sizeof(T)));
  RETURN_IF_ERROR(cuda::CudaStatus(
      cudaMemcpyAsync(host.data(), source.data(), source.size_bytes(),
                      cudaMemcpyDeviceToHost, executor.stream()),
      "copy readout features to host"));
  return host;
}
}  // namespace

absl::StatusOr<ReadoutFeatures> CaptureGpt2ReadoutBatch(
    cuda::Executor& executor, const Layer& model, const DataBatch& batch,
    int width, int vocabulary_size) {
  ASSIGN_OR_RETURN(const int rows, batch.token_count());
  ASSIGN_OR_RETURN(const int supervised_rows, batch.loss_row_count());
  if (width <= 0 || vocabulary_size <= 0 ||
      static_cast<size_t>(rows) >
          std::numeric_limits<size_t>::max() / sizeof(uint16_t) / width ||
      batch.inputs.size_bytes() != static_cast<size_t>(rows) * sizeof(int) ||
      batch.targets.size_bytes() != batch.inputs.size_bytes() ||
      &batch.inputs.executor() != &executor ||
      &batch.targets.executor() != &executor)
    return absl::InvalidArgumentError(
        "invalid readout capture batch/dimensions");
  const auto input_types = model.input_types();
  const auto output_types = model.output_types();
  const ActivationType input_type(
      DataType::INT32,
      {ActivationType::kBatchDimension, batch.sequence_length});
  if (input_types.size() != 1 || input_types[0] != input_type ||
      output_types.size() != 1 || output_types[0].data_type() != DataType::FP32)
    return absl::InvalidArgumentError(
        "readout requires INT32 input and FP32 logits");
  const auto shape = output_types[0].dimensions();
  if (shape.size() != 3 || shape[0] != ActivationType::kBatchDimension ||
      shape[1] != batch.sequence_length || shape[2] < vocabulary_size ||
      static_cast<uint64_t>(shape[2]) >
          std::numeric_limits<size_t>::max() / sizeof(float) / rows)
    return absl::InvalidArgumentError("invalid declared logit shape");

  std::vector<std::string> scopes;
  std::optional<Buffer> features;
  LayerHooks hooks;
  hooks.enter_combinator = [&](cuda::Executor&, absl::string_view name) {
    scopes.emplace_back(name);
    return absl::OkStatus();
  };
  hooks.exit_combinator = [&](cuda::Executor&) {
    if (scopes.empty())
      return absl::InternalError("unbalanced feature capture scopes");
    scopes.pop_back();
    return absl::OkStatus();
  };
  hooks.activation_hook = [&](cuda::Executor&, absl::string_view name,
                              absl::Span<const ActivationType> types,
                              absl::Span<Buffer> outputs) {
    if (name != "LayerNormLayer" || scopes.size() != 1 || scopes[0] != "gpt2")
      return absl::OkStatus();
    const ActivationType expected(
        DataType::BF16,
        {ActivationType::kBatchDimension, batch.sequence_length, width});
    if (features || types.size() != 1 || types[0] != expected ||
        outputs.size() != 1 ||
        outputs[0].size_bytes() !=
            static_cast<size_t>(rows) * width * sizeof(uint16_t))
      return absl::InvalidArgumentError(
          "ambiguous or malformed final BF16 LayerNorm");
    features = outputs[0];
    return absl::OkStatus();
  };
  ASSIGN_OR_RETURN(auto forward, model.fwd(executor, {batch.inputs}, &hooks));
  if (!scopes.empty() || !features || forward.outputs.size() != 1 ||
      forward.outputs[0].size_bytes() !=
          static_cast<size_t>(rows) * shape[2] * sizeof(float))
    return absl::InvalidArgumentError(
        "model does not expose one final GPT-2 LayerNorm/head");
  ASSIGN_OR_RETURN(auto predictions,
                   ExtractTop1Ids(executor, forward.outputs[0], batch.targets,
                                  vocabulary_size));
  ASSIGN_OR_RETURN(auto host_features,
                   QueueCopyD2H<uint16_t>(executor, *features));
  ASSIGN_OR_RETURN(auto host_targets,
                   QueueCopyD2H<int>(executor, batch.targets));
  ASSIGN_OR_RETURN(auto host_predictions,
                   QueueCopyD2H<int>(executor, predictions));
  RETURN_IF_ERROR(executor.Synchronize());
  ReadoutFeatures result;
  result.width = width;
  for (int row = 0; row < rows; ++row) {
    const int target = host_targets[row];
    if (target == -1)
      continue;  // The supplied prompt and right padding are not predictions.
    if (target < 0 || target >= vocabulary_size || host_predictions[row] < 0)
      return absl::InvalidArgumentError(
          "invalid target or nonfinite baseline logits");
    result.labels.push_back(target);
    result.original_predictions.push_back(host_predictions[row]);
    result.sample_indices.push_back(row / batch.sequence_length);
    result.positions.push_back(row % batch.sequence_length);
    for (int dim = 0; dim < width; ++dim) {
      const float value =
          ExpandBf16(host_features[static_cast<size_t>(row) * width + dim]);
      if (!std::isfinite(value))
        return absl::InvalidArgumentError("nonfinite captured feature");
      result.values.push_back(value);
    }
  }
  if (batch.supervised_row_count >= 0 &&
      result.labels.size() != static_cast<size_t>(supervised_rows))
    return absl::InvalidArgumentError(
        "target mask disagrees with supervised row count");
  return result;
}

absl::StatusOr<std::vector<float>> CopyEffectiveBf16Readout(
    cuda::Executor& executor, const Buffer& embedding, int width,
    int vocabulary_size) {
  if (width <= 0 || vocabulary_size <= 0 ||
      static_cast<size_t>(vocabulary_size) >
          std::numeric_limits<size_t>::max() / sizeof(float) / width ||
      embedding.size_bytes() <
          static_cast<size_t>(vocabulary_size) * width * sizeof(float))
    return absl::InvalidArgumentError("invalid embedding shape for readout");
  ASSIGN_OR_RETURN(auto host, QueueCopyD2H<float>(executor, embedding));
  RETURN_IF_ERROR(executor.Synchronize());
  std::vector<float> result(static_cast<size_t>(vocabulary_size) * width);
  for (size_t i = 0; i < result.size(); ++i) {
    if (!std::isfinite(host[i]))
      return absl::InvalidArgumentError("nonfinite embedding weight");
    result[i] = RoundBf16(host[i]);
    if (!std::isfinite(result[i]))
      return absl::InvalidArgumentError("embedding overflows BF16");
  }
  return result;
}
}  // namespace pluto::llm::one_shot_memorizer
