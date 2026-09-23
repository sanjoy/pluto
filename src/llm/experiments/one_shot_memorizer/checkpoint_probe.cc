#include "src/llm/experiments/one_shot_memorizer/checkpoint_probe.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <vector>

#include "absl/status/status.h"
#include "src/cuda/buffer.h"
#include "src/cuda/page_locked_host_array.h"
#include "src/llm/extract_top1_ids.h"
#include "src/util/status_macros.h"

namespace pluto::llm::one_shot_memorizer {
namespace {

struct ProbeCase {
  size_t sentence;
  size_t prefix_length;
  int target;
};

struct ProbeShape {
  size_t context;
  size_t stride;
};

absl::StatusOr<ProbeShape> ValidateModel(cuda::Executor& executor,
                                         const Layer& model,
                                         int vocabulary_size) {
  const auto inputs = model.input_types();
  const auto outputs = model.output_types();
  if (inputs.size() != 1 || outputs.size() != 1)
    return absl::InvalidArgumentError(
        "context probe requires one model input and one output");
  RETURN_IF_ERROR(inputs[0].Validate());
  RETURN_IF_ERROR(outputs[0].Validate());
  const auto input_shape = inputs[0].dimensions();
  const auto output_shape = outputs[0].dimensions();
  if (inputs[0].data_type() != DataType::INT32 || input_shape.size() != 2 ||
      input_shape[0] != ActivationType::kBatchDimension ||
      input_shape[1] > std::numeric_limits<int>::max())
    return absl::InvalidArgumentError(
        "context probe input must be INT32 [batch, context]");
  if (outputs[0].data_type() != DataType::FP32 || output_shape.size() != 3 ||
      output_shape[0] != ActivationType::kBatchDimension ||
      output_shape[1] != input_shape[1] || output_shape[2] < vocabulary_size ||
      output_shape[2] > std::numeric_limits<int>::max())
    return absl::InvalidArgumentError(
        "context probe output must be FP32 [batch, context, "
        "padded_vocabulary]");
  for (const auto& weight : model.weights())
    if (&weight.executor() != &executor)
      return absl::InvalidArgumentError(
          "context probe model weights belong to another executor");
  return ProbeShape{.context = static_cast<size_t>(input_shape[1]),
                    .stride = static_cast<size_t>(output_shape[2])};
}

}  // namespace

absl::StatusOr<std::vector<ContextProbeResult>> ProbeContextWindows(
    cuda::Executor& executor, const Layer& model,
    absl::Span<const std::vector<int>> sentences, int vocabulary_size,
    int eos_token, int prompt_tokens, int batch_size,
    absl::Span<const int> windows, PrefixReplacement replacement) {
  if (vocabulary_size <= 0 || eos_token < 0 || eos_token >= vocabulary_size ||
      prompt_tokens <= 0 || batch_size <= 0)
    return absl::InvalidArgumentError(
        "context probe requires valid vocabulary, EOS, prompt and batch size");
  if (sentences.empty() || windows.empty())
    return absl::InvalidArgumentError(
        "context probe requires nonempty sentences and windows");
  if (replacement != PrefixReplacement::kEos &&
      replacement != PrefixReplacement::kOtherSentence)
    return absl::InvalidArgumentError("unknown context probe replacement");
  if (replacement == PrefixReplacement::kOtherSentence && sentences.size() < 2)
    return absl::InvalidArgumentError(
        "other-sentence replacement requires at least two sentences");
  for (int window : windows)
    if (window < -1)
      return absl::InvalidArgumentError(
          "context probe windows must be -1 (control) or nonnegative");
  ASSIGN_OR_RETURN(const auto shape,
                   ValidateModel(executor, model, vocabulary_size));
  size_t total_cases = 0;
  for (const auto& sentence : sentences) {
    if (sentence.size() < static_cast<size_t>(prompt_tokens) ||
        sentence.size() > shape.context)
      return absl::InvalidArgumentError(
          "context probe sentence length must be between prompt and context");
    for (int token : sentence)
      if (token < 0 || token >= vocabulary_size || token == eos_token)
        return absl::InvalidArgumentError(
            "context probe text token is outside vocabulary or equals EOS");
    const size_t count = sentence.size() - prompt_tokens + 1;
    if (count > std::numeric_limits<size_t>::max() - total_cases ||
        count > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) -
                    total_cases)
      return absl::InvalidArgumentError("context probe target count overflows");
    total_cases += count;
  }
  const size_t capacity =
      std::min(total_cases, static_cast<size_t>(batch_size));
  if (shape.context >
      static_cast<size_t>(std::numeric_limits<int>::max()) / capacity)
    return absl::InvalidArgumentError(
        "context probe batch rows exceed int range");
  const size_t rows = capacity * shape.context;
  if (rows > std::numeric_limits<size_t>::max() / sizeof(int) ||
      rows > std::numeric_limits<size_t>::max() / sizeof(float) / shape.stride)
    return absl::InvalidArgumentError(
        "context probe batch byte size overflows");
  const size_t logits_bytes = rows * shape.stride * sizeof(float);

  std::vector<ProbeCase> cases;
  cases.reserve(total_cases);
  for (size_t sentence_index = 0; sentence_index < sentences.size();
       ++sentence_index) {
    const auto& sentence = sentences[sentence_index];
    for (size_t length = prompt_tokens; length <= sentence.size(); ++length)
      cases.push_back(
          {.sentence = sentence_index,
           .prefix_length = length,
           .target = length == sentence.size() ? eos_token : sentence[length]});
  }

  ASSIGN_OR_RETURN(auto inputs,
                   cuda::PageLockedHostArray<int>::Allocate(executor, rows));
  ASSIGN_OR_RETURN(auto targets,
                   cuda::PageLockedHostArray<int>::Allocate(executor, rows));
  ASSIGN_OR_RETURN(auto predictions,
                   cuda::PageLockedHostArray<int>::Allocate(executor, rows));
  ASSIGN_OR_RETURN(auto device_inputs,
                   cuda::Buffer::Allocate(executor, inputs.size_bytes()));
  ASSIGN_OR_RETURN(auto device_targets,
                   cuda::Buffer::Allocate(executor, targets.size_bytes()));

  std::vector<ContextProbeResult> results;
  results.reserve(windows.size());
  for (int window : windows) {
    ContextProbeResult result{.window = window, .replacement = replacement};
    for (size_t begin = 0; begin < cases.size(); begin += capacity) {
      const size_t count = std::min(capacity, cases.size() - begin);
      // The preceding batch synchronized before these pinned arrays are
      // overwritten. All future positions and unused samples contain EOS;
      // only each case's original final prefix position has a scored target.
      std::fill(inputs.begin(), inputs.end(), eos_token);
      std::fill(targets.begin(), targets.end(), -1);
      for (size_t sample = 0; sample < count; ++sample) {
        const ProbeCase& probe = cases[begin + sample];
        const auto& sentence = sentences[probe.sentence];
        const size_t base = sample * shape.context;
        const size_t kept =
            window == -1
                ? probe.prefix_length
                : std::min(probe.prefix_length, static_cast<size_t>(window));
        const size_t replaced = probe.prefix_length - kept;
        if (replacement == PrefixReplacement::kOtherSentence && replaced != 0) {
          const auto& donor =
              sentences[(probe.sentence + 1) % sentences.size()];
          for (size_t position = 0; position < replaced; ++position)
            inputs[base + position] = donor[position % donor.size()];
        }
        std::copy(sentence.begin() + replaced,
                  sentence.begin() + probe.prefix_length,
                  inputs.begin() + base + replaced);
        targets[base + probe.prefix_length - 1] = probe.target;
      }
      RETURN_IF_ERROR(cuda::CudaStatus(
          cudaMemcpyAsync(device_inputs.data(), inputs.data(),
                          inputs.size_bytes(), cudaMemcpyHostToDevice,
                          executor.stream()),
          "upload context probe inputs"));
      RETURN_IF_ERROR(cuda::CudaStatus(
          cudaMemcpyAsync(device_targets.data(), targets.data(),
                          targets.size_bytes(), cudaMemcpyHostToDevice,
                          executor.stream()),
          "upload context probe target mask"));
      ASSIGN_OR_RETURN(auto forward, model.fwd(executor, {&device_inputs, 1}));
      forward.state = BackwardState{};
      if (forward.outputs.size() != 1 ||
          &forward.outputs[0].executor() != &executor ||
          forward.outputs[0].size_bytes() != logits_bytes)
        return absl::InvalidArgumentError(
            "context probe logits differ from declared shape or executor");
      ASSIGN_OR_RETURN(
          auto ids, ExtractTop1Ids(executor, forward.outputs[0], device_targets,
                                   vocabulary_size));
      RETURN_IF_ERROR(cuda::CudaStatus(
          cudaMemcpyAsync(predictions.data(), ids.data(),
                          predictions.size_bytes(), cudaMemcpyDeviceToHost,
                          executor.stream()),
          "download context probe predictions"));
      RETURN_IF_ERROR(executor.Synchronize());
      for (size_t sample = 0; sample < count; ++sample) {
        const auto& probe = cases[begin + sample];
        const int predicted =
            predictions[sample * shape.context + probe.prefix_length - 1];
        if (predicted != -2 && (predicted < 0 || predicted >= vocabulary_size))
          return absl::DataLossError(
              "context probe produced an invalid token ID");
        ++result.targets;
        result.correct += predicted == probe.target;
        result.nonfinite += predicted == -2;
      }
    }
    results.push_back(result);
  }
  return results;
}

}  // namespace pluto::llm::one_shot_memorizer
