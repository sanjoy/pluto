#include "src/llm/experiments/one_shot_memorizer/causal_trace.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "src/cuda/buffer.h"
#include "src/cuda/page_locked_host_array.h"
#include "src/llm/extract_top1_ids.h"
#include "src/llm/layer_hooks.h"
#include "src/llm/layers/util.h"
#include "src/util/status_macros.h"

namespace pluto::llm::one_shot_memorizer {
namespace {

struct TraceCase {
  size_t sentence;
  size_t prefix_length;
  int target;
};

struct TraceShape {
  size_t context;
  size_t stride;
};

absl::StatusOr<TraceShape> ValidateModel(cuda::Executor& executor,
                                         const Layer& model,
                                         int vocabulary_size) {
  const auto inputs = model.input_types();
  const auto outputs = model.output_types();
  if (inputs.size() != 1 || outputs.size() != 1)
    return absl::InvalidArgumentError("causal trace requires one input/output");
  RETURN_IF_ERROR(inputs[0].Validate());
  RETURN_IF_ERROR(outputs[0].Validate());
  const auto input_shape = inputs[0].dimensions();
  const auto output_shape = outputs[0].dimensions();
  if (inputs[0].data_type() != DataType::INT32 || input_shape.size() != 2 ||
      input_shape[0] != ActivationType::kBatchDimension ||
      input_shape[1] > std::numeric_limits<int>::max())
    return absl::InvalidArgumentError(
        "causal trace input must be INT32 [batch, context]");
  if (outputs[0].data_type() != DataType::FP32 || output_shape.size() != 3 ||
      output_shape[0] != ActivationType::kBatchDimension ||
      output_shape[1] != input_shape[1] || output_shape[2] < vocabulary_size ||
      output_shape[2] > std::numeric_limits<int>::max())
    return absl::InvalidArgumentError(
        "causal trace output must be FP32 [batch, context, padded_vocabulary]");
  for (const auto& weight : model.weights())
    if (&weight.executor() != &executor)
      return absl::InvalidArgumentError(
          "causal trace model weights belong to another executor");
  return TraceShape{.context = static_cast<size_t>(input_shape[1]),
                    .stride = static_cast<size_t>(output_shape[2])};
}

struct CapturedActivation {
  ActivationType type;
  cuda::Buffer buffer;
  size_t row_bytes;
};

// A fresh instance is used for every forward pass: occurrence counters and
// scope stacks cannot leak across clean, corrupt, or single-site patch runs.
class TraceHooks {
 public:
  TraceHooks(cuda::Executor& executor, absl::Span<const CausalTraceSite> sites,
             size_t capacity, size_t context,
             absl::Span<const size_t> query_rows,
             const CapturedActivation* donor = nullptr)
      : executor_(executor),
        sites_(sites),
        rows_(capacity * context),
        context_(context),
        query_rows_(query_rows),
        donor_(donor),
        matches_(sites.size(), 0),
        captured_(sites.size()) {
    hooks_.enter_combinator = [this](cuda::Executor& executor,
                                     absl::string_view name) {
      RETURN_IF_ERROR(CheckExecutor(executor));
      scopes_.emplace_back(name);
      return absl::OkStatus();
    };
    hooks_.exit_combinator = [this](cuda::Executor& executor) {
      RETURN_IF_ERROR(CheckExecutor(executor));
      if (scopes_.empty())
        return absl::FailedPreconditionError("causal trace scope underflow");
      scopes_.pop_back();
      return absl::OkStatus();
    };
    hooks_.activation_hook = [this](cuda::Executor& executor,
                                    absl::string_view name,
                                    absl::Span<const ActivationType> types,
                                    absl::Span<cuda::Buffer> activations) {
      return Observe(executor, name, types, activations);
    };
  }

  TraceHooks(const TraceHooks&) = delete;
  TraceHooks& operator=(const TraceHooks&) = delete;

  LayerHooks& hooks() { return hooks_; }
  const CapturedActivation& captured(size_t index) const {
    return *captured_[index];
  }

  absl::Status Finish() const {
    if (!scopes_.empty())
      return absl::FailedPreconditionError(
          "causal trace scopes were not closed");
    for (size_t index = 0; index < sites_.size(); ++index)
      if (!captured_[index].has_value())
        return absl::NotFoundError(absl::StrCat(
            "causal trace site did not match: ", sites_[index].layer_name,
            " in scope ", sites_[index].enclosing_scope, " at occurrence ",
            sites_[index].occurrence));
    return absl::OkStatus();
  }

 private:
  absl::Status CheckExecutor(const cuda::Executor& executor) const {
    if (&executor != &executor_)
      return absl::InvalidArgumentError(
          "causal trace hook uses another executor");
    return absl::OkStatus();
  }

  absl::StatusOr<size_t> ValidateActivation(
      absl::Span<const ActivationType> types,
      absl::Span<const cuda::Buffer> activations) const {
    if (types.size() != 1 || activations.size() != 1)
      return absl::InvalidArgumentError(
          "causal trace site must expose one activation");
    RETURN_IF_ERROR(types[0].Validate());
    const auto dimensions = types[0].dimensions();
    // Signatures describe physical storage. In particular, the legacy FP16
    // compute policy publishes FP32 here; BF16 occupies exactly two bytes.
    if ((types[0].data_type() != DataType::FP32 &&
         types[0].data_type() != DataType::BF16) ||
        dimensions.size() != 3 ||
        dimensions[0] != ActivationType::kBatchDimension ||
        dimensions[1] != static_cast<int64_t>(context_))
      return absl::InvalidArgumentError(
          "causal trace site must be FP32 or BF16 [batch, context, width]");
    if (&activations[0].executor() != &executor_)
      return absl::InvalidArgumentError(
          "causal trace activation belongs to another executor");
    const size_t element_bytes =
        internal::ActivationElementBytes(types[0].data_type());
    if (static_cast<uint64_t>(dimensions[2]) >
        std::numeric_limits<size_t>::max() / element_bytes)
      return absl::InvalidArgumentError(
          "causal trace activation width overflows");
    const size_t row_bytes = static_cast<size_t>(dimensions[2]) * element_bytes;
    if (rows_ > std::numeric_limits<size_t>::max() / row_bytes ||
        activations[0].size_bytes() != rows_ * row_bytes)
      return absl::InvalidArgumentError(
          "causal trace activation bytes differ from declared shape");
    return row_bytes;
  }

  absl::Status Observe(cuda::Executor& executor, absl::string_view name,
                       absl::Span<const ActivationType> types,
                       absl::Span<cuda::Buffer> activations) {
    RETURN_IF_ERROR(CheckExecutor(executor));
    for (size_t index = 0; index < sites_.size(); ++index) {
      const auto& site = sites_[index];
      if (name != site.layer_name ||
          (!site.enclosing_scope.empty() &&
           std::find(scopes_.begin(), scopes_.end(), site.enclosing_scope) ==
               scopes_.end()))
        continue;
      const size_t occurrence = matches_[index]++;
      if (site.occurrence == -1 && occurrence > 0)
        return absl::InvalidArgumentError(
            absl::StrCat("causal trace site is ambiguous: ", site.layer_name,
                         "; specify enclosing_scope and occurrence"));
      if (site.occurrence >= 0 &&
          occurrence != static_cast<size_t>(site.occurrence))
        continue;
      ASSIGN_OR_RETURN(const size_t row_bytes,
                       ValidateActivation(types, activations));
      if (donor_ != nullptr) {
        if (sites_.size() != 1 || donor_->type != types[0] ||
            donor_->row_bytes != row_bytes ||
            donor_->buffer.size_bytes() != activations[0].size_bytes() ||
            &donor_->buffer.executor() != &executor_)
          return absl::InvalidArgumentError(
              "causal trace donor shape, dtype, or executor differs");
        ASSIGN_OR_RETURN(
            auto replacement,
            cuda::Buffer::Allocate(executor_, activations[0].size_bytes()));
        RETURN_IF_ERROR(cuda::CudaStatus(
            cudaMemcpyAsync(replacement.data(), activations[0].data(),
                            replacement.size_bytes(), cudaMemcpyDeviceToDevice,
                            executor_.stream()),
            "clone causal trace recipient activation"));
        for (size_t row : query_rows_) {
          if (row >= rows_)
            return absl::InvalidArgumentError(
                "causal trace query row is outside activation");
          RETURN_IF_ERROR(cuda::CudaStatus(
              cudaMemcpyAsync(
                  static_cast<char*>(replacement.data()) + row * row_bytes,
                  static_cast<const char*>(donor_->buffer.data()) +
                      row * row_bytes,
                  row_bytes, cudaMemcpyDeviceToDevice, executor_.stream()),
              "patch causal trace final query row"));
        }
        activations[0] = std::move(replacement);
      }
      // Retain handles, not borrowed spans. All producers/consumers and patch
      // copies use one stream; the baseline storage is never modified.
      captured_[index].emplace(
          CapturedActivation{types[0], activations[0], row_bytes});
    }
    return absl::OkStatus();
  }

  cuda::Executor& executor_;
  absl::Span<const CausalTraceSite> sites_;
  size_t rows_;
  size_t context_;
  absl::Span<const size_t> query_rows_;
  const CapturedActivation* donor_;
  std::vector<size_t> matches_;
  std::vector<std::optional<CapturedActivation>> captured_;
  std::vector<std::string> scopes_;
  LayerHooks hooks_;
};

absl::StatusOr<std::vector<int>> RunForward(
    cuda::Executor& executor, const Layer& model, const cuda::Buffer& inputs,
    const cuda::Buffer& targets, int vocabulary_size, size_t logits_bytes,
    absl::Span<const size_t> query_rows, TraceHooks& hooks,
    cuda::PageLockedHostArray<int>& host_predictions) {
  ASSIGN_OR_RETURN(auto forward,
                   model.fwd(executor, {&inputs, 1}, &hooks.hooks()));
  RETURN_IF_ERROR(hooks.Finish());
  forward.state = BackwardState{};
  if (forward.outputs.size() != 1 ||
      &forward.outputs[0].executor() != &executor ||
      forward.outputs[0].size_bytes() != logits_bytes)
    return absl::InvalidArgumentError(
        "causal trace logits differ from declared shape or executor");
  ASSIGN_OR_RETURN(auto ids, ExtractTop1Ids(executor, forward.outputs[0],
                                            targets, vocabulary_size));
  RETURN_IF_ERROR(cuda::CudaStatus(
      cudaMemcpyAsync(host_predictions.data(), ids.data(),
                      host_predictions.size_bytes(), cudaMemcpyDeviceToHost,
                      executor.stream()),
      "download causal trace predictions"));
  RETURN_IF_ERROR(executor.Synchronize());
  std::vector<int> predictions;
  predictions.reserve(query_rows.size());
  for (size_t row : query_rows) {
    const int token = host_predictions[row];
    if (token != -2 && (token < 0 || token >= vocabulary_size))
      return absl::DataLossError(
          "causal trace produced an invalid predicted token");
    predictions.push_back(token);
  }
  return predictions;
}

}  // namespace

absl::StatusOr<std::vector<CausalTraceResult>> TracePrefixMediation(
    cuda::Executor& executor, const Layer& model,
    absl::Span<const std::vector<int>> sentences, int vocabulary_size,
    int eos_token, int prompt_tokens, int batch_size,
    absl::Span<const CausalTraceSite> sites, int retained_tokens) {
  if (vocabulary_size <= 0 || eos_token < 0 || eos_token >= vocabulary_size ||
      prompt_tokens <= 0 || batch_size <= 0 || retained_tokens < 0 ||
      sentences.size() < 2 || sites.empty())
    return absl::InvalidArgumentError(
        "causal trace requires valid dimensions, two sentences, and nonempty "
        "sites");
  for (size_t index = 0; index < sites.size(); ++index) {
    if (sites[index].layer_name.empty() || sites[index].occurrence < -1)
      return absl::InvalidArgumentError("invalid causal trace site selector");
    for (size_t earlier = 0; earlier < index; ++earlier)
      if (sites[earlier].layer_name == sites[index].layer_name &&
          sites[earlier].enclosing_scope == sites[index].enclosing_scope &&
          sites[earlier].occurrence == sites[index].occurrence)
        return absl::InvalidArgumentError(
            "duplicate causal trace site selector");
  }
  ASSIGN_OR_RETURN(const auto shape,
                   ValidateModel(executor, model, vocabulary_size));
  size_t total_cases = 0;
  for (const auto& sentence : sentences) {
    if (sentence.size() < static_cast<size_t>(prompt_tokens) ||
        sentence.size() > shape.context)
      return absl::InvalidArgumentError(
          "causal trace sentence length is outside prompt/context bounds");
    for (int token : sentence)
      if (token < 0 || token >= vocabulary_size || token == eos_token)
        return absl::InvalidArgumentError(
            "causal trace sentence contains an invalid or EOS token");
    const size_t count = sentence.size() - prompt_tokens + 1;
    if (count > std::numeric_limits<size_t>::max() - total_cases ||
        count > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) -
                    total_cases)
      return absl::InvalidArgumentError("causal trace target count overflows");
    total_cases += count;
  }
  const size_t capacity =
      std::min(total_cases, static_cast<size_t>(batch_size));
  if (shape.context >
      static_cast<size_t>(std::numeric_limits<int>::max()) / capacity)
    return absl::InvalidArgumentError(
        "causal trace batch rows exceed int range");
  const size_t rows = capacity * shape.context;
  if (rows > std::numeric_limits<size_t>::max() / sizeof(int) ||
      rows > std::numeric_limits<size_t>::max() / sizeof(float) / shape.stride)
    return absl::InvalidArgumentError("causal trace batch bytes overflow");
  const size_t logits_bytes = rows * shape.stride * sizeof(float);
  std::vector<TraceCase> cases;
  cases.reserve(total_cases);
  for (size_t sentence = 0; sentence < sentences.size(); ++sentence)
    for (size_t length = prompt_tokens; length <= sentences[sentence].size();
         ++length)
      cases.push_back({sentence, length,
                       length == sentences[sentence].size()
                           ? eos_token
                           : sentences[sentence][length]});

  ASSIGN_OR_RETURN(auto clean_inputs,
                   cuda::PageLockedHostArray<int>::Allocate(executor, rows));
  ASSIGN_OR_RETURN(auto corrupt_inputs,
                   cuda::PageLockedHostArray<int>::Allocate(executor, rows));
  ASSIGN_OR_RETURN(auto targets,
                   cuda::PageLockedHostArray<int>::Allocate(executor, rows));
  ASSIGN_OR_RETURN(auto predictions,
                   cuda::PageLockedHostArray<int>::Allocate(executor, rows));
  ASSIGN_OR_RETURN(auto device_clean,
                   cuda::Buffer::Allocate(executor, clean_inputs.size_bytes()));
  ASSIGN_OR_RETURN(
      auto device_corrupt,
      cuda::Buffer::Allocate(executor, corrupt_inputs.size_bytes()));
  ASSIGN_OR_RETURN(auto device_targets,
                   cuda::Buffer::Allocate(executor, targets.size_bytes()));
  std::vector<CausalTraceResult> results;
  results.reserve(sites.size());
  for (const auto& site : sites)
    results.push_back({.site = site});

  for (size_t begin = 0; begin < cases.size(); begin += capacity) {
    const size_t count = std::min(capacity, cases.size() - begin);
    std::fill(clean_inputs.begin(), clean_inputs.end(), eos_token);
    std::fill(corrupt_inputs.begin(), corrupt_inputs.end(), eos_token);
    std::fill(targets.begin(), targets.end(), -1);
    std::vector<size_t> query_rows;
    query_rows.reserve(count);
    int64_t changed_prefixes = 0;
    for (size_t sample = 0; sample < count; ++sample) {
      const auto& probe = cases[begin + sample];
      const auto& sentence = sentences[probe.sentence];
      const auto& donor = sentences[(probe.sentence + 1) % sentences.size()];
      const size_t base = sample * shape.context;
      const size_t replaced =
          probe.prefix_length -
          std::min(probe.prefix_length, static_cast<size_t>(retained_tokens));
      bool changed = false;
      for (size_t position = 0; position < probe.prefix_length; ++position) {
        clean_inputs[base + position] = sentence[position];
        corrupt_inputs[base + position] = position < replaced
                                              ? donor[position % donor.size()]
                                              : sentence[position];
        changed |=
            clean_inputs[base + position] != corrupt_inputs[base + position];
      }
      changed_prefixes += changed;
      query_rows.push_back(base + probe.prefix_length - 1);
      targets[query_rows.back()] = probe.target;
    }
    // These staging buffers are overwritten only after all four kinds of
    // forwards for the preceding batch have synchronized their predictions.
    RETURN_IF_ERROR(cuda::CudaStatus(
        cudaMemcpyAsync(device_clean.data(), clean_inputs.data(),
                        clean_inputs.size_bytes(), cudaMemcpyHostToDevice,
                        executor.stream()),
        "upload clean trace prefixes"));
    RETURN_IF_ERROR(cuda::CudaStatus(
        cudaMemcpyAsync(device_corrupt.data(), corrupt_inputs.data(),
                        corrupt_inputs.size_bytes(), cudaMemcpyHostToDevice,
                        executor.stream()),
        "upload corrupt trace prefixes"));
    RETURN_IF_ERROR(cuda::CudaStatus(
        cudaMemcpyAsync(device_targets.data(), targets.data(),
                        targets.size_bytes(), cudaMemcpyHostToDevice,
                        executor.stream()),
        "upload causal trace targets"));
    TraceHooks clean_hooks(executor, sites, capacity, shape.context,
                           query_rows);
    ASSIGN_OR_RETURN(const auto clean_ids,
                     RunForward(executor, model, device_clean, device_targets,
                                vocabulary_size, logits_bytes, query_rows,
                                clean_hooks, predictions));
    TraceHooks corrupt_hooks(executor, sites, capacity, shape.context,
                             query_rows);
    ASSIGN_OR_RETURN(const auto corrupt_ids,
                     RunForward(executor, model, device_corrupt, device_targets,
                                vocabulary_size, logits_bytes, query_rows,
                                corrupt_hooks, predictions));
    for (size_t site = 0; site < sites.size(); ++site) {
      const auto selected = sites.subspan(site, 1);
      TraceHooks rescue_hooks(executor, selected, capacity, shape.context,
                              query_rows, &clean_hooks.captured(site));
      ASSIGN_OR_RETURN(const auto rescue_ids,
                       RunForward(executor, model, device_corrupt,
                                  device_targets, vocabulary_size, logits_bytes,
                                  query_rows, rescue_hooks, predictions));
      TraceHooks damage_hooks(executor, selected, capacity, shape.context,
                              query_rows, &corrupt_hooks.captured(site));
      ASSIGN_OR_RETURN(const auto damage_ids,
                       RunForward(executor, model, device_clean, device_targets,
                                  vocabulary_size, logits_bytes, query_rows,
                                  damage_hooks, predictions));
      auto& result = results[site];
      result.changed_prefixes += changed_prefixes;
      for (size_t sample = 0; sample < count; ++sample) {
        const int target = cases[begin + sample].target;
        const bool clean_correct = clean_ids[sample] == target;
        const bool corrupt_correct = corrupt_ids[sample] == target;
        const bool rescue_correct = rescue_ids[sample] == target;
        const bool damage_correct = damage_ids[sample] == target;
        ++result.targets;
        result.clean_correct += clean_correct;
        result.corrupt_correct += corrupt_correct;
        result.rescue_correct += rescue_correct;
        result.damage_correct += damage_correct;
        result.baseline_wrong += !corrupt_correct;
        result.rescued += !corrupt_correct && rescue_correct;
        result.newly_broken_by_rescue += corrupt_correct && !rescue_correct;
        result.damaged += clean_correct && !damage_correct;
        result.clean_nonfinite += clean_ids[sample] == -2;
        result.corrupt_nonfinite += corrupt_ids[sample] == -2;
        result.rescue_nonfinite += rescue_ids[sample] == -2;
        result.damage_nonfinite += damage_ids[sample] == -2;
      }
    }
  }
  return results;
}

}  // namespace pluto::llm::one_shot_memorizer
