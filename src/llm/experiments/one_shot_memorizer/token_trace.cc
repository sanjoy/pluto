#include "src/llm/experiments/one_shot_memorizer/token_trace.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "src/cuda/page_locked_host_array.h"
#include "src/llm/layer_hooks.h"
#include "src/util/status_macros.h"

namespace pluto::llm::one_shot_memorizer {
namespace {

using HostBytes = cuda::PageLockedHostArray<uint8_t>;

absl::StatusOr<size_t> Product(size_t a, size_t b) {
  if (a != 0 && b > std::numeric_limits<size_t>::max() / a)
    return absl::OutOfRangeError("token trace tensor byte count overflows");
  return a * b;
}

struct Shape {
  size_t context;
  size_t logit_stride;
  size_t logit_bytes;
};

absl::StatusOr<Shape> ValidateModel(cuda::Executor& executor,
                                    const Layer& model,
                                    absl::Span<const int> prefix,
                                    const TokenTraceOptions& options) {
  if (options.vocabulary_size <= 0 || options.padding_token < 0 ||
      options.padding_token >= options.vocabulary_size || prefix.empty())
    return absl::InvalidArgumentError(
        "token trace requires a nonempty prefix and valid vocabulary/padding");
  for (int token : prefix)
    if (token < 0 || token >= options.vocabulary_size)
      return absl::InvalidArgumentError(
          "prefix token is outside the real vocabulary");
  const auto inputs = model.input_types();
  const auto outputs = model.output_types();
  if (inputs.size() != 1 || outputs.size() != 1)
    return absl::InvalidArgumentError(
        "token trace model must have one input and output");
  RETURN_IF_ERROR(inputs[0].Validate());
  RETURN_IF_ERROR(outputs[0].Validate());
  const auto in = inputs[0].dimensions();
  const auto out = outputs[0].dimensions();
  if (inputs[0].data_type() != DataType::INT32 || in.size() != 2 ||
      in[0] != ActivationType::kBatchDimension ||
      in[1] > std::numeric_limits<int>::max() ||
      prefix.size() > static_cast<size_t>(in[1]))
    return absl::InvalidArgumentError(
        "token trace requires INT32 [batch,context] and a fitting prefix");
  if (outputs[0].data_type() != DataType::FP32 || out.size() != 3 ||
      out[0] != ActivationType::kBatchDimension || out[1] != in[1] ||
      out[2] < options.vocabulary_size ||
      out[2] > std::numeric_limits<int>::max())
    return absl::InvalidArgumentError(
        "token trace requires FP32 [batch,context,padded_vocabulary] logits");
  for (const auto& weight : model.weights())
    if (&weight.executor() != &executor)
      return absl::InvalidArgumentError(
          "token trace model uses another executor");
  ASSIGN_OR_RETURN(auto elements, Product(in[1], out[2]));
  ASSIGN_OR_RETURN(auto bytes, Product(elements, sizeof(float)));
  return Shape{static_cast<size_t>(in[1]), static_cast<size_t>(out[2]), bytes};
}

struct ActivationShape {
  size_t channels;
  size_t element_bytes;
  size_t row_bytes;
};

// All stored rows refer to one sample. No flattening across an inferred batch
// axis is permitted: the model invocation always has exactly one sample.
absl::StatusOr<ActivationShape> ValidateActivation(cuda::Executor& executor,
                                                   const ActivationType& type,
                                                   const Buffer& buffer,
                                                   size_t context) {
  RETURN_IF_ERROR(type.Validate());
  const auto dimensions = type.dimensions();
  if ((type.data_type() != DataType::BF16 &&
       type.data_type() != DataType::FP32) ||
      dimensions.size() != 3 ||
      dimensions[0] != ActivationType::kBatchDimension ||
      dimensions[1] != static_cast<int64_t>(context))
    return absl::InvalidArgumentError(
        "token trace activations must be BF16 or FP32 "
        "[batch,context,channels]");
  const size_t element_bytes = type.data_type() == DataType::BF16 ? 2 : 4;
  ASSIGN_OR_RETURN(auto row_bytes, Product(dimensions[2], element_bytes));
  ASSIGN_OR_RETURN(auto bytes, Product(context, row_bytes));
  if (buffer.size_bytes() != bytes || &buffer.executor() != &executor)
    return absl::InvalidArgumentError(
        "token trace activation shape/executor mismatch");
  return ActivationShape{static_cast<size_t>(dimensions[2]), element_bytes,
                         row_bytes};
}

struct Occurrence {
  std::vector<std::string> scope;
  std::string name;
  int count = 0;
};

struct PendingActivation {
  TokenTraceActivation value;
  HostBytes host;
};

struct PendingAttention {
  TokenTraceAttention value;
  cuda::PageLockedHostArray<float> host;
};

class TraceHooks {
 public:
  TraceHooks(cuda::Executor& executor, const Layer& model, const Shape& shape,
             size_t prefix_length, const TokenTraceOptions& options)
      : executor_(executor),
        model_(model),
        shape_(shape),
        prefix_length_(prefix_length),
        options_(options),
        patch_matches_(options.patches.size(), 0) {
    hooks_.enter_combinator = [this](cuda::Executor& executor,
                                     absl::string_view name) {
      RETURN_IF_ERROR(CheckExecutor(executor));
      scope_.emplace_back(name);
      return absl::OkStatus();
    };
    hooks_.exit_combinator = [this](cuda::Executor& executor) {
      RETURN_IF_ERROR(CheckExecutor(executor));
      if (scope_.empty())
        return absl::FailedPreconditionError("token trace scope underflow");
      scope_.pop_back();
      return absl::OkStatus();
    };
    if (options.capture_activations || !options.patches.empty())
      hooks_.activation_hook = [this](cuda::Executor& executor,
                                      absl::string_view name,
                                      absl::Span<const ActivationType> types,
                                      absl::Span<Buffer> buffers) {
        return Activation(executor, name, types, buffers);
      };
    if (options.capture_attention)
      hooks_.attention_probabilities_hook =
          [this](cuda::Executor& executor, absl::string_view name,
                 const ActivationType& type, const Buffer& buffer) {
            return Attention(executor, name, type, buffer);
          };
  }

  // Callbacks capture this object, so moving/copying it would leave their
  // closure pointers bound to the wrong scope and occurrence counters.
  TraceHooks(const TraceHooks&) = delete;
  TraceHooks& operator=(const TraceHooks&) = delete;
  TraceHooks(TraceHooks&&) = delete;
  TraceHooks& operator=(TraceHooks&&) = delete;

  LayerHooks* hooks() {
    return options_.capture_activations || options_.capture_attention ||
                   !options_.patches.empty()
               ? &hooks_
               : nullptr;
  }

  absl::Status Finish(TokenTraceResult& result) {
    if (!scope_.empty())
      return absl::FailedPreconditionError(
          "token trace scopes were not closed");
    for (size_t i = 0; i < patch_matches_.size(); ++i)
      if (patch_matches_[i] != 1)
        return absl::NotFoundError(
            absl::StrCat("token trace patch site not found: ",
                         options_.patches[i].site.layer_name));
    // Copies were queued immediately after each event, before any consumer
    // could run. Defer CPU reads to one synchronization after the whole pass.
    RETURN_IF_ERROR(executor_.Synchronize());
    for (auto& pending : activations_) {
      auto& value = pending.value;
      value.bytes.assign(pending.host.begin(), pending.host.end());
      value.values.resize(value.row_count * value.channels);
      for (size_t i = 0; i < value.values.size(); ++i) {
        if (value.data_type == DataType::BF16) {
          uint16_t bits;
          std::memcpy(&bits, value.bytes.data() + i * 2, 2);
          value.values[i] =
              std::bit_cast<float>(static_cast<uint32_t>(bits) << 16);
        } else {
          std::memcpy(&value.values[i], value.bytes.data() + i * sizeof(float),
                      sizeof(float));
        }
      }
      result.activations.push_back(std::move(value));
    }
    for (auto& pending : attention_) {
      auto& value = pending.value;
      value.probabilities.assign(pending.host.begin(), pending.host.end());
      for (size_t head = 0; head < value.heads; ++head)
        for (size_t query = 0; query < prefix_length_; ++query) {
          double row_sum = 0;
          for (size_t key = 0; key < prefix_length_; ++key) {
            const float probability =
                value.probabilities[(head * prefix_length_ + query) *
                                        prefix_length_ +
                                    key];
            // Reconstruction can differ from the saved normalization by a
            // few FP32 rounding units. Preserve the value, without clipping.
            if (!std::isfinite(probability) || probability < 0 ||
                probability > 1.0001f || (key > query && probability != 0))
              return absl::DataLossError(
                  "attention hook returned invalid causal probabilities");
            row_sum += probability;
          }
          // These are pre-dropout softmax weights. Capturing every permitted
          // key must preserve unit mass; tolerate FP32 reconstruction error.
          if (std::abs(row_sum - 1) > 1e-4)
            return absl::DataLossError(
                "attention hook returned a nonnormalized probability row");
        }
      result.attention.push_back(std::move(value));
    }
    return absl::OkStatus();
  }

 private:
  absl::Status CheckExecutor(const cuda::Executor& executor) const {
    return &executor == &executor_ ? absl::OkStatus()
                                   : absl::InvalidArgumentError(
                                         "token trace hook executor mismatch");
  }

  int NextOccurrence(absl::string_view name, std::vector<Occurrence>& counts) {
    for (auto& counter : counts)
      if (counter.scope == scope_ && counter.name == name)
        return counter.count++;
    counts.push_back({scope_, std::string(name), 1});
    return 0;
  }

  absl::Status Patch(const TokenTracePatch& patch, const ActivationType& type,
                     const ActivationShape& shape, Buffer& buffer) {
    size_t first_row = prefix_length_ - 1;
    size_t end_row = prefix_length_;
    switch (patch.rows) {
      case TokenTraceRows::kQuery:
        break;
      case TokenTraceRows::kOne:
        if (patch.row >= prefix_length_)
          return absl::InvalidArgumentError(
              "patch row is outside the explicit prefix");
        first_row = patch.row;
        end_row = first_row + 1;
        break;
      case TokenTraceRows::kAllPrefix:
        first_row = 0;
        break;
      default:
        return absl::InvalidArgumentError("unknown token trace row selection");
    }
    if (patch.first_channel >= shape.channels)
      return absl::InvalidArgumentError(
          "patch first channel is outside the activation");
    const size_t channels = patch.channel_count == 0
                                ? shape.channels - patch.first_channel
                                : patch.channel_count;
    if (channels > shape.channels - patch.first_channel)
      return absl::InvalidArgumentError(
          "patch channel range exceeds the activation");
    HostBytes donor;
    if (patch.replacement == TokenTraceReplacement::kDonor) {
      if (patch.donor == nullptr)
        return absl::InvalidArgumentError(
            "donor patch requires a captured donor");
      const auto& source = *patch.donor;
      ASSIGN_OR_RETURN(auto donor_bytes,
                       Product(source.row_count, shape.row_bytes));
      if (source.data_type != type.data_type() ||
          source.channels != shape.channels ||
          source.bytes.size() != donor_bytes || first_row < source.first_row ||
          end_row - source.first_row > source.row_count)
        return absl::InvalidArgumentError(
            "donor dtype/shape/absolute row range differs from patch");
      ASSIGN_OR_RETURN(donor, HostBytes::CopyFrom(executor_, source.bytes));
      // CopyFrom synchronously copies the owned CPU snapshot into fresh pinned
      // storage. Its destructor queues a free on executor_'s stream AFTER all
      // uploads below; no async copy borrows the caller's std::vector memory.
    } else if (patch.replacement != TokenTraceReplacement::kZero ||
               patch.donor != nullptr) {
      return absl::InvalidArgumentError(
          "invalid zero/donor patch configuration");
    }
    ASSIGN_OR_RETURN(auto replacement,
                     Buffer::Allocate(executor_, buffer.size_bytes()));
    RETURN_IF_ERROR(cuda::CudaStatus(
        cudaMemcpyAsync(replacement.data(), buffer.data(), buffer.size_bytes(),
                        cudaMemcpyDeviceToDevice, executor_.stream()),
        "clone token trace activation before patch"));
    const size_t channel_bytes = channels * shape.element_bytes;
    for (size_t row = first_row; row < end_row; ++row) {
      auto* destination = static_cast<uint8_t*>(replacement.data()) +
                          row * shape.row_bytes +
                          patch.first_channel * shape.element_bytes;
      if (patch.replacement == TokenTraceReplacement::kZero) {
        RETURN_IF_ERROR(cuda::CudaStatus(
            cudaMemsetAsync(destination, 0, channel_bytes, executor_.stream()),
            "zero selected token trace channels"));
      } else {
        const auto* source = donor.data() +
                             (row - patch.donor->first_row) * shape.row_bytes +
                             patch.first_channel * shape.element_bytes;
        RETURN_IF_ERROR(cuda::CudaStatus(
            cudaMemcpyAsync(destination, source, channel_bytes,
                            cudaMemcpyHostToDevice, executor_.stream()),
            "patch exact donor activation bytes"));
      }
    }
    buffer = std::move(replacement);
    return absl::OkStatus();
  }

  absl::Status Activation(cuda::Executor& executor, absl::string_view name,
                          absl::Span<const ActivationType> types,
                          absl::Span<Buffer> buffers) {
    RETURN_IF_ERROR(CheckExecutor(executor));
    if (types.size() != buffers.size())
      return absl::InvalidArgumentError(
          "token trace hook type/buffer count differs");
    const int occurrence = NextOccurrence(name, activation_counts_);
    for (size_t i = 0; i < options_.patches.size(); ++i) {
      const auto& patch = options_.patches[i];
      if (patch.site.scope != scope_ || patch.site.layer_name != name)
        continue;
      if (patch.site.occurrence == -1 && occurrence > 0)
        return absl::InvalidArgumentError(
            "token trace patch site is ambiguous; specify occurrence");
      if (patch.site.occurrence != -1 && patch.site.occurrence != occurrence)
        continue;
      if (patch.site.output_index >= buffers.size())
        return absl::InvalidArgumentError(
            "patch output index is outside the hook outputs");
      const size_t output = patch.site.output_index;
      ASSIGN_OR_RETURN(auto shape,
                       ValidateActivation(executor_, types[output],
                                          buffers[output], shape_.context));
      RETURN_IF_ERROR(Patch(patch, types[output], shape, buffers[output]));
      ++patch_matches_[i];
    }
    if (!options_.capture_activations)
      return absl::OkStatus();
    for (size_t output = 0; output < buffers.size(); ++output) {
      ASSIGN_OR_RETURN(auto shape,
                       ValidateActivation(executor_, types[output],
                                          buffers[output], shape_.context));
      // LM-head padding remains represented in this raw donor snapshot. Only
      // the final public logits below discard non-vocabulary columns.
      const bool logits = (name == "LanguageModelingHeadLayer" ||
                           (scope_.empty() && name == model_.name())) &&
                          types[output].data_type() == DataType::FP32 &&
                          shape.channels == shape_.logit_stride;
      const size_t first_row = logits ? prefix_length_ - 1 : 0;
      const size_t row_count = logits ? 1 : prefix_length_;
      ASSIGN_OR_RETURN(auto host, HostBytes::Allocate(
                                      executor_, row_count * shape.row_bytes));
      RETURN_IF_ERROR(cuda::CudaStatus(
          cudaMemcpyAsync(host.data(),
                          static_cast<const uint8_t*>(buffers[output].data()) +
                              first_row * shape.row_bytes,
                          host.size_bytes(), cudaMemcpyDeviceToHost,
                          executor_.stream()),
          "capture token trace prefix activation"));
      activations_.push_back(
          {TokenTraceActivation{
               .site = {scope_, std::string(name), occurrence, output},
               .data_type = types[output].data_type(),
               .first_row = first_row,
               .row_count = row_count,
               .channels = shape.channels},
           std::move(host)});
    }
    return absl::OkStatus();
  }

  absl::Status Attention(cuda::Executor& executor, absl::string_view name,
                         const ActivationType& type, const Buffer& buffer) {
    RETURN_IF_ERROR(CheckExecutor(executor));
    RETURN_IF_ERROR(type.Validate());
    const auto dimensions = type.dimensions();
    if (type.data_type() != DataType::FP32 || dimensions.size() != 4 ||
        dimensions[0] != 1 ||
        dimensions[2] != static_cast<int64_t>(shape_.context) ||
        dimensions[3] != dimensions[2] || &buffer.executor() != &executor_)
      return absl::InvalidArgumentError(
          "attention trace must be FP32 [1,heads,context,context]");
    ASSIGN_OR_RETURN(auto head_elements,
                     Product(shape_.context, shape_.context));
    ASSIGN_OR_RETURN(auto elements, Product(dimensions[1], head_elements));
    ASSIGN_OR_RETURN(auto bytes, Product(elements, sizeof(float)));
    if (buffer.size_bytes() != bytes)
      return absl::InvalidArgumentError(
          "attention trace buffer has incorrect size");
    const size_t heads = dimensions[1];
    ASSIGN_OR_RETURN(auto prefix_elements,
                     Product(prefix_length_, prefix_length_));
    ASSIGN_OR_RETURN(auto captured_elements, Product(heads, prefix_elements));
    ASSIGN_OR_RETURN(auto host, cuda::PageLockedHostArray<float>::Allocate(
                                    executor_, captured_elements));
    for (size_t head = 0; head < heads; ++head)
      RETURN_IF_ERROR(cuda::CudaStatus(
          cudaMemcpy2DAsync(
              host.data() + head * prefix_elements,
              prefix_length_ * sizeof(float),
              static_cast<const float*>(buffer.data()) + head * head_elements,
              shape_.context * sizeof(float), prefix_length_ * sizeof(float),
              prefix_length_, cudaMemcpyDeviceToHost, executor_.stream()),
          "capture causal attention prefix square"));
    const int occurrence = NextOccurrence(name, attention_counts_);
    attention_.push_back(
        {TokenTraceAttention{.site = {scope_, std::string(name), occurrence, 0},
                             .heads = heads,
                             .prefix_length = prefix_length_},
         std::move(host)});
    return absl::OkStatus();
  }

  cuda::Executor& executor_;
  const Layer& model_;
  Shape shape_;
  size_t prefix_length_;
  const TokenTraceOptions& options_;
  std::vector<size_t> patch_matches_;
  std::vector<std::string> scope_;
  std::vector<Occurrence> activation_counts_;
  std::vector<Occurrence> attention_counts_;
  std::vector<PendingActivation> activations_;
  std::vector<PendingAttention> attention_;
  LayerHooks hooks_;
};

}  // namespace

absl::StatusOr<TokenTraceResult> TraceNextToken(
    cuda::Executor& executor, const Layer& model, absl::Span<const int> prefix,
    const TokenTraceOptions& options) {
  static_assert(sizeof(int) == sizeof(int32_t));  // INT32 model input storage.
  ASSIGN_OR_RETURN(const auto shape,
                   ValidateModel(executor, model, prefix, options));
  for (const auto& patch : options.patches)
    if (patch.site.layer_name.empty() || patch.site.occurrence < -1)
      return absl::InvalidArgumentError(
          "patch requires a layer name and occurrence >= -1");
  ASSIGN_OR_RETURN(auto host_input, cuda::PageLockedHostArray<int>::Allocate(
                                        executor, shape.context));
  std::fill(host_input.begin(), host_input.end(), options.padding_token);
  std::copy(prefix.begin(), prefix.end(), host_input.begin());
  ASSIGN_OR_RETURN(auto input,
                   Buffer::Allocate(executor, host_input.size_bytes()));
  RETURN_IF_ERROR(cuda::CudaStatus(
      cudaMemcpyAsync(input.data(), host_input.data(), host_input.size_bytes(),
                      cudaMemcpyHostToDevice, executor.stream()),
      "upload explicit prefix with fixed future padding"));
  TraceHooks hooks(executor, model, shape, prefix.size(), options);
  LayerHooks composed;
  LayerHooks* dispatch = hooks.hooks();
  if (options.compose_hooks) {
    composed =
        options.compose_hooks(dispatch == nullptr ? LayerHooks{} : *dispatch);
    dispatch = &composed;
  }
  ASSIGN_OR_RETURN(auto forward, model.fwd(executor, {&input, 1}, dispatch));
  if (forward.outputs.size() != 1 ||
      forward.outputs[0].size_bytes() != shape.logit_bytes ||
      &forward.outputs[0].executor() != &executor)
    return absl::InvalidArgumentError(
        "token trace model output differs from signature");
  ASSIGN_OR_RETURN(auto scores, cuda::PageLockedHostArray<float>::Allocate(
                                    executor, options.vocabulary_size));
  RETURN_IF_ERROR(cuda::CudaStatus(
      cudaMemcpyAsync(scores.data(),
                      static_cast<const float*>(forward.outputs[0].data()) +
                          (prefix.size() - 1) * shape.logit_stride,
                      scores.size_bytes(), cudaMemcpyDeviceToHost,
                      executor.stream()),
      "read first-unseen-token logits"));
  TokenTraceResult result{.query_row = prefix.size() - 1};
  RETURN_IF_ERROR(hooks.Finish(result));
  result.logits.assign(scores.begin(), scores.end());
  for (size_t i = 0; i < result.logits.size(); ++i) {
    if (!std::isfinite(result.logits[i]))
      return absl::DataLossError(
          "next-token real-vocabulary logits are nonfinite");
    if (result.logits[i] > result.logits[result.predicted_token])
      result.predicted_token = static_cast<int>(i);
  }
  return result;
}

}  // namespace pluto::llm::one_shot_memorizer
