#include "src/llm/experiments/one_shot_memorizer/final_mlp_probe.h"

#include <cuda_runtime_api.h>

#include <bit>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "src/cuda/page_locked_host_array.h"
#include "src/llm/batch_validation.h"
#include "src/llm/extract_top1_ids.h"
#include "src/llm/layer_hooks.h"
#include "src/util/status_macros.h"

namespace pluto::llm::one_shot_memorizer {
namespace {

int BlockIndex(absl::string_view name) {
  constexpr absl::string_view prefix = "transformer_block_";
  if (name.size() <= prefix.size() || name.substr(0, prefix.size()) != prefix)
    return -1;
  int index = -1;
  const auto parsed = std::from_chars(name.data() + prefix.size(),
                                      name.data() + name.size(), index);
  if (parsed.ec != std::errc{} || parsed.ptr != name.data() + name.size() ||
      index < 0 || name != absl::StrCat(prefix, index))
    return -1;
  return index;
}

absl::Status CheckWeights(cuda::Executor& executor, const Layer& layer) {
  for (const auto& weight : layer.weights())
    if (&weight.executor() != &executor)
      return absl::InvalidArgumentError(
          "final MLP layer uses another executor");
  return absl::OkStatus();
}

absl::Status CheckTypes(absl::Span<const ActivationType> types,
                        const ActivationType& expected) {
  if (types.size() != 1 || types[0] != expected)
    return absl::InvalidArgumentError("unexpected final MLP layer signature");
  return expected.Validate();
}

absl::Status CheckActivation(cuda::Executor& executor, const DataBatch& batch,
                             absl::Span<const ActivationType> types,
                             absl::Span<const Buffer> buffers, int width) {
  RETURN_IF_ERROR(CheckTypes(
      types, ActivationType(DataType::BF16, {ActivationType::kBatchDimension,
                                             batch.sequence_length, width})));
  return ValidateBatchBuffers(executor, batch, buffers, types,
                              "final MLP activation");
}

// No extra wrapper combinator is inserted around gpt2: existing experiment
// hooks intentionally identify sites by the original GPT-2 scope stack.
class FinalHooks {
 public:
  FinalHooks(cuda::Executor& executor, const DataBatch& batch, int width,
             int feature_width, int last_block, const Layer* projection,
             const Layer* norm, const Layer* head, LayerHooks* outer)
      : executor_(executor),
        batch_(batch),
        width_(width),
        feature_width_(feature_width),
        last_block_(last_block),
        projection_(projection),
        norm_(norm),
        head_(head),
        outer_(outer) {
    hooks_.enter_combinator = [&](cuda::Executor& actual,
                                  absl::string_view name) {
      RETURN_IF_ERROR(CheckExecutor(actual));
      const int block = BlockIndex(name);
      if (block >= 0 && blocks_.contains(block))
        return absl::InvalidArgumentError("duplicate final MLP block scope");
      // A failed enter must not modify either scope stack.
      if (outer_ && outer_->enter_combinator)
        RETURN_IF_ERROR(outer_->enter_combinator(actual, name));
      scopes_.emplace_back(name);
      if (block >= 0)
        blocks_.insert(block);
      return absl::OkStatus();
    };
    hooks_.exit_combinator = [&](cuda::Executor& actual) {
      RETURN_IF_ERROR(CheckExecutor(actual));
      if (scopes_.empty())
        return absl::FailedPreconditionError("final MLP scope underflow");
      scopes_.pop_back();
      if (outer_ && outer_->exit_combinator)
        return outer_->exit_combinator(actual);
      return absl::OkStatus();
    };
    hooks_.activation_hook = [&](cuda::Executor& actual, absl::string_view name,
                                 absl::Span<const ActivationType> types,
                                 absl::Span<Buffer> outputs) {
      RETURN_IF_ERROR(CheckExecutor(actual));
      return Observe(name, types, outputs);
    };
    if (outer_)
      hooks_.attention_probabilities_hook =
          outer_->attention_probabilities_hook;
  }

  LayerHooks& hooks() { return hooks_; }
  const std::optional<Buffer>& normalized() const { return normalized_; }
  const std::optional<Buffer>& features() const { return features_; }

  absl::Status Finish() const {
    if (!scopes_.empty() || !normalized_ || !features_ || !last_residual_ ||
        !final_normalized_ || !head_seen_ ||
        blocks_.size() != static_cast<size_t>(last_block_) + 1)
      return absl::InvalidArgumentError(
          "missing or ambiguous final GPT-2 sites");
    int expected = 0;
    for (int block : blocks_)
      if (block != expected++)
        return absl::InvalidArgumentError("noncontiguous GPT-2 block indices");
    return absl::OkStatus();
  }

 private:
  absl::Status CheckExecutor(const cuda::Executor& actual) const {
    if (&actual != &executor_)
      return absl::InvalidArgumentError("final MLP hook uses another executor");
    return absl::OkStatus();
  }

  absl::Status Replace(const Layer& layer, const Buffer& source,
                       absl::Span<const ActivationType> source_types,
                       absl::Span<const ActivationType> output_types,
                       absl::Span<Buffer> outputs) {
    if (source_types.size() != 1 || output_types.size() != 1 ||
        outputs.size() != 1)
      return absl::InvalidArgumentError("invalid final MLP replacement arity");
    RETURN_IF_ERROR(CheckTypes(layer.input_types(), source_types[0]));
    RETURN_IF_ERROR(CheckTypes(layer.output_types(), output_types[0]));
    RETURN_IF_ERROR(CheckWeights(executor_, layer));
    RETURN_IF_ERROR(ValidateBatchBuffers(executor_, batch_, {&source, 1},
                                         layer.input_types(),
                                         "replacement input"));
    // Replacement internals never contaminate the original GPT-2 scopes.
    ASSIGN_OR_RETURN(auto result, layer.fwd(executor_, {&source, 1}, nullptr));
    RETURN_IF_ERROR(ValidateBatchBuffers(executor_, batch_, result.outputs,
                                         layer.output_types(),
                                         "replacement output"));
    outputs[0] = std::move(result.outputs[0]);
    return absl::OkStatus();
  }

  absl::Status Observe(absl::string_view name,
                       absl::Span<const ActivationType> types,
                       absl::Span<Buffer> outputs) {
    const bool direct = scopes_.size() == 1 && scopes_[0] == "gpt2";
    const bool block_scope = scopes_.size() >= 3 && scopes_[0] == "gpt2" &&
                             BlockIndex(scopes_[1]) == last_block_ &&
                             scopes_[2] == "ResidualLayer";
    const bool inner =
        block_scope && scopes_.size() == 4 && scopes_[3] == "mlp";
    const bool branch = block_scope && scopes_.size() == 3 && name == "mlp";
    const bool final_norm = direct && name == "LayerNormLayer";
    const bool final_head = direct && name == "LanguageModelingHeadLayer";
    if (branch && projection_) {
      if (!features_)
        return absl::NotFoundError("no fresh GELU for final MLP replacement");
      const ActivationType source(DataType::BF16,
                                  {ActivationType::kBatchDimension,
                                   batch_.sequence_length, feature_width_});
      RETURN_IF_ERROR(
          Replace(*projection_, *features_, {&source, 1}, types, outputs));
    }
    if (final_norm && norm_) {
      if (!last_residual_)
        return absl::NotFoundError(
            "no fresh last-block residual for final norm");
      RETURN_IF_ERROR(Replace(*norm_, *last_residual_, types, types, outputs));
    }
    if (final_head && head_) {
      if (!final_normalized_)
        return absl::NotFoundError("no fresh final norm for replacement head");
      const ActivationType source(
          DataType::BF16,
          {ActivationType::kBatchDimension, batch_.sequence_length, width_});
      RETURN_IF_ERROR(
          Replace(*head_, *final_normalized_, {&source, 1}, types, outputs));
    }
    // The outer observer sees substitutions. Retain handles AFTER it runs so
    // any legitimate outer intervention becomes the next replacement's input.
    if (outer_ && outer_->activation_hook)
      RETURN_IF_ERROR(outer_->activation_hook(executor_, name, types, outputs));
    if (inner && name == "LayerNormLayer") {
      RETURN_IF_ERROR(
          CheckActivation(executor_, batch_, types, outputs, width_));
      if (normalized_)
        return absl::InvalidArgumentError("duplicate final MLP normalization");
      normalized_ = outputs[0];
    }
    if (inner && name == "GeluLayer") {
      if (feature_width_ == 0) {
        if (types.size() != 1 || types[0].dimensions().size() != 3 ||
            types[0].dimensions()[2] <= 0 ||
            types[0].dimensions()[2] > std::numeric_limits<int>::max())
          return absl::InvalidArgumentError("invalid observed GELU width");
        feature_width_ = static_cast<int>(types[0].dimensions()[2]);
      }
      RETURN_IF_ERROR(
          CheckActivation(executor_, batch_, types, outputs, feature_width_));
      if (features_)
        return absl::InvalidArgumentError("duplicate final MLP GELU");
      features_ = outputs[0];
    }
    if (direct && BlockIndex(name) == last_block_) {
      RETURN_IF_ERROR(
          CheckActivation(executor_, batch_, types, outputs, width_));
      if (last_residual_)
        return absl::InvalidArgumentError("duplicate final transformer block");
      last_residual_ = outputs[0];
    }
    if (final_norm) {
      RETURN_IF_ERROR(
          CheckActivation(executor_, batch_, types, outputs, width_));
      if (final_normalized_)
        return absl::InvalidArgumentError(
            "duplicate direct final normalization");
      final_normalized_ = outputs[0];
    }
    if (final_head) {
      if (head_seen_)
        return absl::InvalidArgumentError("duplicate direct final head");
      head_seen_ = true;
    }
    return absl::OkStatus();
  }

  cuda::Executor& executor_;
  const DataBatch& batch_;
  int width_;
  int feature_width_;
  int last_block_;
  const Layer* projection_;
  const Layer* norm_;
  const Layer* head_;
  LayerHooks* outer_;
  LayerHooks hooks_;
  std::vector<std::string> scopes_;
  std::set<int> blocks_;
  std::optional<Buffer> normalized_, features_, last_residual_,
      final_normalized_;
  bool head_seen_ = false;
};

const BackwardState* UniqueChild(const BackwardState& state,
                                 absl::string_view name) {
  const BackwardState* found = nullptr;
  for (const auto& child : state.children) {
    if (child.layer && child.layer->name() == name) {
      if (found)
        return nullptr;
      found = &child;
    }
  }
  return found;
}

// Select by named tree paths, not flattened parameter or activation offsets.
absl::StatusOr<const BackwardState*> LastMlpNormState(const BackwardState& root,
                                                      int last_block) {
  if (!root.layer || root.layer->name() != "gpt2")
    return absl::InvalidArgumentError(
        "capture requires an original GPT-2 state");
  const auto* block =
      UniqueChild(root, absl::StrCat("transformer_block_", last_block));
  if (!block)
    return absl::NotFoundError("missing final block state");
  const BackwardState* mlp = nullptr;
  for (const auto& child : block->children) {
    if (!child.layer || child.layer->name() != "ResidualLayer")
      continue;
    const auto* candidate = UniqueChild(child, "mlp");
    if (candidate) {
      if (mlp)
        return absl::InvalidArgumentError("ambiguous final MLP state");
      mlp = candidate;
    }
  }
  const auto* norm = mlp ? UniqueChild(*mlp, "LayerNormLayer") : nullptr;
  if (!norm || norm->intermediates.empty())
    return absl::NotFoundError("missing final MLP normalization input state");
  return norm;
}

template <typename T>
absl::StatusOr<cuda::PageLockedHostArray<T>> Download(cuda::Executor& executor,
                                                      const Buffer& source) {
  if (&source.executor() != &executor || source.size_bytes() % sizeof(T))
    return absl::InvalidArgumentError("invalid final MLP download buffer");
  ASSIGN_OR_RETURN(auto host, cuda::PageLockedHostArray<T>::Allocate(
                                  executor, source.size_bytes() / sizeof(T)));
  RETURN_IF_ERROR(cuda::CudaStatus(
      cudaMemcpyAsync(host.data(), source.data(), source.size_bytes(),
                      cudaMemcpyDeviceToHost, executor.stream()),
      "download final MLP capture"));
  return host;
}

class FinalMlpReplacement final : public Layer {
 public:
  FinalMlpReplacement(const Layer& model, const Layer* projection,
                      const Layer& norm, const Layer& head, int last_block)
      : model_(model),
        projection_(projection),
        norm_(norm),
        head_(head),
        last_block_(last_block) {}
  absl::string_view name() const override { return "FinalMlpReplacement"; }
  absl::Span<const ActivationType> input_types() const override {
    return model_.input_types();
  }
  absl::Span<const ActivationType> output_types() const override {
    return model_.output_types();
  }
  // Deliberately do not expose mutable aliases to borrowed original weights.
  absl::Span<Buffer> weights() override { return {}; }
  DataType output_type() const override { return model_.output_type(); }

 private:
  absl::StatusOr<FwdResult> fwd_impl(cuda::Executor& executor,
                                     absl::Span<const Buffer> inputs,
                                     LayerHooks* outer) const override {
    if (inputs.size() != 1)
      return absl::InvalidArgumentError("final MLP view requires one input");
    const int context = static_cast<int>(input_types()[0].dimensions()[1]);
    if (inputs[0].size_bytes() % (static_cast<size_t>(context) * sizeof(int)))
      return absl::InvalidArgumentError("incomplete final MLP token batch");
    const size_t batch_size = inputs[0].size_bytes() / sizeof(int) / context;
    if (batch_size == 0 || batch_size > std::numeric_limits<int>::max())
      return absl::InvalidArgumentError("invalid final MLP batch size");
    DataBatch batch{inputs[0], inputs[0], static_cast<int>(batch_size),
                    context};
    RETURN_IF_ERROR(ValidateBatchBuffers(executor, batch, inputs, input_types(),
                                         "final MLP input"));
    RETURN_IF_ERROR(CheckWeights(executor, model_));
    const int width = static_cast<int>(norm_.input_types()[0].dimensions()[2]);
    // With a retained branch we still observe and validate the fresh GELU's
    // width dynamically; it need not equal the residual width.
    const int features =
        projection_
            ? static_cast<int>(projection_->input_types()[0].dimensions()[2])
            : 0;
    // The null-projection path does not need the feature buffer, but capture
    // site validation below learns its declared dimension before observing it.
    FinalHooks hooks(executor, batch, width, features, last_block_, projection_,
                     &norm_, &head_, outer);
    ASSIGN_OR_RETURN(auto result, model_.fwd(executor, inputs, &hooks.hooks()));
    RETURN_IF_ERROR(hooks.Finish());
    RETURN_IF_ERROR(ValidateBatchBuffers(executor, batch, result.outputs,
                                         output_types(), "final MLP logits"));
    result.state = BackwardState{};  // Interventions have no backward rule.
    return result;
  }
  absl::StatusOr<BufferVec> bwd_impl(cuda::Executor&, absl::Span<const Buffer>,
                                     BackwardState, LayerHooks*) override {
    return absl::UnimplementedError("final MLP replacement is forward-only");
  }
  const Layer& model_;
  const Layer* projection_;
  const Layer& norm_;
  const Layer& head_;
  int last_block_;
};

}  // namespace

absl::StatusOr<FinalMlpBatch> CaptureFinalMlpBatch(
    cuda::Executor& executor, const Layer& model, const DataBatch& batch,
    int width, int feature_width, int block_count, int vocabulary_size) {
  ASSIGN_OR_RETURN(const int rows, batch.token_count());
  ASSIGN_OR_RETURN(const int supervised, batch.loss_row_count());
  if (width <= 0 || feature_width <= 0 || block_count <= 0 ||
      vocabulary_size <= 0)
    return absl::InvalidArgumentError("invalid final MLP capture dimensions");
  const ActivationType tokens(DataType::INT32, {ActivationType::kBatchDimension,
                                                batch.sequence_length});
  RETURN_IF_ERROR(CheckTypes(model.input_types(), tokens));
  RETURN_IF_ERROR(
      ValidateBatchInput(executor, batch, batch.inputs, tokens, "tokens"));
  RETURN_IF_ERROR(
      ValidateBatchInput(executor, batch, batch.targets, tokens, "targets"));
  const auto outputs = model.output_types();
  if (outputs.size() != 1 || outputs[0].data_type() != DataType::FP32)
    return absl::InvalidArgumentError("final MLP capture requires FP32 logits");
  RETURN_IF_ERROR(outputs[0].Validate());
  const auto dimensions = outputs[0].dimensions();
  if (dimensions.size() != 3 ||
      dimensions[0] != ActivationType::kBatchDimension ||
      dimensions[1] != batch.sequence_length ||
      dimensions[2] < vocabulary_size ||
      dimensions[2] > std::numeric_limits<int>::max())
    return absl::InvalidArgumentError("invalid final MLP logit shape");
  RETURN_IF_ERROR(CheckWeights(executor, model));
  FinalHooks hooks(executor, batch, width, feature_width, block_count - 1,
                   nullptr, nullptr, nullptr, nullptr);
  ASSIGN_OR_RETURN(auto forward,
                   model.fwd(executor, {batch.inputs}, &hooks.hooks()));
  RETURN_IF_ERROR(hooks.Finish());
  RETURN_IF_ERROR(ValidateBatchBuffers(executor, batch, forward.outputs,
                                       outputs, "logits"));
  ASSIGN_OR_RETURN(const auto* mlp_norm,
                   LastMlpNormState(forward.state, block_count - 1));
  const auto* final_norm = UniqueChild(forward.state, "LayerNormLayer");
  if (!final_norm || !final_norm->layer)
    return absl::NotFoundError("missing direct final normalization state");
  RETURN_IF_ERROR(CheckActivation(executor, batch,
                                  mlp_norm->layer->input_types(),
                                  {&mlp_norm->intermediates[0], 1}, width));
  const auto norm_weights = final_norm->layer->weights();
  if (norm_weights.size() != 2 ||
      norm_weights[0].size_bytes() != width * sizeof(float) ||
      norm_weights[1].size_bytes() != width * sizeof(float))
    return absl::InvalidArgumentError("malformed final normalization weights");
  ASSIGN_OR_RETURN(auto predictions,
                   ExtractTop1Ids(executor, forward.outputs[0], batch.targets,
                                  vocabulary_size));
  ASSIGN_OR_RETURN(auto normalized,
                   Download<uint16_t>(executor, *hooks.normalized()));
  ASSIGN_OR_RETURN(auto features,
                   Download<uint16_t>(executor, *hooks.features()));
  ASSIGN_OR_RETURN(auto residuals,
                   Download<uint16_t>(executor, mlp_norm->intermediates[0]));
  ASSIGN_OR_RETURN(auto targets, Download<int>(executor, batch.targets));
  ASSIGN_OR_RETURN(auto predicted, Download<int>(executor, predictions));
  ASSIGN_OR_RETURN(auto gamma, Download<float>(executor, norm_weights[0]));
  ASSIGN_OR_RETURN(auto beta, Download<float>(executor, norm_weights[1]));
  RETURN_IF_ERROR(executor.Synchronize());
  FinalMlpBatch result;
  result.width = width;
  result.feature_width = feature_width;
  for (int d = 0; d < width; ++d) {
    if (!std::isfinite(gamma[d]) || !std::isfinite(beta[d]))
      return absl::InvalidArgumentError(
          "nonfinite final normalization weights");
    result.finalnorm_gamma.push_back(gamma[d]);
    result.finalnorm_beta.push_back(beta[d]);
  }
  auto append = [](const auto& values, int row, int columns,
                   std::vector<float>& destination) -> absl::Status {
    for (int d = 0; d < columns; ++d) {
      const float value = std::bit_cast<float>(
          uint32_t{values[static_cast<size_t>(row) * columns + d]} << 16);
      if (!std::isfinite(value))
        return absl::InvalidArgumentError(
            "nonfinite supervised final MLP activation");
      destination.push_back(value);
    }
    return absl::OkStatus();
  };
  for (int row = 0; row < rows; ++row) {
    if (targets[row] == -1)
      continue;
    if (targets[row] < 0 || targets[row] >= vocabulary_size ||
        predicted[row] < 0 || predicted[row] >= vocabulary_size)
      return absl::InvalidArgumentError(
          "invalid target or nonfinite final MLP logits");
    result.labels.push_back(targets[row]);
    result.original_predictions.push_back(predicted[row]);
    result.sample_indices.push_back(row / batch.sequence_length);
    result.positions.push_back(row % batch.sequence_length);
    RETURN_IF_ERROR(append(normalized, row, width, result.normalized_inputs));
    RETURN_IF_ERROR(append(features, row, feature_width, result.features));
    RETURN_IF_ERROR(append(residuals, row, width, result.residuals));
  }
  if (batch.supervised_row_count >= 0 &&
      result.labels.size() != static_cast<size_t>(supervised))
    return absl::InvalidArgumentError("final MLP target mask/count disagree");
  return result;
}

absl::StatusOr<std::unique_ptr<Layer>> CreateFinalMlpReplacement(
    const Layer& model, const Layer* projection, const Layer& norm,
    const Layer& head, int last_block) {
  const auto inputs = model.input_types();
  const auto outputs = model.output_types();
  const auto norm_inputs = norm.input_types();
  if (last_block < 0 || inputs.size() != 1 || outputs.size() != 1 ||
      norm_inputs.size() != 1)
    return absl::InvalidArgumentError(
        "invalid final MLP replacement dimensions");
  RETURN_IF_ERROR(inputs[0].Validate());
  RETURN_IF_ERROR(outputs[0].Validate());
  RETURN_IF_ERROR(norm_inputs[0].Validate());
  const auto token_dims = inputs[0].dimensions();
  const auto norm_dims = norm_inputs[0].dimensions();
  const auto logit_dims = outputs[0].dimensions();
  if (inputs[0].data_type() != DataType::INT32 || token_dims.size() != 2 ||
      token_dims[0] != ActivationType::kBatchDimension ||
      token_dims[1] > std::numeric_limits<int>::max() ||
      norm_inputs[0].data_type() != DataType::BF16 || norm_dims.size() != 3 ||
      norm_dims[0] != token_dims[0] || norm_dims[1] != token_dims[1] ||
      norm_dims[2] > std::numeric_limits<int>::max() ||
      outputs[0].data_type() != DataType::FP32 || logit_dims.size() != 3 ||
      logit_dims[0] != token_dims[0] || logit_dims[1] != token_dims[1] ||
      logit_dims[2] > std::numeric_limits<int>::max())
    return absl::InvalidArgumentError(
        "incompatible final MLP token/norm/logit shapes");
  RETURN_IF_ERROR(CheckTypes(norm.output_types(), norm_inputs[0]));
  RETURN_IF_ERROR(CheckTypes(head.input_types(), norm_inputs[0]));
  RETURN_IF_ERROR(CheckTypes(head.output_types(), outputs[0]));
  if (projection) {
    RETURN_IF_ERROR(CheckTypes(projection->output_types(), norm_inputs[0]));
    const auto projected_inputs = projection->input_types();
    if (projected_inputs.size() != 1)
      return absl::InvalidArgumentError(
          "final MLP projection must have one input");
    RETURN_IF_ERROR(projected_inputs[0].Validate());
    const auto shape = projected_inputs[0].dimensions();
    if (projected_inputs[0].data_type() != DataType::BF16 ||
        shape.size() != 3 || shape[0] != token_dims[0] ||
        shape[1] != token_dims[1] || shape[2] > std::numeric_limits<int>::max())
      return absl::InvalidArgumentError(
          "invalid final MLP projection input shape");
  }
  return std::unique_ptr<Layer>(absl::make_unique<FinalMlpReplacement>(
      model, projection, norm, head, last_block));
}

}  // namespace pluto::llm::one_shot_memorizer
