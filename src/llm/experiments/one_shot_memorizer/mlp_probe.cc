#include "src/llm/experiments/one_shot_memorizer/mlp_probe.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <bit>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "src/cuda/page_locked_host_array.h"
#include "src/llm/batch_validation.h"
#include "src/llm/extract_top1_ids.h"
#include "src/llm/layer_hooks.h"
#include "src/util/status_macros.h"

namespace pluto::llm::one_shot_memorizer {
namespace {

float ExpandBf16(uint16_t value) {
  return std::bit_cast<float>(uint32_t{value} << 16);
}

float RoundBf16(float value) {
  uint32_t word = std::bit_cast<uint32_t>(value);
  word += 0x7fff + ((word >> 16) & 1);
  return ExpandBf16(static_cast<uint16_t>(word >> 16));
}

int BlockIndex(absl::string_view name) {
  constexpr absl::string_view prefix = "transformer_block_";
  if (name.size() <= prefix.size() || name.substr(0, prefix.size()) != prefix)
    return -1;
  int block = -1;
  const auto parsed = std::from_chars(name.data() + prefix.size(),
                                      name.data() + name.size(), block);
  if (parsed.ec != std::errc{} || parsed.ptr != name.data() + name.size() ||
      block < 0 || name != absl::StrCat(prefix, block))
    return -1;
  return block;
}

template <class T>
absl::StatusOr<cuda::PageLockedHostArray<T>> QueueDownload(
    cuda::Executor& executor, const Buffer& buffer) {
  if (&buffer.executor() != &executor || buffer.size_bytes() % sizeof(T) != 0)
    return absl::InvalidArgumentError("invalid MLP probe download buffer");
  ASSIGN_OR_RETURN(auto host, cuda::PageLockedHostArray<T>::Allocate(
                                  executor, buffer.size_bytes() / sizeof(T)));
  RETURN_IF_ERROR(cuda::CudaStatus(
      cudaMemcpyAsync(host.data(), buffer.data(), buffer.size_bytes(),
                      cudaMemcpyDeviceToHost, executor.stream()),
      "download MLP probe data"));
  return host;
}

absl::Status ValidateDataset(cuda::Executor& executor, const Layer& model,
                             const PaddedLineDataSetIterator& dataset,
                             int vocabulary_size) {
  if (vocabulary_size <= 0 || dataset.options().shuffle ||
      dataset.options().eos_token < 0 ||
      dataset.options().eos_token >= vocabulary_size ||
      dataset.sample_count() == 0)
    return absl::InvalidArgumentError(
        "MLP probes require an unshuffled corpus and valid vocabulary");
  const auto inputs = model.input_types();
  const auto outputs = model.output_types();
  const ActivationType expected_input(
      DataType::INT32,
      {ActivationType::kBatchDimension, dataset.options().context_length});
  if (inputs.size() != 1 || inputs[0] != expected_input ||
      outputs.size() != 1 || outputs[0].data_type() != DataType::FP32)
    return absl::InvalidArgumentError(
        "MLP probe model requires INT32 tokens and FP32 logits");
  RETURN_IF_ERROR(outputs[0].Validate());
  const auto shape = outputs[0].dimensions();
  if (shape.size() != 3 || shape[0] != ActivationType::kBatchDimension ||
      shape[1] != dataset.options().context_length ||
      shape[2] < vocabulary_size || shape[2] > std::numeric_limits<int>::max())
    return absl::InvalidArgumentError("MLP probe model logit shape is invalid");
  for (const auto& weight : model.weights())
    if (&weight.executor() != &executor)
      return absl::InvalidArgumentError(
          "MLP probe model belongs to another executor");
  for (size_t sample = 0; sample < dataset.sample_count(); ++sample)
    for (int token : dataset.sample_tokens(sample))
      if (token < 0 || token >= vocabulary_size ||
          token == dataset.options().eos_token)
        return absl::InvalidArgumentError(
            "MLP probe corpus contains invalid or reserved EOS tokens");
  return absl::OkStatus();
}

absl::Status ValidateBf16(cuda::Executor& executor,
                          absl::Span<const ActivationType> types,
                          absl::Span<const Buffer> buffers, int batch_size,
                          int sequence_length, int width) {
  const ActivationType expected(
      DataType::BF16,
      {ActivationType::kBatchDimension, sequence_length, width});
  if (batch_size <= 0 || sequence_length <= 0 || width <= 0 ||
      types.size() != 1 || types[0] != expected || buffers.size() != 1)
    return absl::InvalidArgumentError(
        "MLP probe activation has an unexpected BF16 shape");
  // Only dimensions matter to activation validation; prefix-only tracing has
  // no target buffer and must not fabricate one to reuse this hook.
  const int64_t rows = int64_t{batch_size} * sequence_length;
  if (rows > std::numeric_limits<int>::max() ||
      static_cast<size_t>(width) >
          std::numeric_limits<size_t>::max() / sizeof(uint16_t) / rows)
    return absl::InvalidArgumentError("MLP activation dimensions overflow");
  const size_t bytes = static_cast<size_t>(rows) * width * sizeof(uint16_t);
  if (buffers[0].size_bytes() != bytes || &buffers[0].executor() != &executor)
    return absl::InvalidArgumentError("MLP activation size/executor mismatch");
  return absl::OkStatus();
}

absl::Status CombineHookErrors(const absl::Status& first,
                               const absl::Status& second) {
  if (first.ok())
    return second;
  if (second.ok())
    return first;
  return absl::Status(
      first.code(), absl::StrCat(first.message(), "; additional hook failure: ",
                                 second.ToString()));
}

struct BlockCapture {
  int block;
  int width;
  int feature_width;
  const MlpReplacement* replacement = nullptr;
  std::optional<Buffer> normalized = std::nullopt;
  std::optional<Buffer> features = std::nullopt;
  std::optional<Buffer> output = std::nullopt;
};

class MlpHooks {
 public:
  MlpHooks(cuda::Executor& executor, int batch_size, int sequence_length,
           std::vector<BlockCapture> captures)
      : executor_(executor),
        batch_size_(batch_size),
        sequence_length_(sequence_length),
        captures_(std::move(captures)) {
    hooks_.enter_combinator = [&](cuda::Executor& actual,
                                  absl::string_view name) {
      RETURN_IF_ERROR(CheckExecutor(actual));
      const int block = BlockIndex(name);
      if (block >= 0 && !blocks_seen_.insert(block).second)
        return absl::InvalidArgumentError(
            "duplicate GPT-2 block scope in MLP probe");
      scopes_.emplace_back(name);
      return absl::OkStatus();
    };
    hooks_.exit_combinator = [&](cuda::Executor& actual) {
      RETURN_IF_ERROR(CheckExecutor(actual));
      if (scopes_.empty())
        return absl::FailedPreconditionError("MLP probe scope underflow");
      scopes_.pop_back();
      return absl::OkStatus();
    };
    hooks_.activation_hook = [&](cuda::Executor& actual, absl::string_view name,
                                 absl::Span<const ActivationType> types,
                                 absl::Span<Buffer> outputs) {
      RETURN_IF_ERROR(CheckExecutor(actual));
      return Observe(name, types, outputs);
    };
  }
  MlpHooks(const MlpHooks&) = delete;
  MlpHooks& operator=(const MlpHooks&) = delete;

  LayerHooks& hooks() { return hooks_; }
  const std::vector<BlockCapture>& captures() const { return captures_; }
  const std::set<int>& blocks_seen() const { return blocks_seen_; }

  // Both observers see the original model's event stream. At source sites,
  // trace patches must precede source retention; at branch outputs, live
  // substitution must precede trace patches/capture. A fixed callback ordering
  // for every activation would silently retain an unpatched source buffer.
  LayerHooks ComposeTrace(LayerHooks trace) {
    LayerHooks combined;
    combined.enter_combinator =
        [this, enter = std::move(trace.enter_combinator)](
            cuda::Executor& executor, absl::string_view name) {
          RETURN_IF_ERROR(hooks_.enter_combinator(executor, name));
          if (!enter)
            return absl::OkStatus();
          const auto entered = enter(executor, name);
          if (entered.ok())
            return entered;
          // Failed enter skips its matching exit in Layer. Roll back the
          // first observer here so ancestors can still unwind correctly.
          return CombineHookErrors(entered, hooks_.exit_combinator(executor));
        };
    combined.exit_combinator = [this, exit = std::move(trace.exit_combinator)](
                                   cuda::Executor& executor) {
      const auto traced = exit ? exit(executor) : absl::OkStatus();
      const auto replaced = hooks_.exit_combinator(executor);
      return CombineHookErrors(traced, replaced);
    };
    combined.activation_hook = [this,
                                activation = std::move(trace.activation_hook)](
                                   cuda::Executor& executor,
                                   absl::string_view name,
                                   absl::Span<const ActivationType> types,
                                   absl::Span<Buffer> buffers) {
      const bool branch = name == "mlp" && scopes_.size() == 3 &&
                          scopes_[0] == "gpt2" && scopes_[2] == "ResidualLayer";
      if (branch)
        RETURN_IF_ERROR(hooks_.activation_hook(executor, name, types, buffers));
      if (activation)
        RETURN_IF_ERROR(activation(executor, name, types, buffers));
      if (!branch)
        RETURN_IF_ERROR(hooks_.activation_hook(executor, name, types, buffers));
      return absl::OkStatus();
    };
    combined.attention_probabilities_hook =
        std::move(trace.attention_probabilities_hook);
    combined.gradient_hook = std::move(trace.gradient_hook);
    return combined;
  }

  absl::Status Finish() const {
    if (!scopes_.empty())
      return absl::FailedPreconditionError("MLP probe scopes were not closed");
    for (const auto& capture : captures_) {
      if (!capture.output)
        return absl::NotFoundError(
            absl::StrCat("MLP block was not observed: ", capture.block));
      if (capture.replacement == nullptr &&
          (!capture.normalized || !capture.features))
        return absl::NotFoundError(
            "MLP normalization or GELU site was not observed");
    }
    return absl::OkStatus();
  }

 private:
  absl::Status CheckExecutor(const cuda::Executor& actual) const {
    if (&actual != &executor_)
      return absl::InvalidArgumentError("MLP hook uses another executor");
    return absl::OkStatus();
  }

  absl::Status Observe(absl::string_view name,
                       absl::Span<const ActivationType> types,
                       absl::Span<Buffer> outputs) {
    // The branch's own hook runs after its 'mlp' scope has been popped.
    const bool inner = scopes_.size() == 4 && scopes_[3] == "mlp";
    const bool branch = name == "mlp" && scopes_.size() == 3;
    if ((!inner && !branch) || scopes_[0] != "gpt2" ||
        scopes_[2] != "ResidualLayer")
      return absl::OkStatus();
    const int block = BlockIndex(scopes_[1]);
    auto found =
        std::find_if(captures_.begin(), captures_.end(),
                     [&](const auto& item) { return item.block == block; });
    if (found == captures_.end())
      return absl::OkStatus();
    auto& capture = *found;
    const bool need_norm = capture.replacement == nullptr ||
                           capture.replacement->source == MlpSource::kLayerNorm;
    const bool need_gelu = capture.replacement == nullptr ||
                           capture.replacement->source == MlpSource::kGelu;
    if (inner && name == "LayerNormLayer" && need_norm) {
      RETURN_IF_ERROR(ValidateBf16(executor_, types, outputs, batch_size_,
                                   sequence_length_, capture.width));
      if (capture.normalized)
        return absl::InvalidArgumentError("ambiguous MLP normalization hook");
      capture.normalized = outputs[0];
    }
    if (inner && name == "GeluLayer" && need_gelu) {
      RETURN_IF_ERROR(ValidateBf16(executor_, types, outputs, batch_size_,
                                   sequence_length_, capture.feature_width));
      if (capture.features)
        return absl::InvalidArgumentError("ambiguous MLP GELU hook");
      capture.features = outputs[0];
    }
    if (branch) {
      RETURN_IF_ERROR(ValidateBf16(executor_, types, outputs, batch_size_,
                                   sequence_length_, capture.width));
      if (capture.output)
        return absl::InvalidArgumentError("ambiguous MLP branch output");
      if (capture.replacement != nullptr) {
        const auto& source =
            capture.replacement->source == MlpSource::kLayerNorm
                ? capture.normalized
                : capture.features;
        if (!source)
          return absl::NotFoundError(
              "replacement source was not captured in this forward");
        const Layer& projection = *capture.replacement->projection;
        // No hooks on the replacement projection: observing its internal
        // layers would contaminate the currently active model scope/state.
        ASSIGN_OR_RETURN(auto projected,
                         projection.fwd(executor_, {&*source, 1}, nullptr));
        projected.state = BackwardState{};
        RETURN_IF_ERROR(ValidateBf16(executor_, projection.output_types(),
                                     projected.outputs, batch_size_,
                                     sequence_length_, capture.width));
        outputs[0] = std::move(projected.outputs[0]);
      }
      capture.output = outputs[0];
    }
    return absl::OkStatus();
  }

  cuda::Executor& executor_;
  const int batch_size_;
  const int sequence_length_;
  std::vector<BlockCapture> captures_;
  std::vector<std::string> scopes_;
  std::set<int> blocks_seen_;
  LayerHooks hooks_;
};

absl::Status FindOutputLayers(const BackwardState& state, int enclosing_block,
                              std::vector<const Layer*>& outputs) {
  if (state.layer == nullptr)
    return absl::InvalidArgumentError(
        "MLP capture encountered an unidentified forward state");
  const int named_block = BlockIndex(state.layer->name());
  if (named_block >= 0)
    enclosing_block = named_block;
  if (state.layer->name() == "mlp") {
    if (enclosing_block < 0 ||
        static_cast<size_t>(enclosing_block) >= outputs.size() ||
        outputs[enclosing_block] != nullptr || state.children.size() != 4 ||
        state.children.back().layer == nullptr ||
        state.children.back().layer->name() != "FullyConnectedLayer")
      return absl::InvalidArgumentError(
          "MLP forward state does not expose one original output projection");
    outputs[enclosing_block] = state.children.back().layer;
  }
  for (const auto& child : state.children)
    RETURN_IF_ERROR(FindOutputLayers(child, enclosing_block, outputs));
  return absl::OkStatus();
}

absl::Status CopyOutputWeights(cuda::Executor& executor,
                               const BackwardState& state, int context,
                               std::vector<MlpSamples>& blocks) {
  std::vector<const Layer*> output_layers(blocks.size(), nullptr);
  RETURN_IF_ERROR(FindOutputLayers(state, -1, output_layers));
  struct HostWeights {
    cuda::PageLockedHostArray<float> matrix;
    cuda::PageLockedHostArray<float> bias;
  };
  std::vector<HostWeights> hosts;
  for (size_t block = 0; block < blocks.size(); ++block) {
    if (output_layers[block] == nullptr)
      return absl::NotFoundError("missing original MLP output projection");
    const Layer& layer = *output_layers[block];
    const auto inputs = layer.input_types();
    const auto outputs = layer.output_types();
    const auto& sample = blocks[block];
    if (inputs.size() != 1 || outputs.size() != 1 ||
        inputs[0] !=
            ActivationType(DataType::BF16, {ActivationType::kBatchDimension,
                                            context, sample.feature_width}) ||
        outputs[0] !=
            ActivationType(DataType::BF16, {ActivationType::kBatchDimension,
                                            context, sample.width}))
      return absl::InvalidArgumentError(
          "original MLP output projection has an unexpected signature");
    const auto weights = layer.weights();
    if (weights.size() != 2 ||
        weights[0].size_bytes() != static_cast<size_t>(sample.feature_width) *
                                       sample.width * sizeof(float) ||
        weights[1].size_bytes() !=
            static_cast<size_t>(sample.width) * sizeof(float))
      return absl::InvalidArgumentError(
          "original MLP output projection has malformed FP32 weights");
    ASSIGN_OR_RETURN(auto matrix, QueueDownload<float>(executor, weights[0]));
    ASSIGN_OR_RETURN(auto bias, QueueDownload<float>(executor, weights[1]));
    hosts.push_back({std::move(matrix), std::move(bias)});
  }
  RETURN_IF_ERROR(executor.Synchronize());
  for (size_t block = 0; block < blocks.size(); ++block) {
    auto& sample = blocks[block];
    for (float value : hosts[block].matrix) {
      if (!std::isfinite(value) || !std::isfinite(RoundBf16(value)))
        return absl::DataLossError(
            "original MLP output matrix is nonfinite or overflows BF16");
      sample.effective_output_weights.push_back(RoundBf16(value));
    }
    for (float value : hosts[block].bias) {
      if (!std::isfinite(value))
        return absl::DataLossError("original MLP output bias is nonfinite");
      sample.output_bias.push_back(value);  // Bias is NOT a BF16 MMA operand.
    }
  }
  return absl::OkStatus();
}

absl::Status AddMetrics(cuda::Executor& executor, const Layer& model,
                        const DataBatch& batch, const FwdResult& forward,
                        int vocabulary_size, MlpEvaluation& metrics) {
  RETURN_IF_ERROR(ValidateBatchBuffers(executor, batch, forward.outputs,
                                       model.output_types(),
                                       "MLP probe logits"));
  ASSIGN_OR_RETURN(auto ids, ExtractTop1Ids(executor, forward.outputs[0],
                                            batch.targets, vocabulary_size));
  ASSIGN_OR_RETURN(auto targets, QueueDownload<int>(executor, batch.targets));
  ASSIGN_OR_RETURN(auto predictions, QueueDownload<int>(executor, ids));
  RETURN_IF_ERROR(executor.Synchronize());
  int scored = 0;
  for (int sample = 0; sample < batch.batch_size; ++sample) {
    bool exact = true;
    int64_t sentence_targets = 0;
    int64_t sentence_correct = 0;
    for (int position = 0; position < batch.sequence_length; ++position) {
      const size_t row =
          static_cast<size_t>(sample) * batch.sequence_length + position;
      const int target = targets[row];
      if (target == -1)
        continue;
      if (target < 0 || target >= vocabulary_size || predictions[row] < 0 ||
          predictions[row] >= vocabulary_size)
        return absl::DataLossError(
            "MLP evaluation encountered invalid targets or nonfinite logits");
      ++scored;
      ++metrics.targets;
      ++sentence_targets;
      const bool correct = predictions[row] == target;
      metrics.correct_targets += correct;
      sentence_correct += correct;
      exact &= correct;
    }
    ++metrics.sentences;
    metrics.exact_sentences += exact;
    metrics.targets_per_sentence.push_back(sentence_targets);
    metrics.correct_per_sentence.push_back(sentence_correct);
  }
  ASSIGN_OR_RETURN(const int expected, batch.loss_row_count());
  if (scored != expected)
    return absl::InvalidArgumentError(
        "MLP batch target mask disagrees with supervised row count");
  return absl::OkStatus();
}

absl::Status ValidateBatch(cuda::Executor& executor, const Layer& model,
                           const DataBatch& batch) {
  RETURN_IF_ERROR(ValidateBatchBuffers(executor, batch, {&batch.inputs, 1},
                                       model.input_types(), "MLP probe input"));
  const ActivationType target(DataType::INT32, {ActivationType::kBatchDimension,
                                                batch.sequence_length});
  return ValidateBatchInput(executor, batch, batch.targets, target,
                            "MLP probe targets");
}

absl::Status AppendRealRows(const cuda::PageLockedHostArray<uint16_t>& source,
                            const PaddedLineDataSetIterator& dataset,
                            size_t first_sample, const DataBatch& batch,
                            int width, std::vector<float>& destination) {
  for (int sample = 0; sample < batch.batch_size; ++sample) {
    const size_t length = dataset.sample_tokens(first_sample + sample).size();
    const size_t base =
        static_cast<size_t>(sample) * batch.sequence_length * width;
    for (size_t element = 0; element < length * width; ++element) {
      const float value = ExpandBf16(source[base + element]);
      if (!std::isfinite(value))
        return absl::DataLossError("captured MLP activation is nonfinite");
      destination.push_back(value);
    }
  }
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<MlpCorpusCapture> CaptureGpt2Mlps(
    cuda::Executor& executor, const Layer& model,
    PaddedLineDataSetIterator& dataset, int width, int feature_width,
    int block_count, int vocabulary_size) {
  if (width <= 0 || feature_width <= 0 || block_count < 0 ||
      static_cast<size_t>(feature_width) >
          std::numeric_limits<size_t>::max() / sizeof(float) / width)
    return absl::InvalidArgumentError("invalid MLP capture dimensions");
  RETURN_IF_ERROR(ValidateDataset(executor, model, dataset, vocabulary_size));
  RETURN_IF_ERROR(dataset.Reset());
  MlpCorpusCapture result;
  result.blocks.resize(block_count);
  for (auto& block : result.blocks) {
    block.width = width;
    block.feature_width = feature_width;
  }
  MlpEvaluation metrics;
  size_t first_sample = 0;
  for (size_t batch_index = 0; batch_index < dataset.batches_per_epoch();
       ++batch_index) {
    ASSIGN_OR_RETURN(auto batch, dataset.Next());
    RETURN_IF_ERROR(ValidateBatch(executor, model, batch));
    if (static_cast<size_t>(batch.batch_size) >
        dataset.sample_count() - first_sample)
      return absl::InvalidArgumentError(
          "MLP capture batch exceeds remaining corpus samples");
    std::vector<BlockCapture> captures;
    for (int block = 0; block < block_count; ++block)
      captures.push_back(
          {.block = block, .width = width, .feature_width = feature_width});
    MlpHooks hooks(executor, batch.batch_size, batch.sequence_length,
                   std::move(captures));
    ASSIGN_OR_RETURN(auto forward,
                     model.fwd(executor, {&batch.inputs, 1}, &hooks.hooks()));
    RETURN_IF_ERROR(hooks.Finish());
    if (hooks.blocks_seen().size() != static_cast<size_t>(block_count))
      return absl::InvalidArgumentError(
          "MLP capture block count differs from model");
    if (batch_index == 0)
      RETURN_IF_ERROR(CopyOutputWeights(executor, forward.state,
                                        batch.sequence_length, result.blocks));
    struct HostCapture {
      cuda::PageLockedHostArray<uint16_t> normalized;
      cuda::PageLockedHostArray<uint16_t> features;
      cuda::PageLockedHostArray<uint16_t> output;
    };
    std::vector<HostCapture> host;
    for (const auto& capture : hooks.captures()) {
      ASSIGN_OR_RETURN(auto normalized,
                       QueueDownload<uint16_t>(executor, *capture.normalized));
      ASSIGN_OR_RETURN(auto features,
                       QueueDownload<uint16_t>(executor, *capture.features));
      ASSIGN_OR_RETURN(auto output,
                       QueueDownload<uint16_t>(executor, *capture.output));
      host.push_back(
          {std::move(normalized), std::move(features), std::move(output)});
    }
    forward.state = BackwardState{};
    RETURN_IF_ERROR(
        AddMetrics(executor, model, batch, forward, vocabulary_size, metrics));
    // AddMetrics synchronizes the stream, making all feature staging ready.
    for (int block = 0; block < block_count; ++block) {
      RETURN_IF_ERROR(AppendRealRows(host[block].normalized, dataset,
                                     first_sample, batch, width,
                                     result.blocks[block].normalized_inputs));
      RETURN_IF_ERROR(AppendRealRows(host[block].features, dataset,
                                     first_sample, batch, feature_width,
                                     result.blocks[block].features));
      RETURN_IF_ERROR(AppendRealRows(host[block].output, dataset, first_sample,
                                     batch, width,
                                     result.blocks[block].outputs));
    }
    first_sample += batch.batch_size;
  }
  if (first_sample != dataset.sample_count() ||
      metrics.targets != dataset.supervised_row_count())
    return absl::InvalidArgumentError("MLP capture corpus accounting mismatch");
  result.target_count = metrics.targets;
  result.correct_targets = metrics.correct_targets;
  result.sentences = metrics.sentences;
  result.exact_sentences = metrics.exact_sentences;
  return result;
}

namespace {

absl::StatusOr<std::vector<BlockCapture>> ReplacementPrototypes(
    cuda::Executor& executor, int sequence_length,
    absl::Span<const MlpReplacement> replacements) {
  std::set<int> selected;
  std::vector<BlockCapture> prototypes;
  for (const auto& replacement : replacements) {
    if (replacement.block < 0 || !selected.insert(replacement.block).second ||
        replacement.projection == nullptr ||
        (replacement.source != MlpSource::kLayerNorm &&
         replacement.source != MlpSource::kGelu))
      return absl::InvalidArgumentError("invalid or duplicate MLP replacement");
    const auto inputs = replacement.projection->input_types();
    const auto outputs = replacement.projection->output_types();
    if (inputs.size() != 1 || outputs.size() != 1)
      return absl::InvalidArgumentError(
          "MLP replacement requires one input and output");
    for (const auto& type : {inputs[0], outputs[0]}) {
      RETURN_IF_ERROR(type.Validate());
      const auto dims = type.dimensions();
      if (type.data_type() != DataType::BF16 || dims.size() != 3 ||
          dims[0] != ActivationType::kBatchDimension ||
          dims[1] != sequence_length ||
          dims[2] > std::numeric_limits<int>::max())
        return absl::InvalidArgumentError(
            "MLP replacement requires BF16 [batch, context, width]");
    }
    const int source_width = static_cast<int>(inputs[0].dimensions()[2]);
    const int output_width = static_cast<int>(outputs[0].dimensions()[2]);
    if (replacement.source == MlpSource::kLayerNorm &&
        source_width != output_width)
      return absl::InvalidArgumentError(
          "LayerNorm replacement source/output widths must match");
    for (const auto& weight : replacement.projection->weights())
      if (&weight.executor() != &executor)
        return absl::InvalidArgumentError(
            "MLP replacement weights belong to another executor");
    prototypes.push_back({.block = replacement.block,
                          .width = output_width,
                          .feature_width = source_width,
                          .replacement = &replacement});
  }
  return prototypes;
}

}  // namespace

absl::StatusOr<TokenTraceResult> TraceMlpReplacements(
    cuda::Executor& executor, const Layer& model, absl::Span<const int> prefix,
    absl::Span<const MlpReplacement> replacements,
    const TokenTraceOptions& options) {
  if (replacements.empty())
    return TraceNextToken(executor, model, prefix, options);
  if (options.compose_hooks)
    return absl::InvalidArgumentError(
        "MLP replacement tracing reserves the trace hook composer");
  const auto inputs = model.input_types();
  if (inputs.size() != 1)
    return absl::InvalidArgumentError("MLP trace model must have one input");
  RETURN_IF_ERROR(inputs[0].Validate());
  const auto shape = inputs[0].dimensions();
  if (inputs[0].data_type() != DataType::INT32 || shape.size() != 2 ||
      shape[0] != ActivationType::kBatchDimension ||
      shape[1] > std::numeric_limits<int>::max())
    return absl::InvalidArgumentError(
        "MLP trace model requires INT32 [batch,context]");
  const int context = static_cast<int>(shape[1]);
  ASSIGN_OR_RETURN(auto prototypes,
                   ReplacementPrototypes(executor, context, replacements));
  MlpHooks replacements_hook(executor, 1, context, std::move(prototypes));
  TokenTraceOptions composed = options;
  composed.compose_hooks = [&](LayerHooks trace) {
    return replacements_hook.ComposeTrace(std::move(trace));
  };
  // TraceNextToken validates vocabulary, prefix, all model signatures and
  // executor ownership before the original model's one instrumented forward.
  ASSIGN_OR_RETURN(auto result,
                   TraceNextToken(executor, model, prefix, composed));
  RETURN_IF_ERROR(replacements_hook.Finish());
  return result;
}

absl::StatusOr<MlpEvaluation> EvaluateMlpReplacements(
    cuda::Executor& executor, const Layer& model,
    PaddedLineDataSetIterator& dataset,
    absl::Span<const MlpReplacement> replacements, int vocabulary_size) {
  RETURN_IF_ERROR(ValidateDataset(executor, model, dataset, vocabulary_size));
  ASSIGN_OR_RETURN(
      const auto prototypes,
      ReplacementPrototypes(executor, dataset.options().context_length,
                            replacements));
  RETURN_IF_ERROR(dataset.Reset());
  MlpEvaluation result;
  for (size_t batch_index = 0; batch_index < dataset.batches_per_epoch();
       ++batch_index) {
    ASSIGN_OR_RETURN(auto batch, dataset.Next());
    RETURN_IF_ERROR(ValidateBatch(executor, model, batch));
    MlpHooks hooks(executor, batch.batch_size, batch.sequence_length,
                   prototypes);
    ASSIGN_OR_RETURN(auto forward,
                     model.fwd(executor, {&batch.inputs, 1}, &hooks.hooks()));
    RETURN_IF_ERROR(hooks.Finish());
    forward.state = BackwardState{};
    RETURN_IF_ERROR(
        AddMetrics(executor, model, batch, forward, vocabulary_size, result));
  }
  if (result.sentences != static_cast<int64_t>(dataset.sample_count()) ||
      result.targets != dataset.supervised_row_count())
    return absl::InvalidArgumentError(
        "MLP replacement corpus accounting mismatch");
  return result;
}

absl::StatusOr<MlpGreedyEvaluation> VerifyMlpGreedyCompletions(
    cuda::Executor& executor, const Layer& model,
    PaddedLineDataSetIterator& dataset,
    absl::Span<const MlpReplacement> replacements, int vocabulary_size) {
  RETURN_IF_ERROR(ValidateDataset(executor, model, dataset, vocabulary_size));
  ASSIGN_OR_RETURN(
      const auto prototypes,
      ReplacementPrototypes(executor, dataset.options().context_length,
                            replacements));
  const size_t context = dataset.options().context_length;
  const size_t prompt = dataset.options().prompt_tokens;
  const int eos = dataset.options().eos_token;
  const size_t capacity =
      std::min(dataset.sample_count(),
               static_cast<size_t>(dataset.options().batch_size));
  if (capacity == 0 ||
      context > static_cast<size_t>(std::numeric_limits<int>::max()) / capacity)
    return absl::InvalidArgumentError("MLP greedy batch rows exceed int range");
  const size_t rows = capacity * context;
  ASSIGN_OR_RETURN(auto inputs,
                   cuda::PageLockedHostArray<int>::Allocate(executor, rows));
  ASSIGN_OR_RETURN(auto score_mask,
                   cuda::PageLockedHostArray<int>::Allocate(executor, rows));
  ASSIGN_OR_RETURN(auto predictions,
                   cuda::PageLockedHostArray<int>::Allocate(executor, rows));
  ASSIGN_OR_RETURN(auto device_inputs,
                   Buffer::Allocate(executor, inputs.size_bytes()));
  ASSIGN_OR_RETURN(auto device_mask,
                   Buffer::Allocate(executor, score_mask.size_bytes()));
  MlpGreedyEvaluation result;
  result.exact_per_sentence.assign(dataset.sample_count(), false);
  result.first_mismatch_per_sentence.resize(dataset.sample_count());
  for (size_t first = 0; first < dataset.sample_count(); first += capacity) {
    const size_t count = std::min(capacity, dataset.sample_count() - first);
    std::fill(inputs.begin(), inputs.end(), eos);
    std::vector<bool> active(capacity, false);
    std::vector<size_t> used(capacity, prompt);
    for (size_t sample = 0; sample < count; ++sample) {
      const auto original = dataset.sample_tokens(first + sample);
      std::copy(original.begin(), original.begin() + prompt,
                inputs.begin() + sample * context);
      active[sample] = true;
    }
    size_t remaining = count;
    while (remaining != 0) {
      std::fill(score_mask.begin(), score_mask.end(), -1);
      for (size_t sample = 0; sample < count; ++sample)
        if (active[sample])
          // Zero merely enables this row for ExtractTop1Ids. Gold targets do
          // not enter either the model input or its prediction-selection mask.
          score_mask[sample * context + used[sample] - 1] = 0;
      RETURN_IF_ERROR(cuda::CudaStatus(
          cudaMemcpyAsync(device_inputs.data(), inputs.data(),
                          inputs.size_bytes(), cudaMemcpyHostToDevice,
                          executor.stream()),
          "upload generated MLP verification prefixes"));
      RETURN_IF_ERROR(cuda::CudaStatus(
          cudaMemcpyAsync(device_mask.data(), score_mask.data(),
                          score_mask.size_bytes(), cudaMemcpyHostToDevice,
                          executor.stream()),
          "upload MLP generation score mask"));
      const DataBatch batch{
          device_inputs, device_mask, static_cast<int32_t>(capacity),
          static_cast<int32_t>(context), static_cast<int32_t>(remaining)};
      RETURN_IF_ERROR(ValidateBatch(executor, model, batch));
      MlpHooks hooks(executor, batch.batch_size, batch.sequence_length,
                     prototypes);
      ASSIGN_OR_RETURN(auto forward,
                       model.fwd(executor, {&batch.inputs, 1}, &hooks.hooks()));
      RETURN_IF_ERROR(hooks.Finish());
      forward.state = BackwardState{};
      RETURN_IF_ERROR(ValidateBatchBuffers(executor, batch, forward.outputs,
                                           model.output_types(),
                                           "MLP greedy logits"));
      ASSIGN_OR_RETURN(auto ids, ExtractTop1Ids(executor, forward.outputs[0],
                                                device_mask, vocabulary_size));
      RETURN_IF_ERROR(cuda::CudaStatus(
          cudaMemcpyAsync(predictions.data(), ids.data(),
                          predictions.size_bytes(), cudaMemcpyDeviceToHost,
                          executor.stream()),
          "download MLP greedy predictions"));
      RETURN_IF_ERROR(executor.Synchronize());
      for (size_t sample = 0; sample < count; ++sample) {
        if (!active[sample])
          continue;
        const int predicted = predictions[sample * context + used[sample] - 1];
        if (predicted < 0 || predicted >= vocabulary_size)
          return absl::DataLossError(
              "MLP greedy verification encountered nonfinite or invalid "
              "logits");
        ++result.generated_targets;
        const auto original = dataset.sample_tokens(first + sample);
        const int expected =
            used[sample] == original.size() ? eos : original[used[sample]];
        if (predicted != expected || predicted == eos) {
          if (predicted != expected)
            result.first_mismatch_per_sentence[first + sample] =
                MlpGreedyMismatch{static_cast<int>(used[sample]), predicted,
                                  expected};
          result.exact_sentences += predicted == expected;
          result.exact_per_sentence[first + sample] = predicted == expected;
          active[sample] = false;
          --remaining;
          continue;
        }
        if (used[sample] >= context)
          return absl::DataLossError(
              "MLP greedy verification exceeded model context");
        inputs[sample * context + used[sample]] = predicted;
        ++used[sample];
      }
      // Synchronization above also makes both pinned H2D arrays safe to reuse.
    }
    result.sentences += count;
  }
  for (size_t sentence = 0; sentence < dataset.sample_count(); ++sentence)
    if (result.exact_per_sentence[sentence] ==
        result.first_mismatch_per_sentence[sentence].has_value())
      return absl::InternalError("greedy mismatch accounting is inconsistent");
  return result;
}

}  // namespace pluto::llm::one_shot_memorizer
