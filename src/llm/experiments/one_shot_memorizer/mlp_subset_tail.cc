#include "src/llm/experiments/one_shot_memorizer/mlp_subset_tail.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <memory>
#include <string>
#include <utility>

#include "absl/container/flat_hash_set.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "src/llm/experiments/one_shot_memorizer/sentence_ablation.h"
#include "src/llm/layers/combinators.h"
#include "src/llm/layers/embedding.h"
#include "src/llm/layers/fully_connected.h"
#include "src/llm/layers/norm.h"
#include "src/util/status_macros.h"

namespace pluto::llm::one_shot_memorizer {
namespace {

// ResidualLayer supplies the original residual to this adapter and adds the
// returned projection with its ordinary BF16 AddKernel. The adapter's branch
// input is instead the cached GELU: no model computation is approximated and
// no special-purpose GPU arithmetic is introduced.
class CachedProjection final : public Layer {
 public:
  explicit CachedProjection(std::unique_ptr<FullyConnectedLayer> projection)
      : projection_(std::move(projection)) {}
  const Buffer* masked_gelu = nullptr;  // Scoped to the current forward call.
  absl::string_view name() const override { return "CachedProjection"; }
  absl::Span<const ActivationType> input_types() const override {
    return absl::MakeConstSpan(&type_, 1);
  }
  absl::Span<const ActivationType> output_types() const override {
    return input_types();
  }
  absl::Span<Buffer> weights() override { return projection_->weights(); }
  DataType output_type() const override { return DataType::BF16; }

 private:
  absl::StatusOr<FwdResult> fwd_impl(cuda::Executor& executor,
                                     absl::Span<const Buffer> inputs,
                                     LayerHooks*) const override {
    if (inputs.size() != 1 || masked_gelu == nullptr ||
        masked_gelu->size_bytes() != inputs[0].size_bytes() * 4 ||
        &masked_gelu->executor() != &executor)
      return absl::InvalidArgumentError("cached projection row count mismatch");
    return projection_->fwd(executor, {*masked_gelu});
  }
  absl::StatusOr<BufferVec> bwd_impl(cuda::Executor&, absl::Span<const Buffer>,
                                     BackwardState, LayerHooks*) override {
    return absl::UnimplementedError("cached MLP tail is forward-only");
  }
  std::unique_ptr<FullyConnectedLayer> projection_;
  const ActivationType type_{DataType::BF16,
                             {ActivationType::kBatchDimension, 1, 16}};
};

absl::Status CopyMaster(cuda::Executor& executor, const Buffer& source,
                        Buffer& destination) {
  if (source.size_bytes() != destination.size_bytes() ||
      &source.executor() != &executor || &destination.executor() != &executor)
    return absl::FailedPreconditionError("tail master shape/executor mismatch");
  return cuda::CudaStatus(
      cudaMemcpyAsync(destination.data(), source.data(), source.size_bytes(),
                      cudaMemcpyDeviceToDevice, executor.stream()),
      "copy cached MLP tail master");
}

}  // namespace

struct FinalMlpSubsetTail::Impl {
  cuda::Executor& executor;
  int vocabulary;
  // The head borrows its embedding, so destruction order matters.
  std::unique_ptr<EmbeddingLookupLayer> embedding;
  std::unique_ptr<LanguageModelingHeadLayer> head;
  std::unique_ptr<ResidualLayer> residual;
  std::unique_ptr<LayerNormLayer> norm;
  CachedProjection* cached;  // Owned by residual.
};

FinalMlpSubsetTail::FinalMlpSubsetTail(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
FinalMlpSubsetTail::~FinalMlpSubsetTail() = default;

int FinalMlpSubsetTail::vocab_size() const { return impl_->vocabulary; }
int FinalMlpSubsetTail::logit_stride() const {
  return impl_->embedding->padded_vocab_size();
}

absl::StatusOr<std::unique_ptr<FinalMlpSubsetTail>> FinalMlpSubsetTail::Create(
    cuda::Executor& executor, const Layer& model, const Gpt2Config& config) {
  RETURN_IF_ERROR(config.Validate());
  if (config.transformer_block_count < 1 || config.model_width != 16 ||
      config.feed_forward_width != 64 || config.pad_vocabulary ||
      model.output_type() != DataType::BF16)
    return absl::InvalidArgumentError(
        "cached MLP tail requires BF16 GPT-2, width16/ff64, unpadded "
        "embeddings and at least one block");
  const int stride = ((config.vocabulary_size + 15) / 16) * 16;
  const ActivationType input(
      DataType::INT32, {ActivationType::kBatchDimension, kGpt2ContextLength});
  const ActivationType output(DataType::FP32, {ActivationType::kBatchDimension,
                                               kGpt2ContextLength, stride});
  if (model.input_types().size() != 1 || model.output_types().size() != 1 ||
      model.input_types()[0] != input || model.output_types()[0] != output)
    return absl::InvalidArgumentError(
        "cached MLP tail model signature mismatch");
  ASSIGN_OR_RETURN(auto layout, BuildGpt2ParameterLayout(
                                    {.vocabulary_size = config.vocabulary_size,
                                     .model_width = 16,
                                     .feed_forward_width = 64,
                                     .context_length = kGpt2ContextLength,
                                     .transformer_block_count =
                                         config.transformer_block_count}));
  const auto weights = model.weights();
  if (weights.size() != layout.size() + 1 ||
      weights.back().data() != weights.front().data() ||
      weights.back().size_bytes() != weights.front().size_bytes() ||
      &weights.back().executor() != &executor)
    return absl::FailedPreconditionError(
        "cached MLP tail needs the tied GPT-2 inventory");
  absl::flat_hash_set<const void*> seen;
  for (size_t index = 0; index < layout.size(); ++index)
    if (weights[index].size_bytes() !=
            layout[index].element_count * sizeof(float) ||
        &weights[index].executor() != &executor ||
        !seen.insert(weights[index].data()).second)
      return absl::FailedPreconditionError(
          absl::StrCat("cached MLP tail master mismatch at ", index));
  const size_t contraction = layout.size() - 4;
  const std::string prefix =
      absl::StrCat("transformer_block_", config.transformer_block_count - 1,
                   ".mlp.contraction.");
  if (layout[contraction].name != prefix + "weight" ||
      layout[contraction + 1].name != prefix + "bias" ||
      layout[contraction + 2].name != "final_layer_norm.gamma" ||
      layout[contraction + 3].name != "final_layer_norm.beta")
    return absl::FailedPreconditionError(
        "cached MLP tail layout roles mismatch");

  ASSIGN_OR_RETURN(auto projection, FullyConnectedLayer::Create(
                                        executor, 64, 16, DataType::BF16));
  RETURN_IF_ERROR(
      CopyMaster(executor, weights[contraction], projection->weights()[0]));
  RETURN_IF_ERROR(
      CopyMaster(executor, weights[contraction + 1], projection->weights()[1]));
  auto cached = absl::make_unique<CachedProjection>(std::move(projection));
  CachedProjection* cached_pointer = cached.get();
  ASSIGN_OR_RETURN(auto residual, ResidualLayer::Create(std::move(cached)));
  ASSIGN_OR_RETURN(auto norm,
                   LayerNormLayer::Create(executor, 16, 1e-5f, DataType::BF16));
  RETURN_IF_ERROR(
      CopyMaster(executor, weights[contraction + 2], norm->weights()[0]));
  RETURN_IF_ERROR(
      CopyMaster(executor, weights[contraction + 3], norm->weights()[1]));
  ASSIGN_OR_RETURN(auto embedding, EmbeddingLookupLayer::Create(
                                       executor, config.vocabulary_size, 16,
                                       DataType::BF16, 1, false));
  RETURN_IF_ERROR(
      CopyMaster(executor, weights.front(), embedding->weights()[0]));
  ASSIGN_OR_RETURN(auto head,
                   LanguageModelingHeadLayer::Create(embedding.get()));
  auto impl = absl::make_unique<Impl>(Impl{
      executor, config.vocabulary_size, std::move(embedding), std::move(head),
      std::move(residual), std::move(norm), cached_pointer});
  return absl::WrapUnique(new FinalMlpSubsetTail(std::move(impl)));
}

absl::StatusOr<cuda::PageLockedHostArray<float>> FinalMlpSubsetTail::Evaluate(
    cuda::Executor& executor, absl::Span<const FinalMlpSubsetInput> inputs) {
  if (&executor != &impl_->executor || inputs.empty())
    return absl::InvalidArgumentError(
        "cached MLP tail needs its executor and nonempty input");
  const size_t max_rows =
      std::numeric_limits<int>::max() / std::max(64, logit_stride());
  if (inputs.size() > max_rows / 16 * 16)
    return absl::OutOfRangeError(
        "cached MLP tail batch exceeds GPU element range");
  for (const auto& input : inputs)
    if (input.gelu.size() != 64 || input.residual.size() != 16)
      return absl::InvalidArgumentError(
          "cached MLP tail needs GELU64/residual16 rows");
  const size_t rows = (inputs.size() + 15) / 16 * 16;
  ASSIGN_OR_RETURN(
      auto host_gelu,
      cuda::PageLockedHostArray<uint16_t>::Allocate(executor, rows * 64));
  ASSIGN_OR_RETURN(
      auto host_residual,
      cuda::PageLockedHostArray<uint16_t>::Allocate(executor, rows * 16));
  std::fill(host_gelu.begin(), host_gelu.end(), 0);
  std::fill(host_residual.begin(), host_residual.end(), 0);
  for (size_t row = 0; row < inputs.size(); ++row) {
    ASSIGN_OR_RETURN(auto masked, ApplyBf16FeatureSubset(inputs[row].gelu,
                                                         inputs[row].keep_mask));
    std::copy(masked.begin(), masked.end(), host_gelu.data() + row * 64);
    std::copy(inputs[row].residual.begin(), inputs[row].residual.end(),
              host_residual.data() + row * 16);
  }
  ASSIGN_OR_RETURN(auto gelu,
                   Buffer::Allocate(executor, host_gelu.size_bytes()));
  ASSIGN_OR_RETURN(auto residual,
                   Buffer::Allocate(executor, host_residual.size_bytes()));
  RETURN_IF_ERROR(cuda::CudaStatus(
      cudaMemcpyAsync(gelu.data(), host_gelu.data(), host_gelu.size_bytes(),
                      cudaMemcpyHostToDevice, executor.stream()),
      "upload cached masked GELU"));
  RETURN_IF_ERROR(cuda::CudaStatus(
      cudaMemcpyAsync(residual.data(), host_residual.data(),
                      host_residual.size_bytes(), cudaMemcpyHostToDevice,
                      executor.stream()),
      "upload cached residual"));
  impl_->cached->masked_gelu = &gelu;
  auto added = impl_->residual->fwd(executor, {residual});
  impl_->cached->masked_gelu = nullptr;
  if (!added.ok())
    return added.status();
  ASSIGN_OR_RETURN(auto normalized, impl_->norm->fwd(executor, added->outputs));
  ASSIGN_OR_RETURN(auto logits, impl_->head->fwd(executor, normalized.outputs));
  const size_t elements = rows * logit_stride();
  if (logits.outputs.size() != 1 ||
      logits.outputs[0].size_bytes() != elements * sizeof(float))
    return absl::InternalError("cached MLP tail returned malformed logits");
  ASSIGN_OR_RETURN(
      auto host, cuda::PageLockedHostArray<float>::Allocate(executor, elements));
  RETURN_IF_ERROR(cuda::CudaStatus(
      cudaMemcpyAsync(host.data(), logits.outputs[0].data(), host.size_bytes(),
                      cudaMemcpyDeviceToHost, executor.stream()),
      "download cached tail logits"));
  RETURN_IF_ERROR(executor.Synchronize());
  return host;
}

}  // namespace pluto::llm::one_shot_memorizer
