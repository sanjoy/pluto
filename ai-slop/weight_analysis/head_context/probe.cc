#include "ai-slop/weight_analysis/head_context/probe.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "src/cuda/page_locked_host_array.h"
#include "src/llm/layers/attention.h"
#include "src/llm/layers/combinators.h"
#include "src/llm/layers/embedding.h"
#include "src/llm/layers/fully_connected.h"
#include "src/llm/layers/gelu.h"
#include "src/llm/layers/norm.h"
#include "src/llm/recipes/gpt2.h"
#include "src/util/status_macros.h"

namespace pluto::weight_analysis::head_context {
namespace {
constexpr int kWidth = llm::kGpt2ModelWidth;
constexpr int kContext = llm::kGpt2ContextLength;
constexpr int kPadded = llm::kGpt2PaddedVocabularySize;
constexpr int kVocabulary = llm::kGpt2VocabularySize;
constexpr Geometry kGeometry{llm::kGpt2TransformerBlockCount, kContext,
                             llm::kGpt2AttentionHeads,
                             llm::kGpt2AttentionHeadDimension};
static_assert(kGeometry.blocks == 8 && kContext == 1024 && kWidth == 512 &&
              kGeometry.heads == 8 && kGeometry.head_dimension == 64 &&
              llm::kGpt2FeedForwardWidth == 2048 && kVocabulary == 50257 &&
              kPadded == 50272);
using Bytes = cuda::PageLockedHostArray<uint8_t>;

absl::Status Same(const void* actual, const void* expected, size_t bytes,
                  const std::string& label) {
  if (std::memcmp(actual, expected, bytes) != 0)
    return absl::DataLossError(absl::StrCat(label, " bytes differ"));
  return absl::OkStatus();
}

template <class T>
absl::StatusOr<cuda::PageLockedHostArray<T>> Download(
    cuda::Executor& executor, const cuda::Buffer& buffer) {
  if (&buffer.executor() != &executor || buffer.size_bytes() == 0 ||
      buffer.size_bytes() % sizeof(T) != 0) {
    return absl::InvalidArgumentError("invalid head-context download");
  }
  ASSIGN_OR_RETURN(auto result, cuda::PageLockedHostArray<T>::Allocate(
                                    executor, buffer.size_bytes() / sizeof(T)));
  RETURN_IF_ERROR(cuda::CudaStatus(
      cudaMemcpyAsync(result.data(), buffer.data(), buffer.size_bytes(),
                      cudaMemcpyDeviceToHost, executor.stream()),
      "download head-context evidence"));
  // Synchronize while pinned storage is alive, including if execution fails.
  RETURN_IF_ERROR(executor.Synchronize());
  return result;
}

absl::Status FiniteFloats(absl::Span<const float> values) {
  if (!std::all_of(values.begin(), values.end(),
                   [](float x) { return std::isfinite(x); })) {
    return absl::DataLossError("nonfinite head-context weights or logits");
  }
  return absl::OkStatus();
}

absl::Status FiniteBf16(absl::Span<const uint16_t> values) {
  if (!std::all_of(values.begin(), values.end(),
                   [](uint16_t bits) { return (bits & 0x7f80) != 0x7f80; })) {
    return absl::DataLossError("nonfinite head-context activation");
  }
  return absl::OkStatus();
}

absl::Status Shape(cuda::Executor& executor, const cuda::Buffer& buffer,
                   size_t bytes) {
  if (&buffer.executor() != &executor || buffer.size_bytes() != bytes) {
    return absl::InvalidArgumentError(
        "head-context buffer shape/executor differs");
  }
  return absl::OkStatus();
}

// New-package-local validation is necessary: the older diagnostic helper
// targets are private, and their BUILD files are frozen by live observers.
// Validate the entire tree before indexing any private recipe BackwardState
// fields.
absl::Status ValidateState(const llm::BackwardState& state) {
  const auto bad = [] {
    return absl::InvalidArgumentError(
        "unexpected full GPT-2 BackwardState layout");
  };
  if (!state.intermediates.empty() || state.children.size() != 12 ||
      state.children[0].intermediates.size() != 1 ||
      !state.children[0].children.empty() ||
      !state.children[1].intermediates.empty() ||
      !state.children[1].children.empty() ||
      state.children[10].intermediates.size() != 1 ||
      !state.children[10].children.empty() ||
      state.children[11].intermediates.size() != 1 ||
      !state.children[11].children.empty())
    return bad();
  for (int block = 0; block < 8; ++block) {
    const auto& body = state.children[block + 2];
    if (!body.intermediates.empty() || body.children.size() != 2)
      return bad();
    for (int branch = 0; branch < 2; ++branch) {
      const auto& residual = body.children[branch];
      if (residual.intermediates.size() != 1 || residual.children.size() != 1)
        return bad();
      const auto& composed = residual.children[0];
      if (!composed.intermediates.empty() || composed.children.size() != 4)
        return bad();
      for (int leaf = 0; leaf < 4; ++leaf) {
        const auto& saved = composed.children[leaf];
        if (!saved.children.empty() ||
            saved.intermediates.size() != (branch == 0 && leaf == 2 ? 2 : 1))
          return bad();
      }
    }
  }
  return absl::OkStatus();
}

std::vector<size_t> WeightSizes() {
  std::vector<size_t> elements{static_cast<size_t>(kPadded) * kWidth,
                               static_cast<size_t>(kContext) * kWidth};
  for (int block = 0; block < 8; ++block) {
    for (size_t size :
         {512ULL, 512ULL, 512ULL * 1536, 1536ULL, 512ULL * 512, 512ULL, 512ULL,
          512ULL, 512ULL * 2048, 2048ULL, 2048ULL * 512, 512ULL}) {
      elements.push_back(size);
    }
  }
  elements.push_back(kWidth);
  elements.push_back(kWidth);
  for (auto& size : elements)
    size *= sizeof(float);
  return elements;
}

absl::Status CopyWeights(cuda::Executor& executor,
                         absl::Span<const cuda::Buffer> original,
                         absl::Span<cuda::Buffer> copies) {
  if (original.size() != copies.size())
    return absl::InvalidArgumentError("tail weight count differs");
  for (size_t i = 0; i < original.size(); ++i) {
    RETURN_IF_ERROR(Shape(executor, copies[i], original[i].size_bytes()));
    if (&original[i].executor() != &executor ||
        original[i].data() == copies[i].data()) {
      return absl::InvalidArgumentError(
          "tail weights must be separate allocations");
    }
  }
  for (size_t i = 0; i < original.size(); ++i) {
    RETURN_IF_ERROR(cuda::CudaStatus(
        cudaMemcpyAsync(copies[i].data(), original[i].data(),
                        original[i].size_bytes(), cudaMemcpyDeviceToDevice,
                        executor.stream()),
        "copy independent head-context tail weights"));
  }
  return absl::OkStatus();
}

// Feed the genuine native ResidualLayer instead of approximating its BF16
// addition by subtracting saved states or implementing a CPU residual add.
class ConstantBranch final : public llm::Layer {
 public:
  explicit ConstantBranch(cuda::Buffer branch) : branch_(std::move(branch)) {}

  absl::Span<cuda::Buffer> weights() override { return {}; }
  llm::DataType output_type() const override { return llm::DataType::BF16; }

 private:
  absl::StatusOr<llm::FwdResult> fwd_impl(
      cuda::Executor& executor,
      absl::Span<const cuda::Buffer> inputs) const override {
    llm::BackwardState state;
    if (inputs.size() != 1)
      return absl::InvalidArgumentError("invalid constant branch");
    RETURN_IF_ERROR(Shape(executor, branch_, inputs[0].size_bytes()));
    if (&inputs[0].executor() != &executor)
      return absl::InvalidArgumentError("constant branch executor differs");
    state.intermediates.clear();
    state.children.clear();
    return llm::FwdResult{{std::move(branch_)}, std::move(state)};
  }
  absl::StatusOr<llm::BufferVec> bwd_impl(cuda::Executor&,
                                          absl::Span<const cuda::Buffer>,
                                          llm::BackwardState) override {
    return absl::UnimplementedError("head-context probe is forward only");
  }

  cuda::Buffer branch_;
};

absl::StatusOr<std::unique_ptr<llm::ComposedLayer>> Branch(
    cuda::Executor& executor, absl::Span<const cuda::Buffer> weights,
    bool attention) {
  llm::ComposedLayerBuilder builder;
  RETURN_IF_ERROR(builder.add(llm::LayerNormLayer::Create(
      executor, kWidth, 1e-5f, llm::DataType::BF16)));
  const int middle = attention ? 3 * kWidth : llm::kGpt2FeedForwardWidth;
  RETURN_IF_ERROR(builder.add(llm::FullyConnectedLayer::Create(
      executor, kWidth, middle, llm::DataType::BF16)));
  if (attention) {
    RETURN_IF_ERROR(builder.add(llm::AttentionLayer::Create(
        executor, kContext, kGeometry.heads, kWidth, llm::DataType::BF16)));
  } else {
    RETURN_IF_ERROR(
        builder.add(llm::GeluLayer::Create(executor, llm::DataType::BF16)));
  }
  RETURN_IF_ERROR(builder.add(llm::FullyConnectedLayer::Create(
      executor, attention ? kWidth : middle, kWidth, llm::DataType::BF16)));
  ASSIGN_OR_RETURN(auto branch, builder.create());
  RETURN_IF_ERROR(CopyWeights(executor, weights, branch->weights()));
  return branch;
}

struct Snapshot {
  cuda::Buffer buffer;
  Bytes bytes;
  std::string name;
};
}  // namespace

struct Probe::Impl {
  Impl(cuda::Executor& executor, int block, int rows,
       const cuda::Buffer& before, const cuda::Buffer& context,
       const cuda::Buffer& logits)
      : executor(executor),
        block(block),
        rows(rows),
        before(before),
        context(context),
        logits(logits) {}

  absl::Status Save(const cuda::Buffer& buffer, std::string name) {
    ASSIGN_OR_RETURN(auto bytes, Download<uint8_t>(executor, buffer));
    originals.push_back({buffer, std::move(bytes), std::move(name)});
    return absl::OkStatus();
  }

  absl::Status Verify() const {
    for (const auto& saved : originals) {
      ASSIGN_OR_RETURN(auto actual, Download<uint8_t>(executor, saved.buffer));
      RETURN_IF_ERROR(Same(actual.data(), saved.bytes.data(),
                           actual.size_bytes(), saved.name));
    }
    return absl::OkStatus();
  }

  absl::StatusOr<cuda::Buffer> Replay(const cuda::Buffer& changed) const {
    llm::BackwardState projection_state, residual_state, tail_state, norm_state,
        head_state;
    ASSIGN_OR_RETURN(
        auto projected_fwd,
        projection->fwd(executor, absl::MakeConstSpan(&changed, 1)));
        auto projected = std::move(projected_fwd.outputs[0]);
        projection_state = std::move(projected_fwd.state);
        llm::ResidualLayer residual(
            std::make_unique<ConstantBranch>(std::move(projected)));
    ASSIGN_OR_RETURN(auto after_fwd,
                     residual.fwd(executor, absl::MakeConstSpan(&before, 1)));
        auto after = std::move(after_fwd.outputs[0]);
        residual_state = std::move(after_fwd.state);
    ASSIGN_OR_RETURN(auto final_fwd,
                     tail->fwd(executor, absl::MakeConstSpan(&after, 1)));
        auto final = std::move(final_fwd.outputs[0]);
        tail_state = std::move(final_fwd.state);
    ASSIGN_OR_RETURN(auto normalized_fwd,
                     norm->fwd(executor, absl::MakeConstSpan(&final, 1)));
        auto normalized = std::move(normalized_fwd.outputs[0]);
        norm_state = std::move(normalized_fwd.state);
    ASSIGN_OR_RETURN(auto fwd_return_result,
                     head->fwd(executor, absl::MakeConstSpan(&normalized, 1)));
        return std::move(fwd_return_result.outputs[0]);
  }

  cuda::Executor& executor;
  int block, rows;
  cuda::Buffer before, context, logits;
  cuda::PageLockedHostArray<uint16_t> context_bytes;
  cuda::PageLockedHostArray<float> logit_values;
  std::vector<Snapshot> originals;
  std::unique_ptr<llm::FullyConnectedLayer> projection;
  std::unique_ptr<llm::ComposedLayer> tail;
  // Declaration order keeps the tied head's embedding alive until head dies.
  std::unique_ptr<llm::EmbeddingLookupLayer> embedding;
  std::unique_ptr<llm::LayerNormLayer> norm;
  std::unique_ptr<llm::LanguageModelingHeadLayer> head;
};

Probe::Probe(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
Probe::~Probe() = default;

absl::StatusOr<std::unique_ptr<Probe>> Probe::Create(
    cuda::Executor& executor, const llm::Layer& production_model,
    const cuda::Buffer& tokens, const llm::BackwardState& state,
    const cuda::Buffer& logits, int block) {
  RETURN_IF_ERROR(ValidateSelection(
      kGeometry, {block, 0, 0, 0, QueryScope::kAllQueries, 1}, kContext));
  RETURN_IF_ERROR(ValidateState(state));
  if (production_model.output_type() != llm::DataType::BF16)
    return absl::InvalidArgumentError("requires native BF16 GPT-2");
  const auto traversal = production_model.weights();
  if (traversal.size() != 101 || traversal[0].data() != traversal[100].data() ||
      traversal[0].size_bytes() != traversal[100].size_bytes() ||
      &traversal[100].executor() != &executor) {
    return absl::InvalidArgumentError(
        "expected exact 100-weight tied GPT-2 traversal");
  }
  const auto weights = traversal.subspan(0, 100);
  const auto sizes = WeightSizes();
  std::set<const void*> addresses;
  for (size_t i = 0; i < weights.size(); ++i) {
    RETURN_IF_ERROR(Shape(executor, weights[i], sizes[i]));
    if (!addresses.insert(weights[i].data()).second)
      return absl::InvalidArgumentError("unexpected GPT-2 weight alias");
  }
  const auto& branch = state.children[block + 2].children[0];
  const auto& before = branch.intermediates[0];
  const auto& qkv = branch.children[0].children[2].intermediates[0];
  const auto& context = branch.children[0].children[2].intermediates[1];
  const size_t rows_wide = context.size_bytes() / (kWidth * sizeof(uint16_t));
  if (context.size_bytes() % (kWidth * sizeof(uint16_t)) || rows_wide == 0 ||
      rows_wide > std::numeric_limits<int>::max() / kPadded) {
    return absl::InvalidArgumentError("invalid native full-row geometry");
  }
  const int rows = static_cast<int>(rows_wide);
  RETURN_IF_ERROR(ValidateSelection(
      kGeometry, {block, 0, 0, 0, QueryScope::kAllQueries, 1}, rows));
  RETURN_IF_ERROR(Shape(executor, tokens, rows_wide * sizeof(int32_t)));
  RETURN_IF_ERROR(Shape(executor, before, rows_wide * kWidth * 2));
  RETURN_IF_ERROR(Shape(executor, qkv, rows_wide * kWidth * 3 * 2));
  RETURN_IF_ERROR(Shape(executor, context, rows_wide * kWidth * 2));
  RETURN_IF_ERROR(Shape(executor, logits, rows_wide * kPadded * 4));
  if (state.children[0].intermediates[0].data() != tokens.data()) {
    return absl::InvalidArgumentError(
        "BackwardState does not retain the supplied input tokens");
  }
  auto impl =
      std::make_unique<Impl>(executor, block, rows, before, context, logits);
  ASSIGN_OR_RETURN(auto token_ids, Download<int32_t>(executor, tokens));
  if (!std::all_of(token_ids.begin(), token_ids.end(),
                   [](int32_t id) { return id >= 0 && id < kVocabulary; })) {
    return absl::InvalidArgumentError(
        "token outside native logical vocabulary");
  }
  ASSIGN_OR_RETURN(impl->context_bytes, Download<uint16_t>(executor, context));
  RETURN_IF_ERROR(FiniteBf16(impl->context_bytes.span()));
  ASSIGN_OR_RETURN(impl->logit_values, Download<float>(executor, logits));
  RETURN_IF_ERROR(FiniteFloats(impl->logit_values.span()));
  // The production mask uses finite -FLT_MAX, not IEEE negative infinity.
  // Logical-vocabulary scores must still exclude these physical pad columns.
  for (int row = 0; row < rows; ++row) {
    for (int token = kVocabulary; token < kPadded; ++token) {
      if (impl->logit_values[static_cast<size_t>(row) * kPadded + token] !=
          -std::numeric_limits<float>::max()) {
        return absl::DataLossError("native padded logits differ from -FLT_MAX");
      }
    }
  }
  for (const auto* activation : {&before, &qkv}) {
    ASSIGN_OR_RETURN(auto bytes, Download<uint16_t>(executor, *activation));
    RETURN_IF_ERROR(FiniteBf16(bytes.span()));
  }
  RETURN_IF_ERROR(impl->Save(tokens, "original tokens"));
  RETURN_IF_ERROR(impl->Save(before, "original pre-attention residual"));
  RETURN_IF_ERROR(impl->Save(qkv, "original QKV"));
  RETURN_IF_ERROR(impl->Save(context, "original context"));
  RETURN_IF_ERROR(impl->Save(logits, "original full logits"));
  for (size_t i = 0; i < weights.size(); ++i) {
    ASSIGN_OR_RETURN(auto values, Download<float>(executor, weights[i]));
    RETURN_IF_ERROR(FiniteFloats(values.span()));
    RETURN_IF_ERROR(
        impl->Save(weights[i], absl::StrCat("original weight ", i)));
  }
  // Also bind the supplied state/clean outputs to this exact model and input,
  // not merely to a mathematically compatible captured downstream state.
  // This full production replay is separate from the tail identity below.
  {
    ASSIGN_OR_RETURN(
        auto repeat_logits_fwd,
        production_model.fwd(executor, absl::MakeConstSpan(&tokens, 1)));
        if (repeat_logits_fwd.outputs.size() != 1) {
          return absl::FailedPreconditionError(
              "production model must return one logits tensor");
        }
        auto repeat_logits = std::move(repeat_logits_fwd.outputs[0]);

        RETURN_IF_ERROR(ValidateState(repeat_logits_fwd.state));
        RETURN_IF_ERROR(Shape(executor, repeat_logits, logits.size_bytes()));
    ASSIGN_OR_RETURN(auto repeated, Download<float>(executor, repeat_logits));
        RETURN_IF_ERROR(Same(repeated.data(), impl->logit_values.data(),
                             repeated.size_bytes(),
                             "actual production full-logit replay"));
        const auto& repeated_branch =
            repeat_logits_fwd.state.children[block + 2].children[0];
        const cuda::Buffer* repeated_buffers[] = {
            &repeated_branch.intermediates[0],
            &repeated_branch.children[0].children[2].intermediates[0],
            &repeated_branch.children[0].children[2].intermediates[1]};
        const cuda::Buffer* original_buffers[] = {&before, &qkv, &context};
        for (int i = 0; i < 3; ++i) {
          RETURN_IF_ERROR(Shape(executor, *repeated_buffers[i],
                                original_buffers[i]->size_bytes()));
      ASSIGN_OR_RETURN(auto a,
                       Download<uint16_t>(executor, *repeated_buffers[i]));
      ASSIGN_OR_RETURN(auto b,
                       Download<uint16_t>(executor, *original_buffers[i]));
          RETURN_IF_ERROR(Same(a.data(), b.data(), a.size_bytes(),
                               "actual production replay activation"));
    }
  }
  ASSIGN_OR_RETURN(impl->projection,
                   llm::FullyConnectedLayer::Create(executor, kWidth, kWidth,
                                                    llm::DataType::BF16));
  RETURN_IF_ERROR(CopyWeights(executor, weights.subspan(6 + 12 * block, 2),
                              impl->projection->weights()));
  llm::ComposedLayerBuilder tail;
  ASSIGN_OR_RETURN(auto first_mlp,
                   Branch(executor, weights.subspan(8 + 12 * block, 6), false));
  RETURN_IF_ERROR(
      tail.add(std::make_unique<llm::ResidualLayer>(std::move(first_mlp))));
  for (int later = block + 1; later < 8; ++later) {
    for (bool attention : {true, false}) {
      ASSIGN_OR_RETURN(
          auto next,
          Branch(executor, weights.subspan((attention ? 2 : 8) + 12 * later, 6),
                 attention));
      RETURN_IF_ERROR(
          tail.add(std::make_unique<llm::ResidualLayer>(std::move(next))));
    }
  }
  ASSIGN_OR_RETURN(impl->tail, tail.create());
  ASSIGN_OR_RETURN(impl->embedding,
                   llm::EmbeddingLookupLayer::Create(
                       executor, kVocabulary, kWidth, llm::DataType::BF16));
  RETURN_IF_ERROR(
      CopyWeights(executor, weights.subspan(0, 1), impl->embedding->weights()));
  ASSIGN_OR_RETURN(
      impl->norm,
      llm::LayerNormLayer::Create(executor, kWidth, 1e-5f, llm::DataType::BF16));
  RETURN_IF_ERROR(
      CopyWeights(executor, weights.subspan(98, 2), impl->norm->weights()));
  ASSIGN_OR_RETURN(
      impl->head, llm::LanguageModelingHeadLayer::Create(impl->embedding.get()));
  ASSIGN_OR_RETURN(auto attention, llm::AttentionLayer::Create(
                                       executor, kContext, kGeometry.heads,
                                       kWidth, llm::DataType::BF16));

  ASSIGN_OR_RETURN(auto native_context_fwd,
                   attention->fwd(executor, absl::MakeConstSpan(&qkv, 1)));
      auto native_context = std::move(native_context_fwd.outputs[0]);

      RETURN_IF_ERROR(Shape(executor, native_context, context.size_bytes()));
  ASSIGN_OR_RETURN(auto native_bytes,
                   Download<uint16_t>(executor, native_context));
      RETURN_IF_ERROR(Same(native_bytes.data(), impl->context_bytes.data(),
                           native_bytes.size_bytes(),
                           "clean native attention replay"));
  ASSIGN_OR_RETURN(auto native_logits, impl->Replay(native_context));
      RETURN_IF_ERROR(Shape(executor, native_logits, logits.size_bytes()));
  ASSIGN_OR_RETURN(auto native_values, Download<float>(executor, native_logits));
      RETURN_IF_ERROR(Same(native_values.data(), impl->logit_values.data(),
                           native_values.size_bytes(),
                           "clean all-row padded-logit tail replay"));
      RETURN_IF_ERROR(impl->Verify());
      return absl::WrapUnique(new Probe(std::move(impl)));
}

absl::StatusOr<Result> Probe::Apply(cuda::Executor& executor,
                                    const Selection& selection) const {
  RETURN_IF_ERROR(ValidateSelection(kGeometry, selection, impl_->rows));
  if (&executor != &impl_->executor || selection.block != impl_->block)
    return absl::InvalidArgumentError("head-context block/executor differs");
  RETURN_IF_ERROR(impl_->Verify());
  ASSIGN_OR_RETURN(auto expected, cuda::PageLockedHostArray<uint16_t>::Allocate(
                                      executor, impl_->context_bytes.size()));
  RETURN_IF_ERROR(ScaleContext(impl_->context_bytes.span(), expected.span(),
                               impl_->rows, kGeometry, selection));
  ASSIGN_OR_RETURN(auto context,
                   cuda::Buffer::Allocate(executor, expected.size_bytes()));
  RETURN_IF_ERROR(cuda::CudaStatus(
      cudaMemcpyAsync(context.data(), expected.data(), expected.size_bytes(),
                      cudaMemcpyHostToDevice, executor.stream()),
      "upload exact head-context intervention"));
  ASSIGN_OR_RETURN(auto actual, Download<uint16_t>(executor, context));
  RETURN_IF_ERROR(Same(actual.data(), expected.data(), actual.size_bytes(),
                       "declared entire context intervention"));
  // Identity still executes all production tail kernels, never returns saved
  // logits.
  ASSIGN_OR_RETURN(auto logits, impl_->Replay(context));
  RETURN_IF_ERROR(Shape(executor, logits, impl_->logits.size_bytes()));
  ASSIGN_OR_RETURN(auto values, Download<float>(executor, logits));
  RETURN_IF_ERROR(FiniteFloats(values.span()));
  if (selection.scale == 1.0f) {
    RETURN_IF_ERROR(Same(values.data(), impl_->logit_values.data(),
                         values.size_bytes(),
                         "identity all-row padded logits"));
  }
  for (int row = 0; row < impl_->rows; ++row) {
    const int sequence = row / kContext;
    const int query = row % kContext;
    const bool causal_null = sequence != selection.sequence ||
                             (selection.scope == QueryScope::kSelectedQuery &&
                              query < selection.query_position) ||
                             (selection.scope == QueryScope::kOtherQueries &&
                              selection.query_position == 0 && query == 0);
    const size_t offset = static_cast<size_t>(row) * kPadded;
    if (causal_null) {
      RETURN_IF_ERROR(
          Same(values.data() + offset, impl_->logit_values.data() + offset,
               kPadded * sizeof(float), "causally unaffected full logit row"));
    }
    RETURN_IF_ERROR(Same(values.data() + offset + kVocabulary,
                         impl_->logit_values.data() + offset + kVocabulary,
                         (kPadded - kVocabulary) * sizeof(float),
                         "physical padding logits"));
  }
  RETURN_IF_ERROR(impl_->Verify());
  return Result{std::move(context), std::move(logits)};
}

absl::Status Probe::VerifyOriginals(cuda::Executor& executor) const {
  if (&executor != &impl_->executor)
    return absl::InvalidArgumentError("head-context executor differs");
  return impl_->Verify();
}
const cuda::Buffer& Probe::original_context() const { return impl_->context; }
const cuda::Buffer& Probe::clean_logits() const { return impl_->logits; }

}  // namespace pluto::weight_analysis::head_context
