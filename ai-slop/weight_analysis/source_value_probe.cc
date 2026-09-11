#include "ai-slop/weight_analysis/source_value_probe.h"

#include <cuda_runtime_api.h>

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <memory>
#include <utility>

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "ai-slop/weight_analysis/causal_probe.h"
#include "ai-slop/weight_analysis/phrase_probe.h"
#include "src/llm/layers/attention.h"
#include "src/llm/layers/combinators.h"
#include "src/llm/layers/fully_connected.h"
#include "src/llm/layers/gelu.h"
#include "src/llm/layers/norm.h"
#include "src/llm/recipes/gpt2.h"
#include "src/util/status_macros.h"

namespace pluto::weight_analysis {
namespace {
constexpr int kWidth = llm::kGpt2ModelWidth;
constexpr int kContext = llm::kGpt2ContextLength;
constexpr int kHeadWidth = llm::kGpt2AttentionHeadDimension;
constexpr size_t kHeadBytes = kHeadWidth * sizeof(uint16_t);
using Bytes = cuda::PageLockedHostArray<uint8_t>;
// The copied checkpoint slices below are physical recipe indices, not a
// generic transformer convention. Fail compilation if that recipe changes.
static_assert(llm::kGpt2TransformerBlockCount == 8 && kWidth == 512 &&
              kContext == 1024 && llm::kGpt2AttentionHeads == 8 &&
              kHeadWidth == 64 && llm::kGpt2FeedForwardWidth == 2048 &&
              llm::kGpt2VocabularySize == 50257 &&
              llm::kGpt2PaddedVocabularySize == 50272);

absl::Status ValidateScale(float scale) {
  if (scale != 0.0f && scale != 0.5f && scale != 1.0f)
    return absl::InvalidArgumentError("value scale must be 0, 0.5, or 1");
  return absl::OkStatus();
}

absl::Status EqualBytes(const Bytes& actual, const Bytes& expected,
                        absl::string_view description) {
  if (actual.size_bytes() != expected.size_bytes() ||
      std::memcmp(actual.data(), expected.data(), expected.size_bytes()) != 0) {
    return absl::DataLossError(
        absl::StrCat(description, " differs from original bytes"));
  }
  return absl::OkStatus();
}

absl::Status EqualOutsideSlice(const Bytes& actual, const Bytes& expected,
                               size_t offset, size_t length,
                               absl::string_view description) {
  if (actual.size_bytes() != expected.size_bytes() ||
      offset > actual.size_bytes() || length > actual.size_bytes() - offset) {
    return absl::InvalidArgumentError("invalid value-intervention byte slice");
  }
  if (std::memcmp(actual.data(), expected.data(), offset) != 0 ||
      std::memcmp(actual.data() + offset + length,
                  expected.data() + offset + length,
                  actual.size_bytes() - offset - length) != 0) {
    return absl::DataLossError(
        absl::StrCat(description, " changed outside the intended slice"));
  }
  return absl::OkStatus();
}

absl::Status FiniteBytes(const Bytes& bytes, bool fp32) {
  for (size_t i = 0; i < bytes.size_bytes(); i += fp32 ? 4 : 2) {
    float value;
    if (fp32) {
      std::memcpy(&value, bytes.data() + i, sizeof(value));
    } else {
      uint16_t bits;
      std::memcpy(&bits, bytes.data() + i, sizeof(bits));
      value = std::bit_cast<float>(static_cast<uint32_t>(bits) << 16);
    }
    if (!std::isfinite(value))
      return absl::DataLossError("nonfinite source-value replay activation");
  }
  return absl::OkStatus();
}

absl::Status ValidateBuffer(cuda::Executor& executor,
                            const cuda::Buffer& buffer, size_t bytes) {
  if (&buffer.executor() != &executor || buffer.size_bytes() != bytes) {
    return absl::InvalidArgumentError(
        "source-value replay buffer has wrong shape or executor");
  }
  return absl::OkStatus();
}

absl::StatusOr<cuda::Buffer> Clone(cuda::Executor& executor,
                                   const cuda::Buffer& original) {
  ASSIGN_OR_RETURN(auto copy,
                   cuda::Buffer::Allocate(executor, original.size_bytes()));
  RETURN_IF_ERROR(cuda::CudaStatus(
      cudaMemcpyAsync(copy.data(), original.data(), original.size_bytes(),
                      cudaMemcpyDeviceToDevice, executor.stream()),
      "copy original attention activation"));
  return copy;
}

// Feed an already computed branch through the real ResidualLayer. This tiny
// adapter avoids approximating BF16 addition on the host, or duplicating the
// production add kernel. It never changes its saved branch or the input.
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
    if (inputs.size() != 1 || &branch_.executor() != &executor ||
        &inputs[0].executor() != &executor ||
        inputs[0].size_bytes() != branch_.size_bytes()) {
      return absl::InvalidArgumentError("invalid constant residual branch");
    }
    state.intermediates.clear();
    state.children.clear();
    return llm::FwdResult{std::move(branch_), std::move(state)};
  }
  absl::StatusOr<llm::BufferVec> bwd_impl(cuda::Executor&,
                                          absl::Span<const cuda::Buffer>,
                                          llm::BackwardState) override {
    return absl::UnimplementedError("source-value replay is forward only");
  }

  cuda::Buffer branch_;
};

absl::StatusOr<std::unique_ptr<llm::ComposedLayer>> NewMlp(
    cuda::Executor& executor, absl::Span<const cuda::Buffer> weights) {
  llm::ComposedLayerBuilder builder;
  RETURN_IF_ERROR(builder.add(llm::LayerNormLayer::Create(
      executor, kWidth, 1e-5f, llm::DataType::BF16)));
  RETURN_IF_ERROR(builder.add(llm::FullyConnectedLayer::Create(
      executor, kWidth, llm::kGpt2FeedForwardWidth, llm::DataType::BF16)));
  RETURN_IF_ERROR(
      builder.add(llm::GeluLayer::Create(executor, llm::DataType::BF16)));
  RETURN_IF_ERROR(builder.add(llm::FullyConnectedLayer::Create(
      executor, llm::kGpt2FeedForwardWidth, kWidth, llm::DataType::BF16)));
  ASSIGN_OR_RETURN(auto mlp, builder.create());
  RETURN_IF_ERROR(CopyDeviceWeights(executor, weights, mlp->weights()));
  return mlp;
}

absl::StatusOr<std::unique_ptr<llm::ComposedLayer>> NewAttention(
    cuda::Executor& executor, absl::Span<const cuda::Buffer> weights) {
  llm::ComposedLayerBuilder builder;
  RETURN_IF_ERROR(builder.add(llm::LayerNormLayer::Create(
      executor, kWidth, 1e-5f, llm::DataType::BF16)));
  RETURN_IF_ERROR(builder.add(llm::FullyConnectedLayer::Create(
      executor, kWidth, 3 * kWidth, llm::DataType::BF16)));
  RETURN_IF_ERROR(builder.add(
      llm::AttentionLayer::Create(executor, kContext, llm::kGpt2AttentionHeads,
                                  kWidth, llm::DataType::BF16)));
  RETURN_IF_ERROR(builder.add(llm::FullyConnectedLayer::Create(
      executor, kWidth, kWidth, llm::DataType::BF16)));
  ASSIGN_OR_RETURN(auto attention, builder.create());
  RETURN_IF_ERROR(CopyDeviceWeights(executor, weights, attention->weights()));
  return attention;
}
}  // namespace

absl::Status ValidateSourceValueSelection(const SourceValueSelection& selection,
                                          int total_rows) {
  RETURN_IF_ERROR(ValidateScale(selection.scale));
  if (total_rows <= 0 || total_rows % kContext != 0 ||
      total_rows >
          std::numeric_limits<int>::max() / llm::kGpt2FeedForwardWidth ||
      selection.block < 0 ||
      selection.block >= llm::kGpt2TransformerBlockCount ||
      selection.head < 0 || selection.head >= llm::kGpt2AttentionHeads ||
      selection.sequence < 0 || selection.sequence >= total_rows / kContext ||
      selection.query_position < 0 || selection.query_position >= kContext ||
      selection.source_position < 0 || selection.source_position >= kContext) {
    return absl::InvalidArgumentError("invalid source-value attention site");
  }
  return absl::OkStatus();
}

absl::StatusOr<uint16_t> ScaleSourceBf16(uint16_t value, float scale) {
  RETURN_IF_ERROR(ValidateScale(scale));
  if ((value & 0x7f80) == 0x7f80)
    return absl::InvalidArgumentError("cannot scale a nonfinite BF16 value");
  if (scale == 0.0f)
    return uint16_t{0};
  if (scale == 1.0f)
    return value;
  const uint16_t sign = value & 0x8000;
  const uint16_t magnitude = value & 0x7fff;
  if (magnitude >= 0x0100)
    return static_cast<uint16_t>(value - 0x0080);
  // In the subnormal/minimum-exponent range, halving shifts the significand.
  // Add one only for an odd tie; this is round-to-nearest/even. Integer work
  // avoids depending on the CPU's flush-to-zero mode for FP32 subnormals.
  const uint16_t rounded = (magnitude >> 1) + ((magnitude & 3) == 3 ? 1 : 0);
  return static_cast<uint16_t>(sign | rounded);
}

struct SourceValueProbe::Impl {
  Impl(cuda::Executor& executor, int block, int rows,
       const cuda::Buffer& before, const cuda::Buffer& qkv,
       const cuda::Buffer& context, const cuda::Buffer& logits)
      : executor(executor),
        block(block),
        rows(rows),
        before(before),
        qkv(qkv),
        context(context),
        logits(logits) {}

  absl::Status CheckOriginals() const {
    ASSIGN_OR_RETURN(auto a, ReadPrefix(executor, before, rows, kWidth, 2));
    RETURN_IF_ERROR(
        EqualBytes(a, before_bytes, "clean pre-attention residual"));
    ASSIGN_OR_RETURN(auto b, ReadPrefix(executor, qkv, rows, 3 * kWidth, 2));
    RETURN_IF_ERROR(EqualBytes(b, qkv_bytes, "clean QKV"));
    ASSIGN_OR_RETURN(auto c, ReadPrefix(executor, context, rows, kWidth, 2));
    RETURN_IF_ERROR(EqualBytes(c, context_bytes, "clean attention context"));
    ASSIGN_OR_RETURN(auto d, ReadPrefix(executor, logits, rows,
                                        llm::kGpt2PaddedVocabularySize, 4));
    return EqualBytes(d, logits_bytes, "clean full logits");
  }

  absl::StatusOr<cuda::Buffer> Replay(
      const cuda::Buffer& changed_context) const {
    ASSIGN_OR_RETURN(
        auto projected_fwd,
        projection->fwd(executor, absl::MakeConstSpan(&changed_context, 1)));
    auto projected = std::move(projected_fwd.output);

    llm::ResidualLayer residual(
        std::make_unique<ConstantBranch>(std::move(projected)));
    llm::BackwardState add_state, tail_state;
    ASSIGN_OR_RETURN(auto after_attention_fwd,
                     residual.fwd(executor, absl::MakeConstSpan(&before, 1)));
    auto after_attention = std::move(after_attention_fwd.output);
    add_state = std::move(after_attention_fwd.state);
    ASSIGN_OR_RETURN(
        auto final_residual_fwd,
        tail->fwd(executor, absl::MakeConstSpan(&after_attention, 1)));
    auto final_residual = std::move(final_residual_fwd.output);
    tail_state = std::move(final_residual_fwd.state);
    return lens->Apply(executor, final_residual);
  }

  cuda::Executor& executor;
  int block;
  int rows;
  cuda::Buffer before, qkv, context, logits;
  Bytes before_bytes, qkv_bytes, context_bytes, logits_bytes;
  std::unique_ptr<llm::AttentionLayer> attention;
  std::unique_ptr<llm::FullyConnectedLayer> projection;
  std::unique_ptr<llm::ComposedLayer> tail;
  std::unique_ptr<NativeLogitLens> lens;
};

SourceValueProbe::SourceValueProbe(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
SourceValueProbe::~SourceValueProbe() = default;

absl::StatusOr<std::unique_ptr<SourceValueProbe>> SourceValueProbe::Create(
    cuda::Executor& executor, const llm::BackwardState& production_state,
    const cuda::Buffer& clean_logits,
    absl::Span<const cuda::Buffer> original_weights, int block) {
  RETURN_IF_ERROR(
      ValidateSourceValueSelection({block, 0, 0, 0, 0, 1}, kContext));
  RETURN_IF_ERROR(ValidateGpt2State(production_state));
  RETURN_IF_ERROR(ValidateGpt2Weights(original_weights));
  for (const auto& weight : original_weights)
    if (&weight.executor() != &executor)
      return absl::InvalidArgumentError("source weight has wrong executor");
  const auto& branch = production_state.children[block + 2].children[0];
  const auto& before = branch.intermediates[0];
  const auto& attention_state = branch.children[0].children[2];
  const auto& qkv = attention_state.intermediates[0];
  const auto& context = attention_state.intermediates[1];
  const size_t row_bytes = kWidth * sizeof(uint16_t);
  const size_t rows_wide = context.size_bytes() / row_bytes;
  if (context.size_bytes() % row_bytes != 0 || rows_wide == 0 ||
      rows_wide > std::numeric_limits<int>::max() ||
      rows_wide % kContext != 0) {
    return absl::InvalidArgumentError("invalid native GPT-2 attention shape");
  }
  const int rows = static_cast<int>(rows_wide);
  RETURN_IF_ERROR(ValidateSourceValueSelection({block, 0, 0, 0, 0, 1}, rows));
  RETURN_IF_ERROR(ValidateBuffer(executor, before, rows_wide * row_bytes));
  RETURN_IF_ERROR(ValidateBuffer(executor, qkv, rows_wide * row_bytes * 3));
  RETURN_IF_ERROR(ValidateBuffer(executor, context, rows_wide * row_bytes));
  RETURN_IF_ERROR(ValidateBuffer(
      executor, clean_logits,
      rows_wide * llm::kGpt2PaddedVocabularySize * sizeof(float)));

  auto impl = std::make_unique<Impl>(executor, block, rows, before, qkv,
                                     context, clean_logits);
  ASSIGN_OR_RETURN(impl->before_bytes,
                   ReadPrefix(executor, before, rows, kWidth, 2));
  ASSIGN_OR_RETURN(impl->qkv_bytes,
                   ReadPrefix(executor, qkv, rows, 3 * kWidth, 2));
  ASSIGN_OR_RETURN(impl->context_bytes,
                   ReadPrefix(executor, context, rows, kWidth, 2));
  ASSIGN_OR_RETURN(impl->logits_bytes,
                   ReadPrefix(executor, clean_logits, rows,
                              llm::kGpt2PaddedVocabularySize, 4));
  RETURN_IF_ERROR(FiniteBytes(impl->before_bytes, false));
  RETURN_IF_ERROR(FiniteBytes(impl->qkv_bytes, false));
  RETURN_IF_ERROR(FiniteBytes(impl->context_bytes, false));
  RETURN_IF_ERROR(FiniteBytes(impl->logits_bytes, true));

  ASSIGN_OR_RETURN(
      impl->attention,
      llm::AttentionLayer::Create(executor, kContext, llm::kGpt2AttentionHeads,
                                  kWidth, llm::DataType::BF16));
  ASSIGN_OR_RETURN(impl->projection,
                   llm::FullyConnectedLayer::Create(executor, kWidth, kWidth,
                                                    llm::DataType::BF16));
  RETURN_IF_ERROR(CopyDeviceWeights(executor,
                                    original_weights.subspan(6 + 12 * block, 2),
                                    impl->projection->weights()));
  llm::ComposedLayerBuilder tail;
  ASSIGN_OR_RETURN(auto first_mlp, NewMlp(executor, original_weights.subspan(
                                                        8 + 12 * block, 6)));
  RETURN_IF_ERROR(
      tail.add(std::make_unique<llm::ResidualLayer>(std::move(first_mlp))));
  for (int later = block + 1; later < llm::kGpt2TransformerBlockCount;
       ++later) {
    ASSIGN_OR_RETURN(
        auto attention,
        NewAttention(executor, original_weights.subspan(2 + 12 * later, 6)));
    RETURN_IF_ERROR(
        tail.add(std::make_unique<llm::ResidualLayer>(std::move(attention))));
    ASSIGN_OR_RETURN(
        auto mlp, NewMlp(executor, original_weights.subspan(8 + 12 * later, 6)));
    RETURN_IF_ERROR(
        tail.add(std::make_unique<llm::ResidualLayer>(std::move(mlp))));
  }
  ASSIGN_OR_RETURN(impl->tail, tail.create());
  ASSIGN_OR_RETURN(impl->lens,
                   NativeLogitLens::Create(executor, original_weights));

  // Do not certify a replay from a diagram or a few selected logits. Running
  // the real kernels on the captured tensor must recover the entire native
  // context, then every padded logit at every original batch position.

  ASSIGN_OR_RETURN(
      auto replayed_context_fwd,
      impl->attention->fwd(executor, absl::MakeConstSpan(&qkv, 1)));
  auto replayed_context = std::move(replayed_context_fwd.output);

  ASSIGN_OR_RETURN(auto replayed_bytes,
                   ReadPrefix(executor, replayed_context, rows, kWidth, 2));
  RETURN_IF_ERROR(EqualBytes(replayed_bytes, impl->context_bytes,
                             "native attention identity replay"));
  ASSIGN_OR_RETURN(auto replayed_logits, impl->Replay(replayed_context));
  ASSIGN_OR_RETURN(auto logit_bytes,
                   ReadPrefix(executor, replayed_logits, rows,
                              llm::kGpt2PaddedVocabularySize, 4));
  RETURN_IF_ERROR(EqualBytes(logit_bytes, impl->logits_bytes,
                             "native full-logit identity replay"));
  RETURN_IF_ERROR(impl->CheckOriginals());
  return absl::WrapUnique(new SourceValueProbe(std::move(impl)));
}

absl::StatusOr<SourceValueResult> SourceValueProbe::Apply(
    cuda::Executor& executor, const SourceValueSelection& selection) const {
  RETURN_IF_ERROR(ValidateSourceValueSelection(selection, impl_->rows));
  if (&executor != &impl_->executor || selection.block != impl_->block) {
    return absl::InvalidArgumentError(
        "source-value probe block/executor differs");
  }
  RETURN_IF_ERROR(impl_->CheckOriginals());
  const size_t source_row = static_cast<size_t>(selection.sequence) * kContext +
                            selection.source_position;
  const size_t query_row = static_cast<size_t>(selection.sequence) * kContext +
                           selection.query_position;
  const size_t value_offset =
      2 * (source_row * 3 * kWidth + 2 * kWidth + selection.head * kHeadWidth);
  const size_t context_offset =
      2 * (query_row * kWidth + selection.head * kHeadWidth);

  ASSIGN_OR_RETURN(auto qkv, Clone(executor, impl_->qkv));
  ASSIGN_OR_RETURN(auto values, cuda::PageLockedHostArray<uint16_t>::Allocate(
                                    executor, kHeadWidth));
  for (int i = 0; i < kHeadWidth; ++i) {
    uint16_t original;
    std::memcpy(&original, impl_->qkv_bytes.data() + value_offset + 2 * i, 2);
    ASSIGN_OR_RETURN(values[i], ScaleSourceBf16(original, selection.scale));
  }
  RETURN_IF_ERROR(cuda::CudaStatus(
      cudaMemcpyAsync(static_cast<uint8_t*>(qkv.data()) + value_offset,
                      values.data(), kHeadBytes, cudaMemcpyHostToDevice,
                      executor.stream()),
      "write single source value slice"));
  // ReadPrefix synchronizes before the pinned staging allocation can die.
  ASSIGN_OR_RETURN(auto qkv_bytes,
                   ReadPrefix(executor, qkv, impl_->rows, 3 * kWidth, 2));
  RETURN_IF_ERROR(EqualOutsideSlice(qkv_bytes, impl_->qkv_bytes, value_offset,
                                    kHeadBytes, "modified QKV"));
  if (std::memcmp(qkv_bytes.data() + value_offset, values.data(), kHeadBytes))
    return absl::DataLossError("modified V differs from requested BF16 dose");

  ASSIGN_OR_RETURN(
      auto attention_fwd,
      impl_->attention->fwd(executor, absl::MakeConstSpan(&qkv, 1)));
  auto attention = std::move(attention_fwd.output);

  ASSIGN_OR_RETURN(auto context, Clone(executor, impl_->context));
  RETURN_IF_ERROR(cuda::CudaStatus(
      cudaMemcpyAsync(
          static_cast<uint8_t*>(context.data()) + context_offset,
          static_cast<const uint8_t*>(attention.data()) + context_offset,
          kHeadBytes, cudaMemcpyDeviceToDevice, executor.stream()),
      "splice single query head context"));
  ASSIGN_OR_RETURN(auto attention_bytes,
                   ReadPrefix(executor, attention, impl_->rows, kWidth, 2));
  ASSIGN_OR_RETURN(auto context_bytes,
                   ReadPrefix(executor, context, impl_->rows, kWidth, 2));
  RETURN_IF_ERROR(FiniteBytes(attention_bytes, false));
  RETURN_IF_ERROR(EqualOutsideSlice(context_bytes, impl_->context_bytes,
                                    context_offset, kHeadBytes,
                                    "spliced attention context"));
  if (std::memcmp(context_bytes.data() + context_offset,
                  attention_bytes.data() + context_offset, kHeadBytes)) {
    return absl::DataLossError("context splice differs from native attention");
  }
  ASSIGN_OR_RETURN(auto logits, impl_->Replay(context));
  ASSIGN_OR_RETURN(auto logits_bytes,
                   ReadPrefix(executor, logits, impl_->rows,
                              llm::kGpt2PaddedVocabularySize, 4));
  RETURN_IF_ERROR(FiniteBytes(logits_bytes, true));
  if (selection.scale == 1.0f) {
    RETURN_IF_ERROR(EqualBytes(qkv_bytes, impl_->qkv_bytes, "dose-one QKV"));
    RETURN_IF_ERROR(EqualBytes(attention_bytes, impl_->context_bytes,
                               "dose-one full attention"));
    RETURN_IF_ERROR(EqualBytes(logits_bytes, impl_->logits_bytes,
                               "dose-one full padded logits"));
  }
  if (selection.source_position > selection.query_position) {
    // A future V cannot affect a causal query. Checking all downstream logits
    // also detects accidentally splicing the changed source's later queries.
    RETURN_IF_ERROR(EqualBytes(context_bytes, impl_->context_bytes,
                               "future-source null context"));
    RETURN_IF_ERROR(EqualBytes(logits_bytes, impl_->logits_bytes,
                               "future-source null full padded logits"));
  }
  RETURN_IF_ERROR(impl_->CheckOriginals());
  return SourceValueResult{std::move(qkv), std::move(attention),
                           std::move(context), std::move(logits)};
}

const cuda::Buffer& SourceValueProbe::original_qkv() const {
  return impl_->qkv;
}
const cuda::Buffer& SourceValueProbe::original_context() const {
  return impl_->context;
}
const cuda::Buffer& SourceValueProbe::clean_logits() const {
  return impl_->logits;
}

}  // namespace pluto::weight_analysis
