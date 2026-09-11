#include <cuda_runtime_api.h>

#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <set>
#include <utility>

#include "absl/memory/memory.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "ai-slop/weight_analysis/causal_probe.h"
#include "ai-slop/weight_analysis/phrase_probe.h"
#include "src/llm/layers/fully_connected.h"
#include "src/llm/recipes/gpt2.h"
#include "src/util/status_macros.h"

namespace pluto::weight_analysis {
namespace {
constexpr int kWidth = llm::kGpt2ModelWidth;

absl::Status BadState() {
  return absl::FailedPreconditionError(
      "production GPT-2 state topology changed; refusing guessed layer paths");
}

float Bf16Float(uint16_t bits) {
  return std::bit_cast<float>(static_cast<uint32_t>(bits) << 16);
}
}  // namespace

absl::Status ValidateGpt2State(const llm::BackwardState& state) {
  if (!state.intermediates.empty() || state.children.size() != 12 ||
      state.children[0].intermediates.size() != 1 ||
      !state.children[1].intermediates.empty() ||
      state.children[10].intermediates.size() != 1 ||
      state.children[11].intermediates.size() != 1) {
    return BadState();
  }
  for (int block = 0; block < 8; ++block) {
    const auto& body = state.children[block + 2];
    if (!body.intermediates.empty() || body.children.size() != 2)
      return BadState();
    for (int branch = 0; branch < 2; ++branch) {
      const auto& residual = body.children[branch];
      if (residual.intermediates.size() != 1 || residual.children.size() != 1)
        return BadState();
      const auto& sequence = residual.children[0];
      if (!sequence.intermediates.empty() || sequence.children.size() != 4)
        return BadState();
      for (int leaf = 0; leaf < 4; ++leaf) {
        const auto& saved = sequence.children[leaf];
        const size_t expected = branch == 0 && leaf == 2 ? 2 : 1;
        if (!saved.children.empty() || saved.intermediates.size() != expected)
          return BadState();
      }
    }
  }
  return absl::OkStatus();
}

absl::Status CopyDeviceWeights(cuda::Executor& executor,
                               absl::Span<const cuda::Buffer> source,
                               absl::Span<cuda::Buffer> destination) {
  if (source.size() != destination.size())
    return absl::InvalidArgumentError("weight counts differ in native replay");
  // Validate every tensor before queuing any writes, including executor and
  // size checks. Only new diagnostic-layer allocations may be destinations.
  for (size_t i = 0; i < source.size(); ++i) {
    if (&source[i].executor() != &executor ||
        &destination[i].executor() != &executor ||
        source[i].size_bytes() != destination[i].size_bytes() ||
        source[i].data() == destination[i].data()) {
      return absl::InvalidArgumentError("incompatible native replay weight");
    }
  }
  for (size_t i = 0; i < source.size(); ++i) {
    RETURN_IF_ERROR(cuda::CudaStatus(
        cudaMemcpyAsync(destination[i].data(), source[i].data(),
                        source[i].size_bytes(), cudaMemcpyDeviceToDevice,
                        executor.stream()),
        "copy diagnostic layer weight"));
  }
  return absl::OkStatus();
}

absl::StatusOr<cuda::Buffer> ReplayProjection(cuda::Executor& executor,
                                              const cuda::Buffer& input,
                                              const cuda::Buffer& matrix,
                                              const cuda::Buffer& bias,
                                              int input_width,
                                              int output_width) {
  ASSIGN_OR_RETURN(
      auto layer, llm::FullyConnectedLayer::Create(
                      executor, input_width, output_width, llm::DataType::BF16));
  const llm::BufferVec original{matrix, bias};
  RETURN_IF_ERROR(CopyDeviceWeights(executor, original, layer->weights()));

  ASSIGN_OR_RETURN(auto fwd_return_result,
                   layer->fwd(executor, absl::MakeConstSpan(&input, 1)));
  return std::move(fwd_return_result.output);
}

absl::StatusOr<std::vector<TraceFrame>> CollectGpt2Trace(
    cuda::Executor& executor, const llm::BackwardState& state,
    const cuda::Buffer& logits, absl::Span<const cuda::Buffer> weights) {
  RETURN_IF_ERROR(ValidateGpt2State(state));
  RETURN_IF_ERROR(ValidateGpt2Weights(weights));
  std::vector<TraceFrame> frames;
  const auto& positioned = state.children[2].children[0].intermediates[0];
  frames.push_back({"positioned", positioned, kWidth});
  for (int block = 0; block < 8; ++block) {
    const auto& branches = state.children[block + 2].children;
    const auto& attention = branches[0].children[0].children;
    const auto& mlp = branches[1].children[0].children;
    const std::string prefix = absl::StrCat("blocks.", block, ".");
    frames.push_back({prefix + "ln1", attention[1].intermediates[0], kWidth});
    frames.push_back(
        {prefix + "qkv", attention[2].intermediates[0], 3 * kWidth});
    frames.push_back(
        {prefix + "attention", attention[2].intermediates[1], kWidth});
    const int attention_weight = 6 + 12 * block;
    ASSIGN_OR_RETURN(
        auto projected,
        ReplayProjection(executor, attention[3].intermediates[0],
                         weights[attention_weight],
                         weights[attention_weight + 1], kWidth, kWidth));
    frames.push_back({prefix + "attention_projected", std::move(projected),
                      kWidth, false, true});
    frames.push_back(
        {prefix + "after_attention", branches[1].intermediates[0], kWidth});
    frames.push_back({prefix + "ln2", mlp[1].intermediates[0], kWidth});
    frames.push_back(
        {prefix + "fc1", mlp[2].intermediates[0], llm::kGpt2FeedForwardWidth});
    frames.push_back(
        {prefix + "gelu", mlp[3].intermediates[0], llm::kGpt2FeedForwardWidth});
    const int mlp_weight = 12 + 12 * block;
    ASSIGN_OR_RETURN(
        auto mlp_projected,
        ReplayProjection(executor, mlp[3].intermediates[0], weights[mlp_weight],
                         weights[mlp_weight + 1], llm::kGpt2FeedForwardWidth,
                         kWidth));
    frames.push_back({prefix + "mlp_projected", std::move(mlp_projected),
                      kWidth, false, true});
    const auto& after =
        block == 7 ? state.children[10].intermediates[0]
                   : state.children[block + 3].children[0].intermediates[0];
    frames.push_back({prefix + "after_mlp", after, kWidth});
  }
  frames.push_back({"final_norm", state.children[11].intermediates[0], kWidth});
  frames.push_back({"logits", logits, llm::kGpt2PaddedVocabularySize, true});
  return frames;
}

absl::StatusOr<cuda::PageLockedHostArray<uint8_t>> ReadPrefix(
    cuda::Executor& executor, const cuda::Buffer& buffer, int rows, int width,
    size_t element_bytes) {
  if (rows <= 0 || width <= 0 || (element_bytes != 2 && element_bytes != 4) ||
      &buffer.executor() != &executor ||
      static_cast<size_t>(width) >
          std::numeric_limits<size_t>::max() / element_bytes) {
    return absl::InvalidArgumentError("invalid prefix tensor shape/executor");
  }
  const size_t stride = static_cast<size_t>(width) * element_bytes;
  if (buffer.size_bytes() % stride ||
      static_cast<size_t>(rows) > buffer.size_bytes() / stride) {
    return absl::InvalidArgumentError(
        "prefix tensor exceeds device allocation");
  }
  ASSIGN_OR_RETURN(auto output, cuda::PageLockedHostArray<uint8_t>::Allocate(
                                    executor, rows * stride));
  RETURN_IF_ERROR(cuda::CudaStatus(
      cudaMemcpyAsync(output.data(), buffer.data(), output.size_bytes(),
                      cudaMemcpyDeviceToHost, executor.stream()),
      "read native activation prefix"));
  RETURN_IF_ERROR(executor.Synchronize());
  return output;
}

uint16_t AddBf16Bits(uint16_t left, uint16_t right) {
  const float sum = Bf16Float(left) + Bf16Float(right);
  const uint32_t bits = std::bit_cast<uint32_t>(sum);
  return static_cast<uint16_t>((bits + 0x7fff + ((bits >> 16) & 1)) >> 16);
}

absl::Status VerifyResidualReplay(cuda::Executor& executor,
                                  const cuda::Buffer& before,
                                  const cuda::Buffer& branch,
                                  const cuda::Buffer& after, int rows,
                                  int width) {
  ASSIGN_OR_RETURN(auto a, ReadPrefix(executor, before, rows, width, 2));
  ASSIGN_OR_RETURN(auto b, ReadPrefix(executor, branch, rows, width, 2));
  ASSIGN_OR_RETURN(auto c, ReadPrefix(executor, after, rows, width, 2));
  for (size_t i = 0; i < a.size_bytes(); i += 2) {
    uint16_t left, right, actual;
    std::memcpy(&left, a.data() + i, 2);
    std::memcpy(&right, b.data() + i, 2);
    std::memcpy(&actual, c.data() + i, 2);
    if (!std::isfinite(Bf16Float(left)) || !std::isfinite(Bf16Float(right)) ||
        !std::isfinite(Bf16Float(actual)) ||
        AddBf16Bits(left, right) != actual) {
      return absl::DataLossError(absl::StrCat(
          "native branch replay disagrees with residual at element ", i / 2));
    }
  }
  return absl::OkStatus();
}

NativeLogitLens::NativeLogitLens(
    std::unique_ptr<llm::EmbeddingLookupLayer> embedding,
    std::unique_ptr<llm::LayerNormLayer> norm,
    std::unique_ptr<llm::LanguageModelingHeadLayer> head)
    : embedding_(std::move(embedding)),
      norm_(std::move(norm)),
      head_(std::move(head)) {}

absl::StatusOr<std::unique_ptr<NativeLogitLens>> NativeLogitLens::Create(
    cuda::Executor& executor, absl::Span<const cuda::Buffer> weights) {
  RETURN_IF_ERROR(ValidateGpt2Weights(weights));
  ASSIGN_OR_RETURN(auto embedding, llm::EmbeddingLookupLayer::Create(
                                       executor, llm::kGpt2VocabularySize,
                                       kWidth, llm::DataType::BF16));
  RETURN_IF_ERROR(
      CopyDeviceWeights(executor, weights.subspan(0, 1), embedding->weights()));
  ASSIGN_OR_RETURN(auto norm, llm::LayerNormLayer::Create(
                                  executor, kWidth, 1e-5f, llm::DataType::BF16));
  RETURN_IF_ERROR(
      CopyDeviceWeights(executor, weights.subspan(98, 2), norm->weights()));
  ASSIGN_OR_RETURN(auto head,
                   llm::LanguageModelingHeadLayer::Create(embedding.get()));
  return absl::WrapUnique(new NativeLogitLens(
      std::move(embedding), std::move(norm), std::move(head)));
}

absl::StatusOr<cuda::Buffer> NativeLogitLens::Embed(
    cuda::Executor& executor, const cuda::Buffer& tokens) const {
  ASSIGN_OR_RETURN(auto fwd_return_result,
                   embedding_->fwd(executor, absl::MakeConstSpan(&tokens, 1)));
  return std::move(fwd_return_result.output);
}

absl::StatusOr<cuda::Buffer> NativeLogitLens::Apply(
    cuda::Executor& executor, const cuda::Buffer& residual) const {
  llm::BackwardState norm_state, head_state;
  ASSIGN_OR_RETURN(auto normalized_fwd,
                   norm_->fwd(executor, absl::MakeConstSpan(&residual, 1)));
  auto normalized = std::move(normalized_fwd.output);
  norm_state = std::move(normalized_fwd.state);
  ASSIGN_OR_RETURN(auto fwd_return_result,
                   head_->fwd(executor, absl::MakeConstSpan(&normalized, 1)));
  return std::move(fwd_return_result.output);
}

absl::StatusOr<std::vector<std::pair<int, int>>> ParseNeuronInterventions(
    absl::string_view text) {
  std::vector<std::pair<int, int>> result;
  if (text.empty())
    return result;
  std::set<std::pair<int, int>> seen;
  for (absl::string_view entry : absl::StrSplit(text, ',')) {
    std::vector<absl::string_view> fields = absl::StrSplit(entry, ':');
    int block, neuron;
    if (fields.size() != 2 || !absl::SimpleAtoi(fields[0], &block) ||
        !absl::SimpleAtoi(fields[1], &neuron) || block < 0 || block >= 8 ||
        neuron < 0 || neuron >= llm::kGpt2FeedForwardWidth ||
        !seen.insert({block, neuron}).second) {
      return absl::InvalidArgumentError(
          "--ablate_neuron expects unique block:neuron pairs within "
          "0:0..7:2047");
    }
    result.emplace_back(block, neuron);
  }
  return result;
}

}  // namespace pluto::weight_analysis
