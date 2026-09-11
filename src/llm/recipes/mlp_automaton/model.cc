#include "src/llm/recipes/mlp_automaton/model.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "src/llm/layers/combinators.h"
#include "src/llm/layers/embedding.h"
#include "src/llm/layers/fully_connected.h"
#include "src/llm/layers/gelu.h"
#include "src/llm/layers/norm.h"
#include "src/util/status_macros.h"

namespace pluto::llm::mlp_automaton {
namespace {

constexpr int kTile = 16;
constexpr float kEpsilon = 1e-5f;
// Each GPT-2 block contributes twelve unique tensors after E and positions.
constexpr std::array<int, 9> kCheckpointIndices = {
    0,
    8,
    9,
    10,
    11,
    12,
    13,
    2 + 12 * kGpt2TransformerBlockCount,
    3 + 12 * kGpt2TransformerBlockCount};

absl::StatusOr<cuda::PageLockedHostArray<float>> ReadWeight(
    const std::filesystem::path& path, size_t expected_bytes) {
  std::error_code error;
  if (!std::filesystem::is_regular_file(path, error)) {
    return absl::NotFoundError(
        absl::StrCat("missing or unreadable weight: ", path.string()));
  }
  const uintmax_t size = std::filesystem::file_size(path, error);
  if (error || size != expected_bytes || size % sizeof(float) != 0 ||
      size >
          static_cast<uintmax_t>(std::numeric_limits<std::streamsize>::max())) {
    return absl::DataLossError(absl::StrCat(
        "wrong weight size: ", path.string(), "; expected ", expected_bytes));
  }
  ASSIGN_OR_RETURN(auto values, cuda::PageLockedHostArray<float>::Allocate(
                                    expected_bytes / sizeof(float)));
  std::ifstream input(path, std::ios::binary);
  if (!input.read(reinterpret_cast<char*>(values.data()), expected_bytes) ||
      input.peek() != std::ifstream::traits_type::eof()) {
    return absl::DataLossError(
        absl::StrCat("cannot read complete weight: ", path.string()));
  }
  for (float value : values) {
    if (!std::isfinite(value)) {
      return absl::DataLossError(
          absl::StrCat("nonfinite weight in ", path.string()));
    }
  }
  return values;
}

}  // namespace

absl::StatusOr<std::unique_ptr<Layer>> CreateReadout(
    cuda::Executor& executor, const Dimensions& dimensions) {
  constexpr DataType kType = DataType::BF16;
  ComposedLayerBuilder builder;
  RETURN_IF_ERROR(builder.add(EmbeddingLookupLayer::Create(
      executor, dimensions.vocab_size, dimensions.model_width, kType)));
  auto* embedding = static_cast<EmbeddingLookupLayer*>(builder.back());

  ComposedLayerBuilder mlp;
  RETURN_IF_ERROR(mlp.add(LayerNormLayer::Create(
      executor, dimensions.model_width, kEpsilon, kType)));
  RETURN_IF_ERROR(mlp.add(FullyConnectedLayer::Create(
      executor, dimensions.model_width, dimensions.feed_forward_width, kType)));
  RETURN_IF_ERROR(mlp.add(GeluLayer::Create(executor, kType)));
  RETURN_IF_ERROR(mlp.add(FullyConnectedLayer::Create(
      executor, dimensions.feed_forward_width, dimensions.model_width, kType)));
  ASSIGN_OR_RETURN(auto branch, mlp.create());
  RETURN_IF_ERROR(
      builder.add(std::make_unique<ResidualLayer>(std::move(branch))));
  RETURN_IF_ERROR(builder.add(LayerNormLayer::Create(
      executor, dimensions.model_width, kEpsilon, kType)));
  RETURN_IF_ERROR(builder.add(LanguageModelingHeadLayer::Create(embedding)));
  ASSIGN_OR_RETURN(auto result, builder.create());
  return std::unique_ptr<Layer>(std::move(result));
}

absl::Status LoadB0Weights(cuda::Executor& executor, Layer& readout,
                           const std::filesystem::path& directory) {
  if (directory.empty()) {
    return absl::InvalidArgumentError("checkpoint directory must not be empty");
  }
  std::vector<Buffer*> weights;
  std::unordered_set<const void*> seen;
  for (Buffer& weight : readout.weights()) {
    if (&weight.executor() != &executor) {
      return absl::InvalidArgumentError("weight belongs to another executor");
    }
    if (seen.insert(weight.data()).second) weights.push_back(&weight);
  }
  if (weights.size() != kCheckpointIndices.size()) {
    return absl::InvalidArgumentError(
        "B0 loader requires the nine unique CreateReadout weights");
  }
  std::vector<cuda::PageLockedHostArray<float>> staging;
  for (size_t index = 0; index < weights.size(); ++index) {
    const auto path =
        directory / absl::StrCat("weight_", kCheckpointIndices[index], ".bin");
    ASSIGN_OR_RETURN(auto values,
                     ReadWeight(path, weights[index]->size_bytes()));
    staging.push_back(std::move(values));
  }
  // No device bytes change until the last required file has passed validation.
  // If a copy fails, still drain previous copies before releasing staging.
  for (size_t index = 0; index < weights.size(); ++index) {
    const auto status = cuda::CudaStatus(
        cudaMemcpyAsync(weights[index]->data(), staging[index].data(),
                        staging[index].size_bytes(), cudaMemcpyHostToDevice,
                        executor.stream()),
        "upload isolated MLP weight");
    if (!status.ok()) {
      (void)executor.Synchronize();
      return status;
    }
  }
  return executor.Synchronize();
}

absl::StatusOr<cuda::PageLockedHostArray<TopTransition>> ScanVocabulary(
    cuda::Executor& executor, const Layer& readout, int vocab_size,
    int batch_size, absl::FunctionRef<void(int)> progress) {
  if (vocab_size <= 0 || vocab_size > std::numeric_limits<int>::max() - kTile ||
      batch_size <= 0 || batch_size % kTile != 0) {
    return absl::InvalidArgumentError(
        "vocabulary must be positive; batch_size must be a positive multiple "
        "of 16");
  }
  const int padded_vocab = (vocab_size + kTile - 1) / kTile * kTile;
  // Do not allocate a huge unused batch when scanning a tiny test vocabulary.
  batch_size = std::min(batch_size, padded_vocab);
  ASSIGN_OR_RETURN(auto ids,
                   cuda::PageLockedHostArray<int32_t>::Allocate(batch_size));
  ASSIGN_OR_RETURN(
      auto result,
      cuda::PageLockedHostArray<TopTransition>::Allocate(vocab_size));
  for (int start = 0; start < vocab_size;) {
    const int count = std::min(batch_size, vocab_size - start);
    const int rows = (count + kTile - 1) / kTile * kTile;
    for (int row = 0; row < rows; ++row) {
      ids[row] = row < count ? start + row : 0;
    }
    ASSIGN_OR_RETURN(auto tokens,
                     Buffer::Allocate(executor, rows * sizeof(int32_t)));
    RETURN_IF_ERROR(cuda::CudaStatus(
        cudaMemcpyAsync(tokens.data(), ids.data(), tokens.size_bytes(),
                        cudaMemcpyHostToDevice, executor.stream()),
        "upload vocabulary token IDs"));
    // This guard drains queued DMA even on an early layer/launch failure. In
    // the ordinary path the explicit synchronize below also permits ids reuse.
    struct Drain {
      cuda::Executor& executor;
      ~Drain() { (void)executor.Synchronize(); }
    } drain{executor};
    Tape tape;
    ASSIGN_OR_RETURN(auto logits, readout.fwd(executor, {tokens}, &tape));
    ASSIGN_OR_RETURN(auto top, ReadTopTransitions(executor, logits, rows,
                                                  vocab_size, padded_vocab));
    RETURN_IF_ERROR(cuda::CudaStatus(
        cudaMemcpyAsync(result.data() + start, top.data(),
                        count * sizeof(TopTransition), cudaMemcpyDeviceToHost,
                        executor.stream()),
        "download compact vocabulary transitions"));
    RETURN_IF_ERROR(executor.Synchronize());
    for (int row = start; row < start + count; ++row) {
      if (result[row].token < 0 || result[row].token >= vocab_size ||
          !std::isfinite(result[row].probability)) {
        return absl::DataLossError(
            absl::StrCat("nonfinite logits at source token ", row));
      }
    }
    start += count;
    progress(start);
  }
  return result;
}

}  // namespace pluto::llm::mlp_automaton
