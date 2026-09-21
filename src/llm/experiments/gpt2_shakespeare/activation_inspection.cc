#include "src/llm/experiments/gpt2_shakespeare/activation_inspection.h"

#include <cuda_runtime_api.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>

#include "absl/memory/memory.h"
#include "absl/strings/escaping.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "src/cuda/page_locked_host_array.h"
#include "src/llm/vocabulary_readout.h"
#include "src/util/status_macros.h"

namespace pluto::llm {
namespace {

// Validate even unsupported outputs: a diagnostic must not silently turn a
// malformed shape into an invented projection. For symbolic batch, the byte
// size supplies the actual number of complete samples, never a wildcard axis.
absl::Status ValidateStorage(const ActivationType& type, const Buffer& buffer) {
  RETURN_IF_ERROR(type.Validate());
  size_t expected = 0;
  switch (type.data_type()) {
    case DataType::FP32:
    case DataType::INT32:
      expected = sizeof(uint32_t);
      break;
    case DataType::BF16:
    case DataType::FP16:
      expected = sizeof(uint16_t);
      break;
    case DataType::FP8:
      expected = sizeof(uint8_t);
      break;
  }
  const auto dimensions = type.dimensions();
  const bool batched =
      !dimensions.empty() && dimensions[0] == ActivationType::kBatchDimension;
  for (int64_t dimension : dimensions) {
    if (dimension == ActivationType::kBatchDimension)
      continue;
    if (static_cast<uint64_t>(dimension) >
        std::numeric_limits<size_t>::max() / expected)
      return absl::InvalidArgumentError("activation shape byte size overflows");
    expected *= static_cast<size_t>(dimension);
  }
  if (batched ? buffer.size_bytes() == 0 || buffer.size_bytes() % expected != 0
              : buffer.size_bytes() != expected)
    return absl::InvalidArgumentError(
        "activation byte size disagrees with shape");
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<std::unique_ptr<NeighboringVocabInspector>>
NeighboringVocabInspector::Create(cuda::Executor& executor,
                                  const Buffer& token_embedding, int vocab_size,
                                  int model_width, int positions,
                                  int position_offset, double min_prob) {
  if (!std::isfinite(min_prob) || min_prob < 0.0 || min_prob > 1.0)
    return absl::InvalidArgumentError("min_prob must be finite and in [0, 1]");
  if (vocab_size < 3 || model_width <= 0 || positions <= 0 ||
      position_offset < 0 ||
      position_offset > std::numeric_limits<int>::max() - (positions - 1))
    return absl::InvalidArgumentError(
        "inspection requires at least three vocabulary tokens, positive width "
        "and positions, and nonnegative nonoverflowing position numbering");
  if (&token_embedding.executor() != &executor)
    return absl::InvalidArgumentError(
        "token embedding belongs to another executor");
  if (static_cast<size_t>(model_width) >
      std::numeric_limits<size_t>::max() / sizeof(float))
    return absl::InvalidArgumentError("token embedding row size overflows");
  const size_t row_bytes = static_cast<size_t>(model_width) * sizeof(float);
  const size_t rows = token_embedding.size_bytes() / row_bytes;
  if (token_embedding.size_bytes() % row_bytes != 0 ||
      rows < static_cast<size_t>(vocab_size) ||
      rows > static_cast<size_t>(std::numeric_limits<int>::max()))
    return absl::InvalidArgumentError(
        "token embedding must contain complete FP32 vocabulary rows");
  return absl::WrapUnique(new NeighboringVocabInspector(
      executor, token_embedding, vocab_size, model_width, positions,
      position_offset, static_cast<int>(rows), min_prob));
}

absl::Status NeighboringVocabInspector::ValidateExecutor(
    const cuda::Executor& executor) const {
  if (&executor != &executor_)
    return absl::InvalidArgumentError(
        "activation inspection uses another executor");
  return absl::OkStatus();
}

std::string NeighboringVocabInspector::NextPath(absl::string_view name) {
  if (scopes_.empty())
    return absl::StrCat(name, "[", next_root_++, "]");
  Scope& parent = scopes_.back();
  return absl::StrCat(parent.path, "/", name, "[", parent.next_child++, "]");
}

absl::Status NeighboringVocabInspector::EnterCombinator(
    cuda::Executor& executor, absl::string_view layer_name) {
  RETURN_IF_ERROR(ValidateExecutor(executor));
  exited_scope_.reset();
  std::string path = NextPath(layer_name);
  scopes_.push_back({std::string(layer_name), std::move(path)});
  return absl::OkStatus();
}

absl::Status NeighboringVocabInspector::ExitCombinator(
    cuda::Executor& executor) {
  RETURN_IF_ERROR(ValidateExecutor(executor));
  if (scopes_.empty())
    return absl::FailedPreconditionError(
        "activation inspection scope stack is empty");
  exited_scope_ = std::move(scopes_.back());
  scopes_.pop_back();
  return absl::OkStatus();
}

absl::Status NeighboringVocabInspector::ActivationHook(
    cuda::Executor& executor, absl::string_view layer_name,
    absl::Span<const ActivationType> activation_types,
    absl::Span<Buffer> activations) {
  RETURN_IF_ERROR(ValidateExecutor(executor));
  if (activation_types.size() != activations.size())
    return absl::InvalidArgumentError(
        "activation inspection type/buffer arity differs");
  std::string path = exited_scope_ && exited_scope_->name == layer_name
                         ? exited_scope_->path
                         : NextPath(layer_name);
  exited_scope_.reset();
  for (size_t index = 0; index < activations.size(); ++index) {
    const Buffer& activation = activations[index];
    const ActivationType& type = activation_types[index];
    if (&activation.executor() != &executor)
      return absl::InvalidArgumentError(
          "activation buffer belongs to another executor");
    RETURN_IF_ERROR(ValidateStorage(type, activation));
    std::optional<Buffer> neighbors;
    const auto dimensions = type.dimensions();
    // GPT-2 token vectors are [batch, sequence, feature]. Rank-two outputs
    // may instead be one scalar loss per token, not vectors to unembed.
    const bool token_rows = dimensions.size() == 3 &&
                            dimensions[0] == ActivationType::kBatchDimension;
    if (!token_rows || (type.data_type() != DataType::FP32 &&
                        type.data_type() != DataType::BF16))
      continue;
    const int64_t width = dimensions.back();
    const size_t element_bytes =
        type.data_type() == DataType::FP32 ? sizeof(float) : sizeof(uint16_t);
    const size_t rows =
        activation.size_bytes() / (static_cast<size_t>(width) * element_bytes);
    if (static_cast<size_t>(positions_) > rows)
      return absl::InvalidArgumentError(
          "requested positions exceed activation token rows");
    const int64_t padded_vocabulary =
        (static_cast<int64_t>(vocab_size_) + 15) / 16 * 16;
    const bool vocabulary_logits =
        type.data_type() == DataType::FP32 &&
        (width == vocab_size_ || width == embedding_rows_ ||
         width == padded_vocabulary);
    if (width == model_width_ && vocabulary_logits) {
      // Unusual small models may use the same width for hidden vectors and
      // logits. Shape metadata alone cannot distinguish their semantics.
      continue;
    } else if (width == model_width_) {
      ASSIGN_OR_RETURN(neighbors, ReadEmbeddingNeighbors(
                                      executor, activation, type.data_type(),
                                      token_embedding_, positions_, vocab_size_,
                                      model_width_));
    } else if (vocabulary_logits) {
      if (width > std::numeric_limits<int>::max())
        return absl::InvalidArgumentError(
            "vocabulary logit stride overflows int");
      ASSIGN_OR_RETURN(neighbors,
                       ReadTopThreeTokens(executor, activation, positions_,
                                          vocab_size_, static_cast<int>(width)));
    } else {
      continue;
    }
    entries_.push_back(
        {.label = activations.size() == 1
                      ? path
                      : absl::StrCat(path, "/output[", index, "]"),
         .neighbors = std::move(*neighbors)});
  }
  return absl::OkStatus();
}

absl::Status NeighboringVocabInspector::Print(
    cuda::Executor& executor, const tokenizer::Detokenizer& detokenizer,
    absl::Span<const int> input_tokens, std::ostream& output) const {
  RETURN_IF_ERROR(ValidateExecutor(executor));
  if (!scopes_.empty())
    return absl::FailedPreconditionError(
        "cannot print inside an unfinished layer scope");
  if (detokenizer.vocab_size() != vocab_size_)
    return absl::InvalidArgumentError(
        "detokenizer vocabulary differs from inspected model");
  if (input_tokens.size() != static_cast<size_t>(positions_))
    return absl::InvalidArgumentError(
        "input token count must match the inspected positions");
  for (int token : input_tokens)
    if (token < 0 || token >= vocab_size_)
      return absl::InvalidArgumentError(
          "input token is outside the inspected vocabulary");
  const size_t supported = entries_.size();
  if (supported >
      std::numeric_limits<size_t>::max() / static_cast<size_t>(positions_))
    return absl::InvalidArgumentError("inspection result count overflows");
  ASSIGN_OR_RETURN(auto host,
                   cuda::PageLockedHostArray<TopThreeTokens>::Allocate(
                       executor, supported * static_cast<size_t>(positions_)));
  size_t result_index = 0;
  for (const Entry& entry : entries_) {
    RETURN_IF_ERROR(cuda::CudaStatus(
        cudaMemcpyAsync(host.data() + result_index, entry.neighbors.data(),
                        entry.neighbors.size_bytes(), cudaMemcpyDeviceToHost,
                        executor.stream()),
        "cudaMemcpyAsync(activation vocabulary neighbors)"));
    result_index += positions_;
  }
  if (supported != 0)
    RETURN_IF_ERROR(executor.Synchronize());
  for (int position = 0; position < positions_; ++position) {
    bool printed_position = false;
    size_t entry_index = 0;
    for (const Entry& entry : entries_) {
      const TopThreeTokens& neighbors =
          host[entry_index++ * static_cast<size_t>(positions_) + position];
      bool finite = true;
      for (int rank = 0; rank < 3; ++rank)
        finite = finite && neighbors.tokens[rank] >= 0 &&
                 neighbors.tokens[rank] < vocab_size_ &&
                 std::isfinite(neighbors.probabilities[rank]);
      // The GPU reduction marks invalid logical logits with ID -1 and NaN.
      // Omit that row without asking the detokenizer to decode an invalid ID
      // or discarding diagnostics for the remaining positions and layers.
      if (!finite)
        continue;
      bool printed_layer = false;
      for (int rank = 0; rank < 3; ++rank) {
        if (neighbors.probabilities[rank] < min_prob_)
          continue;
        ASSIGN_OR_RETURN(std::string token,
                         detokenizer.Decode(
                             absl::Span<const int>(&neighbors.tokens[rank], 1)));
        if (!printed_position) {
          ASSIGN_OR_RETURN(
              auto input_token,
              detokenizer.Decode(input_tokens.subspan(position, 1)));
          output << "Position " << position_offset_ + position << " (\""
                 << absl::CEscape(input_token) << "\"):\n";
          printed_position = true;
        }
        if (!printed_layer) {
          output << "  " << entry.label << ": ";
          printed_layer = true;
        } else {
          output << ", ";
        }
        const double percentage = 100.0 * neighbors.probabilities[rank];
        // Tiny but nonzero probabilities are informative in large vocabularies;
        // fixed two-decimal formatting would misleadingly display them as zero.
        const std::string formatted =
            percentage > 0.0 && percentage < 0.01
                ? absl::StrFormat("%.4g%%", percentage)
                : absl::StrFormat("%.2f%%", percentage);
        output << '"' << absl::CEscape(token) << "\" (" << formatted << ')';
      }
      if (printed_layer)
        output << '\n';
    }
  }
  if (!output.good())
    return absl::UnknownError("writing activation inspection output failed");
  return absl::OkStatus();
}

}  // namespace pluto::llm
