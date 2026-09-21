#include "src/dataset/padded_line_dataset.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <limits>
#include <numeric>
#include <utility>

#include "absl/log/check.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "src/util/status_macros.h"

namespace pluto {

absl::StatusOr<std::unique_ptr<PaddedLineDataSetIterator>>
PaddedLineDataSetIterator::Create(cuda::Executor& executor,
                                  absl::string_view corpus_text,
                                  const tokenizer::Tokenizer& tokenizer,
                                  PaddedLineDataSetOptions options) {
  // Tokenizer IDs use int, whereas the device dataset schema is int32. Reject
  // incompatible platforms at compile time instead of copying the wrong width.
  static_assert(sizeof(int) == sizeof(int32_t));
  if (options.batch_size <= 0 || options.context_length <= 0 ||
      options.prompt_tokens <= 0 ||
      options.prompt_tokens > options.context_length)
    return absl::InvalidArgumentError("invalid padded-line batch dimensions");
  if (int64_t{options.batch_size} * options.context_length >
      std::numeric_limits<int>::max())
    return absl::InvalidArgumentError("batch token count exceeds int range");
  if (options.eos_token < 0 || options.eos_token >= tokenizer.vocab_size())
    return absl::InvalidArgumentError("EOS token is outside the vocabulary");
  if (corpus_text.empty())
    return absl::InvalidArgumentError("the line corpus must not be empty");
  if (corpus_text.back() == '\n')
    corpus_text.remove_suffix(1);
  std::vector<absl::string_view> lines = absl::StrSplit(corpus_text, '\n');
  if (lines.size() > std::numeric_limits<size_t>::max() /
                         static_cast<size_t>(options.context_length) /
                         sizeof(int))
    return absl::InvalidArgumentError("padded corpus byte size overflows");

  const size_t corpus_rows = lines.size() * options.context_length;
  ASSIGN_OR_RETURN(auto host_inputs, cuda::PageLockedHostArray<int>::Allocate(
                                         executor, corpus_rows));
  ASSIGN_OR_RETURN(auto host_targets, cuda::PageLockedHostArray<int>::Allocate(
                                          executor, corpus_rows));
  std::fill(host_inputs.begin(), host_inputs.end(), options.eos_token);
  std::fill(host_targets.begin(), host_targets.end(), -1);
  std::vector<int> lengths;
  lengths.reserve(lines.size());
  for (size_t line_index = 0; line_index < lines.size(); ++line_index) {
    absl::string_view line = lines[line_index];
    // Accept CRLF as a line ending, but preserve every other text byte.
    if (!line.empty() && line.back() == '\r')
      line.remove_suffix(1);
    if (absl::StripAsciiWhitespace(line).empty())
      return absl::InvalidArgumentError(
          absl::StrCat("line ", line_index + 1, " is empty"));
    ASSIGN_OR_RETURN(auto tokens, tokenizer.Encode(executor, line));
    if (tokens.size() < static_cast<size_t>(options.prompt_tokens) ||
        tokens.size() > static_cast<size_t>(options.context_length))
      return absl::InvalidArgumentError(
          absl::StrCat("line ", line_index + 1, " has ", tokens.size(),
                       " tokens; expected between ", options.prompt_tokens,
                       " and ", options.context_length));
    if (std::any_of(tokens.begin(), tokens.end(), [&](int token) {
          return token < 0 || token >= tokenizer.vocab_size();
        }))
      return absl::InvalidArgumentError(absl::StrCat(
          "line ", line_index + 1, " contains an invalid token ID"));

    const size_t base = line_index * options.context_length;
    std::copy(tokens.begin(), tokens.end(), host_inputs.data() + base);
    // Row P-1 consumes the final prompt token and predicts the first suffix
    // token. In particular, mask P-1 targets, not P targets (an easy
    // off-by-one).
    for (size_t row = options.prompt_tokens - 1; row + 1 < tokens.size(); ++row)
      host_targets[base + row] = tokens[row + 1];
    host_targets[base + tokens.size() - 1] = options.eos_token;
    lengths.push_back(static_cast<int>(tokens.size()));
  }

  ASSIGN_OR_RETURN(auto corpus_inputs,
                   cuda::Buffer::Allocate(executor, host_inputs.size_bytes()));
  ASSIGN_OR_RETURN(auto corpus_targets,
                   cuda::Buffer::Allocate(executor, host_targets.size_bytes()));
  RETURN_IF_ERROR(cuda::CudaStatus(
      cudaMemcpyAsync(corpus_inputs.data(), host_inputs.data(),
                      host_inputs.size_bytes(), cudaMemcpyHostToDevice,
                      executor.stream()),
      "cudaMemcpyAsync(padded corpus inputs)"));
  RETURN_IF_ERROR(cuda::CudaStatus(
      cudaMemcpyAsync(corpus_targets.data(), host_targets.data(),
                      host_targets.size_bytes(), cudaMemcpyHostToDevice,
                      executor.stream()),
      "cudaMemcpyAsync(padded corpus targets)"));
  // host_targets may be released immediately: its pinned allocation is freed
  // on executor after these uploads. No compute-stream synchronization is
  // necessary. host_inputs stays alive for later exact greedy verification.
  const auto allocate_batch =
      [&](size_t samples) -> absl::StatusOr<BatchBuffers> {
    const size_t bytes = samples * options.context_length * sizeof(int);
    ASSIGN_OR_RETURN(auto inputs, cuda::Buffer::Allocate(executor, bytes));
    ASSIGN_OR_RETURN(auto targets, cuda::Buffer::Allocate(executor, bytes));
    return BatchBuffers{std::move(inputs), std::move(targets)};
  };
  ASSIGN_OR_RETURN(auto full_batch, allocate_batch(options.batch_size));
  std::optional<BatchBuffers> partial_batch;
  const size_t remainder = lines.size() % options.batch_size;
  if (remainder != 0) {
    ASSIGN_OR_RETURN(partial_batch, allocate_batch(remainder));
  }
  return absl::WrapUnique(new PaddedLineDataSetIterator(
      executor, options, std::move(host_inputs), std::move(lengths),
      std::move(corpus_inputs), std::move(corpus_targets),
      std::move(full_batch), std::move(partial_batch)));
}

PaddedLineDataSetIterator::PaddedLineDataSetIterator(
    cuda::Executor& executor, PaddedLineDataSetOptions options,
    cuda::PageLockedHostArray<int> host_inputs, std::vector<int> lengths,
    cuda::Buffer corpus_inputs, cuda::Buffer corpus_targets,
    BatchBuffers full_batch, std::optional<BatchBuffers> partial_batch)
    : executor_(executor),
      options_(options),
      host_inputs_(std::move(host_inputs)),
      lengths_(std::move(lengths)),
      corpus_inputs_(std::move(corpus_inputs)),
      corpus_targets_(std::move(corpus_targets)),
      full_batch_(std::move(full_batch)),
      partial_batch_(std::move(partial_batch)),
      order_(lengths_.size()),
      random_(options.seed) {
  for (int length : lengths_)
    supervised_row_count_ += length - options_.prompt_tokens + 1;
  BeginEpoch();
}

void PaddedLineDataSetIterator::BeginEpoch() {
  std::iota(order_.begin(), order_.end(), 0);
  if (options_.shuffle)
    std::shuffle(order_.begin(), order_.end(), random_);
  next_sample_ = 0;
}

absl::StatusOr<DataBatch> PaddedLineDataSetIterator::Next() {
  if (next_sample_ == order_.size())
    BeginEpoch();
  const size_t samples = std::min(static_cast<size_t>(options_.batch_size),
                                  order_.size() - next_sample_);
  const BatchBuffers& batch =
      samples == static_cast<size_t>(options_.batch_size) ? full_batch_
                                                          : *partial_batch_;
  const size_t row_bytes = options_.context_length * sizeof(int);
  int valid_rows = 0;
  for (size_t sample = 0; sample < samples; ++sample) {
    const size_t source_index = order_[next_sample_ + sample];
    const size_t destination_offset = sample * options_.context_length;
    const size_t source_offset = source_index * options_.context_length;
    RETURN_IF_ERROR(cuda::CudaStatus(
        cudaMemcpyAsync(
            static_cast<int*>(batch.inputs.data()) + destination_offset,
            static_cast<const int*>(corpus_inputs_.data()) + source_offset,
            row_bytes, cudaMemcpyDeviceToDevice, executor_.stream()),
        "cudaMemcpyAsync(padded batch inputs)"));
    RETURN_IF_ERROR(cuda::CudaStatus(
        cudaMemcpyAsync(
            static_cast<int*>(batch.targets.data()) + destination_offset,
            static_cast<const int*>(corpus_targets_.data()) + source_offset,
            row_bytes, cudaMemcpyDeviceToDevice, executor_.stream()),
        "cudaMemcpyAsync(padded batch targets)"));
    valid_rows += lengths_[source_index] - options_.prompt_tokens + 1;
  }
  next_sample_ += samples;
  return DataBatch{.inputs = batch.inputs,
                   .targets = batch.targets,
                   .batch_size = static_cast<int32_t>(samples),
                   .sequence_length = options_.context_length,
                   .supervised_row_count = valid_rows};
}

absl::Status PaddedLineDataSetIterator::Reset() {
  random_.seed(options_.seed);
  BeginEpoch();
  return absl::OkStatus();
}

size_t PaddedLineDataSetIterator::batches_per_epoch() const {
  return sample_count() / options_.batch_size +
         (sample_count() % options_.batch_size != 0);
}

absl::Span<const int> PaddedLineDataSetIterator::sample_tokens(
    size_t index) const {
  CHECK_LT(index, sample_count());
  return absl::MakeConstSpan(
      host_inputs_.data() + index * options_.context_length, lengths_[index]);
}

}  // namespace pluto
