#include "src/llm/attention_probability_printer.h"

#include <cuda_runtime_api.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>

#include "absl/strings/escaping.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "src/cuda/page_locked_host_array.h"
#include "src/util/status_macros.h"

namespace pluto::llm {

AttentionProbabilityPrinter::AttentionProbabilityPrinter(
    cuda::Executor& executor)
    : executor_(executor) {
  hooks_.attention_probabilities_hook =
      [this](cuda::Executor& executor, absl::string_view name,
             const ActivationType& type, const cuda::Buffer& probabilities) {
        return Record(executor, name, type, probabilities);
      };
  hooks_.enter_combinator = [this](cuda::Executor& executor,
                                   absl::string_view name) -> absl::Status {
    RETURN_IF_ERROR(ValidateExecutor(executor));
    scopes_.push_back({NextPath(name)});
    return absl::OkStatus();
  };
  hooks_.exit_combinator = [this](cuda::Executor& executor) -> absl::Status {
    RETURN_IF_ERROR(ValidateExecutor(executor));
    if (scopes_.empty())
      return absl::FailedPreconditionError(
          "attention inspection scope stack is empty");
    scopes_.pop_back();
    return absl::OkStatus();
  };
}

absl::Status AttentionProbabilityPrinter::ValidateExecutor(
    const cuda::Executor& executor) const {
  if (&executor != &executor_)
    return absl::InvalidArgumentError(
        "attention inspection uses another executor");
  return absl::OkStatus();
}

std::string AttentionProbabilityPrinter::NextPath(absl::string_view name) {
  if (scopes_.empty())
    return absl::StrCat(name, "[", next_root_++, "]");
  Scope& parent = scopes_.back();
  return absl::StrCat(parent.path, "/", name, "[", parent.next_child++, "]");
}

absl::Status AttentionProbabilityPrinter::Record(
    cuda::Executor& executor, absl::string_view name,
    const ActivationType& type, const cuda::Buffer& probabilities) {
  RETURN_IF_ERROR(ValidateExecutor(executor));
  if (&probabilities.executor() != &executor)
    return absl::InvalidArgumentError(
        "attention probability buffer belongs to another executor");
  RETURN_IF_ERROR(type.Validate());
  const auto dims = type.dimensions();
  // This UI generates one sequence at a time. Unlike the usual symbolic
  // layer signatures, the attention hook resolves every dimension explicitly.
  if (type.data_type() != DataType::FP32 || dims.size() != 4 || dims[0] != 1 ||
      dims[2] != dims[3])
    return absl::InvalidArgumentError(
        "attention inspection expects concrete FP32 [1, heads, sequence, "
        "sequence] probabilities");
  size_t bytes = sizeof(float);
  for (int64_t dim : dims) {
    if (dim <= 0)
      return absl::InvalidArgumentError(
          "attention inspection dimensions must be concrete and positive");
    if (static_cast<uint64_t>(dim) > std::numeric_limits<size_t>::max() / bytes)
      return absl::InvalidArgumentError("attention probability size overflows");
    bytes *= static_cast<size_t>(dim);
  }
  if (probabilities.size_bytes() != bytes)
    return absl::InvalidArgumentError(
        "attention probability buffer size differs from its shape");
  entries_.push_back({NextPath(name), static_cast<size_t>(dims[1]),
                      static_cast<size_t>(dims[2]), probabilities});
  return absl::OkStatus();
}

absl::Status AttentionProbabilityPrinter::Print(
    cuda::Executor& executor, const tokenizer::Detokenizer& detokenizer,
    absl::Span<const int> input_prefix, int produced_token,
    std::ostream& output) {
  RETURN_IF_ERROR(ValidateExecutor(executor));
  if (!scopes_.empty())
    return absl::FailedPreconditionError(
        "cannot print attention inside an unfinished layer scope");
  if (input_prefix.empty())
    return absl::InvalidArgumentError(
        "attention input prefix must not be empty");
  if (produced_token < 0 || produced_token >= detokenizer.vocab_size())
    return absl::InvalidArgumentError("output token is outside the vocabulary");

  // Decode before copying or printing, so an invalid token cannot produce a
  // misleading partial attention report. CEscape keeps quotes, whitespace,
  // and partial UTF-8 byte tokens on a single unambiguous output line.
  ASSIGN_OR_RETURN(auto produced_text, detokenizer.Decode({&produced_token, 1}));
  std::vector<std::string> labels;
  labels.reserve(input_prefix.size());
  for (int token : input_prefix) {
    if (token < 0 || token >= detokenizer.vocab_size())
      return absl::InvalidArgumentError(
          "input token is outside the vocabulary");
    ASSIGN_OR_RETURN(auto text, detokenizer.Decode({&token, 1}));
    labels.push_back(absl::CEscape(text));
  }

  const size_t n = input_prefix.size();
  // A valid zero-transformer model has no attention hooks to record. Keep its
  // generation behavior unchanged, while making the empty report explicit.
  if (entries_.empty()) {
    output << "Output token " << n << " (\"" << absl::CEscape(produced_text)
           << "\", id " << produced_token << "):\n"
           << "  No attention layers.\n";
    if (!output.good())
      return absl::InternalError("failed to write attention inspection output");
    next_root_ = 0;
    return absl::OkStatus();
  }
  if (n > std::numeric_limits<size_t>::max() / n)
    return absl::InvalidArgumentError("attention crop size overflows");
  const size_t square = n * n;
  size_t total = 0;
  for (const Entry& entry : entries_) {
    if (n > entry.sequence)
      return absl::InvalidArgumentError(
          "input prefix exceeds the attention sequence length");
    if (entry.heads > (std::numeric_limits<size_t>::max() - total) / square)
      return absl::InvalidArgumentError("attention crop count overflows");
    total += entry.heads * square;
  }
  ASSIGN_OR_RETURN(auto host,
                   cuda::PageLockedHostArray<float>::Allocate(executor, total));
  size_t offset = 0;
  for (const Entry& entry : entries_)
    for (size_t head = 0; head < entry.heads; ++head) {
      // Source rows still have the complete padded sequence stride. Copy a
      // leading NxN square per head without downloading future/padding rows
      // or columns. The causal upper triangle is omitted when rendering.
      const auto* source =
          static_cast<const float*>(entry.probabilities.data()) +
          head * entry.sequence * entry.sequence;
      RETURN_IF_ERROR(cuda::CudaStatus(
          cudaMemcpy2DAsync(host.data() + offset, n * sizeof(float), source,
                            entry.sequence * sizeof(float), n * sizeof(float),
                            n, cudaMemcpyDeviceToHost, executor.stream()),
          "cudaMemcpy2DAsync(attention inspection)"));
      offset += square;
    }
  RETURN_IF_ERROR(executor.Synchronize());

  // Validate only causally related entries; masked/future entries never affect
  // the report, even if the producer leaves unused storage unspecified.
  for (size_t matrix = 0; matrix < total; matrix += square)
    for (size_t row = 0; row < n; ++row)
      for (size_t col = 0; col <= row; ++col) {
        const float p = host[matrix + row * n + col];
        // Reconstruction from a fused kernel's FP32 log-sum-exp statistics
        // can round a unit probability very slightly above one.
        if (!std::isfinite(p) || p < 0.0f || p > 1.0f + 1e-5f)
          return absl::DataLossError(
              "attention contains an invalid probability");
      }

  output << "Output token " << n << " (\"" << absl::CEscape(produced_text)
         << "\", id " << produced_token << "):\n";
  output << "  Key columns:";
  for (size_t col = 0; col < n; ++col)
    output << (col == 0 ? " " : ", ") << col << " (\"" << labels[col] << "\")";
  output << '\n';
  offset = 0;
  for (const Entry& entry : entries_)
    for (size_t head = 0; head < entry.heads; ++head) {
      output << "  " << entry.path << ", head " << head << ":\n";
      for (size_t row = 0; row < n; ++row) {
        output << "    Query " << row << " (\"" << labels[row] << "\"): [";
        for (size_t col = 0; col <= row; ++col)
          output << (col == 0 ? "" : ", ")
                 << absl::StrFormat("%.6f", host[offset + row * n + col]);
        output << "]\n";
      }
      offset += square;
    }
  if (!output.good())
    return absl::InternalError("failed to write attention inspection output");
  entries_.clear();
  next_root_ = 0;
  return absl::OkStatus();
}

}  // namespace pluto::llm
