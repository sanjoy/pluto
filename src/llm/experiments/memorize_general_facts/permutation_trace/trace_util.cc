#include "src/llm/experiments/memorize_general_facts/permutation_trace/trace_util.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>

#include "absl/status/status.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"

namespace pluto::llm::permutation_trace {
namespace {

template <typename T>
T Read(absl::Span<const uint8_t> data, size_t index) {
  T result;
  std::memcpy(&result, data.data() + index * sizeof(T), sizeof(T));
  return result;
}

double Value(absl::Span<const uint8_t> data, size_t index,
             absl::string_view dtype) {
  if (dtype == "int32")
    return Read<int32_t>(data, index);
  if (dtype == "bf16")
    return std::bit_cast<float>(uint32_t{Read<uint16_t>(data, index)} << 16);
  return Read<float>(data, index);
}

}  // namespace

absl::StatusOr<std::vector<int>> ParsePermutation(absl::string_view text,
                                                  int vocabulary_size,
                                                  int eos) {
  if (vocabulary_size <= 0 || eos < 0 || eos >= vocabulary_size)
    return absl::InvalidArgumentError("invalid vocabulary or EOS");
  std::vector<int> result;
  std::vector<bool> seen(vocabulary_size);
  for (auto word :
       absl::StrSplit(text, absl::ByAnyChar(" \t\r\n"), absl::SkipEmpty())) {
    int id;
    if (!absl::SimpleAtoi(word, &id) || id < 0 || id >= vocabulary_size ||
        seen[id])
      return absl::InvalidArgumentError("permutation is not a bijection");
    seen[id] = true;
    result.push_back(id);
  }
  if (result.size() != static_cast<size_t>(vocabulary_size) ||
      result[eos] != eos)
    return absl::InvalidArgumentError(
        "permutation must cover vocabulary and fix EOS");
  return result;
}

absl::StatusOr<Difference> Compare(const TensorDescription& d,
                                   absl::Span<const uint8_t> baseline,
                                   absl::Span<const uint8_t> candidate,
                                   absl::Span<const int> permutation) {
  if (d.dtype != "fp32" && d.dtype != "bf16" && d.dtype != "int32")
    return absl::InvalidArgumentError("unsupported tensor dtype");
  const size_t scalar_bytes = d.dtype == "bf16" ? 2 : 4;
  if (baseline.size() != candidate.size() ||
      baseline.size() % scalar_bytes != 0)
    return absl::InvalidArgumentError("tensor byte sizes disagree");
  size_t elements = 1;
  for (int64_t dimension : d.shape) {
    if (dimension <= 0 || static_cast<uint64_t>(dimension) >
                              std::numeric_limits<size_t>::max() / elements)
      return absl::InvalidArgumentError("invalid tensor shape");
    elements *= dimension;
  }
  if (elements != baseline.size() / scalar_bytes)
    return absl::InvalidArgumentError("tensor shape and byte size disagree");
  if (d.alignment != "none" && d.alignment != "token_ids" &&
      d.alignment != "vocab_rows" && d.alignment != "vocab_columns")
    return absl::InvalidArgumentError("unknown alignment");
  if ((d.alignment == "vocab_rows" || d.alignment == "vocab_columns") &&
      d.shape.empty())
    return absl::InvalidArgumentError("vocabulary alignment needs dimensions");
  if (d.alignment == "token_ids" && d.dtype != "int32")
    return absl::InvalidArgumentError("token ID alignment needs int32");
  std::vector<int> inverse(permutation.size(), -1);
  for (size_t old = 0; old < permutation.size(); ++old) {
    const int renamed = permutation[old];
    if (renamed < 0 || static_cast<size_t>(renamed) >= permutation.size() ||
        inverse[renamed] != -1)
      return absl::InvalidArgumentError(
          "alignment permutation is not a bijection");
    inverse[renamed] = old;
  }
  if (!permutation.empty() &&
      ((d.alignment == "vocab_rows" &&
        static_cast<uint64_t>(d.shape.front()) < permutation.size()) ||
       (d.alignment == "vocab_columns" &&
        static_cast<uint64_t>(d.shape.back()) < permutation.size())))
    return absl::InvalidArgumentError(
        "vocabulary exceeds aligned tensor dimension");

  Difference result{.elements = elements};
  long double sum_squares = 0;
  for (size_t i = 0; i < elements; ++i) {
    size_t j = i;
    if (!permutation.empty() && d.alignment == "vocab_rows") {
      const size_t stride = elements / d.shape.front();
      const size_t row = i / stride;
      if (row < permutation.size())
        j = static_cast<size_t>(permutation[row]) * stride + i % stride;
    } else if (!permutation.empty() && d.alignment == "vocab_columns") {
      const size_t stride = d.shape.back();
      const size_t column = i % stride;
      if (column < permutation.size())
        j = i - column + permutation[column];
    }
    const double a = Value(baseline, i, d.dtype);
    double b = Value(candidate, j, d.dtype);
    bool equal =
        std::memcmp(baseline.data() + i * scalar_bytes,
                    candidate.data() + j * scalar_bytes, scalar_bytes) == 0;
    if (!permutation.empty() && d.alignment == "token_ids") {
      const int id = Read<int32_t>(candidate, j);
      if (id != -1 && (id < 0 || static_cast<size_t>(id) >= inverse.size()))
        return absl::InvalidArgumentError("out-of-range token ID in capture");
      b = id == -1 ? -1 : inverse[id];
      equal = a == b;
    }
    result.mismatches += !equal;
    if (equal || a == b)
      continue;  // Equal padded -infinity values have zero distance.
    const double difference = std::isfinite(a) && std::isfinite(b)
                                  ? std::abs(a - b)
                                  : std::numeric_limits<double>::infinity();
    result.max_abs = std::max(result.max_abs, difference);
    sum_squares += static_cast<long double>(difference) * difference;
  }
  result.l2 = std::sqrt(sum_squares);
  return result;
}

std::string ShapeText(absl::Span<const int64_t> shape) {
  std::string result;
  for (int64_t dimension : shape) {
    if (!result.empty())
      result += ',';
    absl::StrAppend(&result, dimension);
  }
  return result;
}

}  // namespace pluto::llm::permutation_trace
