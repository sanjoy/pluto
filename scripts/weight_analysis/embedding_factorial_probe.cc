#include "scripts/weight_analysis/embedding_factorial_probe.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>

#include "absl/strings/str_cat.h"
#include "scripts/weight_analysis/token_trace_probe.h"
#include "src/llm/recipes/gpt2.h"
#include "src/util/status_macros.h"

namespace pluto::weight_analysis {

absl::Status ValidateFactorialRows(absl::Span<const int32_t> rows,
                                   int case_count, int rows_per_case,
                                   int context_length) {
  if (case_count <= 0 || rows_per_case <= 0 || context_length <= 0 ||
      rows_per_case > context_length ||
      rows.size() != uint64_t{static_cast<unsigned>(case_count)} *
                         static_cast<unsigned>(rows_per_case)) {
    return absl::InvalidArgumentError("invalid selected-row geometry");
  }
  for (int c = 0; c < case_count; ++c) {
    int previous = -1;
    for (int i = 0; i < rows_per_case; ++i) {
      const int row = rows[static_cast<size_t>(c) * rows_per_case + i];
      if (row <= previous || row >= context_length) {
        return absl::InvalidArgumentError(
            "scored rows must be in range and strictly increasing per case");
      }
      previous = row;
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<std::vector<int32_t>> LoadFactorialRows(
    const std::filesystem::path& path, int case_count, int rows_per_case,
    int context_length) {
  if constexpr (std::endian::native != std::endian::little) {
    return absl::UnimplementedError("selected-row files require little endian");
  }
  if (case_count <= 0 || rows_per_case <= 0 || context_length <= 0 ||
      rows_per_case > context_length) {
    return absl::InvalidArgumentError("invalid selected-row geometry");
  }
  const uint64_t count = uint64_t{static_cast<unsigned>(case_count)} *
                         static_cast<unsigned>(rows_per_case);
  // A bounded metadata file, not an unbounded allocation from an input size.
  if (count > 1024 * 1024 || std::filesystem::is_symlink(path) ||
      !std::filesystem::is_regular_file(path) ||
      std::filesystem::file_size(path) != count * sizeof(int32_t)) {
    return absl::InvalidArgumentError("invalid selected-row file type or size");
  }
  std::vector<int32_t> rows(count);
  std::ifstream file(path, std::ios::binary);
  if (!file.read(reinterpret_cast<char*>(rows.data()),
                 count * sizeof(int32_t)) ||
      file.peek() != EOF) {
    return absl::DataLossError("could not read exact selected-row file");
  }
  RETURN_IF_ERROR(
      ValidateFactorialRows(rows, case_count, rows_per_case, context_length));
  return rows;
}

absl::StatusOr<FactorialTokenScore> ScoreFactorialToken(
    absl::Span<const float> logits, int target) {
  if (logits.empty() || logits.size() > std::numeric_limits<int>::max() ||
      target < 0 || static_cast<size_t>(target) >= logits.size()) {
    return absl::InvalidArgumentError("invalid vocabulary or target");
  }
  int argmax = 0;
  for (size_t i = 0; i < logits.size(); ++i) {
    if (!std::isfinite(logits[i])) {
      return absl::DataLossError("nonfinite logical-vocabulary logit");
    }
    if (logits[i] > logits[argmax]) argmax = i;
  }
  const double maximum = logits[argmax];
  double sum = 0;
  int rank = 1;
  for (size_t i = 0; i < logits.size(); ++i) {
    sum += std::exp(static_cast<double>(logits[i]) - maximum);
    if (logits[i] > logits[target] ||
        (logits[i] == logits[target] && i < static_cast<size_t>(target))) {
      ++rank;
    }
  }
  return FactorialTokenScore{std::log(sum) + (maximum - logits[target]), argmax,
                             rank};
}

absl::StatusOr<std::vector<int>> ChangedEmbeddingRows(
    absl::Span<const float> original, absl::Span<const float> patched,
    int vocabulary, int width) {
  if (vocabulary <= 0 || width <= 0 || original.size() != patched.size() ||
      original.size() % width ||
      original.size() / width < static_cast<size_t>(vocabulary)) {
    return absl::InvalidArgumentError("invalid embedding-table geometry");
  }
  std::vector<int> changed;
  for (size_t row = 0; row < original.size() / width; ++row) {
    for (int col = 0; col < width; ++col) {
      if (!std::isfinite(original[row * width + col]) ||
          !std::isfinite(patched[row * width + col])) {
        return absl::DataLossError("nonfinite embedding weight");
      }
    }
    if (std::memcmp(original.data() + row * width, patched.data() + row * width,
                    width * sizeof(float))) {
      if (row >= static_cast<size_t>(vocabulary)) {
        return absl::InvalidArgumentError("embedding padding changed");
      }
      changed.push_back(row);
    }
  }
  return changed;
}

absl::StatusOr<FactorialMeasurements> EvaluateEmbeddingFactorial(
    cuda::Executor& executor,
    const std::array<const NativeLogitLens*, 2>& lenses,
    const std::array<cuda::Buffer, 2>& residuals,
    const std::array<cuda::Buffer, 2>& native_logits,
    absl::Span<const int32_t> selected_rows,
    absl::Span<const int32_t> targets) {
  constexpr int vocabulary = llm::kGpt2VocabularySize;
  constexpr int padded = llm::kGpt2PaddedVocabularySize;
  constexpr int width = llm::kGpt2ModelWidth;
  if (!lenses[0] || !lenses[1] || selected_rows.empty() ||
      targets.size() != selected_rows.size()) {
    return absl::InvalidArgumentError("missing readout or selected targets");
  }
  size_t rows = 0;
  for (int side = 0; side < 2; ++side) {
    if (&residuals[side].executor() != &executor ||
        &native_logits[side].executor() != &executor ||
        residuals[side].size_bytes() % (width * sizeof(uint16_t))) {
      return absl::InvalidArgumentError("invalid residual shape or executor");
    }
    const size_t side_rows =
        residuals[side].size_bytes() / (width * sizeof(uint16_t));
    if (!side_rows || side_rows > std::numeric_limits<int>::max() ||
        (side && side_rows != rows) ||
        native_logits[side].size_bytes() !=
            side_rows * padded * sizeof(float)) {
      return absl::InvalidArgumentError("incompatible native logit geometry");
    }
    rows = side_rows;
  }
  RETURN_IF_ERROR(ValidateFactorialRows(selected_rows, 1, selected_rows.size(),
                                        static_cast<int>(rows)));
  for (int target : targets) {
    if (target < 0 || target >= vocabulary) {
      return absl::InvalidArgumentError("target is not a logical token ID");
    }
  }
  FactorialMeasurements result;
  for (int input = 0; input < 2; ++input) {
    for (int head = 0; head < 2; ++head) {
      const int cell = input * 2 + head;
      ASSIGN_OR_RETURN(auto logits,
                       lenses[head]->Apply(executor, residuals[input]));
      ASSIGN_OR_RETURN(result.logits[cell],
                       cuda::PageLockedHostArray<float>::Allocate(
                           selected_rows.size() * vocabulary));
      for (size_t i = 0; i < selected_rows.size(); ++i) {
        ASSIGN_OR_RETURN(auto actual,
                         ReadSelectedRow(executor, logits, selected_rows[i],
                                         padded, sizeof(float)));
        if (input == head) {
          ASSIGN_OR_RETURN(
              auto expected,
              ReadSelectedRow(executor, native_logits[input], selected_rows[i],
                              padded, sizeof(float)));
          if (std::memcmp(actual.data(), expected.data(),
                          actual.size_bytes())) {
            return absl::DataLossError(absl::StrCat(
                "native diagonal readout is not byte-identical: ",
                kEmbeddingFactorialCells[cell], " row ", selected_rows[i]));
          }
        }
        const auto row = absl::MakeConstSpan(
            reinterpret_cast<const float*>(actual.data()), vocabulary);
        ASSIGN_OR_RETURN(auto score, ScoreFactorialToken(row, targets[i]));
        result.scores[cell].push_back(score);
        std::memcpy(result.logits[cell].data() + i * vocabulary, row.data(),
                    vocabulary * sizeof(float));
      }
    }
  }
  return result;
}

}  // namespace pluto::weight_analysis
