// CPU-only, label-independent sign readout of raw checkpoint embedding files.
// See embedding_sign_readout.h for the hypothesis and its limitations.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "src/llm/experiments/one_shot_memorizer/embedding_sign_readout.h"
#include "src/util/status_macros.h"

ABSL_FLAG(std::string, initial_embeddings, "",
          "Initial raw FP32 embedding file, for example step_0/weight_0.bin");
ABSL_FLAG(std::string, trained_embeddings, "",
          "Trained raw FP32 embedding file, for example step_512/weight_0.bin");
ABSL_FLAG(int, width, 16, "Columns in each row-major embedding table");
ABSL_FLAG(
    std::string, target_ids, "",
    "Optional comma-separated distinct target row IDs, used only to "
    "score predictions after label-free decoding; empty disables scoring");

namespace pluto::llm::one_shot_memorizer {
namespace {

absl::StatusOr<std::vector<float>> ReadEmbeddings(const std::string& path) {
  static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559);
  if (path.empty())
    return absl::InvalidArgumentError("both embedding file paths are required");
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input)
    return absl::NotFoundError(
        absl::StrCat("cannot open embedding file: ", path));
  const std::streamoff bytes = input.tellg();
  if (bytes <= 0 || bytes % sizeof(float) != 0)
    return absl::InvalidArgumentError(absl::StrCat(
        "embedding file must contain a nonempty FP32 array: ", path));
  if (static_cast<uintmax_t>(bytes) > std::numeric_limits<size_t>::max() ||
      bytes > std::numeric_limits<std::streamsize>::max())
    return absl::OutOfRangeError("embedding file is too large to read");
  std::vector<float> values(static_cast<size_t>(bytes) / sizeof(float));
  input.seekg(0);
  input.read(reinterpret_cast<char*>(values.data()), bytes);
  if (!input)
    return absl::DataLossError(absl::StrCat("short embedding read: ", path));
  return values;
}

absl::StatusOr<absl::flat_hash_set<int>> ParseTargets(absl::string_view text,
                                                      size_t rows) {
  absl::flat_hash_set<int> targets;
  if (text.empty())
    return targets;
  for (absl::string_view field : absl::StrSplit(text, ',')) {
    int id = -1;
    if (!absl::SimpleAtoi(field, &id) || id < 0 ||
        static_cast<size_t>(id) >= rows)
      return absl::InvalidArgumentError(
          absl::StrCat("invalid or out-of-range target row ID: ", field));
    if (!targets.insert(id).second)
      return absl::InvalidArgumentError(
          absl::StrCat("duplicate target row ID: ", id));
  }
  return targets;
}

struct Range {
  double minimum;
  double maximum;
};

void Include(double value, std::optional<Range>& range) {
  if (!range.has_value()) {
    range = Range{value, value};
    return;
  }
  range->minimum = std::min(range->minimum, value);
  range->maximum = std::max(range->maximum, value);
}

void PrintRange(absl::string_view name, const std::optional<Range>& range) {
  if (range.has_value())
    std::cerr << ' ' << name << "_min=" << range->minimum << ' ' << name
              << "_max=" << range->maximum;
  else
    std::cerr << ' ' << name << "=none";
}

// Selection is taken from the core's completed readout. Labels cannot change
// it; this layer only reports membership accuracy against an optional set.
void PrintReadout(absl::string_view variant, const EmbeddingSignScores& readout,
                  const absl::flat_hash_set<int>& targets, bool score_targets) {
  std::optional<Range> negative, nonnegative, target, nontarget;
  size_t next_selected = 0, tp = 0, fp = 0, fn = 0;
  for (size_t row = 0; row < readout.scores.size(); ++row) {
    const bool selected =
        next_selected < readout.selected_ids.size() &&
        static_cast<size_t>(readout.selected_ids[next_selected]) == row;
    next_selected += selected;
    const double score = readout.scores[row];
    Include(score, selected ? negative : nonnegative);
    std::cout << variant << '\t' << row << '\t' << score << '\t' << selected;
    if (score_targets) {
      const bool actual = targets.contains(static_cast<int>(row));
      std::cout << '\t' << actual;
      Include(score, actual ? target : nontarget);
      tp += selected && actual;
      fp += selected && !actual;
      fn += !selected && actual;
    }
    std::cout << '\n';
  }
  std::cerr << "variant=" << variant
            << " selected=" << readout.selected_ids.size();
  PrintRange("negative", negative);
  PrintRange("nonnegative", nonnegative);
  if (score_targets) {
    std::cerr << " targets=" << targets.size() << " tp=" << tp << " fp=" << fp
              << " fn=" << fn;
    PrintRange("target", target);
    PrintRange("nontarget", nontarget);
  }
  std::cerr << '\n';
}

absl::Status Run() {
  ASSIGN_OR_RETURN(auto initial,
                   ReadEmbeddings(absl::GetFlag(FLAGS_initial_embeddings)));
  ASSIGN_OR_RETURN(auto trained,
                   ReadEmbeddings(absl::GetFlag(FLAGS_trained_embeddings)));
  ASSIGN_OR_RETURN(
      auto readout,
      ComputeEmbeddingSignReadout(initial, trained, absl::GetFlag(FLAGS_width)));
  // Optional gold labels are not even parsed until all scores and selected
  // token sets have been computed. Omitting this flag performs the same decode.
  const std::string target_text = absl::GetFlag(FLAGS_target_ids);
  ASSIGN_OR_RETURN(auto targets,
                   ParseTargets(target_text, readout.delta_fp32.scores.size()));
  const bool score_targets = !target_text.empty();
  std::cout << std::setprecision(17) << "variant\tcompact_id\tscore\tselected";
  if (score_targets)
    std::cout << "\tis_target";
  std::cout << '\n';
  std::cerr << std::setprecision(17);
  PrintReadout("delta_fp32", readout.delta_fp32, targets, score_targets);
  PrintReadout("delta_bf16", readout.delta_bf16, targets, score_targets);
  PrintReadout("trained_fp32_exploratory", readout.trained_fp32, targets,
               score_targets);
  PrintReadout("trained_bf16_exploratory", readout.trained_bf16, targets,
               score_targets);
  if (!std::cout || !std::cerr)
    return absl::DataLossError("failed to write embedding sign report");
  return absl::OkStatus();
}

}  // namespace
}  // namespace pluto::llm::one_shot_memorizer

int main(int argc, char** argv) {
  const auto positional = absl::ParseCommandLine(argc, argv);
  if (positional.size() != 1) {
    std::cerr << "Unexpected positional arguments; use --help.\n";
    return 1;
  }
  const auto status = pluto::llm::one_shot_memorizer::Run();
  if (!status.ok()) {
    std::cerr << status << '\n';
    return 1;
  }
  return 0;
}
