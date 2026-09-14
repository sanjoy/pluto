#include "src/llm/experiments/mlp_automaton/history.h"

#include <algorithm>
#include <charconv>
#include <ostream>
#include <system_error>
#include <utility>

#include "absl/container/flat_hash_set.h"
#include "absl/strings/escaping.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "src/util/status_macros.h"

namespace pluto::llm::mlp_automaton {
namespace {

absl::Status FilesystemError(const std::filesystem::path& path,
                             const std::error_code& error) {
  return absl::InternalError(
      absl::StrCat("Cannot inspect ", path.string(), ": ", error.message()));
}

absl::Status ValidateBlocks(absl::Span<const int> blocks) {
  if (blocks.empty())
    return absl::InvalidArgumentError(
        "A path must have at least one MLP block");
  int previous = -1;
  for (int block : blocks) {
    if (block <= previous)
      return absl::InvalidArgumentError(
          "MLP block IDs must be nonnegative, unique, and sorted");
    previous = block;
  }
  return absl::OkStatus();
}

absl::Status ValidatePaths(absl::Span<const CombinedPath> paths) {
  absl::flat_hash_set<absl::string_view> seen;
  for (const CombinedPath& path : paths) {
    if (path.bytes.empty() || !seen.insert(path.bytes).second)
      return absl::InvalidArgumentError(
          "Path texts must be nonempty and unique");
    RETURN_IF_ERROR(ValidateBlocks(path.mlp_blocks));
  }
  return absl::OkStatus();
}

absl::Status ValidateHistories(absl::Span<const PathHistory> histories) {
  absl::flat_hash_set<absl::string_view> seen;
  for (const PathHistory& history : histories) {
    if (history.bytes.empty() || !seen.insert(history.bytes).second ||
        history.ranges.empty())
      return absl::InvalidArgumentError(
          "History texts must be nonempty and unique, with nonempty ranges");
    int64_t previous = -1;
    for (const CheckpointRange& range : history.ranges) {
      if (range.first_step <= previous || range.last_step < range.first_step)
        return absl::InvalidArgumentError(
            "Checkpoint ranges must be nonnegative, ordered, and disjoint");
      RETURN_IF_ERROR(ValidateBlocks(range.mlp_blocks));
      previous = range.last_step;
    }
  }
  return absl::OkStatus();
}

std::string Hex(absl::string_view bytes) {
  constexpr char kHex[] = "0123456789abcdef";
  std::string result;
  result.reserve(bytes.size() * 2);
  for (unsigned char byte : bytes) {
    result.push_back(kHex[byte >> 4]);
    result.push_back(kHex[byte & 15]);
  }
  return result;
}

// CEscape produces ASCII display text; quote that display as a JSON string.
// The extra escaping here is distinct from the byte escapes being displayed.
std::string JsonQuote(absl::string_view display) {
  std::string result = "\"";
  for (char c : display) {
    if (c == '\\' || c == '"')
      result.push_back('\\');
    result.push_back(c);
  }
  result.push_back('"');
  return result;
}

void WriteBlocks(std::ostream& output, absl::Span<const int> blocks,
                 absl::string_view separator) {
  for (size_t index = 0; index < blocks.size(); ++index) {
    if (index != 0)
      output << separator;
    // Do not inherit hex flags or a locale's grouping of decimal integers.
    output << std::to_string(blocks[index]);
  }
}

absl::Status StreamStatus(std::ostream& output) {
  if (!output)
    return absl::DataLossError("Failed to write MLP automaton history");
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<CheckpointSelection> DiscoverCheckpoints(
    const std::filesystem::path& directory) {
  std::error_code error;
  const bool is_directory = std::filesystem::is_directory(directory, error);
  if (error)
    return FilesystemError(directory, error);
  if (!is_directory)
    return absl::InvalidArgumentError(
        absl::StrCat("Not a checkpoint directory: ", directory.string()));

  const auto weight = directory / "weight_0.bin";
  const bool has_weight = std::filesystem::is_regular_file(weight, error);
  if (error && error != std::errc::no_such_file_or_directory)
    return FilesystemError(weight, error);
  if (has_weight)
    return CheckpointSelection{false, {{directory, 0}}};

  error.clear();
  std::filesystem::directory_iterator iterator(directory, error);
  if (error)
    return FilesystemError(directory, error);
  CheckpointSelection result{true, {}};
  while (iterator != std::filesystem::directory_iterator()) {
    const auto path = iterator->path();
    const std::string name = path.filename().string();
    const absl::string_view suffix =
        name.size() > 5 ? absl::string_view(name).substr(5) : "";
    const bool step_name =
        name.starts_with("step_") && !suffix.empty() &&
        std::all_of(suffix.begin(), suffix.end(),
                    [](char c) { return c >= '0' && c <= '9'; });
    if (step_name) {
      const bool child_directory = iterator->is_directory(error);
      if (error)
        return FilesystemError(path, error);
      if (child_directory) {
        int64_t step;
        const auto parsed =
            std::from_chars(suffix.data(), suffix.data() + suffix.size(), step);
        if (parsed.ec != std::errc())
          return absl::InvalidArgumentError(
              absl::StrCat("Checkpoint step is outside int64 range: ", name));
        result.checkpoints.push_back({path, step});
      }
    }
    iterator.increment(error);
    if (error)
      return FilesystemError(directory, error);
  }
  std::sort(
      result.checkpoints.begin(), result.checkpoints.end(),
      [](const Checkpoint& a, const Checkpoint& b) { return a.step < b.step; });
  for (size_t index = 1; index < result.checkpoints.size(); ++index)
    if (result.checkpoints[index - 1].step == result.checkpoints[index].step)
      return absl::InvalidArgumentError(
          absl::StrCat("Multiple checkpoint directories have step ",
                       result.checkpoints[index].step, ": ",
                       result.checkpoints[index - 1].directory.string(),
                       " and ", result.checkpoints[index].directory.string()));
  if (result.checkpoints.empty())
    return absl::NotFoundError(
        absl::StrCat("No checkpoint or step_<number> directories found in ",
                     directory.string()));
  return result;
}

absl::Status HistoryAccumulator::AddCheckpoint(
    int64_t step, absl::Span<const CombinedPath> complete_paths,
    absl::Span<const CombinedPath> sampled_paths) {
  if (step < 0 || step <= last_step_)
    return absl::InvalidArgumentError(
        "Checkpoint steps must be nonnegative and strictly increasing");
  RETURN_IF_ERROR(ValidatePaths(complete_paths));
  RETURN_IF_ERROR(ValidatePaths(sampled_paths));
  absl::flat_hash_map<absl::string_view, const CombinedPath*> complete;
  for (const CombinedPath& path : complete_paths)
    complete.emplace(path.bytes, &path);
  for (const CombinedPath& path : sampled_paths) {
    const auto found = complete.find(path.bytes);
    if (found == complete.end() || found->second->mlp_blocks != path.mlp_blocks)
      return absl::InvalidArgumentError(
          "Sampled paths must have the same membership as complete paths");
  }

  // Do all validation above before mutating state. last_seen_step also detects
  // absence without scanning the full historical vocabulary at every step.
  for (const CombinedPath& path : complete_paths) {
    Entry& entry = entries_[path.bytes];
    if (!entry.ranges.empty() && entry.last_seen_step == last_step_ &&
        entry.ranges.back().mlp_blocks == path.mlp_blocks) {
      entry.ranges.back().last_step = step;
    } else {
      entry.ranges.push_back({step, step, path.mlp_blocks});
    }
    entry.last_seen_step = step;
  }
  for (const CombinedPath& path : sampled_paths)
    entries_.find(path.bytes)->second.sampled = true;
  last_step_ = step;
  return absl::OkStatus();
}

std::vector<PathHistory> HistoryAccumulator::Finish() const {
  std::vector<PathHistory> result;
  for (const auto& [bytes, entry] : entries_)
    if (entry.sampled)
      result.push_back({bytes, entry.ranges});
  std::sort(result.begin(), result.end(),
            [](const PathHistory& a, const PathHistory& b) {
              return a.bytes < b.bytes;
            });
  return result;
}

absl::Status WriteHistoryText(std::ostream& output,
                              absl::Span<const PathHistory> histories) {
  RETURN_IF_ERROR(ValidateHistories(histories));
  for (const PathHistory& history : histories) {
    output << "\"" << absl::CEscape(history.bytes) << "\":\n";
    for (const CheckpointRange& range : history.ranges) {
      output << "  Chkpt " << std::to_string(range.first_step);
      if (range.first_step != range.last_step)
        output << " - " << std::to_string(range.last_step);
      output << " — block ";
      WriteBlocks(output, range.mlp_blocks, ",");
      output << "\n";
    }
  }
  return StreamStatus(output);
}

absl::Status WriteHistoryJson(std::ostream& output,
                              absl::Span<const PathHistory> histories,
                              absl::Span<const int64_t> analyzed_steps) {
  RETURN_IF_ERROR(ValidateHistories(histories));
  int64_t previous = -1;
  for (int64_t step : analyzed_steps) {
    if (step <= previous)
      return absl::InvalidArgumentError(
          "Analyzed steps must be nonnegative and strictly increasing");
    previous = step;
  }
  for (const PathHistory& history : histories)
    for (const CheckpointRange& range : history.ranges)
      if (!std::binary_search(analyzed_steps.begin(), analyzed_steps.end(),
                              range.first_step) ||
          !std::binary_search(analyzed_steps.begin(), analyzed_steps.end(),
                              range.last_step))
        return absl::InvalidArgumentError(
            "History range endpoints must be analyzed checkpoints");

  output << "{\n  \"format\": \"pluto.mlp_automaton.history.v1\",\n"
            "  \"range_semantics\": \"consecutive_analyzed_checkpoints\",\n"
            "  \"analyzed_steps\": [";
  for (size_t index = 0; index < analyzed_steps.size(); ++index) {
    if (index != 0)
      output << ", ";
    output << std::to_string(analyzed_steps[index]);
  }
  output << "],\n  \"paths\": [";
  for (size_t index = 0; index < histories.size(); ++index) {
    const PathHistory& history = histories[index];
    if (index != 0)
      output << ",";
    output << "\n    {\"bytes_hex\": " << JsonQuote(Hex(history.bytes))
           << ", \"bytes_escaped\": " << JsonQuote(absl::CEscape(history.bytes))
           << ", \"ranges\": [";
    for (size_t range_index = 0; range_index < history.ranges.size();
         ++range_index) {
      const CheckpointRange& range = history.ranges[range_index];
      if (range_index != 0)
        output << ", ";
      output << "{\"first_step\": " << std::to_string(range.first_step)
             << ", \"last_step\": " << std::to_string(range.last_step)
             << ", \"mlp_blocks\": [";
      WriteBlocks(output, range.mlp_blocks, ", ");
      output << "]}";
    }
    output << "]}";
  }
  output << "\n  ]\n}\n";
  return StreamStatus(output);
}

}  // namespace pluto::llm::mlp_automaton
