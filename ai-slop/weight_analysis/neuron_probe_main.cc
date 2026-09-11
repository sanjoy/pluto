#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "ai-slop/weight_analysis/causal_probe.h"
#include "src/llm/checkpoint.h"
#include "src/llm/layers/cross_entropy_loss.h"
#include "src/llm/recipes/gpt2.h"
#include "src/util/status_macros.h"

ABSL_FLAG(std::string, checkpoint, "",
          "Read-only full GPT-2 checkpoint directory.");
ABSL_FLAG(std::string, batch_tokens, "",
          "Frozen little-endian int32 input/target batch of 32 passages.");
ABSL_FLAG(
    std::string, neuron_groups, "",
    "Frozen little-endian int32 [2,32,64] feature IDs for blocks 6 and 7.");
ABSL_FLAG(std::string, output_dir, "",
          "New directory for neuron-group validation measurements.");
ABSL_FLAG(int, batch_sequences, 4,
          "Execution microbatch; never changes passage or neuron selection.");

namespace pluto::weight_analysis {
namespace {
using Clock = std::chrono::steady_clock;

// Shapes alone cannot authenticate the head count or historical checkpoint
// producer. Freeze the current recipe here; Python records the source hashes.
static_assert(llm::kGpt2VocabularySize == 50257);
static_assert(llm::kGpt2PaddedVocabularySize == 50272);
static_assert(llm::kGpt2ContextLength == 1024);
static_assert(llm::kGpt2TransformerBlockCount == 8);
static_assert(llm::kGpt2ModelWidth == 512);
static_assert(llm::kGpt2AttentionHeads == 8);
static_assert(llm::kGpt2AttentionHeadDimension == 64);
static_assert(llm::kGpt2FeedForwardWidth == 2048);

constexpr int kBlocks = 2;
constexpr int kGroupsPerBlock = 32;
constexpr int kFeaturesPerGroup = 64;
constexpr int kFeatures = kGroupsPerBlock * kFeaturesPerGroup;
constexpr size_t kGroupsBytes = kBlocks * kFeatures * sizeof(int32_t);
constexpr size_t kArmCount = 3 + kBlocks * (1 + kGroupsPerBlock);
static_assert(kArmCount == 69);
static_assert(kFeatures == llm::kGpt2FeedForwardWidth);
using NeuronGroups =
    std::array<std::array<std::array<int, kFeaturesPerGroup>, kGroupsPerBlock>,
               kBlocks>;

// This is a frozen partition, not a ranking or a request for the driver to
// choose neurons. Decode explicitly so high-bit or big-endian input cannot
// silently become another valid index. Every feature occurs once PER block.
absl::StatusOr<NeuronGroups> LoadNeuronGroups(
    const std::filesystem::path& path) {
  if (!std::filesystem::is_regular_file(path) ||
      std::filesystem::file_size(path) != kGroupsBytes) {
    return absl::InvalidArgumentError(
        "neuron_groups must be exactly 16384 bytes: LE int32 [2,32,64]");
  }
  std::array<unsigned char, kGroupsBytes> bytes;
  std::ifstream input(path, std::ios::binary);
  if (!input.read(reinterpret_cast<char*>(bytes.data()), bytes.size()) ||
      input.peek() != std::char_traits<char>::eof()) {
    return absl::DataLossError("could not read exact neuron_groups contents");
  }
  NeuronGroups result;
  size_t offset = 0;
  for (int block = 0; block < kBlocks; ++block) {
    std::array<bool, kFeatures> seen{};
    for (int group = 0; group < kGroupsPerBlock; ++group) {
      for (int feature = 0; feature < kFeaturesPerGroup; ++feature) {
        const uint32_t id = static_cast<uint32_t>(bytes[offset]) |
                            (static_cast<uint32_t>(bytes[offset + 1]) << 8) |
                            (static_cast<uint32_t>(bytes[offset + 2]) << 16) |
                            (static_cast<uint32_t>(bytes[offset + 3]) << 24);
        offset += sizeof(int32_t);
        if (id >= kFeatures || seen[id]) {
          return absl::InvalidArgumentError(
              absl::StrCat("neuron_groups block ", block + 6,
                           " must contain a separate permutation of 0..2047"));
        }
        seen[id] = true;
        result[block][group][feature] = static_cast<int>(id);
      }
    }
  }
  return result;
}

struct NeuronArm {
  std::string name;
  std::string kind;
  int block;
  std::vector<int> weight_indices;
  std::vector<int> selected_rows;
  float scale;
};

std::vector<NeuronArm> Arms(const NeuronGroups& groups) {
  std::vector<NeuronArm> result = {{"clean_before", "clean", -1, {}, {}, 1.0f},
                                   {"clean_repeat", "clean", -1, {}, {}, 1.0f}};
  for (int index = 0; index < kBlocks; ++index) {
    const int block = index + 6;
    const int weight = 12 + 12 * block;
    result.push_back({absl::StrCat("block", block, "_mlp_half"),
                      "whole_mlp_output",
                      block,
                      {weight, weight + 1},
                      {},
                      0.5f});
    for (int group = 0; group < kGroupsPerBlock; ++group) {
      const auto& rows = groups[index][group];
      result.push_back({absl::StrCat("block", block, "_group",
                                     group < 10 ? "0" : "", group, "_half"),
                        "mlp_output_rows",
                        block,
                        {weight},
                        std::vector<int>(rows.begin(), rows.end()),
                        0.5f});
    }
  }
  result.push_back({"clean_after", "clean", -1, {}, {}, 1.0f});
  return result;
}

absl::Status ValidateCheckpointFiles(const std::filesystem::path& directory) {
  const auto sizes = Gpt2WeightByteSizes();
  size_t count = 0;
  for (const auto& entry : std::filesystem::directory_iterator(directory)) {
    const auto name = entry.path().filename().string();
    if (name.starts_with("weight_") && name.ends_with(".bin"))
      ++count;
  }
  if (count != sizes.size()) {
    return absl::InvalidArgumentError(
        "full checkpoint must contain exactly 100 weight files");
  }
  for (size_t i = 0; i < sizes.size(); ++i) {
    const auto path = directory / absl::StrCat("weight_", i, ".bin");
    if (!std::filesystem::is_regular_file(path) ||
        std::filesystem::file_size(path) != sizes[i]) {
      return absl::InvalidArgumentError(
          absl::StrCat("checkpoint size mismatch: ", path.string()));
    }
  }
  return absl::OkStatus();
}

absl::Status Run() {
  const auto run_start = Clock::now();
  if (absl::GetFlag(FLAGS_checkpoint).empty() ||
      absl::GetFlag(FLAGS_batch_tokens).empty() ||
      absl::GetFlag(FLAGS_neuron_groups).empty() ||
      absl::GetFlag(FLAGS_output_dir).empty() ||
      absl::GetFlag(FLAGS_batch_sequences) <= 0) {
    return absl::InvalidArgumentError(
        "required: --checkpoint --batch_tokens --neuron_groups --output_dir; "
        "--batch_sequences must be positive");
  }
  const auto checkpoint =
      std::filesystem::canonical(absl::GetFlag(FLAGS_checkpoint));
  const auto batch_path =
      std::filesystem::canonical(absl::GetFlag(FLAGS_batch_tokens));
  const auto groups_path =
      std::filesystem::canonical(absl::GetFlag(FLAGS_neuron_groups));
  const auto binary_path = std::filesystem::canonical("/proc/self/exe");
  const auto output_path =
      std::filesystem::absolute(absl::GetFlag(FLAGS_output_dir));
  const auto resolved_output =
      (std::filesystem::canonical(output_path.parent_path()) /
       output_path.filename())
          .lexically_normal();
  for (auto ancestor = resolved_output;; ancestor = ancestor.parent_path()) {
    if (ancestor == checkpoint) {
      return absl::InvalidArgumentError(
          "output directory must not be inside checkpoint");
    }
    if (ancestor == ancestor.parent_path())
      break;
  }
  // Reject malformed partitions before creating a model or running a forward.
  ASSIGN_OR_RETURN(const auto groups, LoadNeuronGroups(groups_path));
  const auto arms = Arms(groups);
  if (arms.size() != kArmCount)
    return absl::InternalError("unexpected frozen intervention count");
  RETURN_IF_ERROR(ValidateCheckpointFiles(checkpoint));
  RETURN_IF_ERROR(CreateNewOutputDirectory(output_path));
  // The executor outlives all allocations and queued asynchronous frees.
  ASSIGN_OR_RETURN(auto executor, cuda::Executor::Create());
  ASSIGN_OR_RETURN(auto batch, LoadPackedBatch(*executor, batch_path,
                                               llm::kGpt2ContextLength,
                                               llm::kGpt2VocabularySize));
  if (batch.passage_count != 32) {
    return absl::InvalidArgumentError(
        "frozen validation protocol requires exactly 32 passages");
  }
  ASSIGN_OR_RETURN(auto model,
                   llm::CreateGpt2(*executor, llm::DataType::BF16, 0));
  ASSIGN_OR_RETURN(auto weights, UniqueWeights(*executor, model->weights()));
  RETURN_IF_ERROR(ValidateGpt2Weights(weights));
  const auto raw_weights = model->weights();
  if (raw_weights.size() != 101 ||
      raw_weights.front().data() != raw_weights.back().data()) {
    return absl::FailedPreconditionError(
        "unexpected GPT-2 tied-embedding traversal");
  }
  RETURN_IF_ERROR(llm::ReadFromDirectory(*executor, *model, checkpoint));
  ASSIGN_OR_RETURN(auto loss_layer, llm::CrossEntropyLossLayer::Create(
                                        *executor, llm::kGpt2VocabularySize,
                                        llm::DataType::BF16));

  std::ostringstream metadata;
  metadata << std::setprecision(17)
           << "{\n\"schema_version\":1,\"complete\":true,"
           << "\n\"context_length\":" << batch.context_length
           << ",\"passage_count\":" << batch.passage_count
           << ",\"vocab_size\":" << llm::kGpt2VocabularySize
           << ",\"padded_vocab_size\":" << llm::kGpt2PaddedVocabularySize
           << ",\"batch_sequences\":" << absl::GetFlag(FLAGS_batch_sequences)
           << ",\n\"batch_tokens_file\":" << JsonQuote(batch_path.string())
           << ",\"batch_tokens_bytes\":" << batch.tokens.size_bytes()
           << ",\n\"neuron_groups_file\":" << JsonQuote(groups_path.string())
           << ",\"neuron_groups_bytes\":" << kGroupsBytes
           << ",\"neuron_groups_shape\":[2,32,64]"
           << ",\"neuron_group_blocks\":[6,7]"
           << ",\"neuron_groups_permutations_verified\":true"
           << ",\n\"binary_file\":" << JsonQuote(binary_path.string())
           << ",\n\"checkpoint_directory\":" << JsonQuote(checkpoint.string())
           << ",\"checkpoint_unique_weight_count\":" << weights.size()
           << ",\"checkpoint_raw_weight_count\":" << raw_weights.size()
           << ",\"non_default_executor\":true,\"optimizer_steps\":0,\"backward_"
              "calls\":0"
           << ",\n\"compute_type\":\"BF16 activations, FP32 master "
              "weights/logits/loss\""
           << ",\"layout\":\"little-endian; input and target halves; "
              "passage-major outputs; W2 rows are input features\""
           << ",\"integrity_hashes\":\"external frozen planner/reporter hashes "
              "input, groups, binary and checkpoint before/after\""
           << ",\n\"checkpoint_weight_bytes\":[";
  for (size_t i = 0; i < weights.size(); ++i) {
    if (i)
      metadata << ',';
    metadata << weights[i].size_bytes();
  }
  metadata << "],\n\"arms\":[\n";
  for (size_t i = 0; i < arms.size(); ++i) {
    const auto& arm = arms[i];
    const auto arm_start = Clock::now();
    std::unique_ptr<WeightIntervention> whole;
    std::unique_ptr<MlpRowIntervention> rows;
    if (!arm.selected_rows.empty()) {
      ASSIGN_OR_RETURN(
          rows, MlpRowIntervention::Capture(
                    *executor, weights[arm.weight_indices.front()], kFeatures,
                    llm::kGpt2ModelWidth, arm.selected_rows));
      RETURN_IF_ERROR(rows->Apply(arm.scale));
    } else if (!arm.weight_indices.empty()) {
      ASSIGN_OR_RETURN(whole, WeightIntervention::Capture(*executor, weights,
                                                          arm.weight_indices));
      RETURN_IF_ERROR(whole->Apply(arm.scale));
    }
    auto measured = EvaluatePassages(*executor, *model, *loss_layer, batch,
                                     absl::GetFlag(FLAGS_batch_sequences),
                                     llm::kGpt2VocabularySize,
                                     llm::kGpt2PaddedVocabularySize);
    // Restore the same model allocation even if measurement fails. A group's
    // bias is never passed to the row helper; whole-branch controls scale both
    // output weights AND bias. Only explicit byte verification certifies an
    // arm.
    if (rows)
      RETURN_IF_ERROR(rows->RestoreAndVerify());
    if (whole)
      RETURN_IF_ERROR(whole->RestoreAndVerify());
    if (!measured.ok())
      return measured.status();
    const std::string loss_file = arm.name + ".losses.f32";
    const std::string argmax_file = arm.name + ".argmax.i32";
    RETURN_IF_ERROR(WriteExclusive(output_path / loss_file,
                                   measured->losses.data(),
                                   measured->losses.size_bytes()));
    RETURN_IF_ERROR(WriteExclusive(output_path / argmax_file,
                                   measured->argmax.data(),
                                   measured->argmax.size_bytes()));
    const double elapsed =
        std::chrono::duration<double>(Clock::now() - arm_start).count();
    if (i)
      metadata << ",\n";
    metadata << "{\"name\":" << JsonQuote(arm.name)
             << ",\"kind\":" << JsonQuote(arm.kind) << ",\"block\":";
    if (arm.block < 0)
      metadata << "null";
    else
      metadata << arm.block;
    metadata << ",\"loss_file\":" << JsonQuote(loss_file)
             << ",\"argmax_file\":" << JsonQuote(argmax_file)
             << ",\"group_weight_indices\":[";
    for (size_t j = 0; j < arm.weight_indices.size(); ++j) {
      if (j)
        metadata << ',';
      metadata << arm.weight_indices[j];
    }
    metadata << "],\"selected_rows\":[";
    for (size_t j = 0; j < arm.selected_rows.size(); ++j) {
      if (j)
        metadata << ',';
      metadata << arm.selected_rows[j];
    }
    metadata << "],\"scale\":" << arm.scale
             << ",\"finite_losses\":true,\"argmax_valid\":true"
             << ",\"restoration_verified_bytes\":true,\"elapsed_seconds\":"
             << elapsed << '}';
    std::cout << (i + 1) << "/" << arms.size() << ' ' << arm.name
              << " complete (" << elapsed << " seconds)" << std::endl;
  }
  RETURN_IF_ERROR(executor->Synchronize());
  metadata << "\n],\"elapsed_seconds\":"
           << std::chrono::duration<double>(Clock::now() - run_start).count()
           << "\n}\n";
  const auto json = metadata.str();
  // A partial or failed run has no complete metadata marker. Files and the
  // directory are exclusive outputs; there is no checkpoint-writing path.
  RETURN_IF_ERROR(
      WriteExclusive(output_path / "metadata.json", json.data(), json.size()));
  return absl::OkStatus();
}
}  // namespace
}  // namespace pluto::weight_analysis

int main(int argc, char** argv) {
  const auto remaining = absl::ParseCommandLine(argc, argv);
  if (remaining.size() != 1) {
    std::cerr << "Unexpected positional arguments\n";
    return 1;
  }
  const auto status = pluto::weight_analysis::Run();
  if (status.ok())
    return 0;
  std::cerr << status << '\n';
  return 1;
}
