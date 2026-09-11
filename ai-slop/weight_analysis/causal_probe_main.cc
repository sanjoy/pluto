#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "ai-slop/weight_analysis/causal_probe.h"
#include "src/llm/checkpoint.h"
#include "src/llm/layers/cross_entropy_loss.h"
#include "src/llm/recipes/gpt2.h"
#include "src/util/status_macros.h"

ABSL_FLAG(std::string, checkpoint, "",
          "Read-only full GPT-2 checkpoint directory.");
ABSL_FLAG(std::string, batch_tokens, "",
          "Frozen little-endian int32 input/target batch.");
ABSL_FLAG(std::string, output_dir, "",
          "New directory for validation measurements.");
ABSL_FLAG(int, batch_sequences, 4,
          "Execution microbatch; never changes passage selection.");

namespace pluto::weight_analysis {
namespace {
using Clock = std::chrono::steady_clock;

// Tensor sizes cannot detect a changed head count. Fail at compile time if
// any part of the frozen architecture diverges from the production recipe.
static_assert(llm::kGpt2VocabularySize == 50257);
static_assert(llm::kGpt2PaddedVocabularySize == 50272);
static_assert(llm::kGpt2ContextLength == 1024);
static_assert(llm::kGpt2TransformerBlockCount == 8);
static_assert(llm::kGpt2ModelWidth == 512);
static_assert(llm::kGpt2AttentionHeads == 8);
static_assert(llm::kGpt2AttentionHeadDimension == 64);
static_assert(llm::kGpt2FeedForwardWidth == 2048);

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
      absl::GetFlag(FLAGS_output_dir).empty() ||
      absl::GetFlag(FLAGS_batch_sequences) <= 0) {
    return absl::InvalidArgumentError(
        "required: --checkpoint --batch_tokens --output_dir; --batch_sequences "
        "must be positive");
  }
  const auto checkpoint =
      std::filesystem::canonical(absl::GetFlag(FLAGS_checkpoint));
  const auto batch_path =
      std::filesystem::canonical(absl::GetFlag(FLAGS_batch_tokens));
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
  RETURN_IF_ERROR(ValidateCheckpointFiles(checkpoint));
  RETURN_IF_ERROR(CreateNewOutputDirectory(output_path));
  ASSIGN_OR_RETURN(auto executor, cuda::Executor::Create());
  ASSIGN_OR_RETURN(auto batch,
                   LoadPackedBatch(batch_path, llm::kGpt2ContextLength,
                                   llm::kGpt2VocabularySize));
  if (batch.passage_count != 32) {
    return absl::InvalidArgumentError(
        "frozen validation protocol requires exactly 32 passages");
  }
  // Executor is declared first so it outlives all buffers, model weights, and
  // queued asynchronous frees. No default stream or guessed stream is used.
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
           << ",\n\"binary_file\":" << JsonQuote(binary_path.string())
           << ",\n\"checkpoint_directory\":" << JsonQuote(checkpoint.string())
           << ",\"checkpoint_unique_weight_count\":" << weights.size()
           << ",\"checkpoint_raw_weight_count\":" << raw_weights.size()
           << ",\"non_default_executor\":true,\"optimizer_steps\":0,\"backward_"
              "calls\":0"
           << ",\n\"compute_type\":\"BF16 activations, FP32 master "
              "weights/logits/loss\""
           << ",\"layout\":\"little-endian; input and target halves; "
              "passage-major outputs\""
           << ",\"integrity_hashes\":\"external frozen planner/reporter hashes "
              "input, binary and checkpoint before/after\""
           << ",\n\"checkpoint_weight_bytes\":[";
  for (size_t i = 0; i < weights.size(); ++i) {
    if (i)
      metadata << ',';
    metadata << weights[i].size_bytes();
  }
  metadata << "],\n\"arms\":[\n";
  const auto arms = CausalArms();
  for (size_t i = 0; i < arms.size(); ++i) {
    const auto& arm = arms[i];
    const auto arm_start = Clock::now();
    std::unique_ptr<WeightIntervention> intervention;
    if (!arm.weight_indices.empty()) {
      ASSIGN_OR_RETURN(
          intervention,
          WeightIntervention::Capture(*executor, weights, arm.weight_indices));
      RETURN_IF_ERROR(intervention->Apply(arm.scale));
    }
    auto measured = EvaluatePassages(*executor, *model, *loss_layer, batch,
                                     absl::GetFlag(FLAGS_batch_sequences),
                                     llm::kGpt2VocabularySize,
                                     llm::kGpt2PaddedVocabularySize);
    // Restore even if forward evaluation rejected a nonfinite result. Only
    // successful explicit verification permits an arm's completion record.
    if (intervention)
      RETURN_IF_ERROR(intervention->RestoreAndVerify());
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
             << ",\"loss_file\":" << JsonQuote(loss_file)
             << ",\"argmax_file\":" << JsonQuote(argmax_file)
             << ",\"group_weight_indices\":[";
    for (size_t j = 0; j < arm.weight_indices.size(); ++j) {
      if (j)
        metadata << ',';
      metadata << arm.weight_indices[j];
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
  // Written last: a failed/partial run never has a complete metadata marker.
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
